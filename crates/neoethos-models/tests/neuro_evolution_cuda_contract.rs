const NEAT_IMPL: &str = include_str!("../src/evolution/neat_impl.rs");
const NEAT_GPU: &str = include_str!("../src/evolution/neat_gpu.rs");
const CRFMNES_IMPL: &str = include_str!("../src/evolution/crfmnes_impl.rs");
const CRFMNES_GPU: &str = include_str!("../src/evolution/crfmnes_gpu.rs");

#[cfg(feature = "neuro-evolution-gpu")]
use std::path::{Path, PathBuf};
#[cfg(feature = "neuro-evolution-gpu")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "neuro-evolution-gpu")]
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
#[cfg(feature = "neuro-evolution-gpu")]
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
#[cfg(feature = "neuro-evolution-gpu")]
use neoethos_models::base::ExpertModel;
#[cfg(feature = "neuro-evolution-gpu")]
use neoethos_models::{NeatExpert, NeuroEvoExpert};

#[test]
fn requested_neuro_evolution_cuda_is_fail_loud_and_device_tested() {
    for (surface, source) in [("NEAT", NEAT_IMPL), ("CR-FM-NES", CRFMNES_IMPL)] {
        assert!(
            !source.contains("cuda_fitness_disabled"),
            "{surface} must not disable CUDA and continue on CPU"
        );
        assert!(
            !source.contains("falling back to cpu fitness evaluation"),
            "{surface} must propagate requested-CUDA failures"
        );
    }

    assert!(
        NEAT_GPU.contains("neat_cuda_population_metrics_launch_real_kernel"),
        "NEAT needs a mandatory real-device kernel test"
    );
    assert!(
        CRFMNES_GPU.contains("crfmnes_cuda_selection_losses_launch_real_kernel"),
        "CR-FM-NES needs a mandatory real-device kernel test"
    );

    for (surface, source) in [("NEAT", NEAT_GPU), ("CR-FM-NES", CRFMNES_GPU)] {
        assert!(
            !source.contains("cuda_available()")
                && !source.contains("is_available()")
                && !source.contains("return; // skip"),
            "{surface} real-device test must not skip when CUDA is unavailable"
        );
    }
}

#[cfg(feature = "neuro-evolution-gpu")]
struct TempArtifactDir(PathBuf);

#[cfg(feature = "neuro-evolution-gpu")]
impl TempArtifactDir {
    fn create(name: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-{name}-cuda-lifecycle-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create evolution CUDA artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(feature = "neuro-evolution-gpu")]
impl Drop for TempArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove evolution CUDA artifact directory {}: {error}",
                self.0.display()
            );
        }
    }
}

#[cfg(feature = "neuro-evolution-gpu")]
fn evolution_fixture(rows: usize, features: usize) -> (FeatureFrame, Vec<i32>) {
    let columns = (0..features)
        .map(|feature_index| {
            FeatureColumnF64::new(
                format!("feature_{feature_index}"),
                (0..rows)
                    .map(|row_index| {
                        let phase = row_index as f64 * (feature_index + 1) as f64 * 0.031;
                        phase.sin() + 0.1 * phase.cos()
                    })
                    .collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("build deterministic evolution CUDA feature")
        })
        .collect::<Vec<_>>();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("build evolution CUDA frame");
    let labels = (0..rows)
        .map(|row_index| match row_index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();
    (frame, labels)
}

#[cfg(feature = "neuro-evolution-gpu")]
fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one evolution worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("evolution CUDA test can acquire one worker")
}

#[cfg(feature = "neuro-evolution-gpu")]
fn assert_probability_matrix(probabilities: &ndarray::Array2<f64>, rows: usize) {
    assert_eq!(probabilities.dim(), (rows, 3));
    for row in probabilities.outer_iter() {
        let sum = row.iter().sum::<f64>();
        assert!((sum - 1.0).abs() <= 1e-6, "probability sum is {sum}");
        assert!(row.iter().all(|value| value.is_finite()));
    }
}

#[cfg(feature = "neuro-evolution-gpu")]
fn exercise_evolution_cuda_lifecycle<F>(
    name: &str,
    mut model: Box<dyn ExpertModel>,
    mut restored: Box<dyn ExpertModel>,
    frame: &FeatureFrame,
    labels: &[i32],
    runtime_file: &str,
    assert_runtime: F,
) where
    F: FnOnce(&serde_json::Value),
{
    let lease = one_worker_lease();
    model
        .fit(frame, labels, &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} CUDA training failed: {error:#}"));
    let trained = model
        .predict_proba(frame, &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} inference failed: {error:#}"));
    assert_probability_matrix(&trained, frame.n_samples());

    let artifact_dir = TempArtifactDir::create(name);
    model
        .save(artifact_dir.path())
        .unwrap_or_else(|error| panic!("persist {name} CUDA artifact: {error:#}"));
    let runtime: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact_dir.path().join(runtime_file))
            .expect("read evolution CUDA artifact"),
    )
    .expect("parse evolution CUDA artifact");
    assert_runtime(&runtime);

    restored
        .load(artifact_dir.path())
        .unwrap_or_else(|error| panic!("restore {name} CUDA artifact: {error:#}"));
    let after_load = restored
        .predict_proba(frame, &lease)
        .unwrap_or_else(|error| panic!("restored {name} inference failed: {error:#}"));
    assert_probability_matrix(&after_load, frame.n_samples());
    for (before, after) in trained.iter().zip(after_load.iter()) {
        assert!(
            (before - after).abs() <= 1e-6,
            "{name} save/load probability drift: trained={before}, restored={after}"
        );
    }
}

#[cfg(feature = "neuro-evolution-gpu")]
#[test]
fn neat_and_neuro_evo_cuda_surfaces_train_infer_save_load() {
    assert!(
        neoethos_models::tree_models::config::nvidia_gpu_count() > 0,
        "mandatory evolution CUDA lifecycle gate requires a visible NVIDIA device"
    );
    let (frame, labels) = evolution_fixture(64, 4);
    exercise_evolution_cuda_lifecycle(
        "neat",
        Box::new(NeatExpert::with_config(4, 24, 8).with_device_policy("gpu:0")),
        Box::new(NeatExpert::with_config(4, 24, 8).with_device_policy("gpu:0")),
        &frame,
        &labels,
        "neat.json",
        |runtime| {
            assert_eq!(runtime["requested_device_policy"], "gpu:0");
            assert_eq!(runtime["effective_device_policy"], "gpu:0");
            assert_eq!(runtime["runtime_backend"], "symbios_neat_cuda_fitness");
        },
    );
    exercise_evolution_cuda_lifecycle(
        "neuro_evo",
        Box::new(
            NeuroEvoExpert::with_config(4, 8, 0.25, 4)
                .with_search_topology(4, 1)
                .with_device_policy("gpu:0"),
        ),
        Box::new(
            NeuroEvoExpert::with_config(4, 8, 0.25, 4)
                .with_search_topology(4, 1)
                .with_device_policy("gpu:0"),
        ),
        &frame,
        &labels,
        "neuro_evo.json",
        |runtime| {
            assert_eq!(runtime["requested_device_policy"], "gpu:0");
            assert_eq!(runtime["effective_device_policy"], "gpu:0");
            assert!(
                runtime["search_backend"]
                    .as_str()
                    .is_some_and(|backend| backend.contains("cuda")),
                "CR-FM-NES artifact did not record CUDA fitness: {runtime}"
            );
        },
    );
}
