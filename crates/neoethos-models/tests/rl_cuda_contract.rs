const RL_IMPL: &str = include_str!("../src/rl/dqn_impl.rs");
const RL_TESTS: &str = include_str!("../src/rl/dqn_impl_tests.rs");

#[cfg(feature = "reinforcement-learning-cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "reinforcement-learning-cuda")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "reinforcement-learning-cuda")]
use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
#[cfg(feature = "reinforcement-learning-cuda")]
use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
#[cfg(feature = "reinforcement-learning-cuda")]
use neoethos_models::TradingReinforcementLearner;

#[test]
fn requested_rl_cuda_is_fail_loud_and_device_tested() {
    assert!(
        !RL_IMPL.contains("retrying on CPU"),
        "requested RL CUDA training must not retry on CPU"
    );
    assert!(
        !RL_IMPL.contains("Err(gpu_err) if effective_backend != \"rlkit_cpu\""),
        "requested RL CUDA failures must propagate"
    );
    assert!(
        RL_IMPL.contains(
            "fn resolve_rl_inference_device(policy: &str) -> Result<(Device, String, String)>"
        ),
        "RL inference device resolution must be fallible"
    );
    assert!(
        RL_TESTS.contains("rl_cuda_tensor_launch_real_kernel"),
        "RL CUDA needs a mandatory real-device tensor test"
    );
    assert!(
        RL_TESTS.contains("explicit_rl_cuda_inference_device_is_fail_loud"),
        "RL CUDA inference needs an invalid-device refusal test"
    );
    assert!(
        !RL_TESTS.contains("cuda_if_available") && !RL_TESTS.contains("return; // skip"),
        "RL CUDA device test must not skip"
    );
}

#[cfg(feature = "reinforcement-learning-cuda")]
struct TempArtifactDir(PathBuf);

#[cfg(feature = "reinforcement-learning-cuda")]
impl TempArtifactDir {
    fn create() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-dqn-cuda-lifecycle-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create DQN CUDA artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(feature = "reinforcement-learning-cuda")]
impl Drop for TempArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove DQN CUDA artifact directory {}: {error}",
                self.0.display()
            );
        }
    }
}

#[cfg(feature = "reinforcement-learning-cuda")]
fn dqn_fixture(rows: usize, features: usize) -> (neoethos_data::FeatureFrame, Vec<i32>) {
    let columns = (0..features)
        .map(|feature_index| {
            FeatureColumnF64::new(
                format!("state_{feature_index}"),
                (0..rows)
                    .map(|row_index| {
                        let phase = row_index as f64 * (feature_index + 1) as f64 * 0.017;
                        phase.sin() + 0.2 * phase.cos()
                    })
                    .collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("build deterministic DQN CUDA feature")
        })
        .collect::<Vec<_>>();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("build DQN CUDA frame");
    let labels = (0..rows)
        .map(|row_index| match row_index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();
    (frame, labels)
}

#[cfg(feature = "reinforcement-learning-cuda")]
#[test]
fn dqn_cuda_surface_train_infer_save_load() {
    assert!(
        neoethos_models::tree_models::config::nvidia_gpu_count() > 0,
        "mandatory DQN CUDA lifecycle gate requires a visible NVIDIA device"
    );
    let (frame, labels) = dqn_fixture(512, 6);
    let width = WorkerLimit::new(1).expect("one DQN worker is valid");
    let lease = CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("DQN CUDA test can acquire one worker");
    let mut learner = TradingReinforcementLearner::new()
        .with_hidden_dims(vec![16, 16])
        .with_train_schedule(2, 128, 16)
        .with_update_schedule(16, 2)
        .with_buffer_capacity(1_024)
        .with_runtime_hints("rlkit", "gpu:0", 1, 2, 0, 1)
        .with_episode_layout(1, 64);

    learner
        .train_on_frame(&frame, &labels, &lease)
        .expect("mandatory DQN CUDA training must succeed without CPU fallback");
    let trained = learner
        .predict_runtime(&frame, &lease)
        .expect("mandatory DQN CUDA inference must succeed");
    assert_eq!(trained.len(), frame.n_samples());

    let artifact_dir = TempArtifactDir::create();
    learner
        .save(artifact_dir.path())
        .expect("persist DQN CUDA artifact");
    assert!(artifact_dir.path().join("q_network.safetensors").exists());
    let artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact_dir.path().join("rl_config.json"))
            .expect("read DQN CUDA runtime artifact"),
    )
    .expect("parse DQN CUDA runtime artifact");
    assert_eq!(artifact["requested_device_policy"], "gpu:0");
    assert_eq!(artifact["effective_device_policy"], "cuda:0");
    assert_eq!(artifact["effective_backend"], "rlkit_cuda");
    assert_eq!(artifact["backend"], "rlkit_cuda");
    assert_eq!(artifact["network_precision"], "fp32");

    let restored = TradingReinforcementLearner::load(artifact_dir.path())
        .expect("restore DQN artifact on exact CUDA ordinal zero");
    let restored_predictions = restored
        .predict_runtime(&frame, &lease)
        .expect("restored DQN CUDA inference must succeed");
    assert_eq!(trained.len(), restored_predictions.len());
    for (before, after) in trained.iter().zip(restored_predictions.iter()) {
        for (before_probability, after_probability) in before
            .class_probabilities()
            .iter()
            .zip(after.class_probabilities().iter())
        {
            assert!(
                (before_probability - after_probability).abs() <= 1e-6,
                "DQN save/load probability drift: trained={before_probability}, restored={after_probability}"
            );
        }
    }
}
