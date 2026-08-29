use ndarray::Array2;
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::BayesianLogitExpert;
use neoethos_models::base::ExpertModel;
use neoethos_models::statistical::common::install_statistical_runtime_from_settings;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const CHILD_ROLE_ENV: &str = "NEOETHOS_BAYES_R5_CHILD_ROLE";
const GPU_POLICY: &str = "gpu:0";
const MODEL_FILE: &str = "model.json";
const METADATA_FILE: &str = "metadata.json";

struct TestArtifactDir(PathBuf);

impl TestArtifactDir {
    fn create(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-bayes-r5-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create parent-owned Bayesian artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove Bayesian R5 artifact directory {}: {error}",
                self.0.display()
            );
        }
    }
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is a legal test budget");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("acquire isolated one-worker lease")
}

/// Deterministic, finite, non-perfectly-separable three-class data. The labels
/// use the public model's accepted {-1, 0, 1} vocabulary. Every 37th row is a
/// deterministic label perturbation, so a real optimiser must converge rather
/// than pass through an impossible perfect-separation shortcut.
fn legal_fixture(rows: usize, features: usize) -> (FeatureFrame, Vec<i32>) {
    assert!(
        rows >= 128,
        "fixture needs enough rows for all temporal partitions"
    );
    assert!(
        features >= 4,
        "fixture needs at least four informative dimensions"
    );

    let latent_classes = (0..rows).map(|row| row % 3).collect::<Vec<_>>();
    let columns = (0..features)
        .map(|feature| {
            let values = latent_classes
                .iter()
                .enumerate()
                .map(|(row, class)| {
                    let class_signal = match (feature % 4, class) {
                        (0, 0) | (1, 1) | (2, 2) => 1.6,
                        (0, 2) | (1, 0) | (2, 1) => -1.1,
                        _ => 0.25,
                    };
                    let phase = (row * (feature + 3)) as f64 * 0.017;
                    class_signal + 0.18 * phase.sin() + 0.07 * (phase * 0.37).cos()
                })
                .collect::<Vec<_>>();
            FeatureColumnF64::new(
                format!("bayes_r5_feature_{feature}"),
                values,
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("construct a legal finite Bayesian feature column")
        })
        .collect::<Vec<_>>();

    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("construct legal public FeatureFrame");
    let labels = latent_classes
        .into_iter()
        .enumerate()
        .map(|(row, class)| {
            let class = if row % 37 == 0 {
                (class + 1) % 3
            } else {
                class
            };
            [-1, 0, 1][class]
        })
        .collect::<Vec<_>>();
    assert_eq!(frame.n_samples(), rows);
    assert_eq!(frame.n_features(), features);
    assert!(labels.iter().any(|label| *label == -1));
    assert!(labels.iter().any(|label| *label == 0));
    assert!(labels.iter().any(|label| *label == 1));
    (frame, labels)
}

fn assert_probability_matrix(probabilities: &Array2<f64>, rows: usize) {
    assert_eq!(probabilities.dim(), (rows, 3));
    assert!(
        !probabilities.is_empty(),
        "empty probabilities cannot satisfy repeatability"
    );
    for row in probabilities.outer_iter() {
        assert!(row.iter().all(|value| value.is_finite()));
        let sum = row.sum();
        assert!((sum - 1.0).abs() <= 1e-10, "probability row sums to {sum}");
        assert!(row.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}

fn assert_exact_bits(left: &Array2<f64>, right: &Array2<f64>, context: &str) {
    assert_eq!(left.dim(), right.dim(), "{context}: shape drift");
    assert!(!left.is_empty(), "{context}: empty matrices are forbidden");
    for (index, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{context}: f64 bit drift at flat probability index {index}"
        );
    }
}

fn assert_actual_fit_threshold(probabilities: &Array2<f64>, labels: &[i32]) {
    assert_probability_matrix(probabilities, labels.len());
    let mut correct = 0usize;
    let mut log_loss = 0.0_f64;
    for (row, label) in probabilities.outer_iter().zip(labels) {
        let expected = match label {
            -1 => 2,
            0 => 0,
            1 => 1,
            unexpected => panic!("fixture emitted illegal label {unexpected}"),
        };
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("three-class probability row is non-empty")
            .0;
        correct += usize::from(predicted == expected);
        log_loss -= row[expected].max(f64::MIN_POSITIVE).ln();
    }
    let accuracy = correct as f64 / labels.len() as f64;
    let mean_log_loss = log_loss / labels.len() as f64;
    assert!(
        accuracy >= 0.80,
        "public Bayesian fit did not converge: accuracy={accuracy:.6} < 0.80"
    );
    assert!(
        mean_log_loss <= 0.65,
        "public Bayesian fit did not converge: log_loss={mean_log_loss:.6} > 0.65"
    );
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap_or_else(|error| {
        panic!(
            "read parent-owned JSON artifact {}: {error}",
            path.display()
        )
    }))
    .unwrap_or_else(|error| panic!("parse JSON artifact {}: {error}", path.display()))
}

fn write_json(path: &Path, value: &Value) {
    let bytes = serde_json::to_vec_pretty(value).expect("serialize parent mutation");
    fs::write(path, bytes)
        .unwrap_or_else(|error| panic!("write parent mutation {}: {error}", path.display()));
}

fn assert_gpu_artifact(artifact_dir: &Path) {
    let model = read_json(&artifact_dir.join(MODEL_FILE));
    assert_eq!(model["model_name"], "bayes_logit");
    assert_eq!(model["requested_device_policy"], GPU_POLICY);
    assert_eq!(model["effective_device_policy"], GPU_POLICY);
    assert_eq!(model["runtime_backend_kind"], "native_cuda");
    assert_eq!(model["runtime_degraded_reason"], Value::Null);
    let backend = model["runtime_backend"]
        .as_str()
        .expect("GPU artifact must persist a typed runtime backend");
    assert!(backend.contains("cuda"), "backend is not CUDA: {backend}");
    assert!(
        backend.contains(GPU_POLICY),
        "backend lost CUDA ordinal: {backend}"
    );
    assert!(
        !backend.contains("cpu"),
        "GPU artifact self-identifies as CPU: {backend}"
    );

    let metadata = read_json(&artifact_dir.join(METADATA_FILE));
    assert_eq!(metadata["model_name"], "bayes_logit");
    assert_eq!(metadata["requested_device_policy"], GPU_POLICY);
    assert_eq!(metadata["effective_device_policy"], GPU_POLICY);
    assert_eq!(metadata["runtime_backend_kind"], "native_cuda");
    assert_eq!(metadata["runtime_degraded_reason"], Value::Null);
}

fn run_cpu_fixture_threshold() {
    let mut settings = neoethos_core::Settings::default();
    settings.models.statistical_device = "cpu".to_string();
    install_statistical_runtime_from_settings(&settings);

    let (frame, labels) = legal_fixture(512, 8);
    let lease = one_worker_lease();
    let mut model = BayesianLogitExpert::new();
    model.epochs = 300;
    model.learning_rate = 0.05;
    model.prior_precision = 0.05;
    ExpertModel::fit(&mut model, &frame, &labels, &lease)
        .expect("public CPU Bayesian fit must converge on the legal fixture");
    let probabilities = ExpertModel::predict_proba(&model, &frame, &lease)
        .expect("public CPU Bayesian prediction must succeed");
    assert_actual_fit_threshold(&probabilities, &labels);
}

fn run_gpu_public_lifecycle() {
    let mut settings = neoethos_core::Settings::default();
    settings.models.statistical_device = GPU_POLICY.to_string();
    install_statistical_runtime_from_settings(&settings);

    let (frame, labels) = legal_fixture(512, 8);
    let lease = one_worker_lease();
    let mut model = BayesianLogitExpert::new();
    model.epochs = 300;
    model.learning_rate = 0.05;
    model.prior_precision = 0.05;

    // On the reviewed preimplementation source this real public call reaches
    // bayesian_impl.rs::ExpertModel::fit -> cpu_backend_for_policy and fails
    // because no production Bayesian GPU lane exists.
    ExpertModel::fit(&mut model, &frame, &labels, &lease)
        .expect("mandatory native-CUDA Bayesian public fit failed");
    let first = ExpertModel::predict_proba(&model, &frame, &lease)
        .expect("mandatory native-CUDA Bayesian public prediction failed");
    let repeated = ExpertModel::predict_proba(&model, &frame, &lease)
        .expect("same-device repeat prediction failed");
    assert_actual_fit_threshold(&first, &labels);
    assert_exact_bits(&first, &repeated, "same-device public prediction");

    let artifact_dir = TestArtifactDir::create("public-lifecycle");
    ExpertModel::save(&model, artifact_dir.path()).expect("save genuine public GPU artifact");
    assert_gpu_artifact(artifact_dir.path());

    let mut restored = BayesianLogitExpert::new();
    ExpertModel::load(&mut restored, artifact_dir.path())
        .expect("load genuine public GPU artifact");
    let after_load = ExpertModel::predict_proba(&restored, &frame, &lease)
        .expect("restored GPU artifact prediction failed");
    assert_exact_bits(&first, &after_load, "public save/load");

    let stable_before_mutation = after_load;
    let model_path = artifact_dir.path().join(MODEL_FILE);
    let metadata_path = artifact_dir.path().join(METADATA_FILE);
    let original_model = fs::read(&model_path).expect("snapshot genuine model artifact bytes");
    let original_metadata =
        fs::read(&metadata_path).expect("snapshot genuine metadata artifact bytes");

    let mut corrupt_model = read_json(&model_path);
    corrupt_model["precision_schema"] = Value::String("neoethos.corrupt.f32".to_string());
    write_json(&model_path, &corrupt_model);
    ExpertModel::load(&mut restored, artifact_dir.path())
        .expect_err("corrupted precision schema must be rejected");
    let after_corrupt_model = ExpertModel::predict_proba(&restored, &frame, &lease)
        .expect("failed model load must preserve receiver state");
    assert_exact_bits(
        &stable_before_mutation,
        &after_corrupt_model,
        "transactional state after model corruption",
    );

    fs::write(&model_path, &original_model).expect("restore genuine model bytes");
    let mut drifted_metadata = read_json(&metadata_path);
    drifted_metadata["effective_device_policy"] = Value::String("cpu".to_string());
    write_json(&metadata_path, &drifted_metadata);
    ExpertModel::load(&mut restored, artifact_dir.path())
        .expect_err("GPU-to-CPU metadata drift must be rejected");
    let after_metadata_drift = ExpertModel::predict_proba(&restored, &frame, &lease)
        .expect("failed metadata load must preserve receiver state");
    assert_exact_bits(
        &stable_before_mutation,
        &after_metadata_drift,
        "transactional state after metadata drift",
    );
    fs::write(&metadata_path, original_metadata).expect("restore genuine metadata bytes");
}

fn run_child(role: &str) -> Output {
    Command::new(std::env::current_exe().expect("locate current integration-test executable"))
        .args(["--exact", "r5_child_public_api", "--ignored", "--nocapture"])
        .env(CHILD_ROLE_ENV, role)
        .output()
        .unwrap_or_else(|error| panic!("launch isolated Bayesian R5 child role {role}: {error}"))
}

fn assert_child_success(role: &str, output: Output) {
    assert!(
        output.status.success(),
        "Bayesian R5 child role `{role}` failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn legal_public_fixture_reaches_real_fit_threshold() {
    assert_child_success("cpu-threshold", run_child("cpu-threshold"));
}

#[test]
#[ignore = "real-card lifecycle is executed only by the single serialized paid R5 parent"]
fn gpu_public_fit_predict_save_load_is_mandatory() {
    assert_child_success("gpu-lifecycle", run_child("gpu-lifecycle"));
}

#[test]
#[ignore = "private child entrypoint; parent tests own role selection and verdicts"]
fn r5_child_public_api() {
    match std::env::var(CHILD_ROLE_ENV).as_deref() {
        Ok("cpu-threshold") => run_cpu_fixture_threshold(),
        Ok("gpu-lifecycle") => run_gpu_public_lifecycle(),
        Ok(unexpected) => panic!("unknown Bayesian R5 child role `{unexpected}`"),
        Err(error) => panic!("Bayesian R5 child role is parent-owned: {error}"),
    }
}
