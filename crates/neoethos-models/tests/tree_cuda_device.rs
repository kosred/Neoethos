#![cfg(feature = "gpu-cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_models::tree_models::config::{
    ParamValue, install_tree_runtime_from_settings, nvidia_gpu_count,
};
use neoethos_models::tree_models::{
    CatBoostExpert, LightGBMExpert, SklearsTreeExpert, TreeModel, XGBoostExpert,
};
use neoethos_models::{
    CalibrationMethod, ConformalPredictionExpert, MetaBlender, MetaDecisionStack,
    ProbabilityCalibrationExpert,
};

const TREE_ROWS: usize = 8_192;
const TREE_FEATURES: usize = 8;
const META_ROWS: usize = 512;
const META_FEATURES: usize = 6;

fn install_mandatory_cuda_runtime() {
    let visible = nvidia_gpu_count();
    assert!(
        visible > 0,
        "mandatory tree CUDA lifecycle gate requires a visible NVIDIA device"
    );

    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let mut settings = neoethos_core::Settings::default();
        settings.models.tree_runtime.device = "gpu:0".to_string();
        settings.models.tree_runtime.gpu_only = true;
        settings.models.tree_runtime.gpu_count = None;
        settings.models.tree_runtime.lightgbm_gpu = true;
        install_tree_runtime_from_settings(&settings);
    });
}

fn fixture(rows: usize, features: usize) -> (FeatureFrame, Vec<i32>) {
    let columns = (0..features)
        .map(|feature_index| {
            FeatureColumnF64::new(
                format!("feature_{feature_index}"),
                (0..rows)
                    .map(|row_index| {
                        let phase = row_index as f64 * (feature_index + 1) as f64 * 0.003_7;
                        phase.sin() + 0.25 * phase.cos() + feature_index as f64 * 0.01
                    })
                    .collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .expect("build deterministic CUDA tree feature")
        })
        .collect::<Vec<_>>();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(rows),
        columns,
    )
    .expect("build deterministic typed CUDA tree frame");
    let labels = (0..rows)
        .map(|row_index| match row_index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();
    (frame, labels)
}

fn one_worker_lease() -> CpuLease {
    let width = WorkerLimit::new(1).expect("one worker is valid");
    CpuPermitBroker::new(width)
        .acquire(CpuPermitRequest::local(width))
        .expect("CUDA model test can acquire one worker")
}

struct TempArtifactDir(PathBuf);

impl TempArtifactDir {
    fn create(prefix: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("neoethos-{prefix}-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).expect("create CUDA model artifact directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempArtifactDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "failed to remove CUDA model artifact directory {}: {error}",
                self.0.display()
            );
        }
    }
}

fn assert_probabilities(probabilities: &ndarray::Array2<f64>, expected_rows: usize) {
    assert_eq!(probabilities.dim(), (expected_rows, 3));
    for row in probabilities.outer_iter() {
        let sum = row.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 0.01, "probability sum is {sum}");
        assert!(
            row.iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }
}

fn assert_probability_parity(trained: &ndarray::Array2<f64>, restored: &ndarray::Array2<f64>) {
    assert_eq!(trained.dim(), restored.dim());
    for (before, after) in trained.iter().zip(restored.iter()) {
        assert!(
            (before - after).abs() <= 1e-8,
            "save/load probability drift: trained={before}, restored={after}"
        );
    }
}

fn read_runtime_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).expect("read CUDA runtime artifact"))
        .expect("parse CUDA runtime artifact")
}

fn assert_xgboost_cuda_runtime(runtime: &serde_json::Value, expected_variant: &str) {
    assert_eq!(runtime["requested_device_policy"], "gpu:0");
    assert_eq!(runtime["effective_device"], "cuda:0");
    assert_eq!(runtime["effective_tree_method"], "hist");
    assert_eq!(runtime["predictor"], "gpu_predictor");
    assert_eq!(runtime["booster_variant"], expected_variant);
    assert_eq!(runtime["gpu_only"], true);
}

fn exercise_cuda_lifecycle<F>(
    name: &str,
    mut model: Box<dyn TreeModel>,
    mut restored: Box<dyn TreeModel>,
    frame: &FeatureFrame,
    labels: &[i32],
    runtime_relative_path: &Path,
    assert_runtime: F,
) where
    F: FnOnce(&serde_json::Value),
{
    let train_rows = frame.n_samples() * 4 / 5;
    let train = frame.row_window(0, train_rows).expect("training window");
    let test = frame
        .row_window(train_rows, frame.n_samples())
        .expect("inference window");
    let lease = one_worker_lease();

    model
        .fit(&train, &labels[..train_rows], &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} CUDA training failed: {error:#}"));
    let trained_probabilities = model
        .predict_proba(&test, &lease)
        .unwrap_or_else(|error| panic!("mandatory {name} CUDA inference failed: {error:#}"));
    assert_probabilities(&trained_probabilities, frame.n_samples() - train_rows);

    let artifact = TempArtifactDir::create(name);
    model
        .save(artifact.path())
        .unwrap_or_else(|error| panic!("persist {name} CUDA model: {error:#}"));
    let runtime = read_runtime_json(&artifact.path().join(runtime_relative_path));
    assert_runtime(&runtime);

    restored
        .load(artifact.path())
        .unwrap_or_else(|error| panic!("restore {name} CUDA model: {error:#}"));
    let restored_probabilities = restored
        .predict_proba(&test, &lease)
        .unwrap_or_else(|error| panic!("restored {name} CUDA inference failed: {error:#}"));
    assert_probabilities(&restored_probabilities, frame.n_samples() - train_rows);
    assert_probability_parity(&trained_probabilities, &restored_probabilities);
}

fn xgboost_params(variant: &str) -> HashMap<String, ParamValue> {
    let mut params = HashMap::from([
        (
            "device".to_string(),
            ParamValue::String("gpu:0".to_string()),
        ),
        ("gpu_only".to_string(), ParamValue::Bool(true)),
        (
            "tree_method".to_string(),
            ParamValue::String("hist".to_string()),
        ),
        (
            "variant".to_string(),
            ParamValue::String(variant.to_string()),
        ),
        ("n_estimators".to_string(), ParamValue::Int(8)),
        ("max_depth".to_string(), ParamValue::Int(4)),
    ]);
    if variant == "rf" {
        params.insert("n_estimators".to_string(), ParamValue::Int(2));
        params.insert("num_parallel_tree".to_string(), ParamValue::Int(4));
        params.insert("subsample".to_string(), ParamValue::Float(0.8));
        params.insert("colsample_bynode".to_string(), ParamValue::Float(0.8));
    }
    if variant == "dart" {
        params.insert("rate_drop".to_string(), ParamValue::Float(0.1));
        params.insert("skip_drop".to_string(), ParamValue::Float(0.5));
    }
    params
}

#[test]
fn xgboost_cuda_named_surfaces_train_infer_save_load() {
    install_mandatory_cuda_runtime();
    let (frame, labels) = fixture(TREE_ROWS, TREE_FEATURES);
    for (name, variant) in [
        ("xgboost", "gbtree"),
        ("xgboost_rf", "rf"),
        ("xgboost_dart", "dart"),
    ] {
        exercise_cuda_lifecycle(
            name,
            Box::new(XGBoostExpert::new(1, Some(xgboost_params(variant)))),
            Box::new(XGBoostExpert::new(99, None)),
            &frame,
            &labels,
            Path::new("xgboost_runtime.json"),
            |runtime| assert_xgboost_cuda_runtime(runtime, variant),
        );
    }
}

#[test]
fn xgboost_cuda_meta_surfaces_train_infer_save_load() {
    install_mandatory_cuda_runtime();
    let (frame, labels) = fixture(META_ROWS, META_FEATURES);
    let surfaces: Vec<(&str, Box<dyn TreeModel>, Box<dyn TreeModel>, PathBuf)> = vec![
        (
            "meta_blender",
            Box::new(MetaBlender::new()),
            Box::new(MetaBlender::new()),
            PathBuf::from("xgboost_backend/xgboost_runtime.json"),
        ),
        (
            "probability_calibrator",
            Box::new(ProbabilityCalibrationExpert::new(
                CalibrationMethod::Identity,
            )),
            Box::new(ProbabilityCalibrationExpert::new(
                CalibrationMethod::Identity,
            )),
            PathBuf::from("calibration_backend/xgboost_backend/xgboost_runtime.json"),
        ),
        (
            "conformal_gate",
            Box::new(ConformalPredictionExpert::new(
                CalibrationMethod::Identity,
                0.10,
            )),
            Box::new(ConformalPredictionExpert::new(
                CalibrationMethod::Identity,
                0.10,
            )),
            PathBuf::from("conformal_backend/xgboost_backend/xgboost_runtime.json"),
        ),
        (
            "meta_stack",
            Box::new(MetaDecisionStack::new(CalibrationMethod::Identity, 0.10)),
            Box::new(MetaDecisionStack::new(CalibrationMethod::Identity, 0.10)),
            PathBuf::from("blender_model/xgboost_backend/xgboost_runtime.json"),
        ),
    ];

    for (name, model, restored, runtime_path) in surfaces {
        exercise_cuda_lifecycle(
            name,
            model,
            restored,
            &frame,
            &labels,
            &runtime_path,
            |runtime| assert_xgboost_cuda_runtime(runtime, "gbtree"),
        );
    }
}

#[test]
fn lightgbm_cuda_surface_train_infer_save_load() {
    install_mandatory_cuda_runtime();
    let (frame, labels) = fixture(TREE_ROWS, TREE_FEATURES);
    let params = HashMap::from([
        (
            "device".to_string(),
            ParamValue::String("gpu:0".to_string()),
        ),
        ("gpu_only".to_string(), ParamValue::Bool(true)),
        ("num_iterations".to_string(), ParamValue::Int(12)),
        ("max_depth".to_string(), ParamValue::Int(4)),
        ("num_leaves".to_string(), ParamValue::Int(15)),
        ("verbosity".to_string(), ParamValue::Int(1)),
    ]);
    exercise_cuda_lifecycle(
        "lightgbm",
        Box::new(LightGBMExpert::new(1, Some(params))),
        Box::new(LightGBMExpert::new(99, None)),
        &frame,
        &labels,
        Path::new("runtime.json"),
        |runtime| {
            assert_eq!(runtime["requested_device_policy"], "gpu:0");
            assert_eq!(runtime["effective_device_type"], "cuda");
            assert_eq!(runtime["cuda_ordinal"], 0);
            assert_eq!(runtime["gpu_only"], true);
        },
    );
}

fn catboost_params(depth: i32, l2_leaf_reg: f64) -> HashMap<String, ParamValue> {
    HashMap::from([
        (
            "device".to_string(),
            ParamValue::String("gpu:0".to_string()),
        ),
        ("gpu_only".to_string(), ParamValue::Bool(true)),
        ("iterations".to_string(), ParamValue::Int(12)),
        ("depth".to_string(), ParamValue::Int(depth)),
        ("learning_rate".to_string(), ParamValue::Float(0.05)),
        ("l2_leaf_reg".to_string(), ParamValue::Float(l2_leaf_reg)),
        (
            "loss_function".to_string(),
            ParamValue::String("MultiClass".to_string()),
        ),
    ])
}

#[test]
fn catboost_cuda_named_surfaces_train_infer_save_load() {
    install_mandatory_cuda_runtime();
    let (frame, labels) = fixture(TREE_ROWS, TREE_FEATURES);
    for (name, depth, l2_leaf_reg) in [("catboost", 4, 6.0), ("catboost_alt", 6, 8.0)] {
        exercise_cuda_lifecycle(
            name,
            Box::new(CatBoostExpert::new_with_params(
                1,
                catboost_params(depth, l2_leaf_reg),
            )),
            Box::new(CatBoostExpert::new(99)),
            &frame,
            &labels,
            Path::new("runtime.json"),
            |runtime| {
                assert_eq!(runtime["requested_device_policy"], "gpu:0");
                assert_eq!(runtime["task_type"], "GPU");
                assert_eq!(runtime["cuda_ordinal"], 0);
                assert!(runtime["visible_nvidia_devices"].as_u64().unwrap_or(0) >= 1);
                assert_eq!(runtime["gpu_only"], true);
                assert_eq!(runtime["depth"], depth);
                assert_eq!(runtime["l2_leaf_reg"], l2_leaf_reg);
            },
        );
    }
}

#[test]
fn sklears_tree_cuda_surface_trains_infers_and_reopens_without_cpu_fallback() {
    install_mandatory_cuda_runtime();
    let (frame, labels) = fixture(TREE_ROWS, TREE_FEATURES);
    exercise_cuda_lifecycle(
        "sklears_tree",
        Box::new(SklearsTreeExpert::new()),
        Box::new(SklearsTreeExpert::new()),
        &frame,
        &labels,
        Path::new("runtime.json"),
        |runtime| {
            assert_eq!(runtime["requested_device_policy"], "gpu:0");
            assert_eq!(runtime["effective_device"], "cuda:0");
            assert_eq!(runtime["runtime_backend_kind"], "native_cuda");
            assert!(runtime["fit_kernel_launches"].as_u64().unwrap_or(0) > 0);
            assert!(runtime["predict_kernel_launches"].as_u64().unwrap_or(0) > 0);
            assert_eq!(runtime["cpu_fallback_used"], false);
        },
    );
}
