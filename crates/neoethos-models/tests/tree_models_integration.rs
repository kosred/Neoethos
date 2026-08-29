//! Native tree-model integration contracts.
//!
//! These tests use deterministic typed fixtures to exercise native training,
//! prediction, persistence, label mapping, and fail-closed device policy. Full
//! canonical-dataset quality evaluation belongs to the later search/training
//! gate, not to this fast package integration target.

#[cfg(any(feature = "lightgbm", feature = "xgboost", feature = "catboost"))]
mod support {
    #[cfg(feature = "lightgbm")]
    use std::path::{Path, PathBuf};
    #[cfg(feature = "lightgbm")]
    use std::time::{SystemTime, UNIX_EPOCH};

    use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
    use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};

    pub fn sample_frame(n_samples: usize, n_features: usize) -> (FeatureFrame, Vec<i32>) {
        let columns = (0..n_features)
            .map(|feature_index| {
                FeatureColumnF64::new(
                    format!("feature_{feature_index}"),
                    (0..n_samples)
                        .map(|row_index| (row_index as f64 * 0.1 + feature_index as f64) % 10.0)
                        .collect(),
                    vec![FeatureCellValidity::Valid; n_samples],
                )
                .expect("build deterministic tree feature")
            })
            .collect::<Vec<_>>();
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(n_samples),
            columns,
        )
        .expect("build deterministic typed tree frame");
        let labels = (0..n_samples)
            .map(|row_index| match row_index % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        (frame, labels)
    }

    pub fn one_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker is valid");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("tree integration test can acquire one worker")
    }

    #[cfg(feature = "lightgbm")]
    pub struct TempArtifactDir(PathBuf);

    #[cfg(feature = "lightgbm")]
    impl TempArtifactDir {
        pub fn create(prefix: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("neoethos-{prefix}-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&path).expect("create temporary tree artifact directory");
            Self(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    #[cfg(feature = "lightgbm")]
    impl Drop for TempArtifactDir {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                eprintln!(
                    "failed to remove tree integration artifact directory {}: {error}",
                    self.0.display()
                );
            }
        }
    }
}

#[cfg(feature = "lightgbm")]
mod lightgbm_tests {
    use std::collections::HashMap;

    use ndarray::Array2;
    use neoethos_models::tree_models::common::remap_labels_to_tree_targets;
    use neoethos_models::tree_models::config::ParamValue;
    use neoethos_models::tree_models::{LightGBMExpert, TreeModel, reorder_to_neutral_buy_sell};

    use super::support::{TempArtifactDir, one_worker_lease, sample_frame};

    fn cpu_lightgbm_params() -> HashMap<String, ParamValue> {
        HashMap::from([
            ("device".to_string(), ParamValue::String("cpu".to_string())),
            ("gpu_only".to_string(), ParamValue::Bool(false)),
        ])
    }

    #[test]
    fn lightgbm_trains_and_predicts_typed_probabilities() {
        let (frame, labels) = sample_frame(1_000, 10);
        let train = frame.row_window(0, 800).expect("training window");
        let test = frame.row_window(800, 1_000).expect("test window");
        let lease = one_worker_lease();
        let mut model = LightGBMExpert::new(1, Some(cpu_lightgbm_params()));

        model
            .fit(&train, &labels[..800], &lease)
            .expect("LightGBM training should succeed");
        let probabilities = model
            .predict_proba(&test, &lease)
            .expect("LightGBM prediction should succeed");
        assert_eq!(probabilities.dim(), (200, 3));
        for row in probabilities.outer_iter() {
            let sum = row.iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 0.01, "probability sum is {sum}");
            assert!(row.iter().all(|value| (0.0..=1.0).contains(value)));
        }
    }

    #[test]
    fn lightgbm_save_load_preserves_typed_prediction_shape() {
        let (frame, labels) = sample_frame(1_000, 10);
        let train = frame.row_window(0, 800).expect("training window");
        let test = frame.row_window(800, 1_000).expect("test window");
        let lease = one_worker_lease();
        let mut model = LightGBMExpert::new(1, Some(cpu_lightgbm_params()));
        model.fit(&train, &labels[..800], &lease).expect("fit");

        let artifact = TempArtifactDir::create("lightgbm-integration");
        model.save(artifact.path()).expect("save");
        let mut loaded = LightGBMExpert::new(1, None);
        loaded.load(artifact.path()).expect("load");
        assert_eq!(
            loaded.predict_proba(&test, &lease).expect("predict").dim(),
            (200, 3)
        );
    }

    #[test]
    fn tree_label_mapping_is_exact_and_typed() {
        let remapped = remap_labels_to_tree_targets(&[-1, 0, 1, -1, 0, 1, 1, 0, -1])
            .expect("remap canonical labels");
        assert_eq!(remapped, vec![2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn tree_output_reordering_refuses_missing_class_and_preserves_exact_order() {
        let binary =
            Array2::from_shape_vec((3, 2), vec![0.7, 0.3, 0.6, 0.4, 0.8, 0.2]).expect("binary");
        let error = reorder_to_neutral_buy_sell(binary, None)
            .expect_err("two classes cannot fabricate a neutral class");
        assert!(error.to_string().contains("exactly 3"));

        let multiclass =
            Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.7, 0.5, 0.3, 0.2]).expect("multiclass");
        let reordered = reorder_to_neutral_buy_sell(multiclass, Some(vec![0, 1, 2]))
            .expect("canonical class order");
        assert_eq!(reordered.row(0).to_vec(), vec![0.1, 0.2, 0.7]);
    }

    #[test]
    fn lightgbm_gpu_only_policy_refuses_resolved_cpu() {
        let (frame, labels) = sample_frame(1_000, 10);
        let train = frame.row_window(0, 800).expect("training window");
        let lease = one_worker_lease();
        let mut model = LightGBMExpert::new(
            1,
            Some(HashMap::from([
                ("device".to_string(), ParamValue::String("gpu".to_string())),
                ("gpu_only".to_string(), ParamValue::Bool(true)),
            ])),
        );

        let error = model
            .fit(&train, &labels[..800], &lease)
            .expect_err("GPU-only policy must never execute resolved CPU training");
        let message = error.to_string();
        assert!(
            message.contains("gpu-only mode is set")
                && message.contains("resolved device is `cpu`")
                && message.contains("models.tree_runtime.lightgbm_gpu"),
            "unexpected GPU-only error: {error}"
        );
    }
}

#[cfg(feature = "xgboost")]
mod xgboost_tests {
    use std::collections::HashMap;

    use neoethos_models::tree_models::config::ParamValue;
    use neoethos_models::tree_models::{TreeModel, XGBoostExpert};

    use super::support::{one_worker_lease, sample_frame};

    #[test]
    fn xgboost_trains_and_predicts_typed_probabilities() {
        let (frame, labels) = sample_frame(500, 5);
        let train = frame.row_window(0, 400).expect("training window");
        let test = frame.row_window(400, 500).expect("test window");
        let lease = one_worker_lease();
        let mut model = XGBoostExpert::new(
            1,
            Some(HashMap::from([
                ("device".to_string(), ParamValue::String("cpu".to_string())),
                ("gpu_only".to_string(), ParamValue::Bool(false)),
            ])),
        );

        model
            .fit(&train, &labels[..400], &lease)
            .expect("XGBoost training should succeed");
        assert_eq!(
            model.predict_proba(&test, &lease).expect("predict").dim(),
            (100, 3)
        );
    }
}

#[cfg(feature = "catboost")]
mod catboost_tests {
    use std::process::Command;

    use neoethos_models::tree_models::{CatBoostExpert, TreeModel};

    use super::support::{one_worker_lease, sample_frame};

    fn catboost_cli_available() -> bool {
        ["catboost", "catboost.exe"].into_iter().any(|candidate| {
            Command::new(candidate)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
        })
    }

    #[test]
    fn catboost_training_contract_is_typed_and_fail_closed() {
        let (frame, labels) = sample_frame(9, 2);
        let lease = one_worker_lease();
        let mut model = CatBoostExpert::new(1);
        let result = model.fit(&frame, &labels, &lease);

        if catboost_cli_available() {
            result.expect("CatBoost training should succeed when its CLI is available");
            assert_eq!(
                model.predict_proba(&frame, &lease).expect("predict").dim(),
                (9, 3)
            );
        } else {
            let error = result.expect_err("missing CatBoost CLI must fail closed");
            let message = error.to_string();
            assert!(
                message.contains("CatBoost CLI")
                    || message.contains("NEOETHOS_BOT_CATBOOST_EXECUTABLE")
                    || message.contains("CATBOOST_EXECUTABLE"),
                "unexpected missing-CLI error: {error}"
            );
        }
    }
}
