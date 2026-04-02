use anyhow::{Context, Result, bail};
#[cfg(feature = "lightgbm")]
use lightgbm3;
use ndarray::Array2;
use polars::prelude::*;
#[cfg(feature = "lightgbm")]
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "lightgbm")]
use std::path::PathBuf;

use crate::base::ExpertModel;
use crate::base::build_runtime_prediction;
use crate::base::feature_columns_from_dataframe;
use crate::runtime::artifacts::TrainingSummaryMetadata;
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

use super::common::{
    LIGHTGBM_MODEL_FILE_NAME, TreeLocalFallbackArtifact, atomic_write,
    build_tree_local_fallback_artifact, dataframe_to_row_major_vec, default_training_summary,
    ensure_feature_columns_match, predict_tree_local_fallback, read_runtime_metadata,
    remap_labels_to_tree_targets, reshape_three_class_probabilities, tree_artifact_paths,
    tree_runtime_metadata, write_runtime_metadata,
};
#[cfg(feature = "lightgbm")]
use super::config::{
    DevicePreference, ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_count, gpu_only_from_params, gpu_only_mode_for, param_bool,
    param_float, param_int, param_string, tree_device_preference_for,
};
#[cfg(not(feature = "lightgbm"))]
use super::config::{
    ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_only_from_params, gpu_only_mode_for,
    tree_device_preference_for,
};
use std::collections::HashMap;

#[cfg(feature = "lightgbm")]
const LIGHTGBM_RUNTIME_FILE_NAME: &str = "runtime.json";
const LIGHTGBM_LOCAL_FALLBACK_FILE_NAME: &str = "lightgbm_local_fallback.json";

#[cfg(feature = "lightgbm")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LightGBMRuntimeProfile {
    device_pref: DevicePreference,
    gpu_only: bool,
    cpu_threads: usize,
    params: HashMap<String, ParamValue>,
}

pub struct LightGBMExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    #[cfg_attr(not(feature = "lightgbm"), allow(dead_code))]
    gpu_only_disabled: bool,
    #[cfg_attr(not(feature = "lightgbm"), allow(dead_code))]
    feature_columns: Vec<String>,
    #[cfg_attr(not(feature = "lightgbm"), allow(dead_code))]
    training_summary: Option<TrainingSummaryMetadata>,
    local_fallback: Option<TreeLocalFallbackArtifact>,
    #[cfg(feature = "lightgbm")]
    model: Option<lightgbm3::Booster>,
    #[cfg(not(feature = "lightgbm"))]
    #[allow(dead_code)]
    model: Option<()>,
}

impl LightGBMExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let params = params.unwrap_or_else(Self::default_params);
        let device_pref =
            device_preference_from_params(&params, tree_device_preference_for("lightgbm"));
        let gpu_only = gpu_only_from_params(&params, gpu_only_mode_for("lightgbm"));
        let cpu_threads = cpu_threads_from_params(&params, cpu_threads_hint_for("lightgbm"));
        Self {
            idx,
            config: TreeModelConfig {
                idx,
                params,
                device_pref,
                gpu_only,
                cpu_threads: Some(cpu_threads),
            },
            gpu_only_disabled: false,
            feature_columns: Vec::new(),
            training_summary: None,
            local_fallback: None,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        params.insert("boosting_type".into(), ParamValue::String("gbdt".into()));
        params.insert("num_iterations".into(), ParamValue::Int(200));
        params.insert("learning_rate".into(), ParamValue::Float(0.05));
        params.insert("max_depth".into(), ParamValue::Int(8));
        params.insert("num_leaves".into(), ParamValue::Int(31));
        params.insert("min_data_in_bin".into(), ParamValue::Int(1));
        params.insert("min_data_in_leaf".into(), ParamValue::Int(1));
        params.insert("feature_fraction".into(), ParamValue::Float(1.0));
        params.insert("bagging_fraction".into(), ParamValue::Float(1.0));
        params.insert("bagging_freq".into(), ParamValue::Int(0));
        params.insert("min_gain_to_split".into(), ParamValue::Float(0.0));
        params.insert("lambda_l1".into(), ParamValue::Float(0.0));
        params.insert("lambda_l2".into(), ParamValue::Float(0.0));
        params.insert("max_bin".into(), ParamValue::Int(255));
        params.insert("verbosity".into(), ParamValue::Int(-1));
        params.insert("drop_rate".into(), ParamValue::Float(0.1));
        params.insert("skip_drop".into(), ParamValue::Float(0.5));
        params.insert("max_drop".into(), ParamValue::Int(50));
        params.insert("uniform_drop".into(), ParamValue::Bool(false));
        params
    }

    fn stored_training_summary(&self) -> TrainingSummaryMetadata {
        self.training_summary
            .clone()
            .unwrap_or_else(|| TrainingSummaryMetadata::new(0, 0, 0))
    }

    #[cfg(feature = "lightgbm")]
    fn runtime_profile(&self) -> LightGBMRuntimeProfile {
        LightGBMRuntimeProfile {
            device_pref: self.config.device_pref,
            gpu_only: self.config.gpu_only,
            cpu_threads: self.config.cpu_threads.unwrap_or(1).max(1),
            params: self.config.params.clone(),
        }
    }

    #[cfg(feature = "lightgbm")]
    fn apply_runtime_profile(&mut self, profile: LightGBMRuntimeProfile) {
        self.config.device_pref = profile.device_pref;
        self.config.gpu_only = profile.gpu_only;
        self.config.cpu_threads = Some(profile.cpu_threads.max(1));
        self.config.params = profile.params;
    }

    #[cfg(feature = "lightgbm")]
    fn runtime_profile_path(root: &Path) -> PathBuf {
        root.join(LIGHTGBM_RUNTIME_FILE_NAME)
    }

    fn local_fallback_path(root: &Path) -> std::path::PathBuf {
        root.join(LIGHTGBM_LOCAL_FALLBACK_FILE_NAME)
    }

    fn persist_local_fallback(&self, root: &Path) -> Result<()> {
        if let Some(artifact) = self.local_fallback.as_ref() {
            let payload =
                serde_json::to_vec_pretty(artifact).context("serialize LightGBM local fallback")?;
            atomic_write(&Self::local_fallback_path(root), &payload)?;
        }
        Ok(())
    }

    fn read_local_fallback(root: &Path) -> Result<Option<TreeLocalFallbackArtifact>> {
        let path = Self::local_fallback_path(root);
        if !path.exists() {
            return Ok(None);
        }
        let payload = std::fs::read(&path)
            .with_context(|| format!("read LightGBM fallback {}", path.display()))?;
        let artifact = serde_json::from_slice(&payload)
            .with_context(|| format!("deserialize LightGBM fallback {}", path.display()))?;
        Ok(Some(artifact))
    }

    #[cfg(feature = "lightgbm")]
    fn read_runtime_profile(root: &Path) -> Result<Option<LightGBMRuntimeProfile>> {
        let path = Self::runtime_profile_path(root);
        if !path.exists() {
            return Ok(None);
        }
        let payload = std::fs::read(&path)
            .with_context(|| format!("read LightGBM runtime profile from {}", path.display()))?;
        let profile = serde_json::from_slice(&payload).with_context(|| {
            format!(
                "deserialize LightGBM runtime profile from {}",
                path.display()
            )
        })?;
        Ok(Some(profile))
    }

    #[cfg(feature = "lightgbm")]
    fn prediction_params(&self) -> String {
        format!(
            "num_threads={}",
            self.config.cpu_threads.unwrap_or(1).max(1)
        )
    }

    #[cfg(feature = "lightgbm")]
    fn build_training_params(&self) -> serde_json::Value {
        let mut params = serde_json::json!({
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 3,
            "num_iterations": param_int(&self.config.params, "num_iterations", 200),
            "learning_rate": param_float(&self.config.params, "learning_rate", 0.05),
            "max_depth": param_int(&self.config.params, "max_depth", 8),
            "num_leaves": param_int(&self.config.params, "num_leaves", 31),
            "min_data_in_bin": param_int(&self.config.params, "min_data_in_bin", 1),
            "min_data_in_leaf": param_int(&self.config.params, "min_data_in_leaf", 1),
            "feature_fraction": param_float(&self.config.params, "feature_fraction", 1.0),
            "bagging_fraction": param_float(&self.config.params, "bagging_fraction", 1.0),
            "bagging_freq": param_int(&self.config.params, "bagging_freq", 0),
            "min_gain_to_split": param_float(&self.config.params, "min_gain_to_split", 0.0),
            "lambda_l1": param_float(&self.config.params, "lambda_l1", 0.0),
            "lambda_l2": param_float(&self.config.params, "lambda_l2", 0.0),
            "max_bin": param_int(&self.config.params, "max_bin", 255),
            "verbosity": param_int(&self.config.params, "verbosity", -1),
            "num_threads": self.config.cpu_threads.unwrap_or(1).max(1),
        });

        let boosting_type = param_string(&self.config.params, "boosting_type", "gbdt");
        params["boosting_type"] = serde_json::json!(boosting_type.clone());

        if boosting_type.eq_ignore_ascii_case("dart") {
            params["drop_rate"] =
                serde_json::json!(param_float(&self.config.params, "drop_rate", 0.1,));
            params["skip_drop"] =
                serde_json::json!(param_float(&self.config.params, "skip_drop", 0.5,));
            params["max_drop"] = serde_json::json!(param_int(&self.config.params, "max_drop", 50));
            params["uniform_drop"] =
                serde_json::json!(param_bool(&self.config.params, "uniform_drop", false,));
        }

        params
    }

    #[cfg(feature = "lightgbm")]
    fn normalize_probabilities(
        probabilities: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if probabilities.len() != rows.saturating_mul(cols) {
            anyhow::bail!(
                "LightGBM prediction shape mismatch: expected {} values for {}x{}, got {}",
                rows * cols,
                rows,
                cols,
                probabilities.len()
            );
        }

        let mut normalized = probabilities;
        for row in normalized.chunks_exact_mut(cols) {
            let mut sum = 0.0_f32;
            for value in row.iter_mut() {
                if !value.is_finite() {
                    anyhow::bail!("LightGBM predicted a non-finite probability: {value}");
                }
                if *value < 0.0 {
                    *value = 0.0;
                }
                sum += *value;
            }
            if sum > f32::EPSILON {
                for value in row.iter_mut() {
                    *value /= sum;
                }
            } else {
                bail!("LightGBM produced a degenerate probability row with zero total mass");
            }
        }

        Ok(normalized)
    }

    #[cfg(feature = "lightgbm")]
    fn runtime_predictions(
        model_name: &str,
        probabilities: &Array2<f32>,
    ) -> Result<Vec<RuntimePrediction>> {
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let confidence = row_values.iter().copied().fold(0.0_f32, f32::max);
            predictions.push(build_runtime_prediction(
                model_name.to_string(),
                ModelFamily::Tree,
                CapabilityState::Implemented,
                row_values,
                Some(confidence),
                Some(confidence < 0.5),
            )?);
        }

        Ok(predictions)
    }
}

impl ExpertModel for LightGBMExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        {
            if x.height() == 0 || y.is_empty() {
                bail!("LightGBM requires non-empty training features and labels");
            }
            if x.height() != y.len() {
                bail!(
                    "LightGBM requires matching feature and label rows: {} features vs {} labels",
                    x.height(),
                    y.len()
                );
            }

            self.feature_columns = feature_columns_from_dataframe(x);
            self.training_summary = Some(default_training_summary(x));
            self.local_fallback = Some(build_tree_local_fallback_artifact(
                x,
                y,
                self.stored_training_summary(),
            )?);
            self.gpu_only_disabled = false;
            self.model = None;
            Ok(())
        }
        #[cfg(feature = "lightgbm")]
        {
            if self.config.gpu_only && gpu_count() == 0 {
                self.gpu_only_disabled = true;
                self.model = None;
                anyhow::bail!("LightGBM gpu-only mode requested but no GPU is available");
            }

            let (flat_x, _rows, cols) = dataframe_to_row_major_vec(x)?;
            let labels = remap_labels_to_tree_targets(y)?;
            if labels.len() != x.height() {
                anyhow::bail!(
                    "LightGBM training row count mismatch: {} features rows, {} labels",
                    x.height(),
                    labels.len()
                );
            }
            let dataset = lightgbm3::Dataset::from_slice(&flat_x, &labels, cols as i32, true)
                .context("create LightGBM dataset from dataframe")?;

            let mut params = self.build_training_params();

            if matches!(
                self.config.device_pref,
                DevicePreference::Gpu | DevicePreference::Auto
            ) && gpu_count() > 0
                && !matches!(self.config.device_pref, DevicePreference::Cpu)
            {
                params["device_type"] = serde_json::json!("gpu");
            } else {
                params["device_type"] = serde_json::json!("cpu");
            }

            let model =
                lightgbm3::Booster::train(dataset, &params).context("train LightGBM booster")?;

            self.feature_columns = feature_columns_from_dataframe(x);
            self.training_summary = Some(default_training_summary(x));
            self.local_fallback = Some(build_tree_local_fallback_artifact(
                x,
                y,
                self.stored_training_summary(),
            )?);
            self.gpu_only_disabled = false;
            self.model = Some(model);
            Ok(())
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(not(feature = "lightgbm"))]
        let _ = x;
        if self.gpu_only_disabled {
            anyhow::bail!("LightGBM disabled: gpu-only mode requested without an available GPU");
        }
        #[cfg(feature = "lightgbm")]
        {
            ensure_feature_columns_match(&self.feature_columns, x)?;
            if self.model.is_none() {
                if let Some(fallback) = self.local_fallback.as_ref() {
                    return predict_tree_local_fallback(fallback, x);
                }
                bail!("LightGBM not trained");
            }
            let model = self.model.as_ref().context("LightGBM not trained")?;
            let (flat_x, rows, cols) = dataframe_to_row_major_vec(x)?;
            let probabilities = model
                .predict_with_params(&flat_x, cols as i32, true, &self.prediction_params())
                .context("predict LightGBM class probabilities")?
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let normalized = Self::normalize_probabilities(probabilities, rows, 3)?;
            reshape_three_class_probabilities(normalized, rows, 3)
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            let fallback = self
                .local_fallback
                .as_ref()
                .context("LightGBM local fallback not trained")?;
            predict_tree_local_fallback(fallback, x)
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        {
            std::fs::create_dir_all(path).with_context(|| {
                format!(
                    "create LightGBM fallback artifact directory {}",
                    path.display()
                )
            })?;
            let metadata = tree_runtime_metadata(
                "lightgbm",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            );
            let (_, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            self.persist_local_fallback(path)?;
            Ok(())
        }
        #[cfg(feature = "lightgbm")]
        {
            std::fs::create_dir_all(path).with_context(|| {
                format!("create LightGBM artifact directory {}", path.display())
            })?;
            let metadata = tree_runtime_metadata(
                "lightgbm",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            );
            let (model_path, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            if let Some(model) = self.model.as_ref() {
                model
                    .save_file(
                        model_path
                            .to_str()
                            .context("LightGBM artifact path must be valid unicode")?,
                    )
                    .with_context(|| format!("save LightGBM artifact {}", model_path.display()))?;
                let runtime_profile = self.runtime_profile();
                atomic_write(
                    &Self::runtime_profile_path(path),
                    &serde_json::to_vec_pretty(&runtime_profile)
                        .context("serialize LightGBM runtime profile")?,
                )?;
            } else if self.local_fallback.is_none() {
                bail!("LightGBM not trained");
            }
            self.persist_local_fallback(path)?;
            Ok(())
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        {
            let (_, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            let metadata = read_runtime_metadata(&metadata_path)?;
            if metadata.model_name != "lightgbm" || metadata.family != ModelFamily::Tree {
                bail!(
                    "LightGBM runtime metadata mismatch: expected tree/lightgbm, got {}/{}",
                    metadata.family,
                    metadata.model_name
                );
            }
            if metadata.feature_columns.is_empty() {
                bail!("LightGBM runtime metadata must contain at least one feature column");
            }
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            self.local_fallback = Self::read_local_fallback(path)?;
            self.model = None;
            self.gpu_only_disabled = false;
            Ok(())
        }
        #[cfg(feature = "lightgbm")]
        {
            let (model_path, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            let metadata = read_runtime_metadata(&metadata_path)?;
            if metadata.model_name != "lightgbm" || metadata.family != ModelFamily::Tree {
                bail!(
                    "LightGBM runtime metadata mismatch: expected tree/lightgbm, got {}/{}",
                    metadata.family,
                    metadata.model_name
                );
            }
            if metadata.feature_columns.is_empty() {
                bail!("LightGBM runtime metadata must contain at least one feature column");
            }
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            let native_model_result = if model_path.exists() {
                Some(
                    lightgbm3::Booster::from_file(
                        model_path
                            .to_str()
                            .context("LightGBM artifact path must be valid unicode")?,
                    )
                    .with_context(|| format!("load LightGBM artifact {}", model_path.display())),
                )
            } else {
                None
            };

            match native_model_result {
                Some(Ok(model)) => {
                    self.model = Some(model);
                    if let Some(profile) = Self::read_runtime_profile(path)? {
                        self.apply_runtime_profile(profile);
                    }
                    self.local_fallback = Self::read_local_fallback(path)?;
                }
                Some(Err(native_err)) => {
                    self.model = None;
                    self.local_fallback = Self::read_local_fallback(path)?;
                    if self.local_fallback.is_none() {
                        return Err(native_err);
                    }
                }
                None => {
                    self.model = None;
                    self.local_fallback = Self::read_local_fallback(path)?;
                    if self.local_fallback.is_none() {
                        bail!(
                            "LightGBM artifact {} is missing both native model and local fallback payload",
                            path.display()
                        );
                    }
                }
            }
            self.gpu_only_disabled = false;
            Ok(())
        }
    }
}

#[cfg(feature = "lightgbm")]
impl LightGBMExpert {
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        Self::runtime_predictions("lightgbm", &probabilities)
    }
}

#[cfg(all(test, feature = "lightgbm"))]
mod tests {
    use super::{ExpertModel, LightGBMExpert};
    use ndarray::Array2;
    use polars::df;
    use polars::prelude::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_three_class_dataset() -> (DataFrame, Series) {
        let mut momentum = Vec::new();
        let mut trend = Vec::new();
        let mut volatility = Vec::new();
        let mut labels = Vec::new();

        for idx in 0..24 {
            let offset = idx as f64 * 0.01;
            momentum.push(0.78 + offset);
            trend.push(0.7 + offset * 0.8);
            volatility.push(0.35 + offset * 0.2);
            labels.push(1_i32);
        }

        for idx in 0..24 {
            let offset = idx as f64 * 0.01;
            momentum.push(-0.02 + offset * 0.05);
            trend.push(-0.03 + offset * 0.04);
            volatility.push(0.12 + offset * 0.03);
            labels.push(0_i32);
        }

        for idx in 0..24 {
            let offset = idx as f64 * 0.01;
            momentum.push(-0.82 - offset);
            trend.push(-0.74 - offset * 0.9);
            volatility.push(0.48 + offset * 0.25);
            labels.push(-1_i32);
        }

        let x = df![
            "momentum" => momentum,
            "trend" => trend,
            "volatility" => volatility,
        ]
        .expect("build training dataframe");
        let y = Series::new("label".into(), labels);
        (x, y)
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{nonce}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn assert_rows_are_non_uniform(probabilities: &Array2<f32>) {
        assert_eq!(probabilities.ncols(), 3);
        assert!(
            probabilities.outer_iter().any(|row| {
                row.iter()
                    .any(|value| (value - (1.0_f32 / 3.0_f32)).abs() > 0.05_f32)
            }),
            "expected at least one non-uniform probability row, got {probabilities:?}"
        );
    }

    #[test]
    fn lightgbm_trains_three_class_probabilities_and_persists_artifacts() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-artifact");

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y).expect("fit should succeed");

        let probabilities = expert.predict_proba(&x).expect("predict should succeed");
        assert_eq!(probabilities.dim(), (x.height(), 3));
        assert_rows_are_non_uniform(&probabilities);

        expert.save(&artifact_dir).expect("save should succeed");
        assert!(
            artifact_dir.join("model.txt").exists(),
            "expected LightGBM model artifact at {}",
            artifact_dir.join("model.txt").display()
        );
        assert!(
            artifact_dir.join("metadata.json").exists(),
            "expected metadata sidecar at {}",
            artifact_dir.join("metadata.json").display()
        );
        assert!(
            artifact_dir.join("runtime.json").exists(),
            "expected runtime sidecar at {}",
            artifact_dir.join("runtime.json").display()
        );

        let mut loaded = LightGBMExpert::new(7, None);
        loaded.load(&artifact_dir).expect("load should succeed");
        let reloaded = loaded
            .predict_proba(&x)
            .expect("reloaded predict should succeed");

        for (lhs, rhs) in probabilities.iter().zip(reloaded.iter()) {
            assert!(
                (lhs - rhs).abs() < 1e-3_f32,
                "expected persisted probabilities to round-trip, left={lhs}, right={rhs}"
            );
        }
    }

    #[test]
    fn lightgbm_loads_fallback_when_native_artifact_is_corrupt() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-corrupt-artifact");

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");

        std::fs::write(artifact_dir.join("model.txt"), b"corrupt lightgbm model")
            .expect("overwrite native model artifact");

        let mut loaded = LightGBMExpert::new(7, None);
        loaded
            .load(&artifact_dir)
            .expect("load should recover from persisted fallback");

        let probabilities = loaded
            .predict_proba(&x)
            .expect("prediction should succeed from fallback");
        assert_eq!(probabilities.dim(), (x.height(), 3));
        for row in probabilities.outer_iter() {
            let sum = row.iter().copied().sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-3_f32);
        }
    }
}
