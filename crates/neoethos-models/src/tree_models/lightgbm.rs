// Tree-models LightGBM expert. Some helpers from `super::common`
// are imported for symmetry with the XGBoost variant — they're
// reached via the trait-object path so the compiler can't see the
// direct call sites. Gated `unused_imports` here keeps the import
// list aligned with the XGBoost sibling so a diff between the two
// is a substantive diff, not import noise.
#![allow(unused_imports)]

use anyhow::{Context, Result, bail};
#[cfg(feature = "lightgbm")]
use lightgbm3;
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::path::PathBuf;

use crate::base::{ExpertModel, feature_columns_from_dataframe};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;

use super::common::{
    LIGHTGBM_MODEL_FILE_NAME, TreeLocalFallbackArtifact, build_tree_local_fallback_artifact,
    build_tree_runtime_predictions, calibrate_three_class_probabilities,
    dataframe_to_row_major_vec, default_training_summary, ensure_feature_columns_match,
    normalize_three_class_probabilities, predict_tree_local_fallback, read_runtime_metadata,
    read_tree_json_artifact, remap_labels_to_tree_targets, reshape_three_class_probabilities,
    tree_artifact_paths, tree_runtime_metadata, validate_tree_local_fallback_artifact,
    write_runtime_metadata, write_tree_json_artifact,
};
#[cfg(feature = "lightgbm")]
use super::config::{
    DevicePreference, ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_count, gpu_only_from_params, gpu_only_mode_for, param_bool,
    param_float, param_int, param_string, tree_device_preference_for,
};
#[cfg(not(feature = "lightgbm"))]
use super::config::{
    DevicePreference, ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_count, gpu_only_from_params, gpu_only_mode_for, param_float,
    param_string, tree_device_preference_for,
};
use std::collections::HashMap;

const LIGHTGBM_RUNTIME_FILE_NAME: &str = "runtime.json";
const LIGHTGBM_LOCAL_FALLBACK_FILE_NAME: &str = "lightgbm_local_fallback.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LightGBMRuntimeArtifact {
    configured_params: HashMap<String, ParamValue>,
    resolved_params: HashMap<String, ParamValue>,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
    device_pref: DevicePreference,
    effective_device_type: String,
    boosting_type: String,
    probability_temperature: f64,
    gpu_only: bool,
    cpu_threads: usize,
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
        params.insert("probability_temperature".into(), ParamValue::Float(1.0));
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

    fn boosting_type(&self) -> String {
        param_string(&self.config.params, "boosting_type", "gbdt").to_lowercase()
    }

    fn probability_temperature(&self) -> f64 {
        let configured = param_float(&self.config.params, "probability_temperature", 1.0);
        if configured.is_finite() && configured > 0.0 {
            configured
        } else {
            1.0
        }
    }

    fn effective_device_type(&self) -> String {
        let gpu_available = gpu_count() > 0 && !self.gpu_only_disabled;
        match self.config.device_pref {
            DevicePreference::Gpu if gpu_available => "gpu".to_string(),
            DevicePreference::Auto if gpu_available => "gpu".to_string(),
            _ => "cpu".to_string(),
        }
    }

    fn resolved_params(&self) -> HashMap<String, ParamValue> {
        let mut params = self.config.params.clone();
        params.insert(
            "boosting_type".into(),
            ParamValue::String(self.boosting_type()),
        );
        params.insert(
            "device_type".into(),
            ParamValue::String(self.effective_device_type()),
        );
        params.insert(
            "probability_temperature".into(),
            ParamValue::Float(self.probability_temperature()),
        );
        params.insert("gpu_only".into(), ParamValue::Bool(self.config.gpu_only));
        params.insert(
            "cpu_threads".into(),
            ParamValue::Int(self.config.cpu_threads.unwrap_or(1).max(1) as i32),
        );
        params
    }

    fn runtime_artifact(&self) -> LightGBMRuntimeArtifact {
        LightGBMRuntimeArtifact {
            configured_params: self.config.params.clone(),
            resolved_params: self.resolved_params(),
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
            device_pref: self.config.device_pref,
            effective_device_type: self.effective_device_type(),
            boosting_type: self.boosting_type(),
            probability_temperature: self.probability_temperature(),
            gpu_only: self.config.gpu_only,
            cpu_threads: self.config.cpu_threads.unwrap_or(1).max(1),
        }
    }

    fn apply_runtime_artifact(&mut self, artifact: LightGBMRuntimeArtifact) {
        self.config.device_pref = artifact.device_pref;
        self.config.gpu_only = artifact.gpu_only;
        self.config.cpu_threads = Some(artifact.cpu_threads.max(1));
        self.config.params = artifact.configured_params;
        self.feature_columns = artifact.feature_columns;
        self.training_summary = Some(artifact.training_summary);
    }

    fn runtime_profile_path(root: &Path) -> PathBuf {
        root.join(LIGHTGBM_RUNTIME_FILE_NAME)
    }

    fn local_fallback_path(root: &Path) -> std::path::PathBuf {
        root.join(LIGHTGBM_LOCAL_FALLBACK_FILE_NAME)
    }

    fn persist_local_fallback(&self, root: &Path) -> Result<()> {
        if let Some(artifact) = self.local_fallback.as_ref() {
            validate_tree_local_fallback_artifact(artifact, &self.feature_columns)?;
            write_tree_json_artifact(
                &Self::local_fallback_path(root),
                artifact,
                "LightGBM local fallback",
            )?;
        }
        Ok(())
    }

    fn read_local_fallback(root: &Path) -> Result<Option<TreeLocalFallbackArtifact>> {
        let path = Self::local_fallback_path(root);
        if !path.exists() {
            return Ok(None);
        }
        let artifact = read_tree_json_artifact(&path, "LightGBM local fallback")?;
        Ok(Some(artifact))
    }

    fn read_runtime_artifact(root: &Path) -> Result<Option<LightGBMRuntimeArtifact>> {
        let path = Self::runtime_profile_path(root);
        if !path.exists() {
            return Ok(None);
        }
        let profile = read_tree_json_artifact(&path, "LightGBM runtime artifact")?;
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

    fn runtime_predictions(
        &self,
        model_name: &str,
        probabilities: &Array2<f32>,
    ) -> Result<Vec<RuntimePrediction>> {
        build_tree_runtime_predictions(
            model_name,
            probabilities,
            self.model.is_some(),
            "lightgbm_native",
            self.local_fallback.as_ref(),
            "native_lightgbm_unavailable",
            "lightgbm_unknown",
        )
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if self.feature_columns.is_empty() {
            bail!("LightGBM runtime state is missing persisted feature columns");
        }
        let summary = self
            .training_summary
            .as_ref()
            .context("LightGBM runtime state is missing training summary metadata")?;
        if summary.dataset_rows == 0 {
            bail!("LightGBM runtime state has zero dataset_rows in training summary");
        }
        if summary.dataset_rows != summary.train_rows + summary.val_rows {
            bail!(
                "LightGBM runtime state has inconsistent training summary: dataset_rows={} train_rows={} val_rows={}",
                summary.dataset_rows,
                summary.train_rows,
                summary.val_rows
            );
        }
        if self.model.is_none() && self.local_fallback.is_none() {
            bail!("LightGBM runtime state has neither a native booster nor a local surrogate");
        }
        if let Some(fallback) = self.local_fallback.as_ref() {
            validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
        }
        Ok(())
    }

    fn validate_runtime_artifact(
        artifact: &LightGBMRuntimeArtifact,
        expected_feature_columns: &[String],
        expected_training_summary: &TrainingSummaryMetadata,
    ) -> Result<()> {
        if artifact.feature_columns.is_empty() {
            bail!("LightGBM runtime artifact must contain feature columns");
        }
        if artifact.feature_columns != expected_feature_columns {
            bail!(
                "LightGBM runtime artifact feature-columns mismatch: expected {:?}, got {:?}",
                expected_feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.training_summary.dataset_rows != expected_training_summary.dataset_rows
            || artifact.training_summary.train_rows != expected_training_summary.train_rows
            || artifact.training_summary.val_rows != expected_training_summary.val_rows
        {
            bail!(
                "LightGBM runtime artifact training-summary mismatch: expected {:?}, got {:?}",
                expected_training_summary,
                artifact.training_summary
            );
        }
        if artifact.training_summary.dataset_rows == 0 {
            bail!("LightGBM runtime artifact must record non-zero dataset_rows");
        }
        if artifact.training_summary.dataset_rows
            != artifact.training_summary.train_rows + artifact.training_summary.val_rows
        {
            bail!("LightGBM runtime artifact training summary is inconsistent");
        }
        if artifact.configured_params.is_empty() {
            bail!("LightGBM runtime artifact must contain configured params");
        }
        if artifact.resolved_params.is_empty() {
            bail!("LightGBM runtime artifact must contain resolved params");
        }
        if !artifact.probability_temperature.is_finite() || artifact.probability_temperature <= 0.0
        {
            bail!("LightGBM runtime artifact probability_temperature must be finite and positive");
        }
        if artifact.cpu_threads == 0 {
            bail!("LightGBM runtime artifact cpu_threads must be greater than zero");
        }
        if artifact.boosting_type.trim().is_empty() {
            bail!("LightGBM runtime artifact boosting_type must not be blank");
        }
        if !matches!(artifact.effective_device_type.as_str(), "cpu" | "gpu") {
            bail!(
                "LightGBM runtime artifact effective_device_type must be 'cpu' or 'gpu', got {}",
                artifact.effective_device_type
            );
        }
        Ok(())
    }

    fn resolve_runtime_metadata(
        path: &Path,
        runtime_artifact: Option<&LightGBMRuntimeArtifact>,
        local_fallback: Option<&TreeLocalFallbackArtifact>,
    ) -> Result<RuntimeArtifactMetadata> {
        let (_, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
        if metadata_path.exists() {
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
            return Ok(metadata);
        }

        let (feature_columns, training_summary) = if let Some(runtime_artifact) = runtime_artifact {
            (
                runtime_artifact.feature_columns.clone(),
                runtime_artifact.training_summary.clone(),
            )
        } else if let Some(local_fallback) = local_fallback {
            (
                local_fallback.feature_columns.clone(),
                local_fallback.training_summary.clone(),
            )
        } else {
            bail!(
                "LightGBM metadata sidecar missing and no runtime/local artifact is available at {}",
                path.display()
            );
        };

        let metadata = tree_runtime_metadata("lightgbm", feature_columns, training_summary)?;
        tracing::warn!(
            path = %path.display(),
            "LightGBM metadata sidecar missing; reconstructing runtime metadata from persisted runtime artifacts"
        );
        Ok(metadata)
    }

    /// M6: shared body for `fit` and `fit_with_validation`. When `val_x`
    /// and `val_y` are supplied, builds a LightGBM eval dataset and uses
    /// `Booster::train_with_valid` so `early_stopping_rounds` from the
    /// training params is honoured. Without external val, falls back to
    /// the legacy `Booster::train` call which trains for the full
    /// `num_iterations`.
    fn fit_internal(
        &mut self,
        x: &DataFrame,
        y: &Series,
        val_x: Option<&DataFrame>,
        val_y: Option<&Series>,
    ) -> Result<()> {
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
            // Validation data is ignored in the no-feature path because the
            // local fallback surrogate does not support eval-set early
            // stopping. Surface a debug log so the caller can see why their
            // val frame was dropped.
            if val_x.is_some() && val_y.is_some() {
                tracing::debug!(
                    "LightGBM compiled without `lightgbm` feature; ignoring supplied val frame"
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

            let valid_dataset = match (val_x, val_y) {
                (Some(vx), Some(vy)) => {
                    if vx.width() != x.width() {
                        anyhow::bail!(
                            "LightGBM validation column count mismatch: train {}, val {}",
                            x.width(),
                            vx.width()
                        );
                    }
                    if vx.height() != vy.len() {
                        anyhow::bail!(
                            "LightGBM validation row/label mismatch: {} rows vs {} labels",
                            vx.height(),
                            vy.len()
                        );
                    }
                    let (vflat, _vrows, vcols) = dataframe_to_row_major_vec(vx)?;
                    let vlabels = remap_labels_to_tree_targets(vy)?;
                    let valid =
                        lightgbm3::Dataset::from_slice(&vflat, &vlabels, vcols as i32, true)
                            .context("create LightGBM validation dataset from dataframe")?;
                    // Default early_stopping_rounds when caller did not
                    // explicitly set one. 50 rounds is a conservative
                    // patience for `num_iterations >= 200`.
                    if !params
                        .get("early_stopping_rounds")
                        .is_some_and(|v| v.is_i64())
                    {
                        params["early_stopping_rounds"] = serde_json::json!(50);
                    }
                    Some(valid)
                }
                (None, None) => None,
                _ => bail!(
                    "LightGBMExpert::fit_with_validation requires both val_x and val_y or neither"
                ),
            };

            let model = lightgbm3::Booster::train_with_valid(dataset, valid_dataset, &params)
                .context("train LightGBM booster")?;

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
}

impl ExpertModel for LightGBMExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        self.fit_internal(x, y, None, None)
    }

    fn fit_with_validation(
        &mut self,
        x: &DataFrame,
        y: &Series,
        val_x: Option<&DataFrame>,
        val_y: Option<&Series>,
    ) -> Result<()> {
        self.fit_internal(x, y, val_x, val_y)
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
                    tracing::warn!(
                        model = "lightgbm",
                        surrogate_kind = %fallback.surrogate_kind,
                        surrogate_rows = fallback.training_summary.dataset_rows,
                        "LightGBM native booster unavailable during predict_proba; using local surrogate fallback"
                    );
                    let probabilities = predict_tree_local_fallback(fallback, x)?;
                    let probabilities = calibrate_three_class_probabilities(
                        probabilities,
                        self.probability_temperature() as f32,
                        "LightGBM",
                    )?;
                    return normalize_three_class_probabilities(probabilities, "LightGBM");
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
            let probabilities = reshape_three_class_probabilities(normalized, rows, 3)?;
            let probabilities = calibrate_three_class_probabilities(
                probabilities,
                self.probability_temperature() as f32,
                "LightGBM",
            )?;
            normalize_three_class_probabilities(probabilities, "LightGBM")
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            let fallback = self
                .local_fallback
                .as_ref()
                .context("LightGBM local fallback not trained")?;
            let probabilities = predict_tree_local_fallback(fallback, x)?;
            let probabilities = calibrate_three_class_probabilities(
                probabilities,
                self.probability_temperature() as f32,
                "LightGBM",
            )?;
            normalize_three_class_probabilities(probabilities, "LightGBM")
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
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
            )?;
            let (_, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            let runtime_profile = self.runtime_artifact();
            Self::validate_runtime_artifact(
                &runtime_profile,
                &self.feature_columns,
                &self.stored_training_summary(),
            )?;
            write_tree_json_artifact(
                &Self::runtime_profile_path(path),
                &runtime_profile,
                "LightGBM runtime artifact",
            )?;
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
            )?;
            let (model_path, metadata_path) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            let runtime_profile = self.runtime_artifact();
            Self::validate_runtime_artifact(
                &runtime_profile,
                &self.feature_columns,
                &self.stored_training_summary(),
            )?;
            write_tree_json_artifact(
                &Self::runtime_profile_path(path),
                &runtime_profile,
                "LightGBM runtime artifact",
            )?;
            if let Some(model) = self.model.as_ref() {
                model
                    .save_file(
                        model_path
                            .to_str()
                            .context("LightGBM artifact path must be valid unicode")?,
                    )
                    .with_context(|| format!("save LightGBM artifact {}", model_path.display()))?;
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
            let runtime_profile = Self::read_runtime_artifact(path)?;
            self.local_fallback = Self::read_local_fallback(path)?;
            let metadata = Self::resolve_runtime_metadata(
                path,
                runtime_profile.as_ref(),
                self.local_fallback.as_ref(),
            )?;
            let metadata_feature_columns = metadata.feature_columns.clone();
            let metadata_training_summary = metadata.training_summary.clone();
            if let Some(runtime_profile) = runtime_profile {
                Self::validate_runtime_artifact(
                    &runtime_profile,
                    &metadata_feature_columns,
                    &metadata_training_summary,
                )?;
                self.apply_runtime_artifact(runtime_profile);
            } else {
                self.feature_columns = metadata_feature_columns;
                self.training_summary = Some(metadata_training_summary);
                tracing::warn!(
                    path = %path.display(),
                    "LightGBM runtime.json missing; using metadata/local fallback to restore runtime state"
                );
            }
            if let Some(fallback) = self.local_fallback.as_ref() {
                validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
            }
            self.model = None;
            self.gpu_only_disabled = false;
            Ok(())
        }
        #[cfg(feature = "lightgbm")]
        {
            let (model_path, _) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            let runtime_profile = Self::read_runtime_artifact(path)?;
            self.local_fallback = Self::read_local_fallback(path)?;
            let metadata = Self::resolve_runtime_metadata(
                path,
                runtime_profile.as_ref(),
                self.local_fallback.as_ref(),
            )?;
            let metadata_feature_columns = metadata.feature_columns.clone();
            let metadata_training_summary = metadata.training_summary.clone();
            if let Some(runtime_profile) = runtime_profile {
                Self::validate_runtime_artifact(
                    &runtime_profile,
                    &metadata_feature_columns,
                    &metadata_training_summary,
                )?;
                self.apply_runtime_artifact(runtime_profile);
            } else {
                self.feature_columns = metadata_feature_columns;
                self.training_summary = Some(metadata_training_summary);
                tracing::warn!(
                    path = %path.display(),
                    "LightGBM runtime.json missing; using metadata/local fallback to restore runtime state"
                );
            }
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
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                    }
                }
                Some(Err(native_err)) => {
                    self.model = None;
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                        tracing::warn!(
                            model = "lightgbm",
                            path = %path.display(),
                            surrogate_kind = %fallback.surrogate_kind,
                            surrogate_rows = fallback.training_summary.dataset_rows,
                            error = %native_err,
                            "failed to restore native LightGBM booster; using local surrogate fallback"
                        );
                    } else {
                        return Err(native_err);
                    }
                }
                None => {
                    self.model = None;
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                        tracing::warn!(
                            model = "lightgbm",
                            path = %path.display(),
                            surrogate_kind = %fallback.surrogate_kind,
                            surrogate_rows = fallback.training_summary.dataset_rows,
                            "LightGBM artifact missing native booster; using local surrogate fallback"
                        );
                    } else {
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

impl LightGBMExpert {
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        self.runtime_predictions("lightgbm", &probabilities)
    }

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`]
    /// adapter so the registry / aggregator can detect column-layout
    /// drift after a retraining session.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }
}

#[cfg(all(test, feature = "lightgbm"))]
mod tests {
    use super::{ExpertModel, LightGBMExpert, build_tree_local_fallback_artifact};
    use crate::runtime::artifacts::TrainingSummaryMetadata;
    use crate::tree_models::config::{DevicePreference, ParamValue};
    use ndarray::Array2;
    use polars::df;
    use polars::prelude::*;
    use std::collections::HashMap;
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

    #[test]
    fn lightgbm_load_uses_runtime_profile_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-missing-metadata-sidecar");

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("metadata.json"))
            .expect("remove metadata sidecar to force fallback path");

        let mut loaded = LightGBMExpert::new(7, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct runtime metadata from runtime profile");

        let probabilities = loaded
            .predict_proba(&x)
            .expect("prediction should succeed after metadata reconstruction");
        assert_eq!(probabilities.dim(), (x.height(), 3));
    }

    #[test]
    fn lightgbm_load_uses_metadata_when_runtime_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-missing-runtime-sidecar");

        let mut expert = LightGBMExpert::new(11, None);
        expert.fit(&x, &y).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("runtime.json"))
            .expect("remove runtime sidecar to force metadata/local fallback path");

        let mut loaded = LightGBMExpert::new(11, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct runtime state from metadata/local fallback");

        let probabilities = loaded
            .predict_proba(&x)
            .expect("prediction should succeed after runtime reconstruction");
        assert_eq!(probabilities.dim(), (x.height(), 3));
    }

    #[test]
    fn lightgbm_validate_runtime_artifact_rejects_invalid_probability_temperature() {
        let artifact = super::LightGBMRuntimeArtifact {
            configured_params: HashMap::from([
                ("boosting_type".into(), ParamValue::String("gbdt".into())),
                ("probability_temperature".into(), ParamValue::Float(1.0)),
            ]),
            resolved_params: HashMap::from([
                ("device_type".into(), ParamValue::String("cpu".into())),
                ("cpu_threads".into(), ParamValue::Int(4)),
            ]),
            feature_columns: vec!["momentum".into(), "trend".into()],
            training_summary: TrainingSummaryMetadata::new(9, 9, 0),
            device_pref: DevicePreference::Cpu,
            effective_device_type: "cpu".into(),
            boosting_type: "gbdt".into(),
            probability_temperature: 0.0,
            gpu_only: false,
            cpu_threads: 4,
        };

        let err = LightGBMExpert::validate_runtime_artifact(
            &artifact,
            &["momentum".into(), "trend".into()],
            &TrainingSummaryMetadata::new(9, 9, 0),
        )
        .expect_err("non-positive probability_temperature should fail");
        assert!(err.to_string().contains("probability_temperature"));
    }

    #[test]
    fn lightgbm_save_rejects_missing_training_summary() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-missing-summary");

        let mut expert = LightGBMExpert::new(7, None);
        expert.feature_columns = vec!["momentum".into(), "trend".into(), "volatility".into()];
        expert.local_fallback = Some(
            build_tree_local_fallback_artifact(
                &x,
                &y,
                TrainingSummaryMetadata::new(x.height(), x.height(), 0),
            )
            .expect("fallback artifact"),
        );
        expert.training_summary = None;

        let err = expert
            .save(&artifact_dir)
            .expect_err("save should fail without training summary");
        assert!(err.to_string().contains("training summary"));
    }
}
