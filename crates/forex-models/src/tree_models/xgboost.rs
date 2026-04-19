use super::common::{
    atomic_write, build_tree_local_fallback_artifact, build_tree_runtime_predictions,
    calibrate_three_class_probabilities, dataframe_to_row_major_vec, default_training_summary,
    ensure_feature_columns_match, normalize_three_class_probabilities, predict_tree_local_fallback,
    read_runtime_metadata, remap_labels_to_tree_targets, tree_artifact_paths,
    tree_runtime_metadata, validate_tree_local_fallback_artifact, write_runtime_metadata,
    TreeLocalFallbackArtifact, XGBOOST_MODEL_FILE_NAME,
};
use super::config::*;
use crate::base::ExpertModel;
use crate::base::{compute_sample_weights, feature_columns_from_dataframe};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;
use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "xgboost")]
use super::common::reshape_three_class_probabilities;
#[cfg(feature = "xgboost")]
use xgb;
#[cfg(feature = "xgboost")]
use xgb::parameters::learning::{
    EvaluationMetric, LearningTaskParametersBuilder, Metrics, Objective,
};
#[cfg(feature = "xgboost")]
use xgb::parameters::tree::{Predictor, TreeBoosterParametersBuilder, TreeMethod};
#[cfg(feature = "xgboost")]
use xgb::parameters::{BoosterParametersBuilder, BoosterType};
#[cfg(feature = "xgboost")]
use xgb::{PredictConfig, PredictType};

const XGBOOST_RUNTIME_FILE_NAME: &str = "xgboost_runtime.json";
const XGBOOST_LOCAL_RUNTIME_FILE_NAME: &str = "xgboost_local_runtime.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct XGBoostRuntimeArtifact {
    configured_params: HashMap<String, ParamValue>,
    resolved_params: HashMap<String, ParamValue>,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
    device_pref: DevicePreference,
    booster_variant: String,
    configured_tree_method: String,
    effective_tree_method: String,
    objective: String,
    predictor: String,
    num_parallel_tree: u32,
    probability_temperature: f64,
    gpu_only: bool,
    cpu_threads: Option<usize>,
}

pub struct XGBoostExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg_attr(not(feature = "xgboost"), allow(dead_code))]
    pub(crate) feature_columns: Vec<String>,
    #[cfg_attr(not(feature = "xgboost"), allow(dead_code))]
    training_summary: Option<TrainingSummaryMetadata>,
    local_fallback: Option<TreeLocalFallbackArtifact>,
    #[cfg(feature = "xgboost")]
    _model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    _model: Option<()>,
}

impl XGBoostExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let params = params.unwrap_or_else(Self::default_params);
        let device_pref =
            device_preference_from_params(&params, tree_device_preference_for("xgboost"));
        let gpu_only = gpu_only_from_params(&params, gpu_only_mode_for("xgboost"));
        let cpu_threads = cpu_threads_from_params(&params, cpu_threads_hint_for("xgboost"));
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
            _model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut p = HashMap::new();
        p.insert("variant".into(), ParamValue::String("gbtree".into()));
        p.insert("n_estimators".into(), ParamValue::Int(800));
        p.insert("max_depth".into(), ParamValue::Int(8));
        p.insert("learning_rate".into(), ParamValue::Float(0.05));
        p.insert(
            "objective".into(),
            ParamValue::String("multi:softprob".into()),
        );
        p.insert("num_class".into(), ParamValue::Int(3));
        p.insert("tree_method".into(), ParamValue::String("hist".into()));
        p.insert("subsample".into(), ParamValue::Float(1.0));
        p.insert("colsample_bytree".into(), ParamValue::Float(1.0));
        p.insert("colsample_bylevel".into(), ParamValue::Float(1.0));
        p.insert("colsample_bynode".into(), ParamValue::Float(1.0));
        p.insert("num_parallel_tree".into(), ParamValue::Int(1));
        p.insert("probability_temperature".into(), ParamValue::Float(1.0));
        p.insert("rate_drop".into(), ParamValue::Float(0.0));
        p.insert("skip_drop".into(), ParamValue::Float(0.0));
        p.insert("one_drop".into(), ParamValue::Bool(false));
        p
    }

    #[cfg(feature = "xgboost")]
    fn configured_tree_method(&self) -> TreeMethod {
        match self.config.params.get("tree_method") {
            Some(ParamValue::String(value)) => match value.as_str() {
                "auto" => TreeMethod::Auto,
                "exact" => TreeMethod::Exact,
                "approx" => TreeMethod::Approx,
                "hist" => TreeMethod::Hist,
                "gpu_exact" => TreeMethod::GpuExact,
                "gpu_hist" => TreeMethod::GpuHist,
                _ => TreeMethod::Hist,
            },
            _ => TreeMethod::Hist,
        }
    }

    #[cfg(feature = "xgboost")]
    fn effective_tree_method(&self) -> TreeMethod {
        let gpu_available = gpu_count() > 0;
        match self.config.device_pref {
            DevicePreference::Gpu if gpu_available => match self.configured_tree_method() {
                TreeMethod::Auto | TreeMethod::Hist => TreeMethod::GpuHist,
                TreeMethod::Exact => TreeMethod::GpuExact,
                other => other,
            },
            DevicePreference::Auto if gpu_available => match self.configured_tree_method() {
                TreeMethod::Auto | TreeMethod::Hist => TreeMethod::GpuHist,
                TreeMethod::Exact => TreeMethod::GpuExact,
                other => other,
            },
            _ => self.configured_tree_method(),
        }
    }

    #[cfg(feature = "xgboost")]
    fn predictor(&self) -> Predictor {
        let gpu_available = gpu_count() > 0;
        match self.config.device_pref {
            DevicePreference::Gpu if gpu_available => Predictor::Gpu,
            DevicePreference::Gpu => Predictor::Cpu,
            DevicePreference::Cpu => Predictor::Cpu,
            DevicePreference::Auto if gpu_available => Predictor::Gpu,
            DevicePreference::Auto => Predictor::Cpu,
        }
    }

    #[cfg(feature = "xgboost")]
    fn booster_variant(&self) -> String {
        param_string(&self.config.params, "variant", "gbtree").to_lowercase()
    }

    #[cfg(feature = "xgboost")]
    fn tree_num_parallel(&self) -> u32 {
        let configured = param_int(&self.config.params, "num_parallel_tree", 1).max(1) as u32;
        if self.booster_variant() == "rf" && configured == 1 {
            64
        } else {
            configured
        }
    }

    #[cfg(feature = "xgboost")]
    fn tree_subsample(&self) -> f32 {
        let configured = param_float(&self.config.params, "subsample", 1.0) as f32;
        if self.booster_variant() == "rf" && (configured - 1.0).abs() < f32::EPSILON {
            0.8
        } else {
            configured
        }
    }

    #[cfg(feature = "xgboost")]
    fn tree_colsample_bytree(&self) -> f32 {
        let configured = param_float(&self.config.params, "colsample_bytree", 1.0) as f32;
        if self.booster_variant() == "rf" && (configured - 1.0).abs() < f32::EPSILON {
            0.8
        } else {
            configured
        }
    }

    #[cfg(feature = "xgboost")]
    fn tree_colsample_bynode(&self) -> f32 {
        let configured = param_float(&self.config.params, "colsample_bynode", 1.0) as f32;
        if self.booster_variant() == "rf" && (configured - 1.0).abs() < f32::EPSILON {
            0.8
        } else {
            configured
        }
    }

    fn probability_temperature(&self) -> f64 {
        let configured = param_float(&self.config.params, "probability_temperature", 1.0);
        if configured.is_finite() && configured > 0.0 {
            configured
        } else {
            1.0
        }
    }

    #[cfg(feature = "xgboost")]
    fn runtime_params(&self) -> HashMap<String, ParamValue> {
        let mut params = self.config.params.clone();
        params.insert("variant".into(), ParamValue::String(self.booster_variant()));
        params.insert(
            "tree_method".into(),
            ParamValue::String(self.effective_tree_method().to_string()),
        );
        params.insert(
            "objective".into(),
            ParamValue::String(param_string(
                &self.config.params,
                "objective",
                "multi:softprob",
            )),
        );
        params.insert(
            "predictor".into(),
            ParamValue::String(self.predictor().to_string()),
        );
        params.insert(
            "num_parallel_tree".into(),
            ParamValue::Int(self.tree_num_parallel() as i32),
        );
        params.insert(
            "probability_temperature".into(),
            ParamValue::Float(self.probability_temperature()),
        );
        params.insert(
            "subsample".into(),
            ParamValue::Float(self.tree_subsample() as f64),
        );
        params.insert(
            "colsample_bytree".into(),
            ParamValue::Float(self.tree_colsample_bytree() as f64),
        );
        params.insert(
            "colsample_bylevel".into(),
            ParamValue::Float(param_float(&self.config.params, "colsample_bylevel", 1.0)),
        );
        params.insert(
            "colsample_bynode".into(),
            ParamValue::Float(self.tree_colsample_bynode() as f64),
        );
        params.insert("gpu_only".into(), ParamValue::Bool(self.config.gpu_only));
        if let Some(cpu_threads) = self.config.cpu_threads {
            params.insert(
                "cpu_threads".into(),
                ParamValue::Int(cpu_threads.max(1) as i32),
            );
        }
        params
    }

    fn runtime_artifact(&self) -> XGBoostRuntimeArtifact {
        XGBoostRuntimeArtifact {
            configured_params: self.config.params.clone(),
            resolved_params: self.runtime_params(),
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
            device_pref: self.config.device_pref,
            booster_variant: self.booster_variant(),
            configured_tree_method: self.configured_tree_method().to_string(),
            effective_tree_method: self.effective_tree_method().to_string(),
            objective: param_string(&self.config.params, "objective", "multi:softprob"),
            predictor: self.predictor().to_string(),
            num_parallel_tree: self.tree_num_parallel(),
            probability_temperature: self.probability_temperature(),
            gpu_only: self.config.gpu_only,
            cpu_threads: self.config.cpu_threads,
        }
    }

    fn local_runtime_artifact(&self) -> Option<TreeLocalFallbackArtifact> {
        self.local_fallback.clone()
    }

    fn persist_local_runtime_artifact(&self, path: &Path) -> Result<()> {
        if let Some(artifact) = self.local_runtime_artifact() {
            validate_tree_local_fallback_artifact(&artifact, &self.feature_columns)?;
            let payload = serde_json::to_vec_pretty(&artifact)
                .context("serialize XGBoost local fallback artifact")?;
            atomic_write(&path.join(XGBOOST_LOCAL_RUNTIME_FILE_NAME), &payload)?;
        }
        Ok(())
    }

    fn read_local_runtime_artifact(path: &Path) -> Result<Option<TreeLocalFallbackArtifact>> {
        let artifact_path = path.join(XGBOOST_LOCAL_RUNTIME_FILE_NAME);
        if !artifact_path.exists() {
            return Ok(None);
        }
        let payload = std::fs::read(&artifact_path).with_context(|| {
            format!(
                "read XGBoost local fallback artifact {}",
                artifact_path.display()
            )
        })?;
        let artifact = serde_json::from_slice(&payload).with_context(|| {
            format!(
                "deserialize XGBoost local fallback artifact {}",
                artifact_path.display()
            )
        })?;
        Ok(Some(artifact))
    }

    #[cfg(feature = "xgboost")]
    fn set_runtime_attributes(&self, model: &mut xgb::Booster) -> Result<()> {
        let feature_refs = self
            .feature_columns
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>();
        model
            .set_feature_names(&feature_refs)
            .context("set XGBoost feature names")?;
        model
            .set_attribute("model_name", "xgboost")
            .context("set XGBoost model_name attribute")?;
        model
            .set_attribute("booster_variant", &self.booster_variant())
            .context("set XGBoost booster_variant attribute")?;
        model
            .set_attribute(
                "configured_tree_method",
                &self.configured_tree_method().to_string(),
            )
            .context("set XGBoost configured_tree_method attribute")?;
        model
            .set_attribute("tree_method", &self.effective_tree_method().to_string())
            .context("set XGBoost tree_method attribute")?;
        model
            .set_attribute(
                "objective",
                &param_string(&self.config.params, "objective", "multi:softprob"),
            )
            .context("set XGBoost objective attribute")?;
        model
            .set_attribute("predictor", &self.predictor().to_string())
            .context("set XGBoost predictor attribute")?;
        model
            .set_attribute("num_parallel_tree", &self.tree_num_parallel().to_string())
            .context("set XGBoost num_parallel_tree attribute")?;
        model
            .set_attribute(
                "probability_temperature",
                &self.probability_temperature().to_string(),
            )
            .context("set XGBoost probability_temperature attribute")?;
        model
            .set_attribute(
                "gpu_only",
                if self.config.gpu_only {
                    "true"
                } else {
                    "false"
                },
            )
            .context("set XGBoost gpu_only attribute")?;
        if let Some(cpu_threads) = self.config.cpu_threads {
            model
                .set_attribute("cpu_threads", &cpu_threads.max(1).to_string())
                .context("set XGBoost cpu_threads attribute")?;
        }
        Ok(())
    }

    #[cfg(feature = "xgboost")]
    fn persist_runtime_artifact(&self, path: &Path) -> Result<()> {
        let payload = serde_json::to_vec_pretty(&self.runtime_artifact())
            .context("serialize XGBoost runtime artifact")?;
        atomic_write(&path.join(XGBOOST_RUNTIME_FILE_NAME), &payload)
    }

    #[cfg(feature = "xgboost")]
    fn read_runtime_artifact(path: &Path) -> Result<Option<XGBoostRuntimeArtifact>> {
        let artifact_path = path.join(XGBOOST_RUNTIME_FILE_NAME);
        if !artifact_path.exists() {
            return Ok(None);
        }
        let payload = std::fs::read(&artifact_path).with_context(|| {
            format!("read XGBoost runtime artifact {}", artifact_path.display())
        })?;
        let artifact = serde_json::from_slice(&payload).with_context(|| {
            format!(
                "deserialize XGBoost runtime artifact {}",
                artifact_path.display()
            )
        })?;
        Ok(Some(artifact))
    }

    #[cfg(feature = "xgboost")]
    fn apply_variant_params(&self, model: &mut xgb::Booster) -> Result<()> {
        if self.booster_variant() != "dart" {
            return Ok(());
        }

        model
            .set_param("booster", "dart")
            .context("set XGBoost booster variant to dart")?;
        model
            .set_param(
                "rate_drop",
                &param_float(&self.config.params, "rate_drop", 0.1).to_string(),
            )
            .context("set XGBoost dart rate_drop")?;
        model
            .set_param(
                "skip_drop",
                &param_float(&self.config.params, "skip_drop", 0.5).to_string(),
            )
            .context("set XGBoost dart skip_drop")?;
        model
            .set_param(
                "one_drop",
                if param_bool(&self.config.params, "one_drop", false) {
                    "1"
                } else {
                    "0"
                },
            )
            .context("set XGBoost dart one_drop")?;
        Ok(())
    }

    fn stored_training_summary(&self) -> TrainingSummaryMetadata {
        self.training_summary
            .clone()
            .unwrap_or_else(|| TrainingSummaryMetadata::new(0, 0, 0))
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if self.feature_columns.is_empty() {
            bail!("XGBoost runtime state is missing persisted feature columns");
        }
        let summary = self
            .training_summary
            .as_ref()
            .context("XGBoost runtime state is missing training summary metadata")?;
        if summary.dataset_rows == 0 {
            bail!("XGBoost runtime state has zero dataset_rows in training summary");
        }
        if summary.dataset_rows != summary.train_rows + summary.val_rows {
            bail!(
                "XGBoost runtime state has inconsistent training summary: dataset_rows={} train_rows={} val_rows={}",
                summary.dataset_rows,
                summary.train_rows,
                summary.val_rows
            );
        }
        if self._model.is_none() && self.local_fallback.is_none() {
            bail!("XGBoost runtime state has neither a native booster nor a local surrogate");
        }
        if let Some(fallback) = self.local_fallback.as_ref() {
            validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
        }
        Ok(())
    }

    fn validate_runtime_artifact(
        artifact: &XGBoostRuntimeArtifact,
        expected_feature_columns: &[String],
        expected_training_summary: &TrainingSummaryMetadata,
    ) -> Result<()> {
        if artifact.feature_columns.is_empty() {
            bail!("XGBoost runtime artifact must contain feature columns");
        }
        if artifact.feature_columns != expected_feature_columns {
            bail!(
                "XGBoost runtime artifact feature-columns mismatch: expected {:?}, got {:?}",
                expected_feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.training_summary.dataset_rows != expected_training_summary.dataset_rows
            || artifact.training_summary.train_rows != expected_training_summary.train_rows
            || artifact.training_summary.val_rows != expected_training_summary.val_rows
        {
            bail!(
                "XGBoost runtime artifact training-summary mismatch: expected {:?}, got {:?}",
                expected_training_summary,
                artifact.training_summary
            );
        }
        if artifact.training_summary.dataset_rows == 0 {
            bail!("XGBoost runtime artifact must record non-zero dataset_rows");
        }
        if artifact.training_summary.dataset_rows
            != artifact.training_summary.train_rows + artifact.training_summary.val_rows
        {
            bail!("XGBoost runtime artifact training summary is inconsistent");
        }
        if !artifact.probability_temperature.is_finite() || artifact.probability_temperature <= 0.0
        {
            bail!("XGBoost runtime artifact probability_temperature must be finite and positive");
        }
        if artifact.num_parallel_tree == 0 {
            bail!("XGBoost runtime artifact num_parallel_tree must be greater than zero");
        }
        for (field, value) in [
            ("booster_variant", artifact.booster_variant.as_str()),
            (
                "configured_tree_method",
                artifact.configured_tree_method.as_str(),
            ),
            (
                "effective_tree_method",
                artifact.effective_tree_method.as_str(),
            ),
            ("objective", artifact.objective.as_str()),
            ("predictor", artifact.predictor.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("XGBoost runtime artifact `{field}` may not be blank");
            }
        }
        if artifact.resolved_params.is_empty() {
            bail!("XGBoost runtime artifact must persist resolved runtime params");
        }
        Ok(())
    }

    fn build_local_fallback_artifact(
        &self,
        x: &DataFrame,
        y: &Series,
    ) -> Result<TreeLocalFallbackArtifact> {
        build_tree_local_fallback_artifact(x, y, self.stored_training_summary())
    }

    fn local_predict_proba_from_artifact(
        artifact: &TreeLocalFallbackArtifact,
        x: &DataFrame,
    ) -> Result<Array2<f32>> {
        predict_tree_local_fallback(artifact, x)
    }

    fn calibrate_probabilities(&self, probabilities: Array2<f32>) -> Result<Array2<f32>> {
        calibrate_three_class_probabilities(
            probabilities,
            self.probability_temperature() as f32,
            "XGBoost",
        )
    }

    fn normalize_probabilities(probabilities: Array2<f32>) -> Result<Array2<f32>> {
        normalize_three_class_probabilities(probabilities, "XGBoost")
    }
}

impl ExpertModel for XGBoostExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if x.height() == 0 || y.is_empty() {
                anyhow::bail!("XGBoost requires non-empty training features and labels");
            }
            if x.height() != y.len() {
                anyhow::bail!(
                    "XGBoost requires matching feature and label rows: {} features vs {} labels",
                    x.height(),
                    y.len()
                );
            }

            if self.config.gpu_only && gpu_count() == 0 {
                self.gpu_only_disabled = true;
                self._model = None;
                anyhow::bail!("XGBoost gpu-only mode requested but no GPU is available");
            }

            let effective_tree_method = self.effective_tree_method();
            if matches!(
                effective_tree_method,
                TreeMethod::GpuExact | TreeMethod::GpuHist
            ) && gpu_count() == 0
            {
                self._model = None;
                anyhow::bail!(
                    "XGBoost tree_method `{}` requires a GPU, but none is available",
                    effective_tree_method
                );
            }

            let (flat_x, n_rows, _n_cols) = dataframe_to_row_major_vec(x)?;
            let labels = remap_labels_to_tree_targets(y)?;
            let sample_weights = compute_sample_weights(y)?;

            let mut dtrain = xgb::DMatrix::from_dense(&flat_x, n_rows)
                .context("create XGBoost training matrix from dataframe")?;
            dtrain
                .set_labels(&labels)
                .context("set XGBoost training labels")?;
            dtrain
                .set_weights(&sample_weights)
                .context("set XGBoost sample weights")?;

            let tree_params =
                TreeBoosterParametersBuilder::default()
                    .eta(param_float(&self.config.params, "learning_rate", 0.05) as f32)
                    .max_depth(param_int(&self.config.params, "max_depth", 8).max(1) as u32)
                    .subsample(self.tree_subsample())
                    .colsample_bytree(self.tree_colsample_bytree())
                    .colsample_bylevel(
                        param_float(&self.config.params, "colsample_bylevel", 1.0) as f32
                    )
                    .colsample_bynode(self.tree_colsample_bynode())
                    .num_parallel_tree(self.tree_num_parallel())
                    .tree_method(effective_tree_method)
                    .predictor(self.predictor())
                    .build()
                    .context("build XGBoost tree booster parameters")?;

            let learning_params = LearningTaskParametersBuilder::default()
                .objective(Objective::MultiSoftprob(3))
                .eval_metrics(Metrics::Custom(vec![EvaluationMetric::MultiClassLogLoss]))
                .build()
                .context("build XGBoost learning parameters")?;

            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .threads(self.config.cpu_threads.map(|threads| threads as u32))
                .verbose(false)
                .build()
                .context("build XGBoost booster parameters")?;

            let boost_rounds = param_int(&self.config.params, "n_estimators", 800).max(1) as u32;

            let mut model = xgb::Booster::new_with_cached_dmats(&booster_params, &[&dtrain])
                .context("create XGBoost booster")?;
            self.apply_variant_params(&mut model)?;
            self.feature_columns = feature_columns_from_dataframe(x);
            self.set_runtime_attributes(&mut model)?;
            for iteration in 0..boost_rounds as i32 {
                model
                    .update(&dtrain, iteration)
                    .with_context(|| format!("update XGBoost booster at iteration {iteration}"))?;
            }
            self.training_summary = Some(default_training_summary(x));
            self.local_fallback = Some(self.build_local_fallback_artifact(x, y)?);
            self.gpu_only_disabled = false;
            self._model = Some(model);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            if x.height() == 0 || y.is_empty() {
                anyhow::bail!("XGBoost requires non-empty training features and labels");
            }
            if x.height() != y.len() {
                anyhow::bail!(
                    "XGBoost requires matching feature and label rows: {} features vs {} labels",
                    x.height(),
                    y.len()
                );
            }

            self.feature_columns = feature_columns_from_dataframe(x);
            self.training_summary = Some(default_training_summary(x));
            self.local_fallback = Some(self.build_local_fallback_artifact(x, y)?);
            self.gpu_only_disabled = false;
            self._model = None;
            Ok(())
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if self.gpu_only_disabled {
            anyhow::bail!("XGBoost disabled: gpu-only mode requested without an available GPU");
        }
        #[cfg(feature = "xgboost")]
        {
            if x.height() == 0 {
                return Ok(Array2::zeros((0, 3)));
            }

            ensure_feature_columns_match(&self.feature_columns, x)?;
            if self._model.is_none() {
                if let Some(fallback) = self.local_fallback.as_ref() {
                    tracing::warn!(
                        model = "xgboost",
                        surrogate_kind = %fallback.surrogate_kind,
                        surrogate_rows = fallback.training_summary.dataset_rows,
                        "XGBoost native booster unavailable during predict_proba; using local surrogate fallback"
                    );
                    let probabilities = Self::local_predict_proba_from_artifact(fallback, x)?;
                    let probabilities = self.calibrate_probabilities(probabilities)?;
                    return Self::normalize_probabilities(probabilities);
                }
                anyhow::bail!("XGBoost not trained");
            }
            let model = self._model.as_ref().context("XGBoost not trained")?;
            let (flat_x, n_rows, _) = dataframe_to_row_major_vec(x)?;
            let dtest = xgb::DMatrix::from_dense(&flat_x, n_rows)
                .context("create XGBoost prediction matrix from dataframe")?;
            let prediction_config = PredictConfig {
                _type: PredictType::Normal,
                training: false,
                iteration_begin: 0,
                iteration_end: 0,
                strict_shape: true,
            };
            let (probabilities, shape) = model
                .predict_matrix(&dtest, &prediction_config.as_json())
                .context("predict XGBoost class probabilities")?;
            let cols = match shape.as_slice() {
                [rows, cols] if *rows as usize == n_rows => *cols as usize,
                [_cols] => {
                    if n_rows == 0 {
                        0
                    } else {
                        probabilities.len() / n_rows
                    }
                }
                _ => {
                    if n_rows == 0 {
                        0
                    } else {
                        probabilities.len() / n_rows
                    }
                }
            };
            let probabilities = reshape_three_class_probabilities(probabilities, n_rows, cols)?;
            let probabilities = self.calibrate_probabilities(probabilities)?;
            Self::normalize_probabilities(probabilities)
        }
        #[cfg(not(feature = "xgboost"))]
        {
            let fallback = self
                .local_fallback
                .as_ref()
                .context("XGBoost local fallback not trained")?;
            let probabilities = Self::local_predict_proba_from_artifact(fallback, x)?;
            let probabilities = self.calibrate_probabilities(probabilities)?;
            Self::normalize_probabilities(probabilities)
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        #[cfg(feature = "xgboost")]
        {
            std::fs::create_dir_all(path)
                .with_context(|| format!("create XGBoost artifact directory {}", path.display()))?;
            let metadata = tree_runtime_metadata(
                "xgboost",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            )?;
            let (model_path, metadata_path) = tree_artifact_paths(path, XGBOOST_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            if let Some(model) = self._model.as_ref() {
                Self::validate_runtime_artifact(
                    &self.runtime_artifact(),
                    &self.feature_columns,
                    &self.stored_training_summary(),
                )?;
                model
                    .save(&model_path)
                    .with_context(|| format!("save XGBoost artifact {}", model_path.display()))?;
                self.persist_runtime_artifact(path)?;
            } else if self.local_fallback.is_none() {
                bail!("XGBoost not trained");
            }
            self.persist_local_runtime_artifact(path)?;
        }
        #[cfg(not(feature = "xgboost"))]
        {
            std::fs::create_dir_all(path)
                .with_context(|| format!("create XGBoost artifact directory {}", path.display()))?;
            let metadata = tree_runtime_metadata(
                "xgboost",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            )?;
            let (_, metadata_path) = tree_artifact_paths(path, XGBOOST_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            self.persist_local_runtime_artifact(path)?;
        }
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let (model_path, metadata_path) = tree_artifact_paths(path, XGBOOST_MODEL_FILE_NAME);
            let runtime_artifact = Self::read_runtime_artifact(path)?;
            self.local_fallback = Self::read_local_runtime_artifact(path)?;
            let metadata: RuntimeArtifactMetadata = if metadata_path.exists() {
                let metadata = read_runtime_metadata(&metadata_path)?;
                if metadata.model_name != "xgboost" || metadata.family != ModelFamily::Tree {
                    bail!(
                        "XGBoost runtime metadata mismatch: expected tree/xgboost, got {}/{}",
                        metadata.family,
                        metadata.model_name
                    );
                }
                if metadata.feature_columns.is_empty() {
                    bail!("XGBoost runtime metadata must contain at least one feature column");
                }
                metadata
            } else {
                let (feature_columns, training_summary) = if let Some(artifact) =
                    runtime_artifact.as_ref()
                {
                    (
                        artifact.feature_columns.clone(),
                        artifact.training_summary.clone(),
                    )
                } else if let Some(fallback) = self.local_fallback.as_ref() {
                    (
                        fallback.feature_columns.clone(),
                        fallback.training_summary.clone(),
                    )
                } else {
                    bail!(
                        "XGBoost metadata sidecar missing and no runtime/local artifact is available at {}",
                        path.display()
                    );
                };
                let metadata = tree_runtime_metadata("xgboost", feature_columns, training_summary)?;
                tracing::warn!(
                    path = %path.display(),
                    "XGBoost metadata sidecar missing; reconstructing from persisted runtime artifacts"
                );
                metadata
            };
            let metadata_feature_columns = metadata.feature_columns.clone();
            let metadata_training_summary = metadata.training_summary.clone();
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            if let Some(fallback) = self.local_fallback.as_ref() {
                validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
            }
            if let Some(artifact) = runtime_artifact {
                Self::validate_runtime_artifact(
                    &artifact,
                    &metadata_feature_columns,
                    &metadata_training_summary,
                )?;
                let XGBoostRuntimeArtifact {
                    configured_params,
                    resolved_params,
                    feature_columns,
                    training_summary,
                    device_pref,
                    booster_variant,
                    configured_tree_method,
                    effective_tree_method,
                    objective,
                    predictor,
                    num_parallel_tree,
                    probability_temperature,
                    gpu_only,
                    cpu_threads,
                } = artifact;
                if feature_columns != metadata_feature_columns {
                    bail!(
                        "XGBoost runtime artifact feature-columns mismatch: metadata has {:?}, runtime artifact has {:?}",
                        metadata_feature_columns,
                        feature_columns
                    );
                }
                if training_summary.dataset_rows != metadata_training_summary.dataset_rows
                    || training_summary.train_rows != metadata_training_summary.train_rows
                    || training_summary.val_rows != metadata_training_summary.val_rows
                {
                    bail!(
                        "XGBoost runtime artifact training-summary mismatch: metadata {:?}, runtime artifact {:?}",
                        metadata_training_summary,
                        training_summary
                    );
                }
                self.config.params = configured_params;
                self.config.device_pref = device_pref;
                self.config.gpu_only = gpu_only;
                self.config.cpu_threads = cpu_threads;
                self.feature_columns = feature_columns;
                self.training_summary = Some(training_summary);
                self.config.params.insert(
                    "probability_temperature".into(),
                    ParamValue::Float(probability_temperature),
                );

                let loaded_resolved_params = self.runtime_params();
                let resolved_variant = self.booster_variant();
                let resolved_tree_method = self.effective_tree_method().to_string();
                let resolved_objective =
                    param_string(&self.config.params, "objective", "multi:softprob");
                let resolved_predictor = self.predictor().to_string();
                let resolved_num_parallel_tree = self.tree_num_parallel();
                if booster_variant != resolved_variant
                    || configured_tree_method != self.configured_tree_method().to_string()
                    || effective_tree_method != resolved_tree_method
                    || objective != resolved_objective
                    || predictor != resolved_predictor
                    || num_parallel_tree != resolved_num_parallel_tree
                    || (probability_temperature - self.probability_temperature()).abs()
                        > f64::EPSILON
                {
                    tracing::warn!(
                        model = "xgboost",
                        stored_variant = %booster_variant,
                        loaded_variant = %resolved_variant,
                        stored_tree_method = %configured_tree_method,
                        loaded_tree_method = %self.configured_tree_method().to_string(),
                        stored_effective_tree_method = %effective_tree_method,
                        loaded_effective_tree_method = %resolved_tree_method,
                        stored_objective = %objective,
                        loaded_objective = %resolved_objective,
                        stored_predictor = %predictor,
                        loaded_predictor = %resolved_predictor,
                        stored_num_parallel_tree = num_parallel_tree,
                        loaded_num_parallel_tree = resolved_num_parallel_tree,
                        stored_probability_temperature = probability_temperature,
                        loaded_probability_temperature = self.probability_temperature(),
                        stored_resolved_params = ?resolved_params,
                        loaded_resolved_params = ?loaded_resolved_params,
                        "XGBoost runtime sidecar did not fully match the restored config"
                    );
                }
            }
            if model_path.exists() {
                match xgb::Booster::load(&model_path)
                    .with_context(|| format!("load XGBoost artifact {}", model_path.display()))
                {
                    Ok(model) => {
                        self._model = Some(model);
                        if let Some(fallback) = self.local_fallback.as_ref() {
                            validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                        }
                    }
                    Err(err) => {
                        self._model = None;
                        if let Some(fallback) = self.local_fallback.as_ref() {
                            tracing::warn!(
                                model = "xgboost",
                                path = %path.display(),
                                surrogate_kind = %fallback.surrogate_kind,
                                surrogate_rows = fallback.training_summary.dataset_rows,
                                error = %err,
                                "failed to restore native XGBoost booster; using local surrogate fallback"
                            );
                        } else {
                            return Err(err);
                        }
                    }
                }
            } else {
                self._model = None;
                if let Some(fallback) = self.local_fallback.as_ref() {
                    tracing::warn!(
                        model = "xgboost",
                        path = %path.display(),
                        surrogate_kind = %fallback.surrogate_kind,
                        surrogate_rows = fallback.training_summary.dataset_rows,
                        "XGBoost artifact missing native booster; using local surrogate fallback"
                    );
                }
            }
            self.gpu_only_disabled = false;
            if self._model.is_none() && self.local_fallback.is_none() {
                bail!(
                    "XGBoost artifact {} is missing both native model and local fallback payload",
                    path.display()
                );
            }
        }
        #[cfg(not(feature = "xgboost"))]
        {
            let (_, metadata_path) = tree_artifact_paths(path, XGBOOST_MODEL_FILE_NAME);
            self.local_fallback = Self::read_local_runtime_artifact(path)?;
            let metadata = if metadata_path.exists() {
                let metadata = read_runtime_metadata(&metadata_path)?;
                if metadata.model_name != "xgboost" || metadata.family != ModelFamily::Tree {
                    anyhow::bail!(
                        "XGBoost runtime metadata mismatch: expected tree/xgboost, got {}/{}",
                        metadata.family,
                        metadata.model_name
                    );
                }
                if metadata.feature_columns.is_empty() {
                    anyhow::bail!(
                        "XGBoost runtime metadata must contain at least one feature column"
                    );
                }
                metadata
            } else if let Some(fallback) = self.local_fallback.as_ref() {
                let metadata = tree_runtime_metadata(
                    "xgboost",
                    fallback.feature_columns.clone(),
                    fallback.training_summary.clone(),
                )?;
                tracing::warn!(
                    path = %path.display(),
                    "XGBoost metadata sidecar missing; reconstructing from local fallback artifact"
                );
                metadata
            } else {
                anyhow::bail!(
                    "XGBoost metadata sidecar missing and local fallback artifact missing at {}",
                    path.display()
                );
            };
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            if let Some(fallback) = self.local_fallback.as_ref() {
                validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
            }
            if self.local_fallback.is_none() {
                anyhow::bail!(
                    "XGBoost local fallback artifact missing at {}",
                    path.display()
                );
            }
            self._model = None;
            self.gpu_only_disabled = false;
        }
        Ok(())
    }
}

impl XGBoostExpert {
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        build_tree_runtime_predictions(
            "xgboost",
            &probabilities,
            self._model.is_some(),
            "xgboost_native",
            self.local_fallback.as_ref(),
            "native_xgboost_unavailable",
            "xgboost_unknown",
        )
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod tests {
    use super::{ExpertModel, ParamValue, XGBoostExpert};
    use ndarray::Array2;
    use polars::df;
    use polars::prelude::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_three_class_dataset() -> (DataFrame, Series) {
        let x = df![
            "momentum" => &[0.96, 0.93, 0.89, 0.07, 0.03, 0.11, -0.94, -0.91, -0.88],
            "trend" => &[0.87, 0.91, 0.86, 0.01, -0.02, 0.04, -0.9, -0.86, -0.93],
            "volatility" => &[0.62, 0.58, 0.6, 0.2, 0.18, 0.23, 0.69, 0.66, 0.64],
        ]
        .expect("build training dataframe");
        let y = Series::new("label".into(), &[1_i32, 1, 1, 0, 0, 0, -1, -1, -1]);
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
    fn xgboost_runtime_params_capture_variant_specific_defaults() {
        let mut params = HashMap::new();
        params.insert("variant".into(), ParamValue::String("rf".into()));
        params.insert("device".into(), ParamValue::String("cpu".into()));
        params.insert("tree_method".into(), ParamValue::String("hist".into()));

        let expert = XGBoostExpert::new(11, Some(params));
        let runtime_params = expert.runtime_params();

        assert_eq!(
            runtime_params.get("variant"),
            Some(&ParamValue::String("rf".into()))
        );
        assert_eq!(
            runtime_params.get("tree_method"),
            Some(&ParamValue::String("hist".into()))
        );
        assert_eq!(
            runtime_params.get("predictor"),
            Some(&ParamValue::String("cpu_predictor".into()))
        );
        assert_eq!(
            runtime_params.get("num_parallel_tree"),
            Some(&ParamValue::Int(64))
        );
        match runtime_params.get("subsample") {
            Some(ParamValue::Float(value)) => assert!((*value - 0.8).abs() < 1e-6),
            other => panic!("unexpected subsample runtime param: {other:?}"),
        }
        match runtime_params.get("colsample_bytree") {
            Some(ParamValue::Float(value)) => assert!((*value - 0.8).abs() < 1e-6),
            other => panic!("unexpected colsample_bytree runtime param: {other:?}"),
        }
    }

    #[test]
    fn xgboost_probability_rows_are_normalized() {
        let probabilities = Array2::from_shape_vec((2, 3), vec![2.0_f32, 1.0, 1.0, 0.0, 0.0, 0.0])
            .expect("build probability matrix");

        let normalized = XGBoostExpert::normalize_probabilities(probabilities).expect("normalize");
        for row in normalized.outer_iter() {
            let sum = row.iter().copied().sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-6_f32);
        }
    }

    #[test]
    fn xgboost_probability_temperature_sharpens_probabilities() {
        let mut params = HashMap::new();
        params.insert("probability_temperature".into(), ParamValue::Float(0.5));

        let expert = XGBoostExpert::new(11, Some(params));
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.6_f32, 0.3, 0.1])
            .expect("build probability matrix");
        let calibrated = expert
            .calibrate_probabilities(probabilities)
            .expect("calibrate");

        let row = calibrated.row(0);
        let sum = row.iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6_f32);
        assert!(
            row[0] > 0.6_f32,
            "expected lower temperature to sharpen the dominant class, got {row:?}"
        );
    }

    #[test]
    fn xgboost_validate_runtime_artifact_rejects_invalid_probability_temperature() {
        let artifact = super::XGBoostRuntimeArtifact {
            configured_params: HashMap::new(),
            resolved_params: HashMap::from([(
                "tree_method".to_string(),
                ParamValue::String("hist".to_string()),
            )]),
            feature_columns: vec!["momentum".to_string()],
            training_summary: TrainingSummaryMetadata::new(9, 9, 0),
            device_pref: super::DevicePreference::Cpu,
            booster_variant: "gbtree".to_string(),
            configured_tree_method: "hist".to_string(),
            effective_tree_method: "hist".to_string(),
            objective: "multi:softprob".to_string(),
            predictor: "cpu_predictor".to_string(),
            num_parallel_tree: 1,
            probability_temperature: 0.0,
            gpu_only: false,
            cpu_threads: Some(4),
        };

        let err = XGBoostExpert::validate_runtime_artifact(
            &artifact,
            &["momentum".to_string()],
            &TrainingSummaryMetadata::new(9, 9, 0),
        )
        .expect_err("non-positive probability_temperature should fail");
        assert!(err.to_string().contains("probability_temperature"));
    }

    #[test]
    fn xgboost_save_rejects_missing_training_summary() {
        let mut expert = XGBoostExpert::new(11, None);
        expert.feature_columns = vec!["momentum".to_string()];
        expert.local_fallback = Some(super::TreeLocalFallbackArtifact {
            feature_columns: vec!["momentum".to_string()],
            class_centroids: vec![vec![0.1], vec![0.2], vec![0.3]],
            class_variances: vec![vec![1.0], vec![1.0], vec![1.0]],
            class_support: vec![1, 1, 1],
            training_summary: TrainingSummaryMetadata::new(9, 9, 0),
            distance_location: 0.0,
            distance_scale: 1.0,
            surrogate_kind: "gaussian_centroid".to_string(),
        });
        let artifact_dir = unique_temp_dir("xgboost-missing-summary");

        let err = expert
            .save(&artifact_dir)
            .expect_err("missing training summary should fail");
        assert!(err.to_string().contains("training summary"));

        let _ = std::fs::remove_dir_all(&artifact_dir);
    }

    #[test]
    fn xgboost_trains_three_class_probabilities_and_persists_artifacts() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("xgboost-artifact");

        let mut expert = XGBoostExpert::new(11, None);
        expert.fit(&x, &y).expect("fit should succeed");

        let probabilities = expert.predict_proba(&x).expect("predict should succeed");
        assert_eq!(probabilities.dim(), (x.height(), 3));
        assert_rows_are_non_uniform(&probabilities);

        expert.save(&artifact_dir).expect("save should succeed");
        assert!(
            artifact_dir.join("model.bin").exists(),
            "expected XGBoost model artifact at {}",
            artifact_dir.join("model.bin").display()
        );
        assert!(
            artifact_dir.join("metadata.json").exists(),
            "expected metadata sidecar at {}",
            artifact_dir.join("metadata.json").display()
        );

        let mut loaded = XGBoostExpert::new(11, None);
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
    fn xgboost_load_uses_runtime_artifacts_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("xgboost-missing-metadata-sidecar");

        let mut expert = XGBoostExpert::new(11, None);
        expert.fit(&x, &y).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("metadata.json"))
            .expect("remove metadata sidecar to trigger reconstruction");

        let mut loaded = XGBoostExpert::new(11, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct metadata from persisted runtime artifacts");
        let probabilities = loaded.predict_proba(&x).expect("prediction should succeed");
        assert_eq!(probabilities.dim(), (x.height(), 3));
    }
}
