// Tree-models XGBoost expert. Native-only imports and helpers are gated by the
// exact `xgboost` feature so standalone feature builds remain warning-clean.

use super::common::build_tree_runtime_predictions;
#[cfg(feature = "xgboost")]
use super::common::{
    XGBOOST_MODEL_FILE_NAME, calibrate_three_class_probabilities, default_training_summary,
    ensure_feature_columns_match, feature_frame_to_tree_f32_row_major,
    normalize_three_class_probabilities, read_runtime_metadata, read_tree_json_artifact,
    remap_labels_to_tree_targets, tree_artifact_paths, tree_runtime_metadata,
    write_runtime_metadata, write_tree_json_artifact,
};
use super::config::*;
use crate::base::ExpertModel;
#[cfg(feature = "xgboost")]
use crate::base::{compute_sample_weights, feature_columns_from_frame};
#[cfg(feature = "xgboost")]
use crate::common::CudaDevicePolicy;
#[cfg(feature = "xgboost")]
use crate::runtime::artifacts::RuntimeArtifactMetadata;
use crate::runtime::artifacts::TrainingSummaryMetadata;
#[cfg(feature = "xgboost")]
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;
use anyhow::{Context, Result, bail};
use ndarray::Array2;
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
#[cfg(feature = "xgboost")]
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

#[cfg(feature = "xgboost")]
const XGBOOST_RUNTIME_FILE_NAME: &str = "xgboost_runtime.json";

/// One-time runtime probe: does the *linked* libxgboost actually support the
/// CUDA device? A present GPU (`gpu_count() > 0`) is NOT enough —
/// the `xgb` crate's bundled libxgboost is built CPU-only by default, so a
/// CUDA-device booster fails at `update()` ("update XGBoost booster at
/// iteration 0"). That single failure used to sink SIX models at once (xgboost,
/// xgboost_dart, meta_stack, meta_blender, probability_calibrator,
/// conformal_gate) because they all route through this expert. We detect the
/// capability ONCE by training a tiny booster with the production XGBoost 2+
/// spelling (`tree_method=hist`, `device=cuda:N`). Explicit or auto-resolved GPU
/// training fails loudly when that probe fails; CPU remains valid only when no
/// card is visible or the operator explicitly selects it.
#[cfg(feature = "xgboost")]
fn xgboost_cuda_runtime_available(cuda_ordinal: usize) -> bool {
    use std::sync::{Mutex, OnceLock};
    static AVAILABLE: OnceLock<Mutex<HashMap<usize, bool>>> = OnceLock::new();
    let available = AVAILABLE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = available
        .lock()
        .expect("XGBoost CUDA probe cache lock poisoned")
        .get(&cuda_ordinal)
        .copied()
    {
        return cached;
    }

    let result = {
        if nvidia_gpu_count() == 0 {
            return false;
        }
        let probe = || -> Result<()> {
            // 3 rows × 2 features, one row per class so MultiSoftprob(3) is valid.
            let flat = [0.0f32, 0.0, 1.0, 1.0, 0.5, 0.5];
            let mut dtrain =
                xgb::DMatrix::from_dense(&flat, 3).context("xgboost gpu probe: DMatrix")?;
            dtrain
                .set_labels(&[0.0f32, 1.0, 2.0])
                .context("xgboost gpu probe: labels")?;
            let tree_params = TreeBoosterParametersBuilder::default()
                .tree_method(TreeMethod::Hist)
                .predictor(Predictor::Gpu)
                .build()
                .context("xgboost gpu probe: tree params")?;
            let learning_params = LearningTaskParametersBuilder::default()
                .objective(Objective::MultiSoftprob(3))
                .build()
                .context("xgboost gpu probe: learning params")?;
            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .verbose(false)
                .build()
                .context("xgboost gpu probe: booster params")?;
            let mut booster = xgb::Booster::new_with_cached_dmats(&booster_params, &[&dtrain])
                .context("xgboost gpu probe: create booster")?;
            let device = format!("cuda:{cuda_ordinal}");
            booster
                .set_param("device", &device)
                .with_context(|| format!("xgboost gpu probe: set device={device}"))?;
            booster
                .update(&dtrain, 0)
                .context("xgboost gpu probe: update booster")?;
            Ok(())
        };
        match probe() {
            Ok(()) => {
                tracing::info!(
                    target: "neoethos_models::tree_models::xgboost",
                    cuda_ordinal,
                    "XGBoost CUDA runtime probe succeeded"
                );
                true
            }
            Err(error) => {
                tracing::warn!(
                    target: "neoethos_models::tree_models::xgboost",
                    cuda_ordinal,
                    %error,
                    "XGBoost CUDA runtime probe failed"
                );
                false
            }
        }
    };
    available
        .lock()
        .expect("XGBoost CUDA probe cache lock poisoned")
        .insert(cuda_ordinal, result);
    result
}

#[cfg(feature = "xgboost")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct XGBoostRuntimeArtifact {
    configured_params: HashMap<String, ParamValue>,
    resolved_params: HashMap<String, ParamValue>,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
    requested_device_policy: String,
    device_pref: DevicePreference,
    booster_variant: String,
    configured_tree_method: String,
    effective_tree_method: String,
    effective_device: String,
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
    pub(crate) feature_columns: Vec<String>,
    training_summary: Option<TrainingSummaryMetadata>,
    #[cfg(feature = "xgboost")]
    _model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    _model: Option<()>,
}

impl XGBoostExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let params = params.unwrap_or_else(Self::default_params);
        let requested_device_policy = tree_device_policy_from_params(&params, "xgboost");
        let device_pref =
            device_preference_from_params(&params, tree_device_preference_for("xgboost"));
        let gpu_only = gpu_only_from_params(&params, gpu_only_mode_for("xgboost"));
        let cpu_threads = cpu_threads_from_params(&params, cpu_threads_hint_for("xgboost"));
        Self {
            idx,
            config: TreeModelConfig {
                idx,
                params,
                requested_device_policy,
                device_pref,
                gpu_only,
                cpu_threads: Some(cpu_threads),
            },
            gpu_only_disabled: false,
            feature_columns: Vec::new(),
            training_summary: None,
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
    fn effective_tree_method(&self) -> Result<TreeMethod> {
        match self.resolved_cuda_device()? {
            crate::common::ResolvedCudaDevicePolicy::Cuda { .. } => {
                match self.configured_tree_method() {
                    TreeMethod::GpuHist => Ok(TreeMethod::Hist),
                    TreeMethod::GpuExact => Ok(TreeMethod::Exact),
                    other => Ok(other),
                }
            }
            crate::common::ResolvedCudaDevicePolicy::Cpu => match self.configured_tree_method() {
                TreeMethod::GpuHist => Ok(TreeMethod::Hist),
                TreeMethod::GpuExact => Ok(TreeMethod::Exact),
                other => Ok(other),
            },
        }
    }

    #[cfg(feature = "xgboost")]
    fn resolved_cuda_device(&self) -> Result<crate::common::ResolvedCudaDevicePolicy> {
        let resolved = self.config.resolved_cuda_device()?;
        if let crate::common::ResolvedCudaDevicePolicy::Cuda { ordinal } = resolved
            && !xgboost_cuda_runtime_available(ordinal)
        {
            bail!(
                "XGBoost resolved CUDA from policy `{}`, but the linked CUDA runtime probe failed; refusing CPU fallback",
                self.config.requested_device_policy
            );
        }
        Ok(resolved)
    }

    /// `cuda` or `cpu` for XGBoost's `device` parameter.
    ///
    /// XGBoost 2.0 replaced `tree_method = gpu_hist` with `tree_method = hist`
    /// plus `device = cuda`, and 3.0 warns on every booster built the old way:
    /// "The tree method `gpu_hist` is deprecated since 2.0.0." The old spelling
    /// still resolves today, but it is a name the library has already announced
    /// it will stop honouring, and the warning is one line in a training log
    /// nobody reads to the end.
    #[cfg(feature = "xgboost")]
    fn device_param(&self) -> Result<String> {
        match self.resolved_cuda_device()? {
            crate::common::ResolvedCudaDevicePolicy::Cpu => Ok("cpu".to_string()),
            crate::common::ResolvedCudaDevicePolicy::Cuda {
                ordinal: cuda_ordinal,
            } => Ok(format!("cuda:{cuda_ordinal}")),
        }
    }

    #[cfg(feature = "xgboost")]
    fn predictor(&self) -> Result<Predictor> {
        match self.resolved_cuda_device()? {
            crate::common::ResolvedCudaDevicePolicy::Cuda { .. } => Ok(Predictor::Gpu),
            crate::common::ResolvedCudaDevicePolicy::Cpu => Ok(Predictor::Cpu),
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

    #[cfg(feature = "xgboost")]
    fn probability_temperature(&self) -> f64 {
        let configured = param_float(&self.config.params, "probability_temperature", 1.0);
        if configured.is_finite() && configured > 0.0 {
            configured
        } else {
            1.0
        }
    }

    #[cfg(feature = "xgboost")]
    fn runtime_params(&self) -> Result<HashMap<String, ParamValue>> {
        let mut params = self.config.params.clone();
        params.insert("variant".into(), ParamValue::String(self.booster_variant()));
        params.insert(
            "tree_method".into(),
            ParamValue::String(self.effective_tree_method()?.to_string()),
        );
        params.insert("device".into(), ParamValue::String(self.device_param()?));
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
            ParamValue::String(self.predictor()?.to_string()),
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
        Ok(params)
    }

    #[cfg(feature = "xgboost")]
    fn runtime_artifact(&self) -> Result<XGBoostRuntimeArtifact> {
        Ok(XGBoostRuntimeArtifact {
            configured_params: self.config.params.clone(),
            resolved_params: self.runtime_params()?,
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
            requested_device_policy: self.config.requested_device_policy.clone(),
            device_pref: self.config.device_pref,
            booster_variant: self.booster_variant(),
            configured_tree_method: self.configured_tree_method().to_string(),
            effective_tree_method: self.effective_tree_method()?.to_string(),
            effective_device: self.device_param()?,
            objective: param_string(&self.config.params, "objective", "multi:softprob"),
            predictor: self.predictor()?.to_string(),
            num_parallel_tree: self.tree_num_parallel(),
            probability_temperature: self.probability_temperature(),
            gpu_only: self.config.gpu_only,
            cpu_threads: self.config.cpu_threads,
        })
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
            .set_attribute("tree_method", &self.effective_tree_method()?.to_string())
            .context("set XGBoost tree_method attribute")?;
        model
            .set_attribute("device", &self.device_param()?)
            .context("set XGBoost device attribute")?;
        model
            .set_attribute(
                "objective",
                &param_string(&self.config.params, "objective", "multi:softprob"),
            )
            .context("set XGBoost objective attribute")?;
        model
            .set_attribute("predictor", &self.predictor()?.to_string())
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
        write_tree_json_artifact(
            &path.join(XGBOOST_RUNTIME_FILE_NAME),
            &self.runtime_artifact()?,
            "XGBoost runtime artifact",
        )
    }

    #[cfg(feature = "xgboost")]
    fn apply_runtime_device(&self, model: &mut xgb::Booster) -> Result<()> {
        let device = self.device_param()?;
        model
            .set_param("device", &device)
            .with_context(|| format!("apply XGBoost runtime device {device}"))
    }

    #[cfg(feature = "xgboost")]
    fn read_runtime_artifact(path: &Path) -> Result<Option<XGBoostRuntimeArtifact>> {
        let artifact_path = path.join(XGBOOST_RUNTIME_FILE_NAME);
        if !artifact_path.exists() {
            return Ok(None);
        }
        let artifact = read_tree_json_artifact(&artifact_path, "XGBoost runtime artifact")?;
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

    #[cfg(feature = "xgboost")]
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
        if self._model.is_none() {
            bail!("XGBoost runtime state is missing its native booster");
        }
        Ok(())
    }

    #[cfg(feature = "xgboost")]
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
            ("effective_device", artifact.effective_device.as_str()),
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
        let requested_device = parse_tree_cuda_device_policy(&artifact.requested_device_policy)
            .with_context(|| {
                format!(
                    "validate XGBoost requested device policy `{}`",
                    artifact.requested_device_policy
                )
            })?;
        let effective_cuda_ordinal = if artifact.effective_device == "cpu" {
            None
        } else if let Some(raw_ordinal) = artifact.effective_device.strip_prefix("cuda:") {
            Some(raw_ordinal.parse::<usize>().with_context(|| {
                format!(
                    "XGBoost runtime artifact has invalid effective CUDA ordinal `{}`",
                    artifact.effective_device
                )
            })?)
        } else {
            bail!(
                "XGBoost runtime artifact effective_device must be `cpu` or `cuda:<ordinal>`, got {}",
                artifact.effective_device
            );
        };
        match (requested_device, effective_cuda_ordinal) {
            (CudaDevicePolicy::Cpu, Some(_)) => {
                bail!("XGBoost runtime artifact requested CPU but recorded CUDA execution")
            }
            (CudaDevicePolicy::Gpu { .. }, None) => {
                bail!("XGBoost runtime artifact requested explicit CUDA but recorded CPU execution")
            }
            (CudaDevicePolicy::Gpu { ordinal: requested }, Some(recorded))
                if requested != recorded =>
            {
                bail!(
                    "XGBoost runtime artifact CUDA ordinal mismatch: requested {requested}, recorded {recorded}"
                )
            }
            (CudaDevicePolicy::Auto, Some(recorded)) if recorded != 0 => {
                bail!("XGBoost Auto runtime artifact must record CUDA ordinal 0, got {recorded}")
            }
            _ => {}
        }
        if artifact.resolved_params.get("device")
            != Some(&ParamValue::String(artifact.effective_device.clone()))
        {
            bail!(
                "XGBoost runtime artifact resolved device parameter does not match effective_device"
            );
        }
        let expected_predictor = if effective_cuda_ordinal.is_some() {
            "gpu_predictor"
        } else {
            "cpu_predictor"
        };
        if artifact.predictor != expected_predictor
            || artifact.resolved_params.get("predictor")
                != Some(&ParamValue::String(expected_predictor.to_string()))
        {
            bail!(
                "XGBoost runtime artifact predictor is inconsistent with effective_device: expected {expected_predictor}, got {}",
                artifact.predictor
            );
        }
        Ok(())
    }

    #[cfg(feature = "xgboost")]
    fn calibrate_probabilities(&self, probabilities: Array2<f64>) -> Result<Array2<f64>> {
        calibrate_three_class_probabilities(
            probabilities,
            self.probability_temperature(),
            "XGBoost",
        )
    }

    #[cfg(feature = "xgboost")]
    fn normalize_probabilities(probabilities: Array2<f64>) -> Result<Array2<f64>> {
        normalize_three_class_probabilities(probabilities, "XGBoost")
    }

    /// M6: shared body for `fit` and `fit_with_validation`. When external
    /// val data is supplied, builds an evaluation `DMatrix` and breaks the
    /// boosting loop early once `mlogloss` has not improved on the val
    /// frame for `early_stopping_rounds` iterations (default 50). Without
    /// external val, runs the full `n_estimators` rounds as before.
    fn fit_internal(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
        lease_width: usize,
    ) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if x.n_samples() == 0 || y.is_empty() {
                anyhow::bail!("XGBoost requires non-empty training features and labels");
            }
            if x.n_samples() != y.len() {
                anyhow::bail!(
                    "XGBoost requires matching feature and label rows: {} features vs {} labels",
                    x.n_samples(),
                    y.len()
                );
            }
            self.config.cpu_threads = Some(
                self.config
                    .cpu_threads
                    .unwrap_or(lease_width)
                    .min(lease_width)
                    .max(1),
            );

            let resolved_cuda_device = self.resolved_cuda_device()?;
            if self.config.gpu_only
                && matches!(
                    resolved_cuda_device,
                    crate::common::ResolvedCudaDevicePolicy::Cpu
                )
            {
                self.gpu_only_disabled = true;
                self._model = None;
                anyhow::bail!(
                    "XGBoost gpu-only mode resolved CPU from policy `{}` because no NVIDIA device is visible",
                    self.config.requested_device_policy
                );
            }

            let effective_tree_method = self.effective_tree_method()?;

            let (flat_x, n_rows, _n_cols) = feature_frame_to_tree_f32_row_major(x)?;
            let labels = remap_labels_to_tree_targets(y)?;
            let sample_weights = compute_sample_weights(y)?
                .into_iter()
                .map(|weight| {
                    let narrowed = weight as f32;
                    if !narrowed.is_finite() || (weight != 0.0 && narrowed == 0.0) {
                        bail!("XGBoost cannot represent sample weight {weight} as f32");
                    }
                    Ok(narrowed)
                })
                .collect::<Result<Vec<_>>>()?;

            let mut dtrain = xgb::DMatrix::from_dense(&flat_x, n_rows)
                .context("create XGBoost training matrix from typed feature frame")?;
            dtrain
                .set_labels(&labels)
                .context("set XGBoost training labels")?;
            dtrain
                .set_weights(&sample_weights)
                .context("set XGBoost sample weights")?;

            let dval = match (val_x, val_y) {
                (Some(vx), Some(vy)) => {
                    if vx.n_features() != x.n_features() {
                        anyhow::bail!(
                            "XGBoost validation column count mismatch: train {}, val {}",
                            x.n_features(),
                            vx.n_features()
                        );
                    }
                    if vx.n_samples() != vy.len() {
                        anyhow::bail!(
                            "XGBoost validation row/label mismatch: {} rows vs {} labels",
                            vx.n_samples(),
                            vy.len()
                        );
                    }
                    let (vflat, v_rows, _vcols) = feature_frame_to_tree_f32_row_major(vx)?;
                    let vlabels = remap_labels_to_tree_targets(vy)?;
                    let mut dval = xgb::DMatrix::from_dense(&vflat, v_rows)
                        .context("create XGBoost validation matrix from typed feature frame")?;
                    dval.set_labels(&vlabels)
                        .context("set XGBoost validation labels")?;
                    Some(dval)
                }
                (None, None) => None,
                _ => bail!(
                    "XGBoostExpert::fit_with_validation requires both val_x and val_y or neither"
                ),
            };

            // v0.5 ML-integration Stage 1(a): plumb the L1/L2/min-child/gamma
            // regularizers into the booster. Previously these were never set,
            // so any `reg_lambda` / `reg_alpha` / `gamma` / `min_child_weight`
            // in `config.params` was silently dropped. Fallbacks are XGBoost's
            // own neutral defaults (gamma 0, min_child_weight 1, lambda 1,
            // alpha 0) so that with the regularized seed map OFF (keys absent)
            // the booster behaves exactly as before.
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
                    .min_child_weight(
                        param_float(&self.config.params, "min_child_weight", 1.0) as f32
                    )
                    .gamma(param_float(&self.config.params, "gamma", 0.0) as f32)
                    .lambda(param_float(&self.config.params, "reg_lambda", 1.0) as f32)
                    .alpha(param_float(&self.config.params, "reg_alpha", 0.0) as f32)
                    .num_parallel_tree(self.tree_num_parallel())
                    .tree_method(effective_tree_method)
                    .predictor(self.predictor()?)
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
            let early_stopping_rounds =
                param_int(&self.config.params, "early_stopping_rounds", 50).max(1) as i32;

            let mut model = xgb::Booster::new_with_cached_dmats(&booster_params, &[&dtrain])
                .context("create XGBoost booster")?;
            // Stated explicitly rather than inferred from the tree method, so a
            // run says which device it trained on instead of leaving it to a
            // deprecated alias. Reported at info because "which device" is the
            // question this file exists to answer.
            let device = self.device_param()?;
            self.apply_runtime_device(&mut model)?;
            tracing::info!(
                target: "neoethos_models::tree_models::xgboost",
                device = %device,
                "XGBoost booster device"
            );
            self.apply_variant_params(&mut model)?;
            self.feature_columns = feature_columns_from_frame(x);
            self.set_runtime_attributes(&mut model)?;

            let mut best_loss = f32::INFINITY;
            let mut best_iter: i32 = 0;
            let mut rounds_without_improvement: i32 = 0;
            for iteration in 0..boost_rounds as i32 {
                model
                    .update(&dtrain, iteration)
                    .with_context(|| format!("update XGBoost booster at iteration {iteration}"))?;
                if let Some(dval) = dval.as_ref() {
                    let metrics = model
                        .evaluate(dval)
                        .context("evaluate XGBoost booster against val matrix")?;
                    let val_loss = metrics
                        .get("mlogloss")
                        .or_else(|| metrics.get("merror"))
                        .copied()
                        .unwrap_or(f32::INFINITY);
                    if val_loss < best_loss {
                        best_loss = val_loss;
                        best_iter = iteration;
                        rounds_without_improvement = 0;
                    } else {
                        rounds_without_improvement += 1;
                        if rounds_without_improvement >= early_stopping_rounds {
                            tracing::info!(
                                model = "xgboost",
                                iteration,
                                best_iter,
                                best_loss,
                                "XGBoost early-stopping triggered after {early_stopping_rounds} rounds without val improvement"
                            );
                            break;
                        }
                    }
                }
            }

            self.training_summary = Some(default_training_summary(x));
            self.gpu_only_disabled = false;
            self._model = Some(model);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            let _ = (x, y, val_x, val_y, lease_width);
            bail!("XGBoost native backend unavailable: compile with the `xgboost` feature")
        }
    }
}

impl ExpertModel for XGBoostExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        lease.scope(|| self.fit_internal(x, y, None, None, lease.width().get()))
    }

    fn fit_with_validation(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
        lease: &CpuLease,
    ) -> Result<()> {
        lease.scope(|| self.fit_internal(x, y, val_x, val_y, lease.width().get()))
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        lease.scope(|| {
            if self.gpu_only_disabled {
                anyhow::bail!("XGBoost disabled: gpu-only mode requested without an available GPU");
            }
            #[cfg(feature = "xgboost")]
            {
                if x.n_samples() == 0 {
                    return Ok(Array2::zeros((0, 3)));
                }

                ensure_feature_columns_match(&self.feature_columns, x)?;
                let model = self._model.as_ref().context("XGBoost not trained")?;
                let (flat_x, n_rows, _) = feature_frame_to_tree_f32_row_major(x)?;
                let dtest = xgb::DMatrix::from_dense(&flat_x, n_rows)
                    .context("create XGBoost prediction matrix from typed feature frame")?;
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
                    [_cols] => probabilities.len().checked_div(n_rows).unwrap_or(0),
                    _ => probabilities.len().checked_div(n_rows).unwrap_or(0),
                };
                let probabilities = reshape_three_class_probabilities(probabilities, n_rows, cols)?;
                let probabilities = self.calibrate_probabilities(probabilities)?;
                Self::normalize_probabilities(probabilities)
            }
            #[cfg(not(feature = "xgboost"))]
            {
                let _ = x;
                bail!("XGBoost native backend unavailable: compile with the `xgboost` feature")
            }
        })
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
            let model = self._model.as_ref().context("XGBoost not trained")?;
            Self::validate_runtime_artifact(
                &self.runtime_artifact()?,
                &self.feature_columns,
                &self.stored_training_summary(),
            )?;
            model
                .save(&model_path)
                .with_context(|| format!("save XGBoost artifact {}", model_path.display()))?;
            self.persist_runtime_artifact(path)?;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            let _ = path;
            bail!("XGBoost native backend unavailable: compile with the `xgboost` feature")
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let (model_path, metadata_path) = tree_artifact_paths(path, XGBOOST_MODEL_FILE_NAME);
            let runtime_artifact = Self::read_runtime_artifact(path)?;
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
                let (feature_columns, training_summary) =
                    if let Some(artifact) = runtime_artifact.as_ref() {
                        (
                            artifact.feature_columns.clone(),
                            artifact.training_summary.clone(),
                        )
                    } else {
                        bail!(
                            "XGBoost metadata sidecar and runtime artifact are missing at {}",
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
                    requested_device_policy,
                    device_pref,
                    booster_variant,
                    configured_tree_method,
                    effective_tree_method,
                    effective_device,
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
                self.config.requested_device_policy = requested_device_policy;
                self.config.device_pref = device_pref;
                self.config.gpu_only = gpu_only;
                self.config.cpu_threads = cpu_threads;
                self.feature_columns = feature_columns;
                self.training_summary = Some(training_summary);
                self.config.params.insert(
                    "probability_temperature".into(),
                    ParamValue::Float(probability_temperature),
                );

                let loaded_resolved_params = self.runtime_params()?;
                let resolved_variant = self.booster_variant();
                let resolved_tree_method = self.effective_tree_method()?.to_string();
                let resolved_device = self.device_param()?;
                let resolved_objective =
                    param_string(&self.config.params, "objective", "multi:softprob");
                let resolved_predictor = self.predictor()?.to_string();
                let resolved_num_parallel_tree = self.tree_num_parallel();
                if gpu_only && resolved_device == "cpu" {
                    bail!(
                        "XGBoost gpu-only artifact cannot relocate to CPU because no NVIDIA device is visible"
                    );
                }
                let auto_cpu_relocation = matches!(
                    parse_tree_cuda_device_policy(&self.config.requested_device_policy)?,
                    CudaDevicePolicy::Auto
                ) && effective_device.starts_with("cuda:")
                    && resolved_device == "cpu"
                    && !gpu_only;
                if booster_variant != resolved_variant
                    || configured_tree_method != self.configured_tree_method().to_string()
                    || effective_tree_method != resolved_tree_method
                    || (!auto_cpu_relocation && effective_device != resolved_device)
                    || objective != resolved_objective
                    || (!auto_cpu_relocation && predictor != resolved_predictor)
                    || num_parallel_tree != resolved_num_parallel_tree
                    || (probability_temperature - self.probability_temperature()).abs()
                        > f64::EPSILON
                {
                    bail!(
                        "XGBoost runtime sidecar drift after restore: stored variant={booster_variant} resolved={resolved_variant}; stored tree_method={effective_tree_method} resolved={resolved_tree_method}; stored device={effective_device} resolved={resolved_device}; stored objective={objective} resolved={resolved_objective}; stored predictor={predictor} resolved={resolved_predictor}; stored num_parallel_tree={num_parallel_tree} resolved={resolved_num_parallel_tree}; stored probability_temperature={probability_temperature} resolved={}; stored params={resolved_params:?} resolved params={loaded_resolved_params:?}",
                        self.probability_temperature()
                    );
                }
            }
            if !model_path.exists() {
                bail!(
                    "XGBoost native model artifact is missing at {}",
                    model_path.display()
                );
            }
            let mut model = xgb::Booster::load(&model_path)
                .with_context(|| format!("load XGBoost artifact {}", model_path.display()))?;
            self.apply_runtime_device(&mut model)?;
            self._model = Some(model);
            self.gpu_only_disabled = false;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            let _ = path;
            bail!("XGBoost native backend unavailable: compile with the `xgboost` feature")
        }
    }
}

impl XGBoostExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`]
    /// adapter so the registry / aggregator can detect column-layout
    /// drift after a retraining session.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        build_tree_runtime_predictions("xgboost", &probabilities, "xgboost_native")
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod tests {
    use super::{ExpertModel, ParamValue, TrainingSummaryMetadata, XGBoostExpert};
    use ndarray::Array2;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
    use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_three_class_dataset() -> (FeatureFrame, Vec<i32>) {
        let rows = 9;
        let columns = [
            (
                "momentum",
                vec![0.96, 0.93, 0.89, 0.07, 0.03, 0.11, -0.94, -0.91, -0.88],
            ),
            (
                "trend",
                vec![0.87, 0.91, 0.86, 0.01, -0.02, 0.04, -0.9, -0.86, -0.93],
            ),
            (
                "volatility",
                vec![0.62, 0.58, 0.6, 0.2, 0.18, 0.23, 0.69, 0.66, 0.64],
            ),
        ]
        .into_iter()
        .map(|(name, values)| {
            FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; rows])
                .expect("valid typed feature column")
        })
        .collect::<Vec<_>>();
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
        .expect("build typed training frame");
        (frame, vec![1_i32, 1, 1, 0, 0, 0, -1, -1, -1])
    }

    fn single_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one is a valid worker limit");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("single-worker model test lease")
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

    fn assert_rows_are_non_uniform(probabilities: &Array2<f64>) {
        assert_eq!(probabilities.ncols(), 3);
        assert!(
            probabilities.outer_iter().any(|row| {
                row.iter()
                    .any(|value| (value - (1.0_f64 / 3.0_f64)).abs() > 0.05_f64)
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
        let runtime_params = expert.runtime_params().expect("resolve CPU runtime params");

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
        let probabilities = Array2::from_shape_vec((2, 3), vec![0.8_f64, 0.6, 0.6, 0.1, 0.2, 0.1])
            .expect("build probability matrix");

        let normalized = XGBoostExpert::normalize_probabilities(probabilities).expect("normalize");
        for (row_index, row) in normalized.outer_iter().enumerate() {
            let sum = row.iter().copied().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-6_f64);
            let expected = if row_index == 0 {
                [0.4_f64, 0.3, 0.3]
            } else {
                [0.25_f64, 0.5, 0.25]
            };
            for (actual, expected) in row.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn xgboost_probability_temperature_sharpens_probabilities() {
        let mut params = HashMap::new();
        params.insert("probability_temperature".into(), ParamValue::Float(0.5));

        let expert = XGBoostExpert::new(11, Some(params));
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.6_f64, 0.3, 0.1])
            .expect("build probability matrix");
        let calibrated = expert
            .calibrate_probabilities(probabilities)
            .expect("calibrate");

        let row = calibrated.row(0);
        let sum = row.iter().copied().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-6_f64);
        assert!(
            row[0] > 0.6_f64,
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
            requested_device_policy: "cpu".to_string(),
            device_pref: super::DevicePreference::Cpu,
            booster_variant: "gbtree".to_string(),
            configured_tree_method: "hist".to_string(),
            effective_tree_method: "hist".to_string(),
            effective_device: "cpu".to_string(),
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
        let lease = single_worker_lease();

        let mut expert = XGBoostExpert::new(11, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");

        let probabilities = expert
            .predict_proba(&x, &lease)
            .expect("predict should succeed");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
        assert_rows_are_non_uniform(&probabilities);

        expert.save(&artifact_dir).expect("save should succeed");
        assert!(
            artifact_dir.join("model.ubj").exists(),
            "expected XGBoost model artifact at {}",
            artifact_dir.join("model.ubj").display()
        );
        assert!(
            artifact_dir.join("metadata.json").exists(),
            "expected metadata sidecar at {}",
            artifact_dir.join("metadata.json").display()
        );

        let mut loaded = XGBoostExpert::new(11, None);
        loaded.load(&artifact_dir).expect("load should succeed");
        let reloaded = loaded
            .predict_proba(&x, &lease)
            .expect("reloaded predict should succeed");

        for (lhs, rhs) in probabilities.iter().zip(reloaded.iter()) {
            assert!(
                (lhs - rhs).abs() < 1e-3_f64,
                "expected persisted probabilities to round-trip, left={lhs}, right={rhs}"
            );
        }
    }

    #[test]
    fn xgboost_load_uses_runtime_artifacts_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("xgboost-missing-metadata-sidecar");
        let lease = single_worker_lease();

        let mut expert = XGBoostExpert::new(11, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("metadata.json"))
            .expect("remove metadata sidecar to trigger reconstruction");

        let mut loaded = XGBoostExpert::new(11, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct metadata from persisted runtime artifacts");
        let probabilities = loaded
            .predict_proba(&x, &lease)
            .expect("prediction should succeed");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
    }
}
