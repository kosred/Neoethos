use anyhow::{Context, Result, bail};
#[cfg(feature = "lightgbm")]
use lightgbm3;
use ndarray::Array2;
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
#[cfg(feature = "lightgbm")]
use serde::{Deserialize, Serialize};
use std::path::Path;
#[cfg(feature = "lightgbm")]
use std::path::PathBuf;

use crate::base::ExpertModel;
#[cfg(feature = "lightgbm")]
use crate::base::feature_columns_from_frame;
#[cfg(feature = "lightgbm")]
use crate::common::{CudaDevicePolicy, ResolvedCudaDevicePolicy, resolve_cuda_device_policy};
#[cfg(feature = "lightgbm")]
use crate::runtime::artifacts::RuntimeArtifactMetadata;
use crate::runtime::artifacts::TrainingSummaryMetadata;
#[cfg(feature = "lightgbm")]
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;

use super::common::build_tree_runtime_predictions;
#[cfg(feature = "lightgbm")]
use super::common::{
    LIGHTGBM_MODEL_FILE_NAME, calibrate_three_class_probabilities, default_training_summary,
    ensure_feature_columns_match, feature_frame_to_tree_f32_row_major,
    normalize_three_class_probabilities, read_runtime_metadata, read_tree_json_artifact,
    remap_labels_to_tree_targets, tree_artifact_paths, tree_runtime_metadata,
    write_runtime_metadata, write_tree_json_artifact,
};
#[cfg(feature = "lightgbm")]
use super::config::{
    DevicePreference, ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_only_from_params, gpu_only_mode_for, lightgbm_gpu_allowed,
    nvidia_gpu_count, param_bool, param_float, param_int, param_string,
    parse_tree_cuda_device_policy, tree_device_policy_from_params, tree_device_preference_for,
};
#[cfg(not(feature = "lightgbm"))]
use super::config::{
    ParamValue, TreeModelConfig, cpu_threads_from_params, cpu_threads_hint_for,
    device_preference_from_params, gpu_only_from_params, gpu_only_mode_for,
    tree_device_policy_from_params, tree_device_preference_for,
};
use std::collections::HashMap;

#[cfg(feature = "lightgbm")]
const LIGHTGBM_RUNTIME_FILE_NAME: &str = "runtime.json";

#[cfg(feature = "lightgbm")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LightGBMRuntimeArtifact {
    configured_params: HashMap<String, ParamValue>,
    resolved_params: HashMap<String, ParamValue>,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
    device_pref: DevicePreference,
    requested_device_policy: String,
    effective_device_type: String,
    cuda_ordinal: Option<usize>,
    boosting_type: String,
    probability_temperature: f64,
    gpu_only: bool,
    cpu_threads: usize,
}

pub struct LightGBMExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    feature_columns: Vec<String>,
    training_summary: Option<TrainingSummaryMetadata>,
    #[cfg(feature = "lightgbm")]
    model: Option<lightgbm3::Booster>,
    #[cfg(not(feature = "lightgbm"))]
    model: Option<()>,
}

impl LightGBMExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let params = params.unwrap_or_else(Self::default_params);
        let requested_device_policy = tree_device_policy_from_params(&params, "lightgbm");
        let device_pref =
            device_preference_from_params(&params, tree_device_preference_for("lightgbm"));
        let gpu_only = gpu_only_from_params(&params, gpu_only_mode_for("lightgbm"));
        let cpu_threads = cpu_threads_from_params(&params, cpu_threads_hint_for("lightgbm"));
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
            feature_columns: Vec::new(),
            training_summary: None,
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

    #[cfg(feature = "lightgbm")]
    fn stored_training_summary(&self) -> TrainingSummaryMetadata {
        self.training_summary
            .clone()
            .unwrap_or_else(|| TrainingSummaryMetadata::new(0, 0, 0))
    }

    #[cfg(feature = "lightgbm")]
    fn boosting_type(&self) -> String {
        param_string(&self.config.params, "boosting_type", "gbdt").to_lowercase()
    }

    #[cfg(feature = "lightgbm")]
    fn probability_temperature(&self) -> f64 {
        let configured = param_float(&self.config.params, "probability_temperature", 1.0);
        if configured.is_finite() && configured > 0.0 {
            configured
        } else {
            1.0
        }
    }

    /// The device LightGBM will actually be handed — and therefore the one
    /// recorded in the artifact.
    ///
    /// This function is the single answer to "which device did this train
    /// on". It used to be the constant `"cpu"`, while `fit_internal`
    /// separately wrote `device_type=gpu` into the training params whenever a
    /// card was visible. Both could not be right: on any `nvidia-smi` host the
    /// run asked LightGBM for the OpenCL learner (which is not in our build,
    /// so it Fatal'd at fit) and the artifact recorded `cpu` for a model that
    /// never trained. `fit_internal` now calls this instead of deciding for
    /// itself, so there is one decision with one record of it.
    ///
    /// Four things must all hold before this returns `cuda`:
    ///   1. the operator opted in (`models.tree_runtime.lightgbm_gpu`);
    ///   2. the build linked the CUDA learner (`lightgbm-gpu` feature);
    ///   3. the operator's device preference is not an explicit `cpu`;
    ///   4. a GPU is actually visible.
    /// An explicit CUDA request fails when any prerequisite is missing. Auto
    /// remains CPU only when the LightGBM-specific opt-in is off or no NVIDIA
    /// device is visible; after Auto selects CUDA, every setup/training error
    /// is returned to the caller.
    ///
    /// Note the vocabulary: `cuda`, never `gpu`. In LightGBM those name two
    /// different tree learners, and `gpu` is the OpenCL one we do not build.
    #[cfg(feature = "lightgbm")]
    fn resolved_cuda_device(&self) -> Result<ResolvedCudaDevicePolicy> {
        let requested = parse_tree_cuda_device_policy(&self.config.requested_device_policy)?;
        if !lightgbm_gpu_allowed() {
            if matches!(requested, CudaDevicePolicy::Gpu { .. }) {
                bail!(
                    "LightGBM CUDA policy `{}` cannot be honoured because models.tree_runtime.lightgbm_gpu is false",
                    self.config.requested_device_policy
                );
            }
            return Ok(ResolvedCudaDevicePolicy::Cpu);
        }

        let resolved = self.config.resolved_cuda_device()?;
        if matches!(resolved, ResolvedCudaDevicePolicy::Cuda { .. })
            && !cfg!(feature = "lightgbm-gpu")
        {
            bail!(
                "LightGBM resolved `{}` to CUDA, but this binary was built without the `lightgbm-gpu` feature",
                self.config.requested_device_policy
            );
        }
        Ok(resolved)
    }

    #[cfg(feature = "lightgbm")]
    fn resolved_device_parts(&self) -> Result<(&'static str, Option<usize>)> {
        Ok(match self.resolved_cuda_device()? {
            ResolvedCudaDevicePolicy::Cpu => ("cpu", None),
            ResolvedCudaDevicePolicy::Cuda { ordinal } => ("cuda", Some(ordinal)),
        })
    }

    #[cfg(feature = "lightgbm")]
    fn resolved_params_for(
        &self,
        device_type: &str,
        cuda_ordinal: Option<usize>,
    ) -> Result<HashMap<String, ParamValue>> {
        let mut params = self.config.params.clone();
        params.insert(
            "boosting_type".into(),
            ParamValue::String(self.boosting_type()),
        );
        params.insert(
            "device_type".into(),
            ParamValue::String(device_type.to_string()),
        );
        params.remove("gpu_device_id");
        if let Some(cuda_ordinal) = cuda_ordinal {
            let cuda_ordinal = i32::try_from(cuda_ordinal)
                .context("LightGBM CUDA ordinal exceeds the supported i32 parameter range")?;
            params.insert("gpu_device_id".into(), ParamValue::Int(cuda_ordinal));
        }
        params.insert(
            "probability_temperature".into(),
            ParamValue::Float(self.probability_temperature()),
        );
        params.insert("gpu_only".into(), ParamValue::Bool(self.config.gpu_only));
        params.insert(
            "cpu_threads".into(),
            ParamValue::Int(self.config.cpu_threads.unwrap_or(1).max(1) as i32),
        );
        Ok(params)
    }

    #[cfg(feature = "lightgbm")]
    fn runtime_artifact(&self) -> Result<LightGBMRuntimeArtifact> {
        let (effective_device_type, cuda_ordinal) = self.resolved_device_parts()?;
        Ok(LightGBMRuntimeArtifact {
            configured_params: self.config.params.clone(),
            resolved_params: self.resolved_params_for(effective_device_type, cuda_ordinal)?,
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
            device_pref: self.config.device_pref,
            requested_device_policy: self.config.requested_device_policy.clone(),
            effective_device_type: effective_device_type.to_string(),
            cuda_ordinal,
            boosting_type: self.boosting_type(),
            probability_temperature: self.probability_temperature(),
            gpu_only: self.config.gpu_only,
            cpu_threads: self.config.cpu_threads.unwrap_or(1).max(1),
        })
    }

    #[cfg(feature = "lightgbm")]
    fn apply_runtime_artifact(&mut self, artifact: LightGBMRuntimeArtifact) {
        self.config.device_pref = artifact.device_pref;
        self.config.requested_device_policy = artifact.requested_device_policy;
        self.config.gpu_only = artifact.gpu_only;
        self.config.cpu_threads = Some(artifact.cpu_threads.max(1));
        self.config.params = artifact.configured_params;
        self.feature_columns = artifact.feature_columns;
        self.training_summary = Some(artifact.training_summary);
    }

    #[cfg(feature = "lightgbm")]
    fn runtime_profile_path(root: &Path) -> PathBuf {
        root.join(LIGHTGBM_RUNTIME_FILE_NAME)
    }

    #[cfg(feature = "lightgbm")]
    fn read_runtime_artifact(root: &Path) -> Result<Option<LightGBMRuntimeArtifact>> {
        let path = Self::runtime_profile_path(root);
        if !path.exists() {
            return Ok(None);
        }
        let profile = read_tree_json_artifact(&path, "LightGBM runtime artifact")?;
        Ok(Some(profile))
    }

    #[cfg(feature = "lightgbm")]
    fn prediction_params(&self, lease_width: usize) -> String {
        format!(
            "num_threads={}",
            self.config
                .cpu_threads
                .unwrap_or(lease_width)
                .min(lease_width)
                .max(1)
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

    fn runtime_predictions(
        &self,
        model_name: &str,
        probabilities: &Array2<f64>,
    ) -> Result<Vec<RuntimePrediction>> {
        build_tree_runtime_predictions(model_name, probabilities, "lightgbm_native")
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
        if self.model.is_none() {
            bail!("LightGBM runtime state is missing its native booster");
        }
        Ok(())
    }

    #[cfg(feature = "lightgbm")]
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
        let requested_device = parse_tree_cuda_device_policy(&artifact.requested_device_policy)
            .with_context(|| {
                format!(
                    "LightGBM runtime artifact has invalid requested device policy `{}`",
                    artifact.requested_device_policy
                )
            })?;
        // `cuda` = the CUDA tree learner (the one we build). `gpu` = the
        // OpenCL learner; still accepted so artifacts written before
        // 2026-08-02 load, but nothing produces it any more.
        if !matches!(
            artifact.effective_device_type.as_str(),
            "cpu" | "gpu" | "cuda"
        ) {
            bail!(
                "LightGBM runtime artifact effective_device_type must be 'cpu', 'cuda' or \
                 (legacy) 'gpu', got {}",
                artifact.effective_device_type
            );
        }
        match artifact.effective_device_type.as_str() {
            "cpu" => {
                if artifact.cuda_ordinal.is_some() {
                    bail!(
                        "LightGBM CPU runtime artifact must not record a CUDA ordinal, got {:?}",
                        artifact.cuda_ordinal
                    );
                }
                if matches!(requested_device, CudaDevicePolicy::Gpu { .. }) {
                    bail!(
                        "LightGBM runtime artifact requested explicit CUDA but recorded CPU execution"
                    );
                }
                if artifact.resolved_params.contains_key("gpu_device_id") {
                    bail!(
                        "LightGBM CPU runtime artifact must not contain a gpu_device_id parameter"
                    );
                }
            }
            "cuda" | "gpu" => {
                let cuda_ordinal = artifact
                    .cuda_ordinal
                    .context("LightGBM CUDA runtime artifact must record the exact CUDA ordinal")?;
                if matches!(requested_device, CudaDevicePolicy::Cpu) {
                    bail!("LightGBM runtime artifact requested CPU but recorded CUDA execution");
                }
                if let CudaDevicePolicy::Gpu { ordinal } = requested_device
                    && ordinal != cuda_ordinal
                {
                    bail!(
                        "LightGBM runtime artifact CUDA ordinal mismatch: requested {ordinal}, recorded {cuda_ordinal}"
                    );
                }
                let recorded_param = artifact.resolved_params.get("gpu_device_id");
                let expected_param = i32::try_from(cuda_ordinal).context(
                    "LightGBM runtime artifact CUDA ordinal exceeds the supported i32 parameter range",
                )?;
                if recorded_param != Some(&ParamValue::Int(expected_param)) {
                    bail!(
                        "LightGBM runtime artifact gpu_device_id mismatch: expected {expected_param}, got {:?}",
                        recorded_param
                    );
                }
            }
            _ => unreachable!("validated LightGBM device vocabulary above"),
        }
        Ok(())
    }

    #[cfg(feature = "lightgbm")]
    fn validate_runtime_device_for_load(artifact: &LightGBMRuntimeArtifact) -> Result<()> {
        let requested = parse_tree_cuda_device_policy(&artifact.requested_device_policy)?;
        let visible_nvidia_devices = nvidia_gpu_count();
        let resolved = if !lightgbm_gpu_allowed() {
            if matches!(requested, CudaDevicePolicy::Gpu { .. }) {
                bail!(
                    "LightGBM CUDA artifact policy `{}` cannot be honoured because models.tree_runtime.lightgbm_gpu is false",
                    artifact.requested_device_policy
                );
            }
            ResolvedCudaDevicePolicy::Cpu
        } else {
            resolve_cuda_device_policy(&artifact.requested_device_policy, visible_nvidia_devices)?
        };
        if matches!(resolved, ResolvedCudaDevicePolicy::Cuda { .. })
            && !cfg!(feature = "lightgbm-gpu")
        {
            bail!(
                "LightGBM artifact resolves CUDA from policy `{}`, but this build lacks `lightgbm-gpu`",
                artifact.requested_device_policy
            );
        }
        let recorded = match artifact.effective_device_type.as_str() {
            "cpu" => ResolvedCudaDevicePolicy::Cpu,
            "cuda" | "gpu" => ResolvedCudaDevicePolicy::Cuda {
                ordinal: artifact
                    .cuda_ordinal
                    .context("LightGBM CUDA artifact is missing its recorded ordinal")?,
            },
            other => bail!("LightGBM artifact has unsupported recorded device `{other}`"),
        };
        if artifact.gpu_only && matches!(resolved, ResolvedCudaDevicePolicy::Cpu) {
            bail!(
                "LightGBM gpu-only artifact cannot relocate to CPU because no NVIDIA device is visible"
            );
        }
        let auto_cpu_relocation = matches!(requested, CudaDevicePolicy::Auto)
            && matches!(recorded, ResolvedCudaDevicePolicy::Cuda { .. })
            && matches!(resolved, ResolvedCudaDevicePolicy::Cpu)
            && visible_nvidia_devices == 0
            && !artifact.gpu_only;
        if !auto_cpu_relocation && recorded != resolved {
            bail!(
                "LightGBM runtime device drift on load: recorded {:?}, resolved {:?} from policy `{}`",
                recorded,
                resolved,
                artifact.requested_device_policy
            );
        }
        Ok(())
    }

    #[cfg(feature = "lightgbm")]
    fn resolve_runtime_metadata(
        path: &Path,
        runtime_artifact: Option<&LightGBMRuntimeArtifact>,
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
        } else {
            bail!(
                "LightGBM metadata sidecar and runtime artifact are missing at {}",
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
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
        lease_width: usize,
    ) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        {
            let _ = (x, y, val_x, val_y, lease_width);
            bail!("LightGBM native backend unavailable: compile with the `lightgbm` feature")
        }
        #[cfg(feature = "lightgbm")]
        {
            if x.n_samples() == 0 || y.is_empty() {
                bail!("LightGBM requires non-empty training features and labels");
            }
            if x.n_samples() != y.len() {
                bail!(
                    "LightGBM requires matching feature and label rows: {} features vs {} labels",
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
            // Resolve the device once and fail before allocating a dataset or
            // entering native training. `gpu_only` means no CPU fallback,
            // regardless of whether the missing prerequisite is the config
            // opt-in, CUDA build feature, explicit device policy, or hardware.
            let (device_type, cuda_ordinal) = self.resolved_device_parts()?;
            if self.config.gpu_only && device_type != "cuda" {
                anyhow::bail!(
                    "LightGBM gpu-only mode is set but the resolved device is `{device_type}`. \
                     Check, in this order: models.tree_runtime.lightgbm_gpu (must be true), \
                     that this binary was built with --features gpu-cuda (the CUDA tree \
                     learner), models.tree_runtime.device (must not be `cpu`), and that a \
                     GPU is visible to this process."
                );
            }

            let mut params = self.build_training_params();

            // ONE device decision, made by resolved_device_parts() and used
            // both here and in the artifact. The block that used to live here
            // wrote "gpu" — LightGBM's OpenCL learner — whenever a card was
            // visible, without ever consulting the strict resolver, which
            // was simultaneously reporting "cpu" into the runtime artifact.
            params["device_type"] = serde_json::json!(device_type);
            if device_type == "cuda" {
                let cuda_ordinal = cuda_ordinal
                    .context("LightGBM CUDA resolution did not produce a device ordinal")?;
                params["gpu_device_id"] = serde_json::json!(cuda_ordinal);
                // LightGBM 4.6 has no CUDA multiclass metric implementation;
                // requesting multi_logloss emits a warning and copies scores
                // back for CPU evaluation on every round. NeoEthos performs
                // its canonical held-out evaluation after fit, so CUDA mode
                // trains the requested fixed iteration budget with no native
                // CPU metric fallback. CUDA also does not support the sparse
                // feature optimization and requires it disabled at dataset
                // construction time.
                params["metric"] = serde_json::json!("None");
                params["is_enable_sparse"] = serde_json::json!(false);
            }
            tracing::info!(
                target: "neoethos_models::lightgbm",
                idx = self.idx,
                device_type = device_type,
                cuda_ordinal = ?cuda_ordinal,
                num_threads = self.config.cpu_threads.unwrap_or(1).max(1),
                "LightGBM training device resolved"
            );

            let (flat_x, _rows, cols) = feature_frame_to_tree_f32_row_major(x)?;
            let labels = remap_labels_to_tree_targets(y)?;
            if labels.len() != x.n_samples() {
                anyhow::bail!(
                    "LightGBM training row count mismatch: {} features rows, {} labels",
                    x.n_samples(),
                    labels.len()
                );
            }
            let dataset = lightgbm3::Dataset::from_slice_with_params(
                &flat_x,
                &labels,
                cols as i32,
                true,
                &params,
            )
            .context("create LightGBM dataset from typed feature frame")?;

            let valid_dataset = match (val_x, val_y) {
                (Some(vx), Some(vy)) => {
                    if vx.n_features() != x.n_features() {
                        anyhow::bail!(
                            "LightGBM validation column count mismatch: train {}, val {}",
                            x.n_features(),
                            vx.n_features()
                        );
                    }
                    if vx.n_samples() != vy.len() {
                        anyhow::bail!(
                            "LightGBM validation row/label mismatch: {} rows vs {} labels",
                            vx.n_samples(),
                            vy.len()
                        );
                    }
                    let (vflat, _vrows, vcols) = feature_frame_to_tree_f32_row_major(vx)?;
                    let vlabels = remap_labels_to_tree_targets(vy)?;
                    let valid = lightgbm3::Dataset::from_slice_with_reference_and_params(
                        &vflat,
                        &vlabels,
                        vcols as i32,
                        true,
                        Some(&dataset),
                        &params,
                    )
                    .context("create LightGBM validation dataset from typed feature frame")?;
                    // Default early_stopping_rounds when caller did not
                    // explicitly set one. 50 rounds is a conservative
                    // patience for `num_iterations >= 200`.
                    if device_type != "cuda"
                        && !params
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

            self.feature_columns = feature_columns_from_frame(x);
            self.training_summary = Some(default_training_summary(x));
            self.model = Some(model);
            Ok(())
        }
    }
}

impl ExpertModel for LightGBMExpert {
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
            #[cfg(feature = "lightgbm")]
            {
                ensure_feature_columns_match(&self.feature_columns, x)?;
                let model = self.model.as_ref().context("LightGBM not trained")?;
                let (flat_x, rows, cols) = feature_frame_to_tree_f32_row_major(x)?;
                let probabilities = model
                    .predict_with_params(
                        &flat_x,
                        cols as i32,
                        true,
                        &self.prediction_params(lease.width().get()),
                    )
                    .context("predict LightGBM class probabilities")?;
                let probabilities = Array2::from_shape_vec((rows, 3), probabilities)
                    .context("reshape LightGBM class probabilities")?;
                let probabilities = calibrate_three_class_probabilities(
                    probabilities,
                    self.probability_temperature(),
                    "LightGBM",
                )?;
                normalize_three_class_probabilities(probabilities, "LightGBM")
            }
            #[cfg(not(feature = "lightgbm"))]
            {
                let _ = x;
                bail!("LightGBM native backend unavailable: compile with the `lightgbm` feature")
            }
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        #[cfg(not(feature = "lightgbm"))]
        {
            let _ = path;
            bail!("LightGBM native backend unavailable: compile with the `lightgbm` feature")
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
            let runtime_profile = self.runtime_artifact()?;
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
            let model = self.model.as_ref().context("LightGBM not trained")?;
            model
                .save_file(
                    model_path
                        .to_str()
                        .context("LightGBM artifact path must be valid unicode")?,
                )
                .with_context(|| format!("save LightGBM artifact {}", model_path.display()))?;
            Ok(())
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        {
            let _ = path;
            bail!("LightGBM native backend unavailable: compile with the `lightgbm` feature")
        }
        #[cfg(feature = "lightgbm")]
        {
            let (model_path, _) = tree_artifact_paths(path, LIGHTGBM_MODEL_FILE_NAME);
            let runtime_profile = Self::read_runtime_artifact(path)?;
            let metadata = Self::resolve_runtime_metadata(path, runtime_profile.as_ref())?;
            let metadata_feature_columns = metadata.feature_columns.clone();
            let metadata_training_summary = metadata.training_summary.clone();
            if let Some(runtime_profile) = runtime_profile {
                Self::validate_runtime_artifact(
                    &runtime_profile,
                    &metadata_feature_columns,
                    &metadata_training_summary,
                )?;
                Self::validate_runtime_device_for_load(&runtime_profile)?;
                self.apply_runtime_artifact(runtime_profile);
            } else {
                self.feature_columns = metadata_feature_columns;
                self.training_summary = Some(metadata_training_summary);
                tracing::warn!(
                    path = %path.display(),
                    "LightGBM runtime.json missing; using metadata to restore runtime state"
                );
            }
            if !model_path.exists() {
                bail!(
                    "LightGBM native model artifact is missing at {}",
                    model_path.display()
                );
            }
            self.model = Some(
                lightgbm3::Booster::from_file(
                    model_path
                        .to_str()
                        .context("LightGBM artifact path must be valid unicode")?,
                )
                .with_context(|| format!("load LightGBM artifact {}", model_path.display()))?,
            );
            Ok(())
        }
    }
}

impl LightGBMExpert {
    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
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
    use super::{ExpertModel, LightGBMExpert};
    use crate::runtime::artifacts::TrainingSummaryMetadata;
    use crate::tree_models::config::{DevicePreference, ParamValue};
    use ndarray::Array2;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
    use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn frame_from_columns(columns: Vec<(&str, Vec<f64>)>) -> FeatureFrame {
        let rows = columns.first().map(|(_, values)| values.len()).unwrap_or(0);
        let columns = columns
            .into_iter()
            .map(|(name, values)| {
                FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; rows])
                    .expect("valid typed feature column")
            })
            .collect::<Vec<_>>();
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
        .expect("build typed feature frame")
    }

    fn single_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one is a valid worker limit");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("single-worker model test lease")
    }

    fn sample_three_class_dataset() -> (FeatureFrame, Vec<i32>) {
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

        (
            frame_from_columns(vec![
                ("momentum", momentum),
                ("trend", trend),
                ("volatility", volatility),
            ]),
            labels,
        )
    }

    /// A training fold where only 2 of the 3 classes appear (no -1/sell). The
    /// multiclass model is still configured `num_class=3`; this MUST train, not
    /// fail — some real symbol/TF folds genuinely lack one direction.
    #[cfg(feature = "lightgbm")]
    #[test]
    fn fit_tolerates_two_class_training_fold() {
        let mut momentum = Vec::new();
        let mut trend = Vec::new();
        let mut volatility = Vec::new();
        let mut labels = Vec::new();
        for idx in 0..40 {
            let o = idx as f64 * 0.01;
            momentum.push(0.78 + o);
            trend.push(0.7 + o * 0.8);
            volatility.push(0.35 + o * 0.2);
            labels.push(1_i32);
        }
        for idx in 0..40 {
            let o = idx as f64 * 0.01;
            momentum.push(-0.02 + o * 0.05);
            trend.push(-0.03 + o * 0.04);
            volatility.push(0.12 + o * 0.03);
            labels.push(0_i32);
        }
        let x = frame_from_columns(vec![
            ("momentum", momentum),
            ("trend", trend),
            ("volatility", volatility),
        ]);
        let lease = single_worker_lease();
        let mut expert = LightGBMExpert::new(7, None);
        let res = expert.fit(&x, &labels, &lease);
        assert!(res.is_ok(), "2-class fold must train, got: {:?}", res.err());
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
    fn lightgbm_trains_three_class_probabilities_and_persists_artifacts() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-artifact");
        let lease = single_worker_lease();

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");

        let probabilities = expert
            .predict_proba(&x, &lease)
            .expect("predict should succeed");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
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
    fn lightgbm_rejects_corrupt_native_artifact_without_a_surrogate() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-corrupt-artifact");
        let lease = single_worker_lease();

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");

        std::fs::write(artifact_dir.join("model.txt"), b"corrupt lightgbm model")
            .expect("overwrite native model artifact");

        let mut loaded = LightGBMExpert::new(7, None);
        let error = loaded
            .load(&artifact_dir)
            .expect_err("corrupt native artifact must fail closed");
        assert!(error.to_string().contains("load LightGBM artifact"));
    }

    #[test]
    fn lightgbm_load_uses_runtime_profile_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-missing-metadata-sidecar");
        let lease = single_worker_lease();

        let mut expert = LightGBMExpert::new(7, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("metadata.json"))
            .expect("remove metadata sidecar to force runtime-profile reconstruction");

        let mut loaded = LightGBMExpert::new(7, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct runtime metadata from runtime profile");

        let probabilities = loaded
            .predict_proba(&x, &lease)
            .expect("prediction should succeed after metadata reconstruction");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
    }

    #[test]
    fn lightgbm_load_uses_metadata_when_runtime_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("lightgbm-missing-runtime-sidecar");
        let lease = single_worker_lease();

        let mut expert = LightGBMExpert::new(11, None);
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::remove_file(artifact_dir.join("runtime.json"))
            .expect("remove runtime sidecar to force metadata reconstruction");

        let mut loaded = LightGBMExpert::new(11, None);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct runtime state from metadata");

        let probabilities = loaded
            .predict_proba(&x, &lease)
            .expect("prediction should succeed after runtime reconstruction");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
    }

    /// The default config keeps LightGBM on the CPU, whatever the host has.
    ///
    /// This is the guard on the SELECTION half of the 2026-08-02 device fix.
    /// Enabling the CUDA tree learner in the build is a capability change and
    /// is unconditional; letting a run USE it changes which trees get grown,
    /// so it waits for `models.tree_runtime.lightgbm_gpu`. If someone later
    /// flips that default, this test is what tells them they did.
    #[test]
    fn lightgbm_device_stays_cpu_until_the_operator_opts_in() {
        // `Auto` is the shipped default device preference — the case that
        // would resolve to `cuda` if the config gate were open.
        let mut expert = LightGBMExpert::new(0, None);
        expert.config.device_pref = DevicePreference::Auto;
        assert!(
            !crate::tree_models::config::lightgbm_gpu_allowed(),
            "models.tree_runtime.lightgbm_gpu must default to false"
        );
        assert_eq!(
            expert
                .resolved_device_parts()
                .expect("default LightGBM device policy must resolve"),
            ("cpu", None),
            "default config must resolve LightGBM to the CPU learner"
        );
        // And the artifact must record the same string the trainer was given
        // — the two disagreeing is the defect this replaced.
        //
        // 2026-08-09: this assertion never ran. `runtime_artifact()` builds
        // `training_summary` BEFORE resolving the device, and on an unfitted
        // expert `stored_training_summary()` calls
        // `TrainingSummaryMetadata::new(0, 0, 0)`, whose `dataset_rows > 0`
        // assert panics — so the test aborted before it could compare a single
        // device string. Give the expert the summary a fitted one would carry,
        // so the device-parity check this test exists for actually executes.
        expert.training_summary = Some(TrainingSummaryMetadata::new(9, 7, 2));
        assert_eq!(
            expert
                .runtime_artifact()
                .expect("default LightGBM runtime artifact must resolve")
                .effective_device_type,
            "cpu",
            "artifact device must match the resolved training device"
        );
    }

    /// `gpu` was LightGBM's OpenCL learner and we never built it; `cuda` is
    /// the learner we do build. Artifacts written before the fix carry `gpu`,
    /// so loading them must still work.
    #[test]
    fn lightgbm_runtime_artifact_accepts_cuda_and_legacy_gpu() {
        let make = |device: &str| {
            let cuda_ordinal: Option<usize> = matches!(device, "cuda" | "gpu").then_some(0);
            let mut resolved_params =
                HashMap::from([("device_type".into(), ParamValue::String(device.into()))]);
            if let Some(cuda_ordinal) = cuda_ordinal {
                resolved_params.insert(
                    "gpu_device_id".into(),
                    ParamValue::Int(
                        i32::try_from(cuda_ordinal).expect("test CUDA ordinal must fit in i32"),
                    ),
                );
            }
            super::LightGBMRuntimeArtifact {
                configured_params: HashMap::from([(
                    "boosting_type".into(),
                    ParamValue::String("gbdt".into()),
                )]),
                resolved_params,
                feature_columns: vec!["momentum".into(), "trend".into()],
                training_summary: TrainingSummaryMetadata::new(9, 9, 0),
                device_pref: DevicePreference::Auto,
                requested_device_policy: "auto".into(),
                effective_device_type: device.into(),
                cuda_ordinal,
                boosting_type: "gbdt".into(),
                probability_temperature: 1.0,
                gpu_only: false,
                cpu_threads: 4,
            }
        };
        let columns = ["momentum".to_string(), "trend".to_string()];
        let summary = TrainingSummaryMetadata::new(9, 9, 0);
        for device in ["cpu", "cuda", "gpu"] {
            LightGBMExpert::validate_runtime_artifact(&make(device), &columns, &summary)
                .unwrap_or_else(|err| panic!("device `{device}` should validate: {err}"));
        }
        let err = LightGBMExpert::validate_runtime_artifact(&make("opencl"), &columns, &summary)
            .expect_err("an unknown device name must not validate");
        assert!(err.to_string().contains("effective_device_type"));
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
            requested_device_policy: "cpu".into(),
            effective_device_type: "cpu".into(),
            cuda_ordinal: None,
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
        let artifact_dir = unique_temp_dir("lightgbm-missing-summary");

        let mut expert = LightGBMExpert::new(7, None);
        expert.feature_columns = vec!["momentum".into(), "trend".into(), "volatility".into()];
        expert.training_summary = None;

        let err = expert
            .save(&artifact_dir)
            .expect_err("save should fail without training summary");
        assert!(err.to_string().contains("training summary"));
    }
}
