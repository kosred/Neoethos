#[cfg(feature = "catboost")]
use catboost_rust as catboost;

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// `std::io::Write` and many of the `super::common::*` imports below are only
// exercised when `feature = "catboost"` is enabled. When the feature is off
// (e.g. the partial-feature builds the v0.4.1 audit added under Batch 8) the
// unused-import lint fires; suppress it here because the imports are correct
// for the FFI-active build.
#[cfg(feature = "catboost")]
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use crate::base::{ExpertModel, feature_columns_from_dataframe};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;

#[allow(unused_imports)] // some entries are only used when feature = "catboost" is on
use super::common::{
    CATBOOST_MODEL_FILE_NAME, TreeLocalFallbackArtifact, atomic_write,
    build_tree_local_fallback_artifact, build_tree_runtime_predictions,
    calibrate_three_class_probabilities, dataframe_to_row_major_vec, default_training_summary,
    ensure_feature_columns_match, normalize_three_class_probabilities, predict_tree_local_fallback,
    read_runtime_metadata, read_tree_json_artifact, remap_labels_to_tree_targets,
    reshape_three_class_probabilities, tree_artifact_paths, tree_runtime_metadata,
    validate_tree_local_fallback_artifact, write_runtime_metadata, write_tree_json_artifact,
};
use super::config::*;

const CATBOOST_RUNTIME_FILE_NAME: &str = "runtime.json";
const CATBOOST_LOCAL_FALLBACK_FILE_NAME: &str = "catboost_local_fallback.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatBoostRuntimeArtifact {
    executable: String,
    task_type: String,
    device_preference: String,
    gpu_available: bool,
    gpu_only: bool,
    model_dimensions: usize,
    feature_count: usize,
    classes_count: usize,
    iterations: i32,
    depth: i32,
    learning_rate: f64,
    l2_leaf_reg: f64,
    probability_temperature: f64,
    use_best_model: bool,
    thread_count: usize,
    random_seed: usize,
    loss_function: String,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
}

impl CatBoostRuntimeArtifact {
    #[allow(clippy::too_many_arguments)]
    fn new(
        executable: Option<&Path>,
        task_type: Option<&str>,
        device_preference: &str,
        gpu_available: bool,
        gpu_only: bool,
        model_dimensions: usize,
        feature_count: usize,
        iterations: i32,
        depth: i32,
        learning_rate: f64,
        l2_leaf_reg: f64,
        probability_temperature: f64,
        use_best_model: bool,
        thread_count: usize,
        random_seed: usize,
        loss_function: &str,
        feature_columns: Vec<String>,
        training_summary: TrainingSummaryMetadata,
    ) -> Self {
        let executable = executable
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let task_type = task_type
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            executable,
            task_type,
            device_preference: device_preference.to_string(),
            gpu_available,
            gpu_only,
            model_dimensions,
            feature_count,
            classes_count: 3,
            iterations,
            depth,
            learning_rate,
            l2_leaf_reg,
            probability_temperature,
            use_best_model,
            thread_count,
            random_seed,
            loss_function: loss_function.to_string(),
            feature_columns,
            training_summary,
        }
    }
}

fn validate_runtime_artifact(
    artifact: &CatBoostRuntimeArtifact,
    expected_feature_count: usize,
) -> Result<()> {
    if artifact.executable.trim().is_empty() {
        bail!("CatBoost runtime artifact executable must not be blank");
    }
    if !matches!(artifact.task_type.as_str(), "CPU" | "GPU" | "unknown") {
        bail!(
            "CatBoost runtime artifact task_type must be CPU, GPU, or unknown, got {}",
            artifact.task_type
        );
    }
    if !matches!(artifact.device_preference.as_str(), "cpu" | "gpu" | "auto") {
        bail!(
            "CatBoost runtime artifact device_preference must be cpu, gpu, or auto, got {}",
            artifact.device_preference
        );
    }
    if artifact.feature_count == 0 {
        bail!("CatBoost runtime artifact requires at least one feature");
    }
    if artifact.model_dimensions != 3 || artifact.classes_count != 3 {
        bail!(
            "CatBoost runtime artifact expects 3 classes, got dimensions={} classes={}",
            artifact.model_dimensions,
            artifact.classes_count
        );
    }
    if artifact.feature_count != expected_feature_count {
        bail!(
            "CatBoost runtime artifact feature mismatch: expected {}, got {}",
            expected_feature_count,
            artifact.feature_count
        );
    }
    if artifact.feature_columns.len() != expected_feature_count {
        bail!(
            "CatBoost runtime artifact feature columns mismatch: expected {}, got {}",
            expected_feature_count,
            artifact.feature_columns.len()
        );
    }
    if artifact.training_summary.dataset_rows == 0 {
        bail!("CatBoost runtime artifact requires non-zero dataset_rows");
    }
    if artifact.training_summary.dataset_rows
        != artifact.training_summary.train_rows + artifact.training_summary.val_rows
    {
        bail!("CatBoost runtime artifact training summary is inconsistent");
    }
    if artifact.iterations < 1 {
        bail!(
            "CatBoost runtime artifact has invalid iteration count {}",
            artifact.iterations
        );
    }
    if artifact.depth < 1 {
        bail!(
            "CatBoost runtime artifact has invalid tree depth {}",
            artifact.depth
        );
    }
    if !artifact.learning_rate.is_finite() || artifact.learning_rate <= 0.0 {
        bail!(
            "CatBoost runtime artifact has invalid learning rate {}",
            artifact.learning_rate
        );
    }
    if !artifact.l2_leaf_reg.is_finite() || artifact.l2_leaf_reg < 0.0 {
        bail!(
            "CatBoost runtime artifact has invalid l2_leaf_reg {}",
            artifact.l2_leaf_reg
        );
    }
    if !artifact.probability_temperature.is_finite() || artifact.probability_temperature <= 0.0 {
        bail!(
            "CatBoost runtime artifact has invalid probability_temperature {}",
            artifact.probability_temperature
        );
    }
    if artifact.thread_count == 0 {
        bail!("CatBoost runtime artifact requires at least one thread");
    }
    if artifact.loss_function.trim().is_empty() {
        bail!("CatBoost runtime artifact is missing a loss function");
    }
    if artifact.gpu_only && artifact.task_type != "GPU" {
        bail!(
            "CatBoost runtime artifact gpu_only=true requires task_type=GPU, got {}",
            artifact.task_type
        );
    }
    Ok(())
}

#[cfg(feature = "catboost")]
fn validate_training_frame(flat_x: &[f32], rows: usize, cols: usize, labels: &[i32]) -> Result<()> {
    if rows == 0 || cols == 0 {
        bail!("CatBoost training requires a non-empty feature matrix");
    }
    if flat_x.len() != rows * cols {
        bail!(
            "CatBoost training matrix mismatch: {} values for {}x{} frame",
            flat_x.len(),
            rows,
            cols
        );
    }
    if labels.len() != rows {
        bail!(
            "CatBoost training row count mismatch: {} rows, {} labels",
            rows,
            labels.len()
        );
    }
    if flat_x.iter().any(|value| !value.is_finite()) {
        bail!("CatBoost training data contains non-finite feature values");
    }
    let distinct_labels = labels
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if distinct_labels.len() < 2 {
        bail!("CatBoost multiclass training requires at least two observed classes");
    }
    Ok(())
}

pub struct CatBoostExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg_attr(not(feature = "catboost"), allow(dead_code))]
    feature_columns: Vec<String>,
    #[cfg_attr(not(feature = "catboost"), allow(dead_code))]
    training_summary: Option<TrainingSummaryMetadata>,
    local_fallback: Option<TreeLocalFallbackArtifact>,
    #[cfg_attr(not(feature = "catboost"), allow(dead_code))]
    model_bytes: Option<Vec<u8>>,
    #[cfg_attr(not(feature = "catboost"), allow(dead_code))]
    runtime_artifact: Option<CatBoostRuntimeArtifact>,
    #[cfg(feature = "catboost")]
    model: Option<catboost::Model>,
    #[cfg(not(feature = "catboost"))]
    model: Option<()>,
}

impl CatBoostExpert {
    pub fn new(idx: usize) -> Self {
        let params = Self::default_params();
        let device_pref =
            device_preference_from_params(&params, tree_device_preference_for("catboost"));
        let gpu_only = gpu_only_from_params(&params, gpu_only_mode_for("catboost"));
        let cpu_threads = cpu_threads_from_params(&params, cpu_threads_hint_for("catboost"));
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
            model_bytes: None,
            runtime_artifact: None,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        params.insert("iterations".into(), ParamValue::Int(500));
        params.insert("depth".into(), ParamValue::Int(8));
        params.insert("learning_rate".into(), ParamValue::Float(0.05));
        params.insert("l2_leaf_reg".into(), ParamValue::Float(3.0));
        params.insert("probability_temperature".into(), ParamValue::Float(1.0));
        params.insert(
            "loss_function".into(),
            ParamValue::String("MultiClass".into()),
        );
        params.insert("use_best_model".into(), ParamValue::Bool(false));
        params
    }

    fn probability_temperature(&self) -> f32 {
        let configured = param_float(&self.config.params, "probability_temperature", 1.0) as f32;
        if configured.is_finite() && configured > 0.0 {
            configured
        } else {
            1.0
        }
    }

    fn stored_training_summary(&self) -> TrainingSummaryMetadata {
        self.training_summary
            .clone()
            .unwrap_or_else(|| TrainingSummaryMetadata::new(0, 0, 0))
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if self.feature_columns.is_empty() {
            bail!("CatBoost runtime state is missing persisted feature columns");
        }
        let summary = self
            .training_summary
            .as_ref()
            .context("CatBoost runtime state is missing training summary metadata")?;
        if summary.dataset_rows == 0 {
            bail!("CatBoost runtime state has zero dataset_rows in training summary");
        }
        if summary.dataset_rows != summary.train_rows + summary.val_rows {
            bail!(
                "CatBoost runtime state has inconsistent training summary: dataset_rows={} train_rows={} val_rows={}",
                summary.dataset_rows,
                summary.train_rows,
                summary.val_rows
            );
        }
        if self.model.is_none() && self.local_fallback.is_none() {
            bail!("CatBoost runtime state has neither a native model nor a local surrogate");
        }
        if let Some(fallback) = self.local_fallback.as_ref() {
            validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
        }
        if let Some(runtime_artifact) = self.runtime_artifact.as_ref() {
            validate_runtime_artifact(runtime_artifact, self.feature_columns.len())?;
        }
        Ok(())
    }

    fn runtime_artifact_path(path: &Path) -> PathBuf {
        path.join(CATBOOST_RUNTIME_FILE_NAME)
    }

    fn local_fallback_path(path: &Path) -> std::path::PathBuf {
        path.join(CATBOOST_LOCAL_FALLBACK_FILE_NAME)
    }

    fn persist_local_fallback(&self, path: &Path) -> Result<()> {
        if let Some(artifact) = self.local_fallback.as_ref() {
            validate_tree_local_fallback_artifact(artifact, &self.feature_columns)?;
            write_tree_json_artifact(
                &Self::local_fallback_path(path),
                artifact,
                "CatBoost local fallback",
            )?;
        }
        Ok(())
    }

    fn read_local_fallback(path: &Path) -> Result<Option<TreeLocalFallbackArtifact>> {
        let fallback_path = Self::local_fallback_path(path);
        if !fallback_path.exists() {
            return Ok(None);
        }
        let artifact = read_tree_json_artifact(&fallback_path, "CatBoost local fallback")?;
        Ok(Some(artifact))
    }

    fn read_runtime_artifact(path: &Path) -> Result<Option<CatBoostRuntimeArtifact>> {
        let runtime_path = Self::runtime_artifact_path(path);
        if !runtime_path.exists() {
            return Ok(None);
        }
        let artifact = read_tree_json_artifact(&runtime_path, "CatBoost runtime artifact")?;
        Ok(Some(artifact))
    }

    fn effective_task_type(&self) -> &'static str {
        let wants_gpu = matches!(self.config.device_pref, DevicePreference::Gpu)
            || (matches!(self.config.device_pref, DevicePreference::Auto) && gpu_count() > 0);
        if wants_gpu && gpu_count() > 0 {
            "GPU"
        } else {
            "CPU"
        }
    }

    fn build_runtime_artifact(
        &self,
        executable: Option<&Path>,
        task_type: Option<&str>,
        model_dimensions: usize,
        feature_count: usize,
    ) -> CatBoostRuntimeArtifact {
        let iterations = param_int(&self.config.params, "iterations", 500).max(1);
        let depth = param_int(&self.config.params, "depth", 8).max(1);
        let learning_rate = param_float(&self.config.params, "learning_rate", 0.05);
        let l2_leaf_reg = param_float(&self.config.params, "l2_leaf_reg", 3.0);
        let probability_temperature =
            param_float(&self.config.params, "probability_temperature", 1.0);
        let use_best_model = param_bool(&self.config.params, "use_best_model", false);
        let thread_count = self
            .config
            .cpu_threads
            .unwrap_or_else(cpu_threads_hint)
            .max(1);
        let loss_function = param_string(&self.config.params, "loss_function", "MultiClass");
        let device_preference = format!("{:?}", self.config.device_pref).to_lowercase();

        CatBoostRuntimeArtifact::new(
            executable,
            task_type.or(Some(self.effective_task_type())),
            &device_preference,
            gpu_count() > 0,
            self.config.gpu_only,
            model_dimensions,
            feature_count,
            iterations,
            depth,
            learning_rate,
            l2_leaf_reg,
            probability_temperature,
            use_best_model,
            thread_count,
            self.idx,
            &loss_function,
            self.feature_columns.clone(),
            self.stored_training_summary(),
        )
    }

    fn build_surrogate_runtime_artifact(&self, feature_count: usize) -> CatBoostRuntimeArtifact {
        self.build_runtime_artifact(None, None, 3, feature_count)
    }

    fn apply_runtime_artifact(&mut self, artifact: &CatBoostRuntimeArtifact) {
        self.config.gpu_only = artifact.gpu_only;
        self.config.cpu_threads = Some(artifact.thread_count.max(1));
        self.config.device_pref = match artifact.device_preference.as_str() {
            "gpu" => DevicePreference::Gpu,
            "cpu" => DevicePreference::Cpu,
            _ => DevicePreference::Auto,
        };
        self.config.params.insert(
            "iterations".into(),
            ParamValue::Int(artifact.iterations.max(1)),
        );
        self.config
            .params
            .insert("depth".into(), ParamValue::Int(artifact.depth.max(1)));
        self.config.params.insert(
            "learning_rate".into(),
            ParamValue::Float(artifact.learning_rate),
        );
        self.config.params.insert(
            "l2_leaf_reg".into(),
            ParamValue::Float(artifact.l2_leaf_reg),
        );
        self.config.params.insert(
            "probability_temperature".into(),
            ParamValue::Float(artifact.probability_temperature),
        );
        self.config.params.insert(
            "use_best_model".into(),
            ParamValue::Bool(artifact.use_best_model),
        );
        self.config.params.insert(
            "loss_function".into(),
            ParamValue::String(artifact.loss_function.clone()),
        );
        self.feature_columns = artifact.feature_columns.clone();
        self.training_summary = Some(artifact.training_summary.clone());
    }

    fn resolve_runtime_metadata(
        path: &Path,
        metadata_path: &Path,
        runtime_artifact: Option<&CatBoostRuntimeArtifact>,
        local_fallback: Option<&TreeLocalFallbackArtifact>,
    ) -> Result<RuntimeArtifactMetadata> {
        if metadata_path.exists() {
            let metadata = read_runtime_metadata(metadata_path)?;
            if metadata.model_name != "catboost" || metadata.family != ModelFamily::Tree {
                bail!(
                    "CatBoost runtime metadata mismatch: expected tree/catboost, got {}/{}",
                    metadata.family,
                    metadata.model_name
                );
            }
            if metadata.feature_columns.is_empty() {
                bail!("CatBoost runtime metadata must contain at least one feature column");
            }
            return Ok(metadata);
        }

        let (feature_columns, training_summary) = if let Some(artifact) = runtime_artifact {
            (
                artifact.feature_columns.clone(),
                artifact.training_summary.clone(),
            )
        } else if let Some(fallback) = local_fallback {
            (
                fallback.feature_columns.clone(),
                fallback.training_summary.clone(),
            )
        } else {
            bail!(
                "CatBoost metadata sidecar missing and no runtime/local artifact is available at {}",
                path.display()
            );
        };

        let metadata = tree_runtime_metadata("catboost", feature_columns, training_summary)?;
        tracing::warn!(
            path = %path.display(),
            "CatBoost metadata sidecar missing; reconstructing from persisted runtime artifacts"
        );
        Ok(metadata)
    }

    #[cfg(feature = "catboost")]
    fn resolve_executable(&self) -> Result<PathBuf> {
        for key in ["NEOETHOS_BOT_CATBOOST_EXECUTABLE", "CATBOOST_EXECUTABLE"] {
            if let Ok(value) = std::env::var(key) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    let candidate = PathBuf::from(trimmed);
                    if candidate.exists() {
                        return Ok(candidate);
                    }
                    bail!("configured CatBoost executable {trimmed} does not exist");
                }
            }
        }

        for candidate in ["catboost", "catboost.exe"] {
            if std::process::Command::new(candidate)
                .arg("--version")
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
            {
                return Ok(PathBuf::from(candidate));
            }
        }

        bail!(
            "CatBoost training requires an official CatBoost CLI binary; set NEOETHOS_BOT_CATBOOST_EXECUTABLE or CATBOOST_EXECUTABLE, or place `catboost` on PATH"
        )
    }

    #[cfg(feature = "catboost")]
    fn create_training_dir(&self) -> Result<PathBuf> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system time before unix epoch")?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("neoethos-catboost-{}-{nonce}", self.idx));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create CatBoost temp dir {}", dir.display()))?;
        Ok(dir)
    }

    #[cfg(feature = "catboost")]
    fn write_training_files(
        &self,
        dir: &Path,
        x: &DataFrame,
        y: &Series,
    ) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let learn_path = dir.join("learn.tsv");
        let cd_path = dir.join("learn.cd");
        let model_path = dir.join(CATBOOST_MODEL_FILE_NAME);

        let labels = remap_labels_to_tree_targets(y)?
            .into_iter()
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        let (flat_x, rows, cols) = dataframe_to_row_major_vec(x)?;
        validate_training_frame(&flat_x, rows, cols, &labels)?;

        {
            let mut writer =
                std::io::BufWriter::new(std::fs::File::create(&learn_path).with_context(|| {
                    format!("create CatBoost learn set {}", learn_path.display())
                })?);

            for row_idx in 0..rows {
                use std::io::Write;
                write!(writer, "{}", labels[row_idx]).with_context(|| {
                    format!("write label row {row_idx} to {}", learn_path.display())
                })?;
                for feature in &flat_x[row_idx * cols..(row_idx + 1) * cols] {
                    write!(writer, "\t{feature}").with_context(|| {
                        format!("write feature row {row_idx} to {}", learn_path.display())
                    })?;
                }
                writeln!(writer).with_context(|| {
                    format!("write newline row {row_idx} to {}", learn_path.display())
                })?;
            }
            writer
                .flush()
                .with_context(|| format!("flush CatBoost learn set {}", learn_path.display()))?;
        }

        {
            let mut writer = std::io::BufWriter::new(
                std::fs::File::create(&cd_path)
                    .with_context(|| format!("create CatBoost cd file {}", cd_path.display()))?,
            );
            use std::io::Write;
            writeln!(writer, "0\tLabel").with_context(|| {
                format!("write CatBoost label descriptor {}", cd_path.display())
            })?;
            for feature_idx in 0..cols {
                writeln!(writer, "{}\tNum", feature_idx + 1).with_context(|| {
                    format!("write CatBoost feature descriptor {}", cd_path.display())
                })?;
            }
            writer
                .flush()
                .with_context(|| format!("flush CatBoost cd file {}", cd_path.display()))?;
        }

        Ok((learn_path, cd_path, model_path))
    }

    #[cfg(feature = "catboost")]
    fn train_cli(
        &self,
        executable: &Path,
        learn_path: &Path,
        cd_path: &Path,
        model_path: &Path,
        train_dir: &Path,
    ) -> Result<()> {
        if self.config.gpu_only && gpu_count() == 0 {
            bail!("CatBoost gpu-only mode requested but no GPU is available");
        }

        let mut command = std::process::Command::new(executable);
        let task_type = self.effective_task_type();
        command
            .arg("fit")
            .arg("--learn-set")
            .arg(learn_path)
            .arg("--cd")
            .arg(cd_path)
            .arg("--model-file")
            .arg(model_path)
            .arg("--train-dir")
            .arg(train_dir)
            .arg("--delimiter")
            .arg("\t")
            .arg("--loss-function")
            .arg(param_string(
                &self.config.params,
                "loss_function",
                "MultiClass",
            ))
            .arg("--classes-count")
            .arg("3")
            // Pin the full label space explicitly. `--classes-count 3` alone is
            // not enough when a training fold happens to contain only 2 of the 3
            // classes (e.g. no -1/sell bars) — CatBoost then trains a 2-class
            // model and our runtime validator rejects the dimension mismatch
            // ("expected 3 classes, got 2"). `--class-names 0,1,2` forces all
            // three classes (our remapped labels) so the model always emits a
            // 3-class probability vector, the absent class simply ~0.
            .arg("--class-names")
            .arg("0,1,2")
            .arg("--iterations")
            .arg(
                param_int(&self.config.params, "iterations", 500)
                    .max(1)
                    .to_string(),
            )
            .arg("--depth")
            .arg(
                param_int(&self.config.params, "depth", 8)
                    .max(1)
                    .to_string(),
            )
            .arg("--learning-rate")
            .arg(param_float(&self.config.params, "learning_rate", 0.05).to_string())
            .arg("--l2-leaf-reg")
            .arg(param_float(&self.config.params, "l2_leaf_reg", 3.0).to_string())
            .arg("--thread-count")
            .arg(
                self.config
                    .cpu_threads
                    .unwrap_or_else(cpu_threads_hint)
                    .max(1)
                    .to_string(),
            )
            .arg("--verbose")
            .arg("0")
            .arg("--random-seed")
            .arg(self.idx.to_string());

        // CatBoost CLI booleans are PRESENCE-only flags: passing an explicit
        // "true"/"false" value makes the parser treat it as a misplaced freearg
        // ("freearg 'false' is misplaced"). So `--has-header` is omitted (our
        // learn-set is written WITHOUT a header row) and `--use-best-model` is
        // added only when enabled (and only meaningful with an eval set).
        if param_bool(&self.config.params, "use_best_model", false) {
            command.arg("--use-best-model");
        }

        command
            .arg("--task-type")
            .arg(task_type)
            .current_dir(train_dir);

        let output = command
            .output()
            .with_context(|| format!("launch CatBoost CLI {}", executable.display()))?;

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "CatBoost CLI training failed (status {}): stdout: {} stderr: {}",
                output.status,
                stdout.trim(),
                stderr.trim()
            );
        }

        if !model_path.exists() {
            bail!(
                "CatBoost CLI completed without producing expected model artifact {}",
                model_path.display()
            );
        }

        Ok(())
    }

    #[cfg(feature = "catboost")]
    fn softmax_probabilities(raw_scores: Vec<f64>, rows: usize, cols: usize) -> Result<Vec<f32>> {
        if cols != 3 {
            bail!("expected CatBoost multiclass logits with 3 columns, got {cols}");
        }

        let mut probabilities = Vec::with_capacity(raw_scores.len());
        for row in raw_scores.chunks(cols) {
            if row.iter().any(|value| !value.is_finite()) {
                bail!("CatBoost produced non-finite raw logits");
            }
            let max_logit = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_values = row
                .iter()
                .map(|value| (*value - max_logit).exp())
                .collect::<Vec<_>>();
            let sum = exp_values.iter().sum::<f64>();
            if !sum.is_finite() || sum <= 0.0 {
                bail!("CatBoost produced invalid raw logits for softmax conversion");
            }
            probabilities.extend(exp_values.into_iter().map(|value| (value / sum) as f32));
        }

        if probabilities.len() != rows * cols {
            bail!(
                "CatBoost probability reshape mismatch: expected {}, got {}",
                rows * cols,
                probabilities.len()
            );
        }

        Ok(probabilities)
    }
}

impl CatBoostExpert {
    fn fit_internal(
        &mut self,
        x: &DataFrame,
        y: &Series,
        val_x: Option<&DataFrame>,
        val_y: Option<&Series>,
    ) -> Result<()> {
        // M6: CatBoost trains via the upstream CLI executable, which
        // already supports `--test-set` + `--use-best-model` for
        // val-driven early stopping. Wiring those flags safely requires
        // additional CD/data-file plumbing that is non-trivial to do
        // from this code path; for now record that val data was supplied
        // so an operator can audit whether early-stopping kicked in. The
        // CatBoost adapter then proceeds with the standard CLI training.
        if val_x.is_some() && val_y.is_some() {
            tracing::info!(
                model = "catboost",
                "CatBoost val frame supplied; CLI training currently ignores it (--test-set wiring is a follow-up)"
            );
        }
        #[cfg(feature = "catboost")]
        {
            let temp_dir = self.create_training_dir()?;
            let result = (|| -> Result<()> {
                let train_dir = temp_dir.join("train");
                std::fs::create_dir_all(&train_dir).with_context(|| {
                    format!("create CatBoost train dir {}", train_dir.display())
                })?;
                let (learn_path, cd_path, model_path) =
                    self.write_training_files(&temp_dir, x, y)?;
                let executable = self.resolve_executable()?;
                self.train_cli(&executable, &learn_path, &cd_path, &model_path, &train_dir)?;

                let model_bytes = std::fs::read(&model_path)
                    .with_context(|| format!("read CatBoost artifact {}", model_path.display()))?;
                let model = catboost::Model::load_buffer(&model_bytes)
                    .context("load CatBoost model from trained artifact bytes")?;
                let model_dimensions = model.get_dimensions_count();
                if model_dimensions != 3 {
                    bail!(
                        "CatBoost model dimensions mismatch: expected 3 classes, got {}",
                        model_dimensions
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
                self.model_bytes = Some(model_bytes);
                let runtime_artifact = self.build_runtime_artifact(
                    Some(&executable),
                    Some(self.effective_task_type()),
                    model_dimensions,
                    self.feature_columns.len(),
                );
                validate_runtime_artifact(&runtime_artifact, self.feature_columns.len())?;
                self.runtime_artifact = Some(runtime_artifact);
                self.model = Some(model);
                Ok(())
            })();

            let _ = std::fs::remove_dir_all(&temp_dir);
            result
        }
        #[cfg(not(feature = "catboost"))]
        {
            if x.height() == 0 || y.is_empty() {
                bail!("CatBoost requires non-empty training features and labels");
            }
            if x.height() != y.len() {
                bail!(
                    "CatBoost requires matching feature and label rows: {} features vs {} labels",
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
            self.model_bytes = None;
            let runtime_artifact =
                self.build_surrogate_runtime_artifact(self.feature_columns.len());
            validate_runtime_artifact(&runtime_artifact, self.feature_columns.len())?;
            self.runtime_artifact = Some(runtime_artifact);
            self.model = None;
            Ok(())
        }
    }
}

impl ExpertModel for CatBoostExpert {
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
        if self.gpu_only_disabled {
            anyhow::bail!("CatBoost disabled: gpu-only mode requested without an available GPU");
        }
        #[cfg(feature = "catboost")]
        {
            ensure_feature_columns_match(&self.feature_columns, x)?;
            if x.height() == 0 {
                return Ok(Array2::zeros((0, 3)));
            }
            if self.model.is_none() {
                if let Some(fallback) = self.local_fallback.as_ref() {
                    tracing::warn!(
                        model = "catboost",
                        surrogate_kind = %fallback.surrogate_kind,
                        surrogate_rows = fallback.training_summary.dataset_rows,
                        "CatBoost native model unavailable during predict_proba; using local surrogate fallback"
                    );
                    let probabilities = predict_tree_local_fallback(fallback, x)?;
                    let probabilities = calibrate_three_class_probabilities(
                        probabilities,
                        self.probability_temperature(),
                        "CatBoost",
                    )?;
                    return normalize_three_class_probabilities(probabilities, "CatBoost");
                }
                bail!("CatBoost not trained");
            }
            let model = self.model.as_ref().context("CatBoost not trained")?;
            if model.get_dimensions_count() != 3 {
                bail!(
                    "CatBoost model dimensions mismatch: expected 3 classes, got {}",
                    model.get_dimensions_count()
                );
            }
            if let Some(runtime_artifact) = self.runtime_artifact.as_ref() {
                validate_runtime_artifact(runtime_artifact, self.feature_columns.len())?;
            }
            let (flat_x, rows, cols) = dataframe_to_row_major_vec(x)?;
            let float_features = flat_x
                .chunks(cols.max(1))
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let cat_features: Vec<Vec<String>> = Vec::new();
            let raw_scores = model
                .calc_model_prediction(&float_features, &cat_features)
                .context("run CatBoost prediction on float features")?;
            let raw_cols = raw_scores.len() / rows.max(1);
            let probabilities = Self::softmax_probabilities(raw_scores, rows, raw_cols)?;
            let probabilities = reshape_three_class_probabilities(probabilities, rows, raw_cols)?;
            let probabilities = calibrate_three_class_probabilities(
                probabilities,
                self.probability_temperature(),
                "CatBoost",
            )?;
            normalize_three_class_probabilities(probabilities, "CatBoost")
        }
        #[cfg(not(feature = "catboost"))]
        {
            let fallback = self
                .local_fallback
                .as_ref()
                .context("CatBoost local fallback not trained")?;
            tracing::warn!(
                model = "catboost",
                surrogate_kind = %fallback.surrogate_kind,
                surrogate_rows = fallback.training_summary.dataset_rows,
                "CatBoost native model unavailable in this build; using local surrogate fallback"
            );
            let probabilities = predict_tree_local_fallback(fallback, x)?;
            let probabilities = calibrate_three_class_probabilities(
                probabilities,
                self.probability_temperature(),
                "CatBoost",
            )?;
            normalize_three_class_probabilities(probabilities, "CatBoost")
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        #[cfg(feature = "catboost")]
        {
            std::fs::create_dir_all(path).with_context(|| {
                format!("create CatBoost artifact directory {}", path.display())
            })?;
            let metadata = tree_runtime_metadata(
                "catboost",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            )?;
            let (model_path, metadata_path) = tree_artifact_paths(path, CATBOOST_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            let runtime_artifact = self.runtime_artifact.clone().unwrap_or_else(|| {
                if self.model_bytes.is_some() {
                    self.build_runtime_artifact(
                        self.resolve_executable().ok().as_deref(),
                        Some(self.effective_task_type()),
                        3,
                        self.feature_columns.len(),
                    )
                } else {
                    self.build_surrogate_runtime_artifact(self.feature_columns.len())
                }
            });
            validate_runtime_artifact(&runtime_artifact, self.feature_columns.len())?;
            let runtime_path = Self::runtime_artifact_path(path);
            write_tree_json_artifact(
                &runtime_path,
                &runtime_artifact,
                "CatBoost runtime artifact",
            )?;
            if let Some(model_bytes) = self.model_bytes.as_ref() {
                atomic_write(&model_path, model_bytes)?;
            } else if self.local_fallback.is_none() {
                bail!("CatBoost model bytes unavailable; train or load before saving");
            }
            self.persist_local_fallback(path)?;
            Ok(())
        }
        #[cfg(not(feature = "catboost"))]
        {
            std::fs::create_dir_all(path).with_context(|| {
                format!(
                    "create CatBoost fallback artifact directory {}",
                    path.display()
                )
            })?;
            let metadata = tree_runtime_metadata(
                "catboost",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            )?;
            let (_, metadata_path) = tree_artifact_paths(path, CATBOOST_MODEL_FILE_NAME);
            write_runtime_metadata(&metadata_path, &metadata)?;
            let runtime_artifact = self.runtime_artifact.clone().unwrap_or_else(|| {
                self.build_surrogate_runtime_artifact(self.feature_columns.len())
            });
            validate_runtime_artifact(&runtime_artifact, self.feature_columns.len())?;
            let runtime_path = Self::runtime_artifact_path(path);
            write_tree_json_artifact(
                &runtime_path,
                &runtime_artifact,
                "CatBoost runtime artifact",
            )?;
            self.persist_local_fallback(path)?;
            Ok(())
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            let (model_path, metadata_path) = tree_artifact_paths(path, CATBOOST_MODEL_FILE_NAME);
            let persisted_runtime_artifact = Self::read_runtime_artifact(path)?;
            self.local_fallback = Self::read_local_fallback(path)?;
            let metadata = Self::resolve_runtime_metadata(
                path,
                &metadata_path,
                persisted_runtime_artifact.as_ref(),
                self.local_fallback.as_ref(),
            )?;
            let metadata_feature_columns = metadata.feature_columns.clone();
            let metadata_training_summary = metadata.training_summary.clone();
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            if let Some(runtime_artifact) = persisted_runtime_artifact.as_ref() {
                validate_runtime_artifact(runtime_artifact, self.feature_columns.len())?;
                if runtime_artifact.feature_count != metadata_feature_columns.len() {
                    bail!(
                        "CatBoost runtime artifact feature mismatch with metadata: runtime={} metadata={}",
                        runtime_artifact.feature_count,
                        metadata_feature_columns.len()
                    );
                }
                self.apply_runtime_artifact(runtime_artifact);
            }
            let native_model_result = if model_path.exists() {
                Some((|| -> Result<(Vec<u8>, catboost::Model)> {
                    let model_bytes = std::fs::read(&model_path).with_context(|| {
                        format!("read CatBoost artifact {}", model_path.display())
                    })?;
                    let model = catboost::Model::load_buffer(&model_bytes).with_context(|| {
                        format!("load CatBoost model from {}", model_path.display())
                    })?;
                    if model.get_dimensions_count() != 3 {
                        bail!(
                            "CatBoost model dimensions mismatch: expected 3 classes, got {}",
                            model.get_dimensions_count()
                        );
                    }
                    if model.get_float_features_count() != self.feature_columns.len() {
                        bail!(
                            "CatBoost feature count mismatch: model expects {}, metadata has {}",
                            model.get_float_features_count(),
                            self.feature_columns.len()
                        );
                    }
                    Ok((model_bytes, model))
                })())
            } else {
                None
            };

            match native_model_result {
                Some(Ok((model_bytes, model))) => {
                    let runtime_artifact = persisted_runtime_artifact.unwrap_or_else(|| {
                        self.build_runtime_artifact(
                            None,
                            Some(self.effective_task_type()),
                            model.get_dimensions_count(),
                            self.feature_columns.len(),
                        )
                    });
                    validate_runtime_artifact(&runtime_artifact, self.feature_columns.len())?;
                    if metadata_training_summary.dataset_rows == 0 {
                        bail!(
                            "CatBoost metadata training summary must record non-zero dataset_rows"
                        );
                    }
                    if metadata_training_summary.dataset_rows
                        != metadata_training_summary.train_rows + metadata_training_summary.val_rows
                    {
                        bail!("CatBoost metadata training summary is inconsistent");
                    }
                    self.apply_runtime_artifact(&runtime_artifact);
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                    }
                    self.model_bytes = Some(model_bytes);
                    self.runtime_artifact = Some(runtime_artifact);
                    self.model = Some(model);
                }
                Some(Err(native_err)) => {
                    self.model_bytes = None;
                    self.runtime_artifact = persisted_runtime_artifact;
                    self.model = None;
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                        tracing::warn!(
                            model = "catboost",
                            path = %path.display(),
                            surrogate_kind = %fallback.surrogate_kind,
                            surrogate_rows = fallback.training_summary.dataset_rows,
                            error = %native_err,
                            "failed to restore native CatBoost model; using local surrogate fallback"
                        );
                    } else {
                        return Err(native_err);
                    }
                }
                None => {
                    self.model_bytes = None;
                    self.runtime_artifact = persisted_runtime_artifact;
                    self.model = None;
                    self.local_fallback = Self::read_local_fallback(path)?;
                    if let Some(fallback) = self.local_fallback.as_ref() {
                        validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
                        tracing::warn!(
                            model = "catboost",
                            path = %path.display(),
                            surrogate_kind = %fallback.surrogate_kind,
                            surrogate_rows = fallback.training_summary.dataset_rows,
                            "CatBoost artifact missing native model; using local surrogate fallback"
                        );
                    } else {
                        bail!(
                            "CatBoost artifact {} is missing both native model and local fallback payload",
                            path.display()
                        );
                    }
                }
            }
            self.gpu_only_disabled = false;
            Ok(())
        }
        #[cfg(not(feature = "catboost"))]
        {
            let (_, metadata_path) = tree_artifact_paths(path, CATBOOST_MODEL_FILE_NAME);
            let persisted_runtime_artifact = Self::read_runtime_artifact(path)?;
            self.local_fallback = Self::read_local_fallback(path)?;
            let metadata = Self::resolve_runtime_metadata(
                path,
                &metadata_path,
                persisted_runtime_artifact.as_ref(),
                self.local_fallback.as_ref(),
            )?;
            self.feature_columns = metadata.feature_columns;
            self.training_summary = Some(metadata.training_summary);
            if let Some(runtime_artifact) = persisted_runtime_artifact.as_ref() {
                validate_runtime_artifact(runtime_artifact, self.feature_columns.len())?;
                self.apply_runtime_artifact(runtime_artifact);
            }
            if let Some(fallback) = self.local_fallback.as_ref() {
                validate_tree_local_fallback_artifact(fallback, &self.feature_columns)?;
            }
            self.model_bytes = None;
            self.runtime_artifact = persisted_runtime_artifact;
            self.model = None;
            self.gpu_only_disabled = false;
            Ok(())
        }
    }
}

impl CatBoostExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`]
    /// adapter so the registry / aggregator can detect column-layout
    /// drift after a retraining session.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        build_tree_runtime_predictions(
            "catboost",
            &probabilities,
            self.model.is_some(),
            "catboost_native",
            self.local_fallback.as_ref(),
            "native_catboost_unavailable",
            "catboost_unknown",
        )
    }
}

#[cfg(all(test, feature = "catboost"))]
mod tests {
    use super::{CatBoostExpert, ExpertModel};
    use crate::base::feature_columns_from_dataframe;
    use crate::runtime::artifacts::TrainingSummaryMetadata;
    use crate::tree_models::common::{
        build_tree_local_fallback_artifact, default_training_summary,
    };
    use polars::df;
    use polars::prelude::*;
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

    #[test]
    fn catboost_loads_fallback_when_native_artifact_is_corrupt() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("catboost-corrupt-artifact");

        let mut expert = CatBoostExpert::new(9);
        let training_summary = default_training_summary(&x);
        expert.feature_columns = feature_columns_from_dataframe(&x);
        expert.training_summary = Some(training_summary.clone());
        expert.local_fallback = Some(
            build_tree_local_fallback_artifact(&x, &y, training_summary)
                .expect("build fallback artifact"),
        );

        expert.save(&artifact_dir).expect("save should succeed");
        std::fs::write(artifact_dir.join("model.cbm"), b"corrupt catboost model")
            .expect("overwrite native model artifact");

        let mut loaded = CatBoostExpert::new(9);
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
    fn catboost_validate_runtime_artifact_rejects_gpu_only_cpu_runtime() {
        let artifact = super::CatBoostRuntimeArtifact {
            executable: "unknown".into(),
            task_type: "CPU".into(),
            device_preference: "gpu".into(),
            gpu_available: false,
            gpu_only: true,
            model_dimensions: 3,
            feature_count: 3,
            classes_count: 3,
            iterations: 100,
            depth: 8,
            learning_rate: 0.05,
            l2_leaf_reg: 3.0,
            probability_temperature: 1.0,
            use_best_model: false,
            thread_count: 4,
            random_seed: 1,
            loss_function: "MultiClass".into(),
            feature_columns: vec![
                "momentum".to_string(),
                "trend".to_string(),
                "volatility".to_string(),
            ],
            training_summary: TrainingSummaryMetadata::new(9, 9, 0),
        };

        let err = super::validate_runtime_artifact(&artifact, 3)
            .expect_err("gpu_only cpu runtime should fail");
        assert!(err.to_string().contains("gpu_only"));
    }

    #[test]
    fn catboost_save_rejects_missing_training_summary() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("catboost-missing-summary");

        let mut expert = CatBoostExpert::new(9);
        expert.feature_columns = feature_columns_from_dataframe(&x);
        expert.training_summary = None;
        expert.local_fallback = Some(
            build_tree_local_fallback_artifact(&x, &y, default_training_summary(&x))
                .expect("build fallback artifact"),
        );

        let err = expert
            .save(&artifact_dir)
            .expect_err("save should fail without training summary");
        assert!(err.to_string().contains("training summary"));
    }

    #[test]
    fn catboost_load_uses_runtime_artifacts_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("catboost-metadata-missing");

        let mut expert = CatBoostExpert::new(17);
        let training_summary = default_training_summary(&x);
        expert.feature_columns = feature_columns_from_dataframe(&x);
        expert.training_summary = Some(training_summary.clone());
        expert.local_fallback = Some(
            build_tree_local_fallback_artifact(&x, &y, training_summary)
                .expect("build fallback artifact"),
        );

        expert.save(&artifact_dir).expect("save should succeed");
        let metadata_path = artifact_dir.join("metadata.json");
        assert!(
            metadata_path.exists(),
            "expected metadata sidecar at {}",
            metadata_path.display()
        );
        std::fs::remove_file(&metadata_path)
            .expect("remove metadata sidecar to trigger reconstruction");

        let mut loaded = CatBoostExpert::new(17);
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct metadata from runtime artifacts");
        let probabilities = loaded
            .predict_proba(&x)
            .expect("prediction should succeed after metadata reconstruction");
        assert_eq!(probabilities.dim(), (x.height(), 3));
    }
}
