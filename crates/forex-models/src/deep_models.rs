use anyhow::{bail, Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction_with_details,
    canonical_three_class_label_mapping, dataframe_to_float32_array,
    feature_columns_from_dataframe, three_class_runtime_confidence, ExpertModel,
};
use crate::burn_models::{
    normalize_burn_device_policy, predict_proba_on_device as burn_predict_proba_on_device,
    resolve_infer_device, resolve_train_device,
    train_model_with_report_with_selection as burn_train_model_with_report_with_selection,
    validate_burn_device_selection, BurnDeviceSelection, BurnKAN, BurnKANConfig, BurnMLP,
    BurnMLPConfig, BurnNBeats, BurnNBeatsConfig, BurnNBeatsx, BurnNBeatsxConfig, BurnPatchTST,
    BurnPatchTSTConfig, BurnTabNet, BurnTabNetConfig, BurnTiDE, BurnTiDEConfig, BurnTiDENf,
    BurnTiDENfConfig, BurnTimesNet, BurnTimesNetConfig, BurnTrainingReport, BurnTransformer,
    BurnTransformerConfig, InferBackend, TrainBackend, TrainConfig,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

const METADATA_FILE_NAME: &str = "metadata.json";
const CONFIG_FILE_NAME: &str = "config.json";
const MODEL_RECORD_BASENAME: &str = "model";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepModelKind {
    Mlp,
    NBeats,
    NBeatsxNf,
    TiDE,
    TiDENf,
    TabNet,
    Kan,
    Transformer,
    PatchTst,
    TimesNet,
}

impl DeepModelKind {
    pub fn model_name(self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::NBeats => "nbeats",
            Self::NBeatsxNf => "nbeatsx_nf",
            Self::TiDE => "tide",
            Self::TiDENf => "tide_nf",
            Self::TabNet => "tabnet",
            Self::Kan => "kan",
            Self::Transformer => "transformer",
            Self::PatchTst => "patchtst",
            Self::TimesNet => "timesnet",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepArtifactConfig {
    kind: DeepModelKind,
    params: HashMap<String, String>,
    #[serde(default)]
    burn_training_report: Option<BurnTrainingReport>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum RuntimeDeepModel {
    Mlp(BurnMLP<InferBackend>),
    NBeats(BurnNBeats<InferBackend>),
    NBeatsxNf(BurnNBeatsx<InferBackend>),
    TiDE(BurnTiDE<InferBackend>),
    TiDENf(BurnTiDENf<InferBackend>),
    TabNet(BurnTabNet<InferBackend>),
    Kan(BurnKAN<InferBackend>),
    Transformer(BurnTransformer<InferBackend>),
    PatchTst(BurnPatchTST<InferBackend>),
    TimesNet(BurnTimesNet<InferBackend>),
}

impl RuntimeDeepModel {
    fn save_to(&self, base_path: &Path) -> Result<()> {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let base_name = base_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("deep-model record base path is missing a file name")?;
        let temp_base_path = base_path.with_file_name(format!("{base_name}_tmp"));
        let target_record_path = base_path.with_extension("mpk");
        let temp_record_path = temp_base_path.with_extension("mpk");

        match self {
            Self::Mlp(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::NBeats(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::NBeatsxNf(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::TiDE(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::TiDENf(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::TabNet(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::Kan(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::Transformer(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::PatchTst(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
            Self::TimesNet(model) => model.clone().save_file(temp_base_path.clone(), &recorder),
        }
        .with_context(|| format!("persist Burn model record to {}", temp_base_path.display()))?;

        if target_record_path.exists() {
            std::fs::remove_file(&target_record_path).with_context(|| {
                format!(
                    "remove previous deep-model record before rotation {}",
                    target_record_path.display()
                )
            })?;
        }
        std::fs::rename(&temp_record_path, &target_record_path).with_context(|| {
            format!(
                "rename deep-model record into {}",
                target_record_path.display()
            )
        })?;
        Ok(())
    }

    fn predict_probabilities(
        &self,
        features: &Array2<f32>,
        batch_size: usize,
        device: &<InferBackend as burn::tensor::backend::Backend>::Device,
    ) -> Result<Array2<f32>> {
        match self {
            Self::Mlp(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::NBeats(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::NBeatsxNf(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::TiDE(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::TiDENf(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::TabNet(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::Kan(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::Transformer(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::PatchTst(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
            Self::TimesNet(model) => {
                burn_predict_proba_on_device::<InferBackend, _>(model, features, batch_size, device)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BurnDeepExpert {
    kind: DeepModelKind,
    seed: u64,
    params: HashMap<String, String>,
    model: Option<RuntimeDeepModel>,
    feature_columns: Vec<String>,
    training_summary: Option<TrainingSummaryMetadata>,
    burn_training_report: Option<BurnTrainingReport>,
    persisted_runtime_selection: Option<BurnDeviceSelection>,
    host_runtime_selection: Option<BurnDeviceSelection>,
}

impl BurnDeepExpert {
    pub fn new(kind: DeepModelKind, seed: u64, params: Option<HashMap<String, String>>) -> Self {
        Self {
            kind,
            seed,
            params: params.unwrap_or_default(),
            model: None,
            feature_columns: Vec::new(),
            training_summary: None,
            burn_training_report: None,
            persisted_runtime_selection: None,
            host_runtime_selection: None,
        }
    }

    pub fn model_name(&self) -> &'static str {
        self.kind.model_name()
    }

    fn train_config(&self) -> TrainConfig {
        TrainConfig {
            lr: self.float_param("lr", 1e-3),
            batch_size: self.usize_param("batch_size", 64),
            max_epochs: self.usize_param("max_epochs", 100),
            patience: self.usize_param("patience", 8),
            n_classes: 3,
            seed: self.u64_param("seed", self.seed),
        }
    }

    fn metadata(&self) -> Result<RuntimeArtifactMetadata> {
        let training_summary = self.training_summary.clone().with_context(|| {
            format!(
                "{} model is missing training summary metadata; fit or load before saving",
                self.model_name()
            )
        })?;

        Self::validate_training_summary(&training_summary)?;

        if self.feature_columns.is_empty() {
            bail!(
                "{} model is missing feature columns; fit or load before saving",
                self.model_name()
            );
        }

        Ok(build_runtime_artifact_metadata(
            self.model_name(),
            ModelFamily::Deep,
            CapabilityState::Implemented,
            self.feature_columns.clone(),
            canonical_three_class_label_mapping(),
            training_summary,
        ))
    }

    fn validate_runtime_params(params: &HashMap<String, String>) -> Result<()> {
        let runtime_keys = [
            "requested_device_policy",
            "effective_device_policy",
            "execution_backend",
        ];
        let present = runtime_keys
            .iter()
            .filter(|key| params.get(**key).is_some())
            .count();
        if present != 0 && present != runtime_keys.len() {
            bail!(
                "deep-model runtime params must persist requested_device_policy, effective_device_policy, and execution_backend together"
            );
        }
        for key in runtime_keys {
            if let Some(value) = params.get(key) {
                if value.trim().is_empty() {
                    bail!("deep-model runtime param `{key}` may not be blank");
                }
            }
        }
        for key in ["requested_device_policy", "effective_device_policy"] {
            if let Some(value) = params.get(key) {
                let normalized = normalize_burn_device_policy(value);
                if !Self::is_supported_device_policy(&normalized) {
                    bail!(
                        "deep-model runtime param `{key}` uses unsupported device policy `{}`",
                        normalized
                    );
                }
            }
        }
        if let Some(value) = params.get("execution_backend") {
            if !Self::is_supported_execution_backend(value) {
                bail!(
                    "deep-model runtime param `execution_backend` uses unsupported backend `{}`",
                    value
                );
            }
        }
        if let Some(device) = params.get("device") {
            if device.trim().is_empty() {
                bail!("deep-model runtime param `device` may not be blank");
            }
        }
        if let (Some(device), Some(requested_runtime)) =
            (params.get("device"), params.get("requested_device_policy"))
        {
            let normalized_device = normalize_burn_device_policy(device);
            let normalized_requested = normalize_burn_device_policy(requested_runtime);
            if normalized_device != normalized_requested {
                bail!(
                    "deep-model legacy `device` param `{}` conflicts with persisted requested_device_policy `{}`",
                    normalized_device,
                    normalized_requested
                );
            }
        }
        if let (Some(requested_policy), Some(effective_policy), Some(execution_backend)) = (
            params.get("requested_device_policy"),
            params.get("effective_device_policy"),
            params.get("execution_backend"),
        ) {
            validate_burn_device_selection(&BurnDeviceSelection {
                requested_policy: requested_policy.clone(),
                effective_policy: effective_policy.clone(),
                execution_backend: execution_backend.clone(),
            })
            .context("deep-model runtime params are internally inconsistent")?;
        }
        Ok(())
    }

    fn is_supported_device_policy(normalized: &str) -> bool {
        normalized == "auto"
            || normalized == "cpu"
            || normalized.starts_with("cuda:")
            || normalized.starts_with("gpu:")
    }

    fn is_supported_execution_backend(backend: &str) -> bool {
        matches!(
            backend.trim(),
            "ndarray_cpu" | "wgpu_cpu" | "wgpu_default" | "wgpu_discrete_gpu"
        )
    }

    fn runtime_selection_from_report(report: &BurnTrainingReport) -> BurnDeviceSelection {
        BurnDeviceSelection {
            requested_policy: report.requested_device_policy.clone(),
            effective_policy: report.effective_device_policy.clone(),
            execution_backend: report.execution_backend.clone(),
        }
    }

    fn validate_burn_training_report(
        &self,
        summary: &TrainingSummaryMetadata,
        runtime_selection: Option<&BurnDeviceSelection>,
        report: Option<&BurnTrainingReport>,
    ) -> Result<()> {
        let report = report.with_context(|| {
            format!(
                "{} model is missing Burn training report metadata",
                self.model_name()
            )
        })?;
        if report.dataset_rows != summary.dataset_rows
            || report.train_rows != summary.train_rows
            || report.val_rows != summary.val_rows
        {
            bail!(
                "{} Burn training report rows do not match persisted training summary",
                self.model_name()
            );
        }
        for (field_name, value) in [
            (
                "requested_device_policy",
                report.requested_device_policy.as_str(),
            ),
            (
                "effective_device_policy",
                report.effective_device_policy.as_str(),
            ),
            ("execution_backend", report.execution_backend.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!(
                    "{} Burn training report `{field_name}` may not be blank",
                    self.model_name()
                );
            }
        }
        let report_runtime = Self::runtime_selection_from_report(report);
        validate_burn_device_selection(&report_runtime).with_context(|| {
            format!(
                "{} Burn training report runtime provenance is internally inconsistent",
                self.model_name()
            )
        })?;
        if let Some(selection) = runtime_selection {
            if report_runtime.requested_policy != selection.requested_policy
                || report_runtime.effective_policy != selection.effective_policy
                || report_runtime.execution_backend != selection.execution_backend
            {
                bail!(
                    "{} Burn training report runtime provenance does not match persisted runtime selection",
                    self.model_name()
                );
            }
        }
        Ok(())
    }

    fn validate_model_params(&self) -> Result<()> {
        for key in [
            "hidden_dim",
            "n_layers",
            "n_blocks",
            "n_steps",
            "n_heads",
            "dim_ff",
            "patch_size",
            "n_periods",
            "batch_size",
            "max_epochs",
            "patience",
        ] {
            if let Some(value) = self.params.get(key) {
                let parsed = value.trim().parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("deep-model param `{key}` must parse as a positive integer")
                })?;
                if parsed == 0 {
                    bail!("deep-model param `{key}` must be greater than zero");
                }
            }
        }
        if let Some(value) = self.params.get("seed") {
            value
                .trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("deep-model param `seed` must parse as u64"))?;
        }
        if let Some(value) = self.params.get("lr") {
            let parsed = value.trim().parse::<f64>().map_err(|_| {
                anyhow::anyhow!("deep-model param `lr` must parse as a finite positive float")
            })?;
            if !parsed.is_finite() || parsed <= 0.0 {
                bail!("deep-model param `lr` must be finite and positive");
            }
        }
        if let Some(value) = self.params.get("dropout") {
            let parsed = value.trim().parse::<f64>().map_err(|_| {
                anyhow::anyhow!("deep-model param `dropout` must parse as a finite float")
            })?;
            if !parsed.is_finite() || !(0.0..1.0).contains(&parsed) {
                bail!("deep-model param `dropout` must be finite and inside [0, 1)");
            }
        }
        Ok(())
    }

    fn artifact_config(&self) -> Result<DeepArtifactConfig> {
        Self::validate_runtime_params(&self.params)?;
        self.validate_model_params()?;
        let summary = self.training_summary.as_ref().with_context(|| {
            format!(
                "{} model is missing training summary metadata",
                self.model_name()
            )
        })?;
        let runtime_selection = Self::runtime_selection_from_params(&self.params)?;
        self.validate_burn_training_report(
            summary,
            runtime_selection.as_ref(),
            self.burn_training_report.as_ref(),
        )?;
        Ok(DeepArtifactConfig {
            kind: self.kind,
            params: self.params.clone(),
            burn_training_report: self.burn_training_report.clone(),
        })
    }

    fn batch_size(&self) -> usize {
        self.usize_param("batch_size", 64)
    }

    fn string_param(&self, key: &str, default: &str) -> String {
        self.params
            .get(key)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_string()
    }

    fn usize_param(&self, key: &str, default: usize) -> usize {
        self.params
            .get(key)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }

    fn u64_param(&self, key: &str, default: u64) -> u64 {
        self.params
            .get(key)
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(default)
    }

    fn float_param(&self, key: &str, default: f64) -> f64 {
        self.params
            .get(key)
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(default)
    }

    fn compatible_head_count(&self, hidden_dim: usize, default: usize) -> usize {
        let requested = self.usize_param("n_heads", default).max(1);
        if hidden_dim.is_multiple_of(requested) {
            return requested;
        }

        (1..=requested)
            .rev()
            .find(|candidate| hidden_dim.is_multiple_of(*candidate))
            .unwrap_or(1)
    }

    fn mlp_config(&self, input_dim: usize) -> BurnMLPConfig {
        BurnMLPConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 256))
            .with_n_layers(self.usize_param("n_layers", 3))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.1))
    }

    fn nbeats_config(&self, input_dim: usize) -> BurnNBeatsConfig {
        BurnNBeatsConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 64))
            .with_n_blocks(self.usize_param("n_blocks", 3))
            .with_n_classes(3)
    }

    fn nbeatsx_nf_config(&self, input_dim: usize) -> BurnNBeatsxConfig {
        BurnNBeatsxConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 96))
            .with_n_blocks(self.usize_param("n_blocks", 4))
            .with_n_classes(3)
    }

    fn tide_config(&self, input_dim: usize) -> BurnTiDEConfig {
        BurnTiDEConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 128))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.1))
    }

    fn tide_nf_config(&self, input_dim: usize) -> BurnTiDENfConfig {
        BurnTiDENfConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 160))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.05))
    }

    fn tabnet_config(&self, input_dim: usize) -> BurnTabNetConfig {
        BurnTabNetConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 64))
            .with_n_steps(self.usize_param("n_steps", 3))
            .with_n_classes(3)
    }

    fn kan_config(&self, input_dim: usize) -> BurnKANConfig {
        BurnKANConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 32))
            .with_n_classes(3)
    }

    fn transformer_config(&self, input_dim: usize) -> BurnTransformerConfig {
        let hidden_dim = self.usize_param("hidden_dim", 128);
        BurnTransformerConfig::new(input_dim)
            .with_hidden_dim(hidden_dim)
            .with_n_heads(self.compatible_head_count(hidden_dim, 8))
            .with_n_layers(self.usize_param("n_layers", 4))
            .with_dim_ff(self.usize_param("dim_ff", 512))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.1))
    }

    fn patchtst_config(&self, input_dim: usize) -> BurnPatchTSTConfig {
        let hidden_dim = self.usize_param("hidden_dim", 192);
        BurnPatchTSTConfig::new(input_dim)
            .with_hidden_dim(hidden_dim)
            .with_patch_size(self.usize_param("patch_size", 8))
            .with_n_heads(self.compatible_head_count(hidden_dim, 6))
            .with_n_layers(self.usize_param("n_layers", 3))
            .with_dim_ff(self.usize_param("dim_ff", 384))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.10))
    }

    fn timesnet_config(&self, input_dim: usize) -> BurnTimesNetConfig {
        BurnTimesNetConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 192))
            .with_n_periods(self.usize_param("n_periods", 4))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.05))
    }

    fn runtime_selection_from_params(
        params: &HashMap<String, String>,
    ) -> Result<Option<BurnDeviceSelection>> {
        let requested = params.get("requested_device_policy").cloned();
        let effective = params.get("effective_device_policy").cloned();
        let backend = params.get("execution_backend").cloned();
        match (requested, effective, backend) {
            (Some(requested_policy), Some(effective_policy), Some(execution_backend)) => {
                Ok(Some(BurnDeviceSelection {
                    requested_policy,
                    effective_policy,
                    execution_backend,
                }))
            }
            (None, None, None) => Ok(None),
            _ => {
                bail!(
                    "deep-model runtime params must persist requested_device_policy, effective_device_policy, and execution_backend together"
                )
            }
        }
    }

    fn configured_requested_device_policy(&self) -> String {
        self.persisted_runtime_selection
            .clone()
            .or_else(|| {
                Self::runtime_selection_from_params(&self.params)
                    .ok()
                    .flatten()
            })
            .as_ref()
            .map(|selection| selection.requested_policy.clone())
            .unwrap_or_else(|| self.string_param("device", "auto"))
    }

    fn resolve_runtime_infer_device(
        &self,
    ) -> (
        <InferBackend as burn::tensor::backend::Backend>::Device,
        BurnDeviceSelection,
    ) {
        let requested_device = self.configured_requested_device_policy();
        resolve_infer_device(&requested_device)
    }

    fn init_runtime_model(&self, input_dim: usize) -> RuntimeDeepModel {
        let (device, _) = self.resolve_runtime_infer_device();
        match self.kind {
            DeepModelKind::Mlp => {
                RuntimeDeepModel::Mlp(self.mlp_config(input_dim).init::<InferBackend>(&device))
            }
            DeepModelKind::NBeats => RuntimeDeepModel::NBeats(
                self.nbeats_config(input_dim).init::<InferBackend>(&device),
            ),
            DeepModelKind::NBeatsxNf => RuntimeDeepModel::NBeatsxNf(
                self.nbeatsx_nf_config(input_dim)
                    .init::<InferBackend>(&device),
            ),
            DeepModelKind::TiDE => {
                RuntimeDeepModel::TiDE(self.tide_config(input_dim).init::<InferBackend>(&device))
            }
            DeepModelKind::TiDENf => RuntimeDeepModel::TiDENf(
                self.tide_nf_config(input_dim).init::<InferBackend>(&device),
            ),
            DeepModelKind::TabNet => RuntimeDeepModel::TabNet(
                self.tabnet_config(input_dim).init::<InferBackend>(&device),
            ),
            DeepModelKind::Kan => {
                RuntimeDeepModel::Kan(self.kan_config(input_dim).init::<InferBackend>(&device))
            }
            DeepModelKind::Transformer => RuntimeDeepModel::Transformer(
                self.transformer_config(input_dim)
                    .init::<InferBackend>(&device),
            ),
            DeepModelKind::PatchTst => RuntimeDeepModel::PatchTst(
                self.patchtst_config(input_dim)
                    .init::<InferBackend>(&device),
            ),
            DeepModelKind::TimesNet => RuntimeDeepModel::TimesNet(
                self.timesnet_config(input_dim)
                    .init::<InferBackend>(&device),
            ),
        }
    }

    fn training_summary_from_report(report: &BurnTrainingReport) -> TrainingSummaryMetadata {
        TrainingSummaryMetadata::new(report.dataset_rows, report.train_rows, report.val_rows)
    }

    fn train_runtime_model(
        &self,
        input_dim: usize,
        features: &Array2<f32>,
        labels: &[i32],
    ) -> Result<(
        RuntimeDeepModel,
        TrainingSummaryMetadata,
        BurnDeviceSelection,
        BurnTrainingReport,
    )> {
        let train_config = self.train_config();
        let requested_device = self.configured_requested_device_policy();
        let (device, device_selection) = resolve_train_device(&requested_device);
        match self.kind {
            DeepModelKind::Mlp => {
                let model = self.mlp_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::Mlp(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::NBeats => {
                let model = self.nbeats_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::NBeats(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::NBeatsxNf => {
                let model = self
                    .nbeatsx_nf_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::NBeatsxNf(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::TiDE => {
                let model = self.tide_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::TiDE(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::TiDENf => {
                let model = self.tide_nf_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::TiDENf(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::TabNet => {
                let model = self.tabnet_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::TabNet(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::Kan => {
                let model = self.kan_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::Kan(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::Transformer => {
                let model = self
                    .transformer_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::Transformer(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::PatchTst => {
                let model = self
                    .patchtst_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::PatchTst(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
            DeepModelKind::TimesNet => {
                let model = self
                    .timesnet_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) =
                    burn_train_model_with_report_with_selection::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                    )?;
                Ok((
                    RuntimeDeepModel::TimesNet(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                    report,
                ))
            }
        }
    }

    fn labels_from_series(y: &Series) -> Result<Vec<i32>> {
        let labels = y
            .cast(&DataType::Int32)
            .context("cast deep-model labels to Int32")?;
        let values = labels.i32().context("access deep-model labels as Int32")?;

        values
            .into_iter()
            .map(|value| match value {
                Some(label @ -1..=1) => Ok(label),
                Some(other) => {
                    bail!("unsupported deep-model label: {other}; expected one of -1, 0, 1")
                }
                None => bail!("deep-model labels may not contain nulls"),
            })
            .collect()
    }

    fn model_record_path(path: &Path) -> PathBuf {
        path.join(MODEL_RECORD_BASENAME)
    }

    fn metadata_path(path: &Path) -> PathBuf {
        path.join(METADATA_FILE_NAME)
    }

    fn config_path(path: &Path) -> PathBuf {
        path.join(CONFIG_FILE_NAME)
    }

    fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create artifact directory {}", parent.display()))?;
        }

        let temp_path = path.with_extension("tmp");
        let payload = serde_json::to_vec_pretty(value)
            .with_context(|| format!("serialize {}", path.display()))?;
        std::fs::write(&temp_path, payload)
            .with_context(|| format!("write temporary artifact {}", temp_path.display()))?;
        if path.exists() {
            std::fs::remove_file(path)
                .with_context(|| format!("remove previous artifact {}", path.display()))?;
        }
        std::fs::rename(&temp_path, path)
            .with_context(|| format!("rename artifact into {}", path.display()))?;
        Ok(())
    }

    fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
        let payload = std::fs::read(path)
            .with_context(|| format!("read deep-model artifact {}", path.display()))?;
        serde_json::from_slice(&payload)
            .with_context(|| format!("deserialize deep-model artifact {}", path.display()))
    }

    fn staged_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("tmp_artifact")
    }

    fn backup_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("bak_artifact")
    }

    fn cleanup_artifact_dir(path: &Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("remove staged deep-model artifact {}", path.display()))?;
        }
        Ok(())
    }

    fn replace_artifact_directory(staged_path: &Path, target_path: &Path) -> Result<()> {
        let backup_path = Self::backup_artifact_dir(target_path);
        Self::cleanup_artifact_dir(&backup_path)?;
        if target_path.exists() {
            std::fs::rename(target_path, &backup_path).with_context(|| {
                format!(
                    "stage previous deep-model artifact into backup {}",
                    backup_path.display()
                )
            })?;
        }
        if let Err(error) = std::fs::rename(staged_path, target_path) {
            if backup_path.exists() {
                let _ = std::fs::rename(&backup_path, target_path);
            }
            bail!(
                "rename staged deep-model artifact into {} failed: {}",
                target_path.display(),
                error
            );
        }
        Self::cleanup_artifact_dir(&backup_path)?;
        Ok(())
    }

    fn validate_training_summary(summary: &TrainingSummaryMetadata) -> Result<()> {
        if summary.dataset_rows != summary.train_rows + summary.val_rows {
            bail!(
                "deep-model training summary is inconsistent: dataset_rows={} but train_rows + val_rows = {}",
                summary.dataset_rows,
                summary.train_rows + summary.val_rows
            );
        }

        Ok(())
    }

    fn validate_loaded_metadata(
        metadata: &RuntimeArtifactMetadata,
        expected_model_name: &str,
    ) -> Result<()> {
        if metadata.model_name != expected_model_name {
            bail!(
                "deep artifact model mismatch: expected {}, got {}",
                expected_model_name,
                metadata.model_name
            );
        }

        if metadata.family != ModelFamily::Deep {
            bail!(
                "deep artifact family mismatch: expected {:?}, got {:?}",
                ModelFamily::Deep,
                metadata.family
            );
        }

        if metadata.state != CapabilityState::Implemented {
            bail!(
                "deep artifact state mismatch: expected {:?}, got {:?}",
                CapabilityState::Implemented,
                metadata.state
            );
        }

        if metadata.label_mapping != canonical_three_class_label_mapping() {
            bail!("deep artifact label mapping mismatch");
        }

        if metadata.feature_columns.is_empty() {
            bail!("deep artifact metadata must contain at least one feature column");
        }

        Self::validate_training_summary(&metadata.training_summary)
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        if self.feature_columns.is_empty() {
            bail!(
                "{} runtime state is missing persisted feature columns",
                self.model_name()
            );
        }
        let summary = self.training_summary.as_ref().with_context(|| {
            format!(
                "{} runtime state is missing training summary metadata",
                self.model_name()
            )
        })?;
        Self::validate_training_summary(summary)?;
        Self::validate_runtime_params(&self.params)?;
        self.validate_model_params()?;
        let runtime_selection = Self::runtime_selection_from_params(&self.params)?;
        self.validate_burn_training_report(
            summary,
            runtime_selection.as_ref(),
            self.burn_training_report.as_ref(),
        )?;
        Ok(())
    }

    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let persisted_runtime_selection = self.persisted_runtime_selection.clone().or_else(|| {
            Self::runtime_selection_from_params(&self.params)
                .ok()
                .flatten()
        });
        let live_host_runtime_selection = if self.model.is_some()
            && !self.feature_columns.is_empty()
            && self.training_summary.is_some()
        {
            Some(self.resolve_runtime_infer_device().1)
        } else {
            self.host_runtime_selection.clone()
        };
        let execution_backend = live_host_runtime_selection
            .as_ref()
            .map(|selection| selection.execution_backend.clone())
            .or_else(|| {
                self.host_runtime_selection
                    .as_ref()
                    .map(|selection| selection.execution_backend.clone())
            })
            .or_else(|| {
                persisted_runtime_selection
                    .as_ref()
                    .map(|selection| selection.execution_backend.clone())
            });
        let mut degraded = Vec::new();
        let persisted = persisted_runtime_selection.as_ref();
        let host = live_host_runtime_selection.as_ref();

        if persisted.is_none() {
            degraded.push("deep_runtime_device_metadata_missing".to_string());
        }
        if self.burn_training_report.is_none() {
            degraded.push("deep_runtime_training_report_missing".to_string());
        }
        if self.model.is_none() {
            degraded.push("deep_runtime_model_missing".to_string());
        }
        if let (Some(cached_host), Some(live_host)) = (
            self.host_runtime_selection.as_ref(),
            live_host_runtime_selection.as_ref(),
        ) {
            if cached_host.requested_policy != live_host.requested_policy
                || cached_host.effective_policy != live_host.effective_policy
                || cached_host.execution_backend != live_host.execution_backend
            {
                degraded.push("deep_runtime_host_cache_stale".to_string());
            }
        }
        if let Some(persisted) = persisted {
            if persisted.requested_policy != persisted.effective_policy {
                degraded.push("deep_requested_device_unavailable".to_string());
            }
        }
        if let (Some(report), Some(persisted)) = (self.burn_training_report.as_ref(), persisted) {
            let report_runtime = Self::runtime_selection_from_report(report);
            if report_runtime.requested_policy != persisted.requested_policy
                || report_runtime.effective_policy != persisted.effective_policy
                || report_runtime.execution_backend != persisted.execution_backend
            {
                degraded.push("deep_runtime_report_metadata_drift".to_string());
            }
        }
        if let (Some(persisted), Some(host)) = (persisted, host) {
            if persisted.effective_policy != host.effective_policy {
                degraded.push("deep_runtime_device_re_resolved".to_string());
            }
            if persisted.execution_backend != host.execution_backend {
                degraded.push("deep_runtime_backend_re_resolved".to_string());
            }
        }

        (
            execution_backend,
            if degraded.is_empty() {
                None
            } else {
                Some(degraded.join("; "))
            },
        )
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                self.model_name(),
                ModelFamily::Deep,
                CapabilityState::Implemented,
                row_values,
                Some(confidence),
                Some(abstain_recommended),
                execution_backend.clone(),
                degraded_reason.clone(),
            )?);
        }
        Ok(predictions)
    }
}

impl ExpertModel for BurnDeepExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let features = dataframe_to_float32_array(x)
            .with_context(|| format!("build {} feature matrix", self.model_name()))?;
        let labels = Self::labels_from_series(y)?;
        if features.nrows() != labels.len() {
            bail!(
                "{} training feature/label mismatch: {} rows vs {} labels",
                self.model_name(),
                features.nrows(),
                labels.len()
            );
        }
        let input_dim = features.ncols();

        self.feature_columns = feature_columns_from_dataframe(x);
        self.validate_model_params()?;
        let (model, summary, device_selection, burn_training_report) =
            self.train_runtime_model(input_dim, &features, &labels)?;
        self.training_summary = Some(summary);
        self.burn_training_report = Some(burn_training_report);
        self.params.insert(
            "requested_device_policy".to_string(),
            device_selection.requested_policy,
        );
        self.params.insert(
            "effective_device_policy".to_string(),
            device_selection.effective_policy,
        );
        self.params.insert(
            "execution_backend".to_string(),
            device_selection.execution_backend,
        );
        self.persisted_runtime_selection = Self::runtime_selection_from_params(&self.params)?;
        self.host_runtime_selection = self.persisted_runtime_selection.clone();
        self.model = Some(model);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        self.ensure_runtime_state_ready()?;
        let model = self
            .model
            .as_ref()
            .with_context(|| format!("{} model is not trained or loaded", self.model_name()))?;

        let actual_columns = feature_columns_from_dataframe(x);
        if !self.feature_columns.is_empty() && self.feature_columns != actual_columns {
            bail!(
                "feature column mismatch for persisted deep model; expected {:?}, got {:?}",
                self.feature_columns,
                actual_columns
            );
        }

        let features = dataframe_to_float32_array(x)
            .with_context(|| format!("build {} inference matrix", self.model_name()))?;
        let (device, _) = self.resolve_runtime_infer_device();
        let probabilities = model.predict_probabilities(&features, self.batch_size(), &device)?;
        if probabilities.ncols() != 3 {
            bail!(
                "{} should output 3 probability columns, got {}",
                self.model_name(),
                probabilities.ncols()
            );
        }
        Ok(probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        let model = self
            .model
            .as_ref()
            .with_context(|| format!("{} model is not trained or loaded", self.model_name()))?;
        let metadata = self.metadata()?;
        let config = self.artifact_config()?;
        let staged_path = Self::staged_artifact_dir(path);
        Self::cleanup_artifact_dir(&staged_path)?;
        std::fs::create_dir_all(&staged_path).with_context(|| {
            format!(
                "create staged deep-model directory {}",
                staged_path.display()
            )
        })?;
        if let Err(error) = (|| -> Result<()> {
            model.save_to(&Self::model_record_path(&staged_path))?;
            Self::write_json(&Self::metadata_path(&staged_path), &metadata)?;
            Self::write_json(&Self::config_path(&staged_path), &config)?;
            Ok(())
        })() {
            let _ = Self::cleanup_artifact_dir(&staged_path);
            return Err(error);
        }
        Self::replace_artifact_directory(&staged_path, path)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = Self::read_json(&Self::metadata_path(path))?;
        let config: DeepArtifactConfig = Self::read_json(&Self::config_path(path))?;
        if config.kind != self.kind {
            bail!(
                "deep artifact kind mismatch: expected {}, got {}",
                self.model_name(),
                config.kind.model_name()
            );
        }

        Self::validate_loaded_metadata(&metadata, self.model_name())?;
        Self::validate_runtime_params(&config.params)?;
        let persisted_runtime_selection = Self::runtime_selection_from_params(&config.params)?;
        self.validate_burn_training_report(
            &metadata.training_summary,
            persisted_runtime_selection.as_ref(),
            config.burn_training_report.as_ref(),
        )?;
        let next_params = config.params;
        let next_feature_columns = metadata.feature_columns;
        let next_training_summary = Some(metadata.training_summary);
        let mut next_state = self.clone();
        next_state.params = next_params.clone();
        next_state.burn_training_report = config.burn_training_report;
        next_state.persisted_runtime_selection = persisted_runtime_selection;
        next_state.validate_model_params()?;
        let next_model = next_state.init_runtime_model(next_feature_columns.len());

        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let base_path = Self::model_record_path(path);
        let (device, host_runtime_selection) = next_state.resolve_runtime_infer_device();
        if let Some(persisted_runtime_selection) = next_state.persisted_runtime_selection.as_ref() {
            if persisted_runtime_selection.requested_policy
                != host_runtime_selection.requested_policy
                || persisted_runtime_selection.effective_policy
                    != host_runtime_selection.effective_policy
                || persisted_runtime_selection.execution_backend
                    != host_runtime_selection.execution_backend
            {
                bail!(
                    "{} runtime identity drift between persisted {:?} and host {:?}",
                    self.model_name(),
                    persisted_runtime_selection,
                    host_runtime_selection
                );
            }
        }
        let loaded = match next_model {
            RuntimeDeepModel::Mlp(model) => RuntimeDeepModel::Mlp(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::NBeats(model) => RuntimeDeepModel::NBeats(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::NBeatsxNf(model) => RuntimeDeepModel::NBeatsxNf(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::TiDE(model) => RuntimeDeepModel::TiDE(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::TiDENf(model) => RuntimeDeepModel::TiDENf(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::TabNet(model) => RuntimeDeepModel::TabNet(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::Kan(model) => RuntimeDeepModel::Kan(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::Transformer(model) => RuntimeDeepModel::Transformer(
                model
                    .load_file(base_path, &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::PatchTst(model) => RuntimeDeepModel::PatchTst(
                model
                    .load_file(base_path.clone(), &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
            RuntimeDeepModel::TimesNet(model) => RuntimeDeepModel::TimesNet(
                model
                    .load_file(base_path, &recorder, &device)
                    .with_context(|| format!("load {} Burn record", self.model_name()))?,
            ),
        };
        next_state.params = next_params;
        next_state.host_runtime_selection = Some(host_runtime_selection);
        next_state.feature_columns = next_feature_columns;
        next_state.training_summary = next_training_summary;
        next_state.model = Some(loaded);
        *self = next_state;
        Ok(())
    }
}

macro_rules! define_deep_expert {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: BurnDeepExpert,
        }

        impl $name {
            pub fn new(seed: u64, params: Option<HashMap<String, String>>) -> Self {
                Self {
                    inner: BurnDeepExpert::new($kind, seed, params),
                }
            }

            pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
                self.inner.predict_runtime(x)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new(42, None)
            }
        }

        impl ExpertModel for $name {
            fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
                self.inner.fit(x, y)
            }

            fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
                self.inner.predict_proba(x)
            }

            fn save(&self, path: &Path) -> Result<()> {
                self.inner.save(path)
            }

            fn load(&mut self, path: &Path) -> Result<()> {
                self.inner.load(path)
            }
        }
    };
}

define_deep_expert!(MLPExpert, DeepModelKind::Mlp);
define_deep_expert!(NBeatsExpert, DeepModelKind::NBeats);
define_deep_expert!(NBeatsxNfExpert, DeepModelKind::NBeatsxNf);
define_deep_expert!(TiDEExpert, DeepModelKind::TiDE);
define_deep_expert!(TiDENfExpert, DeepModelKind::TiDENf);
define_deep_expert!(TabNetExpert, DeepModelKind::TabNet);
define_deep_expert!(KANExpert, DeepModelKind::Kan);
define_deep_expert!(TransformerExpert, DeepModelKind::Transformer);
define_deep_expert!(PatchTSTExpert, DeepModelKind::PatchTst);
define_deep_expert!(TimesNetExpert, DeepModelKind::TimesNet);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_requires_training_summary() {
        let expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);

        let err = expert
            .metadata()
            .expect_err("missing training summary must fail");
        assert!(err
            .to_string()
            .contains("missing training summary metadata"));
    }

    #[test]
    fn metadata_uses_training_summary_and_feature_columns() -> Result<()> {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));

        let metadata = expert.metadata()?;
        assert_eq!(metadata.model_name, "mlp");
        assert_eq!(metadata.feature_columns, vec!["rsi", "atr"]);
        assert_eq!(metadata.training_summary.dataset_rows, 100);
        assert_eq!(metadata.training_summary.train_rows, 80);
        assert_eq!(metadata.training_summary.val_rows, 20);
        Ok(())
    }

    #[test]
    fn validate_loaded_metadata_rejects_inconsistent_training_summary() {
        let metadata = RuntimeArtifactMetadata::new(
            "mlp",
            ModelFamily::Deep,
            CapabilityState::Implemented,
            vec!["rsi".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 7, 2),
        );

        let err = BurnDeepExpert::validate_loaded_metadata(&metadata, "mlp")
            .expect_err("inconsistent training summary must fail");
        assert!(err.to_string().contains("training summary is inconsistent"));
    }

    #[test]
    fn validate_runtime_params_rejects_partial_runtime_triplet() {
        let params = HashMap::from([
            ("requested_device_policy".to_string(), "cpu".to_string()),
            ("execution_backend".to_string(), "ndarray_cpu".to_string()),
        ]);

        let err = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("partial runtime triplet should fail");
        assert!(err.to_string().contains("persist"));
    }

    #[test]
    fn validate_runtime_params_rejects_conflicting_legacy_device_param() {
        let params = HashMap::from([
            ("device".to_string(), "cuda:0".to_string()),
            ("requested_device_policy".to_string(), "cpu".to_string()),
            ("effective_device_policy".to_string(), "cpu".to_string()),
            ("execution_backend".to_string(), "ndarray_cpu".to_string()),
        ]);

        let err = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("conflicting legacy device param should fail");
        assert!(err.to_string().contains("conflicts"));
    }

    #[test]
    fn validate_runtime_params_rejects_unknown_execution_backend() {
        let params = HashMap::from([
            ("requested_device_policy".to_string(), "cpu".to_string()),
            ("effective_device_policy".to_string(), "cpu".to_string()),
            ("execution_backend".to_string(), "metal_gpu".to_string()),
        ]);

        let err = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("unknown execution backend should fail");
        assert!(err.to_string().contains("unsupported backend"));
    }

    #[test]
    fn validate_runtime_params_rejects_internally_incoherent_runtime_triplet() {
        let params = HashMap::from([
            ("requested_device_policy".to_string(), "cpu".to_string()),
            ("effective_device_policy".to_string(), "cpu".to_string()),
            (
                "execution_backend".to_string(),
                "wgpu_discrete_gpu".to_string(),
            ),
        ]);

        let err = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("incoherent runtime triplet should fail");
        assert!(err
            .to_string()
            .contains("runtime params are internally inconsistent"));
    }

    #[test]
    fn ensure_runtime_state_ready_rejects_invalid_model_params() {
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            7,
            Some(HashMap::from([
                ("requested_device_policy".to_string(), "cpu".to_string()),
                ("effective_device_policy".to_string(), "cpu".to_string()),
                ("execution_backend".to_string(), "ndarray_cpu".to_string()),
                ("dropout".to_string(), "1.2".to_string()),
            ])),
        );
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));

        let err = expert
            .ensure_runtime_state_ready()
            .expect_err("invalid dropout must fail");
        assert!(err.to_string().contains("dropout"));
    }

    #[test]
    fn ensure_runtime_state_ready_requires_runtime_device_metadata() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));

        let err = expert
            .ensure_runtime_state_ready()
            .expect_err("missing runtime device metadata should fail");
        assert!(err.to_string().contains("runtime params"));
    }

    #[test]
    fn ensure_runtime_state_ready_requires_burn_training_report() {
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            7,
            Some(HashMap::from([
                ("requested_device_policy".to_string(), "cpu".to_string()),
                ("effective_device_policy".to_string(), "cpu".to_string()),
                ("execution_backend".to_string(), "ndarray_cpu".to_string()),
            ])),
        );
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));

        let err = expert
            .ensure_runtime_state_ready()
            .expect_err("missing Burn training report should fail");
        assert!(err.to_string().contains("Burn training report"));
    }

    #[test]
    fn runtime_details_mark_requested_device_drift_as_degraded() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert
            .params
            .insert("execution_backend".to_string(), "wgpu".to_string());
        expert
            .params
            .insert("requested_device_policy".to_string(), "cuda:0".to_string());
        expert
            .params
            .insert("effective_device_policy".to_string(), "cpu".to_string());

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("wgpu"));
        assert!(degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("deep_requested_device_unavailable"));
        assert!(degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("deep_runtime_model_missing"));
    }

    #[test]
    fn runtime_details_mark_missing_burn_training_report() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.persisted_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: "ndarray_cpu".to_string(),
        });

        let (_, degraded_reason) = expert.runtime_details();
        assert!(degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("deep_runtime_training_report_missing"));
    }

    #[test]
    fn runtime_details_mark_missing_runtime_metadata() {
        let expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend, None);
        assert!(degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("deep_runtime_device_metadata_missing"));
    }

    #[test]
    fn runtime_details_mark_re_resolved_runtime_identity_as_degraded() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.persisted_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: "ndarray_cpu".to_string(),
        });
        expert.host_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "default".to_string(),
            execution_backend: "wgpu_default".to_string(),
        });

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("wgpu_default"));
        let degraded_reason = degraded_reason.expect("runtime re-resolution should be degraded");
        assert!(degraded_reason.contains("deep_runtime_device_re_resolved"));
        assert!(degraded_reason.contains("deep_runtime_backend_re_resolved"));
    }

    #[test]
    fn runtime_details_prefer_live_host_runtime_over_stale_cached_host() {
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            7,
            Some(HashMap::from([("device".to_string(), "cpu".to_string())])),
        );
        let model = expert.init_runtime_model(2);
        let live_backend = expert.resolve_runtime_infer_device().1.execution_backend;
        expert.model = Some(model);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));
        expert.persisted_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: live_backend.clone(),
        });
        expert.host_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "default".to_string(),
            execution_backend: "wgpu_default".to_string(),
        });

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some(live_backend.as_str()));
        assert!(degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("deep_runtime_host_cache_stale"));
    }

    #[test]
    fn fit_persists_effective_burn_device_metadata() -> Result<()> {
        let rsi = (0..140)
            .map(|idx| 0.1_f32 + idx as f32 * 0.01)
            .collect::<Vec<_>>();
        let atr = (0..140)
            .map(|idx| 1.0_f32 + idx as f32 * 0.01)
            .collect::<Vec<_>>();
        let labels = (0..140)
            .map(|idx| match idx % 3 {
                0 => 0_i32,
                1 => 1_i32,
                _ => -1_i32,
            })
            .collect::<Vec<_>>();
        let df = DataFrame::new(vec![
            Series::new("rsi".into(), rsi).into(),
            Series::new("atr".into(), atr).into(),
        ])?;
        let labels = Series::new("label".into(), labels);
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            7,
            Some(HashMap::from([
                ("device".to_string(), "cpu".to_string()),
                ("max_epochs".to_string(), "2".to_string()),
                ("batch_size".to_string(), "4".to_string()),
            ])),
        );
        expert.fit(&df, &labels)?;

        assert_eq!(
            expert
                .params
                .get("requested_device_policy")
                .map(String::as_str),
            Some("cpu")
        );
        assert!(expert.params.contains_key("effective_device_policy"));
        assert!(expert.params.contains_key("execution_backend"));
        assert!(expert.persisted_runtime_selection.is_some());
        assert!(expert.host_runtime_selection.is_some());
        Ok(())
    }

    #[test]
    fn artifact_config_persists_burn_training_report() -> Result<()> {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 100,
            train_rows: 80,
            val_rows: 20,
            embargo_rows: 5,
            class_weights: vec![1.0, 1.0, 1.0],
            best_loss: 0.2,
            best_epoch: Some(3),
            epochs_ran: 4,
            final_train_loss: 0.25,
            learning_rate: 1e-3,
            batch_size: 32,
            patience: 8,
            seed: 7,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            execution_backend: "ndarray_cpu".to_string(),
        });

        let artifact = expert.artifact_config()?;
        assert!(artifact.burn_training_report.is_some());
        Ok(())
    }

    #[test]
    fn artifact_config_rejects_burn_training_report_row_drift() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 101,
            train_rows: 81,
            val_rows: 20,
            embargo_rows: 5,
            class_weights: vec![1.0, 1.0, 1.0],
            best_loss: 0.2,
            best_epoch: Some(3),
            epochs_ran: 4,
            final_train_loss: 0.25,
            learning_rate: 1e-3,
            batch_size: 32,
            patience: 8,
            seed: 7,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            execution_backend: "ndarray_cpu".to_string(),
        });

        let err = expert
            .artifact_config()
            .expect_err("row-drifted burn report should fail");
        assert!(err.to_string().contains("Burn training report rows"));
    }

    #[test]
    fn artifact_config_rejects_burn_training_report_runtime_incoherence() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 100,
            train_rows: 80,
            val_rows: 20,
            embargo_rows: 5,
            class_weights: vec![1.0, 1.0, 1.0],
            best_loss: 0.2,
            best_epoch: Some(3),
            epochs_ran: 4,
            final_train_loss: 0.25,
            learning_rate: 1e-3,
            batch_size: 32,
            patience: 8,
            seed: 7,
            requested_device_policy: "cpu".to_string(),
            effective_device_policy: "cpu".to_string(),
            execution_backend: "wgpu_discrete_gpu".to_string(),
        });

        let err = expert
            .artifact_config()
            .expect_err("runtime-incoherent burn report should fail");
        assert!(err
            .to_string()
            .contains("runtime provenance is internally inconsistent"));
    }

    #[test]
    fn artifact_config_requires_burn_training_report() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 20));

        let err = expert
            .artifact_config()
            .expect_err("missing burn training report should fail");
        assert!(err.to_string().contains("Burn training report"));
    }
}
