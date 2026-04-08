use anyhow::{bail, Context, Result};
use burn::module::{AutodiffModule, Module};
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_artifact_metadata, canonical_three_class_label_mapping,
    dataframe_to_float32_array, feature_columns_from_dataframe, ExpertModel,
};
use crate::burn_models::{
    predict_proba_on_device as burn_predict_proba_on_device, resolve_infer_device,
    resolve_train_device, train_model_with_report as burn_train_model_with_report,
    BurnDeviceSelection, BurnKAN, BurnKANConfig, BurnMLP, BurnMLPConfig, BurnNBeats,
    BurnNBeatsConfig, BurnNBeatsx, BurnNBeatsxConfig, BurnPatchTST, BurnPatchTSTConfig, BurnTabNet,
    BurnTabNetConfig, BurnTiDE, BurnTiDEConfig, BurnTiDENf, BurnTiDENfConfig, BurnTimesNet,
    BurnTimesNetConfig, BurnTrainingReport, BurnTransformer, BurnTransformerConfig, InferBackend,
    TrainBackend, TrainConfig,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

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

    fn artifact_config(&self) -> DeepArtifactConfig {
        DeepArtifactConfig {
            kind: self.kind,
            params: self.params.clone(),
        }
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

    fn init_runtime_model(&self, input_dim: usize) -> RuntimeDeepModel {
        let requested_device = self.string_param("device", "auto");
        let (device, _) = resolve_infer_device(&requested_device);
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
    )> {
        let train_config = self.train_config();
        let requested_device = self.string_param("device", "auto");
        let (device, device_selection) = resolve_train_device(&requested_device);
        match self.kind {
            DeepModelKind::Mlp => {
                let model = self.mlp_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::Mlp(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::NBeats => {
                let model = self.nbeats_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::NBeats(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::NBeatsxNf => {
                let model = self
                    .nbeatsx_nf_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::NBeatsxNf(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::TiDE => {
                let model = self.tide_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::TiDE(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::TiDENf => {
                let model = self.tide_nf_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::TiDENf(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::TabNet => {
                let model = self.tabnet_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::TabNet(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::Kan => {
                let model = self.kan_config(input_dim).init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::Kan(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::Transformer => {
                let model = self
                    .transformer_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::Transformer(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::PatchTst => {
                let model = self
                    .patchtst_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::PatchTst(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
                ))
            }
            DeepModelKind::TimesNet => {
                let model = self
                    .timesnet_config(input_dim)
                    .init::<TrainBackend>(&device);
                let (trained, report) = burn_train_model_with_report::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                )?;
                Ok((
                    RuntimeDeepModel::TimesNet(trained.valid()),
                    Self::training_summary_from_report(&report),
                    device_selection.clone(),
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
        let (model, summary, device_selection) =
            self.train_runtime_model(input_dim, &features, &labels)?;
        self.training_summary = Some(summary);
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
        self.model = Some(model);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
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
        let requested_device = self.string_param("device", "auto");
        let (device, _) = resolve_infer_device(&requested_device);
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
        let model = self
            .model
            .as_ref()
            .with_context(|| format!("{} model is not trained or loaded", self.model_name()))?;
        let metadata = self.metadata()?;

        std::fs::create_dir_all(path)
            .with_context(|| format!("create deep-model directory {}", path.display()))?;
        model.save_to(&Self::model_record_path(path))?;
        Self::write_json(&Self::metadata_path(path), &metadata)?;
        Self::write_json(&Self::config_path(path), &self.artifact_config())?;
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
        let next_params = config.params;
        let next_feature_columns = metadata.feature_columns;
        let next_training_summary = Some(metadata.training_summary);
        let mut next_state = self.clone();
        next_state.params = next_params.clone();
        let next_model = next_state.init_runtime_model(next_feature_columns.len());

        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let base_path = Self::model_record_path(path);
        let requested_device = next_state.string_param("device", "auto");
        let (device, device_selection) = resolve_infer_device(&requested_device);
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
        next_state.params.insert(
            "requested_device_policy".to_string(),
            device_selection.requested_policy,
        );
        next_state.params.insert(
            "effective_device_policy".to_string(),
            device_selection.effective_policy,
        );
        next_state.params.insert(
            "execution_backend".to_string(),
            device_selection.execution_backend,
        );
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
        Ok(())
    }
}
