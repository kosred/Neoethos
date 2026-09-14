use anyhow::{Context, Result, bail};
use burn::module::{AutodiffModule, Module};
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::DType;
use ndarray::Array2;
use neoethos_core::storage::json::{
    JsonBackupWriteConfig, read_json as read_json_artifact,
    write_json_with_backup as write_json_artifact_with_backup,
};
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    feature_columns_from_frame, feature_frame_to_f64_array, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata, validate_model_labels,
};
use crate::burn_models::{
    BurnDeviceSelection, BurnKAN, BurnKANConfig, BurnMLP, BurnMLPConfig, BurnNBeats,
    BurnNBeatsConfig, BurnNBeatsx, BurnNBeatsxConfig, BurnPatchTST, BurnPatchTSTConfig, BurnTabNet,
    BurnTabNetConfig, BurnTiDE, BurnTiDEConfig, BurnTiDENf, BurnTiDENfConfig, BurnTimesNet,
    BurnTimesNetConfig, BurnTrainingReport, BurnTransformer, BurnTransformerConfig, InferBackend,
    TrainBackend, TrainConfig, cast_module_float_tensors, normalize_burn_device_policy,
    predict_proba_on_device as burn_predict_proba_on_device, resolve_infer_device,
    resolve_train_device,
    train_model_with_report_with_external_val as burn_train_model_with_report_with_external_val,
    train_model_with_transport_v1, validate_burn_device_selection,
};
#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend", test))]
use crate::burn_models::{BurnResidentDatasetPlanV1, resident_tensor_bytes_v1};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, normalize_training_precision_policy,
};
use crate::runtime::prediction::RuntimePrediction;

const METADATA_FILE_NAME: &str = "metadata.json";
const CONFIG_FILE_NAME: &str = "config.json";
const MODEL_RECORD_BASENAME: &str = "model";

/// Exact parameter shape of BurnMLPConfig::init (including linear biases and
/// both LayerNorm parameters). All admission arithmetic precedes tensor allocation.
#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend", test))]
fn mlp_parameter_count(input: usize, hidden: usize, layers: usize) -> Result<usize> {
    if input == 0 || hidden == 0 || layers == 0 {
        bail!("MLP capacity requires positive input, hidden and layer dimensions");
    }
    input
        .checked_mul(hidden)
        .and_then(|n| {
            hidden
                .checked_mul(hidden)?
                .checked_mul(layers - 1)?
                .checked_add(n)
        })
        .and_then(|n| layers.checked_mul(3)?.checked_mul(hidden)?.checked_add(n))
        .and_then(|n| hidden.checked_mul(3)?.checked_add(3)?.checked_add(n))
        .context("MLP parameter count overflow")
}

/// Conservative padded live-set estimate, not an allocator/RSS reservation.
/// Eight FP32 parameter copies cover weights, gradients, AdamW moments, best
/// weights and update/cast temporaries. Training reserves sixteen activation
/// copies; validation reserves eight and uses its ENTIRE row span, as the
/// current trainer does. Allocator/kernel workspace headroom is reserved again
/// when translating live free memory into an operation budget.
#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend", test))]
fn mlp_training_bytes(
    input: usize,
    hidden: usize,
    layers: usize,
    batch_rows: usize,
    validation_rows: usize,
    alignment: usize,
) -> Result<usize> {
    mlp_parameter_count(input, hidden, layers)?; // shared shape/overflow guard
    let tensor = |height, width| resident_tensor_bytes_v1(height, width, usize::MAX, alignment);
    // Exact initialized topology: first linear, remaining linears, three
    // hidden vectors per layer (bias + LayerNorm gamma/beta), output + bias.
    let parameters = tensor(input, hidden)?
        .checked_add(if layers > 1 {
            tensor(hidden, hidden)?
                .checked_mul(layers - 1)
                .context("MLP hidden parameter bytes overflow")?
        } else {
            0
        })
        .and_then(|n| {
            tensor(1, hidden)
                .ok()?
                .checked_mul(3)?
                .checked_mul(layers)?
                .checked_add(n)
        })
        .and_then(|n| n.checked_add(tensor(hidden, 3).ok()?))
        .and_then(|n| n.checked_add(tensor(1, 3).ok()?))
        .context("MLP padded parameter bytes overflow")?;
    let activations = |rows| -> Result<usize> {
        tensor(rows, input)?
            .checked_add(
                tensor(rows, hidden)?
                    .checked_mul(layers)
                    .context("MLP hidden activation bytes overflow")?,
            )
            .and_then(|n| n.checked_add(tensor(rows, 3).ok()?))
            .context("MLP padded activation bytes overflow")
    };
    let training = activations(batch_rows)?
        .checked_mul(16)
        .context("MLP training activation bytes overflow")?;
    let validation = activations(validation_rows)?
        .checked_mul(8)
        .context("MLP validation activation bytes overflow")?;
    parameters
        .checked_mul(8)
        .and_then(|n| n.checked_add(training.max(validation)))
        .context("MLP training live-set bytes overflow")
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend", test))]
fn mlp_admitted_width(
    input: usize,
    requested: usize,
    layers: usize,
    batch_rows: usize,
    validation_rows: usize,
    budget: usize,
    max_tensor_bytes: usize,
    alignment: usize,
    automatic: bool,
) -> Result<usize> {
    let max_tensor_bytes = max_tensor_bytes.min((u32::MAX as usize).saturating_mul(4));
    let fits = |width: usize| -> bool {
        let weight_axis = input.max(if layers > 1 { width } else { 0 }).max(3);
        // Both orientations bound linear weights and their transpose/copy;
        // pitched extents, not only logical cell counts, must fit one page.
        resident_tensor_bytes_v1(weight_axis, width, max_tensor_bytes, alignment).is_ok()
            && resident_tensor_bytes_v1(width, weight_axis, max_tensor_bytes, alignment).is_ok()
            && resident_tensor_bytes_v1(
                batch_rows.max(validation_rows),
                input.max(width).max(3),
                max_tensor_bytes,
                alignment,
            )
            .is_ok()
            && mlp_training_bytes(input, width, layers, batch_rows, validation_rows, alignment)
                .is_ok_and(|n| n <= budget)
    };
    if !fits(requested) {
        bail!(
            "MLP configured minimum architecture does not fit current memory admission (hidden_dim={requested}, budget_bytes={budget}, validation_rows={validation_rows})"
        );
    }
    if !automatic {
        return Ok(requested);
    }
    // The padded estimate and per-tensor dimensions are monotone in width.
    // No catalogue width ceiling: both available memory and actual rows/features
    // determine the upper bound. Explicit depth and batch size are unchanged.
    let mut low = requested;
    let mut high = budget / 32;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend", test))]
fn mlp_fractional_width(requested: usize, admitted: usize, fraction: f64) -> Result<usize> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 || admitted < requested {
        bail!("MLP capacity_fraction must be finite in (0, 1] within the admitted width range");
    }
    let span = admitted - requested;
    Ok(requested + ((span as f64 * fraction).floor() as usize).min(span))
}

#[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
fn mlp_host_bytes(rows: usize, input: usize, bytes_per_cell: usize) -> Result<usize> {
    rows.checked_mul(input)
        .and_then(|n| n.checked_mul(bytes_per_cell))
        .and_then(|n| rows.checked_mul(32)?.checked_add(n))
        .context("MLP host staging byte count overflow")
}

/// Checked, backend-local narrowing for Burn, whose public tensor input is
/// intrinsically f32. Shared model input remains `FeatureFrame` f64+validity;
/// values that cannot survive this explicit boundary fail closed.
fn deep_backend_f32_matrix(frame: &FeatureFrame) -> Result<Array2<f32>> {
    let source = feature_frame_to_f64_array(frame).context("materialize typed deep-model frame")?;
    let mut narrowed = Vec::with_capacity(source.len());
    for ((row, column), value) in source.indexed_iter() {
        if value.abs() > f32::MAX as f64 {
            bail!(
                "deep backend f32 adapter cannot represent feature row {row} column {column}: {value}"
            );
        }
        let converted = *value as f32;
        if !converted.is_finite() {
            bail!("deep backend f32 adapter produced non-finite feature row {row} column {column}");
        }
        if *value != 0.0 && converted == 0.0 {
            bail!(
                "deep backend f32 adapter underflowed non-zero feature row {row} column {column}: {value}"
            );
        }
        narrowed.push(converted);
    }
    Array2::from_shape_vec(source.dim(), narrowed).context("shape deep backend f32 feature matrix")
}

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
    #[serde(default)]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
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
        device: &<InferBackend as burn::tensor::backend::BackendTypes>::Device,
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
    // Declared after the model, shared by clones, and retained until its last
    // tensor handle is dropped on the exact originating Fusion/CubeCL stream.
    #[cfg(feature = "burn-rocm-backend")]
    rocm_residency: Option<std::sync::Arc<crate::burn_rocm_backend::RocmModelResidency>>,
}

impl BurnDeepExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`]
    /// adapter so the registry / aggregator can detect column-layout
    /// drift after a retraining session.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

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
            #[cfg(feature = "burn-rocm-backend")]
            rocm_residency: None,
        }
    }

    #[cfg(feature = "burn-rocm-backend")]
    fn ensure_rocm_residency(
        &mut self,
    ) -> Result<std::sync::Arc<crate::burn_rocm_backend::RocmModelResidency>> {
        let policy = self.configured_requested_device_policy();
        let ordinal = crate::common::parse_rocm_device_ordinal(&policy)?;
        if let Some(owner) = self.rocm_residency.as_ref() {
            if owner.ordinal() != ordinal {
                bail!("ROCm model device changed while its tensor handles are still retained");
            }
            return Ok(std::sync::Arc::clone(owner));
        }
        let owner =
            std::sync::Arc::new(crate::burn_rocm_backend::RocmModelResidency::new(&policy)?);
        self.rocm_residency = Some(std::sync::Arc::clone(&owner));
        Ok(owner)
    }

    fn on_runtime_stream<T>(&self, operation: impl FnOnce() -> T) -> T {
        #[cfg(feature = "burn-rocm-backend")]
        if let Some(owner) = self.rocm_residency.as_ref() {
            return owner.executes(operation);
        }
        operation()
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

    fn automatic_mlp_capacity(&self) -> Result<bool> {
        match self.params.get("capacity_mode").map(|v| v.trim()) {
            None | Some("fixed") => Ok(false),
            Some("auto") if self.kind == DeepModelKind::Mlp => Ok(true),
            Some("auto") => bail!("automatic capacity is currently supported only for MLP"),
            Some(other) => {
                bail!("invalid deep-model capacity_mode `{other}`; expected fixed or auto")
            }
        }
    }

    fn mlp_capacity_fraction(&self) -> Result<f64> {
        let fraction = self
            .params
            .get("capacity_fraction")
            .map(|value| {
                value
                    .parse::<f64>()
                    .context("invalid MLP capacity_fraction")
            })
            .transpose()?
            .unwrap_or(1.0);
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            bail!("MLP capacity_fraction must be finite in (0, 1]");
        }
        Ok(fraction)
    }

    #[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
    fn admitted_mlp_config(
        &self,
        input: usize,
        rows: usize,
        external_validation_rows: Option<usize>,
        selection: &BurnDeviceSelection,
    ) -> Result<(BurnMLPConfig, usize, usize, BurnResidentDatasetPlanV1)> {
        #[cfg(feature = "burn-cuda-backend")]
        let (free, total, max_page_size, alignment) = {
            let ordinal = selection
                .effective_policy
                .strip_prefix("gpu:")
                .and_then(|v| v.parse::<usize>().ok())
                .context("MLP memory admission requires the exact selected CUDA ordinal")?;
            // Inspect the same runtime/device as Burn, without registering an
            // allocation owner before the trainer's own lifetime has begun.
            let client = <cubecl::cuda::CudaRuntime as cubecl::prelude::Runtime>::client(
                &cubecl::cuda::CudaDevice::new(ordinal),
            );
            let context = cudarc::driver::CudaContext::new(ordinal)
                .context("retain MLP CUDA memory-query context")?;
            let (free, total) = context
                .mem_get_info()
                .context("query live MLP CUDA memory")?;
            if total == 0 {
                bail!("MLP CUDA memory query returned zero capacity");
            }
            (
                free,
                total,
                usize::try_from(client.properties().memory.max_page_size)
                    .context("MLP CUDA maximum tensor size exceeds usize")?,
                usize::try_from(client.properties().memory.alignment)
                    .context("MLP CUDA allocation alignment exceeds usize")?,
            )
        };
        #[cfg(feature = "burn-rocm-backend")]
        let (free, total, max_page_size, alignment) = {
            let ordinal = crate::common::parse_rocm_device_ordinal(&selection.effective_policy)?;
            let (free, total, max_page_size) =
                crate::burn_rocm_backend::mlp_memory_snapshot(ordinal)?;
            let client = <cubecl::hip::HipRuntime as cubecl::prelude::Runtime>::client(
                &cubecl::hip::AmdDevice::new(ordinal),
            );
            let alignment = usize::try_from(client.properties().memory.alignment)
                .context("MLP ROCm allocation alignment exceeds usize")?;
            (free, total, max_page_size, alignment)
        };
        let reserve = (total / 10).max(256 * 1024 * 1024).min(total);
        // Leave half of the remaining free memory for allocator pages, fusion
        // and matmul workspaces not described by the logical live-set estimate.
        let mut budget = free.min(total).saturating_sub(reserve) / 2;
        if let Some(value) = self.params.get("memory_budget_gb") {
            let gb = value
                .parse::<f64>()
                .context("MLP memory_budget_gb must be numeric")?;
            if !gb.is_finite() || gb <= 0.0 || gb * 1_073_741_824.0 > usize::MAX as f64 {
                bail!("MLP memory_budget_gb must be finite, positive and representable");
            }
            budget = budget.min((gb * 1_073_741_824.0) as usize);
        }
        let train = self.train_config();
        let dataset_plan = BurnResidentDatasetPlanV1::checked(
            rows,
            input,
            external_validation_rows,
            train.batch_size,
            max_page_size,
            alignment,
        )?;
        let validation_rows = dataset_plan.validation_rows();
        let model_budget = budget.checked_sub(dataset_plan.peak_bytes()).context(
            "MLP resident dataset plus gather/upload workspace exceeds the admitted memory budget",
        )?;
        let all_rows = rows
            .checked_add(external_validation_rows.unwrap_or(0))
            .context("MLP dataset row count overflow")?;
        // Split arrays plus a largest full-split TensorData flattening copy can
        // coexist during the one-time upload. This is distinct from VRAM.
        let remaining_host = mlp_host_bytes(all_rows, input, 8)?
            .checked_add(mlp_host_bytes(dataset_plan.batch_rows(), input, 12)?)
            .context("MLP remaining host allocation estimate overflow")?;
        if remaining_host as u64 > neoethos_core::available_memory_bytes() / 2 {
            bail!("MLP training copies exceed current available host RAM");
        }
        let mut config = self.mlp_config(input);
        let automatic = self.automatic_mlp_capacity()?;
        let requested = if automatic {
            self.params
                .get("capacity_requested_hidden_dim")
                .map(|value| {
                    value
                        .parse::<usize>()
                        .context("invalid MLP requested capacity width")
                })
                .transpose()?
                .unwrap_or(config.hidden_dim)
        } else {
            config.hidden_dim
        };
        let admitted_width = mlp_admitted_width(
            input,
            requested,
            config.n_layers,
            dataset_plan.batch_rows(),
            validation_rows,
            model_budget,
            max_page_size,
            alignment,
            automatic,
        )?;
        let fraction = if automatic {
            self.mlp_capacity_fraction()?
        } else {
            1.0
        };
        let width = mlp_fractional_width(requested, admitted_width, fraction)?;
        config.hidden_dim = width;
        let estimate = mlp_training_bytes(
            input,
            width,
            config.n_layers,
            dataset_plan.batch_rows(),
            validation_rows,
            alignment,
        )?
        .checked_add(dataset_plan.peak_bytes())
        .context("MLP total admitted training bytes overflow")?;
        tracing::info!(target: "neoethos_models::burn", requested_hidden_dim=requested,
            resolved_hidden_dim=width, layers=config.n_layers, batch_size=train.batch_size,
            validation_rows, free_vram_bytes=free, budget_bytes=budget,
            train_rows=dataset_plan.train_rows(), resident_dataset_peak_bytes=dataset_plan.peak_bytes(),
            model_budget_bytes=model_budget, transport="inner-backend-resident",
            estimated_training_bytes=estimate, automatic, capacity_fraction=fraction,
            admitted_hidden_dim=admitted_width, "MLP live-memory capacity admission");
        Ok((config, requested, estimate, dataset_plan))
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

        try_build_runtime_artifact_metadata(
            self.model_name(),
            ModelFamily::Deep,
            CapabilityState::Implemented,
            self.feature_columns.clone(),
            canonical_three_class_label_mapping(),
            training_summary,
        )
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
            if let Some(value) = params.get(key)
                && value.trim().is_empty()
            {
                bail!("deep-model runtime param `{key}` may not be blank");
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
        if let Some(value) = params.get("execution_backend")
            && !Self::is_supported_execution_backend(value)
        {
            bail!(
                "deep-model runtime param `execution_backend` uses unsupported backend `{}`",
                value
            );
        }
        if let Some(value) = params.get("training_precision") {
            let normalized = normalize_training_precision_policy(value);
            if !matches!(
                normalized.as_str(),
                "auto" | "fp32" | "bf16" | "fp8" | "bf4"
            ) {
                bail!(
                    "deep-model runtime param `training_precision` uses unsupported precision `{}`",
                    value
                );
            }
        }
        if let Some(value) = params.get("training_precision_reason")
            && value.trim().is_empty()
        {
            bail!("deep-model runtime param `training_precision_reason` may not be blank");
        }
        if let Some(device) = params.get("device")
            && device.trim().is_empty()
        {
            bail!("deep-model runtime param `device` may not be blank");
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
        #[cfg(feature = "burn-rocm-backend")]
        {
            return matches!(normalized, "auto" | "cpu")
                || crate::common::parse_rocm_device_ordinal(normalized).is_ok();
        }
        #[cfg(not(feature = "burn-rocm-backend"))]
        {
            matches!(normalized, "auto" | "cpu" | "gpu")
                || normalized
                    .strip_prefix("gpu:")
                    .is_some_and(|ordinal| ordinal.parse::<usize>().is_ok())
        }
    }

    fn is_supported_execution_backend(backend: &str) -> bool {
        matches!(backend.trim(), "ndarray_cpu" | "cuda")
            || (cfg!(feature = "burn-rocm-backend") && backend.trim() == "rocm")
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
            || report.embargo_rows != summary.embargo_rows
            || report.val_rows != summary.val_rows
            || report
                .train_rows
                .checked_add(report.embargo_rows)
                .and_then(|rows| rows.checked_add(report.val_rows))
                != Some(report.dataset_rows)
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
            ("training_precision", report.training_precision.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!(
                    "{} Burn training report `{field_name}` may not be blank",
                    self.model_name()
                );
            }
        }
        if !matches!(report.training_precision.as_str(), "fp32" | "bf16") {
            bail!(
                "{} Burn training report `training_precision` must be an implemented runtime precision, got `{}`",
                self.model_name(),
                report.training_precision
            );
        }
        if let Some(reason) = report.training_precision_reason.as_ref()
            && reason.trim().is_empty()
        {
            bail!(
                "{} Burn training report `training_precision_reason` may not be blank",
                self.model_name()
            );
        }
        let report_runtime = Self::runtime_selection_from_report(report);
        validate_burn_device_selection(&report_runtime).with_context(|| {
            format!(
                "{} Burn training report runtime provenance is internally inconsistent",
                self.model_name()
            )
        })?;
        if let Some(selection) = runtime_selection
            && (report_runtime.requested_policy != selection.requested_policy
                || report_runtime.effective_policy != selection.effective_policy
                || report_runtime.execution_backend != selection.execution_backend)
        {
            bail!(
                "{} Burn training report runtime provenance does not match persisted runtime selection",
                self.model_name()
            );
        }
        Ok(())
    }

    fn validate_model_params(&self) -> Result<()> {
        self.automatic_mlp_capacity()?;
        self.mlp_capacity_fraction()?;
        for key in [
            "hidden_dim",
            "n_layers",
            "n_blocks",
            "n_steps",
            "n_heads",
            "dim_ff",
            "patch_size",
            "n_periods",
            "token_count",
            "grid_size",
            "batch_size",
            "max_epochs",
            "patience",
            "capacity_requested_hidden_dim",
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
        if let Some(value) = self.params.get("grid_size") {
            let parsed = value.trim().parse::<usize>().map_err(|_| {
                anyhow::anyhow!("deep-model param `grid_size` must parse as a positive integer")
            })?;
            if !(3..=33).contains(&parsed) {
                bail!("deep-model param `grid_size` must be between 3 and 33");
            }
        }
        if let Some(value) = self.params.get("token_count") {
            let parsed = value.trim().parse::<usize>().map_err(|_| {
                anyhow::anyhow!("deep-model param `token_count` must parse as a positive integer")
            })?;
            if !(1..=256).contains(&parsed) {
                bail!("deep-model param `token_count` must be between 1 and 256");
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
        if let Some(value) = self.params.get("relaxation_factor") {
            let parsed = value.trim().parse::<f64>().map_err(|_| {
                anyhow::anyhow!("deep-model param `relaxation_factor` must parse as a finite float")
            })?;
            if !parsed.is_finite() || parsed < 1.0 {
                bail!("deep-model param `relaxation_factor` must be finite and >= 1.0");
            }
        }
        Ok(())
    }

    fn artifact_config(&self) -> Result<DeepArtifactConfig> {
        Self::validate_runtime_params(&self.params)?;
        self.validate_model_params()?;
        let runtime_metadata = self.metadata()?;
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
            runtime_metadata: Some(runtime_metadata),
        })
    }

    fn resolve_loaded_metadata(
        path: &Path,
        config: &DeepArtifactConfig,
        expected_model_name: &str,
    ) -> Result<RuntimeArtifactMetadata> {
        let metadata_path = Self::metadata_path(path);
        if metadata_path.exists() {
            let sidecar: RuntimeArtifactMetadata = Self::read_json(&metadata_path)?;
            if let Some(embedded) = config.runtime_metadata.as_ref()
                && &sidecar != embedded
            {
                bail!(
                    "deep artifact {} metadata sidecar mismatch with embedded runtime metadata",
                    path.display()
                );
            }
            return Ok(sidecar);
        }
        if let Some(metadata) = config.runtime_metadata.clone() {
            tracing::warn!(
                model = %expected_model_name,
                path = %path.display(),
                "deep-model metadata sidecar missing; using runtime metadata embedded in config"
            );
            return Ok(metadata);
        }
        bail!(
            "deep artifact {} is missing metadata sidecar and embedded runtime metadata",
            path.display()
        );
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
            .with_relaxation_factor(self.float_param("relaxation_factor", 1.5))
    }

    fn kan_config(&self, input_dim: usize) -> BurnKANConfig {
        BurnKANConfig::new(input_dim)
            .with_hidden_dim(self.usize_param("hidden_dim", 32))
            .with_n_layers(self.usize_param("n_layers", 3))
            .with_grid_size(self.usize_param("grid_size", 9))
            .with_n_classes(3)
            .with_dropout(self.float_param("dropout", 0.05))
    }

    fn transformer_config(&self, input_dim: usize) -> BurnTransformerConfig {
        let hidden_dim = self.usize_param("hidden_dim", 128);
        BurnTransformerConfig::new(input_dim)
            .with_hidden_dim(hidden_dim)
            .with_n_heads(self.compatible_head_count(hidden_dim, 8))
            .with_n_layers(self.usize_param("n_layers", 4))
            .with_token_count(self.usize_param("token_count", 8))
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

    fn configured_requested_training_precision(&self) -> Option<String> {
        self.params
            .get("training_precision")
            .map(|value| normalize_training_precision_policy(value))
    }

    fn resolve_runtime_infer_device(
        &self,
    ) -> Result<(
        <InferBackend as burn::tensor::backend::BackendTypes>::Device,
        BurnDeviceSelection,
    )> {
        let requested_device = self.configured_requested_device_policy();
        resolve_infer_device(&requested_device)
    }

    fn runtime_model_dtype(
        &self,
        device: &<InferBackend as burn::tensor::backend::BackendTypes>::Device,
    ) -> Result<DType> {
        match self
            .configured_requested_training_precision()
            .unwrap_or_else(|| "fp32".to_string())
            .as_str()
        {
            "bf16" => {
                if <InferBackend as burn::tensor::backend::Backend>::supports_dtype(
                    device,
                    DType::BF16,
                ) {
                    Ok(DType::BF16)
                } else {
                    bail!(
                        "{} runtime backend does not support persisted bf16 model precision",
                        self.model_name()
                    );
                }
            }
            "fp32" | "auto" => Ok(DType::F32),
            other => bail!(
                "{} runtime precision `{}` is not loadable for inference; expected fp32 or bf16",
                self.model_name(),
                other
            ),
        }
    }

    fn init_runtime_model(&self, input_dim: usize) -> Result<RuntimeDeepModel> {
        let (device, _) = self.resolve_runtime_infer_device()?;
        let runtime_dtype = self.runtime_model_dtype(&device)?;
        match self.kind {
            DeepModelKind::Mlp => Ok(RuntimeDeepModel::Mlp(cast_module_float_tensors(
                self.mlp_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::NBeats => Ok(RuntimeDeepModel::NBeats(cast_module_float_tensors(
                self.nbeats_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::NBeatsxNf => Ok(RuntimeDeepModel::NBeatsxNf(cast_module_float_tensors(
                self.nbeatsx_nf_config(input_dim)
                    .init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::TiDE => Ok(RuntimeDeepModel::TiDE(cast_module_float_tensors(
                self.tide_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::TiDENf => Ok(RuntimeDeepModel::TiDENf(cast_module_float_tensors(
                self.tide_nf_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::TabNet => Ok(RuntimeDeepModel::TabNet(cast_module_float_tensors(
                self.tabnet_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::Kan => Ok(RuntimeDeepModel::Kan(cast_module_float_tensors(
                self.kan_config(input_dim).init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::Transformer => {
                Ok(RuntimeDeepModel::Transformer(cast_module_float_tensors(
                    self.transformer_config(input_dim)
                        .init::<InferBackend>(&device),
                    runtime_dtype,
                )))
            }
            DeepModelKind::PatchTst => Ok(RuntimeDeepModel::PatchTst(cast_module_float_tensors(
                self.patchtst_config(input_dim)
                    .init::<InferBackend>(&device),
                runtime_dtype,
            ))),
            DeepModelKind::TimesNet => Ok(RuntimeDeepModel::TimesNet(cast_module_float_tensors(
                self.timesnet_config(input_dim)
                    .init::<InferBackend>(&device),
                runtime_dtype,
            ))),
        }
    }

    fn training_summary_from_report(report: &BurnTrainingReport) -> TrainingSummaryMetadata {
        TrainingSummaryMetadata::new(
            report.dataset_rows,
            report.train_rows,
            report.embargo_rows,
            report.val_rows,
        )
    }

    /// Trains a runtime model and optionally accepts an explicit external
    /// validation pair so the early-stopping signal can match the HPO
    /// objective. Passing `None` for both validation inputs reverts to Burn's
    /// internal time_series_split holdout.
    #[allow(clippy::too_many_arguments)]
    fn train_runtime_model_with_val(
        &mut self,
        input_dim: usize,
        features: &Array2<f32>,
        labels: &[i32],
        external_val_x: Option<&Array2<f32>>,
        external_val_y: Option<&[i32]>,
    ) -> Result<(
        RuntimeDeepModel,
        TrainingSummaryMetadata,
        BurnDeviceSelection,
        BurnTrainingReport,
    )> {
        let train_config = self.train_config();
        let requested_device = self.configured_requested_device_policy();
        let requested_training_precision = self.configured_requested_training_precision();
        let (device, device_selection) = resolve_train_device(&requested_device)?;
        match self.kind {
            DeepModelKind::Mlp => {
                #[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
                let (mlp_config, requested_width, estimated_bytes, dataset_plan) = self
                    .admitted_mlp_config(
                        input_dim,
                        features.nrows(),
                        external_val_x.map(|values| values.nrows()),
                        &device_selection,
                    )?;
                #[cfg(not(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend")))]
                let mlp_config = {
                    if self.automatic_mlp_capacity()? {
                        bail!(
                            "automatic MLP capacity requires a native CUDA or ROCm training backend"
                        );
                    }
                    self.mlp_config(input_dim)
                };
                #[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
                let dataset_plan = Some(dataset_plan);
                #[cfg(not(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend")))]
                let dataset_plan = None;
                let model = mlp_config.init::<TrainBackend>(&device);
                let (trained, report) = train_model_with_transport_v1::<TrainBackend, _>(
                    model,
                    features,
                    labels,
                    &train_config,
                    &device,
                    &device_selection,
                    requested_training_precision.as_deref(),
                    external_val_x,
                    external_val_y,
                    dataset_plan,
                )?;
                #[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
                {
                    // Persist the actual initialized architecture, not a plan
                    // that would be re-resolved against another machine on load.
                    self.params
                        .insert("hidden_dim".into(), mlp_config.hidden_dim.to_string());
                    self.params.insert(
                        "capacity_requested_hidden_dim".into(),
                        requested_width.to_string(),
                    );
                    self.params.insert(
                        "capacity_estimated_training_bytes".into(),
                        estimated_bytes.to_string(),
                    );
                }
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
                    burn_train_model_with_report_with_external_val::<TrainBackend, _>(
                        model,
                        features,
                        labels,
                        &train_config,
                        &device,
                        &device_selection,
                        requested_training_precision.as_deref(),
                        external_val_x,
                        external_val_y,
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
        write_json_artifact_with_backup(
            path,
            value,
            JsonBackupWriteConfig {
                artifact_label: "deep-model artifact",
                temp_extension: "tmp",
                backup_extension: "bak",
            },
        )
    }

    fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
        read_json_artifact(path, "deep-model")
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
                if let Err(restore_err) = std::fs::rename(&backup_path, target_path) {
                    tracing::error!(
                        target: "neoethos_models::artifact",
                        backup = %backup_path.display(),
                        target = %target_path.display(),
                        error = %restore_err,
                        "failed to restore backup after staged-rename failure;                      artifact directory may be in an inconsistent state"
                    );
                }
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
        let partition_rows = summary
            .train_rows
            .checked_add(summary.embargo_rows)
            .and_then(|rows| rows.checked_add(summary.val_rows));
        if Some(summary.dataset_rows) != partition_rows {
            bail!(
                "deep-model training summary is inconsistent: dataset_rows={} but train_rows + embargo_rows + val_rows = {:?}",
                summary.dataset_rows,
                partition_rows
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
        let runtime_selection = Self::runtime_selection_from_params(&self.params)?.with_context(
            || {
                format!(
                    "{} runtime params must persist requested_device_policy, effective_device_policy, and execution_backend together",
                    self.model_name()
                )
            },
        )?;
        self.validate_burn_training_report(
            summary,
            Some(&runtime_selection),
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
            self.resolve_runtime_infer_device()
                .ok()
                .map(|(_, selection)| selection)
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
        ) && (cached_host.requested_policy != live_host.requested_policy
            || cached_host.effective_policy != live_host.effective_policy
            || cached_host.execution_backend != live_host.execution_backend)
        {
            degraded.push("deep_runtime_host_cache_stale".to_string());
        }
        if let Some(persisted) = persisted
            && persisted.requested_policy != persisted.effective_policy
        {
            degraded.push("deep_requested_device_unavailable".to_string());
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

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
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

    /// M5: shared body for `fit` and `fit_with_validation`. When the caller
    /// supplies an explicit validation pair (HPO path), it is forwarded to
    /// `train_runtime_model_with_val` so Burn drives early stopping against
    /// the same val data the HPO objective scores. When no external val is
    /// supplied we fall back to Burn's internal time_series_split holdout.
    fn fit_internal(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
    ) -> Result<()> {
        #[cfg(feature = "burn-rocm-backend")]
        {
            self.validate_model_params()?;
            let owner = self.ensure_rocm_residency()?;
            return owner.executes(|| self.fit_internal_on_stream(x, y, val_x, val_y));
        }
        #[cfg(not(feature = "burn-rocm-backend"))]
        self.fit_internal_on_stream(x, y, val_x, val_y)
    }

    fn fit_internal_on_stream(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
    ) -> Result<()> {
        self.validate_model_params()?;
        #[cfg(any(feature = "burn-cuda-backend", feature = "burn-rocm-backend"))]
        if self.kind == DeepModelKind::Mlp {
            let rows = x
                .n_samples()
                .checked_add(val_x.map_or(0, FeatureFrame::n_samples))
                .context("MLP host dataset row count overflow")?;
            if mlp_host_bytes(rows, x.n_features(), 12)? as u64
                > neoethos_core::available_memory_bytes() / 2
            {
                bail!("MLP input materialization exceeds current available host RAM");
            }
        }
        let features = deep_backend_f32_matrix(x)
            .with_context(|| format!("build {} feature matrix", self.model_name()))?;
        validate_model_labels(y, features.nrows())
            .with_context(|| format!("validate {} training labels", self.model_name()))?;
        let input_dim = features.ncols();

        let val_arrays = match (val_x, val_y) {
            (Some(vx), Some(vy)) => {
                x.ensure_semantically_compatible(vx).with_context(|| {
                    format!(
                        "validate {} train/validation feature plan",
                        self.model_name()
                    )
                })?;
                let vx_array = deep_backend_f32_matrix(vx).with_context(|| {
                    format!("build {} validation feature matrix", self.model_name())
                })?;
                validate_model_labels(vy, vx_array.nrows())
                    .with_context(|| format!("validate {} validation labels", self.model_name()))?;
                let val_columns = feature_columns_from_frame(vx);
                let train_columns = feature_columns_from_frame(x);
                if val_columns != train_columns {
                    bail!(
                        "{} validation column mismatch: train {:?} vs val {:?}",
                        self.model_name(),
                        train_columns,
                        val_columns
                    );
                }
                Some((vx_array, vy.to_vec()))
            }
            (None, None) => None,
            _ => bail!(
                "{} fit_with_validation requires both val_x and val_y or neither",
                self.model_name()
            ),
        };

        self.feature_columns = feature_columns_from_frame(x);
        self.validate_model_params()?;
        let (val_x_ref, val_y_ref) = match val_arrays.as_ref() {
            Some((vx, vy)) => (Some(vx), Some(vy.as_slice())),
            None => (None, None),
        };
        let (model, summary, device_selection, burn_training_report) =
            self.train_runtime_model_with_val(input_dim, &features, y, val_x_ref, val_y_ref)?;
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
        if let Some(report) = self.burn_training_report.as_ref() {
            self.params.insert(
                "training_precision".to_string(),
                report.training_precision.clone(),
            );
            if let Some(reason) = report.training_precision_reason.as_ref() {
                self.params
                    .insert("training_precision_reason".to_string(), reason.clone());
            } else {
                self.params.remove("training_precision_reason");
            }
        }
        self.persisted_runtime_selection = Self::runtime_selection_from_params(&self.params)?;
        self.host_runtime_selection = self.persisted_runtime_selection.clone();
        self.model = Some(model);
        Ok(())
    }
}

impl ExpertModel for BurnDeepExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        lease.scope(|| self.fit_internal(x, y, None, None))
    }

    fn fit_with_validation(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        val_x: Option<&FeatureFrame>,
        val_y: Option<&[i32]>,
        lease: &CpuLease,
    ) -> Result<()> {
        lease.scope(|| self.fit_internal(x, y, val_x, val_y))
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        lease.scope(|| {
            self.on_runtime_stream(|| {
                self.ensure_runtime_state_ready()?;
                let model = self.model.as_ref().with_context(|| {
                    format!("{} model is not trained or loaded", self.model_name())
                })?;

                let actual_columns = feature_columns_from_frame(x);
                if !self.feature_columns.is_empty() && self.feature_columns != actual_columns {
                    bail!(
                        "feature column mismatch for persisted deep model; expected {:?}, got {:?}",
                        self.feature_columns,
                        actual_columns
                    );
                }

                let features = deep_backend_f32_matrix(x)
                    .with_context(|| format!("build {} inference matrix", self.model_name()))?;
                let (device, _) = self.resolve_runtime_infer_device()?;
                let probabilities =
                    model.predict_probabilities(&features, self.batch_size(), &device)?;
                if probabilities.ncols() != 3 {
                    bail!(
                        "{} should output 3 probability columns, got {}",
                        self.model_name(),
                        probabilities.ncols()
                    );
                }
                Ok(probabilities.mapv(f64::from))
            })
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.on_runtime_stream(|| {
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
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let config: DeepArtifactConfig = Self::read_json(&Self::config_path(path))?;
        let metadata: RuntimeArtifactMetadata =
            Self::resolve_loaded_metadata(path, &config, self.model_name())?;
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
        #[cfg(feature = "burn-rocm-backend")]
        let owner = next_state.ensure_rocm_residency()?;
        let complete_load = || -> Result<()> {
            let next_model = next_state.init_runtime_model(next_feature_columns.len())?;

            let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            let base_path = Self::model_record_path(path);
            let (device, host_runtime_selection) = next_state.resolve_runtime_infer_device()?;
            if let Some(persisted_runtime_selection) =
                next_state.persisted_runtime_selection.as_ref()
                && (persisted_runtime_selection.requested_policy
                    != host_runtime_selection.requested_policy
                    || persisted_runtime_selection.effective_policy
                        != host_runtime_selection.effective_policy
                    || persisted_runtime_selection.execution_backend
                        != host_runtime_selection.execution_backend)
            {
                bail!(
                    "{} runtime identity drift between persisted {:?} and host {:?}",
                    self.model_name(),
                    persisted_runtime_selection,
                    host_runtime_selection
                );
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
        };
        #[cfg(feature = "burn-rocm-backend")]
        {
            owner.executes(complete_load)
        }
        #[cfg(not(feature = "burn-rocm-backend"))]
        {
            complete_load()
        }
    }
}

#[cfg(feature = "burn-rocm-backend")]
impl Drop for BurnDeepExpert {
    fn drop(&mut self) {
        // Fusion records DropOps on the owning stream even if the caller
        // moved this expert to another host thread before destruction.
        if let Some(owner) = self.rocm_residency.as_ref() {
            owner.drop_handles(|| drop(self.model.take()));
        }
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

            pub fn predict_runtime(
                &self,
                x: &FeatureFrame,
                lease: &CpuLease,
            ) -> Result<Vec<RuntimePrediction>> {
                self.inner.predict_runtime(x, lease)
            }

            /// Read-only view of the trained feature column names +
            /// ordering — proxies to the inner `BurnDeepExpert`.
            /// Required by the inference-side `ExpertModel` adapter.
            pub fn feature_columns(&self) -> &[String] {
                self.inner.feature_columns()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new(42, None)
            }
        }

        impl ExpertModel for $name {
            fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
                self.inner.fit(x, y, lease)
            }

            fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
                self.inner.predict_proba(x, lease)
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
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};

    /// Genuine HIP training/inference/record reload, never a host-only pass.
    /// Execute explicitly on the later AMD acceptance host, not in host subsets.
    #[cfg(feature = "burn-rocm-backend")]
    #[test]
    fn burn_rocm_real_mlp_three_epoch_reload_and_cross_thread_cleanup() -> Result<()> {
        use burn_fusion::{inspect::FusionInspector, stream::StreamId};
        assert_eq!(
            std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
            Ok("1"),
            "this test requires explicit real-device acceptance, not a skipped GPU test"
        );
        let inspector = FusionInspector::install(StreamId::current());
        let device = burn_rocm::RocmDevice::new(0);
        <InferBackend as burn::tensor::backend::Backend>::sync(&device)
            .map_err(|error| anyhow::anyhow!("ROCm baseline sync failed: {error:?}"))?;
        inspector.set_baseline();
        let frame = typed_frame(vec![
            ("rsi", (0..224).map(|i| (i % 17) as f64 / 17.0).collect()),
            (
                "atr",
                (0..224).map(|i| 1.0 + (i % 11) as f64 / 11.0).collect(),
            ),
        ])?;
        let train = frame.select_rows(&(0..160).collect::<Vec<_>>())?;
        let validation = frame.select_rows(&(160..224).collect::<Vec<_>>())?;
        let labels = (0..224).map(|i| [-1, 0, 1][i % 3]).collect::<Vec<_>>();
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            17,
            Some(HashMap::from([
                ("device".into(), "rocm:0".into()),
                ("max_epochs".into(), "3".into()),
                ("patience".into(), "3".into()),
                ("batch_size".into(), "16".into()),
                ("hidden_dim".into(), "16".into()),
                ("n_layers".into(), "2".into()),
                ("dropout".into(), "0".into()),
                ("training_precision".into(), "fp32".into()),
            ])),
        );
        let lease = one_worker_lease();
        expert.fit_with_validation(
            &train,
            &labels[..160],
            Some(&validation),
            Some(&labels[160..]),
            &lease,
        )?;
        let report = expert
            .burn_training_report
            .as_ref()
            .context("missing real ROCm training report")?;
        assert_eq!(report.epochs_ran, 3);
        assert_eq!(report.execution_backend, "rocm");
        assert_eq!(report.effective_device_policy, "rocm:0");
        assert_eq!(report.training_precision, "fp32");
        let before = expert.predict_proba(&validation, &lease)?;
        assert_eq!(before.dim(), (64, 3));
        for row in before.rows() {
            assert!(
                row.iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            );
            assert!((row.sum() - 1.0).abs() < 1e-5);
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("neoethos-rocm-mlp-{}-{nonce}", std::process::id()));
        assert!(!path.exists());
        expert.save(&path)?;
        let mut loaded = BurnDeepExpert::new(DeepModelKind::Mlp, 17, None);
        loaded.load(&path)?;
        let retained = std::sync::Arc::downgrade(loaded.rocm_residency.as_ref().unwrap());
        let after = std::thread::spawn(move || -> Result<Array2<f64>> {
            let values = loaded.predict_proba(&validation, &one_worker_lease())?;
            drop(loaded);
            Ok(values)
        })
        .join()
        .map_err(|_| anyhow::anyhow!("ROCm cross-thread model operation panicked"))??;
        assert!(retained.upgrade().is_none());
        assert_eq!(before.mapv(f64::to_bits), after.mapv(f64::to_bits));
        drop(expert);
        <InferBackend as burn::tensor::backend::Backend>::sync(&device)
            .map_err(|error| anyhow::anyhow!("ROCm terminal sync failed: {error:?}"))?;
        assert!(inspector.new_handles_since_baseline().is_empty());
        std::fs::remove_dir_all(path)?;
        Ok(())
    }

    #[test]
    fn mlp_capacity_parameter_estimate_matches_the_actual_module() -> Result<()> {
        let config = BurnMLPConfig::new(5).with_hidden_dim(7).with_n_layers(3);
        let model = config.init::<burn_ndarray::NdArray<f32>>(&Default::default());
        assert_eq!(mlp_parameter_count(5, 7, 3)?, model.num_params());
        assert_eq!(mlp_parameter_count(5, 7, 3)?, 220);
        Ok(())
    }

    #[test]
    fn mlp_capacity_charges_each_padded_tensor_in_tiny_models() -> Result<()> {
        // I=H=L=B=V=1: six parameter tensors, each256B; eight copies.
        // Three activation tensors, each256B; max(16 train,8 validation) copies.
        // The old logical formula was640B, less than parameters alone (1536B).
        assert_eq!(mlp_parameter_count(1, 1, 1)?, 10);
        assert_eq!(mlp_training_bytes(1, 1, 1, 1, 1, 256)?, 24_576);
        assert_eq!(mlp_training_bytes(1, 1, 1, 1, 1, 512)?, 49_152);
        assert!(mlp_admitted_width(1, 1, 1, 1, 1, 640, 1 << 20, 256, false).is_err());
        assert_eq!(
            mlp_admitted_width(1, 1, 1, 1, 1, 24_576, 1 << 20, 256, false)?,
            1
        );
        Ok(())
    }

    #[test]
    fn mlp_capacity_padded_estimate_is_monotone_across_alignment_boundaries() -> Result<()> {
        for alignment in [32, 256, 512] {
            let mut previous = 0;
            for width in 1..=130 {
                let current = mlp_training_bytes(7, width, 3, 17, 65, alignment)?;
                assert!(current >= previous, "width {width}, alignment {alignment}");
                previous = current;
            }
        }
        // Logical matrix17x3=204B, but its aligned physical extent is512B.
        assert!(mlp_admitted_width(1, 1, 1, 1, 17, 1 << 20, 300, 256, false).is_err());
        Ok(())
    }

    #[test]
    fn mlp_capacity_reserves_the_dataset_before_resolving_requested_width() -> Result<()> {
        let plan = BurnResidentDatasetPlanV1::checked(200, 3, Some(50), 17, 1 << 24, 256)?;
        let model_only =
            mlp_training_bytes(3, 37, 3, plan.batch_rows(), plan.validation_rows(), 256)?;
        let combined = model_only.checked_add(plan.peak_bytes()).unwrap();
        assert_eq!(plan.peak_bytes(), 8960);
        assert!(
            mlp_admitted_width(
                3,
                37,
                3,
                17,
                50,
                model_only - plan.peak_bytes(),
                1 << 24,
                256,
                false
            )
            .is_err()
        );
        assert_eq!(
            mlp_admitted_width(
                3,
                37,
                3,
                17,
                50,
                combined - plan.peak_bytes(),
                1 << 24,
                256,
                false
            )?,
            37
        );
        // More cache memory must not be silently funded by shrinking the
        // configured architecture, even when automatic growth is requested.
        assert!(mlp_admitted_width(3, 37, 3, 17, 50, model_only - 1, 1 << 24, 256, true).is_err());
        Ok(())
    }

    #[test]
    fn mlp_capacity_grows_with_memory_and_accounts_for_full_validation() -> Result<()> {
        const MIB: usize = 1024 * 1024;
        let small = mlp_admitted_width(128, 64, 3, 64, 1000, 256 * MIB, 1024 * MIB, 256, true)?;
        let large = mlp_admitted_width(128, 64, 3, 64, 1000, 1024 * MIB, 1024 * MIB, 256, true)?;
        assert!(large > small && small > 64);
        let longer_validation =
            mlp_admitted_width(128, 64, 3, 64, 10_000, 1024 * MIB, 1024 * MIB, 256, true)?;
        assert!(longer_validation < large);
        assert!(mlp_training_bytes(128, large, 3, 64, 1000, 256)? <= 1024 * MIB);
        assert!(mlp_training_bytes(128, large + 1, 3, 64, 1000, 256)? > 1024 * MIB);
        let low_fraction = mlp_fractional_width(64, large, 0.25)?;
        let high_fraction = mlp_fractional_width(64, large, 0.75)?;
        assert!(64 <= low_fraction && low_fraction < high_fraction && high_fraction <= large);
        assert!(mlp_fractional_width(64, large, 0.75)? > mlp_fractional_width(64, small, 0.75)?);
        Ok(())
    }

    #[test]
    fn mlp_capacity_preserves_fixed_dimensions_and_rejects_impossible_shapes() -> Result<()> {
        const MIB: usize = 1024 * 1024;
        for budget in [256 * MIB, 1024 * MIB] {
            assert_eq!(
                mlp_admitted_width(128, 37, 3, 64, 1000, budget, MIB, 256, false)?,
                37
            );
        }
        assert!(mlp_admitted_width(128, 64, 3, 64, 1000, 1, MIB, 256, true).is_err());
        assert!(mlp_admitted_width(128, 64, 3, 64, 1000, usize::MAX, 1, 256, true).is_err());
        assert!(
            mlp_admitted_width(
                usize::MAX,
                64,
                3,
                64,
                1000,
                usize::MAX,
                usize::MAX,
                256,
                true
            )
            .is_err()
        );
        assert!(mlp_parameter_count(128, 0, 3).is_err());
        assert!(mlp_parameter_count(128, 64, 0).is_err());
        assert_eq!(
            mlp_admitted_width(2, 10_000, 1, 1, 1, 16 * MIB, MIB, 256, false)?,
            10_000
        );
        for bad in ["0", "-1", "1.1", "NaN", "inf", "invalid"] {
            let expert = BurnDeepExpert::new(
                DeepModelKind::Mlp,
                1,
                Some(HashMap::from([("capacity_fraction".into(), bad.into())])),
            );
            assert!(expert.mlp_capacity_fraction().is_err());
        }
        Ok(())
    }

    #[test]
    fn mlp_capacity_saved_dimensions_reload_without_resizing() -> Result<()> {
        let config = DeepArtifactConfig {
            kind: DeepModelKind::Mlp,
            params: HashMap::from([
                ("capacity_mode".into(), "auto".into()),
                ("capacity_requested_hidden_dim".into(), "64".into()),
                ("hidden_dim".into(), "777".into()),
                ("n_layers".into(), "3".into()),
                ("capacity_fraction".into(), "0.375".into()),
            ]),
            burn_training_report: None,
            runtime_metadata: None,
        };
        let decoded: DeepArtifactConfig = serde_json::from_slice(&serde_json::to_vec(&config)?)?;
        let expert = BurnDeepExpert::new(decoded.kind, 1, Some(decoded.params));
        assert!(expert.automatic_mlp_capacity()?);
        assert_eq!(expert.mlp_config(128).hidden_dim, 777);
        assert_eq!(expert.mlp_config(128).n_layers, 3);
        assert_eq!(expert.mlp_capacity_fraction()?, 0.375);
        let legacy = BurnDeepExpert::new(DeepModelKind::Mlp, 1, None);
        assert!(!legacy.automatic_mlp_capacity()?);
        assert_eq!(legacy.mlp_config(128).hidden_dim, 256);
        Ok(())
    }

    fn one_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker is valid");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("isolated deep-model test lease")
    }

    fn typed_frame(columns: Vec<(&str, Vec<f64>)>) -> Result<FeatureFrame> {
        let rows = columns.first().map_or(0, |(_, values)| values.len());
        let columns = columns
            .into_iter()
            .map(|(name, values)| {
                FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; rows])
            })
            .collect::<Result<Vec<_>>>()?;
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
    }

    #[test]
    fn deep_backend_f32_adapter_preserves_shape_and_row_order() -> Result<()> {
        let frame = typed_frame(vec![("rsi", vec![0.25, 0.5]), ("atr", vec![1.25, 1.5])])?;
        let matrix = deep_backend_f32_matrix(&frame)?;

        assert_eq!(matrix.dim(), (2, 2));
        assert_eq!(matrix[(0, 0)].to_bits(), 0.25_f32.to_bits());
        assert_eq!(matrix[(0, 1)].to_bits(), 1.25_f32.to_bits());
        assert_eq!(matrix[(1, 0)].to_bits(), 0.5_f32.to_bits());
        assert_eq!(matrix[(1, 1)].to_bits(), 1.5_f32.to_bits());
        Ok(())
    }

    #[test]
    fn deep_backend_f32_adapter_rejects_nonzero_underflow() -> Result<()> {
        let below_smallest_f32_subnormal = f64::from(f32::from_bits(1)) / 4.0;
        assert_ne!(below_smallest_f32_subnormal, 0.0);
        let frame = typed_frame(vec![("rsi", vec![below_smallest_f32_subnormal])])?;
        let error = deep_backend_f32_matrix(&frame)
            .expect_err("non-zero f64 feature must not silently narrow to zero");
        assert!(error.to_string().contains("underflowed non-zero"));
        Ok(())
    }

    #[test]
    fn metadata_requires_training_summary() {
        let expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);

        let err = expert
            .metadata()
            .expect_err("missing training summary must fail");
        assert!(
            err.to_string()
                .contains("missing training summary metadata")
        );
    }

    #[test]
    fn metadata_uses_training_summary_and_feature_columns() -> Result<()> {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));

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
            TrainingSummaryMetadata::raw_for_validation(10, 7, 0, 2),
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
    fn validate_runtime_params_rejects_retired_wgpu_provenance() {
        let params = HashMap::from([
            ("requested_device_policy".to_string(), "gpu:0".to_string()),
            ("effective_device_policy".to_string(), "gpu:0".to_string()),
            (
                "execution_backend".to_string(),
                "wgpu_integrated_gpu".to_string(),
            ),
        ]);

        let error = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("retired WGPU runtime provenance must fail closed");
        assert!(error.to_string().contains("unsupported backend"));
    }

    #[test]
    fn validate_runtime_params_rejects_internally_incoherent_runtime_triplet() {
        let params = HashMap::from([
            ("requested_device_policy".to_string(), "cpu".to_string()),
            ("effective_device_policy".to_string(), "cpu".to_string()),
            ("execution_backend".to_string(), "cuda".to_string()),
        ]);

        let err = BurnDeepExpert::validate_runtime_params(&params)
            .expect_err("incoherent runtime triplet should fail");
        assert!(
            err.to_string()
                .contains("runtime params are internally inconsistent")
        );
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
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));

        let err = expert
            .ensure_runtime_state_ready()
            .expect_err("invalid dropout must fail");
        assert!(err.to_string().contains("dropout"));
    }

    #[test]
    fn ensure_runtime_state_ready_requires_runtime_device_metadata() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));

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
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));

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
            .insert("execution_backend".to_string(), "cuda".to_string());
        expert
            .params
            .insert("requested_device_policy".to_string(), "cuda:0".to_string());
        expert
            .params
            .insert("effective_device_policy".to_string(), "cpu".to_string());

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("cuda"));
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("deep_requested_device_unavailable")
        );
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("deep_runtime_model_missing")
        );
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
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("deep_runtime_training_report_missing")
        );
    }

    #[test]
    fn runtime_details_mark_missing_runtime_metadata() {
        let expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend, None);
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("deep_runtime_device_metadata_missing")
        );
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
            requested_policy: "gpu:0".to_string(),
            effective_policy: "gpu:0".to_string(),
            execution_backend: "cuda".to_string(),
        });

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("cuda"));
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
        let model = expert
            .init_runtime_model(2)
            .expect("runtime model should initialize");
        let live_backend = expert
            .resolve_runtime_infer_device()
            .expect("runtime device must resolve")
            .1
            .execution_backend;
        expert.model = Some(model);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));
        expert.persisted_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: live_backend.clone(),
        });
        expert.host_runtime_selection = Some(BurnDeviceSelection {
            requested_policy: "gpu:0".to_string(),
            effective_policy: "gpu:0".to_string(),
            execution_backend: "cuda".to_string(),
        });

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some(live_backend.as_str()));
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("deep_runtime_host_cache_stale")
        );
    }

    #[test]
    fn fit_persists_effective_burn_device_metadata() -> Result<()> {
        let rsi = (0..140)
            .map(|idx| 0.1_f64 + idx as f64 * 0.01)
            .collect::<Vec<_>>();
        let atr = (0..140)
            .map(|idx| 1.0_f64 + idx as f64 * 0.01)
            .collect::<Vec<_>>();
        let labels = (0..140)
            .map(|idx| match idx % 3 {
                0 => 0_i32,
                1 => 1_i32,
                _ => -1_i32,
            })
            .collect::<Vec<_>>();
        let frame = typed_frame(vec![("rsi", rsi), ("atr", atr)])?;
        let lease = one_worker_lease();
        let mut expert = BurnDeepExpert::new(
            DeepModelKind::Mlp,
            7,
            Some(HashMap::from([
                ("device".to_string(), "cpu".to_string()),
                ("max_epochs".to_string(), "2".to_string()),
                ("batch_size".to_string(), "4".to_string()),
            ])),
        );
        expert.fit(&frame, &labels, &lease)?;

        assert_eq!(
            expert
                .params
                .get("requested_device_policy")
                .map(String::as_str),
            Some("cpu")
        );
        assert!(expert.params.contains_key("effective_device_policy"));
        assert!(expert.params.contains_key("execution_backend"));
        assert!(expert.params.contains_key("training_precision"));
        assert!(expert.persisted_runtime_selection.is_some());
        assert!(expert.host_runtime_selection.is_some());
        Ok(())
    }

    #[test]
    fn artifact_config_persists_burn_training_report() -> Result<()> {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 5, 15));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 100,
            train_rows: 80,
            val_rows: 15,
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
            training_precision: "fp32".to_string(),
            training_precision_reason: None,
        });

        let artifact = expert.artifact_config()?;
        assert!(artifact.burn_training_report.is_some());
        assert!(artifact.runtime_metadata.is_some());
        Ok(())
    }

    #[test]
    fn resolve_loaded_metadata_uses_config_runtime_metadata_when_sidecar_missing() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("deep-metadata-fallback-{nonce}"));
        std::fs::create_dir_all(&artifact_dir)?;

        let runtime_metadata = RuntimeArtifactMetadata::new(
            "mlp",
            ModelFamily::Deep,
            CapabilityState::Implemented,
            vec!["rsi".to_string(), "atr".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(100, 80, 0, 20),
        );
        let config = DeepArtifactConfig {
            kind: DeepModelKind::Mlp,
            params: HashMap::new(),
            burn_training_report: None,
            runtime_metadata: Some(runtime_metadata.clone()),
        };
        BurnDeepExpert::write_json(&BurnDeepExpert::config_path(&artifact_dir), &config)?;
        let loaded_config: DeepArtifactConfig =
            BurnDeepExpert::read_json(&BurnDeepExpert::config_path(&artifact_dir))?;

        let resolved =
            BurnDeepExpert::resolve_loaded_metadata(&artifact_dir, &loaded_config, "mlp")?;
        assert_eq!(resolved.feature_columns, runtime_metadata.feature_columns);
        assert_eq!(
            resolved.training_summary.dataset_rows,
            runtime_metadata.training_summary.dataset_rows
        );
        assert_eq!(
            resolved.training_summary.train_rows,
            runtime_metadata.training_summary.train_rows
        );
        assert_eq!(
            resolved.training_summary.val_rows,
            runtime_metadata.training_summary.val_rows
        );

        std::fs::remove_dir_all(&artifact_dir)?;
        Ok(())
    }

    #[test]
    fn resolve_loaded_metadata_rejects_sidecar_embedded_mismatch() -> Result<()> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("deep-metadata-mismatch-{nonce}"));
        std::fs::create_dir_all(&artifact_dir)?;

        let sidecar = RuntimeArtifactMetadata::new(
            "mlp",
            ModelFamily::Deep,
            CapabilityState::Implemented,
            vec!["rsi".to_string(), "atr".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(100, 80, 0, 20),
        );
        let embedded = RuntimeArtifactMetadata::new(
            "mlp",
            ModelFamily::Deep,
            CapabilityState::Implemented,
            vec!["rsi".to_string(), "atr".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(100, 81, 0, 19),
        );
        let config = DeepArtifactConfig {
            kind: DeepModelKind::Mlp,
            params: HashMap::new(),
            burn_training_report: None,
            runtime_metadata: Some(embedded),
        };
        BurnDeepExpert::write_json(&BurnDeepExpert::metadata_path(&artifact_dir), &sidecar)?;
        BurnDeepExpert::write_json(&BurnDeepExpert::config_path(&artifact_dir), &config)?;
        let loaded_config: DeepArtifactConfig =
            BurnDeepExpert::read_json(&BurnDeepExpert::config_path(&artifact_dir))?;

        let err = BurnDeepExpert::resolve_loaded_metadata(&artifact_dir, &loaded_config, "mlp")
            .expect_err("mismatched sidecar/embedded metadata should fail");
        assert!(err.to_string().contains("metadata sidecar mismatch"));

        std::fs::remove_dir_all(&artifact_dir)?;
        Ok(())
    }

    #[test]
    fn artifact_config_rejects_burn_training_report_row_drift() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 5, 15));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 101,
            train_rows: 81,
            val_rows: 15,
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
            training_precision: "fp32".to_string(),
            training_precision_reason: None,
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
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 5, 15));
        expert.burn_training_report = Some(BurnTrainingReport {
            dataset_rows: 100,
            train_rows: 80,
            val_rows: 15,
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
            execution_backend: "cuda".to_string(),
            training_precision: "fp32".to_string(),
            training_precision_reason: None,
        });

        let err = expert
            .artifact_config()
            .expect_err("runtime-incoherent burn report should fail");
        assert!(
            err.to_string()
                .contains("runtime provenance is internally inconsistent")
        );
    }

    #[test]
    fn artifact_config_requires_burn_training_report() {
        let mut expert = BurnDeepExpert::new(DeepModelKind::Mlp, 7, None);
        expert.feature_columns = vec!["rsi".to_string(), "atr".to_string()];
        expert.training_summary = Some(TrainingSummaryMetadata::new(100, 80, 0, 20));

        let err = expert
            .artifact_config()
            .expect_err("missing burn training report should fail");
        assert!(err.to_string().contains("Burn training report"));
    }
}
