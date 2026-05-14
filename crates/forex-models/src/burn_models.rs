// Production Burn Neural Network Models
//
// Pure-Rust deep learning models using Burn 0.20.
// Default backend is NdArray CPU, with an optional pure-Rust WGPU lane.
// Replaces legacy models (deep.py, mlp.py) — no legacy, no GIL.
//
// Production features matching legacy:
// - Class-weighted loss for imbalanced data
// - Index-order-aware train/val split with embargo
//   The caller must pass rows already sorted chronologically.
// - Early stopping with configurable patience
// - Mini-batch training with shuffling
// - Label protocol mapping (-1 → 2)

use burn::backend::Autodiff;
use burn::module::{Module, ModuleMapper, Param};
use burn::nn;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, FloatDType, TensorData};
#[cfg(not(feature = "burn-wgpu-backend"))]
use burn_ndarray::NdArray;
#[cfg(feature = "burn-wgpu-backend")]
use burn_wgpu::{Wgpu, WgpuDevice, graphics, init_setup};

use crate::hardware::HardwareInfo;
use crate::runtime::capabilities::{
    normalize_training_precision_policy, requested_training_precision_policy,
};
use anyhow::Context;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
#[cfg(feature = "burn-wgpu-backend")]
use std::collections::HashSet;
#[cfg(feature = "burn-wgpu-backend")]
use std::sync::{Mutex, OnceLock};
use tracing::info;

/// Backend types
#[cfg(feature = "burn-wgpu-backend")]
pub type TrainBackend = Autodiff<Wgpu>;
#[cfg(feature = "burn-wgpu-backend")]
pub type InferBackend = Wgpu;
#[cfg(not(feature = "burn-wgpu-backend"))]
pub type TrainBackend = Autodiff<NdArray>;
#[cfg(not(feature = "burn-wgpu-backend"))]
pub type InferBackend = NdArray;

#[cfg(feature = "burn-wgpu-backend")]
fn initialize_wgpu_runtime(device: &<InferBackend as Backend>::Device, policy_key: &str) {
    static INIT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let initialized = INIT.get_or_init(|| Mutex::new(HashSet::new()));
    let mut initialized = initialized.lock().expect("wgpu init cache poisoned");
    let init_key = format!("{policy_key}::{device:?}");
    if initialized.insert(init_key) {
        init_setup::<graphics::Vulkan>(device, Default::default());
    }
}

pub fn active_burn_backend_name() -> &'static str {
    #[cfg(feature = "burn-wgpu-backend")]
    {
        "wgpu"
    }
    #[cfg(not(feature = "burn-wgpu-backend"))]
    {
        "ndarray_cpu"
    }
}

pub fn selection_execution_backend(selection: &BurnDeviceSelection) -> &str {
    selection.execution_backend.as_str()
}

fn backend_name_for_type<B: Backend>() -> String {
    let backend_type = std::any::type_name::<B>().to_ascii_lowercase();
    if backend_type.contains("wgpu") {
        "wgpu".to_string()
    } else if backend_type.contains("ndarray") {
        "ndarray_cpu".to_string()
    } else {
        active_burn_backend_name().to_string()
    }
}

fn external_execution_backend_for<B: Backend>() -> String {
    format!("external:{}", backend_name_for_type::<B>())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnDeviceSelection {
    pub requested_policy: String,
    pub effective_policy: String,
    pub execution_backend: String,
}

pub fn normalize_burn_device_policy(policy: &str) -> String {
    // Burn-specific synonym: `"default"` is the burn ecosystem's term
    // for "let the framework pick"; map it to our `"auto"` canonical
    // token before delegating to the shared helper.
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized == "default" {
        return "auto".to_string();
    }
    // Burn also accepts the `wgpu` / `wgpu:N` form because the burn-wgpu
    // backend exists; statistical / runtime / rl callers do not, so it
    // goes in as an `extra_prefixes` token rather than into the shared
    // vendor list. See Batch 9 consolidation (docs/audits/research/
    // gpu_consolidation_audit.md) for the contract.
    crate::common::normalize_vendor_device_policy(&normalized, &["wgpu"])
}

fn is_supported_burn_device_policy(normalized: &str) -> bool {
    matches!(
        normalized,
        "auto"
            | "cpu"
            | "gpu"
            | "cuda"
            | "wgpu"
            | "default"
            | "rocm"
            | "metal"
            | "vulkan"
            | "external_device"
    ) || normalized.starts_with("cuda:")
        || normalized.starts_with("gpu:")
        || normalized.starts_with("wgpu:")
        || normalized.starts_with("rocm:")
        || normalized.starts_with("metal:")
        || normalized.starts_with("vulkan:")
}

fn normalize_effective_burn_device_policy(policy: &str) -> String {
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized == "external_device" {
        normalized
    } else {
        normalize_burn_device_policy(&normalized)
    }
}

pub(crate) fn validate_burn_device_selection(
    selection: &BurnDeviceSelection,
) -> anyhow::Result<()> {
    let requested = normalize_burn_device_policy(&selection.requested_policy);
    let effective = normalize_effective_burn_device_policy(&selection.effective_policy);
    let execution_backend = selection.execution_backend.trim().to_ascii_lowercase();

    if !is_supported_burn_device_policy(&requested) {
        return Err(anyhow::anyhow!(
            "Burn runtime requested device policy is unsupported: {}",
            selection.requested_policy
        ));
    }

    let effective_supported = effective == "external_device"
        || effective == "cpu"
        || effective == "default"
        || effective == "gpu"
        || effective.starts_with("gpu:");
    if !effective_supported {
        return Err(anyhow::anyhow!(
            "Burn runtime effective device policy is unsupported: {}",
            selection.effective_policy
        ));
    }

    if execution_backend.is_empty() {
        return Err(anyhow::anyhow!(
            "Burn runtime execution backend must not be empty"
        ));
    }

    match execution_backend.as_str() {
        "ndarray_cpu" => {
            if effective != "cpu" && effective != "external_device" {
                return Err(anyhow::anyhow!(
                    "Burn runtime ndarray_cpu backend is incompatible with effective policy {effective}"
                ));
            }
        }
        "wgpu" => {
            if effective != "external_device" {
                return Err(anyhow::anyhow!(
                    "Burn runtime generic wgpu backend requires external_device effective policy, got {effective}"
                ));
            }
        }
        "wgpu_cpu" => {
            if effective != "cpu" && effective != "external_device" {
                return Err(anyhow::anyhow!(
                    "Burn runtime wgpu_cpu backend is incompatible with effective policy {effective}"
                ));
            }
        }
        "wgpu_default" => {
            if effective != "default" && effective != "external_device" {
                return Err(anyhow::anyhow!(
                    "Burn runtime wgpu_default backend is incompatible with effective policy {effective}"
                ));
            }
        }
        "wgpu_discrete_gpu" | "wgpu_integrated_gpu" => {
            if effective != "external_device"
                && effective != "gpu"
                && !effective.starts_with("gpu:")
            {
                return Err(anyhow::anyhow!(
                    "Burn runtime GPU backend {execution_backend} is incompatible with effective policy {effective}"
                ));
            }
        }
        other if other.starts_with("external:") => {
            if effective != "external_device" {
                return Err(anyhow::anyhow!(
                    "Burn runtime external backend {other} requires external_device effective policy, got {effective}"
                ));
            }
        }
        other => {
            return Err(anyhow::anyhow!(
                "Burn runtime execution backend is unsupported: {other}"
            ));
        }
    }

    Ok(())
}

#[cfg(feature = "burn-wgpu-backend")]
fn parse_wgpu_gpu_index(normalized: &str) -> Option<usize> {
    normalized
        .strip_prefix("cuda:")
        .or_else(|| normalized.strip_prefix("gpu:"))
        .or_else(|| normalized.strip_prefix("wgpu:"))
        .or_else(|| normalized.strip_prefix("rocm:"))
        .or_else(|| normalized.strip_prefix("metal:"))
        .or_else(|| normalized.strip_prefix("vulkan:"))
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(feature = "burn-wgpu-backend")]
fn resolve_wgpu_device_policy(normalized: &str) -> (WgpuDevice, String, String) {
    match normalized {
        "cpu" => (WgpuDevice::Cpu, "cpu".to_string(), "wgpu_cpu".to_string()),
        "auto" | "gpu" | "cuda" | "wgpu" | "default" | "rocm" | "metal" | "vulkan" => (
            WgpuDevice::DefaultDevice,
            "default".to_string(),
            "wgpu_default".to_string(),
        ),
        other => {
            if let Some(index) = parse_wgpu_gpu_index(other) {
                (
                    WgpuDevice::DiscreteGpu(index),
                    format!("gpu:{index}"),
                    "wgpu_discrete_gpu".to_string(),
                )
            } else {
                (
                    WgpuDevice::DefaultDevice,
                    "default".to_string(),
                    "wgpu_default".to_string(),
                )
            }
        }
    }
}

pub fn resolve_infer_device(
    policy: &str,
) -> (<InferBackend as Backend>::Device, BurnDeviceSelection) {
    let requested_policy = normalize_burn_device_policy(policy);
    #[cfg(feature = "burn-wgpu-backend")]
    {
        let (device, effective_policy, execution_backend) =
            resolve_wgpu_device_policy(&requested_policy);
        initialize_wgpu_runtime(&device, &effective_policy);
        return (
            device,
            BurnDeviceSelection {
                requested_policy,
                effective_policy,
                execution_backend,
            },
        );
    }
    #[cfg(not(feature = "burn-wgpu-backend"))]
    {
        (
            <InferBackend as Backend>::Device::default(),
            BurnDeviceSelection {
                requested_policy,
                effective_policy: "cpu".to_string(),
                execution_backend: "ndarray_cpu".to_string(),
            },
        )
    }
}

pub fn resolve_train_device(
    policy: &str,
) -> (<TrainBackend as Backend>::Device, BurnDeviceSelection) {
    let requested_policy = normalize_burn_device_policy(policy);
    #[cfg(feature = "burn-wgpu-backend")]
    {
        let (device, effective_policy, execution_backend) =
            resolve_wgpu_device_policy(&requested_policy);
        initialize_wgpu_runtime(&device, &effective_policy);
        return (
            device,
            BurnDeviceSelection {
                requested_policy,
                effective_policy,
                execution_backend,
            },
        );
    }
    #[cfg(not(feature = "burn-wgpu-backend"))]
    {
        (
            <TrainBackend as Backend>::Device::default(),
            BurnDeviceSelection {
                requested_policy,
                effective_policy: "cpu".to_string(),
                execution_backend: "ndarray_cpu".to_string(),
            },
        )
    }
}

pub fn default_infer_device() -> <InferBackend as Backend>::Device {
    resolve_infer_device("auto").0
}

pub fn default_train_device() -> <TrainBackend as Backend>::Device {
    resolve_train_device("auto").0
}

fn parse_accelerator_index(policy: &str) -> Option<usize> {
    policy
        .strip_prefix("cuda:")
        .or_else(|| policy.strip_prefix("gpu:"))
        .or_else(|| policy.strip_prefix("wgpu:"))
        .or_else(|| policy.strip_prefix("rocm:"))
        .or_else(|| policy.strip_prefix("metal:"))
        .or_else(|| policy.strip_prefix("vulkan:"))
        .and_then(|value| value.parse::<usize>().ok())
}

pub trait ManagedBurnBackend: Backend {
    fn managed_device_and_selection() -> (Self::Device, BurnDeviceSelection);

    fn managed_device() -> Self::Device {
        Self::managed_device_and_selection().0
    }

    fn managed_selection() -> BurnDeviceSelection {
        Self::managed_device_and_selection().1
    }
}

impl ManagedBurnBackend for TrainBackend {
    fn managed_device_and_selection() -> (Self::Device, BurnDeviceSelection) {
        resolve_train_device("auto")
    }
}

impl ManagedBurnBackend for InferBackend {
    fn managed_device_and_selection() -> (Self::Device, BurnDeviceSelection) {
        resolve_infer_device("auto")
    }
}

// ============================================================================
// SHARED UTILITIES — matching legacy base.py
// ============================================================================

/// Map labels from {-1, 0, 1} to {2, 0, 1} matching legacy protocol.
fn map_labels(y: &[i32]) -> anyhow::Result<Vec<i64>> {
    y.iter()
        .map(|&v| match v {
            -1 => Ok(2i64),
            0 => Ok(0i64),
            1 => Ok(1i64),
            other => Err(anyhow::anyhow!(
                "Burn models only support labels in {{-1, 0, 1}}, received {other}"
            )),
        })
        .collect()
}

/// Compute class weights (inverse frequency) matching legacy compute_class_weights().
fn compute_class_weights(y: &[i64], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &label in y {
        let idx = label as usize;
        if idx < n_classes {
            counts[idx] += 1;
        }
    }
    let total = y.len() as f32;
    counts
        .iter()
        .map(|&c| {
            if c == 0 {
                1.0
            } else {
                total / (n_classes as f32 * c as f32)
            }
        })
        .collect()
}

fn scalar_loss_value(values: burn::tensor::TensorData, context: &str) -> anyhow::Result<f32> {
    let values = values
        .to_vec::<f32>()
        .with_context(|| context.to_string())?;
    let Some(value) = values.first().copied() else {
        return Err(anyhow::anyhow!(
            "{context}: Burn scalar tensor did not contain any values"
        ));
    };
    if !value.is_finite() {
        return Err(anyhow::anyhow!(
            "{context}: Burn scalar tensor contained non-finite value {value}"
        ));
    }
    Ok(value)
}

fn float_dtype(dtype: DType) -> FloatDType {
    match dtype {
        DType::BF16 => FloatDType::BF16,
        DType::F16 => FloatDType::F16,
        DType::Flex32 => FloatDType::Flex32,
        _ => FloatDType::F32,
    }
}

fn cast_tensor_to_dtype<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dtype: DType,
) -> Tensor<B, D> {
    if tensor.dtype() == dtype {
        tensor
    } else {
        tensor.cast(float_dtype(dtype))
    }
}

pub(crate) fn cast_module_float_tensors<B: Backend, M: Module<B>>(module: M, dtype: DType) -> M {
    struct FloatTensorDTypeMapper {
        dtype: DType,
    }

    impl<B: Backend> ModuleMapper<B> for FloatTensorDTypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            let tensor = cast_tensor_to_dtype(tensor, self.dtype);
            Param::from_mapped_value(id, tensor, mapper)
        }
    }

    let mut mapper = FloatTensorDTypeMapper { dtype };
    module.map(&mut mapper)
}

/// Index-order-aware train/val split with embargo gap.
/// The caller must provide rows in chronological order; this helper does not
/// infer or validate timestamps.
fn time_series_split(
    n_samples: usize,
    val_ratio: f32,
    min_train: usize,
    embargo: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let val_size = ((n_samples as f32) * val_ratio).ceil() as usize;
    let val_size = val_size
        .max(1)
        .min(n_samples.saturating_sub(min_train + embargo));
    let train_end = n_samples
        .saturating_sub(val_size + embargo)
        .max(min_train.min(n_samples));
    let val_start = (train_end + embargo).min(n_samples);
    (0..train_end, val_start..n_samples)
}

/// Early stopping tracker matching legacy EarlyStopper.
struct EarlyStopper {
    patience: usize,
    min_delta: f32,
    counter: usize,
    best_loss: Option<f32>,
}

impl EarlyStopper {
    fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: None,
        }
    }
    fn check(&mut self, val_loss: f32) -> bool {
        match self.best_loss {
            None => {
                self.best_loss = Some(val_loss);
                false
            }
            Some(best) if val_loss > best - self.min_delta => {
                self.counter += 1;
                self.counter >= self.patience
            }
            _ => {
                self.best_loss = Some(val_loss);
                self.counter = 0;
                false
            }
        }
    }
}

/// Convert ndarray::Array2<f32> to Burn Tensor<B, 2> with the requested runtime dtype.
fn array2_to_tensor_with_dtype<B: Backend>(
    data: &Array2<f32>,
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, 2> {
    let (rows, cols) = (data.nrows(), data.ncols());
    let flat: Vec<f32> = data.iter().copied().collect();
    Tensor::from_data_dtype(TensorData::new(flat, [rows, cols]), device, dtype)
}

/// Convert ndarray::Array2<f32> to Burn Tensor<B, 2>
fn array2_to_tensor<B: Backend>(data: &Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    array2_to_tensor_with_dtype(data, device, DType::F32)
}

/// Convert i64 labels to Burn Int Tensor<B, 1>
fn labels_to_tensor<B: Backend>(labels: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::from_data(TensorData::new(labels.to_vec(), [labels.len()]), device)
}

// ============================================================================
// BURN MLP — matches legacy MLPExpert (mlp.py)
// Uses LayerNorm instead of BatchNorm to avoid 3D reshape complications
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnMLP<B: Backend> {
    layers: Vec<nn::Linear<B>>,
    norms: Vec<nn::LayerNorm<B>>,
    dropout: nn::Dropout,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnMLPConfig {
    pub input_dim: usize,
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnMLP<B> {
        let mut layers = Vec::new();
        let mut norms = Vec::new();
        let mut dim = self.input_dim;
        for _ in 0..self.n_layers.max(1) {
            layers.push(nn::LinearConfig::new(dim, self.hidden_dim).init(device));
            norms.push(nn::LayerNormConfig::new(self.hidden_dim).init(device));
            dim = self.hidden_dim;
        }
        BurnMLP {
            layers,
            norms,
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnMLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for (linear, norm) in self.layers.iter().zip(self.norms.iter()) {
            h = linear.forward(h);
            h = norm.forward(h);
            h = burn::tensor::activation::relu(h);
            h = self.dropout.forward(h);
        }
        self.output.forward(h)
    }
}

// ============================================================================
// BURN N-BEATS — matches legacy NBeatsExpert (deep.py)
// ============================================================================

#[derive(Module, Debug)]
pub struct NBeatsBlock<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    fc4: nn::Linear<B>,
    theta_b: nn::Linear<B>,
    theta_f: nn::Linear<B>,
}

impl<B: Backend> NBeatsBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = burn::tensor::activation::relu(self.fc1.forward(x));
        let h = burn::tensor::activation::relu(self.fc2.forward(h));
        let h = burn::tensor::activation::relu(self.fc3.forward(h));
        let h = burn::tensor::activation::relu(self.fc4.forward(h));
        (self.theta_b.forward(h.clone()), self.theta_f.forward(h))
    }
}

#[derive(Module, Debug)]
pub struct BurnNBeats<B: Backend> {
    embed: nn::Linear<B>,
    blocks: Vec<NBeatsBlock<B>>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnNBeatsConfig {
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_blocks: usize,
    #[config(default = 3)]
    pub n_classes: usize,
}

impl BurnNBeatsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnNBeats<B> {
        let embed = nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device);
        let blocks = (0..self.n_blocks)
            .map(|_| NBeatsBlock {
                fc1: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc3: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc4: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                theta_b: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
                theta_f: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
            })
            .collect();
        let output = nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device);
        BurnNBeats {
            embed,
            blocks,
            output,
        }
    }
}

impl<B: Backend> BurnNBeats<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut residual = self.embed.forward(x);
        let mut forecast = Tensor::zeros(residual.dims(), &residual.device());
        for block in &self.blocks {
            let (backcast, fore) = block.forward(residual.clone());
            residual = residual - backcast;
            forecast = forecast + fore;
        }
        self.output.forward(forecast)
    }
}

// ============================================================================
// BURN N-BEATSx-NF — dedicated N-BEATS variant with exogenous gating
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnNBeatsx<B: Backend> {
    input_norm: nn::LayerNorm<B>,
    embed: nn::Linear<B>,
    blocks: Vec<NBeatsBlock<B>>,
    exogenous_gate: nn::Linear<B>,
    skip_proj: nn::Linear<B>,
    fusion_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnNBeatsxConfig {
    pub input_dim: usize,
    #[config(default = 96)]
    pub hidden_dim: usize,
    #[config(default = 4)]
    pub n_blocks: usize,
    #[config(default = 3)]
    pub n_classes: usize,
}

impl BurnNBeatsxConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnNBeatsx<B> {
        let embed = nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device);
        let blocks = (0..self.n_blocks)
            .map(|_| NBeatsBlock {
                fc1: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc3: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc4: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                theta_b: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
                theta_f: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
            })
            .collect();

        BurnNBeatsx {
            input_norm: nn::LayerNormConfig::new(self.input_dim).init(device),
            embed,
            blocks,
            exogenous_gate: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            skip_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fusion_norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnNBeatsx<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let normalized = self.input_norm.forward(x.clone());
        let skip = burn::tensor::activation::gelu(self.skip_proj.forward(normalized.clone()));
        let gate =
            burn::tensor::activation::sigmoid(self.exogenous_gate.forward(normalized.clone()));

        let mut residual = self.embed.forward(normalized) * gate.clone() + skip.clone();
        let mut forecast = Tensor::zeros(residual.dims(), &residual.device());
        for block in &self.blocks {
            let (backcast, fore) = block.forward(residual.clone());
            residual = residual - backcast * gate.clone();
            forecast = forecast + fore;
        }

        let fused = self.fusion_norm.forward(forecast + skip);
        self.output.forward(fused)
    }
}

// ============================================================================
// BURN TiDE — residual dense encoder-decoder for tabular time-series
// ============================================================================

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    fc: nn::Linear<B>,
    norm: nn::LayerNorm<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> ResidualBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = burn::tensor::activation::relu(self.fc.forward(x.clone()));
        self.norm.forward(x + self.dropout.forward(out))
    }
}

#[derive(Module, Debug)]
pub struct BurnTiDE<B: Backend> {
    feature_proj: nn::Linear<B>,
    enc1: ResidualBlock<B>,
    enc2: ResidualBlock<B>,
    temporal_link: nn::Linear<B>,
    dec1: ResidualBlock<B>,
    dec2: ResidualBlock<B>,
    output: nn::Linear<B>,
    raw_skip: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnTiDEConfig {
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnTiDEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTiDE<B> {
        let mk_res = || ResidualBlock {
            fc: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
        };
        BurnTiDE {
            feature_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc1: mk_res(),
            enc2: mk_res(),
            temporal_link: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            dec1: mk_res(),
            dec2: mk_res(),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
            raw_skip: nn::LinearConfig::new(self.input_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnTiDE<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let skip = self.raw_skip.forward(x.clone());
        let h = self.feature_proj.forward(x);
        let h = self.enc1.forward(h);
        let h = self.enc2.forward(h);
        let h = self.temporal_link.forward(h);
        let h = self.dec1.forward(h);
        let h = self.dec2.forward(h);
        self.output.forward(h) + skip
    }
}

// ============================================================================
// BURN TiDE-NF — dedicated TiDE variant with seasonal/frequency gating
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnTiDENf<B: Backend> {
    feature_proj: nn::Linear<B>,
    seasonal_proj: nn::Linear<B>,
    context_gate: nn::Linear<B>,
    enc1: ResidualBlock<B>,
    enc2: ResidualBlock<B>,
    temporal_link: nn::Linear<B>,
    horizon_gate: nn::Linear<B>,
    dec1: ResidualBlock<B>,
    dec2: ResidualBlock<B>,
    output_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
    raw_skip: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnTiDENfConfig {
    pub input_dim: usize,
    #[config(default = 160)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.05)]
    pub dropout: f64,
}

impl BurnTiDENfConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTiDENf<B> {
        let mk_res = || ResidualBlock {
            fc: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
        };

        BurnTiDENf {
            feature_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            seasonal_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            context_gate: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc1: mk_res(),
            enc2: mk_res(),
            temporal_link: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            horizon_gate: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            dec1: mk_res(),
            dec2: mk_res(),
            output_norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
            raw_skip: nn::LinearConfig::new(self.input_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnTiDENf<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let skip = self.raw_skip.forward(x.clone());
        let base = self.feature_proj.forward(x.clone());
        let seasonal = burn::tensor::activation::gelu(self.seasonal_proj.forward(x.clone()));
        let gate = burn::tensor::activation::sigmoid(self.context_gate.forward(x));

        let h = self.enc1.forward(base * gate + seasonal.clone());
        let h = self.enc2.forward(h);
        let h = self.temporal_link.forward(h);
        let horizon =
            burn::tensor::activation::sigmoid(self.horizon_gate.forward(seasonal.clone()));
        let h = self.dec1.forward(h * horizon + seasonal.clone());
        let h = self.dec2.forward(h);
        let h = self.output_norm.forward(h + seasonal);
        self.output.forward(h) + skip
    }
}

// ============================================================================
// BURN TabNet — attentive feature selection with GLU decision steps
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnTabNet<B: Backend> {
    initial_norm: nn::LayerNorm<B>,
    feat_fc1: nn::Linear<B>,
    feat_fc2: nn::Linear<B>,
    attn_fc: nn::Linear<B>,
    attn_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
    n_steps: usize,
    hidden_dim: usize,
    input_dim: usize,
    relaxation_factor: f32,
}

#[derive(Config, Debug)]
pub struct BurnTabNetConfig {
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_steps: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 1.5)]
    pub relaxation_factor: f64,
}

impl BurnTabNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTabNet<B> {
        BurnTabNet {
            initial_norm: nn::LayerNormConfig::new(self.input_dim).init(device),
            feat_fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim * 2).init(device),
            feat_fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim * 2).init(device),
            attn_fc: nn::LinearConfig::new(self.hidden_dim, self.input_dim)
                .with_bias(false)
                .init(device),
            attn_norm: nn::LayerNormConfig::new(self.input_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
            n_steps: self.n_steps,
            hidden_dim: self.hidden_dim,
            input_dim: self.input_dim,
            relaxation_factor: self.relaxation_factor as f32,
        }
    }
}

/// GLU: split last dim in half, sigmoid-gate one half
fn glu<B: Backend>(x: Tensor<B, 2>, half: usize) -> Tensor<B, 2> {
    let a = x.clone().slice([0..x.dims()[0], 0..half]);
    let b = x.clone().slice([0..x.dims()[0], half..(half * 2)]);
    a * burn::tensor::activation::sigmoid(b)
}

impl<B: Backend> BurnTabNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let x_norm = self.initial_norm.forward(x.clone());
        let mut prior: Tensor<B, 2> = Tensor::ones([batch, self.input_dim], &x.device());
        let mut out_accum: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_dim], &x.device());

        for _ in 0..self.n_steps {
            // Feature transform with GLU
            let feat = glu(self.feat_fc1.forward(x_norm.clone()), self.hidden_dim);
            let feat = glu(self.feat_fc2.forward(feat), self.hidden_dim);

            // Attention mask
            let attn = self.attn_fc.forward(feat.clone());
            let attn = self.attn_norm.forward(attn * prior.clone());
            let mask = burn::tensor::activation::softmax(attn, 1);

            // Update prior (decay attention)
            let threshold =
                Tensor::<B, 2>::ones_like(&mask) * self.relaxation_factor - mask.clone();
            prior = prior * threshold.clamp(0.0, 1.0);

            // Masked feature → transform → accumulate
            let masked = x_norm.clone() * mask;
            let step_out = glu(self.feat_fc1.forward(masked), self.hidden_dim);
            let step_out = burn::tensor::activation::relu(glu(
                self.feat_fc2.forward(step_out),
                self.hidden_dim,
            ));
            out_accum = out_accum + step_out;
        }

        self.output.forward(out_accum)
    }
}

// ============================================================================
// BURN KAN — Kolmogorov-Arnold style edge functions using grid/RBF bases
// ============================================================================

const KAN_GRID_MIN: f32 = -3.0;
const KAN_GRID_MAX: f32 = 3.0;

#[derive(Module, Debug)]
pub struct KANLayer<B: Backend> {
    base: nn::Linear<B>,
    spline: nn::Linear<B>,
    gate: nn::Linear<B>,
    norm: nn::LayerNorm<B>,
    dropout: nn::Dropout,
    grid_size: usize,
}

fn kan_grid_center(index: usize, grid_size: usize) -> f32 {
    if grid_size <= 1 {
        0.0
    } else {
        let fraction = index as f32 / (grid_size - 1) as f32;
        KAN_GRID_MIN + (KAN_GRID_MAX - KAN_GRID_MIN) * fraction
    }
}

fn kan_grid_gamma(grid_size: usize) -> f32 {
    if grid_size <= 1 {
        1.0
    } else {
        let width = (KAN_GRID_MAX - KAN_GRID_MIN) / (grid_size - 1) as f32;
        1.0 / (width * width).max(1e-6)
    }
}

impl<B: Backend> KANLayer<B> {
    fn basis_expand(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let bounded = x.clamp(KAN_GRID_MIN, KAN_GRID_MAX);
        let gamma = kan_grid_gamma(self.grid_size);
        let mut basis = Vec::with_capacity(self.grid_size);
        for grid_idx in 0..self.grid_size {
            let center = kan_grid_center(grid_idx, self.grid_size);
            let radial = ((bounded.clone() - center).powi_scalar(2) * -gamma).exp();
            basis.push(radial);
        }
        Tensor::cat(basis, 1)
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let basis = self.basis_expand(x.clone());
        let base = burn::tensor::activation::silu(self.base.forward(x.clone()));
        let spline = self.spline.forward(basis);
        let gate = burn::tensor::activation::sigmoid(self.gate.forward(x));
        let h = base + spline * gate;
        burn::tensor::activation::gelu(self.dropout.forward(self.norm.forward(h)))
    }
}

#[derive(Module, Debug)]
pub struct BurnKAN<B: Backend> {
    layers: Vec<KANLayer<B>>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnKANConfig {
    pub input_dim: usize,
    #[config(default = 32)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 9)]
    pub grid_size: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.05)]
    pub dropout: f64,
}

impl BurnKANConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnKAN<B> {
        let hidden_dim = self.hidden_dim.max(4);
        let n_layers = self.n_layers.max(1);
        let grid_size = self.grid_size.clamp(3, 33);
        let mut layers = Vec::with_capacity(n_layers);
        let mut dim = self.input_dim;
        for _ in 0..n_layers {
            layers.push(KANLayer {
                base: nn::LinearConfig::new(dim, hidden_dim).init(device),
                spline: nn::LinearConfig::new(dim * grid_size, hidden_dim).init(device),
                gate: nn::LinearConfig::new(dim, hidden_dim).init(device),
                norm: nn::LayerNormConfig::new(hidden_dim).init(device),
                dropout: nn::DropoutConfig::new(self.dropout).init(),
                grid_size,
            });
            dim = hidden_dim;
        }
        BurnKAN {
            layers,
            output: nn::LinearConfig::new(hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnKAN<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h);
        }
        self.output.forward(h)
    }
}

// ============================================================================
// BURN TRANSFORMER — feature-token transformer for tabular time-series features
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnTransformer<B: Backend> {
    token_proj: nn::Linear<B>,
    encoder: Vec<SequenceTransformerBlock<B>>,
    final_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
    token_size: usize,
    token_count: usize,
    input_dim: usize,
}

#[derive(Config, Debug)]
pub struct BurnTransformerConfig {
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 8)]
    pub token_count: usize,
    #[config(default = 512)]
    pub dim_ff: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTransformer<B> {
        let hidden_dim = self.hidden_dim.max(16);
        let token_count = self.token_count.max(1).min(self.input_dim.max(1));
        let token_size = self.input_dim.div_ceil(token_count);
        let requested_heads = self.n_heads.max(1);
        let compatible_heads = (1..=requested_heads)
            .rev()
            .find(|candidate| hidden_dim.is_multiple_of(*candidate))
            .unwrap_or(1);
        let ff_dim = self.dim_ff.max(hidden_dim);
        let encoder = (0..self.n_layers)
            .map(|_| SequenceTransformerBlock {
                q_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                k_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                v_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                out_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                ff1: nn::LinearConfig::new(hidden_dim, ff_dim).init(device),
                ff2: nn::LinearConfig::new(ff_dim, hidden_dim).init(device),
                norm1: nn::LayerNormConfig::new(hidden_dim).init(device),
                norm2: nn::LayerNormConfig::new(hidden_dim).init(device),
                dropout: nn::DropoutConfig::new(self.dropout).init(),
                n_heads: compatible_heads,
            })
            .collect();
        BurnTransformer {
            token_proj: nn::LinearConfig::new(token_size, hidden_dim).init(device),
            encoder,
            final_norm: nn::LayerNormConfig::new(hidden_dim).init(device),
            output: nn::LinearConfig::new(hidden_dim, self.n_classes).init(device),
            token_size,
            token_count,
            input_dim: self.input_dim,
        }
    }
}

impl<B: Backend> BurnTransformer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let mut tokens = Vec::with_capacity(self.token_count);
        for token_idx in 0..self.token_count {
            let start = token_idx * self.token_size;
            let end = ((token_idx + 1) * self.token_size).min(self.input_dim);
            let mut token = x.clone().slice([0..batch, start..end]);
            let observed = end.saturating_sub(start);
            if observed < self.token_size {
                let padding =
                    Tensor::<B, 2>::zeros([batch, self.token_size - observed], &x.device());
                token = Tensor::cat(vec![token, padding], 1);
            }
            tokens.push(
                burn::tensor::activation::gelu(self.token_proj.forward(token))
                    .unsqueeze_dim::<3>(1),
            );
        }

        let mut h = Tensor::cat(tokens, 1);
        for block in &self.encoder {
            h = block.forward(h);
        }
        let pooled = h.mean_dim(1).squeeze_dim::<2>(1);
        let pooled = self.final_norm.forward(pooled);
        self.output.forward(pooled)
    }
}

#[derive(Module, Debug)]
pub struct SequenceTransformerBlock<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    out_proj: nn::Linear<B>,
    ff1: nn::Linear<B>,
    ff2: nn::Linear<B>,
    norm1: nn::LayerNorm<B>,
    norm2: nn::LayerNorm<B>,
    dropout: nn::Dropout,
    n_heads: usize,
}

impl<B: Backend> SequenceTransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        let n_heads = self.n_heads.max(1);
        let head_dim = d_model / n_heads;
        let flat_rows = batch * seq_len;
        let flat = x.clone().reshape([flat_rows, d_model]);

        let q = self
            .q_proj
            .forward(flat.clone())
            .reshape([batch, seq_len, n_heads, head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(flat.clone())
            .reshape([batch, seq_len, n_heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(flat)
            .reshape([batch, seq_len, n_heads, head_dim])
            .swap_dims(1, 2);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn = burn::tensor::activation::softmax(scores, 3);
        let attn_flat = attn.matmul(v).swap_dims(1, 2).reshape([flat_rows, d_model]);
        let attn_proj = self.dropout.forward(self.out_proj.forward(attn_flat));
        let h = self
            .norm1
            .forward(
                (x + attn_proj.reshape([batch, seq_len, d_model])).reshape([flat_rows, d_model]),
            )
            .reshape([batch, seq_len, d_model]);

        let h_flat = h.clone().reshape([flat_rows, d_model]);
        let ff = self.ff2.forward(
            self.dropout
                .forward(burn::tensor::activation::gelu(self.ff1.forward(h_flat))),
        );
        self.norm2
            .forward(
                (h + self.dropout.forward(ff).reshape([batch, seq_len, d_model]))
                    .reshape([flat_rows, d_model]),
            )
            .reshape([batch, seq_len, d_model])
    }
}

// ============================================================================
// BURN PatchTST — dedicated patch-based transformer for tabular sequences
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnPatchTST<B: Backend> {
    patch_proj: nn::Linear<B>,
    patch_encoder: Vec<SequenceTransformerBlock<B>>,
    merge_proj: nn::Linear<B>,
    skip_proj: nn::Linear<B>,
    head_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
    patch_size: usize,
    patch_count: usize,
    hidden_dim: usize,
    input_dim: usize,
}

#[derive(Config, Debug)]
pub struct BurnPatchTSTConfig {
    pub input_dim: usize,
    #[config(default = 192)]
    pub hidden_dim: usize,
    #[config(default = 8)]
    pub patch_size: usize,
    #[config(default = 6)]
    pub n_heads: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 384)]
    pub dim_ff: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnPatchTSTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnPatchTST<B> {
        let patch_size = self.patch_size.max(1);
        let patch_count = self.input_dim.div_ceil(patch_size);
        let hidden_dim = self.hidden_dim.max(16);
        let requested_heads = self.n_heads.max(1);
        let compatible_heads = (1..=requested_heads)
            .rev()
            .find(|candidate| hidden_dim.is_multiple_of(*candidate))
            .unwrap_or(1);
        let ff_dim = self.dim_ff.max(hidden_dim);
        let patch_encoder = (0..self.n_layers.max(1))
            .map(|_| SequenceTransformerBlock {
                q_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                k_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                v_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                out_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                ff1: nn::LinearConfig::new(hidden_dim, ff_dim).init(device),
                ff2: nn::LinearConfig::new(ff_dim, hidden_dim).init(device),
                norm1: nn::LayerNormConfig::new(hidden_dim).init(device),
                norm2: nn::LayerNormConfig::new(hidden_dim).init(device),
                dropout: nn::DropoutConfig::new(self.dropout).init(),
                n_heads: compatible_heads,
            })
            .collect();

        BurnPatchTST {
            patch_proj: nn::LinearConfig::new(patch_size, hidden_dim).init(device),
            patch_encoder,
            merge_proj: nn::LinearConfig::new(hidden_dim * patch_count, hidden_dim).init(device),
            skip_proj: nn::LinearConfig::new(self.input_dim, hidden_dim).init(device),
            head_norm: nn::LayerNormConfig::new(hidden_dim).init(device),
            output: nn::LinearConfig::new(hidden_dim, self.n_classes).init(device),
            patch_size,
            patch_count,
            hidden_dim,
            input_dim: self.input_dim,
        }
    }
}

impl<B: Backend> BurnPatchTST<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let mut tokens = Vec::with_capacity(self.patch_count);

        for patch_idx in 0..self.patch_count {
            let start = patch_idx * self.patch_size;
            let end = ((patch_idx + 1) * self.patch_size).min(self.input_dim);
            let mut patch = x.clone().slice([0..batch, start..end]);
            let observed = end.saturating_sub(start);
            if observed < self.patch_size {
                let padding =
                    Tensor::<B, 2>::zeros([batch, self.patch_size - observed], &x.device());
                patch = Tensor::cat(vec![patch, padding], 1);
            }

            let token = burn::tensor::activation::gelu(self.patch_proj.forward(patch));
            tokens.push(token.unsqueeze_dim::<3>(1));
        }

        let mut sequence = Tensor::cat(tokens, 1);
        for block in &self.patch_encoder {
            sequence = block.forward(sequence);
        }
        let merged = sequence.reshape([batch, self.patch_count * self.hidden_dim]);
        let fused = burn::tensor::activation::gelu(
            self.merge_proj.forward(merged) + self.skip_proj.forward(x),
        );
        let fused = self.head_norm.forward(fused);
        self.output.forward(fused)
    }
}

// ============================================================================
// BURN TimesNet — dedicated multi-period mixer
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnTimesNet<B: Backend> {
    input_proj: nn::Linear<B>,
    raw_period_projs: Vec<nn::Linear<B>>,
    period_mixers: Vec<nn::Linear<B>>,
    period_norms: Vec<nn::LayerNorm<B>>,
    period_weight_proj: nn::Linear<B>,
    gate_proj: nn::Linear<B>,
    fusion_norm: nn::LayerNorm<B>,
    decoder: Vec<ResidualBlock<B>>,
    output: nn::Linear<B>,
    hidden_dim: usize,
}

#[derive(Config, Debug)]
pub struct BurnTimesNetConfig {
    pub input_dim: usize,
    #[config(default = 192)]
    pub hidden_dim: usize,
    #[config(default = 4)]
    pub n_periods: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.05)]
    pub dropout: f64,
}

impl BurnTimesNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTimesNet<B> {
        let hidden_dim = self.hidden_dim.max(16);
        let n_periods = self.n_periods.max(2);
        let raw_period_projs = (0..n_periods)
            .map(|_| nn::LinearConfig::new(self.input_dim, hidden_dim).init(device))
            .collect();
        let period_mixers = (0..n_periods)
            .map(|_| nn::LinearConfig::new(hidden_dim, hidden_dim).init(device))
            .collect();
        let period_norms = (0..n_periods)
            .map(|_| nn::LayerNormConfig::new(hidden_dim).init(device))
            .collect();
        let decoder = (0..2)
            .map(|_| ResidualBlock {
                fc: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
                norm: nn::LayerNormConfig::new(hidden_dim).init(device),
                dropout: nn::DropoutConfig::new(self.dropout).init(),
            })
            .collect();

        BurnTimesNet {
            input_proj: nn::LinearConfig::new(self.input_dim, hidden_dim).init(device),
            raw_period_projs,
            period_mixers,
            period_norms,
            period_weight_proj: nn::LinearConfig::new(hidden_dim, n_periods).init(device),
            gate_proj: nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            fusion_norm: nn::LayerNormConfig::new(hidden_dim).init(device),
            decoder,
            output: nn::LinearConfig::new(hidden_dim, self.n_classes).init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> BurnTimesNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let base = burn::tensor::activation::gelu(self.input_proj.forward(x.clone()));
        let period_weights =
            burn::tensor::activation::softmax(self.period_weight_proj.forward(base.clone()), 1);
        let mut periodic: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_dim], &base.device());

        for (period_idx, ((raw_proj, mixer), norm)) in self
            .raw_period_projs
            .iter()
            .zip(self.period_mixers.iter())
            .zip(self.period_norms.iter())
            .enumerate()
        {
            let raw_period = raw_proj.forward(x.clone());
            let mixed = mixer.forward(base.clone());
            let branch = norm.forward(burn::tensor::activation::gelu(raw_period + mixed));
            let weight = period_weights
                .clone()
                .slice([0..batch, period_idx..(period_idx + 1)])
                .repeat_dim(1, self.hidden_dim);
            periodic = periodic + branch * weight;
        }

        let gate = burn::tensor::activation::sigmoid(self.gate_proj.forward(base.clone()));
        let mut h = self.fusion_norm.forward(base + periodic * gate);
        for block in &self.decoder {
            h = block.forward(h);
        }
        self.output.forward(h)
    }
}

// ============================================================================
// BURN EXPERT TRAIT — uniform interface for all models
// ============================================================================

/// Trait for all Burn models providing a consistent forward pass.
pub trait BurnForward<B: Backend> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}

impl<B: Backend> BurnForward<B> for BurnMLP<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnNBeats<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnNBeatsx<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnTiDE<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnTiDENf<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnTabNet<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnKAN<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnTransformer<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnPatchTST<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}
impl<B: Backend> BurnForward<B> for BurnTimesNet<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x)
    }
}

// ============================================================================
// TRAINING CONFIGURATION
// ============================================================================

/// Training configuration matching legacy deep.py defaults.
pub struct TrainConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub n_classes: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            batch_size: 64,
            max_epochs: 100,
            patience: 8,
            n_classes: 3,
            seed: 42,
        }
    }
}

/// Training report returned by the richer Burn runtime helper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnTrainingReport {
    pub dataset_rows: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    pub embargo_rows: usize,
    pub class_weights: Vec<f32>,
    pub best_loss: f32,
    pub best_epoch: Option<usize>,
    pub epochs_ran: usize,
    pub final_train_loss: f32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub patience: usize,
    pub seed: u64,
    pub requested_device_policy: String,
    pub effective_device_policy: String,
    pub execution_backend: String,
    pub training_precision: String,
    pub training_precision_reason: Option<String>,
}

impl BurnTrainingReport {
    pub fn best_observed_loss(&self) -> f32 {
        self.best_loss
    }
}

fn validate_train_config(config: &TrainConfig) -> anyhow::Result<()> {
    if !config.lr.is_finite() || config.lr <= 0.0 {
        return Err(anyhow::anyhow!(
            "Burn training learning rate must be finite and positive"
        ));
    }
    if config.batch_size == 0 {
        return Err(anyhow::anyhow!(
            "Burn training batch_size must be greater than zero"
        ));
    }
    if config.max_epochs == 0 {
        return Err(anyhow::anyhow!(
            "Burn training max_epochs must be greater than zero"
        ));
    }
    if config.patience == 0 {
        return Err(anyhow::anyhow!(
            "Burn training patience must be greater than zero"
        ));
    }
    if config.n_classes < 2 {
        return Err(anyhow::anyhow!(
            "Burn training requires at least two classes"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BurnExecutionPrecision {
    Fp32,
    Bf16,
}

impl BurnExecutionPrecision {
    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }

    fn dtype(self) -> DType {
        match self {
            Self::Fp32 => DType::F32,
            Self::Bf16 => DType::BF16,
        }
    }
}

fn requested_burn_training_precision(requested_precision: Option<&str>) -> String {
    requested_precision
        .map(normalize_training_precision_policy)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| requested_training_precision_policy("burn"))
}

fn resolve_burn_training_precision_for_backend<B: Backend>(
    selection: &BurnDeviceSelection,
    device: &B::Device,
    requested_precision: Option<&str>,
) -> (BurnExecutionPrecision, Option<String>) {
    fn env_flag(name: &str, default: bool) -> bool {
        match std::env::var(name) {
            Ok(value) => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Err(_) => default,
        }
    }

    let requested = requested_burn_training_precision(requested_precision);

    let gpu_requested = selection.effective_policy == "default"
        || selection.effective_policy == "gpu"
        || selection.effective_policy.starts_with("gpu:")
        || matches!(
            selection.execution_backend.as_str(),
            "wgpu_default" | "wgpu_discrete_gpu" | "wgpu_integrated_gpu"
        );
    let supports_bf16 = if gpu_requested {
        let hardware = HardwareInfo::detect();
        let accelerator_index = parse_accelerator_index(&selection.effective_policy).unwrap_or(0);
        hardware.gpu_supports_bf16(accelerator_index) && B::supports_dtype(device, DType::BF16)
    } else {
        B::supports_dtype(device, DType::BF16)
    };
    let model_supports_bf16 = env_flag("FOREX_BURN_MODEL_SUPPORTS_BF16", true);
    let bf16_supported = supports_bf16 && model_supports_bf16;

    match requested.as_str() {
        "bf16" if bf16_supported => (BurnExecutionPrecision::Bf16, None),
        "auto" if bf16_supported => (BurnExecutionPrecision::Bf16, None),
        "fp8" | "bf4" => {
            let selected = if bf16_supported {
                BurnExecutionPrecision::Bf16
            } else {
                BurnExecutionPrecision::Fp32
            };
            (
                selected,
                Some(format!(
                    "requested precision `{requested}`; Burn runtime currently executes training tensors as fp32/bf16 only, using `{}`",
                    selected.label()
                )),
            )
        }
        "bf16" => (
            BurnExecutionPrecision::Fp32,
            Some(format!(
                "requested precision `{requested}` unsupported by the active Burn backend/device/model implementation; using `fp32`"
            )),
        ),
        "auto" => (
            BurnExecutionPrecision::Fp32,
            Some(
                "auto precision fallback to fp32 for current Burn backend/device/model implementation"
                    .to_string(),
            ),
        ),
        _ => (BurnExecutionPrecision::Fp32, None),
    }
}

// ============================================================================
// CROSS-ENTROPY LOSS
// ============================================================================

/// Weighted cross-entropy loss matching legacy nn.CrossEntropyLoss(weight=...).
fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    class_weights: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let n_classes = class_weights.len();
    let batch_size = logits.dims()[0];
    let logits_dtype = logits.dtype();
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let one_hot = targets
        .one_hot(n_classes)
        .float()
        .cast(float_dtype(logits_dtype));
    let weights = Tensor::<B, 1>::from_data_dtype(
        TensorData::new(class_weights.to_vec(), [n_classes]),
        device,
        logits_dtype,
    );
    let weighted = log_probs * one_hot * weights.unsqueeze_dim(0);
    weighted.sum().neg() / (batch_size as f32)
}

fn normalize_probability_rows(
    mut probabilities: Vec<f32>,
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 || probabilities.len() != rows.saturating_mul(cols) {
        return None;
    }

    for row in probabilities.chunks_exact_mut(cols) {
        let mut sum = 0.0_f32;
        for value in row.iter_mut() {
            if !value.is_finite() {
                return None;
            }
            if *value < 0.0 {
                return None;
            }
            sum += *value;
        }
        if sum > f32::EPSILON {
            for value in row.iter_mut() {
                *value /= sum;
            }
        } else {
            return None;
        }
    }

    Some(probabilities)
}

fn validate_feature_matrix(x_data: &Array2<f32>, context: &str) -> anyhow::Result<()> {
    if x_data.ncols() == 0 {
        return Err(anyhow::anyhow!(
            "{context} requires at least one feature column"
        ));
    }
    if x_data.iter().any(|value| !value.is_finite()) {
        return Err(anyhow::anyhow!(
            "{context} received non-finite feature values"
        ));
    }
    Ok(())
}

// ============================================================================
// TRAINING LOOP — generic over all Burn models
// Matches legacy deep.py: class weights, time-series split, early stopping
// ============================================================================

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

/// Train any Burn model with production features.
///
/// Includes: time-series split with embargo, class-weighted loss,
/// early stopping, mini-batch, Adam optimizer, label protocol mapping.
pub fn train_model<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
) -> anyhow::Result<(M, f32)>
where
    B: AutodiffBackend + ManagedBurnBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    let (model, report) = train_model_with_report::<B, M>(model, x_data, y_raw, config)?;
    Ok((model, report.best_observed_loss()))
}

/// Train any Burn model and return a detailed runtime report alongside the model.
pub fn train_model_with_report<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
) -> anyhow::Result<(M, BurnTrainingReport)>
where
    B: AutodiffBackend + ManagedBurnBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    let (device, selection) = B::managed_device_and_selection();
    train_model_with_report_with_selection::<B, M>(
        model, x_data, y_raw, config, &device, &selection,
    )
}

pub fn train_model_with_report_on_device<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
    device: &B::Device,
    selection: &BurnDeviceSelection,
) -> anyhow::Result<(M, BurnTrainingReport)>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    train_model_with_report_with_selection::<B, M>(model, x_data, y_raw, config, device, selection)
}

pub fn train_model_with_report_with_selection<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
    device: &B::Device,
    selection: &BurnDeviceSelection,
) -> anyhow::Result<(M, BurnTrainingReport)>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    train_model_with_report_with_selection_and_precision::<B, M>(
        model, x_data, y_raw, config, device, selection, None,
    )
}

pub fn train_model_with_report_with_selection_and_precision<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
    device: &B::Device,
    selection: &BurnDeviceSelection,
    requested_precision: Option<&str>,
) -> anyhow::Result<(M, BurnTrainingReport)>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    train_model_with_report_with_external_val::<B, M>(
        model,
        x_data,
        y_raw,
        config,
        device,
        selection,
        requested_precision,
        None,
        None,
    )
}

/// M5: variant that accepts an explicit external validation frame from the
/// HPO orchestrator. When `external_val_x` and `external_val_y` are
/// supplied, Burn skips its internal `time_series_split` 15% holdout and
/// drives early stopping against the same val data the HPO objective uses.
/// This keeps train/val/early-stopping consistent across the model
/// pipeline so HPO scores reflect what the trained weights will actually
/// generalise to.
#[allow(clippy::too_many_arguments)]
pub fn train_model_with_report_with_external_val<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
    device: &B::Device,
    selection: &BurnDeviceSelection,
    requested_precision: Option<&str>,
    external_val_x: Option<&Array2<f32>>,
    external_val_y: Option<&[i32]>,
) -> anyhow::Result<(M, BurnTrainingReport)>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B> + Clone,
{
    validate_train_config(config)?;
    validate_burn_device_selection(selection)?;
    let n_samples = x_data.nrows();

    validate_feature_matrix(x_data, "Burn training")?;

    if n_samples != y_raw.len() {
        return Err(anyhow::anyhow!(
            "Burn training feature/label mismatch: {} rows vs {} labels",
            n_samples,
            y_raw.len()
        ));
    }

    let use_external_val = matches!((external_val_x, external_val_y), (Some(_), Some(_)));
    if let (Some(vx), Some(vy)) = (external_val_x, external_val_y) {
        validate_feature_matrix(vx, "Burn external validation")?;
        if vx.ncols() != x_data.ncols() {
            return Err(anyhow::anyhow!(
                "Burn external validation column mismatch: train {}, val {}",
                x_data.ncols(),
                vx.ncols()
            ));
        }
        if vx.nrows() != vy.len() {
            return Err(anyhow::anyhow!(
                "Burn external validation row/label mismatch: {} rows vs {} labels",
                vx.nrows(),
                vy.len()
            ));
        }
    }

    // 1. Label mapping: -1 → 2
    let y_mapped = map_labels(y_raw)?;
    let (training_precision, training_precision_reason) =
        resolve_burn_training_precision_for_backend::<B>(selection, device, requested_precision);
    let training_dtype = training_precision.dtype();

    // 2. Resolve train/val ranges. When the caller supplied an external val
    // frame, treat the entire `x_data` as training rows. Otherwise fall
    // back to the legacy index-order-aware split with embargo.
    let (train_range, val_range, embargo) = if use_external_val {
        (0..n_samples, 0..0, 0usize)
    } else {
        let embargo = ((n_samples as f32 * 0.005).ceil() as usize).max(10);
        let (train_range, val_range) = time_series_split(n_samples, 0.15, 100, embargo);
        (train_range, val_range, embargo)
    };
    let n_train = train_range.len();

    let external_val_labels_mapped = if let Some(vy) = external_val_y {
        Some(map_labels(vy)?)
    } else {
        None
    };

    let val_rows_for_report = if use_external_val {
        external_val_x.map(|vx| vx.nrows()).unwrap_or(0)
    } else {
        val_range.len()
    };
    let dataset_rows_for_report = n_train + val_rows_for_report + embargo;

    if n_train == 0 || val_rows_for_report == 0 {
        return Err(anyhow::anyhow!(
            "Burn training requires enough rows for a validation set; rows={}, train_rows={}, val_rows={}, embargo={}",
            n_samples,
            n_train,
            val_rows_for_report,
            embargo
        ));
    }
    info!(
        "Burn training: {} train, {} val, embargo={}, external_val={}",
        n_train, val_rows_for_report, embargo, use_external_val
    );

    // 3. Class weights
    let train_labels: Vec<i64> = y_mapped[train_range.clone()].to_vec();
    let class_weights = compute_class_weights(&train_labels, config.n_classes);

    // 4. Keep ndarray slices as the long-lived representation and materialize Burn tensors
    // per batch/validation pass so we do not retain large tensor graphs across the full run.
    let x_train_array = x_data
        .slice(ndarray::s![train_range.clone(), ..])
        .to_owned();
    let x_val_array = if use_external_val {
        external_val_x.map(|vx| vx.to_owned())
    } else if val_range.is_empty() {
        None
    } else {
        Some(x_data.slice(ndarray::s![val_range.clone(), ..]).to_owned())
    };
    let y_val_labels = if use_external_val {
        external_val_labels_mapped
    } else if val_range.is_empty() {
        None
    } else {
        Some(y_mapped[val_range.clone()].to_vec())
    };
    let val_is_empty = x_val_array.is_none();

    // 5. Optimizer + early stopping
    // Use AdamW with 5e-4 decoupled weight decay as recommended for noisy time-series
    let mut optim = AdamWConfig::new().with_weight_decay(5e-4).init();
    let mut early_stop = EarlyStopper::new(config.patience, 1e-4);
    let mut best_loss = f32::INFINITY;
    let mut best_epoch = None;
    let mut final_train_loss = f32::INFINITY;
    let mut model = cast_module_float_tensors(model, training_dtype);
    let mut best_model_snapshot: Option<M> = None;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..n_train).collect();
    let mut epochs_ran = 0usize;

    for epoch in 0..config.max_epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0usize;

        let mut start = 0;
        while start < n_train {
            let end = (start + config.batch_size).min(n_train);
            let loss_val = {
                // Tight scope keeps tensors and autograd state from lingering across batches.
                let batch_rows = indices[start..end].to_vec();
                let x_batch_array = x_train_array.select(ndarray::Axis(0), &batch_rows);
                let y_batch_labels = batch_rows
                    .iter()
                    .map(|&idx| train_labels[idx])
                    .collect::<Vec<_>>();
                let x_batch =
                    array2_to_tensor_with_dtype::<B>(&x_batch_array, device, training_dtype);
                let y_batch = labels_to_tensor::<B>(&y_batch_labels, device);

                let logits = BurnForward::forward_pass(&model, x_batch);
                let loss = cross_entropy_loss(logits, y_batch, &class_weights, device);
                let loss_val =
                    scalar_loss_value(loss.clone().into_data(), "extract Burn training loss")?;

                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.lr, model, grads_params);
                Ok::<f32, anyhow::Error>(loss_val)
            }?;

            epoch_loss += loss_val;
            n_batches += 1;
            start = end;
        }

        let train_epoch_loss = if n_batches > 0 {
            epoch_loss / n_batches as f32
        } else {
            f32::INFINITY
        };
        final_train_loss = train_epoch_loss;
        if train_epoch_loss.is_finite() && val_is_empty && train_epoch_loss < best_loss {
            best_loss = train_epoch_loss;
            best_epoch = Some(epoch);
            best_model_snapshot = Some(model.clone());
        }

        // Validation on holdout (sequential, no shuffle)
        if !val_is_empty {
            let x_val = x_val_array
                .as_ref()
                .expect("validation array must exist when validation range is non-empty");
            let y_val = y_val_labels
                .as_ref()
                .expect("validation labels must exist when validation range is non-empty");
            let x_val = array2_to_tensor_with_dtype::<B>(x_val, device, training_dtype);
            let y_val = labels_to_tensor::<B>(y_val, device);
            let val_logits = BurnForward::forward_pass(&model, x_val);
            let val_loss = cross_entropy_loss(val_logits, y_val, &class_weights, device);
            let vl = scalar_loss_value(val_loss.into_data(), "extract Burn validation loss")?;

            if vl < best_loss {
                best_loss = vl;
                best_epoch = Some(epoch);
                best_model_snapshot = Some(model.clone());
            }
            if early_stop.check(vl) {
                info!("Early stop at epoch {} (val_loss={:.6})", epoch, vl);
                epochs_ran = epoch + 1;
                break;
            }
            if epoch % 10 == 0 {
                info!(
                    "Epoch {}: train={:.6} val={:.6}",
                    epoch, train_epoch_loss, vl
                );
            }
        } else if epoch % 10 == 0 {
            info!("Epoch {}: train={:.6}", epoch, train_epoch_loss);
        }

        epochs_ran = epoch + 1;
    }

    if !best_loss.is_finite() {
        best_loss = final_train_loss;
        if best_loss.is_finite() && best_epoch.is_none() {
            best_epoch = Some(epochs_ran.saturating_sub(1));
        }
    }
    if let Some(best_model) = best_model_snapshot {
        model = best_model;
    }

    Ok((
        model,
        BurnTrainingReport {
            dataset_rows: dataset_rows_for_report,
            train_rows: n_train,
            val_rows: val_rows_for_report,
            embargo_rows: embargo,
            class_weights,
            best_loss,
            best_epoch,
            epochs_ran,
            final_train_loss,
            learning_rate: config.lr,
            batch_size: config.batch_size,
            patience: config.patience,
            seed: config.seed,
            requested_device_policy: selection.requested_policy.clone(),
            effective_device_policy: selection.effective_policy.clone(),
            execution_backend: selection.execution_backend.clone(),
            training_precision: training_precision.label().to_string(),
            training_precision_reason,
        },
    ))
}

// ============================================================================
// PREDICTION HELPER
// ============================================================================

/// Run inference and return probabilities as (n_samples, 3) array.
pub fn predict_proba<B: ManagedBurnBackend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
) -> anyhow::Result<Array2<f32>> {
    predict_proba_checked::<B, M>(model, x_data, batch_size)
}

pub fn predict_proba_on_device<B: Backend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
    device: &B::Device,
) -> anyhow::Result<Array2<f32>> {
    Ok(predict_proba_checked_on_device::<B, M>(model, x_data, batch_size, device)?.0)
}

pub fn predict_proba_on_device_with_selection<B: Backend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
    device: &B::Device,
    selection: &BurnDeviceSelection,
) -> anyhow::Result<(Array2<f32>, BurnDeviceSelection)> {
    predict_proba_checked_on_device_with_selection::<B, M>(
        model, x_data, batch_size, device, selection,
    )
}

/// Run inference and return validated probabilities as (n_samples, 3) array.
pub fn predict_proba_checked<B: ManagedBurnBackend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
) -> anyhow::Result<Array2<f32>> {
    Ok(predict_proba_checked_with_selection::<B, M>(model, x_data, batch_size)?.0)
}

pub fn predict_proba_checked_with_selection<B: ManagedBurnBackend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
) -> anyhow::Result<(Array2<f32>, BurnDeviceSelection)> {
    let (device, selection) = B::managed_device_and_selection();
    predict_proba_checked_on_device_with_selection::<B, M>(
        model, x_data, batch_size, &device, &selection,
    )
}

pub fn predict_proba_checked_on_device<B: Backend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
    device: &B::Device,
) -> anyhow::Result<(Array2<f32>, BurnDeviceSelection)> {
    let selection = BurnDeviceSelection {
        requested_policy: "external_device".to_string(),
        effective_policy: "external_device".to_string(),
        execution_backend: external_execution_backend_for::<B>(),
    };
    predict_proba_checked_on_device_with_selection::<B, M>(
        model, x_data, batch_size, device, &selection,
    )
}

pub fn predict_proba_checked_on_device_with_selection<B: Backend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
    device: &B::Device,
    selection: &BurnDeviceSelection,
) -> anyhow::Result<(Array2<f32>, BurnDeviceSelection)> {
    validate_burn_device_selection(selection)?;
    if batch_size == 0 {
        return Err(anyhow::anyhow!(
            "Burn prediction batch_size must be greater than zero"
        ));
    }
    let n_samples = x_data.nrows();
    if n_samples == 0 {
        return Ok((Array2::zeros((0, 3)), selection.clone()));
    }
    validate_feature_matrix(x_data, "Burn prediction")?;

    let mut all_probs: Vec<f32> = Vec::with_capacity(n_samples * 3);

    let mut start = 0;
    while start < n_samples {
        let end = (start + batch_size).min(n_samples);
        let data: Vec<f32> = {
            let batch = array2_to_tensor::<B>(
                &x_data.slice(ndarray::s![start..end, ..]).to_owned(),
                device,
            );
            let logits = model.forward_pass(batch);
            let probs = burn::tensor::activation::softmax(logits, 1);
            probs
                .into_data()
                .to_vec()
                .context("extract Burn prediction probabilities")?
        };
        all_probs.extend(data);
        start = end;
    }

    let normalized = normalize_probability_rows(all_probs, n_samples, 3)
        .ok_or_else(|| anyhow::anyhow!("Burn prediction probabilities failed validation"))?;
    let probabilities = Array2::from_shape_vec((n_samples, 3), normalized)
        .map_err(|_| anyhow::anyhow!("Burn prediction probabilities failed to reshape"))?;
    Ok((probabilities, selection.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_proba_checked_returns_empty_array_for_empty_input() -> anyhow::Result<()> {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((0, 2));

        let probabilities = predict_proba_checked::<InferBackend, _>(&model, &x, 16)?;
        assert_eq!(probabilities.shape(), &[0, 3]);
        Ok(())
    }

    #[test]
    fn predict_proba_rejects_zero_batch_size() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((1, 2));

        let err =
            predict_proba::<InferBackend, _>(&model, &x, 0).expect_err("batch_size=0 must fail");
        assert!(
            err.to_string()
                .contains("batch_size must be greater than zero")
        );
    }

    #[test]
    fn predict_proba_rejects_zero_feature_columns() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((1, 0));

        let err = predict_proba::<InferBackend, _>(&model, &x, 16)
            .expect_err("zero feature columns must fail early");
        assert!(err.to_string().contains("at least one feature column"));
    }

    #[test]
    fn predict_proba_rejects_non_finite_inputs() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::from_shape_vec((1, 2), vec![0.0_f32, f32::NAN]).expect("shape input");

        let err = predict_proba::<InferBackend, _>(&model, &x, 16)
            .expect_err("non-finite inputs must fail early");
        assert!(err.to_string().contains("non-finite feature values"));
    }

    #[test]
    fn train_model_rejects_non_finite_inputs_before_tensorization() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<TrainBackend>(&device);
        let x = Array2::from_shape_vec(
            (128, 2),
            (0..256)
                .map(|idx| {
                    if idx == 17 {
                        f32::INFINITY
                    } else {
                        idx as f32 * 0.01
                    }
                })
                .collect(),
        )
        .expect("shape input");
        let labels = (0..128)
            .map(|idx| match idx % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();

        let err = train_model::<TrainBackend, _>(model, &x, &labels, &TrainConfig::default())
            .expect_err("non-finite training inputs must fail early");
        assert!(err.to_string().contains("non-finite feature values"));
    }

    #[test]
    fn normalize_burn_device_policy_defaults_to_auto() {
        assert_eq!(normalize_burn_device_policy(""), "auto");
        assert_eq!(normalize_burn_device_policy("  CUDA:2 "), "gpu:2");
        assert_eq!(normalize_burn_device_policy("rocm:3"), "gpu:3");
        assert_eq!(normalize_burn_device_policy("metal"), "gpu");
        assert_eq!(normalize_burn_device_policy("vulkan:1"), "gpu:1");
        assert_eq!(normalize_burn_device_policy("cuda"), "gpu");
        assert_eq!(normalize_burn_device_policy("wgpu"), "gpu");
    }

    #[test]
    fn supported_burn_device_policy_recognizes_expected_aliases() {
        assert!(is_supported_burn_device_policy("auto"));
        assert!(is_supported_burn_device_policy("cpu"));
        assert!(is_supported_burn_device_policy("cuda:2"));
        assert!(is_supported_burn_device_policy("gpu:1"));
        assert!(is_supported_burn_device_policy("metal"));
        assert!(is_supported_burn_device_policy("rocm:2"));
        assert!(is_supported_burn_device_policy("external_device"));
    }

    #[test]
    fn burn_training_precision_fp8_request_degrades_truthfully() {
        let selection = BurnDeviceSelection {
            requested_policy: "gpu".to_string(),
            effective_policy: "gpu".to_string(),
            execution_backend: "wgpu_discrete_gpu".to_string(),
        };
        let device = <TrainBackend as Backend>::Device::default();
        let (precision, reason) = resolve_burn_training_precision_for_backend::<TrainBackend>(
            &selection,
            &device,
            Some("fp8"),
        );
        assert!(matches!(
            precision,
            BurnExecutionPrecision::Bf16 | BurnExecutionPrecision::Fp32
        ));
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("requested precision `fp8`")
        );
    }

    #[test]
    fn burn_training_precision_cpu_bf16_request_reflects_backend_support() {
        let selection = BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: "ndarray_cpu".to_string(),
        };
        let device = <TrainBackend as Backend>::Device::default();
        let (precision, reason) = resolve_burn_training_precision_for_backend::<TrainBackend>(
            &selection,
            &device,
            Some("bf16"),
        );
        if <TrainBackend as Backend>::supports_dtype(&device, DType::BF16) {
            assert_eq!(precision, BurnExecutionPrecision::Bf16);
            assert!(reason.is_none());
        } else {
            assert_eq!(precision, BurnExecutionPrecision::Fp32);
            assert!(
                reason
                    .as_deref()
                    .unwrap_or_default()
                    .contains("unsupported by the active Burn backend/device/model")
            );
        }
    }

    #[test]
    fn train_model_rejects_invalid_train_config_before_tensor_work() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<TrainBackend>(&device);
        let x = Array2::from_shape_vec((128, 2), (0..256).map(|idx| idx as f32 * 0.01).collect())
            .expect("shape input");
        let labels = (0..128)
            .map(|idx| match idx % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();
        let config = TrainConfig {
            batch_size: 0,
            ..TrainConfig::default()
        };

        let err = train_model::<TrainBackend, _>(model, &x, &labels, &config)
            .expect_err("invalid train config must fail early");
        assert!(
            err.to_string()
                .contains("batch_size must be greater than zero")
        );
    }

    #[test]
    fn train_model_with_report_on_device_requires_explicit_selection() -> anyhow::Result<()> {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<TrainBackend>(&device);
        let x = Array2::from_shape_vec((160, 2), (0..320).map(|idx| idx as f32 * 0.01).collect())
            .expect("shape input");
        let labels = (0..160)
            .map(|idx| match idx % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();
        let selection = resolve_train_device("cpu").1;

        let (_trained, report) = train_model_with_report_on_device::<TrainBackend, _>(
            model,
            &x,
            &labels,
            &TrainConfig::default(),
            &device,
            &selection,
        )?;

        assert_eq!(report.requested_device_policy, selection.requested_policy);
        assert_eq!(report.effective_device_policy, selection.effective_policy);
        assert_eq!(report.execution_backend, selection.execution_backend);
        assert!(!report.training_precision.trim().is_empty());
        Ok(())
    }

    #[test]
    fn resolve_train_device_cpu_policy_reports_consistent_backend() {
        let (_device, selection) = resolve_train_device("cpu");
        assert_eq!(selection.requested_policy, "cpu");
        assert_eq!(selection.effective_policy, "cpu");
        #[cfg(feature = "burn-wgpu-backend")]
        assert_eq!(selection.execution_backend, "wgpu_cpu");
        #[cfg(not(feature = "burn-wgpu-backend"))]
        assert_eq!(selection.execution_backend, "ndarray_cpu");
    }

    #[cfg(feature = "burn-wgpu-backend")]
    #[test]
    fn resolve_train_device_cuda_alias_reports_wgpu_runtime_truthfully() {
        let (_device, selection) = resolve_train_device("cuda:2");
        assert_eq!(selection.requested_policy, "gpu:2");
        assert_eq!(selection.effective_policy, "gpu:2");
        assert_eq!(selection.execution_backend, "wgpu_discrete_gpu");
    }

    #[cfg(feature = "burn-wgpu-backend")]
    #[test]
    fn resolve_infer_device_default_gpu_reports_wgpu_default_backend() {
        let (_device, selection) = resolve_infer_device("gpu");
        assert_eq!(selection.requested_policy, "gpu");
        assert_eq!(selection.effective_policy, "default");
        assert_eq!(selection.execution_backend, "wgpu_default");
    }

    #[test]
    fn predict_proba_checked_with_selection_returns_runtime_provenance() -> anyhow::Result<()> {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((2, 2));

        let (_probabilities, selection) =
            predict_proba_checked_with_selection::<InferBackend, _>(&model, &x, 16)?;
        assert!(!selection.requested_policy.trim().is_empty());
        assert!(!selection.effective_policy.trim().is_empty());
        assert!(!selection.execution_backend.trim().is_empty());
        Ok(())
    }

    #[test]
    fn predict_proba_on_device_with_selection_preserves_supplied_provenance() -> anyhow::Result<()>
    {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((2, 2));
        let selection = BurnDeviceSelection {
            requested_policy: "external_device".to_string(),
            effective_policy: "external_device".to_string(),
            execution_backend: active_burn_backend_name().to_string(),
        };

        let (_probabilities, returned_selection) = predict_proba_on_device_with_selection::<
            InferBackend,
            _,
        >(&model, &x, 16, &device, &selection)?;
        assert_eq!(returned_selection, selection);
        Ok(())
    }

    #[test]
    fn predict_proba_checked_on_device_reports_typed_external_provenance() -> anyhow::Result<()> {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((2, 2));

        let (_probabilities, selection) =
            predict_proba_checked_on_device::<InferBackend, _>(&model, &x, 16, &device)?;
        assert_eq!(selection.requested_policy, "external_device");
        assert_eq!(selection.effective_policy, "external_device");
        assert!(selection.execution_backend.starts_with("external:"));
        Ok(())
    }

    #[test]
    fn train_model_with_report_rejects_incoherent_runtime_selection() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<TrainBackend>(&device);
        let x = Array2::<f32>::zeros((12, 2));
        let labels = (0..12)
            .map(|idx| match idx % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect::<Vec<_>>();
        let selection = BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: "wgpu_discrete_gpu".to_string(),
        };

        let err = train_model_with_report_on_device::<TrainBackend, _>(
            model,
            &x,
            &labels,
            &TrainConfig::default(),
            &device,
            &selection,
        )
        .expect_err("incoherent runtime provenance must fail");
        assert!(
            err.to_string()
                .contains("incompatible with effective policy cpu"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn predict_proba_on_device_with_selection_rejects_incoherent_runtime_selection() {
        let device = Default::default();
        let model = BurnMLPConfig::new(2).init::<InferBackend>(&device);
        let x = Array2::<f32>::zeros((2, 2));
        let selection = BurnDeviceSelection {
            requested_policy: "cpu".to_string(),
            effective_policy: "cpu".to_string(),
            execution_backend: "wgpu_discrete_gpu".to_string(),
        };

        let err = predict_proba_on_device_with_selection::<InferBackend, _>(
            &model, &x, 16, &device, &selection,
        )
        .expect_err("incoherent runtime provenance must fail");
        assert!(
            err.to_string()
                .contains("incompatible with effective policy cpu"),
            "unexpected error: {err}"
        );
    }
}
