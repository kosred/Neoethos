//! CUDA implementation of Online Passive-Aggressive PB-v2.
//!
//! This facade keeps kernel ownership bounded. Fit and predict retain device
//! buffers only for one call: parameters are persisted as the model artifact
//! and uploaded again for each inference call. There is no cross-call device
//! residency claim and no CPU numerical fallback on the CUDA route.

mod device_utils;
mod fit;
mod inference;
mod predict;
mod preprocess;
mod update;

pub(crate) use device_utils::validate_passive_aggressive_cuda_device_identity;
pub(crate) use fit::try_fit_passive_aggressive_cuda_full_pipeline;
pub(crate) use predict::try_predict_passive_aggressive_cuda_full_pipeline;
#[cfg(test)]
pub(crate) use update::try_fit_passive_aggressive_cuda;

pub(super) const CLASS_COUNT: usize = 3;
pub(super) const PA_CUBE_UNITS: usize = 256;
pub(super) const DEVICE_ARITHMETIC_REDUCTION_FAULT: u32 = 1;
pub(super) const DEVICE_ARITHMETIC_UPDATE_FAULT: u32 = 2;
pub(super) const DEVICE_LABEL_MAP_FAULT: u32 = 10;
pub(super) const DEVICE_MISSING_CLASS_0_FAULT: u32 = 20;
pub(super) const DEVICE_MISSING_CLASS_1_FAULT: u32 = 21;
pub(super) const DEVICE_MISSING_CLASS_2_FAULT: u32 = 22;
pub(super) const DEVICE_SCALER_INPUT_FAULT: u32 = 30;
pub(super) const DEVICE_SCALER_ARITHMETIC_FAULT: u32 = 31;
pub(super) const DEVICE_SCALER_OUTPUT_FAULT: u32 = 32;
pub(super) const DEVICE_TRANSFORM_ARITHMETIC_FAULT: u32 = 40;
pub(super) const DEVICE_INFERENCE_ARITHMETIC_FAULT: u32 = 50;
pub(super) const DEVICE_MEMORY_HEADROOM_PERCENT: usize = 10;
pub(super) const DEVICE_MEMORY_MIN_HEADROOM_BYTES: usize = 256 * 1024 * 1024;
