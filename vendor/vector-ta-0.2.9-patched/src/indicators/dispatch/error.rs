use crate::indicators::registry::IndicatorInputKind;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum IndicatorDispatchError {
    #[error("unknown indicator: {id}")]
    UnknownIndicator { id: String },
    #[error("unknown output '{output}' for indicator '{indicator}'")]
    UnknownOutput { indicator: String, output: String },
    #[error("missing required input {input:?} for indicator '{indicator}'")]
    MissingRequiredInput {
        indicator: String,
        input: IndicatorInputKind,
    },
    #[error("invalid parameter '{key}' for indicator '{indicator}': {reason}")]
    InvalidParam {
        indicator: String,
        key: String,
        reason: String,
    },
    #[error("unsupported capability '{capability}' for indicator '{indicator}'")]
    UnsupportedCapability {
        indicator: String,
        capability: &'static str,
    },
    #[error("data length mismatch: {details}")]
    DataLengthMismatch { details: String },
    #[error("kernel unavailable: {details}")]
    KernelUnavailable { details: String },
    #[error("compute failed for indicator '{indicator}': {details}")]
    ComputeFailed { indicator: String, details: String },
    /// An f64 CUDA request named an indicator that has no f64 kernel.
    ///
    /// This is deliberately a HARD failure with the indicator named. The two
    /// tempting alternatives are both already present in this crate and both
    /// are wrong here:
    ///
    /// * serving the f32 kernel — indicator values feed a
    ///   `combined >= long_threshold` comparison, so a one-ULP move flips a
    ///   trade, and the f32 lane was measured 54% wrong at 200k bars;
    /// * computing on the host and returning it as device work — which is
    ///   exactly what the nine empty `*_f64` stub kernels and their wrappers do
    ///   today (`possible_rsi_wrapper.rs:104-152`).
    #[error(
        "no f64 CUDA kernel for indicator '{indicator}'. The f64 lane does not fall back to f32 \
         and does not fall back to the CPU. Indicators with an f64 kernel: {available}"
    )]
    CudaF64KernelMissing {
        indicator: String,
        available: String,
    },
}
