pub mod compiled;
pub mod cpu_batch;
pub mod cpu_single;
#[cfg(feature = "cuda")]
pub mod cuda;
/// The f64 CUDA lane. ADDITIVE — `cuda` and `cuda_non_ma_generated` are the
/// f32 lane and are untouched.
#[cfg(feature = "cuda")]
pub mod cuda_f64;
#[cfg(feature = "cuda")]
pub mod cuda_non_ma_generated;
pub mod error;
pub mod types;

pub use compiled::{compile_call, run_compiled_cpu, CompiledIndicatorCall};
pub use cpu_batch::{compute_cpu_batch, compute_cpu_batch_strict};
pub use cpu_single::compute_cpu;
pub use error::IndicatorDispatchError;
pub use types::{
    IndicatorBatchOutput, IndicatorBatchRequest, IndicatorComputeOutput, IndicatorComputeRequest,
    IndicatorDataRef, IndicatorParamSet, IndicatorSeries, ParamKV, ParamValue,
};

#[cfg(feature = "cuda")]
pub use compiled::run_compiled_cuda;
#[cfg(feature = "cuda")]
pub use cuda::{
    compute_cuda, compute_cuda_device, compute_pattern_recognition_cuda_bitmask,
    compute_pattern_recognition_cuda_device_bitmask,
};
#[cfg(feature = "cuda")]
pub use cuda_f64::{
    compute_cuda_device_f64, f64_kernel_for, resolve_f64_entry_point, resolve_f64_kernel,
    CudaOutputTargetF64, F64FirstValidRule, F64InputKind, F64KernelSpec, IndicatorCudaDataRefF64,
    IndicatorCudaDeviceDataRefF64, IndicatorCudaDeviceRequestF64, IndicatorCudaOutputF64,
    IndicatorCudaSeriesF64, F64_KERNELS,
};
#[cfg(feature = "cuda")]
pub use types::{
    CudaOutputTarget, DeviceMatrixF32, IndicatorCudaBitmaskRequest, IndicatorCudaDataRef,
    IndicatorCudaDeviceBitmaskRequest, IndicatorCudaDeviceDataRef, IndicatorCudaDeviceRequest,
    IndicatorCudaOutput, IndicatorCudaRequest, IndicatorCudaSeries,
    PatternRecognitionCudaBitmaskOutput,
};
