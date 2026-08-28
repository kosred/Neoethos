pub mod compiled;
pub mod cpu_batch;
pub mod cpu_single;
#[cfg(feature = "cuda-build-native")]
pub mod cuda;
/// The f64 CUDA lane. ADDITIVE — `cuda` and `cuda_non_ma_generated` are the
/// f32 lane and are untouched.
#[cfg(feature = "cuda-build-native")]
pub mod cuda_f64;
#[cfg(feature = "cuda-build-native")]
pub mod cuda_non_ma_generated;
pub mod error;
pub mod types;

pub use compiled::{CompiledIndicatorCall, compile_call, run_compiled_cpu};
pub use cpu_batch::{compute_cpu_batch, compute_cpu_batch_strict};
pub use cpu_single::compute_cpu;
pub use error::IndicatorDispatchError;
pub use types::{
    IndicatorBatchOutput, IndicatorBatchRequest, IndicatorComputeOutput, IndicatorComputeRequest,
    IndicatorDataRef, IndicatorParamSet, IndicatorSeries, ParamKV, ParamValue,
};

#[cfg(feature = "cuda-build-native")]
pub use compiled::run_compiled_cuda;
#[cfg(feature = "cuda-build-native")]
pub use cuda::{compute_cuda, compute_cuda_device};
#[cfg(feature = "cuda-build-native")]
pub use cuda_f64::{
    CudaOutputTargetF64, F64_KERNELS, F64FirstValidRule, F64InputKind, F64KernelSpec,
    IndicatorCudaDataRefF64, IndicatorCudaDeviceDataRefF64, IndicatorCudaDeviceRequestF64,
    IndicatorCudaOutputF64, IndicatorCudaSeriesF64, compute_cuda_device_f64, f64_kernel_for,
    has_f64_resident_output_route, resolve_f64_entry_point, resolve_f64_kernel,
};
#[cfg(feature = "cuda-build-native")]
pub use types::{
    CudaOutputTarget, DeviceMatrixF32, IndicatorCudaDataRef, IndicatorCudaDeviceDataRef,
    IndicatorCudaDeviceRequest, IndicatorCudaOutput, IndicatorCudaRequest, IndicatorCudaSeries,
};
