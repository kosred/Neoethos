//! Stable Rust wrapper around the Stage 1 CUDA C ABI scaffold.

use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::device::{
    DatasetHeader, GeneDescriptor, NeoPopulationCounters, NeoPopulationEvent,
    NeoPopulationMetricRow, NeoPopulationOutcome, NeoPopulationSettings, ScenarioDescriptor,
};
use thiserror::Error;

#[cfg(feature = "cuda")]
pub mod data_population_workspace_plan_v1;
#[cfg(feature = "cuda")]
pub mod resident_classic_ta_v3;
#[cfg(feature = "cuda")]
pub mod resident_feature_store_v3;
#[cfg(feature = "cuda-device-fixtures")]
pub mod resident_feature_store_v3_device_fixture;
#[cfg(feature = "cuda")]
pub mod resident_footprint_v2;
#[cfg(feature = "cuda")]
// The pre-existing V1 generation owner was source-contract-only. Root it
// privately for the first V3 -> Search consumer without widening its API.
#[allow(dead_code)]
mod resident_generation_v1;
#[cfg(feature = "cuda")]
pub mod resident_higher_timeframe_alignment_v3;
#[cfg(feature = "cuda-device-fixtures")]
pub mod resident_higher_timeframe_alignment_v3_device_fixture;
#[cfg(feature = "cuda")]
pub mod resident_quant_v3;
#[cfg(feature = "cuda")]
pub mod resident_regime_v3;
#[cfg(feature = "cuda")]
pub mod resident_robust_normalization_v2;
#[cfg(feature = "cuda")]
mod resident_scoring_v2;
#[cfg(any(
    feature = "cuda",
    feature = "resident-search-slice2-compile-contract",
    all(test, feature = "resident-search-slice2-host-contract")
))]
#[cfg_attr(
    not(all(test, feature = "resident-search-slice2-host-contract")),
    allow(dead_code)
)]
mod resident_search_slice2_admission_v2;
#[cfg(any(feature = "cuda", feature = "resident-search-slice2-compile-contract"))]
pub mod resident_search_slice2_v3;
#[cfg(feature = "cuda")]
pub mod resident_search_v2;
#[cfg(feature = "cuda")]
pub mod resident_session_v2;
#[cfg(feature = "cuda-device-fixtures")]
pub mod resident_session_v2_device_fixture;
#[cfg(feature = "cuda")]
pub mod resident_smc_v3;
#[cfg(feature = "cuda")]
pub mod resident_trim_prefilter_v1;

#[cfg(all(test, feature = "cuda"))]
mod resident_archive_knn_v2_tests;

#[cfg(all(test, feature = "cuda-device-fixtures"))]
mod resident_search_generation_v2_device_tests;

pub mod full_discovery_workspace_plan_v1;
pub mod physical_gpu_inventory_v1;
pub mod run_device_admission_v1;

mod population;

#[cfg(feature = "cuda")]
pub use data_population_workspace_plan_v1::{
    AdmittedNativeCudaDataPopulationRunV1, DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1,
    DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1, DataPopulationWorkspacePlanErrorCodeV1,
    DataPopulationWorkspacePlanErrorV1, DataPopulationWorkspacePreflightRequestV1,
    SealedDataPopulationExecutionLimitsV1, SealedDataPopulationGpuWorkspacePlanV1,
    SealedNativeCudaDataPopulationPreflightFactsV1, bind_data_population_gpu_workspace_plan_v1,
    native_cuda_data_population_preflight_facts_v1, seal_data_population_gpu_workspace_plan_v1,
};
pub use full_discovery_workspace_plan_v1::{
    AdmittedFullDiscoveryGpuRunV1, FullDiscoveryGpuRunReceiptV1,
    FullDiscoveryWorkspacePlanErrorCodeV1, FullDiscoveryWorkspacePlanErrorV1,
    SealedFullDiscoveryGpuWorkspacePlanV1, bind_full_discovery_workspace_plan_v1,
};
pub use physical_gpu_inventory_v1::{
    PhysicalGpuInventoryErrorCodeV1, PhysicalGpuInventoryErrorV1, PhysicalGpuInventoryPlatformV1,
    PhysicalGpuInventoryRecordV1, SealedNoPhysicalGpuReceiptV1,
    SealedPhysicalGpuInventoryReceiptV1, probe_physical_gpu_inventory_v1,
};
pub use run_device_admission_v1::{
    DiscoveryRunDeviceAdmissionErrorCodeV1, DiscoveryRunDeviceAdmissionErrorV1,
    SealedDiscoveryRunDeviceAdmissionV1, acquire_discovery_run_device_admission_v1,
};

pub use population::{
    CudaPopulationDeviceIdentityV1, CudaPopulationError, HostPopulationMetricsReceiptV1,
    PopulationDatasetView, PopulationDiagnostics, PopulationEvaluationViewV1,
    PopulationGeneStorePlanV1, PopulationGeneView, PopulationMetricsOnlyPlanV1,
    PopulationParentDatasetInputV1, PopulationParentDatasetV1, PopulationParentDevicePlanV1,
    PopulationResidencyCountersV1, PopulationSession, PopulationTimestampModeV1,
    PopulationViewKindV1, ResidentAdaptiveBaseRequestV1, ResidentAdaptiveBaseViewTokenV1,
    TerminalCompactPopulationResultReceiptV1, population_status_message,
};
#[cfg(feature = "cuda")]
pub use population::{ResidentAdaptiveBaseViewTokenIdentityV1, ResidentPopulationMetricsV1};
// Callers that decide what to do about a failure need to name the failure.
// Without these, the only handle on a status was its rendered message, and a
// caller matching on that text silently stopped working when the wording moved.
pub use population::{
    RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1, STATUS_ADAPTIVE_BASE_DEGENERATE, STATUS_ALLOCATION_FAILED,
    STATUS_EVENT_CAPACITY, STATUS_LAUNCH_FAILED,
};
// A caller sizing a batch has to know what the kernel reserves per candidate.
pub use population::MAX_TRADES_PER_CANDIDATE;

/// Number of SMC slots carried by one row of the canonical SMC contract.
pub const SMC_SLOTS: usize = 11;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaFirstHitEvent {
    pub entry_bar: u32,
    pub last_bar: u32,
    /// `1` for long, `-1` for short.
    pub direction: i32,
    /// `0` for stop-first, `1` for target-first.
    pub precedence: i32,
    pub stop_price: f64,
    pub target_price: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaFirstHitResult {
    /// `-1` when no exit occurs inside the requested window.
    pub exit_bar: i32,
    /// `0` none, `1` stop-loss, `2` take-profit.
    pub exit_reason: i32,
}

unsafe extern "C" {
    fn neoethos_gpu_cuda_abi_version() -> u32;
    fn neoethos_gpu_cuda_runtime_available() -> i32;
    fn neoethos_gpu_cuda_probe_device_count_v1(out_count: *mut u32) -> i32;
    fn neoethos_gpu_cuda_device_count() -> i32;
    fn neoethos_gpu_cuda_device_free_memory(device: i32) -> u64;
    fn neoethos_gpu_cuda_smoke(input: *const u32, output: *mut u32, len: usize) -> i32;
    fn neoethos_gpu_cuda_warp_first_hit(
        highs: *const f64,
        lows: *const f64,
        rows: usize,
        events: *const CudaFirstHitEvent,
        results: *mut CudaFirstHitResult,
        event_count: usize,
    ) -> i32;
}

const NEO_CUDA_DEVICE_PROBE_OK: i32 = 0;
const NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT: i32 = -50;
const NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE: i32 = -51;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CudaDeviceEnumerationErrorV1 {
    #[error("native CUDA adapter is not compiled into this binary")]
    NativeAdapterUnavailable,
    #[error("CUDA device enumeration failed with runtime status {0}")]
    RuntimeFailure(i32),
    #[error("native CUDA device enumeration returned invalid output")]
    InvalidNativeOutput,
}

pub fn probe_cuda_device_count_v1() -> Result<u32, CudaDeviceEnumerationErrorV1> {
    let mut count = u32::MAX;
    // SAFETY: `count` is a valid exclusive output pointer for the duration of
    // the call. The native contract writes it only after CUDA success.
    let status = unsafe { neoethos_gpu_cuda_probe_device_count_v1(&mut count) };
    match status {
        NEO_CUDA_DEVICE_PROBE_OK if count != u32::MAX => Ok(count),
        NEO_CUDA_DEVICE_PROBE_OK | NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT => {
            Err(CudaDeviceEnumerationErrorV1::InvalidNativeOutput)
        }
        NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE => {
            Err(CudaDeviceEnumerationErrorV1::NativeAdapterUnavailable)
        }
        runtime_status => Err(CudaDeviceEnumerationErrorV1::RuntimeFailure(runtime_status)),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaSmokeError {
    #[error("native CUDA ABI mismatch: Rust={rust}, native={native}")]
    AbiMismatch { rust: u32, native: u32 },
    #[error("CUDA runtime/device is unavailable")]
    RuntimeUnavailable,
    #[error("invalid CUDA first-hit input: {0}")]
    InvalidInput(&'static str),
    #[error("CUDA smoke kernel failed with status {0}")]
    NativeFailure(i32),
    #[error("CUDA warp first-hit kernel failed with status {0}")]
    WarpFirstHitFailure(i32),
}

pub fn native_abi_version() -> u32 {
    // SAFETY: no arguments, no memory access, stable C ABI.
    unsafe { neoethos_gpu_cuda_abi_version() }
}

/// Reviewed build manifest embedded by `build.rs` for a real CUDA build.
/// Honest no-CUDA adapter builds return `None`; they cannot emit native success.
pub const fn cuda_build_manifest_v1() -> Option<&'static str> {
    option_env!("NEOETHOS_CUDA_BUILD_MANIFEST_V1")
}

pub fn validate_abi() -> Result<(), CudaSmokeError> {
    let native = native_abi_version();
    if native == ABI_VERSION {
        Ok(())
    } else {
        Err(CudaSmokeError::AbiMismatch {
            rust: ABI_VERSION,
            native,
        })
    }
}

pub fn runtime_available() -> bool {
    // SAFETY: no arguments, no memory access, stable C ABI.
    unsafe { neoethos_gpu_cuda_runtime_available() == 1 }
}

/// Number of visible CUDA devices.
///
/// This exists so the CubeCL CUDA lane can bounds-check a device index without
/// depending on `tch`, which would drag a multi-gigabyte libtorch install onto
/// every machine that only wants to run a GPU benchmark.
pub fn device_count() -> usize {
    // SAFETY: no arguments, no memory access, stable C ABI.
    let count = unsafe { neoethos_gpu_cuda_device_count() };
    count.max(0) as usize
}

/// Free memory on `device`, in bytes, or `None` when it cannot be determined.
///
/// A session's event capacity has to be a function of the hardware, never of
/// the caller's parameters — that is the whole of the never-OOM invariant. This
/// is what makes that possible in-process, instead of shelling out to
/// `nvidia-smi` and parsing it. `None` means unknown, and a caller must refuse
/// to guess rather than assume a comfortable number.
pub fn device_free_memory_bytes(device: usize) -> Option<u64> {
    if !runtime_available() {
        return None;
    }
    // SAFETY: takes an integer by value, returns an integer, stable C ABI.
    let free = unsafe { neoethos_gpu_cuda_device_free_memory(device as i32) };
    (free > 0).then_some(free)
}

pub fn smoke_add_one(input: &[u32]) -> Result<Vec<u32>, CudaSmokeError> {
    validate_abi()?;
    if !runtime_available() {
        return Err(CudaSmokeError::RuntimeUnavailable);
    }
    let mut output = vec![0_u32; input.len()];
    // SAFETY: pointers are valid for `input.len()` elements and non-overlapping.
    let status =
        unsafe { neoethos_gpu_cuda_smoke(input.as_ptr(), output.as_mut_ptr(), input.len()) };
    if status == 0 {
        Ok(output)
    } else {
        Err(CudaSmokeError::NativeFailure(status))
    }
}

pub fn warp_first_hit(
    highs: &[f64],
    lows: &[f64],
    events: &[CudaFirstHitEvent],
) -> Result<Vec<CudaFirstHitResult>, CudaSmokeError> {
    validate_abi()?;
    if highs.is_empty() || highs.len() != lows.len() {
        return Err(CudaSmokeError::InvalidInput(
            "high/low arrays must be equal and non-empty",
        ));
    }
    if highs.iter().chain(lows).any(|value| !value.is_finite()) {
        return Err(CudaSmokeError::InvalidInput("prices must be finite"));
    }
    for event in events {
        if event.entry_bar >= event.last_bar || event.last_bar as usize >= highs.len() {
            return Err(CudaSmokeError::InvalidInput("event bar window is invalid"));
        }
        if !matches!(event.direction, -1 | 1) {
            return Err(CudaSmokeError::InvalidInput("direction must be -1 or 1"));
        }
        if !matches!(event.precedence, 0 | 1) {
            return Err(CudaSmokeError::InvalidInput("precedence must be 0 or 1"));
        }
        if !event.stop_price.is_finite() || !event.target_price.is_finite() {
            return Err(CudaSmokeError::InvalidInput("stop/target must be finite"));
        }
    }
    if events.is_empty() {
        return Ok(Vec::new());
    }
    if !runtime_available() {
        return Err(CudaSmokeError::RuntimeUnavailable);
    }

    let mut results = vec![CudaFirstHitResult::default(); events.len()];
    // SAFETY: all pointers cover the declared lengths, remain alive for the call,
    // and the output does not alias any input.
    let status = unsafe {
        neoethos_gpu_cuda_warp_first_hit(
            highs.as_ptr(),
            lows.as_ptr(),
            highs.len(),
            events.as_ptr(),
            results.as_mut_ptr(),
            events.len(),
        )
    };
    if status == 0 {
        Ok(results)
    } else {
        Err(CudaSmokeError::WarpFirstHitFailure(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_and_rust_share_the_same_abi_version() {
        validate_abi().unwrap();
    }

    #[test]
    fn first_hit_f64_abi_contract_is_stable() {
        use core::mem::{align_of, offset_of, size_of};

        type WarpFirstHitFn = fn(
            &[f64],
            &[f64],
            &[CudaFirstHitEvent],
        ) -> Result<Vec<CudaFirstHitResult>, CudaSmokeError>;

        let _: WarpFirstHitFn = warp_first_hit;
        let _: unsafe extern "C" fn(
            *const f64,
            *const f64,
            usize,
            *const CudaFirstHitEvent,
            *mut CudaFirstHitResult,
            usize,
        ) -> i32 = neoethos_gpu_cuda_warp_first_hit;

        assert_eq!(size_of::<CudaFirstHitEvent>(), 32);
        assert_eq!(align_of::<CudaFirstHitEvent>(), 8);
        assert_eq!(offset_of!(CudaFirstHitEvent, stop_price), 16);
        assert_eq!(offset_of!(CudaFirstHitEvent, target_price), 24);
        assert_eq!(core::mem::size_of::<CudaFirstHitResult>(), 8);

        let header = include_str!("../native/neoethos_gpu_cuda.h");
        let kernel = include_str!("../native/prototype_b.cu");
        let stub = include_str!("../native/stub.cpp");
        let layout_asserts = include_str!("../native/layout_asserts.cpp");
        for required in [
            "double stop_price;",
            "double target_price;",
            "const double* highs",
            "const double* lows",
        ] {
            assert!(
                header.contains(required),
                "native header is missing f64 first-hit contract `{required}`"
            );
        }
        for required in [
            "__device__ std::int32_t first_hit_reason(double high,",
            "double low,",
            "__global__ void warp_first_hit_kernel(const double* highs,",
            "double* device_highs = nullptr;",
            "double* device_lows = nullptr;",
            "rows * sizeof(double)",
        ] {
            assert!(
                kernel.contains(required),
                "native kernel is missing f64 first-hit contract `{required}`"
            );
        }
        assert!(stub.contains("const double*"));
        assert!(layout_asserts.contains("sizeof(NeoFirstHitEvent) == 32"));
        assert!(layout_asserts.contains("alignof(NeoFirstHitEvent) == 8"));
        assert!(
            layout_asserts.contains(
                "using NeoWarpFirstHitFn = std::int32_t (*)(const double*, const double*"
            )
        );

        for source in [header, kernel, stub, layout_asserts] {
            assert!(
                !source.contains("const float* highs")
                    && !source.contains("const float* lows")
                    && !source.contains("float stop_price")
                    && !source.contains("float target_price"),
                "first-hit ABI still contains a narrowing f32 surface"
            );
        }
    }

    #[test]
    fn invalid_first_hit_input_is_rejected_before_ffi() {
        let event = CudaFirstHitEvent {
            entry_bar: 0,
            last_bar: 1,
            direction: 0,
            precedence: 0,
            stop_price: 95.0,
            target_price: 105.0,
        };
        assert_eq!(
            warp_first_hit(&[100.0, 101.0], &[100.0, 99.0], &[event]),
            Err(CudaSmokeError::InvalidInput("direction must be -1 or 1"))
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_reports_runtime_unavailable_without_fabricating_success() {
        assert!(!runtime_available());
        assert_eq!(
            smoke_add_one(&[1, 2, 3]),
            Err(CudaSmokeError::RuntimeUnavailable)
        );
        let event = CudaFirstHitEvent {
            entry_bar: 0,
            last_bar: 1,
            direction: 1,
            precedence: 0,
            stop_price: 95.0,
            target_price: 105.0,
        };
        assert_eq!(
            warp_first_hit(&[100.0, 106.0], &[100.0, 94.0], &[event]),
            Err(CudaSmokeError::RuntimeUnavailable)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn real_cuda_smoke_executes_f64_first_hit_without_narrowing() {
        assert!(
            runtime_available(),
            "the cuda feature's real-device gate requires a visible CUDA device"
        );
        let output = smoke_add_one(&[1, 2, 41]).unwrap();
        assert_eq!(output, vec![2, 3, 42]);

        let events = [CudaFirstHitEvent {
            entry_bar: 0,
            last_bar: 2,
            direction: 1,
            precedence: 0,
            stop_price: 95.0,
            target_price: 105.0,
        }];
        let result = warp_first_hit(&[100.0, 106.0, 101.0], &[100.0, 94.0, 99.0], &events).unwrap();
        assert_eq!(
            result,
            vec![CudaFirstHitResult {
                exit_bar: 1,
                exit_reason: 1,
            }]
        );

        let lower = f64::from_bits(1.0_f64.to_bits() - 1);
        let upper = f64::from_bits(1.0_f64.to_bits() + 1);
        let nearest_f32_boundary = f64::from(1.0_f32);
        assert!(lower < nearest_f32_boundary && nearest_f32_boundary < upper);
        let precision_event = [CudaFirstHitEvent {
            entry_bar: 0,
            last_bar: 2,
            direction: 1,
            precedence: 0,
            stop_price: 0.0,
            target_price: 1.0,
        }];
        let precision_result =
            warp_first_hit(&[0.5_f64, lower, upper], &[0.5_f64; 3], &precision_event).unwrap();
        assert_eq!(
            precision_result,
            vec![CudaFirstHitResult {
                exit_bar: 2,
                exit_reason: 2,
            }],
            "bar 1 must remain below the f64 target; an f32 boundary would exit early"
        );
    }
}
