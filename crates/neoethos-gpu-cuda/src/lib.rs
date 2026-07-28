//! Stable Rust wrapper around the Stage 1 CUDA C ABI scaffold.

use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::device::{
    DatasetHeader, GeneDescriptor, NeoPopulationCounters, NeoPopulationEvent,
    NeoPopulationMetricRow, NeoPopulationOutcome, NeoPopulationSettings, ScenarioDescriptor,
};
use thiserror::Error;

mod population;

pub use population::{
    CudaPopulationError, PopulationDatasetView, PopulationDiagnostics, PopulationGeneView,
    PopulationSession, population_status_message,
};

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
    pub stop_price: f32,
    pub target_price: f32,
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
    fn neoethos_gpu_cuda_device_count() -> i32;
    fn neoethos_gpu_cuda_device_free_memory(device: i32) -> u64;
    fn neoethos_gpu_cuda_smoke(input: *const u32, output: *mut u32, len: usize) -> i32;
    fn neoethos_gpu_cuda_warp_first_hit(
        highs: *const f32,
        lows: *const f32,
        rows: usize,
        events: *const CudaFirstHitEvent,
        results: *mut CudaFirstHitResult,
        event_count: usize,
    ) -> i32;
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
    highs: &[f32],
    lows: &[f32],
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
    fn first_hit_layout_is_stable() {
        assert_eq!(core::mem::size_of::<CudaFirstHitEvent>(), 24);
        assert_eq!(core::mem::align_of::<CudaFirstHitEvent>(), 4);
        assert_eq!(core::mem::size_of::<CudaFirstHitResult>(), 8);
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
    fn real_cuda_smoke_is_explicitly_gpu_gated() {
        if std::env::var("NEOETHOS_RUN_CUDA_SMOKE").as_deref() != Ok("1") {
            eprintln!("CUDA smoke skipped; set NEOETHOS_RUN_CUDA_SMOKE=1 on a GPU runner");
            return;
        }
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
    }
}
