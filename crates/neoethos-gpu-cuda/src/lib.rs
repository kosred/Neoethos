//! Stable Rust wrapper around the Stage 1 CUDA C ABI scaffold.

use neoethos_gpu_contracts::ABI_VERSION;
use thiserror::Error;

unsafe extern "C" {
    fn neoethos_gpu_cuda_abi_version() -> u32;
    fn neoethos_gpu_cuda_runtime_available() -> i32;
    fn neoethos_gpu_cuda_smoke(input: *const u32, output: *mut u32, len: usize) -> i32;
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaSmokeError {
    #[error("native CUDA ABI mismatch: Rust={rust}, native={native}")]
    AbiMismatch { rust: u32, native: u32 },
    #[error("CUDA runtime/device is unavailable")]
    RuntimeUnavailable,
    #[error("CUDA smoke kernel failed with status {0}")]
    NativeFailure(i32),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_and_rust_share_the_same_abi_version() {
        validate_abi().unwrap();
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_reports_runtime_unavailable_without_fabricating_success() {
        assert!(!runtime_available());
        assert_eq!(
            smoke_add_one(&[1, 2, 3]),
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
    }
}
