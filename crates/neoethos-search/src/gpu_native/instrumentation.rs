//! Optional NVTX stage ranges.
//!
//! The `nvtx` feature links NVIDIA's `nvToolsExt` only on profiler builds. The
//! ordinary CPU/iGPU builds retain the same API as a zero-cost no-op.

#[derive(Debug)]
pub struct StageRange {
    #[cfg(feature = "nvtx")]
    active: bool,
}

impl StageRange {
    pub fn push(name: &'static str) -> Self {
        #[cfg(feature = "nvtx")]
        {
            use std::ffi::CString;
            let active = CString::new(name)
                .ok()
                .is_some_and(|name| unsafe { nvtxRangePushA(name.as_ptr()) >= 0 });
            return Self { active };
        }

        #[cfg(not(feature = "nvtx"))]
        {
            let _ = name;
            Self {}
        }
    }
}

impl Drop for StageRange {
    fn drop(&mut self) {
        #[cfg(feature = "nvtx")]
        if self.active {
            unsafe {
                nvtxRangePop();
            }
        }
    }
}

#[macro_export]
macro_rules! gpu_stage_range {
    ($name:expr) => {
        let _neoethos_gpu_stage_range = $crate::gpu_native::instrumentation::StageRange::push($name);
    };
}

#[cfg(feature = "nvtx")]
#[link(name = "nvToolsExt")]
unsafe extern "C" {
    fn nvtxRangePushA(message: *const std::ffi::c_char) -> i32;
    fn nvtxRangePop() -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_range_has_raii_shape() {
        let _range = StageRange::push("neoethos.test.range");
    }
}
