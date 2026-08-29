//! Exact native-SASS CUDA module loading.
//!
//! `build.rs` compiles one ELF cubin per kernel and per selected architecture,
//! verifies every artifact with `cuobjdump`, and generates the registry included
//! below. Runtime selection requires the current device's exact `sm_X` entry.
//! No alternate artifact, driver compilation, approximate architecture, or CPU
//! substitution path exists.

use std::sync::{Mutex, OnceLock};

use cust::{
    context::CurrentContext,
    device::DeviceAttribute,
    error::{CudaError, CudaResult},
    module::Module,
};

use crate::native_sass::select_exact_native_cubin;

mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/vector_ta_native_cubin_registry.rs"
    ));
}

pub const COMPILED_ARCHS: &[u32] = generated::COMPILED_ARCHS;
pub const COMPILED_ARCH_SOURCE: &str = generated::COMPILED_ARCH_SOURCE;
pub const COMPILED_NVCC_VERSION: &str = generated::NVCC_VERSION;
pub const NATIVE_CUBIN_COUNT: usize = generated::NATIVE_CUBIN_COUNT;

fn debug_enabled() -> bool {
    matches!(
        std::env::var("CUDA_MODULE_LOAD_DEBUG").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("True")
    )
}

fn current_device_capability() -> CudaResult<(i32, i32)> {
    let device = CurrentContext::get_device()?;
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    Ok((major, minor))
}

fn failure_slot() -> &'static Mutex<Option<String>> {
    static SLOT: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

pub fn last_module_load_failure() -> Option<String> {
    match failure_slot().lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

fn record_failure(message: String) {
    eprintln!("{message}");
    let mut guard = match failure_slot().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = Some(message);
}

fn compiled_architectures() -> String {
    COMPILED_ARCHS
        .iter()
        .map(|arch| format!("sm_{arch}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn load_module_for_current_context(stem: &str) -> CudaResult<Module> {
    let (major, minor) = match current_device_capability() {
        Ok(capability) => capability,
        Err(error) => {
            record_failure(format!(
                "vector-ta: cannot read the current CUDA device architecture for {stem:?}: \
                 {error:?}; refusing to guess or use another execution path"
            ));
            return Err(error);
        }
    };

    let cubin = match select_exact_native_cubin(stem, major, minor, generated::NATIVE_CUBINS) {
        Ok(cubin) => cubin,
        Err(error) => {
            record_failure(format!(
                "vector-ta: exact native cubin selection failed: {error}\n  device: sm_{major}{minor}\n  \
                 compiled SASS: {}\n  architecture source: {COMPILED_ARCH_SOURCE}\n  nvcc: \
                 {COMPILED_NVCC_VERSION}\n  registry entries: {NATIVE_CUBIN_COUNT}\n  Refusing \
                 alternate artifacts, driver compilation, approximate architecture, or CPU \
                 substitution.",
                compiled_architectures()
            ));
            return Err(CudaError::InvalidImage);
        }
    };

    match Module::from_cubin(cubin, &[]) {
        Ok(module) => {
            if debug_enabled() {
                eprintln!(
                    "[cuda-module-loader] {stem}: loaded exact sm_{major}{minor} native cubin \
                     (source={COMPILED_ARCH_SOURCE}; nvcc={COMPILED_NVCC_VERSION})"
                );
            }
            Ok(module)
        }
        Err(error) => {
            record_failure(format!(
                "vector-ta: CUDA driver rejected the verified exact sm_{major}{minor} native cubin \
                 for {stem:?}: {error:?}; source={COMPILED_ARCH_SOURCE}; \
                 nvcc={COMPILED_NVCC_VERSION}. Refusing every fallback path."
            ));
            Err(error)
        }
    }
}

#[macro_export]
macro_rules! load_cuda_embedded_module {
    ($stem:literal) => {{ $crate::cuda::module_loader::load_module_for_current_context($stem) }};
}
