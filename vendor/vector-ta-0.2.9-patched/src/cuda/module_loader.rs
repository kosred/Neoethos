//! Architecture-agnostic CUDA module loading.
//!
//! # What this used to do, and why it was wrong
//!
//! Upstream embedded ONE cubin per kernel, compiled for `sm_89`, and preferred
//! it only when the running device reported compute capability EXACTLY 8.9:
//!
//! ```text
//! const TARGET_CUBIN_MAJOR: i32 = 8;
//! const TARGET_CUBIN_MINOR: i32 = 9;
//! ...
//! Ok(major == TARGET_CUBIN_MAJOR && minor == TARGET_CUBIN_MINOR)
//! ```
//!
//! Two cards were named in source — one in those constants, one in the
//! `_sm89.cubin` filename the macro reached for — and everything else fell
//! through to a PTX that `build.rs` had also compiled for a single
//! architecture. Since PTX runs FORWARD only, a build made for Ada would not
//! load on Ampere at all. Pinning one architecture in source is the defect
//! class that cost this project eight months.
//!
//! # What it does now
//!
//! `build.rs` emits, per kernel source:
//!
//! * `<stem>.fatbin` — a fat binary carrying real SASS for every architecture
//!   in `VECTOR_TA_CUDA_ARCHS` (default `sm_80, sm_86, sm_89, sm_90`) PLUS
//!   embedded PTX at the highest of them; and
//! * `<stem>.ptx` — standalone PTX at the LOWEST of them, as a fallback for
//!   the case where the fatbin container itself cannot be loaded.
//!
//! Loading order here is therefore:
//!
//! 1. **fatbin.** The driver selects the exact SASS for the present device, or
//!    JITs the embedded PTX when the device is newer than anything compiled.
//!    This is the path that satisfies sm_80 / sm_86 / sm_89 / sm_90 from ONE
//!    build with no source change and no rebuild flag change.
//! 2. **PTX.** Only if the fatbin is absent (a prebuilt-staged tree that
//!    supplied none) or unloadable.
//! 3. **A named, actionable error** — never a silent success, never a quiet
//!    degradation to another precision or to the host.
//!
//! # Why the error is printed here rather than returned
//!
//! `load_module_for_current_context` returns `CudaResult<Module>` and is
//! reached from 309 call sites through [`load_cuda_embedded_module!`], each
//! inside a wrapper with its own error enum that carries
//! `#[from] cust::error::CudaError`. `CudaError` is a bare C enum and cannot
//! carry a message. Widening the signature would rewrite 309 wrappers; instead
//! the diagnostic is printed unconditionally (not behind a debug flag) AND
//! stashed in [`last_module_load_failure`] so a caller that wants to build a
//! richer error — `neoethos_data::core::gpu_indicators::GpuIndicatorEngine`
//! does — can quote it verbatim. The `Err` still propagates; nothing is
//! swallowed.

use std::env;
use std::sync::Mutex;
use std::sync::OnceLock;

use cust::{
    context::CurrentContext,
    device::DeviceAttribute,
    error::{CudaError, CudaResult},
    module::{Module, ModuleJitOption, OptLevel},
};

const FORCE_PTX_ENV: &str = "VECTOR_TA_CUDA_FORCE_PTX";
const FORCE_FATBIN_ENV: &str = "VECTOR_TA_CUDA_FORCE_FATBIN";
const DEBUG_ENV: &str = "CUDA_MODULE_LOAD_DEBUG";

/// The architectures `build.rs` compiled real SASS for, e.g. `"sm_80,sm_86"`.
/// `"unknown"` when the kernels were staged from a prebuilt tree rather than
/// compiled here — in which case we genuinely cannot name what is inside, and
/// say so instead of guessing.
pub const COMPILED_ARCHS: &str = env!("VECTOR_TA_CUDA_ARCHS");

/// The virtual architecture of the PTX embedded in the fatbin for forward
/// JIT onto cards newer than anything compiled.
pub const COMPILED_PTX_ARCH: &str = env!("VECTOR_TA_CUDA_PTX_ARCH");

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("True")
    )
}

fn debug_enabled() -> bool {
    env_flag(DEBUG_ENV)
}

/// Compute capability of the device bound to the current context, as
/// `(major, minor)`. Read from the driver — never inferred from a build-time
/// string, because the entire point is to compare COMPILED against PRESENT.
fn current_device_capability() -> CudaResult<(i32, i32)> {
    let device = CurrentContext::get_device()?;
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    Ok((major, minor))
}

fn device_arch_string() -> String {
    match current_device_capability() {
        Ok((major, minor)) => format!("sm_{major}{minor}"),
        Err(err) => format!("<unreadable: {err:?}>"),
    }
}

fn failure_slot() -> &'static Mutex<Option<String>> {
    static SLOT: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// The full diagnostic from the most recent module-load failure, if any.
///
/// Exists because `CudaError` cannot carry a message (see the module docs).
/// Callers that build a richer error should quote this rather than paraphrase
/// it. Reading does not clear it — the value is overwritten by the next
/// failure.
pub fn last_module_load_failure() -> Option<String> {
    match failure_slot().lock() {
        Ok(g) => g.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

fn record_failure(message: String) {
    // Print unconditionally. A module that will not load is not a debug-level
    // event: every indicator that needs it is about to fail, and the operator
    // needs the two architectures side by side to know what to rebuild.
    eprintln!("{message}");
    let mut guard = match failure_slot().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    *guard = Some(message);
}

fn load_ptx_module(ptx: &str) -> CudaResult<Module> {
    let jit_opts = &[
        ModuleJitOption::DetermineTargetFromContext,
        ModuleJitOption::OptLevel(OptLevel::O2),
    ];

    Module::from_ptx(ptx, jit_opts)
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
}

/// The `.target` directive of a PTX module, e.g. `sm_80` — so the error can
/// state what the fallback PTX was actually compiled for rather than what we
/// believe it was.
fn ptx_target(ptx: &str) -> Option<&str> {
    ptx.lines()
        .find_map(|line| line.trim().strip_prefix(".target "))
        .map(|t| t.split(',').next().unwrap_or(t).trim())
}

pub(crate) fn load_module_for_current_context(
    ptx: &str,
    fatbin: &[u8],
    stem: &str,
) -> CudaResult<Module> {
    let force_ptx = env_flag(FORCE_PTX_ENV);
    let force_fatbin = env_flag(FORCE_FATBIN_ENV);

    // A zero-byte fatbin is the placeholder `build.rs` writes when a kernel was
    // filtered out or a prebuilt tree supplied none. It is not an error — it
    // just means this stem has PTX only.
    let have_fatbin = !fatbin.is_empty();

    let mut fatbin_err: Option<CudaError> = None;

    if have_fatbin && !force_ptx {
        match Module::from_fatbin(fatbin, &[]) {
            Ok(module) => {
                if debug_enabled() {
                    eprintln!(
                        "[cuda-module-loader] {stem}: loaded fatbin ({COMPILED_ARCHS} + \
                         {COMPILED_PTX_ARCH} PTX) on {}",
                        device_arch_string()
                    );
                }
                return Ok(module);
            }
            Err(err) => {
                fatbin_err = Some(err);
                if debug_enabled() {
                    eprintln!(
                        "[cuda-module-loader] {stem}: fatbin load failed ({err:?}); trying the \
                         standalone PTX fallback"
                    );
                }
            }
        }
    }

    if force_fatbin && !force_ptx {
        // The operator explicitly demanded the fatbin lane; falling through to
        // PTX would hide exactly what they asked to observe.
        let message = format!(
            "vector-ta: {VECTOR_TA_FORCE_FATBIN_LABEL} is set and the fatbin for `{stem}` could \
             not be loaded on {device}.\n  compiled SASS : {COMPILED_ARCHS}\n  \
             embedded PTX  : {COMPILED_PTX_ARCH}\n  driver error  : {fatbin_err:?}\n\
             Refusing to fall back to the standalone PTX because {VECTOR_TA_FORCE_FATBIN_LABEL} \
             was requested.",
            device = device_arch_string(),
            VECTOR_TA_FORCE_FATBIN_LABEL = FORCE_FATBIN_ENV,
        );
        record_failure(message);
        return Err(fatbin_err.unwrap_or(CudaError::InvalidImage));
    }

    match load_ptx_module(ptx) {
        Ok(module) => {
            if debug_enabled() {
                eprintln!(
                    "[cuda-module-loader] {stem}: loaded standalone PTX ({}) on {}",
                    ptx_target(ptx).unwrap_or("<no .target>"),
                    device_arch_string()
                );
            }
            Ok(module)
        }
        Err(ptx_err) => {
            let device = device_arch_string();
            let ptx_arch = ptx_target(ptx).unwrap_or("<no .target directive>");
            let message = format!(
                "\n\
                 === vector-ta: no CUDA module for `{stem}` can run on this device ===\n\
                 \n\
                   device present : {device}\n\
                   fatbin SASS    : {COMPILED_ARCHS}{fatbin_note}\n\
                   fatbin PTX     : {COMPILED_PTX_ARCH} (forward-JIT for newer cards)\n\
                   standalone PTX : {ptx_arch}\n\
                 \n\
                   fatbin error   : {fatbin_err:?}\n\
                   PTX error      : {ptx_err:?}\n\
                 \n\
                 Both SASS and PTX run FORWARD only: code built for a NEWER architecture never \
                 runs on an OLDER device. If `{device}` is below every architecture listed above, \
                 rebuild with it included — the set is a build input, not a source constant:\n\
                 \n\
                   CUDA_ARCHS={device_num},80,86,89,90 cargo build -p vector-ta \
                 --features cuda-build-ptx\n\
                 \n\
                 If `{device}` is at or above them, the fatbin should have JIT-ed the embedded \
                 PTX; that failing points at the driver rather than the build — check that the \
                 installed driver is new enough for the CUDA toolkit these kernels were compiled \
                 with.\n\
                 \n\
                 NOT falling back to the CPU and NOT falling back to a lower precision. This is \
                 an error by design.\n\
                 ==========================================================================\n",
                fatbin_note = if have_fatbin {
                    ""
                } else {
                    " (none embedded for this stem)"
                },
                device_num = device
                    .strip_prefix("sm_")
                    .unwrap_or(&device)
                    .to_string(),
            );
            record_failure(message);
            Err(ptx_err)
        }
    }
}

/// Load the module compiled from `<stem>.cu`.
///
/// Both artifacts are embedded at compile time: the multi-architecture fatbin
/// (preferred) and the standalone PTX (fallback). Neither names a card —
/// `build.rs` decides the set and records it in `VECTOR_TA_CUDA_ARCHS`.
#[macro_export]
macro_rules! load_cuda_embedded_module {
    ($stem:literal) => {{
        $crate::cuda::module_loader::load_module_for_current_context(
            include_str!(concat!(env!("OUT_DIR"), "/", $stem, ".ptx")),
            include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, ".fatbin")),
            $stem,
        )
    }};
}
