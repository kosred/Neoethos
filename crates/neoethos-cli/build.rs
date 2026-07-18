//! Force the linker to keep `libtorch_cuda` in the final binary when a CUDA
//! GPU feature is enabled. tch-rs only emits `cargo:rustc-link-lib=
//! torch_cuda`, but modern linkers strip the library because no symbols
//! from it are referenced directly — `tch::Cuda::device_count()` then
//! returns 0 even on a CUDA-enabled host.
//!
//! 2026-07-18 deep-audit fixes (mirrors crates/neoethos-app/build.rs):
//! - GATE: the old check looked only at the `gpu` ALIAS env, so a direct
//!   `--features gpu-nvidia` build silently skipped the force-link.
//! - MSVC: the old code emitted GNU-ld syntax unconditionally — link.exe
//!   does not understand `-Wl,…`/`-l…`, so every Windows GPU link would
//!   have FAILED. Windows uses `/INCLUDE:` on an exported torch_cuda
//!   symbol (`at::cuda::warp_size()`) instead.

fn main() {
    let gpu_on = std::env::var("CARGO_FEATURE_GPU").is_ok()
        || std::env::var("CARGO_FEATURE_GPU_NVIDIA").is_ok();
    if gpu_on {
        let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
        if let Ok(libtorch) = std::env::var("LIBTORCH") {
            if msvc {
                println!("cargo:rustc-link-arg-bins=/LIBPATH:{libtorch}/lib");
                println!("cargo:rustc-link-arg-bins=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
            } else {
                println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
                println!("cargo:rustc-link-arg-bins=-L{libtorch}/lib");
                println!("cargo:rustc-link-arg-bins=-ltorch_cuda");
                println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
            }
            println!("cargo:rerun-if-env-changed=LIBTORCH");
        } else {
            println!(
                "cargo:warning=neoethos-cli built with a CUDA GPU feature but LIBTORCH env \
                 not set; libtorch_cuda will not be force-linked and \
                 tch::Cuda::device_count() may return 0"
            );
        }
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU_NVIDIA");
}
