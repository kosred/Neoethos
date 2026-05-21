//! Force the linker to keep `libtorch_cuda` in the final binary when
//! the `gpu` feature is enabled. tch-rs only emits `cargo:rustc-link-lib=
//! torch_cuda`, but modern linkers strip the library because no symbols
//! from it are referenced directly — `tch::Cuda::device_count()` then
//! returns 0 even on a CUDA-enabled host. Adding `-Wl,--no-as-needed`
//! around the link flag prevents the strip and lets the runtime see
//! the GPU.
//!
//! See `docs/audits/gpu_migration_2026-05-11.md` for context.

fn main() {
    if std::env::var("CARGO_FEATURE_GPU").is_ok() {
        if let Ok(libtorch) = std::env::var("LIBTORCH") {
            // Search path for the linker.
            println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg-bins=-L{libtorch}/lib");
            println!("cargo:rustc-link-arg-bins=-ltorch_cuda");
            println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
            println!("cargo:rerun-if-env-changed=LIBTORCH");
        } else {
            println!(
                "cargo:warning=neoethos-cli built with `gpu` feature but LIBTORCH env not set; \
                 libtorch_cuda will not be force-linked and tch::Cuda::device_count() may return 0"
            );
        }
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU");
}
