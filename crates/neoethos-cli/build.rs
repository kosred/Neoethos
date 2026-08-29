//! Build script for `neoethos-cli`.
//!
//! REMOVED 2026-08-02: this file used to force-link `libtorch_cuda` so that
//! `tch::Cuda::device_count()` would see the GPU at runtime — it emitted
//! `/INCLUDE:?warp_size@cuda@at@@YAHXZ` on MSVC and
//! `-Wl,--no-as-needed -ltorch_cuda` on GNU whenever a CUDA GPU feature was on
//! and `LIBTORCH` was set.
//!
//! Nothing provides that symbol any more. `tch` is optional in
//! neoethos-models and is enabled by NO feature — every call site sits behind
//! `#[cfg(feature = "tch")]` — and d4df966a dropped the `dep:tch` that
//! neoethos-search once carried. Forcing the linker to resolve a symbol from a
//! library that is not on the link line is LNK2001 followed by LNK1120: the
//! next Windows GPU release build would have failed at the link step.
//!
//! Linux tree-model runtimes are staged beside the final executable by their
//! owning sys crates. The final binary therefore carries a single `$ORIGIN`
//! RUNPATH. Windows already searches the executable directory for DLLs; adding
//! a linker search path there would weaken that portable, adjacent-file rule.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("Cargo must provide CARGO_CFG_TARGET_OS to neoethos-cli/build.rs");
    if target_os == "linux" {
        println!("cargo:rustc-link-arg-bin=neoethos-cli=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg-bin=neoethos-cli=-Wl,-rpath,$ORIGIN/../lib/neoethos-cli");
    }
}
