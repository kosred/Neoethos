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
//! The build script is kept (rather than deleted) so `cargo` does not have to
//! re-resolve the package layout, and so this note survives where the next
//! person will look for it.

fn main() {
    // Nothing to do. Kept deliberately empty — see the module comment.
    println!("cargo:rerun-if-changed=build.rs");
}
