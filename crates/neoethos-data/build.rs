//! Refuses one build configuration, and records nothing.
//!
//! # What this file used to do, and why that was wrong
//!
//! It used to resolve a CUDA architecture — `CUDA_ARCHS`, else `CUDA_ARCH`,
//! else the minimum compute capability reported by `nvidia-smi` — and bake it
//! into `NEOETHOS_VECTOR_TA_PTX_ARCH` so the runtime preflight could quote it.
//!
//! Two things were wrong with that.
//!
//! 1. **It was a second source of truth for a number vector-ta already
//!    exports.** `vendor/vector-ta-0.2.9-patched/build.rs` compiles ONE fatbin
//!    covering `[80, 86, 89, 90]` plus embedded `compute_90` PTX for forward
//!    JIT, and records exactly that in `VECTOR_TA_CUDA_ARCHS` /
//!    `VECTOR_TA_CUDA_PTX_ARCH`, surfaced as
//!    `vector_ta::cuda::module_loader::{COMPILED_ARCHS, COMPILED_PTX_ARCH}`.
//!    The number this script computed was a fact about the BUILD HOST, printed
//!    in the diagnostic that exists to explain an arch mismatch — so a 4090
//!    build box producing a fully portable binary reported `sm_89` in the
//!    message meant to explain why an sm_86 card failed. Two independent
//!    sources for one number is how the two-Settings and two-config-files
//!    defects started.
//!
//! 2. **It made the build depend on the build host having a card.** Resolution
//!    fell through to `nvidia-smi`, and failure PANICKED — so
//!    `cargo build -p neoethos-data --features gpu-cuda` could not run in a
//!    CUDA container, a cross-build, or a CI image with the toolkit and no
//!    device, even though vector-ta no longer needs to know the host's card at
//!    all. Worse, the panic's own remedy told the operator to export
//!    `CUDA_ARCH=sm_86` — which in vector-ta's `target_archs()` takes the
//!    single-value branch and builds a SINGLE-ARCHITECTURE fatbin that will not
//!    load on an A100. The error message recreated the exact trap it existed to
//!    prevent.
//!
//! Both are gone. `core::indicator_telemetry::VECTOR_TA_PTX_ARCH` is now a
//! re-export of vector-ta's own constant, so there is one answer and it
//! describes the KERNELS rather than the machine that compiled them.
//!
//! # What remains
//!
//! One refusal: `CUDA_FAST_MATH` must be off for a `gpu-cuda` build. That is
//! not an arch question and it is not redundant with anything.
//!
//! To narrow the fatbin for a faster iteration loop, use the LIST form —
//! `CUDA_ARCHS=86` — which vector-ta reads directly. Never `CUDA_ARCH=`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_FAST_MATH");

    if std::env::var_os("CARGO_FEATURE_GPU_CUDA").is_none() {
        return;
    }

    // ── Fast math is a parity question, not a performance one ────────────────
    //
    // `--use_fast_math` implies FMA contraction, flush-to-zero denormals and
    // approximate div/rcp/sqrt. vector-ta's kernels compute the indicator
    // columns that become `dataset.indicators`, and the fused Prototype B walk
    // multiplies them by the gene weights — so a changed indicator flips
    // `combined >= long_threshold` and therefore flips trades. The 147 GPU
    // parity fixtures hand that matrix in directly and would never see it.
    //
    // vector-ta's build.rs defaults it OFF (`fast_math_requested`) and returns
    // false for the f64 lane BEFORE it reads the env var at all, so the f64
    // kernels are never fast-mathed regardless. Refusing here covers the rest
    // of the crate: a NeoEthos binary that computes ANY indicator with
    // approximate arithmetic cannot make a parity claim, and there is no way to
    // tell from the run that it was built that way.
    //
    // NOTE for the rented-card checklist: this panic means you CANNOT prove the
    // f64 lane's fast-math opt-out with
    // `CUDA_FAST_MATH=1 cargo build -p neoethos-data --features gpu-cuda` — it
    // aborts before nvcc is invoked. Prove it against vector-ta directly:
    // `CUDA_FAST_MATH=1 cargo build -p vector-ta --features cuda-build-ptx`,
    // then grep the emitted `neoethos_f64_kernels.ptx` for `approx`.
    if let Ok(value) = std::env::var("CUDA_FAST_MATH") {
        if value != "0" {
            panic!(
                "neoethos-data was built with `--features gpu-cuda` and CUDA_FAST_MATH={value:?}.\n\
                 \n\
                 `--use_fast_math` turns on FMA contraction, flush-to-zero denormals and \
                 approximate div/sqrt/rcp in vector-ta's indicator kernels. Those kernels feed \
                 `dataset.indicators`, which the Prototype B walk multiplies by the gene weights \
                 — one contracted multiply-add is enough to move a stop/target boundary and \
                 change which trades a strategy takes. `neoethos-gpu-cuda/build.rs` forbids \
                 exactly this with `-fmad=false` and a measured 0.62 %% divergence to justify it.\n\
                 \n\
                 Unset CUDA_FAST_MATH (or set it to 0) and rebuild."
            );
        }
    }
}
