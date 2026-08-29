//! Enforces the data crate's CUDA precision contract.
//!
//! CUDA architecture selection has exactly one repository-wide authority:
//! `NEOETHOS_CUDA_ARCHS`, validated by `vendor/cuda_build_arch.rs`. This build
//! script does not resolve, infer, or override architectures. It only rejects
//! approximate CUDA arithmetic for the indicator pipeline.

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
