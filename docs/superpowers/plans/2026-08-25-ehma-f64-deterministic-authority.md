# EHMA Deterministic f64 Authority V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace EHMA's divergent CPU/CUDA f64 routes with one accuracy-oriented exact-bit authority while preserving f32 behavior.

**Architecture:** A deterministic mirrored Hann coefficient builder supplies one compensated, scaled, anchored chronological window evaluator. Every CPU f64 API and the strict CUDA f64 kernel transcribe that same schedule; no libm/libdevice result or SIMD reassociation defines the output.

**Tech Stack:** Rust nightly f64/AVX dispatch, CUDA C++ strict-f64 kernels, standalone Rust source-contract tests, RTX 3090 device fixtures.

---

## Chunk 1: Contract and CPU authority

### Task 1: Add the RED source/numeric contract

**Files:**
- Create: `vendor/vector-ta-0.2.9-patched/tests/ehma_deterministic_f64_authority_contract.rs`

- [ ] Add the official Ehlers source URLs and the exact reviewed-routeable
  fixture inputs/bits.
- [ ] Assert the V2 identity, deterministic msun coefficient helpers,
  power-of-two scale, `TwoProd`/`TwoSum`, route unification, stream unification,
  strict CUDA transcription, and unchanged f32 entry points.
- [ ] Compile the contract directly with `rustc --test` and run it before any
  production edit.

Run:

```powershell
rustc --edition=2024 --test vendor/vector-ta-0.2.9-patched/tests/ehma_deterministic_f64_authority_contract.rs -o target/ehma_deterministic_f64_authority_contract.exe
target/ehma_deterministic_f64_authority_contract.exe --nocapture
```

Expected: numeric/source-independent assertions pass, production-source
assertions fail because V2 is absent.

### Task 2: Implement the CPU coefficient authority

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ehma.rs`

- [ ] Add bit-pinned pi constants and the V2 semantic identity.
- [ ] Transcribe the bounded FreeBSD-msun sine/reduction schedule already used
  by the strict Correlation Cycle authority.
- [ ] Add double-double half-angle construction, `2*sin^2`, exact mirroring,
  and compensated coefficient accumulation.
- [ ] Add private coefficient tests for period 1, period 14 expected bits, and
  symmetry over every period 1..512.

### Task 3: Implement the common CPU window evaluator

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/ehma.rs`

- [ ] Add floor-power-of-two scaling, canonical qNaN handling, `TwoSum`,
  `TwoProd`, and the chronological indexed window helper.
- [ ] Route direct Scalar/Auto/AVX/WASM and both batch implementations through
  the helper; remove duplicate forward/reversed weight logic and AVX reduction.
- [ ] Replace `EhmaStream`'s rotating recurrence with the same mirrored weights,
  coefficient, logical-ring evaluation, and gap recovery.
- [ ] Add exact route tests for row 13, constants, large offsets, subnormals,
  period edges, gaps, and recovery.
- [ ] Run rustfmt on the one Rust production file and re-run the source contract.

## Chunk 2: Strict CUDA transcription and handoff

### Task 4: Implement the strict CUDA f64 authority

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/ehma_kernel.cu`

- [ ] Leave every f32 line before the `S4 f64 LANE` marker unchanged.
- [ ] Transcribe the same constants, angle residual, msun sine, mirrored
  coefficients, coefficient sum, scale, anchored compensated dot, and
  non-finite behavior into the f64 section.
- [ ] Use strict RN intrinsics for intentional FMA/division/multiplication and
  keep separate operations separate under the existing `-fmad=false` policy.
- [ ] Re-run the standalone source contract and verify the f32 prefix hash is
  identical to its predecessor.

### Task 5: Local light verification and exact handoff

**Files:**
- Verify: all files above

- [ ] Run rustfmt and `git diff --check` on the touched paths.
- [ ] Re-read the complete diff, confirming no unrelated dirty hunk changed.
- [ ] Record predecessor/successor SHA-256 hashes and the exact RED/GREEN
  source-contract output.
- [ ] Hand the parent these RTX commands without running local Cargo/NVCC:

```bash
cargo +nightly-2026-04-07 test --locked \
  --manifest-path vendor/vector-ta-0.2.9-patched/Cargo.toml \
  --features nightly-avx --lib ehma --release -- --nocapture --test-threads=1

cargo +nightly-2026-04-07 test --locked \
  -p neoethos-data --features gpu-cuda-device-fixtures --release \
  core::hpc_ta::gpu_resident_classic_ta_v3_device_tests::resident_classic_ta_v3_reviewed_routeable_subset_through_halftrend_is_exact_and_leak_free \
  -- --exact --nocapture --test-threads=1
```

Run both with `CUDA_VISIBLE_DEVICES=0`, `NEOETHOS_REQUIRE_GPU=1`, warning-denied
Rust flags, and all CUDA architecture/fast-math overrides unset. If exact
parity passes, repeat the device fixture under Compute Sanitizer and benchmark
V1 versus V2 before the full Windows/Linux/CUDA release matrix.
