# Deviation f64 Stable Authority V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cancellation-prone VectorTA population-deviation implementation with one numerically stable, bit-identical CPU/stream/CUDA f64 authority that recovers throughput by parallelizing independent windows.

**Architecture:** A private Rust per-window implementation first applies one bit-derived global power-of-two input scale, then owns an anchored, Neumaier-compensated two-pass population variance. Scalar and AVX entry points call or lane-vectorize that single authority so CPU architecture cannot change the answer. The strict-f64 CUDA kernel gives independent output windows to CUDA threads and transcribes the same operation order; the existing f32 implementation remains separate.

**Tech Stack:** Rust 2024, vendored VectorTA, x86 AVX dispatch entry points, CUDA C++17 strict-f64, `rustc -Dwarnings`, Cargo nightly, RTX3090, Compute Sanitizer.

---

## Chunk 1: Authority tests and CPU state machine

### Task 1: Freeze the failing numerical and architecture-parity tests

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/deviation.rs`
- Create: `vendor/vector-ta-0.2.9-patched/tests/deviation_stable_f64_authority_contract.rs`

- [ ] Add a unit test using the nine exact RTX input bit patterns and population period 9.
- [ ] Assert the stable authority bit pattern `0x3efaabdbf86838c1` for scalar and stream.
- [ ] Under `nightly-avx`, assert explicit AVX2, AVX-512 when supported, and Auto match the same bits.
- [ ] Add constant, variance-collapse, large-offset/tiny-variance, subnormal, finite-extreme, monotone-drift, periods 1/2/9/50/200, and interior NaN/Inf recovery fixtures.
- [ ] Add a source contract requiring anchored scaling/Neumaier/two-pass APIs in Rust and CUDA and forbidding raw moments or rolling M2 in strict-f64 sections.
- [ ] Run the contract and focused unit tests before production changes; record failures caused by missing stable authority or divergent bits, not compile errors.

Commands:

```powershell
rustc --edition 2024 -Dwarnings --test vendor/vector-ta-0.2.9-patched/tests/deviation_stable_f64_authority_contract.rs -o target/audit-logs/deviation-stable-f64-authority-red.exe
$env:CARGO_MANIFEST_DIR=(Resolve-Path 'vendor/vector-ta-0.2.9-patched').Path
target/audit-logs/deviation-stable-f64-authority-red.exe --nocapture
$env:RUSTFLAGS='-Dwarnings'
cargo +nightly-2026-04-07 test --locked --manifest-path vendor/vector-ta-0.2.9-patched/Cargo.toml --lib deviation_stable_f64_authority_v1 -- --nocapture --test-threads=1
```

Expected: warning-clean compile, intentional assertion failures reproducing the raw-moment/AVX divergence.

### Task 2: Implement one private Rust stable per-window authority

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/deviation.rs`

- [ ] Add one Neumaier accumulator with fixed f64 operation order.
- [ ] Add exact bit-derived `floor_power_of_two_input_scale_v2`, including subnormal inputs.
- [ ] Scan and scale finite raw inputs before subtraction, then implement anchored shifted mean and centered-square passes in oldest-to-newest order.
- [ ] Emit canonical NaN on any non-finite input/intermediate and positive zero for a constant finite window.
- [ ] Evaluate every output window independently; do not retain rolling numerical state.
- [ ] Route `standard_deviation_rolling_into`, the finite fast path, legacy scalar API, matrix batch/prefix path, row APIs, devtype 3 alias, and `DeviationStream` through this per-window authority.
- [ ] First make AVX2/AVX-512 entry points delegate devtypes 0/3 to the shared authority; then lane-vectorize across independent output windows only if the exact-bit tests remain green.
- [ ] Run the focused unit and contract gates; require exact bits and warning-free output.

### Task 3: Validate CPU accuracy and bounded throughput

**Files:**
- Modify only if a test exposes a bug: `vendor/vector-ta-0.2.9-patched/src/indicators/deviation.rs`
- Evidence: `target/audit-logs/math-authority-v1/deviation-f64-v1-*`

- [ ] Compare every test window to an independent high-precision centered oracle.
- [ ] Record maximum ULP and relative error for each fixture/period.
- [ ] Benchmark the old raw implementation from the recorded pre-edit executable against V1 on representative 16,503-row period sweeps.
- [ ] Measure the O(N*period) behavior and investigate any material whole-search regression; never restore an inaccurate raw or rolling fallback to meet a microbenchmark.

## Chunk 2: Strict-f64 CUDA transcription and device proof

### Task 4: Write the strict-CUDA RED source checks

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/tests/deviation_stable_f64_authority_contract.rs`
- Modify after RED only: `vendor/vector-ta-0.2.9-patched/kernels/cuda/deviation_kernel.cu`
- Modify after RED only: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`

- [ ] Require CUDA global-input scaling before subtraction, Neumaier, two-pass, canonical-invalid, and independent-window tokens.
- [ ] Require explicit CUDA RN add/sub/div/mul/FMA/sqrt operations, with FMA only where Rust uses `mul_add`.
- [ ] Require the wrapper to launch Deviation with `GridSize::xy(output_blocks, combos)` before the generic sequential branch.
- [ ] Forbid strict-f64 raw `sumsq / n - mean * mean` and architecture-dependent reduction trees.
- [ ] Run and observe the CUDA-specific contract failure before editing the kernel.

### Task 5: Transcribe Stable Population Variance V1 into strict CUDA

**Files:**
- Modify: `vendor/vector-ta-0.2.9-patched/kernels/cuda/deviation_kernel.cu`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`

- [ ] Replace only `neoethos_deviation_batch_f64` and its private f64 helpers.
- [ ] Preserve the row/period/first-valid ABI, warmup, canonical NaN, and finite-only period-one zero.
- [ ] Change strict-f64 scheduling to parallelize independent output windows while retaining combination ownership and bounds.
- [ ] Match Rust's raw-input max scan, bit-derived scale, normalized anchor, Neumaier branches, two passes, and explicit rounding exactly.
- [ ] Leave all f32 kernels byte-identical.
- [ ] Run the source contract, rustfmt for Rust files, and `git diff --check`.

### Task 6: Build and test on RTX3090

**Files:**
- No new production files.
- Evidence: `/workspace/audit-logs/math-authority-v1/deviation-f64-v1-*`

- [ ] Overlay only the frozen changed paths after SHA verification and preserve remote predecessors.
- [ ] Run warning-denied release no-run builds with ambient architecture/fast-math variables unset.
- [ ] Run the exact deviation source contract and focused CPU tests remotely.
- [ ] Run `resident_classic_ta_v3_reviewed_routeable_subset_through_halftrend_is_exact_and_leak_free`; require exact CPU-f64/GPU-f64 bits.
- [ ] Resolve the exact built test executable, then run the same filter under Compute Sanitizer memcheck/leak-check; require one test, zero errors, zero leaks, no skip/fallback.

## Chunk 3: Regression closure and handoff

### Task 7: Run aggregate VectorTA/Data gates

**Files:**
- No planned edits.

- [ ] Run VectorTA warning-denied all-targets no-run with native CUDA.
- [ ] Run gpu-cuda and Data warning-denied release all-targets no-run.
- [ ] Run all previously frozen mathematical authority contracts and the exact 45-gap census.
- [ ] Compare complete INFO, WARNING, and ERROR logs and retain SHA256 evidence.

### Task 8: Freeze the result and resume the next real P0

**Files:**
- Update: `target/audit-logs/math-authority-v1/*` evidence only.

- [ ] Record pre/post SHA256 for every modified production/test path.
- [ ] Record RED, local GREEN, RTX GREEN, sanitizer, and benchmark evidence separately.
- [ ] State explicitly that this closes only deviation f64 authority, not the full indicator or Search objective.
- [ ] Resume the next highest-severity proven math/runtime mismatch from the frozen audit queue.

No commit step is included while the shared worktree contains unrelated user and agent changes. Integration will use exact-path/hash ownership rather than a broad commit that could capture unrelated work.
