# Exact Classic-TA CUDA Routing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the canonical, post-budget Classic/vector-ta feature graph on real f64 CUDA under `GpuOnly`, with exact receipts and no CPU/f32 fallback.

**Architecture:** Split allocation-free plan construction from execution. The planner derives ordered typed output nodes from the same admissions and working set used by production; preflight resolves every node before constructing one resident `GpuIndicatorEngine`; execution downloads f64 only at the existing host `FeatureFrame` boundary.

**Tech Stack:** Rust nightly, anyhow, rayon CPU reference, vendored vector-ta f64 CUDA registry/runtime, source-contract tests, RTX 3090 device tests.

---

## Chunk 1: Exact plan and fail-closed routing

### Task 1: Pin the production structural contract

**Files:**

- Create: `crates/neoethos-data/tests/classic_cuda_production_route_source_contract.rs`
- Inspect: `crates/neoethos-data/src/core/hpc_ta.rs`
- Inspect: `crates/neoethos-data/src/core/gpu_indicators.rs`

- [x] **Step 1: Write the failing source contract**

Assert that the `GpuOnly` branch calls a named exact-plan executor, that the
executor constructs exactly one `GpuIndicatorEngine`, and that its body contains
neither `compute_cpu`, `cpu_multi_period_all`, `Kernel::Auto`, nor a fallback
branch.

- [x] **Step 2: Run the standalone test to verify RED**

Run: `rustc --edition 2024 --test crates/neoethos-data/tests/classic_cuda_production_route_source_contract.rs -o target/source-contracts/classic_cuda_production_route_source_contract.exe && target/source-contracts/classic_cuda_production_route_source_contract.exe --nocapture`

Expected: FAIL because production currently unconditionally rejects `GpuOnly`
and has no exact-plan executor.

### Task 2: Build the exact typed Classic plan

**Files:**

- Create: `crates/neoethos-data/src/core/classic_cuda_plan.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Test: `crates/neoethos-data/src/core/classic_cuda_plan.rs`

- [x] **Step 1: Write planner unit tests**

Cover base admission order, `output_ids_for` exclusions, every dynamically listed typed pattern
outputs, historical period order, installed extended working-set order, and a
complete ordered missing-route manifest.

- [x] **Step 2: Implement plan types**

Add typed `ClassicCudaPlan`, `ClassicCudaNode`, parameter request, and value-kind
types. Construct the plan only after `VocabularyBudget`, `admit_indicators`, and
extended groups are finalized, but before `Candles`, CPU dispatch, CUDA context,
or output allocation.

- [x] **Step 3: Make preflight exact**

Resolve each node through the vector-ta f64 named-output authority. Return every
missing route in plan order. Do not consult raw `ALL_INDICATORS` independently
of admission and do not special-case a missing route out of the schema.

- [ ] **Step 4: Run planner tests in the coordinated Cargo lane**

Run: `cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda classic_cuda_plan -- --nocapture`

Expected: all planner tests pass with zero warnings under `RUSTFLAGS=-Dwarnings`.

## Chunk 2: One-engine f64 execution

### Task 3: Execute supported nodes and materialize only at FeatureFrame

**Files:**

- Modify: `crates/neoethos-data/src/core/gpu_indicators.rs`
- Modify: `crates/neoethos-data/src/core/classic_cuda_plan.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Test: `crates/neoethos-data/src/core/gpu_indicators.rs`

- [ ] **Step 1: Write a failing real-device lifecycle test**

Build a representative exact plan containing base, named multi-output,
historical, and extended nodes. Assert one engine/session, no input re-upload,
f64 device results, stable column order, exact shapes, and one host
materialization boundary.

- [x] **Step 2: Implement the minimal executor**

Create one engine after successful preflight, launch nodes in plan order, retain
device matrices until all launches succeed, synchronize, download f64, and build
the existing column/ledger/report shape. Do not add an error arm that invokes a
CPU helper.

- [x] **Step 3: Wire `GpuOnly` to the executor**

The CPU body remains the `CpuOnly`/`Auto` route. `GpuOnly` returns directly from
the exact CUDA executor so it cannot fall through into `Kernel::Auto`.

- [x] **Step 4: Re-run the standalone source contract**

Expected: PASS and a non-zero test count.

### Task 4: Close every admitted output gap without changing schema

**Files:**

- Modify as required: `crates/neoethos-data/src/core/gpu_indicators.rs`
- Modify as required: `vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs`
- Modify as required: relevant vendored f64 CUDA kernel/wrapper files
- Test: `crates/neoethos-data/src/core/gpu_indicators.rs`

- [ ] **Step 1: Print the exact admitted gap manifest**

Use the planner result, not `gpu_only_classic_ta_output_gaps()` over the raw
registry. Record indicator, output, parameter identity, and value kind.

- [ ] **Step 2: Treat each missing route as RED**

Add one exact CPU-oracle parity test per truly missing output family before its
implementation. `ma`, `ma_batch`, and `ma_stream` remain requested if admitted;
their dispatcher nature is a routing problem, not permission to delete them.
`pattern_recognition` retains its typed discrete library-declared schema.

- [ ] **Step 3: Implement and verify one family at a time**

No exclusions, placeholder values, f32 widening, or CPU substitution.

## Chunk 3: Verification and evidence

### Task 5: Compile and run the complete data gate

**Files:**

- Verify: `crates/neoethos-data/**`
- Verify: affected vendored vector-ta files

- [ ] **Step 1: Format and inspect the exact diff**

Run pinned rustfmt on changed Rust files. Inspect INFO, then WARNING, then ERROR
from the full build log.

- [ ] **Step 2: Run warning-denied all-target compilation**

Run: `cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda --all-targets --no-run`

- [ ] **Step 3: Run pure planner and fail-loud tests without a card dependency**

Expected: exact test count, zero skips, zero warnings.

- [ ] **Step 4: Run RTX 3090 real-device tests**

Set the coordinated CUDA environment, run the complete `gpu_indicators` and
`hpc_ta` CUDA groups, capture 100 ms `nvidia-smi` telemetry, and prove non-zero
GPU utilization/memory/power while the named kernels execute.

- [ ] **Step 5: Run canonical `GpuOnly` feature build parity**

Compare names, order, values, validity, ledger, and execution report against the
CPU reference at the established f64 tolerances. Inject one unsupported output
and prove the run fails before the first CUDA launch and with zero CPU work.

- [ ] **Step 6: Commit only after root review**

Stage the exact data/vendor/test/docs paths; do not absorb unrelated dirty
worktree changes.
