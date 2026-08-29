# GPU-Resident ClassicTa V3 Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:subagent-driven-development` when the runtime permits delegated workers; otherwise use `superpowers:executing-plans`. This shared worktree currently forbids new subagents, so execute locally and preserve unrelated edits.

**Goal:** Replace `ResidentFeatureProducerV3::ClassicTa` in the strict GPU-only missing-capability census with one exact, bounded, same-primary-context resident producer. The producer consumes the immutable admitted Classic plan, launches only real vector-ta f64 CUDA routes, appends ordered batches of at most 64 columns into the V3 resident store, and never downloads or reuploads feature values.

**Architecture:** Data remains the canonical registry/accounting/planning authority and creates a non-authoritative immutable launch recipe before full-workspace admission. `neoethos-gpu-cuda` owns an opaque `ResidentClassicTaExecutorV3`, reconstructs no plan, revalidates recipe/device/build/route identities against the moved run-device carrier, and owns all vector-ta outputs, parameter buffers, validity bytes, events, and stream-ordered retirement. The parent dataset must retain exact OHLCV plus clock/SMC arrays on the carried context and stream; derived `hlc3`, `hl2`, and `hlcc4` are produced device-side. Capability admission remains fail-closed until every admitted route can execute through this authority.

**Tech stack:** Rust 2024/nightly, `cust` 0.3.2 primary-context ownership, optional vendored `vector-ta` native f64 CUDA, `neoethos-gpu-contracts` V3 receipts, source-contract tests, coordinated RTX 3090 warning-denied and Compute Sanitizer gates.

**Non-negotiable boundary:** No public context/stream/raw-pointer getter, no caller-minted authority, no second context or stream, no host FeatureFrame, no feature-value D2H/H2D, no f32/CPU fallback, no host synchronization, no ledger/schema/identity change. A public or crate-public recipe is only validated data; the opaque run-device carrier is authority.

---

## Task 1: Pin the parent-input successor and executor boundary RED

**Files:**

- Modify: `crates/neoethos-data/tests/gpu_resident_classic_ta_v3_red_contract.rs`
- Create: `crates/neoethos-gpu-cuda/tests/resident_classic_ta_v3_source_contract.rs`
- Inspect: `crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs`
- Inspect: `crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs`
- Inspect: `crates/neoethos-gpu-cuda/src/resident_smc_v3.rs`

- [ ] Assert the opaque parent contract retains exact open/high/low/close/volume/timestamps plus calendar/SMC rows and binds raw-bit content hashes for every canonical input.
- [ ] Assert `ResidentClassicTaExecutorV3` lives in `neoethos-gpu-cuda`, receives only a validated recipe and `GpuOnlyRunDeviceAdmissionV3`, and has no public raw context/stream/buffer constructor.
- [ ] Assert every launch batch is nonempty, monotonic, contiguous, at most 64 output columns, and carries exact output names/routes/parameter tuples.
- [ ] Assert source contains no `Context::new`, `Stream::new`, upload helper, download helper, `synchronize`, CPU dispatcher, f32 type, or fallback branch.
- [ ] Run the two direct `rustc --test -D warnings` source contracts and preserve intentional RED evidence.

## Task 2: Make the opaque parent a complete one-upload Classic input authority

**Files:**

- Modify: `crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-contracts/tests/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_smc_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/native/resident_smc_v3.cu`
- Modify: `crates/neoethos-data/tests/gpu_resident_smc_v3_source_contract.rs`

- [ ] RED: prove open/volume are absent from retained parent layout/bytes/hashes and are dropped or never uploaded.
- [ ] Extend the sealed parent layout and source trait with exact open and volume device buffers and f64-bit hashes; update checked steady/peak byte accounting. Treat this as a declared successor hash, not silent drift.
- [ ] Retain the existing one-time SMC open allocation and add volume to the same pinned H2D transaction; record the producer event only after both inputs and every SMC parent/feature output are ready.
- [ ] Preserve the original f64 bit patterns and typed validity; do not canonicalize nonfinite payloads.
- [ ] Prove stream-ordered error/drop handling for the two new retained buffers and no ordinary `DeviceBuffer` free while work may be queued.

## Task 3: Freeze one immutable Data Classic launch recipe before workspace admission

**Files:**

- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/src/core/classic_cuda_plan.rs`
- Create: `crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Test: `crates/neoethos-data/tests/gpu_resident_classic_ta_v3_red_contract.rs`

- [ ] RED: prove the current `ClassicTaRunPlan` exposes no immutable resident projection and that rebuilding the plan would require a second RAM/budget probe.
- [ ] Add a crate-visible projection that borrows/copies only the already-resolved admitted base IDs, exact historical/extended groups and periods, admission free-bytes evidence, and working-set identity. Do not call `VocabularyBudget::for_run`, `current_extended_sweep_working_set`, or registry admission twice.
- [ ] Convert the existing fully preflighted `ResolvedClassicCudaLaunch` graph into an ordered gpu-cuda recipe with a domain-separated SHA-256 over rows, stage, indicator/output/column names, value kind, exact parameter types/bits, input kind, first-valid rule, and expected entry point.
- [ ] Reject any unresolved/gapped/discrete route before the run-device carrier is consumed. The recipe is never authority and has no constructor outside Data planning.
- [ ] Bind exact batch boundaries to resolved launch groups; never split one multi-output launch and never exceed 64 columns.

## Task 4: Add the opaque gpu-cuda Classic executor and optional vector-ta edge

**Files:**

- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`
- Modify as required: `vendor/vector-ta-0.2.9-patched/src/cuda/device_types_f64.rs`
- Test: `crates/neoethos-gpu-cuda/tests/resident_classic_ta_v3_source_contract.rs`

- [ ] Add optional `vector-ta` under the existing gpu-cuda `cuda` feature without creating a dependency cycle or changing the CPU/default graph.
- [ ] Construct `CudaSession::from_parts` and `CudaF64Indicators::from_session` only from the carrier-owned primary context and run stream; revalidate ordinal, UUID/context identity, SASS/build manifests, route-plan hash, and exact math authority before any output allocation.
- [ ] Derive `hlc3`, `hl2`, and `hlcc4` in one exact f64 CUDA kernel/launch set on the same stream, with CPU operation order pinned; retain them only as executor scratch and charge them in the Classic producer batch ledger.
- [ ] Move/refactor the existing typed launch dispatch into the executor rather than creating a second arithmetic or output-route authority. Data sends the resolved recipe; gpu-cuda matches every recipe variant to the existing vector-ta f64 primary/named method and checks returned indicator/output IDs and shapes.
- [ ] Add `into_resident_parts_v3` ownership transfer for vector-ta named results so parameter/scratch/output buffers can be retired stream-ordered without `F64NamedOutputsResult::Drop` synchronizing. Preserve the standalone compatibility API and its current Drop behavior outside the resident path.
- [ ] Do not expose `CudaF64Indicators`, `CudaSession`, context/stream `Arc`s, device pointers, or raw output buffers across the crate boundary.

## Task 5: Produce exact device validity and opaque appendable batches

**Files:**

- Modify: `crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs`
- Create: `crates/neoethos-gpu-cuda/native/resident_classic_ta_v3.cu`
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Test: `crates/neoethos-gpu-cuda/tests/resident_classic_ta_v3_source_contract.rs`

- [ ] Implement device-side validity classification matching `hpc_ta::classify_classic_ta_validity`: warmup prefix, valid finite, nonfinite after start, compute failure, and missing-input/preflight placeholder semantics. Infinity is a fail-loud producer error where the CPU authority rejects it.
- [ ] Preserve every output f64 bit. Validity is producer-owned u8 0..9; the shared resident store alone packs it to canonical u4.
- [ ] Return opaque one-shot `PendingResidentClassicTaBatchV3` values. Each owns output matrices plus parameter/scratch and one ready event; it can only append to `ResidentFeatureStoreAssemblerV3` and then retire via event polling.
- [ ] Enforce exactly one in-flight producer batch: append, poll retirement, then launch the next group. Peak accounting records final store + max one Classic group/scratch, never all Classic outputs.
- [ ] Add native symbol/build-source contract so an uncompiled CUDA source cannot masquerade as integration.

## Task 6: Wire the A2 factory without weakening the missing census

**Files:**

- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs`
- Modify: `crates/neoethos-data/tests/gpu_resident_classic_ta_v3_red_contract.rs`
- Modify: `crates/neoethos-data/tests/gpu_resident_feature_store_v3_source_contract.rs`

- [ ] Preflight the exact Classic recipe before `into_gpu_only_run_device_admission_v3` consumes the full-workspace carrier.
- [ ] After SMC batch retirement, move the same opaque carrier-owned executor state through all Classic groups; no caller may obtain or replace its context/stream/parent buffers.
- [ ] Add `resident_classic_ta_capability_v3()` only after the executor proves every admitted recipe route and output. Then update the exact missing list from nine to eight in canonical enum order.
- [ ] Keep `materialize_gpu_only_feature_store_v3` fail-before-materialization while the remaining eight capabilities are absent. Do not fabricate dataset/plan/normalization hashes to reach the carrier.

## Task 7: Source freeze and coordinated RTX gates

**Files:** all paths changed above only.

- [ ] Run pinned rustfmt on changed Rust files and inspect every diff against recorded pre-hashes; preserve unrelated shared hunks.
- [ ] Run direct source-contract binaries with `-D warnings`; record exact counts and log hashes.
- [ ] Static census: no second context/stream/upload/download/sync/f32/CPU/fallback, no public raw accessors, all 10→9→8 capability transitions chronological, all changed manifests/build sources linked.
- [ ] Request root-run warning-denied gates, without claiming local Cargo/device validation:
  - `cargo +nightly-2026-04-07 test -p neoethos-gpu-cuda --features cuda --all-targets --no-run --locked`
  - `cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda --all-targets --no-run --locked`
  - focused Data source/planner contracts and gpu-cuda resident Classic contracts with exact nonzero counts.
- [ ] On RTX 3090, run exact CPU-oracle parity for base + admitted sweeps, output names/order/bits/validity, batch sizes 1/31/32/33/63/64, delayed-stream lifetime, injected invalid/nonfinite output refusal, and changed-last-bit observation after event wait.
- [ ] Run Compute Sanitizer memcheck + initcheck/leak proof; require ERROR SUMMARY 0, LEAK SUMMARY 0, no ignored/skipped/fallback tests, zero full-frame D2H/H2D, exactly one input upload, and max one live producer group.
- [ ] Freeze exact path hashes and append a durable checkpoint. Do not commit or merge without root/user direction.

