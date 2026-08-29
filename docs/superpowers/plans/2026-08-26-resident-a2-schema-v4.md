# Resident A2 Schema V4 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Data-owned global schema/identity assembler and connect exact resident producers without changing canonical CPU feature order or minting authority from caller data.

**Architecture:** Column producers emit move-only local route/batch drafts and exact memory receipts. Data seals them once in the canonical schema-v4 order, while runtime transforms stay outside column coverage; final `FeaturePlanV1` and provenance are created only after normalization evidence.

**Tech Stack:** Rust, sha2, neoethos feature/GPU contracts, cust/CUDA C ABI, standalone `rustc --test` source contracts, rustfmt.

---

## Chunk 1: Global schema and source authority

### Task 1: Freeze the Data-owned schema order and typed draft invariants

**Files:**
- Create: `crates/neoethos-data/tests/gpu_resident_feature_recipe_v4_source_contract.rs`
- Create: `crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`

- [ ] Write a failing structural contract requiring the seven-family schema
  order, separate transform inventory, move-only local route/batch drafts,
  contiguous `1..=64` batches, duplicate-name rejection, and one global seal.
- [ ] Run the standalone test and confirm failure because the module is absent.
- [ ] Implement producer-local drafts and a Data-owned assembler that assigns
  global ordinals once and hashes the complete route contract.
- [ ] Accept private typed canonical parameters and derive tuple hashes in Data;
  reject caller/draft-supplied parameter hashes.
- [ ] Reconstruct capability-manifest order by producer independently of schema
  insertion order; keep production sealing unreachable until all seven column
  drafts and all three transform receipts exist.
- [ ] Add mutation fixtures for producer reordering, missing producer, batch
  gap, guessed zero hash, and adding transform pseudo-columns.
- [ ] Run the standalone test GREEN and rustfmt only touched Rust files.

### Task 2: Carry exact pinned source bindings and leases

**Files:**
- Create: `crates/neoethos-data/tests/pinned_resident_source_descriptor_v1_source_contract.rs`
- Modify: `crates/neoethos-data/src/core/pinned_canonical_series_v1.rs`
- Modify: `crates/neoethos-data/src/core/canonical_ohlcv.rs`

- [ ] Write RED assertions for a move-only crate-private resident descriptor,
  all direct generations, exact manifests/segments, and retained leases.
- [ ] Add a consuming conversion from `PinnedCanonicalSeriesV1`; derive
  `SourceArtifactBindingV1` only from private manifest fields.
- [ ] Restrict the first descriptor to full-generation segments; arbitrary
  cutoffs require a later indexed timestamp authority.
- [ ] Prove no decode/materialization method is called and no hash/segment can
  be supplied by the caller.
- [ ] Run source contracts GREEN and rustfmt touched files.

## Chunk 2: Classic local draft and exact pre-device memory

### Task 3: Defer Classic global receipt creation

**Files:**
- Modify: `crates/neoethos-data/tests/gpu_resident_classic_ta_v3_red_contract.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs`

- [ ] Add RED assertions forbidding zero-based `ResidentFeatureRouteV3` creation
  during Classic preflight and requiring typed local output drafts.
- [ ] Split the Classic plan into its local gpu-cuda recipe plus Data route
  fragments; preserve local destination indices for launch/pack execution.
- [ ] Add one Data sealing method that accepts the true global start ordinal and
  creates each hashed route exactly once.
- [ ] Project globally sealed bindings onto Classic local destinations so
  runtime append compares the admitted absolute span, not local ordinals.
- [ ] Prove local recipe bytes/math are unchanged by ordinal assignment.

### Task 4: Add the Classic pre-device memory receipt

**Files:**
- Modify: `crates/neoethos-gpu-cuda/tests/resident_classic_ta_v3_source_contract.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs`
- Modify only if unavoidable: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`

- [ ] Write RED assertions for exact per-launch output/additional-retained/
  scratch bytes before run-device consumption, sourced from the same runtime
  allocation authority.
- [ ] Expose a move-only memory draft that reuses owner-defined sizing logic;
  do not copy formula tables into Data.
- [ ] Bind the receipt to recipe identity, rows, launch order, and output width.
- [ ] Keep HCE/Fisher/FRAMA/FWMA bytes untouched and run source contracts only.

## Chunk 3: Session semantic-v2

### Task 5: Freeze the CPU oracle and native ABI

**Files:**
- Create: `crates/neoethos-data/tests/resident_session_v2_oracle.rs`
- Create: `crates/neoethos-gpu-cuda/tests/resident_session_v2_source_contract.rs`

- [ ] Freeze all 23 names/order, UTC windows, reset/state/ATR arithmetic order,
  mandatory volume, canonical NaN, validity codes, and the v2 dual-clock quirk
  where values infer timestamp units but validity consumes canonical ms.
- [ ] Require present-volume resident admission before device acquisition and
  retain an absent-volume/MissingInput CPU-only oracle at the Cargo/RTX gate.
- [ ] Freeze bits `0x7ff8000000000000`, validity `0/1/5`, `207N` retained bytes,
  zero additional/scratch/H2D/D2H, one launch/event, zero host sync, 736 bytes
  of pointer tables, and 1,377 bytes of isolated metadata.
- [ ] Add CUDA RED assertions for one sequential f64 kernel, exact allocations,
  zero scratch/feature D2H, same context/non-default stream, and one event.
- [ ] Run the structural CUDA source contract RED locally. The behavioral CPU
  integration oracle requires Cargo `--extern` resolution and is deferred to
  the parent-owned RTX Cargo gate.

### Task 6: Implement and integrate Session-v2

**Files:**
- Create: `crates/neoethos-gpu-cuda/native/resident_session_v2.cu`
- Create: `crates/neoethos-gpu-cuda/src/resident_session_v2.rs`
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Modify: `crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs`
- Modify: `crates/neoethos-data/src/core/feature_registry.rs`
- Modify: `crates/neoethos-data/tests/gpu_resident_feature_recipe_v4_source_contract.rs`
- Modify: `crates/neoethos-data/tests/gpu_only_feature_workspace_preflight_v3_source_contract.rs`
- Create: `crates/neoethos-data/tests/session_semantic_v2_source_closure_contract.rs`

- [ ] Implement one-thread ascending-row CUDA state machine and opaque owner.
- [ ] Preserve two independent clock/state lanes and the first-sixteen-nonzero
  timestamp-unit inference; do not silently correct this under semantic-v2.
- [ ] Add typed allocation/lifetime/runtime receipts and generic batch append.
- [ ] Add Data component sealing/binding/runtime-equality validation.
- [ ] Carry the move-only Session receipt through recipe preflight, bound
  admission, seal token, and sealed store; expose runtime evidence only through
  `session_runtime_receipt_v2`.
- [ ] Bind exact Session route domain/indicator/output/stage/no-period and typed
  dual-clock/windows/ATR/output-session parameters before global sealing.
- [ ] Add `features.rs` to the Session-v2 semantic source closure in
  `feature_registry.rs` with a focused closure source contract, preserving all
  Classic-v9/Fisher bytes outside that hunk.
- [ ] Schedule Session only after preceding admitted schema ranges are ready and
  retire its pack event before the next producer.
- [ ] Keep Session capability unadvertised until its complete admitted route and
  runtime receipt path is reachable.

Completing the Session owner/local draft/component is not capability promotion:
Quant precedes Session, so production sealing remains fail-closed until the
preceding Quant span exists.

## Chunk 4: Existing producers and final identity

### Task 7: Add local drafts for existing SMC, Regime, and Footprint

**Files:**
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Modify: focused Data source contracts.

- [ ] Derive exact local fragments from existing registry/output authorities.
- [ ] Bind existing owner-supplied batch memory and semantic capabilities.
- [ ] Prove SMC precedes Classic in schema while runtime consumes globally
  sealed bindings in ascending admitted spans.

### Task 7A: Schedule Regime and Footprint through admitted routes

**Files:**
- Modify: `crates/neoethos-data/tests/gpu_resident_feature_store_v3_source_contract.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`

- [ ] RED-require exact route-driven Regime then Footprint apply calls before
  Robust normalization.
- [ ] Use existing prepared inputs and component receipts on the same carrier.
- [ ] Retire every producer batch event before the next append.
- [ ] Validate their existing runtime receipts during final seal.

### Task 8: Finalize plan and provenance after normalization evidence

**Files:**
- Create: `crates/neoethos-data/tests/gpu_resident_feature_identity_v4_source_contract.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`

- [ ] RED-forbid `final_identity_hashes()` before runtime sealing.
- [ ] Carry typed source bindings and plan-node templates in the seal token.
- [ ] Move the complete resident source descriptor and leases through preflight,
  admission, seal token, sealed store, and Search-consumer ownership.
- [ ] Build enabled/disabled final plans correctly after the validated fit
  digest, then construct and retain exact provenance.
- [ ] Pass only derived identities into the low-level sealed-store request.

## Chunk 5: Quant and HTF closure

### Task 9: Define the mandatory Quant-v3 typed-input migration

**Files:**
- Create: `crates/neoethos-data/tests/resident_quant_v3_oracle.rs`
- Create: `crates/neoethos-data/tests/resident_quant_v3_source_contract.rs`
- Modify after semantic approval: `crates/neoethos-data/src/core/quant_features.rs`
- Modify after Classic-v9 freezes: `crates/neoethos-data/src/core/feature_registry.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/core/timestamps.rs`
- Modify: `crates/neoethos-data/src/core/canonical_ohlcv.rs`
- Modify: `crates/neoethos-data/src/core/pinned_canonical_series_v1.rs`
- Modify: `crates/neoethos-data/src/core/direct_timeframes.rs`
- Modify: dataset-contract identity/temporal semantic closure.
- Future isolated Data/gpu-cuda Quant input and producer files.

- [ ] Inventory every currently permanent `MissingInput` column.
- [ ] Seal observations-per-year explicitly; derive D1/W1 facts only from
  pinned causal direct parents.
- [ ] For irreducible external market facts, freeze an explicit Quant-v3
  remove/replace migration before changing the schema.
- [ ] Seal validated uniform `timeframe_millis`, Session-v2-compatible UTC
  boundaries, versioned `trading_sessions_per_year=252`, and checked
  bars-per-session/day/week; never infer these empirically.
- [ ] Restrict Quant-v3 base recipes to fixed intraday timeframes whose grid
  divides the Asian/UTC-day session and supplies at least twelve bars per
  session (canonical M30 or finer); bind `quant_orb_4/8/12` to the 00:00 UTC
  Session-v2 Asian open and reject coarser/calendar bases before admission.
- [ ] Make D1/W1 hidden mandatory dependencies even when not selected as
  emitted HTFs, using next-direct-bar-open causal availability.
- [ ] Bind move-only annualization/D1/W1/session receipts, source bindings,
  retained parent/availability bytes, scratch, events, and deduplication with
  HTF-owned parents before admission.
- [ ] Freeze the exact 63-name v3 order and fail-closed v2 migration policy
  before changing the registry.
- [ ] Keep RobustNormalization-v2 unchanged and prove all 63 Quant-v3 columns
  have valid typed semantics; no permanent `MissingInput` may be promoted.
- [ ] Stop before production Quant changes unless the oracle freezes all 63
  formulas, validity/warmup rules, DTO canonical bytes, and migration policy.
- [ ] Implement only one complete, versioned 63-column capability.

### Task 10: Add retained direct-timeframe HTF ownership

**Files:**
- Future isolated Data/gpu-cuda HTF files.
- Modify: `crates/neoethos-data/src/core/pinned_canonical_series_v1.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`

- [ ] Carry selected direct parents and leases into the resident factory.
- [ ] RED-freeze existing causal availability and exact appended schema order.
- [ ] Implement resident alignment, exact memory/event receipts, and zero feature
  D2H.
- [ ] Promote the full manifest only after Quant/Session/HTF and all transforms
  are reachable through one materializer.

## Chunk 6: Verification and RTX handoff

### Task 11: Freeze the source overlay and remote gates

- [ ] Run all focused standalone source contracts and CPU oracles.
- [ ] Run rustfmt `--check` only on touched Rust files and inspect exact diffs.
- [ ] Freeze predecessor/successor SHA-256 for every touched/new file.
- [ ] Remote RTX: warning-denied Cargo all-targets, selected-CC NVCC/SASS,
  required-card component fixtures, enabled/disabled normalization, canonical
  full run, and first real search.
