# VRAM-first resident feature projection implementation plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the impossible full resident feature store with exact two-pass CUDA screening and compact materialization sized from the selected GPU.

**Architecture:** Resolve one immutable recipe, score it in bounded producer batches, seal a small selected-column map, replay the recipe into a compact store, and size Search from the same free-VRAM/topology authority. CPU memory policy remains separate.

**Tech Stack:** Rust nightly, CUDA C++, CUB, existing resident producer/prefilter/workspace APIs.

---

### Task 1: RED contracts

**Files:**
- Modify: `crates/neoethos-gpu-cuda/tests/resident_trim_prefilter_v1_source_contract.rs`
- Modify: `crates/neoethos-data/tests/gpu_resident_feature_store_v3_source_contract.rs`
- Modify: `crates/neoethos-search/tests/canonical_native_discovery_entry_v1_contract.rs`

- [ ] Require a versioned streaming score-batch ABI with local stride/global ordinal.
- [ ] Require compact allocation only after a sealed selected-map receipt.
- [ ] Forbid `rows * parent_columns` final-store allocation before selection.
- [ ] Require the production entry to consume the two-pass continuation.
- [ ] Run the three focused contracts and observe the expected failures.

### Task 2: Carry exact device topology in the existing admission

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/run_device_admission_v1.rs`
- Modify: `crates/neoethos-gpu-cuda/src/data_population_workspace_plan_v1.rs`

- [ ] Capture SM count and warp size during the existing selected-device property pass.
- [ ] Hash and carry them through the move-only run admission and Data+population facts.
- [ ] Reject zero/drifted topology; do not issue a second memory/device query.
- [ ] Run focused layout/admission tests to GREEN.

### Task 3: Streaming CUDA screening pass

**Files:**
- Modify: `crates/neoethos-gpu-cuda/native/resident_trim_prefilter_v1_abi.cuh`
- Modify: `crates/neoethos-gpu-cuda/native/resident_trim_prefilter_v1.cu`
- Modify: `crates/neoethos-gpu-cuda/src/resident_trim_prefilter_v1.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs`

- [ ] Split prefilter lifecycle into begin, score feature-major batch, finalize rank/map.
- [ ] Reuse exact labels, folds, correlations, stable rank, quotas, and map seal.
- [ ] Add bounded selected-count/map readback with exact map digest under V2 authority.
- [ ] Keep one <=64-column normalized scratch batch live and release it by event.
- [ ] Run source/layout tests to GREEN.

### Task 4: Immutable replay and compact materialization

**Files:**
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs`

- [ ] Split immutable recipe/source authority from per-pass one-shot runtimes.
- [ ] First traversal scores and discards batches; seal selected routes in parent order.
- [ ] Seal `max(screen_peak, compact_materialization_peak, search_peak)` from selected count.
- [ ] Replay producers and expose only selected local columns to the existing packer.
- [ ] Preserve compact normalization/content identities and release empty intersections.
- [ ] Prove small-fixture compact values/validity equal the old full-store projection.

### Task 5: Production Search and adaptive sizing

**Files:**
- Modify: `crates/neoethos-search/src/gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs`
- Modify: `crates/neoethos-search/src/prepared_discovery_run_input_v3.rs`
- Modify: `crates/neoethos-search/src/canonical_native_discovery_run_v1.rs`
- Modify: `crates/neoethos-search/src/resident_population_auto_sizing_receipt_v2.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_population_eval.rs`
- Modify: `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`

- [ ] Replace reducing-prefilter/row-cap refusals with the two-pass continuation.
- [ ] Reserve configured population plus one scenario before Data projection.
- [ ] Maximize population/scenarios under the shared VRAM predicate.
- [ ] Bind SM/warp facts to launch geometry and remove fixed Ampere occupancy constants.
- [ ] Preserve exact owner recovery and fail-closed cancellation/release.

### Task 6: Verification, commit, push, GPU

- [ ] Run focused RED/GREEN tests, warning-clean affected-crate checks, rustfmt, and `git diff --check`.
- [ ] Commit completed slices directly on `master` and push GitHub.
- [ ] On a funded RTX 5090, compile exact master and run the real native path.
- [ ] Record selected count, exact VRAM snapshots/peaks, population, launch geometry, parity, and timing.

