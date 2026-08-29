# Resident Search local production wiring implementation plan

> **For Codex:** execute directly on `master`, commit each completed slice, and keep tests focused on wiring rather than restarting broad audits.

**Goal:** remove the local no-op/refusal boundaries and connect the existing resident CUDA components into the production prepared-discovery path.

**Architecture:** one move-only Rust owner binds the existing trim, admission, scoring, archive and generation owners to the seven-function native split ABI. Typed V3 transitions drive the same-stream multi-generation chain. A bounded terminal receipt continues into the existing quality/portfolio/validation/holdout result path. No CPU fallback or host feature materialization is permitted.

**Tech stack:** Rust nightly, CUDA C++, existing NeoEthos GPU/Search crates, locked/offline Cargo.

---

### Task 1: Bind the native archive ABI in Rust

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_archive_knn_v2_native.rs`
- Modify if required: `crates/neoethos-gpu-cuda/src/lib.rs`

Add exact `repr(C)` owner/pending/terminal mirrors and `extern "C"` declarations for the seven frozen ABI calls. Keep raw handles crate-private and add focused ABI/layout compile checks.

### Task 2: Build the move-only resident Slice 2 owner

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`

Join trimmed-population lifetime, generation/scoring owners, dependency receipt, gene view, calibration/bind authority and native archive handle. Bind once. On every error return the exact owner; on drop/release free only the native view and leave outer allocations to their owner.

### Task 3: Wire typed generation transitions

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs`
- Modify: `crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs`

Replace all fail-closed transition bodies and `queue_generation_v2` no-op with native calls. Enforce score/rank -> stage -> evolve/publish ordering, repeated generations, terminal-only-after-publish and retained authority on `NotReady`.

### Task 4: Seal the full workspace plan before materialization

**Files:**
- Modify: `crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs`
- Modify: `crates/neoethos-app/src/app_services/discovery.rs`

Implement the allocation-free native scratch/preflight provider that mints `OpaqueResidentTrimPrefilterPreflightV1`, then seal `SealedFullDiscoveryGpuWorkspacePlanV1` before the application materializes the resident Data store. This removes the first preparation refusal without allocating or fabricating trim authority.

### Task 5: Connect trim output to Slice 2 admission

**Files:**
- Modify: `crates/neoethos-search/src/gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs`
- Modify: `crates/neoethos-search/src/prepared_discovery_run_input_v3.rs`

Turn the sealed trim output into the exact owned population source and admission/calibration input. Preserve the resident dataset owner and eliminate placeholder trim identity/facts on the native path.

### Task 6: Run the resident multi-generation loop

**Files:**
- Modify: `crates/neoethos-search/src/prepared_discovery_run_input_v3.rs`
- Reuse/modify: `crates/neoethos-search/src/gpu_full_discovery/gpu_resident_generation_pipeline_v1.rs`
- Modify: `crates/neoethos-search/src/lib.rs`

Make the resident generation pipeline compiled and callable. Drive the configured generation count through the typed owner and finish with terminal enqueue/completion. Permit only bounded polling and terminal readback.

### Task 7: Connect post-GA quality, portfolio and validation stages

**Files:**
- Implement the currently contract-only modules under `crates/neoethos-search/src/gpu_full_discovery/` for post-GA, portfolio, resident validation and robustness.
- Modify their existing source-contract tests only when exact production names require synchronization.

Connect existing native/library operations and canonical formulas; do not recalculate or redesign them. Return a bounded finalist/evidence carrier suitable for `DiscoveryResult` finalization.

### Task 8: Remove Search and application refusals

**Files:**
- Modify: `crates/neoethos-search/src/prepared_discovery_run_input_v3.rs`
- Modify: `crates/neoethos-app/src/app_services/discovery.rs`

Replace `run_native_cuda_prepared_discovery_v3` refusal and both application plan/materializer refusal closures with the production pipeline. Keep missing CUDA authority as an explicit error and prohibit CPU fallback.

### Task 9: Focused local verification and commits

Run, with `CARGO_INCREMENTAL=0`, `RUSTFLAGS=-Dwarnings`, `--locked --offline`:

1. GPU CUDA compile-contract library checks and focused ownership/state tests.
2. Search library/all-target checks for the resident compile feature.
3. App native workspace bridge source/integration smoke checks.
4. Scoped rustfmt and `git diff --check`.

Commit each finished task directly on `master`. Stop only for a concrete compiler/ownership incompatibility; do not start another broad audit cycle.

### Task 10: Hand the same master to the NVIDIA gate

After local tracked state is clean, run the exact committed `master` on the target NVIDIA card with native CUDA compilation, device execution, sanitizer/error checks, allocation/sync/D2H counters, deterministic parity and throughput. Do not patch a separate GPU worktree.
