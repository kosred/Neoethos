# Resident Search local production wiring design

**Status:** Approved (option A) on 2026-08-29.

## Goal

Connect the already implemented resident CUDA trim, scoring, archive, ranking, generation, and validation components into one production-owned discovery run on `master`. The local milestone ends only when the current no-op generation transition and the two explicit full-discovery refusals are gone, the Rust ownership chain calls the native ABI, and the application receives a real bounded `DiscoveryResult` path.

This is wiring work. Existing quantitative formulas, kernels, layouts, model implementations, and reviewed ABI contracts are reused. They are changed only when compilation or a direct integration failure proves a concrete incompatibility.

## Non-negotiable constraints

- CUDA work never falls back to CPU.
- No host feature-frame materialization in the resident path.
- One admitted device/context/stream/pool ownership chain.
- No intermediate full-array D2H transfer or host synchronization.
- Only bounded terminal receipts/survivors may cross to the host.
- Fallible transitions return the move-only owner unchanged; `NotReady` retains it.
- A device fault cannot publish a half-advanced generation/archive state.
- Cleanup follows the existing outer-owner order and frees each allocation once.
- Implementation and commits stay on `master`; no new worktree is created.

## Existing production pieces to connect

- Resident trim/prefilter: `gpu_resident_trim_prefilter_view_v1.rs`.
- Slice 2 admission, calibration, preallocation and bind authority: `resident_search_slice2_admission_v2.rs`.
- Native split archive ABI and kernels: `resident_archive_knn_v2_abi.cuh` and `resident_archive_knn_v2.cu`.
- Resident scoring and generation native owners/helpers.
- Typed public state surface: `resident_search_slice2_v3.rs`.
- Prepared discovery entry and sealed resident dataset ownership in `prepared_discovery_run_input_v3.rs`.
- Existing canonical final quality, portfolio, validation and holdout semantics in Search.

## Wiring architecture

### 1. Rust native owner

Add the missing Rust FFI declarations and raw ABI mirrors for bind, score/rank, stage, evolve/publish, terminal enqueue, terminal completion and release. A single move-only Rust owner retains the trimmed population lifetime, generation owner, scoring owner, boxed dependency receipt, gene view, calibration/bind authority and native archive owner.

### 2. Typed generation state machine

Replace the no-op `queue_generation_v2` and fail-closed V3 transition bodies with calls through the owner/backend. The legal sequence is:

`GenerationChain -> RankEnqueued -> ArchiveStaged -> GenerationChain`

and, after the final published generation:

`GenerationChain -> TerminalPending -> TerminalReceipt`.

The host ordinal is validation only; the retained device seal remains generation/store authority.

### 3. Resident source bridge

Bind the sealed trim result to the already defined scoring population source without exposing detached pointers. Admission is executed once, native bind is executed once, and all generations reuse the admitted arenas and stream.

Before materialization, implement the missing allocation-free native scratch/preflight provider required to mint `OpaqueResidentTrimPrefilterPreflightV1` and seal `SealedFullDiscoveryGpuWorkspacePlanV1`. This provider runs at the preparation boundary, so the application can produce the full workspace plan before the resident Data store is materialized.

### 4. Full resident discovery continuation

Replace the prepared-run refusal with the actual resident loop: trim/prefilter, admission/calibration, repeated score/rank/stage/evolve-publish, terminal seal and bounded completion. The continuation feeds the existing canonical quality, portfolio, robustness, validation and outer-holdout semantics through resident/native stages and a bounded finalist receipt, not through the old host feature-frame path.

### 5. Application entry

Replace the native plan/materializer refusal closures with the real prepared plan and resident materialization. The application dispatch remains fail-closed when CUDA authority or a required production stage is unavailable; it never silently routes to CPU.

## Verification boundary

During wiring, use only focused compile and ownership smoke checks needed to prevent broken integration. Do not restart broad mathematical audits or mutation campaigns. Before opening the GPU card, the exact `master` must satisfy:

- no no-op generation transition;
- no unconditional full-discovery/native-plan/materializer refusal on the admitted CUDA path;
- a real pre-materialization native scratch/preflight provider seals the full workspace plan;
- warning-clean local compile of the affected crates/features;
- focused host ownership/state smoke checks;
- clean tracked worktree with every implementation commit on `master`.

Real CUDA compilation, sanitizer/device execution, counters, parity and throughput are the next gate and must run from that same committed `master`.

## Explicit non-goals

- Re-deriving already frozen formulas or model mathematics.
- Replacing working kernels or libraries without a concrete integration failure.
- CPU fallback, fake `DiscoveryResult`, or treating Generation 0 as full discovery.
- ROCm/HIP work.
- Model training changes.
