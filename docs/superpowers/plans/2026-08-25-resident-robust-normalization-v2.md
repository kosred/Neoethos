# Resident Robust Normalization V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact, GPU-resident semantic-v2 robust-normalization transform and a move-only shared/Data canonical split seam without host feature materialization.

**Architecture:** Data seals the canonical 80/20 range from its pinned/canonical row authority and the startup-installed normalization mode, carries that move-only split with the phase-zero workspace preflight, then freezes allocation, run-context, stream, event and lifetime authority when the exact schema exists. gpu-cuda transforms the final bar-major store in place before canonical SHA using deterministic total-order device sorting in batches of at most 64 columns. The incomplete full-workspace factory remains fail-closed and the CLI is unchanged.

**Tech Stack:** Rust 2024 workspace, cust CUDA ownership, CUDA C++17, SHA-256 capability identities, source-contract TDD.

---

## Chunk 1: Contracts and exact planning

### Task 1: Repair RED source contracts

**Files:**
- Create: `crates/neoethos-data/tests/resident_robust_normalization_v2_source_contract.rs`
- Create: `crates/neoethos-search/tests/canonical_normalization_receipt_v2_source_contract.rs`

- [ ] Replace the false Search-minter assertion with a Data-owned split derived
      only from pinned/canonical row authority and installed config.
- [ ] Add assertions for exact u4 allocation/alignment, atomic word access,
      fit digest, event synchronization before the 4-byte verdict,
      enabled/disabled accounting and unchanged four-producer pending census.
- [ ] Run standalone source oracles and record the expected RED failures.

### Task 2: Typed shared/Data seam

**Files:**
- Create: `crates/neoethos-data/src/core/gpu_resident_robust_normalization_v2.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Modify: `crates/neoethos-search/src/lib.rs`

- [ ] Define a non-Clone, non-serializable split with private fields and no
      range/mode/byte/hash/fit constructor.
- [ ] Derive row count only from pinned/canonical Data authority and mode only
      from the startup-installed configuration; remove the Search minter.
- [ ] Carry the split in `PreparedGpuOnlyFeatureWorkspacePreflightV3`, consume
      it once when exact feature-column count exists, and revalidate the split.
- [ ] Keep the current incomplete full-plan factory and CLI bail unchanged.

## Chunk 2: Isolated resident CUDA runtime

### Task 3: RED arithmetic and ABI contract

**Files:**
- Create: `crates/neoethos-gpu-cuda/src/resident_robust_normalization_v2.rs`
- Create: `crates/neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu`

- [ ] Test checked `next_power_of_two`, `ceil(columns/64)`, scratch bytes,
      `columns*48` metadata, exact bitonic plus fit-hash launch count, disabled
      zero extents/launches/events, and `round_up(cells/2, 4)` validity bytes.
- [ ] Freeze the exact ABI and CPU authority string before implementation.

### Task 4: Device implementation

**Files:**
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Create: `crates/neoethos-gpu-cuda/src/resident_robust_normalization_v2.rs`
- Create: `crates/neoethos-gpu-cuda/native/resident_robust_normalization_v2.cu`

- [ ] Fill fixed scratch from packed u4 validity and bar-major values.
- [ ] Sort values with Rust-total-order-equivalent bitonic stages.
- [ ] Compute ordered fallback statistics, replace scratch with deviations,
      sort again, and finalize six-u64 fit rows.
- [ ] Apply normalization in place with atomic u4 reason updates and canonical
      NaN payloads.
- [ ] Hash fit metadata on device into the retired scratch prefix, record one
      event, explicitly synchronize it, then read one 4-byte verdict and one
      32-byte digest. Return only allocation/launch/event/control-plane
      evidence; never values or validity payloads.

## Chunk 3: Data authority and connection audit

### Task 5: Component receipt

**Files:**
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`
- Modify: `crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs`
- Modify: `crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs`
- Modify: `crates/neoethos-gpu-contracts/tests/resident_feature_store_v3.rs`

- [ ] Add move-only allocation/lifetime/component receipts, bind them to the
      moved admission's exact context/stream, and validate range, extents,
      launches, event/sync counts, D2H bounds and fit digest.
- [ ] Charge retained fit metadata to steady device residency exactly once and
      validate the sealed owner against that admitted steady extent.
- [ ] Invoke exactly once after final producer-batch retirement and before
      canonical SHA. Disabled mode still emits a typed zero-work receipt.
- [ ] Keep RobustNormalization pending until source/compile/device gates pass;
      document the remaining complete-workspace factory/identity connection.

### Task 6: Lightweight verification and freeze

- [ ] Run rustfmt checks only for touched Rust files.
- [ ] Run source oracles; do not run Cargo, NVCC or a device locally.
- [ ] Record every pre/post SHA-256 and distinguish source-connected,
      compile-checked and device-validated states.
- [ ] Do not commit from the shared dirty worktree; root owns integration.
