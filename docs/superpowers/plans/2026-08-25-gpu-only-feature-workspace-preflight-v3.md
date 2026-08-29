# GPU-Only Feature Workspace Preflight V3 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first move-only, no-decode Data preflight seam for a future complete native Discovery workspace plan while preserving the current exact fail-closed producer frontier.

**Architecture:** A focused Data module consumes `PinnedCanonicalSeriesV1`, validates only pinned manifest metadata, and seals the crate-owned ordered producer census. The existing resident materializer exposes a crate-private census sealer; no caller can supply bytes, hashes, device identity, or producer receipts.

**Tech Stack:** Rust 2024, standalone source-contract tests, `rustfmt` source checks.

---

## Chunk 1: Move-only Data preflight frontier

### Task 1: Freeze the source contract

**Files:**
- Create: `crates/neoethos-data/tests/gpu_only_feature_workspace_preflight_v3_source_contract.rs`
- Create: `crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs`

- [ ] Write a standalone source contract that requires a move-only pinned-series token, no decode/materialization calls, the exact ordered eight-producer frontier, and the exact ordered receipt backlog. Keep the next Data-owned component sealer as an explicitly ignored, compile-safe RED handoff.
- [ ] Compile and run the source contract directly; verify RED because the module and exports do not exist.
- [ ] Add the minimal preflight module and crate-private complete-census sealer.
- [ ] Register and export only the move-only preflight API and descriptive backlog types.
- [ ] Run `rustfmt --check` on touched Rust sources.
- [ ] Compile and run the standalone source contract; verify GREEN.
- [ ] Rehash touched files and report that full workspace sealing remains unavailable until the ordered producer/receipt backlog is closed.
