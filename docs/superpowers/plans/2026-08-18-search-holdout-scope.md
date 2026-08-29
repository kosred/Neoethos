# Exact Search Holdout Scope Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make discovery holdout execution and every persisted search result carry the exact selection and evidence windows derived from one validated canonical input.

**Architecture:** Keep `CanonicalSearchRunInputV1` as the strict full-input boundary. Add private discovery-core typed full/split inputs that construct values and canonical scopes together, then store their scopes on `DiscoveryResult`; writers consume only the stored selection scope.

**Tech Stack:** Rust, `neoethos-search`, canonical data-selection contracts, focused unit tests.

---

## Chunk 1: RED scope contract

### Task 1: Specify exact full and holdout windows

**Files:**
- Modify: `crates/neoethos-search/src/discovery_tests.rs`
- Test: `crates/neoethos-search/src/discovery_tests.rs`

- [ ] **Step 1: Write the failing 100-row split test**

Add a test that asks the wished-for private typed split API to construct the existing fixture's windows and asserts:

```rust
assert_eq!(selection.role(), CanonicalSearchWindowRoleV1::InSample);
assert_eq!((selection.row_start(), selection.row_end()), (0, 80));
assert_eq!(selection.timestamp_start_ms(), timestamps[0]);
assert_eq!(selection.timestamp_end_ms(), timestamps[79]);
assert_eq!(holdout.role(), CanonicalSearchWindowRoleV1::Holdout);
assert_eq!((holdout.row_start(), holdout.row_end()), (80, 100));
assert_eq!(holdout.timestamp_start_ms(), timestamps[80]);
assert_eq!(holdout.timestamp_end_ms(), timestamps[99]);
```

- [ ] **Step 2: Write refusal tests**

Add focused tests for swapped roles, a gap, an overlap, an empty/missing suffix, and fewer than 64 in-sample rows. Each must assert an actionable boundary/role error.

- [ ] **Step 3: Request Cargo GO**

Do not run Cargo automatically. Ask the parent to confirm the vector lane is stable.

- [ ] **Step 4: Verify RED after GO**

Run:

```powershell
cargo test -p neoethos-search --lib discovery::tests::holdout -- --nocapture
```

Expected: compilation/test failure because the typed split API and stored result scopes do not exist yet.

## Chunk 2: GREEN typed input and stored scopes

### Task 2: Implement the private typed discovery windows

**Files:**
- Modify: `crates/neoethos-search/src/discovery.rs:1490-1561`
- Modify: `crates/neoethos-search/src/discovery.rs:4961-5128`
- Test: `crates/neoethos-search/src/discovery_tests.rs`

- [ ] **Step 1: Add the minimal private typed wrapper**

Build full or split windows only from `&CanonicalSearchRunInputV1`. Store `Cow<FeatureFrame>`, `Cow<Ohlcv>`, and the exact scope together. A split constructor computes `is_end`, enforces the 64-row floor and non-empty suffix, and validates the paired roles/boundaries.

- [ ] **Step 2: Make the value runner accept the typed selection input**

Replace its raw `features`, `ohlcv`, and `search_input_receipt` parameters with one typed selection input. Use the typed holdout values for forward and prop evaluation.

- [ ] **Step 3: Store exact scopes on `DiscoveryResult`**

Add the stored selection scope and optional holdout scope. Keep the explicit fields public only for existing cross-crate literal compatibility, add fail-closed `validate_evaluated_scopes()`, and make `selection_scope()` / `holdout_scope()` validate before returning. Remove `artifact_scope(role)` so it cannot relabel or reconstruct authority; do not add a default or permissive constructor.

- [ ] **Step 4: Update discovery-core result fixtures**

Update only `discovery_tests.rs` literals/helpers in this owned slice. Record other crate callers for separate migration.

- [ ] **Step 5: Update owned writers**

Make `save_portfolio_json` and `save_promotion_summary_json` clone `result.selection_scope()` explicitly.

- [ ] **Step 6: Verify GREEN after GO**

Run the focused command from Task 1 and then:

```powershell
cargo test -p neoethos-search --lib discovery::tests -- --nocapture
```

Expected: all focused discovery tests pass with no warnings or errors attributable to this slice.

## Chunk 3: Review and handoff

### Task 3: Verify scope and enumerate migrations

**Files:**
- Review: `crates/neoethos-search/src/discovery.rs`
- Review: `crates/neoethos-search/src/discovery_tests.rs`

- [ ] **Step 1: Inspect the exact diff**

Run `git diff --check` and a focused `git diff` for the owned files. Confirm unrelated dirty hunks are preserved.

- [ ] **Step 2: Search for old callers**

Run:

```powershell
rg -n "artifact_scope\(|DiscoveryResult \{" crates/neoethos-search crates/neoethos-app crates/neoethos-cli crates/neoethos-autoresearch
```

Report every out-of-scope caller needing `selection_scope()`/`holdout_scope()` or fixture-field migration; do not edit app, models, or `live_portfolio.rs` in this slice.

- [ ] **Step 3: Report verification honestly**

If Cargo GO was not granted, report the implementation as edited but uncompiled/unverified. Never claim GREEN without fresh command output.
