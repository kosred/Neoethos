# Canonical Trendbar Bulk Acquisition Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, resumable, exact-account cTrader canonical-timeframe acquisition pipeline for ten-plus years of search/training data, with no tick dataset path.

**Architecture:** First repair `hasMore` inside the existing authenticated one-cell capture. Then add an immutable acquisition plan/checkpoint and final per-symbol series plus full matrix receipts. Search/training consume only exact final receipts.

**Tech Stack:** Rust 2024, cTrader Open API JSON messages, Vortex 0.67, SHA-256 content addressing, `anyhow`, existing NeoEthos dataset contracts.

---

## Chunk 1: Complete one-cell broker paging

### Task 1: Same-session `hasMore` pagination

**Files:**
- Modify: `crates/neoethos-broker-history/src/service_tests.rs`
- Modify: `crates/neoethos-broker-history/src/service.rs`

- [ ] **Step 1: Write the failing terminal-pagination test**

Add
`explicit_has_more_reissues_same_logical_window_with_strictly_older_exclusive_to_on_same_session_and_publishes_all_rows`.
The fake connector records full `HistoricalPageRequest` values. Script a newer
ascending page with `has_more=true` followed by an older terminal page. Assert
one connection, two unique request ids, unchanged lower bound, second exclusive
logical upper bound equal to the first page's oldest timestamp, and exact
oldest-first reopened receipt rows/range.

- [ ] **Step 2: Run the exact test and verify RED**

Run:
`RUSTFLAGS='-Dwarnings' cargo test -p neoethos-broker-history --lib service_tests::explicit_has_more_reissues_same_logical_window_with_strictly_older_exclusive_to_on_same_session_and_publishes_all_rows -- --exact --nocapture`

Expected: FAIL because current production aborts on explicit `hasMore=true`.

- [ ] **Step 3: Implement the minimal logical-chunk paging loop**

Keep one authenticated session. Accumulate accepted subpages newest-to-oldest,
retreat the exclusive logical upper cursor to the oldest accepted timestamp,
and terminate only on `has_more=false`. Enforce a bounded subpage count and all
existing identity/order/range checks before adding any page to the spool.

- [ ] **Step 4: Re-run the exact test and verify GREEN**

Expected: exactly one passed, zero failed/ignored, zero warnings.

- [ ] **Step 5: Add refusal tests one at a time**

Add and RED/GREEN individually:

- `has_more_without_nonempty_strictly_older_boundary_fails_before_publication`
- `has_more_pagination_is_bounded_and_cancellation_preserves_no_partial_generation`

- [ ] **Step 6: Run the focused service suite**

Run:
`RUSTFLAGS='-Dwarnings' cargo test -p neoethos-broker-history --lib service_tests -- --nocapture`

Expected: all selected tests pass; zero warnings/errors.

## Chunk 2: Exact plan and resumable receipts

### Task 2: Versioned acquisition plan and checkpoint

**Files:**
- Create: `crates/neoethos-broker-history/src/historical_series_acquisition_v1.rs`
- Create: `crates/neoethos-broker-history/tests/historical_series_acquisition_v1_contract.rs`
- Modify: `crates/neoethos-broker-history/src/lib.rs`

- [ ] Write executable RED tests for exact environment/account, explicit
  upper bound, canonical order/uniqueness, fixed 2016 lower bound, policy id,
  content hashes, tamper refusal, and checkpoint resume.
- [ ] Run the exact warning-denied integration test and record RED.
- [ ] Implement only the strict plan/checkpoint store and reopen validation.
- [ ] Re-run exact test and full broker-history warning-denied tests.

### Task 3: Final series and matrix authority

**Files:**
- Modify: `crates/neoethos-broker-history/src/historical_series_acquisition_v1.rs`
- Modify: `crates/neoethos-broker-history/tests/historical_series_acquisition_v1_contract.rs`

- [ ] RED-test one exact `CanonicalDatasetSeriesReceiptV1` per symbol and one
  canonical ordered matrix receipt across every required symbol.
- [ ] Reject incomplete, extra, duplicate, mixed-account, mixed-server, or
  mixed-window matrices.
- [ ] Implement minimal final authority publication and exact reopen.
- [ ] Run warning-denied focused and workspace tests.

## Chunk 3: Production acquisition and pipeline handoff

### Task 4: Exact production runner

- [ ] Replace implicit enabled/first-account selection with explicit exact
  environment/account plan binding and exact token-store loading.
- [ ] Run a bounded real-broker canonical timeframe smoke and reopen all
  immutable receipts.
- [ ] Execute the full configured symbol/timeframe matrix with resumable
  checkpoints and archive complete logs.
- [ ] Verify actual oldest/newest coverage and report any broker-limited series.
- [ ] Start search/training only from the final explicit matrix receipts.

