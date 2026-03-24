# cTrader Data Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bars-first operator bootstrap flow that fetches missing historical data from `cTrader`, falls back to `MT5` when needed, cleans and validates the result, and writes it into the existing local parquet layout used by discovery and training.

**Architecture:** Introduce a focused bootstrap orchestration service that sits above existing `cTrader` history helpers and the current local parquet contract. The bootstrap service must inspect existing coverage, fetch uncovered trailing UTC ranges through a fixed `Local -> cTrader -> MT5` ladder, normalize and validate bars into a canonical schema, and write them atomically without changing the downstream research pipeline. Progress must move through an explicit bootstrap job snapshot path so the `System` UI can render truthful operator status.

**Tech Stack:** Rust, `forex-app` app services, `forex-data`, `polars` parquet IO, existing `cTrader` history transport, extended `MT5` bridge path, egui operator UI.

---

## Chunk 1: Bootstrap Planning And Cleaning Core

### Task 1: Add failing chunk-planning tests

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- year request to time-range conversion
- chunk planning for at least `M1`, `M15`, and `H1`
- non-overlapping chunks that fully cover the requested range

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: FAIL because the bootstrap module and planner do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- bootstrap request structs
- timeframe-aware chunk planner
- explicit chunk list generation

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

### Task 2: Add failing normalization and validation tests

**Files:**
- Create: `crates/forex-app/src/app_services/bootstrap_writer.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- duplicate timestamps are deduplicated deterministically
- rows are sorted ascending by timestamp
- invalid OHLC rows fail validation
- non-finite values fail validation

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: FAIL because normalization and validation helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- canonical `NormalizedBar`
- cleaning helpers
- validation helpers

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

## Chunk 2: Parquet Writing And Coverage Detection

### Task 3: Add failing parquet-writer tests

**Files:**
- Create: `crates/forex-app/src/app_services/bootstrap_writer.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- writer creates `symbol=<PAIR>/timeframe=<TF>/data.parquet`
- schema matches existing contract
- atomic write leaves no corrupt partial file when replacement fails

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app bootstrap_writer -- --nocapture`
Expected: FAIL because the writer does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a writer that:
- converts cleaned bars into the current parquet schema
- writes to a temporary file first
- renames into place atomically

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app bootstrap_writer -- --nocapture`
Expected: PASS

### Task 4: Add failing local-coverage tests

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`
- Modify: `crates/forex-data/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- existing parquet coverage is detected correctly
- fully covered requests skip remote fetch
- uncovered tail/head segments are reported accurately

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: FAIL because coverage inspection is incomplete or missing.

- [ ] **Step 3: Write minimal implementation**

Add:
- local coverage inspection helpers
- requested-range vs covered-range reporting
- trailing UTC range semantics using `years * 365 days`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

## Chunk 3: Source Ladder Orchestration

### Task 5: Add failing cTrader-source orchestration tests

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_data.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- chunked `cTrader` bar fetch over multiple windows
- partial `cTrader` coverage produces missing segments
- `cTrader` source warnings propagate to the bootstrap result

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: FAIL because chunked bootstrap orchestration is not wired yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- reusable bar-page fetch helper in `ctrader_data.rs`
- bootstrap orchestration over chunk plans
- `cTrader` source coverage aggregation

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

### Task 6: Add failing MT5 historical-bar bridge tests

**Files:**
- Modify: `crates/mt5-bridge/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- range-based MT5 bars retrieval for a symbol/timeframe
- timestamp normalization to UTC milliseconds or nanoseconds compatible with bootstrap normalization
- explicit failure when MT5 is disconnected or returns no rates

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p mt5-bridge -- --nocapture`
Expected: FAIL because the bridge does not expose historical bars yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- typed MT5 historical-bar snapshot
- documented range-based bars retrieval through the Python bridge

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p mt5-bridge -- --nocapture`
Expected: PASS

### Task 7: Add failing MT5-fallback tests

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- uncovered segments fall back from `cTrader` to `MT5`
- unavailable `MT5` leaves explicit degraded result instead of fake success
- `sources_used` and `missing_segments` are reported accurately

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: FAIL because fallback ladder behavior is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- source ladder execution
- `Local -> cTrader -> MT5` coverage merge
- explicit degraded result state for partially satisfied requests

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

## Chunk 4: App State And UI Wiring

### Task 8: Add failing bootstrap job-state and UI tests

**Files:**
- Modify: `crates/forex-app/src/app_state.rs`
- Modify: `crates/forex-app/src/app_services/jobs.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- bootstrap form defaults
- operator selections for pairs/timeframes/years
- bootstrap `JobSnapshot` transport to UI
- progress/result rendering for success, degraded, and failed states

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: FAIL because bootstrap UI state does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- bootstrap form state in `AppState`
- bootstrap job kind and explicit snapshot transport
- `Data Bootstrap` operator section in `system_status.rs`
- service wiring to launch and display bootstrap progress/results

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: PASS

## Chunk 5: Full Verification

### Task 9: Verify crate and workspace

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_bootstrap.rs`
- Create: `crates/forex-app/src/app_services/bootstrap_writer.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_data.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Modify: `crates/forex-app/src/app_state.rs`
- Modify: `crates/forex-app/src/app_services/jobs.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify: `crates/forex-data/src/lib.rs`
- Modify: `crates/mt5-bridge/src/lib.rs`

- [ ] **Step 1: Run bootstrap-focused tests**

Run: `cargo test -p forex-app ctrader_bootstrap -- --nocapture`
Expected: PASS

- [ ] **Step 2: Run forex-app tests**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run forex-app clippy**

Run: `cargo clippy -p forex-app --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 4: Run workspace tests**

Run: `cargo test --workspace -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 6: Run startup smoke**

Run: `target/debug/forex-app.exe --headless --local --config config.yaml`
Expected: startup succeeds and writes `SYSTEM`/`APP` records into `logs/forex-ai.log`

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-03-24-ctrader-data-bootstrap-design.md docs/superpowers/plans/2026-03-24-ctrader-data-bootstrap.md crates/forex-app/src/app_services/ctrader_bootstrap.rs crates/forex-app/src/app_services/bootstrap_writer.rs crates/forex-app/src/app_services/ctrader_data.rs crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/jobs.rs crates/forex-app/src/app_state.rs crates/forex-app/src/ui/system_status.rs crates/forex-data/src/lib.rs crates/mt5-bridge/src/lib.rs
git commit -m "Add cTrader data bootstrap flow"
```
