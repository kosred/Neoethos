# cTrader Platform Adapter Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cTrader` the tier-1 backend by implementing the documented `cTrader Open API` surface in staged tranches, starting with symbol discovery and historical market data.

**Architecture:** Build a small shared `cTrader` protocol layer for documented message handling, then layer focused adapters for data, account, streaming, and execution on top. Expose only broker-agnostic app snapshots to the UI and keep raw protocol handling isolated.

**Tech Stack:** Rust, `tungstenite`, `serde`, `serde_json`, existing `forex-app` service layer, official `cTrader Open API` JSON/websocket protocol.

---

## Chunk 1: cTrader Symbols + Historical Data

### Task 1: Add cTrader protocol message helpers for symbols and historical bars

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_messages.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Test: `crates/forex-app/src/app_services/ctrader_messages.rs`

- [ ] **Step 1: Write failing tests for documented symbol/trendbar payload builders**

Add tests for:
- `ProtoOASymbolsListReq`
- `ProtoOAGetTrendbarsReq`
- optional `ProtoOAGetTickDataReq` if included in this tranche

Expected checks:
- documented payload type ids
- required fields present
- request uses `ctidTraderAccountId`
- trendbars request uses `symbolId`, `period`, `fromTimestamp`, `toTimestamp`, `count`

- [ ] **Step 2: Run targeted test to verify failure**

Run:
`cargo test -p forex-app ctrader_messages -- --nocapture`

Expected:
- FAIL because the new module/builders do not exist yet

- [ ] **Step 3: Implement minimal message builders and envelope helpers**

Add:
- payload type constants
- typed builder functions
- tiny parsing helpers for envelope matching

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app ctrader_messages -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_messages.rs crates/forex-app/src/app_services/mod.rs
git commit -m "feat: add ctrader symbol and trendbar message helpers"
```

### Task 2: Add cTrader historical/symbol data adapter

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_data.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- Test: `crates/forex-app/src/app_services/ctrader_data.rs`

- [ ] **Step 1: Write failing tests for symbol list parsing and trendbar normalization**

Add tests for:
- parsing `ProtoOASymbolsListRes`
- parsing `ProtoOAGetTrendbarsRes`
- converting relative trendbar prices to absolute OHLC using symbol digits
- explicit failure on wrong payload types or malformed data

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app ctrader_data -- --nocapture`

Expected:
- FAIL because adapter/parsers do not exist yet

- [ ] **Step 3: Implement minimal symbol and historical bar adapter**

Add:
- `CTraderSymbolInfo`
- `HistoricalBar`
- `HistoricalBarsResult`
- parsers/mappers for official JSON responses
- fail-closed error handling

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app ctrader_data -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_data.rs crates/forex-app/src/app_services/ctrader_live_auth.rs
git commit -m "feat: add ctrader symbols and historical bars adapter"
```

### Task 3: Wire cTrader historical data into app-level market chart snapshots

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/ui/trading/chart_panel.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`
- Test: `crates/forex-app/src/ui/trading/chart_panel.rs`

- [ ] **Step 1: Write failing tests for cTrader-backed chart snapshots**

Add tests for:
- selecting `cTrader` as active adapter and loading real historical chart data
- explicit degraded state when cTrader data is not available
- no fake placeholders when adapter results exist

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app market_chart_snapshot -- --nocapture`

Expected:
- FAIL because chart snapshots still only use local parquet path

- [ ] **Step 3: Implement minimal cTrader chart data integration**

Add:
- broker-agnostic chart data request path in `TradingSession`
- cTrader historical bar loading through the new adapter
- chart snapshot mapping for real candles

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app market_chart_snapshot -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/trading.rs crates/forex-app/src/ui/trading/chart_panel.rs
git commit -m "feat: wire ctrader historical data into chart snapshots"
```

### Task 4: Verify the first cTrader data tranche end-to-end

**Files:**
- Verify only

- [ ] **Step 1: Run crate verification**

Run:
`cargo test -p forex-app -- --nocapture`

Expected:
- PASS

- [ ] **Step 2: Run crate lint verification**

Run:
`cargo clippy -p forex-app --all-targets -- -D warnings`

Expected:
- PASS

- [ ] **Step 3: Run workspace verification**

Run:
`cargo test --workspace -- --nocapture`

Expected:
- PASS

- [ ] **Step 4: Run workspace lint verification**

Run:
`cargo clippy --workspace --all-targets -- -D warnings`

Expected:
- PASS

- [ ] **Step 5: Run startup smoke**

Run:
`target/debug/forex-app.exe --headless --local --config config.yaml`

Expected:
- startup succeeds
- canonical log records `APP` startup
- process may need manual stop after verification

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "test: verify ctrader data tranche"
```

## Chunk 2: cTrader Account Auth + Reconcile

### Task 5: Add account-auth protocol helpers

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_messages.rs`
- Create: `crates/forex-app/src/app_services/ctrader_account.rs`
- Test: `crates/forex-app/src/app_services/ctrader_account.rs`

- [ ] **Step 1: Write failing tests for `ProtoOAAccountAuthReq` / `ProtoOAAccountAuthRes`**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal account-auth contract**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

### Task 6: Add reconcile snapshots for current positions and pending orders

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_account.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for `ProtoOAReconcileReq` / `ProtoOAReconcileRes` mapping**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal reconcile adapter and snapshot wiring**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 3: Live Market Subscriptions

### Task 7: Add spot/trendbar/depth subscription contracts

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_streaming.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_messages.rs`
- Test: `crates/forex-app/src/app_services/ctrader_streaming.rs`

- [ ] **Step 1: Write failing tests for subscription message builders and event parsers**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal streaming contract**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 4: Trading Operations

### Task 8: Add order-operation message contracts

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_execution.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_messages.rs`
- Test: `crates/forex-app/src/app_services/ctrader_execution.rs`

- [ ] **Step 1: Write failing tests for new/cancel/amend/close message builders**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal execution adapter contract**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

### Task 9: Surface order execution results and errors into the app

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/ui/trading/execution_panel.rs`
- Modify: `crates/forex-app/src/ui/trading/bottom_strip.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for `ProtoOAExecutionEvent` / `ProtoOAOrderErrorEvent` mapping**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal execution result wiring**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Chunk 5: Multi-account Execution

### Task 10: Add multi-account fan-out on top of verified cTrader account sessions

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_execution.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for multi-account execution target fan-out**
- [ ] **Step 2: Run targeted failing tests**
- [ ] **Step 3: Implement minimal fan-out execution and per-account result tracking**
- [ ] **Step 4: Re-run tests**
- [ ] **Step 5: Commit**

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-23-ctrader-platform-adapter-foundation.md`. Ready to execute.
