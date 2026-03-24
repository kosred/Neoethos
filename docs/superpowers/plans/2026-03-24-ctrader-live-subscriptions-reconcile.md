# cTrader Live Subscriptions And Reconcile Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add documented cTrader account-authenticated trader/reconcile snapshots plus live spots/live trendbars to the app so charts and execution surfaces can use real cTrader runtime data.

**Architecture:** Extend the shared cTrader message layer with the documented account/subscription payloads, add focused `ctrader_account` and `ctrader_streaming` adapters, then wire them into `TradingSession` so the UI stays snapshot-driven and honest about degraded states.

**Tech Stack:** Rust, `serde`, `serde_json`, existing `forex-app` app services, `tungstenite`, official cTrader Open API JSON/WebSocket protocol.

---

## Chunk 1: Protocol Contracts

### Task 1: Add failing tests for documented account/subscription message builders

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_messages.rs`
- Test: `crates/forex-app/src/app_services/ctrader_messages.rs`

- [ ] **Step 1: Write failing tests for account/trader/reconcile/subscription payload builders**

Add tests for:
- `ProtoOATraderReq`
- `ProtoOAReconcileReq`
- `ProtoOASubscribeSpotsReq`
- `ProtoOAUnsubscribeSpotsReq`
- `ProtoOASubscribeLiveTrendbarReq`
- `ProtoOAUnsubscribeLiveTrendbarReq`

- [ ] **Step 2: Run targeted test to verify failure**

Run:
`cargo test -p forex-app ctrader_messages -- --nocapture`

Expected:
- FAIL because the new builders/constants do not exist yet

- [ ] **Step 3: Implement the minimal documented message builders**

Add only the documented payload ids and required fields for the messages above.

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app ctrader_messages -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_messages.rs
git commit -m "feat: add ctrader account and subscription message helpers"
```

## Chunk 2: Account Auth + Reconcile Adapter

### Task 2: Add failing tests for cTrader account and reconcile parsing

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_account.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Test: `crates/forex-app/src/app_services/ctrader_account.rs`

- [ ] **Step 1: Write failing tests for trader/reconcile parsing**

Add tests for:
- `ProtoOATraderRes` mapping into a small trader snapshot
- `ProtoOAReconcileRes` mapping into normalized positions/pending orders
- explicit failure on wrong payload type
- explicit failure on error envelopes

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app ctrader_account -- --nocapture`

Expected:
- FAIL because the adapter does not exist yet

- [ ] **Step 3: Implement the minimal account adapter**

Add:
- trader snapshot structs
- reconcile snapshot structs
- parsers for documented JSON envelopes
- minimal transport-backed loader for:
  - app auth
  - account auth
  - trader request
  - reconcile request

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app ctrader_account -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/ctrader_account.rs
git commit -m "feat: add ctrader trader and reconcile adapter"
```

### Task 3: Wire reconcile snapshots into the app execution surface

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for cTrader reconcile-backed execution snapshots**

Add tests for:
- `cTrader` execution surface shows real positions/pending orders when account-authenticated data exists
- missing token/account auth still yields degraded warnings instead of fake data

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app execution_surface_snapshot -- --nocapture`

Expected:
- FAIL because the cTrader execution surface is still unwired

- [ ] **Step 3: Implement minimal reconcile integration**

Wire `TradingSession` so `cTrader` can load trader/reconcile state and project it into `ExecutionSurfaceSnapshot`.

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app execution_surface_snapshot -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/trading.rs
git commit -m "feat: wire ctrader reconcile into execution surface"
```

## Chunk 3: Live Spot + Live Trendbar Adapter

### Task 4: Add failing tests for cTrader spot/trendbar event parsing

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_streaming.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Test: `crates/forex-app/src/app_services/ctrader_streaming.rs`

- [ ] **Step 1: Write failing tests for subscription responses and spot events**

Add tests for:
- `ProtoOASubscribeSpotsRes`
- `ProtoOASubscribeLiveTrendbarRes`
- `ProtoOASpotEvent` parsing into live quote/trendbar updates
- explicit failure on wrong payload or malformed spot data

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app ctrader_streaming -- --nocapture`

Expected:
- FAIL because the streaming adapter does not exist yet

- [ ] **Step 3: Implement the minimal streaming adapter**

Add:
- spot snapshot structs
- live trendbar update structs
- subscription response parsers
- spot event parser
- small transport-backed helper to:
  - app auth
  - account auth
  - subscribe spots
  - subscribe live trendbars
  - read a bounded number of matching spot events

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app ctrader_streaming -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/ctrader_streaming.rs
git commit -m "feat: add ctrader spot and trendbar streaming adapter"
```

### Task 5: Wire live cTrader updates into the market chart snapshot

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for cTrader live market updates**

Add tests for:
- live spot/trendbar updates refining the chart snapshot when cTrader runtime data exists
- explicit degraded state when subscriptions are unavailable

- [ ] **Step 2: Run targeted tests to verify failure**

Run:
`cargo test -p forex-app market_chart_snapshot -- --nocapture`

Expected:
- FAIL because chart snapshots only use historical data and no live subscription layer

- [ ] **Step 3: Implement minimal live market integration**

Wire `TradingSession` so `cTrader` market snapshots:
- start from historical bars
- optionally merge live spot/trendbar updates into the newest bar(s)
- remain explicit and empty when runtime prerequisites are missing

- [ ] **Step 4: Re-run targeted tests**

Run:
`cargo test -p forex-app market_chart_snapshot -- --nocapture`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/trading.rs
git commit -m "feat: wire ctrader live subscriptions into chart snapshot"
```

## Chunk 4: Verification

### Task 6: Verify the full tranche

**Files:**
- Verify only

- [ ] **Step 1: Run targeted account tests**

Run:
`cargo test -p forex-app ctrader_account -- --nocapture`

Expected:
- PASS

- [ ] **Step 2: Run targeted streaming tests**

Run:
`cargo test -p forex-app ctrader_streaming -- --nocapture`

Expected:
- PASS

- [ ] **Step 3: Run full forex-app tests**

Run:
`cargo test -p forex-app -- --nocapture`

Expected:
- PASS

- [ ] **Step 4: Run forex-app lint verification**

Run:
`cargo clippy -p forex-app --all-targets -- -D warnings`

Expected:
- PASS

- [ ] **Step 5: Run workspace verification**

Run:
`cargo test --workspace -- --nocapture`

Expected:
- PASS

- [ ] **Step 6: Run workspace lint verification**

Run:
`cargo clippy --workspace --all-targets -- -D warnings`

Expected:
- PASS

- [ ] **Step 7: Run startup smoke**

Run:
`target/debug/forex-app.exe --headless --local --config config.yaml`

Expected:
- startup succeeds
- canonical log records `SYSTEM setup_logging`
- canonical log records `APP headless_local_start`

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "test: verify ctrader live subscriptions and reconcile tranche"
```

Plan complete and saved to `docs/superpowers/plans/2026-03-24-ctrader-live-subscriptions-reconcile.md`. Ready to execute?
