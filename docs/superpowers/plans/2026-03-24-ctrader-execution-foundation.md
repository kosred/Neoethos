# cTrader Execution Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add recent `cTrader` deals/fills parity and documented order-operation request builders so the execution surface can show typed fills and the adapter has a correct order-message foundation.

**Architecture:** Keep the change split across three boundaries: documented JSON message builders in `ctrader_messages.rs`, runtime loading/parsing in `ctrader_account.rs`, and UI-facing execution mapping in `trading.rs`. Do not add live order submission yet; only establish the typed contracts and verified runtime surface.

**Tech Stack:** Rust, serde/serde_json, anyhow, tungstenite transport helpers, existing `forex-app` test suite

---

## Chunk 1: Message-Layer Contracts

### Task 1: Add failing tests for documented cTrader deal/order request builders

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/ctrader_messages.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `deal_list_request_uses_documented_payload_and_filters`
- `new_order_request_uses_documented_trade_payload`
- `amend_order_request_uses_documented_identifiers_and_optional_fields`
- `cancel_order_request_uses_documented_order_id`
- `close_position_request_uses_documented_position_id_and_volume`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_messages -- --nocapture`

Expected: FAIL because the new builders/constants do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- payload type constants for deal list and order-operation messages
- request builders for:
  - deal list
  - new order
  - amend order
  - cancel order
  - close position
- response mapping updates only where documented and needed

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_messages -- --nocapture`

Expected: PASS

## Chunk 2: Runtime Deals/Fills Parity

### Task 2: Add failing tests for cTrader deal parsing and runtime loading

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/ctrader_account.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `deal_list_response_parses_recent_deals`
- `account_runtime_loader_authenticates_then_loads_trader_reconcile_and_deals`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_account -- --nocapture`

Expected: FAIL because the runtime snapshot does not yet include recent deals.

- [ ] **Step 3: Write minimal implementation**

Add:
- typed `CTraderDealSnapshot`
- runtime snapshot field for recent deals
- deal-list response parsing
- deal-list request in the runtime transport sequence
- minimal documented lookback/maxRows request choices

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_account -- --nocapture`

Expected: PASS

## Chunk 3: Execution Surface Parity

### Task 3: Add failing tests for cTrader fills on the UI execution surface

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write the failing tests**

Add a test that connected `cTrader` runtime snapshots produce:
- position lines
- pending order lines
- recent fill lines in `bot_timeline`
- diagnostics containing recent-fill count

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app execution_surface_snapshot_uses_ctrader_reconcile_runtime_when_connected -- --nocapture`

Expected: FAIL because `cTrader` currently produces no fills/timeline.

- [ ] **Step 3: Write minimal implementation**

Add:
- `format_ctrader_deal_line`
- runtime diagnostic count for recent deals
- map `recent_deals` into `bot_timeline`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app execution_surface_snapshot_uses_ctrader_reconcile_runtime_when_connected -- --nocapture`

Expected: PASS

## Chunk 4: Verification

### Task 4: Run full verification

**Files:**
- Verify only

- [ ] **Step 1: Run forex-app tests**

Run: `cargo test -p forex-app -- --nocapture`

- [ ] **Step 2: Run forex-app clippy**

Run: `cargo clippy -p forex-app --all-targets -- -D warnings`

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace -- --nocapture`

- [ ] **Step 4: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_messages.rs crates/forex-app/src/app_services/ctrader_account.rs crates/forex-app/src/app_services/trading.rs docs/superpowers/specs/2026-03-24-ctrader-execution-foundation-design.md docs/superpowers/plans/2026-03-24-ctrader-execution-foundation.md
git commit -m "Add cTrader deals and execution message foundation"
```
