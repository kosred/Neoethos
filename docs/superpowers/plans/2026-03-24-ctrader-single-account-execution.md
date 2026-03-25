# cTrader Single-Account Execution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real single-account `cTrader` execution with documented order requests, execution/error-event parsing, lot-size validation, and operator-facing `History/Journal` rows with P/L.

**Architecture:** Keep request builders in `ctrader_messages.rs`, execution orchestration in a new `ctrader_execution.rs`, and app-facing state/UI mapping in `trading.rs` and `execution_panel.rs`. This tranche is deliberately single-account only so the execution contract is stable before multi-account fan-out is introduced.

**Tech Stack:** Rust, serde/serde_json, anyhow, tungstenite transport helpers, existing `forex-app` UI/services/tests

---

## Chunk 1: Message/Event Contracts

### Task 1: Add failing tests for execution-event and order-error matching

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/ctrader_messages.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `order_requests_accept_execution_event_as_success_response`
- `order_requests_accept_order_error_event_as_terminal_response`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_messages -- --nocapture`

Expected: FAIL because the message matching does not yet fully express the single-account execution contract.

- [ ] **Step 3: Write minimal implementation**

Use documented payload types and matching rules for:
- `ProtoOAExecutionEvent`
- `ProtoOAOrderErrorEvent`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_messages -- --nocapture`

Expected: PASS

## Chunk 2: Execution Backend

### Task 2: Add failing tests for single-account execution outcomes

**Files:**
- Create: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/ctrader_execution.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/mod.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- successful new market order maps to `Filled` or `Accepted`
- order error event maps to `Failed`
- cancel request maps to `Cancelled`
- close request maps to `Filled`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_execution -- --nocapture`

Expected: FAIL because the backend does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:
- typed request structs for app-side order ticket
- typed `ExecutionOutcome`
- production/stub execution backend
- documented event parsing for execution/error events

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_execution -- --nocapture`

Expected: PASS

## Chunk 3: Trading Service And Lot Size Validation

### Task 3: Add failing tests for lot size validation and journal rows

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- invalid lot size blocks order submission
- successful cTrader execution updates status and journal rows
- journal rows include lot size, status, gross P/L, fees, net P/L when available

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app execution_surface_snapshot_uses_ctrader -- --nocapture`

Expected: FAIL because trading service does not yet own single-account execution state.

- [ ] **Step 3: Write minimal implementation**

Add:
- order ticket state/service contract
- lot size to protocol-volume conversion
- runtime refresh after successful execution outcome
- journal/history row formatting with P/L and fee fields

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app execution_surface_snapshot_uses_ctrader -- --nocapture`

Expected: PASS

## Chunk 4: Execution Panel

### Task 4: Add failing tests for operator order ticket controls

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- connected cTrader runtime enables operator actions
- disconnected or invalid lot size disables actions with explicit reason
- panel exposes lot size and history/journal summary fields

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app execution_panel -- --nocapture`

Expected: FAIL because the UI does not yet render an order ticket/journal contract.

- [ ] **Step 3: Write minimal implementation**

Add:
- lot size input model
- `Buy Market`, `Sell Market`, `Cancel Selected`, `Close Selected`
- thin rendering over the service-layer state

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app execution_panel -- --nocapture`

Expected: PASS

## Chunk 5: Verification

### Task 5: Run full verification

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
git add crates/forex-app/src/app_services/ctrader_messages.rs crates/forex-app/src/app_services/ctrader_execution.rs crates/forex-app/src/app_services/trading.rs crates/forex-app/src/ui/trading/execution_panel.rs docs/superpowers/specs/2026-03-24-ctrader-single-account-execution-design.md docs/superpowers/plans/2026-03-24-ctrader-single-account-execution.md
git commit -m "Add cTrader single-account execution"
```
