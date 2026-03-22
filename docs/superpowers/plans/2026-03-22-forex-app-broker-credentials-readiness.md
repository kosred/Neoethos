# Forex App Broker Credentials Readiness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add app-level broker credentials and execution-target readiness to `forex-app` so adapter selection becomes truly configurable and remote adapters become auth-ready instead of placeholder-only.

**Architecture:** Introduce a focused broker configuration module owned by `TradingSession`, then surface readiness and account targets through `System` and `Execution`. Keep real remote authentication out of scope and make all unwired behavior explicit.

**Tech Stack:** Rust, `eframe`/`egui`, current `TradingSession` service layer, typed readiness validation, workspace UI tests, canonical sectioned logging

---

## File Structure

- Create: `crates/forex-app/src/app_services/broker_config.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify: `crates/forex-app/src/ui/trading/execution_panel.rs`
- Modify: `crates/forex-app/src/workspace/viewer.rs` only if wiring changes are needed

## Chunk 1: Broker Readiness Model

### Task 1: Add failing readiness tests

**Files:**
- Create: `crates/forex-app/src/app_services/broker_config.rs`
- Test: inline tests in `broker_config.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- incomplete `cTrader` config reports missing OAuth fields
- complete `cTrader` config reports `ReadyForAuth`
- incomplete `DXtrade` config reports missing remote fields
- enabled execution targets count is correct

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app ctrader_readiness_requires_oauth_fields_and_counts_targets -- --nocapture`
Expected: FAIL with missing broker readiness types or methods

- [ ] **Step 3: Implement the minimal readiness model**

Add:
- `BrokerSettingsState`
- `BrokerAccountTarget`
- `BrokerSessionState`
- `AdapterReadinessSnapshot`
- adapter validation helpers

- [ ] **Step 4: Re-run focused tests**

Run: `cargo test -p forex-app broker_config -- --nocapture`
Expected: PASS

## Chunk 2: Trading Session Integration

### Task 2: Teach `TradingSession` to use broker readiness

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: inline tests in `trading.rs`

- [ ] **Step 1: Write failing tests**

Add tests for:
- selecting a remote adapter keeps configuration state available
- unconfigured remote connect reports missing credentials
- configured remote adapter reports auth-ready but still unwired

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app connect_sets_missing_credentials_status_for_unready_remote_adapter -- --nocapture`
Expected: FAIL with missing readiness/session methods

- [ ] **Step 3: Implement the minimal session integration**

Add:
- config ownership inside `TradingSession`
- readiness/query helpers
- connect gating for remote adapters
- explicit status messages and log events

- [ ] **Step 4: Re-run focused tests**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: PASS

## Chunk 3: System And Execution UI

### Task 3: Surface configuration and gating in the operator UI

**Files:**
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify: `crates/forex-app/src/ui/trading/execution_panel.rs`

- [ ] **Step 1: Write failing UI-facing tests**

Add tests for:
- `System` shows readiness and target counts for selected remote adapter
- `Execution` exposes disabled connect state when adapter is not ready

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app system_status_dashboard_surfaces_selected_remote_adapter_metadata -- --nocapture`
Expected: FAIL for missing readiness fields or summaries

- [ ] **Step 3: Implement the minimal operator UI**

Add:
- adapter-specific config form
- readiness summary rows
- account target toggles
- disabled connect button with explicit reason

- [ ] **Step 4: Re-run focused tests**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS

## Chunk 4: Full Verification

### Task 4: Verify the tranche end-to-end

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-03-22-forex-app-broker-credentials-readiness-design.md`
- Modify if needed: `docs/superpowers/plans/2026-03-22-forex-app-broker-credentials-readiness.md`

- [ ] **Step 1: Run full verification**

Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 2: Run headless smoke**

Run:

```powershell
target/debug/forex-app.exe --headless --local --config config.yaml
```

Expected:
- startup succeeds
- canonical `logs/forex-ai.log` records `SYSTEM setup_logging SUCCESS`
- canonical `logs/forex-ai.log` records `APP headless_local_start SUCCESS`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-03-22-forex-app-broker-credentials-readiness-design.md docs/superpowers/plans/2026-03-22-forex-app-broker-credentials-readiness.md crates/forex-app/src/app_services/broker_config.rs crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/trading.rs crates/forex-app/src/ui/system_status.rs crates/forex-app/src/ui/trading/execution_panel.rs
git commit -m "Add broker credential readiness surfaces"
```
