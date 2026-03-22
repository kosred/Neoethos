# Forex App cTrader Auth Session Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first real `cTrader` auth/session state machine to `forex-app`, including authorization URL construction, auth-code handling, token request contracts, and operator-visible auth state.

**Architecture:** Introduce a focused `ctrader_auth` service module and let `TradingSession` own the active auth snapshot. Keep networking out of scope and surface the state machine through the `System` panel using typed state instead of ad-hoc strings.

**Tech Stack:** Rust, `eframe`/`egui`, current `TradingSession` service layer, typed auth state machine, workspace UI tests, canonical sectioned logging

---

## File Structure

- Create: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify if needed: `crates/forex-app/src/app_services/broker_config.rs`

## Chunk 1: cTrader Auth Model

### Task 1: Add failing tests for the state machine

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Test: inline tests in `ctrader_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- configured settings produce a valid authorize URL
- receiving an authorization code advances the state
- token request contract is built from the auth code
- account summaries are retained and counted

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: FAIL with missing `ctrader_auth` types or methods

- [ ] **Step 3: Implement the minimal auth model**

Add:
- `CTraderAuthState`
- `CTraderAuthSnapshot`
- `CTraderTokenExchangeRequest`
- `CTraderAccountSummary`
- authorize URL builder and state transitions

- [ ] **Step 4: Re-run the focused lane**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: PASS

## Chunk 2: Trading Session Integration

### Task 2: Teach `TradingSession` to own cTrader auth progression

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: inline tests in `trading.rs`

- [ ] **Step 1: Write failing tests**

Add tests for:
- `start_ctrader_auth()` exposes `AwaitingAuthorizationCode`
- `receive_ctrader_authorization_code()` stores code and updates snapshot
- `build_ctrader_token_exchange_request()` returns the correct request and state

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app start_ctrader_auth_exposes_authorize_url_when_ready -- --nocapture`
Expected: FAIL with missing trading-session methods

- [ ] **Step 3: Implement the minimal integration**

Keep the session typed and do not add network calls.

- [ ] **Step 4: Re-run the focused lane**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: PASS

## Chunk 3: System UI Auth Surface

### Task 3: Surface the auth state in the operator UI

**Files:**
- Modify: `crates/forex-app/src/ui/system_status.rs`

- [ ] **Step 1: Write failing UI-facing tests**

Add tests for:
- the `System` dashboard shows `ReadyToAuthorize` for configured `cTrader`
- the dashboard shows `AuthorizationCodeReceived` after code intake
- the dashboard shows account count when account summaries exist

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app system_status_dashboard_surfaces_ctrader_auth_state -- --nocapture`
Expected: FAIL with missing auth summary rows

- [ ] **Step 3: Implement the minimal UI**

Add:
- `cTrader Auth` section
- start auth action
- auth code field
- auth state summary rows

- [ ] **Step 4: Re-run the focused lane**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS

## Chunk 4: Full Verification

### Task 4: Verify and finalize the tranche

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-03-22-forex-app-ctrader-auth-session-design.md`
- Modify if needed: `docs/superpowers/plans/2026-03-22-forex-app-ctrader-auth-session.md`

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
git add docs/superpowers/specs/2026-03-22-forex-app-ctrader-auth-session-design.md docs/superpowers/plans/2026-03-22-forex-app-ctrader-auth-session.md crates/forex-app/src/app_services/ctrader_auth.rs crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/trading.rs crates/forex-app/src/ui/system_status.rs
git commit -m "Add cTrader auth session state"
```
