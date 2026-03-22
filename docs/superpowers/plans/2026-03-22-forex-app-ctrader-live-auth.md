# Forex App cTrader Live Auth Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-safe live `cTrader` login to `forex-app` using the system browser, a loopback callback listener, secure token storage, and restored session state.

**Architecture:** Extend the existing typed `cTrader` auth state machine with a focused live-auth orchestration module and a focused secure-store wrapper. Keep `TradingSession` as the service owner, drive the flow from `System`, and fail closed whenever browser launch, callback capture, token exchange, or secure persistence fails.

**Tech Stack:** Rust, `tokio`, `reqwest`, `serde`, `keyring`, `open`, `eframe`/`egui`, canonical sectioned logging

---

## File Structure

- Create: `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- Create: `crates/forex-app/src/app_services/secure_store.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify: `crates/forex-app/src/app_services/broker_config.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify: `crates/forex-app/Cargo.toml`

## Chunk 1: Live Auth Contracts

### Task 1: Extend the typed cTrader auth model

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Test: inline tests in `ctrader_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `ListeningForCallback` and `ExchangingToken` state transitions
- restored token bundle snapshot state
- callback port and persistence status in the auth snapshot

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: FAIL with missing live-auth state or snapshot fields

- [ ] **Step 3: Write the minimal implementation**

Add:
- new live auth states
- callback port / persistence status fields
- token bundle types
- restored-session state support

- [ ] **Step 4: Re-run focused tests and verify green**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: PASS

## Chunk 2: Loopback Listener And Token Exchange

### Task 2: Add the live auth orchestration module

**Files:**
- Create: `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- Modify: `crates/forex-app/Cargo.toml`
- Test: inline tests in `ctrader_live_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- authorize URL uses the configured redirect URI and callback port
- loopback callback parsing accepts only valid `code` responses
- token exchange request is built correctly for the documented endpoint

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: FAIL with missing live-auth module or helpers

- [ ] **Step 3: Write the minimal implementation**

Add:
- loopback config
- local callback parser
- token request/response types
- system-browser launch helper
- token exchange client helper

- [ ] **Step 4: Re-run focused tests and verify green**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: PASS

## Chunk 3: Secure Storage

### Task 3: Add secure token persistence

**Files:**
- Create: `crates/forex-app/src/app_services/secure_store.rs`
- Test: inline tests in `secure_store.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- save/load/delete round-trip using the secure-store abstraction
- incomplete token data is rejected

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app secure_store -- --nocapture`
Expected: FAIL with missing secure-store wrapper

- [ ] **Step 3: Write the minimal implementation**

Add:
- typed `save_ctrader_token_bundle`
- typed `load_ctrader_token_bundle`
- typed `clear_ctrader_token_bundle`
- fail-closed validation around required token fields

- [ ] **Step 4: Re-run focused tests and verify green**

Run: `cargo test -p forex-app secure_store -- --nocapture`
Expected: PASS

## Chunk 4: Trading Session Integration

### Task 4: Integrate live auth into `TradingSession`

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Modify if needed: `crates/forex-app/src/app_services/broker_config.rs`
- Test: inline tests in `trading.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- selecting `cTrader` can restore a saved session
- start-live-auth maps browser/listener/token/save failures to `Failed`
- successful live auth persists the token bundle and marks restored/auth-ready state
- clearing the saved session removes restored state

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: FAIL with missing live-auth session methods

- [ ] **Step 3: Write the minimal implementation**

Add:
- `start_ctrader_live_auth(...)`
- restore-on-demand helper
- clear-saved-session helper
- service-level mapping from low-level failures into operator-visible status

- [ ] **Step 4: Re-run focused tests and verify green**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: PASS

## Chunk 5: System UI Surface

### Task 5: Surface the live auth controls in `System`

**Files:**
- Modify: `crates/forex-app/src/ui/system_status.rs`

- [ ] **Step 1: Write the failing UI-facing tests**

Add tests for:
- `System` shows live `cTrader` auth waiting state
- `System` shows restored-session status
- `System` shows clear-saved-session action state

- [ ] **Step 2: Run focused tests and verify red**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: FAIL with missing live-auth UI state

- [ ] **Step 3: Write the minimal UI**

Add:
- `Start cTrader Login`
- callback/listener status
- saved-session status
- `Clear Saved Session`

- [ ] **Step 4: Re-run focused tests and verify green**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: PASS

## Chunk 6: Full Verification And Finalization

### Task 6: Verify and finalize the tranche

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-03-22-forex-app-ctrader-live-auth-design.md`
- Modify if needed: `docs/superpowers/plans/2026-03-22-forex-app-ctrader-live-auth.md`

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
git add crates/forex-app/Cargo.toml crates/forex-app/src/app_services/ctrader_auth.rs crates/forex-app/src/app_services/ctrader_live_auth.rs crates/forex-app/src/app_services/secure_store.rs crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/trading.rs crates/forex-app/src/app_services/broker_config.rs crates/forex-app/src/ui/system_status.rs docs/superpowers/specs/2026-03-22-forex-app-ctrader-live-auth-design.md docs/superpowers/plans/2026-03-22-forex-app-ctrader-live-auth.md
git commit -m "Add cTrader live auth flow"
```
