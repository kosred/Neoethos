# cTrader Token Refresh Session Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add documented `cTrader` refresh-token support so restored sessions can survive access-token expiry and update secure storage safely.

**Architecture:** Extend the existing `ctrader_auth` token model with freshness helpers, add a refresh exchange path in `ctrader_live_auth`, and have `TradingSession` refresh stale bundles before cTrader account/runtime/chart operations. The refresh path must fail closed and overwrite secure storage only with a valid refreshed bundle.

**Tech Stack:** Rust, `reqwest::blocking`, `serde`, secure OS keyring storage, existing `forex-app` service/test harness.

---

## Chunk 1: Spec-Aligned Tests

### Task 1: Add failing token-freshness tests

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- expired token bundles
- near-expiry token bundles
- healthy token bundles

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: FAIL because freshness helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add token freshness helpers to `CTraderTokenBundle`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: PASS

### Task 2: Add failing refresh-contract tests

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_live_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- refresh URL uses `grant_type=refresh_token`
- refreshed token responses parse into a new `CTraderTokenBundle`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: FAIL because refresh helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add refresh request/response helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: PASS

## Chunk 2: Trading Session Auto-Refresh

### Task 3: Add failing trading-session refresh tests

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- restored expired bundle is refreshed before account runtime request
- refreshed bundle is persisted back to the secure store
- refresh failure blocks cTrader runtime/chart access

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader -- --nocapture`
Expected: FAIL because `TradingSession` does not refresh yet.

- [ ] **Step 3: Write minimal implementation**

Add an internal `ensure_fresh_ctrader_token_bundle()` seam and wire it into the cTrader runtime/chart/account paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader -- --nocapture`
Expected: PASS

## Chunk 3: Full Verification

### Task 4: Verify crate and workspace

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Modify: `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- Modify: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Run forex-app tests**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS

- [ ] **Step 2: Run forex-app clippy**

Run: `cargo clippy -p forex-app --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 3: Run workspace tests**

Run: `cargo test --workspace -- --nocapture`
Expected: PASS

- [ ] **Step 4: Run workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 5: Run startup smoke**

Run: `target/debug/forex-app.exe --headless --local --config config.yaml`
Expected: startup succeeds and writes `SYSTEM`/`APP` records into `logs/forex-ai.log`

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-03-24-ctrader-token-refresh-session-hardening-design.md docs/superpowers/plans/2026-03-24-ctrader-token-refresh-session-hardening.md crates/forex-app/src/app_services/ctrader_auth.rs crates/forex-app/src/app_services/ctrader_live_auth.rs crates/forex-app/src/app_services/trading.rs
git commit -m "Harden cTrader token refresh lifecycle"
```
