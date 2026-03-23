# Forex App cTrader Account Discovery Target Sync Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real post-auth `cTrader` account discovery, surface the discovered catalog in the UI, and synchronize discovered accounts into execution targets without breaking other adapters.

**Architecture:** Extend the existing `TradingSession` cTrader seams instead of redesigning them. Keep discovered accounts separate from generic execution targets, then merge by `account_id` so operator target choices survive discovery refreshes.

**Tech Stack:** Rust, `egui/eframe`, existing `cTrader` auth service seams, workspace tests, `cargo test`, `clippy`

---

## Chunk 1: Auth And Discovery State

### Task 1: Enrich cTrader auth state for discovered accounts

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_auth.rs`
- Test: `crates/forex-app/src/app_services/ctrader_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- discovered accounts are retained in snapshot
- restored token state is different from accounts-available state
- enabled target count is derived from discovered accounts

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: FAIL due to missing discovered-account behavior

- [ ] **Step 3: Write minimal implementation**

Add:
- richer discovered-account fields
- explicit discovered-account snapshot visibility
- dedicated methods such as `set_discovered_accounts(...)`

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_auth -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_auth.rs
git commit -m "feat: enrich ctrader auth account state"
```

### Task 2: Add cTrader account discovery transport contract

**Files:**
- Modify: `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- Test: `crates/forex-app/src/app_services/ctrader_live_auth.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- discovery request/result types
- app-auth + account-list flow contract shape
- discovered accounts parsed into the app-facing model

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: FAIL because discovery transport path is missing

- [ ] **Step 3: Write minimal implementation**

Add:
- `CTraderAccountDiscoveryRequest`
- `CTraderAccountDiscoveryResult`
- production/test backend seam for account discovery

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app ctrader_live_auth -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/ctrader_live_auth.rs
git commit -m "feat: add ctrader account discovery transport"
```

## Chunk 2: Service Orchestration And Target Sync

### Task 3: Extend broker config for richer cTrader readiness and compatibility-safe target metadata

**Files:**
- Modify: `crates/forex-app/src/app_services/broker_config.rs`
- Test: `crates/forex-app/src/app_services/broker_config.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `SessionRestored` vs `AccountsAvailable`
- compatible target counting after discovery sync
- optional metadata does not break MT5/DXtrade readiness

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app broker_config -- --nocapture`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Extend:
- `BrokerSessionState`
- optionally `BrokerAccountTarget` with compatibility-safe metadata

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app broker_config -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/broker_config.rs
git commit -m "feat: refine broker readiness for discovered accounts"
```

### Task 4: Orchestrate cTrader discovery and safe target merge in TradingSession

**Files:**
- Modify: `crates/forex-app/src/app_services/trading.rs`
- Test: `crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- post-auth discovery populates discovered accounts
- discovery sync preserves existing `enabled_for_execution`
- missing accounts become unavailable/stale instead of disappearing
- restored session does not skip account discovery

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Add:
- `discover_ctrader_accounts()`
- sync/upsert merge by `account_id`
- better `connect()`/status messaging for cTrader
- safer handling of cTrader state in reset/select flows

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/trading.rs
git commit -m "feat: sync discovered ctrader accounts into targets"
```

## Chunk 3: UI Surface And Final Verification

### Task 5: Replace cTrader freeform targets with discovered-account selection in System UI

**Files:**
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Test: `crates/forex-app/src/ui/system_status.rs`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- discovered accounts section renders
- enabled target count reflects discovered selection
- cTrader auth dashboard shows session-restored vs accounts-available correctly

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Render:
- discovered account table
- target enable toggles
- fallback guidance when discovery is unavailable

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p forex-app system_status -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/ui/system_status.rs
git commit -m "feat: show discovered ctrader accounts in system ui"
```

### Task 6: Run full verification and update docs if needed

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

- [ ] **Step 1: Run focused crate verification**

Run:
- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 2: Run full workspace verification**

Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 3: Run runtime smoke**

Run:
- `target/debug/forex-app.exe --headless --local --config config.yaml`

Expected: PASS and canonical log shows successful app startup

- [ ] **Step 4: Update audit artifacts if behavior/status changed materially**

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md
git commit -m "docs: update audit artifacts for ctrader account discovery"
```
