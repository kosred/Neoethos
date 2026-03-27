# Truthfulness + Safety Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove misleading operator-facing runtime states, enforce a real AI auto-trade backend gate, and make capability reporting honest across the app shell.

**Architecture:** The work is split into three layers: a truth snapshot/gating layer in the trading service, UI cleanup to remove synthetic operator values, and a capability honesty pass across dashboard/system/execution surfaces. The existing app shell stays intact; only the operator truth contract changes.

**Tech Stack:** Rust, egui/eframe, tokio, tracing, existing canonical sectioned logging, existing forex-app service/UI modules.

---

## File Structure

- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs`
  - remove or neutralize placeholder operator defaults
  - hold explicit truth-state fields only where needed
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`
  - add backend auto-trade gate
  - expose honest truth/capability snapshots
  - journal denied AI/bot actions
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/dashboard.rs`
  - stop showing synthetic equity as live state
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/ai_insights.rs`
  - stop showing hardcoded probabilities/compliance/latency as truth
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/system_status.rs`
  - surface truthful capability/readiness state
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs`
  - fix misleading ribbon metrics
- Optional Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs`
  - shared unavailable/degraded rendering helpers
- Optional Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs`
  - reflect denied AI/bot execution reasons honestly

## Chunk 1: Backend Truth And Safety Gate

### Task 1: Add failing tests for auto-trade backend gating

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for AI execution denial**

Add tests covering:

- AI/bot execution denied when `auto_trade_enabled == false`
- manual execution path is not denied by the auto-trade gate alone
- denied execution appends an explicit journal reason

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app trading -- --nocapture
```

Expected: one or more new tests fail because the backend gate does not yet exist.

- [ ] **Step 3: Implement minimal backend gate**

In `trading.rs`, add:

- explicit helper to classify execution origin:
  - manual
  - AI/bot
- gate function that denies AI/bot execution when auto-trade is disabled
- journal/log helper for denied execution

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app trading -- --nocapture
```

Expected: new gating tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs
git commit -m "feat: enforce backend auto-trade safety gate"
```

### Task 2: Add truthful capability snapshot tests

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for unavailable/stubbed truth states**

Add tests asserting that snapshot fields return explicit states such as:

- `Unavailable`
- `Not wired`
- `Stubbed`

rather than synthetic numeric/operator values.

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app trading -- --nocapture
```

Expected: snapshot truth tests fail before implementation.

- [ ] **Step 3: Implement capability truth mapping**

Update snapshot builders in `trading.rs` to expose honest status text for:

- AI signal availability
- compliance/news state availability
- runtime telemetry availability
- equity/feed availability

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app trading -- --nocapture
```

Expected: truth snapshot tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs
git commit -m "feat: add truthful capability snapshots"
```

## Chunk 2: UI Truthfulness Cleanup

### Task 3: Remove synthetic dashboard truth

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/dashboard.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs`

- [ ] **Step 1: Write failing UI tests for fake equity/default messages**

Add tests asserting:

- default dashboard does not render a synthetic equity curve as live runtime truth
- safe-mode/auto-trade messaging reflects actual execution-gate semantics

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app dashboard -- --nocapture
```

Expected: new tests fail before cleanup.

- [ ] **Step 3: Implement minimal dashboard cleanup**

Change dashboard behavior to:

- show explicit unavailable/no-live-equity state when no real feed exists
- stop implying live execution authority from a cosmetic switch alone

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app dashboard -- --nocapture
```

Expected: dashboard truth tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/dashboard.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs
git commit -m "fix: remove synthetic dashboard runtime states"
```

### Task 4: Remove synthetic AI insight truth

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/ai_insights.rs`
- Optional Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs`

- [ ] **Step 1: Write failing tests for hardcoded AI/compliance labels**

Add tests asserting the panel no longer presents:

- hardcoded confidence values
- hardcoded compliance safety
- hardcoded latency

as live truth when no source exists.

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app ai_insights -- --nocapture
```

Expected: tests fail before implementation.

- [ ] **Step 3: Implement truthful AI insight rendering**

Render explicit states such as:

- `Model signal unavailable`
- `Compliance state unavailable`
- `Latency unavailable`

unless backed by real values.

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app ai_insights -- --nocapture
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/ai_insights.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs
git commit -m "fix: make AI insights panel truthful"
```

### Task 5: Fix top-ribbon misleading metrics

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs`

- [ ] **Step 1: Write failing test for misleading CPU/metric rendering**

Add a focused test asserting the ribbon does not label core count as CPU percentage.

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app main -- --nocapture
```

Expected: new ribbon truth test fails before implementation.

- [ ] **Step 3: Implement truthful ribbon behavior**

Change the ribbon to show:

- a real metric if available
- otherwise explicit unavailable/unknown text

Do not synthesize CPU percent from core count.

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app main -- --nocapture
```

Expected: ribbon truth test passes.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs
git commit -m "fix: remove misleading ribbon metrics"
```

## Chunk 3: System And Execution Honesty Pass

### Task 6: Make system status report capability truth

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/system_status.rs`

- [ ] **Step 1: Write failing tests for capability honesty**

Add tests asserting `System Status` surfaces:

- disconnected
- unavailable
- not wired
- stubbed

correctly instead of optimistic labels.

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app system_status -- --nocapture
```

Expected: tests fail before implementation.

- [ ] **Step 3: Implement honest system status rendering**

Wire the panel to the service truth/capability snapshot rather than local optimistic messaging.

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app system_status -- --nocapture
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/system_status.rs
git commit -m "fix: make system status capability reporting honest"
```

### Task 7: Align execution panel with real permission semantics

**Files:**
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs`
- Modify: `C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs`

- [ ] **Step 1: Write failing tests for denied AI/bot execution visibility**

Add tests asserting:

- AI/bot denied actions appear explicitly in journal rows
- manual controls do not misleadingly appear blocked by the AI switch alone

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```powershell
cargo test -p forex-app execution_panel -- --nocapture
```

Expected: tests fail before implementation.

- [ ] **Step 3: Implement execution panel truth pass**

Show real denial reasons and ensure the UI reflects the backend gate outcome honestly.

- [ ] **Step 4: Re-run targeted tests**

Run:

```powershell
cargo test -p forex-app execution_panel -- --nocapture
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs
git commit -m "fix: align execution panel with backend safety semantics"
```

## Chunk 4: Final Verification

### Task 8: Run full verification and save final clean state

**Files:**
- No code changes required unless verification fails

- [ ] **Step 1: Run forex-app verification**

```powershell
cargo test -p forex-app -- --nocapture
cargo clippy -p forex-app --all-targets -- -D warnings
```

Expected: PASS

- [ ] **Step 2: Run workspace verification**

```powershell
cargo test --workspace -- --nocapture
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS

- [ ] **Step 3: Run headless smoke**

```powershell
target/debug/forex-app.exe --headless --local --config config.yaml
```

Expected: startup succeeds without misleading runtime state regressions.

- [ ] **Step 4: Inspect canonical log**

Check:

`C:/Users/konst/development/forex-ai/logs/forex-ai.log`

Expected:

- startup records present
- denied AI execution actions are journaled/logged if exercised

- [ ] **Step 5: Commit final tranche**

```powershell
git add C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/trading.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/dashboard.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/ai_insights.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/system_status.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading/execution_panel.rs
git commit -m "fix: harden operator truthfulness and execution safety"
```

Plan complete and saved to `C:/Users/konst/development/forex-ai/docs/superpowers/plans/2026-03-27-truthfulness-safety-hardening.md`. Ready to execute?
