# Forex App Dockable Workspace Shell Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dockable workspace shell to `forex-app`, using the approved chart-first rail direction, while preserving the existing backend service boundaries.

**Architecture:** Introduce `egui_dock` and a small workspace layer that owns layout, tabs, and rendering dispatch only. Keep app services unchanged, embed the current `Discovery`, `Training`, and `System` views into the dockable workspace, and add first-shell trading subpanels as focused modules.

**Tech Stack:** Rust, `eframe`/`egui`, `egui_dock`, existing app-service layer, workspace tests, canonical sectioned logging

---

## File Structure

Planned ownership:

- Create: `crates/forex-app/src/workspace/mod.rs`
- Create: `crates/forex-app/src/workspace/tabs.rs`
- Create: `crates/forex-app/src/workspace/layout.rs`
- Create: `crates/forex-app/src/workspace/viewer.rs`
- Create: `crates/forex-app/src/ui/trading/chart_panel.rs`
- Create: `crates/forex-app/src/ui/trading/watchlist_panel.rs`
- Create: `crates/forex-app/src/ui/trading/execution_panel.rs`
- Create: `crates/forex-app/src/ui/trading/news_panel.rs`
- Create: `crates/forex-app/src/ui/trading/bottom_strip.rs`
- Modify: `crates/forex-app/Cargo.toml`
- Modify: `crates/forex-app/src/app_state.rs`
- Modify: `crates/forex-app/src/main.rs`
- Modify: `crates/forex-app/src/ui/trading.rs`

## Chunk 1: Workspace Model

### Task 1: Add failing tests for the dockable workspace state

**Files:**
- Create: `crates/forex-app/src/workspace/tabs.rs`
- Create: `crates/forex-app/src/workspace/layout.rs`
- Create: `crates/forex-app/src/workspace/mod.rs`
- Test: inline tests in `tabs.rs` and `layout.rs`

- [ ] **Step 1: Write failing tests**

Add tests for:
- the default workspace includes `Chart`, `Watchlist`, `Execution`, `News`, `Bottom Strip`, `Discovery`, `Training`, and `System`
- the default layout places `Chart` in the main center area
- tab labels remain stable

- [ ] **Step 2: Run the focused test lane and verify red**

Run: `cargo test -p forex-app workspace -- --nocapture`
Expected: FAIL with missing workspace modules/types

- [ ] **Step 3: Implement the minimal workspace model**

Add:
- workspace tab enum
- default dock state
- workspace state wrapper

- [ ] **Step 4: Re-run the focused lane**

Run: `cargo test -p forex-app workspace -- --nocapture`
Expected: PASS

## Chunk 2: Trading Workspace Panels

### Task 2: Add first-shell trading panels

**Files:**
- Create: `crates/forex-app/src/ui/trading/chart_panel.rs`
- Create: `crates/forex-app/src/ui/trading/watchlist_panel.rs`
- Create: `crates/forex-app/src/ui/trading/execution_panel.rs`
- Create: `crates/forex-app/src/ui/trading/news_panel.rs`
- Create: `crates/forex-app/src/ui/trading/bottom_strip.rs`
- Modify: `crates/forex-app/src/ui/trading.rs`

- [ ] **Step 1: Write failing tests for the new panel shells**

Add tests for:
- chart panel shows timeframe controls and bot marker placeholders
- watchlist panel surfaces selected symbol/runtime context
- execution panel exposes primary actions
- news panel keeps a distinct operator rail
- bottom strip groups positions/orders/timeline/notes sections

- [ ] **Step 2: Run the focused lane and verify red**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: FAIL with missing panel modules/helpers

- [ ] **Step 3: Implement the minimal panel shells**

Keep them visual/operator-first and do not open scope into full charting or live news feeds.

- [ ] **Step 4: Re-run the focused lane**

Run: `cargo test -p forex-app trading -- --nocapture`
Expected: PASS

## Chunk 3: Dock Viewer Integration

### Task 3: Replace top-tab ownership with dockable shell routing

**Files:**
- Create: `crates/forex-app/src/workspace/viewer.rs`
- Modify: `crates/forex-app/src/app_state.rs`
- Modify: `crates/forex-app/src/main.rs`
- Modify: `crates/forex-app/Cargo.toml`

- [ ] **Step 1: Run the current forex-app lane as baseline**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS before the shell refactor

- [ ] **Step 2: Add `egui_dock` and implement the workspace viewer**

The viewer should:
- route each workspace tab to its renderer
- keep `Discovery`, `Training`, and `System` wired to the existing verified modules
- route the trading workspace panels through the new panel files

- [ ] **Step 3: Replace the top-tab central rendering path**

Keep startup, logging, service message processing, and headless mode intact.

- [ ] **Step 4: Re-run the forex-app lane**

Run:
- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`

Expected: PASS

## Chunk 4: Workspace Verification

### Task 4: Verify the shell change did not regress the workspace

**Files:**
- Modify if needed: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify if needed: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

- [ ] **Step 1: Run workspace verification**

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
- app starts successfully
- canonical `logs/forex-ai.log` records the run in `APP`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-03-22-forex-app-dockable-workspace-shell-design.md docs/superpowers/plans/2026-03-22-forex-app-dockable-workspace-shell.md crates/forex-app/Cargo.toml crates/forex-app/src/app_state.rs crates/forex-app/src/main.rs crates/forex-app/src/workspace crates/forex-app/src/ui/trading.rs crates/forex-app/src/ui/trading
git commit -m "Add forex-app dockable workspace shell"
```
