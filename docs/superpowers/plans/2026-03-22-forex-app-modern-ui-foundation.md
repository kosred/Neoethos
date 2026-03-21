# Forex App Modern UI Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable visual system to `forex-app` so the current operator shell looks modern, stays cross-platform-safe, and reduces repeated dashboard rendering logic.

**Architecture:** Add a focused UI theme module, centralize shared dashboard primitives in `ui/components.rs`, and then rewire the active tabs to use the shared rendering layer. Keep the current service contracts and only modernize the visual shell around them.

**Tech Stack:** Rust, `eframe`/`egui`, workspace tests, existing app-service layer, canonical sectioned logging

---

## File Structure

Planned ownership:

- Create: `crates/forex-app/src/ui/theme.rs`
  - app-wide palette and spacing tokens
  - `apply_theme(&egui::Context)`
  - shared frame helpers

- Modify: `crates/forex-app/src/ui/mod.rs`
  - export the new theme module

- Modify: `crates/forex-app/src/ui/components.rs`
  - define shared dashboard card/section models
  - add reusable card and section renderers
  - keep report and log helpers

- Modify: `crates/forex-app/src/ui/discovery.rs`
  - keep data grouping local
  - remove duplicated summary/section rendering

- Modify: `crates/forex-app/src/ui/training.rs`
  - keep data grouping local
  - remove duplicated summary/section rendering

- Modify: `crates/forex-app/src/ui/trading.rs`
  - keep adapter-specific grouping local
  - remove duplicated summary/section rendering

- Modify: `crates/forex-app/src/ui/system_status.rs`
  - keep system-specific grouping local
  - remove duplicated summary/section rendering

- Modify: `crates/forex-app/src/main.rs`
  - apply theme during app construction
  - keep shell wiring clean

## Chunk 1: Theme Foundation

### Task 1: Add failing tests for the app theme

**Files:**
- Create: `crates/forex-app/src/ui/theme.rs`
- Test: inline tests in `crates/forex-app/src/ui/theme.rs`

- [ ] **Step 1: Write failing tests**

Add tests for:
- theme application changes the default visuals
- the operator accent palette stays stable

- [ ] **Step 2: Run the focused test lane and verify red**

Run: `cargo test -p forex-app theme -- --nocapture`
Expected: FAIL with missing module or missing theme helpers

- [ ] **Step 3: Implement the minimal theme module**

Add:
- palette constants
- theme application function
- small frame helpers for cards and sections

- [ ] **Step 4: Re-run the focused theme tests**

Run: `cargo test -p forex-app theme -- --nocapture`
Expected: PASS

## Chunk 2: Shared Dashboard Primitives

### Task 2: Centralize summary cards and section rendering

**Files:**
- Modify: `crates/forex-app/src/ui/components.rs`
- Test: inline tests in `crates/forex-app/src/ui/components.rs`

- [ ] **Step 1: Write failing tests for shared dashboard primitives**

Add tests for:
- card/section helpers preserve labels and values
- status and severity colors remain stable under the new theme

- [ ] **Step 2: Run the focused test lane and verify red**

Run: `cargo test -p forex-app components -- --nocapture`
Expected: FAIL with missing shared dashboard helpers

- [ ] **Step 3: Implement the minimal shared components**

Add:
- shared `DashboardCard`
- shared `DashboardSection`
- reusable summary-card renderer
- reusable section-grid renderer

- [ ] **Step 4: Re-run the focused test lane**

Run: `cargo test -p forex-app components -- --nocapture`
Expected: PASS

## Chunk 3: Apply The New Visual System

### Task 3: Rewire the active operator tabs to the shared rendering layer

**Files:**
- Modify: `crates/forex-app/src/ui/discovery.rs`
- Modify: `crates/forex-app/src/ui/training.rs`
- Modify: `crates/forex-app/src/ui/trading.rs`
- Modify: `crates/forex-app/src/ui/system_status.rs`
- Modify: `crates/forex-app/src/main.rs`

- [ ] **Step 1: Run the existing forex-app UI tests as the safety net**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS before refactor

- [ ] **Step 2: Apply the theme in app startup**

Wire `theme::apply_theme` into app creation.

- [ ] **Step 3: Replace duplicated card/section rendering**

Keep each tab responsible only for:
- grouping its own metrics
- passing those groups into shared renderers

- [ ] **Step 4: Re-run the full forex-app lane**

Run:
- `cargo test -p forex-app -- --nocapture`
- `cargo clippy -p forex-app --all-targets -- -D warnings`

Expected: PASS

## Chunk 4: Workspace Verification

### Task 4: Verify the modernization tranche does not regress the app shell

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
git add docs/superpowers/specs/2026-03-22-forex-app-modern-ui-foundation-design.md docs/superpowers/plans/2026-03-22-forex-app-modern-ui-foundation.md crates/forex-app/src/main.rs crates/forex-app/src/ui/mod.rs crates/forex-app/src/ui/theme.rs crates/forex-app/src/ui/components.rs crates/forex-app/src/ui/discovery.rs crates/forex-app/src/ui/training.rs crates/forex-app/src/ui/trading.rs crates/forex-app/src/ui/system_status.rs
git commit -m "Add forex-app modern UI foundation"
```
