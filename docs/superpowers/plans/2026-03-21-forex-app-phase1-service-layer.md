# Forex App Phase 1 Service Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the forex-app mock discovery/training UI flows with a real application-service layer that exposes explicit job state, progress, cancellation, reports, and canonical log integration.

**Architecture:** Split `crates/forex-app/src/main.rs` into small focused modules. Add app-owned job and service abstractions for `Discovery` and `Training`, keep actual backend execution in `forex-data`, `forex-search`, and `forex-models`, and wire the UI tabs to thin rendering modules that consume service snapshots instead of owning backend logic.

**Tech Stack:** Rust, `eframe`/`egui`, Tokio tasks and channels, existing workspace crates (`forex-data`, `forex-search`, `forex-models`, `mt5-bridge`, `forex-core`), workspace/unit tests, canonical sectioned logging

---

## File Structure

Planned ownership:

- Create: `crates/forex-app/src/app_state.rs`
  - UI state container
  - active tab, selected symbol, service snapshots, log path

- Create: `crates/forex-app/src/app_services/mod.rs`
  - service facade exports

- Create: `crates/forex-app/src/app_services/jobs.rs`
  - `JobId`
  - `JobKind`
  - `JobState`
  - `JobProgress`
  - `JobReport`
  - cancellation primitives

- Create: `crates/forex-app/src/app_services/discovery.rs`
  - discovery request validation
  - async job launch
  - progress/report updates
  - canonical log updates

- Create: `crates/forex-app/src/app_services/training.rs`
  - training request validation
  - async job launch
  - progress/report updates
  - canonical log updates

- Create: `crates/forex-app/src/ui/mod.rs`
  - UI module exports

- Create: `crates/forex-app/src/ui/components.rs`
  - status badges
  - warning/error panels
  - `Open Log` helper

- Create: `crates/forex-app/src/ui/trading.rs`
  - trading tab rendering only

- Create: `crates/forex-app/src/ui/discovery.rs`
  - discovery tab rendering only

- Create: `crates/forex-app/src/ui/training.rs`
  - training tab rendering only

- Modify: `crates/forex-app/src/main.rs`
  - bootstrap only
  - app construction
  - no mock discovery/training logic

- Modify: `crates/forex-app/Cargo.toml`
  - add any small crate-local dependencies only if the implementation actually needs them

- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
  - record that the discovery/training mock-path findings are resolved if verification passes

- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
  - add the new app service verification lane if needed

## Chunk 1: Job Model And Shared App State

### Task 1: Add the app-owned job contract

**Files:**
- Create: `crates/forex-app/src/app_services/jobs.rs`
- Test: inline tests in `crates/forex-app/src/app_services/jobs.rs`

- [ ] **Step 1: Write failing unit tests for the job model**

Add tests for:
- valid initial job state is `Queued`
- `Running -> Cancelled` is representable
- `current` report/progress snapshots can be updated
- invalid or missing report data does not require fake defaults

- [ ] **Step 2: Run the focused crate tests to verify red**

Run: `cargo test -p forex-app jobs -- --nocapture`
Expected: FAIL with missing module/types/tests

- [ ] **Step 3: Implement the minimal job types**

Add:
- `JobId`
- `JobKind`
- `JobState`
- `JobProgress`
- `JobReport`
- `JobSnapshot`
- cancellation flag primitive

- [ ] **Step 4: Re-run the focused crate tests**

Run: `cargo test -p forex-app jobs -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/jobs.rs
git commit -m "Add forex-app job service contract"
```

### Task 2: Add focused app state

**Files:**
- Create: `crates/forex-app/src/app_state.rs`
- Modify: `crates/forex-app/src/main.rs`
- Test: inline tests in `crates/forex-app/src/app_state.rs`

- [ ] **Step 1: Write failing unit tests for app state defaults**

Add tests for:
- symbol selection initializes from discovered symbols
- discovery and training job slots start empty
- canonical log path is available in state

- [ ] **Step 2: Run the focused crate tests to verify red**

Run: `cargo test -p forex-app app_state -- --nocapture`
Expected: FAIL with missing module/types

- [ ] **Step 3: Implement minimal app state**

Move the non-rendering UI state out of `main.rs` into `app_state.rs`.

- [ ] **Step 4: Re-run the focused crate tests**

Run: `cargo test -p forex-app app_state -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_state.rs crates/forex-app/src/main.rs
git commit -m "Extract focused forex-app state"
```

## Chunk 2: Discovery Service

### Task 3: Replace the mock discovery loop with a real service

**Files:**
- Create: `crates/forex-app/src/app_services/discovery.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Test: inline tests in `crates/forex-app/src/app_services/discovery.rs`

- [ ] **Step 1: Write failing tests for discovery request and result mapping**

Add tests for:
- invalid request fails before launch
- cancellation request maps to `Cancelled`
- explicit empty-portfolio backend failure maps to `Failed`
- successful report snapshot can carry candidate/accepted/rejected counters

- [ ] **Step 2: Run the focused crate tests to verify red**

Run: `cargo test -p forex-app discovery_service -- --nocapture`
Expected: FAIL with missing module/types

- [ ] **Step 3: Implement the minimal discovery service**

Implement:
- request type
- service start
- safe cancellation checkpoints
- progress/report snapshot updates
- canonical `DISCOVERY` log records

Do not add fake progress simulation.

- [ ] **Step 4: Re-run the focused crate tests**

Run: `cargo test -p forex-app discovery_service -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/discovery.rs
git commit -m "Add forex-app discovery service"
```

## Chunk 3: Training Service

### Task 4: Replace the log-only training action with a real service

**Files:**
- Create: `crates/forex-app/src/app_services/training.rs`
- Modify: `crates/forex-app/src/app_services/mod.rs`
- Test: inline tests in `crates/forex-app/src/app_services/training.rs`

- [ ] **Step 1: Write failing tests for training request and result mapping**

Add tests for:
- invalid request fails before launch
- unsupported backend outcome maps to explicit `Failed` or `Degraded`
- cancellation request maps to `Cancelled`
- report summary can store completed and failed model names

- [ ] **Step 2: Run the focused crate tests to verify red**

Run: `cargo test -p forex-app training_service -- --nocapture`
Expected: FAIL with missing module/types

- [ ] **Step 3: Implement the minimal training service**

Implement:
- request type
- service start
- safe cancellation checkpoints
- progress/report snapshot updates
- canonical `TRAINING` log records

Do not keep `info!("Training logic triggered.")` as the active path.

- [ ] **Step 4: Re-run the focused crate tests**

Run: `cargo test -p forex-app training_service -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/forex-app/src/app_services/mod.rs crates/forex-app/src/app_services/training.rs
git commit -m "Add forex-app training service"
```

## Chunk 4: UI Extraction And Wiring

### Task 5: Extract shared UI components and thin tab renderers

**Files:**
- Create: `crates/forex-app/src/ui/mod.rs`
- Create: `crates/forex-app/src/ui/components.rs`
- Create: `crates/forex-app/src/ui/trading.rs`
- Create: `crates/forex-app/src/ui/discovery.rs`
- Create: `crates/forex-app/src/ui/training.rs`
- Modify: `crates/forex-app/src/main.rs`
- Test: inline tests in `components.rs` where feasible

- [ ] **Step 1: Write failing tests for small UI helper behavior**

Add tests for:
- status badge helper maps each job state correctly
- `Open Log` helper resolves the canonical log path
- summary formatters preserve warning and error counts

- [ ] **Step 2: Run the focused crate tests to verify red**

Run: `cargo test -p forex-app ui -- --nocapture`
Expected: FAIL with missing helper modules

- [ ] **Step 3: Implement the shared UI helpers**

Add:
- status badge helper
- warning/error panel helper
- `Open Log` helper

- [ ] **Step 4: Extract the trading/discovery/training tabs**

Move the rendering code out of `main.rs` and wire the tabs to:
- service snapshots
- `Start`
- `Stop`
- `Open Log`

- [ ] **Step 5: Re-run the focused crate tests**

Run: `cargo test -p forex-app ui -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/forex-app/src/ui crates/forex-app/src/main.rs
git commit -m "Split forex-app UI tabs and shared components"
```

## Chunk 5: Runtime Integration

### Task 6: Wire app bootstrap to the new service layer

**Files:**
- Modify: `crates/forex-app/src/main.rs`
- Modify: `crates/forex-app/Cargo.toml` only if needed

- [ ] **Step 1: Write or extend bootstrap-focused tests**

Add tests for:
- app bootstrap still uses configured data dir
- services are present in app construction
- no mock discovery/training path remains reachable

- [ ] **Step 2: Run the app crate tests to verify red**

Run: `cargo test -p forex-app -- --nocapture`
Expected: red only for the newly added bootstrap expectations

- [ ] **Step 3: Implement the minimal bootstrap wiring**

Ensure `main.rs` is reduced to:
- args parsing
- settings load
- logging setup
- app creation
- headless behavior

- [ ] **Step 4: Re-run the app crate tests**

Run: `cargo test -p forex-app -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run crate-level lint verification**

Run: `cargo clippy -p forex-app --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/forex-app/src/main.rs crates/forex-app/Cargo.toml
git commit -m "Wire forex-app bootstrap to service layer"
```

## Chunk 6: Full Verification And Audit Update

### Task 7: Verify the supported path and update audit artifacts

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

- [ ] **Step 1: Run real runtime smoke verification**

Run:
- `cargo run -p forex-app -- --headless --config config.yaml`
- `cargo run -p forex-app -- --headless --local --config config.yaml`

Expected:
- app starts cleanly
- no mock discovery/training path remains
- canonical `APP` logging remains intact

- [ ] **Step 2: Inspect the canonical log**

Check:
- `logs/forex-ai.log`

Expected:
- `APP`, `DISCOVERY`, and `TRAINING` sections update correctly
- cancellation records `Cancelled`
- no extra runtime log files are created

- [ ] **Step 3: Run workspace verification**

Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 4: Update the audit artifacts**

Update:
- the finding that `forex-app` discovery is a mock loop
- the finding that `forex-app` training is log-only
- the verification matrix for the new app-service lane

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md
git commit -m "Document forex-app service layer verification"
```
