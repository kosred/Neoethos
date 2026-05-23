# egui Removal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy egui desktop UI from `neoethos-app` while preserving the Rust server/backend behavior consumed by the Flutter UI.

**Architecture:** Flutter is the only user-facing UI. `neoethos-app` remains a Rust backend binary for HTTP server, headless runs, reauth, and API-test workflows. The legacy `src/ui/**` tree is treated as quarantined historical code: nothing is copied into Flutter, and any candidate non-render logic must pass tests and service-layer review before reuse.

**Tech Stack:** Rust 2024, Axum, Tokio, Clap, Flutter, Riverpod, Dio.

---

## Ground Rules

- No egui widget, layout, dock, theme, copy, mock state, or wizard UI code is ported to Flutter.
- `crates/neoethos-app/src/ui/**` and `crates/neoethos-app/src/workspace/**` are not source of truth. They are legacy code and may contain old bugs.
- If a behavior exists in `app_services`, `server`, `neoethos-core`, `neoethos-data`, or `neoethos-models`, use that implementation instead of anything from `src/ui/**`.
- If logic only exists under `src/ui/**`, it must be audited, tested, and moved into a non-UI Rust module before egui deletion.
- Flutter must remain a thin server-driven client. It renders API results and calls backend endpoints; it must not inherit egui business rules.
- User-facing branding should be `neoethos` everywhere. Avoid `neoethos app`, `NeoEthos App`, or visible `app` suffixes in product labels. Internal crate and binary names can remain `neoethos-app` unless a separate rename project is approved.

## Current Evidence

- `neoethos-app` defaults to HTTP server mode and only launches egui with `--gui`: `crates/neoethos-app/src/main.rs`.
- egui is still compiled through `eframe`, `egui`, and `egui_dock`: `crates/neoethos-app/Cargo.toml`.
- The Rust server routes already cover the Flutter backend client surface: `crates/neoethos-app/src/server/mod.rs` and `experiments/forex-flutter-ui/lib/api/backend_client.dart`.
- The egui UI/workspace tree is large enough to remove deliberately: `crates/neoethos-app/src/ui/**` and `crates/neoethos-app/src/workspace/**`.
- The working tree is already dirty. Do not revert unrelated user changes.

## Acceptance Gates

- [x] `cargo check -p neoethos-app --no-default-features` passes.
- [x] `cargo test -p neoethos-app` passes, or pre-existing unrelated failures are documented.
- [x] `cargo test -p neoethos-core` passes, or pre-existing unrelated failures are documented.
- [x] From `experiments/forex-flutter-ui`, `flutter analyze` passes.
- [x] From `experiments/forex-flutter-ui`, `flutter test` passes.
- [x] `rg -n "\b(egui|eframe|egui_dock)\b" crates/neoethos-app/src crates/neoethos-app/Cargo.toml` returns no live dependency or source references.
- [x] `cargo tree -p neoethos-app -i egui` reports no dependency path.
- [x] Every `_dio.get` and `_dio.post` path in `experiments/forex-flutter-ui/lib/api/backend_client.dart` still has a matching route in `crates/neoethos-app/src/server/mod.rs`.
- [x] User-facing labels in Flutter and packaging use `neoethos`, not `neoethos app`.

## File Structure

### Files to Modify

- `crates/neoethos-app/Cargo.toml`
  - Remove `eframe`, `egui`, and `egui_dock`.
  - Keep backend/server/headless dependencies.

- `crates/neoethos-app/src/main.rs`
  - Remove egui imports, `--gui`, `eframe::run_native`, icon texture helpers, `ForexApp`, and `impl eframe::App`.
  - Keep `--server`, default server dispatch, `--headless`, `--reauth`, and `--api-test`.

- `crates/neoethos-app/src/app_state.rs`
  - Audit for UI-only panel state.
  - Keep or move runtime config needed by server/headless only.

- `crates/neoethos-app/src/server/**`
  - Only modify if a verified non-UI behavior must be exposed to Flutter.
  - Add route tests for any new or changed endpoint.

- `crates/neoethos-app/src/app_services/**`
  - Move validated non-render logic here when it belongs to backend service orchestration.
  - Add unit tests around moved behavior before wiring it to routes.

- `experiments/forex-flutter-ui/**`
  - Branding-only changes for `neoethos` labels if needed.
  - No egui-derived UI behavior.

- `packaging/**`
  - Update user-facing product names and shortcuts to `neoethos`.
  - Keep internal backend binary paths unless explicitly renamed later.

### Files to Delete After Audit

- `crates/neoethos-app/src/ui/**`
- `crates/neoethos-app/src/workspace/**`

### Files Not to Use as Source of Truth

- `crates/neoethos-app/src/ui/theme.rs`
- `crates/neoethos-app/src/ui/dashboard.rs`
- `crates/neoethos-app/src/ui/trading/**`
- `crates/neoethos-app/src/ui/system/**`
- `crates/neoethos-app/src/ui/wizard/**`

These files can be inspected to understand historical intent, but they must not be copied into Flutter or treated as correct without independent tests.

---

## Chunk 1: Baseline and Safety

### Task 1: Record Baseline State

**Files:**
- Read: `crates/neoethos-app/Cargo.toml`
- Read: `crates/neoethos-app/src/main.rs`
- Read: `crates/neoethos-app/src/app_state.rs`
- Read: `crates/neoethos-app/src/server/mod.rs`
- Read: `experiments/forex-flutter-ui/lib/api/backend_client.dart`

- [ ] **Step 1: Confirm dirty worktree**

Run:

```powershell
git status --short --branch
```

Expected: Existing modified files may be present. Do not revert them.

- [ ] **Step 2: Capture egui dependency paths**

Run:

```powershell
rg -n "\b(egui|eframe|egui_dock)\b" crates/neoethos-app/src crates/neoethos-app/Cargo.toml Cargo.lock
cargo tree -p neoethos-app -i egui
```

Expected: Current egui references are visible and can be compared after removal.

- [ ] **Step 3: Run backend baseline check**

Run:

```powershell
cargo check -p neoethos-app --no-default-features
```

Expected: PASS, or document exact pre-existing failures before changing code.

- [ ] **Step 4: Run Flutter baseline checks**

Run:

```powershell
Set-Location experiments/forex-flutter-ui
flutter analyze
flutter test
```

Expected: PASS, or document exact pre-existing failures before changing code.

- [ ] **Step 5: Commit baseline note if needed**

Only commit if a baseline documentation file is created. Do not commit unrelated dirty work.

---

## Chunk 2: Quarantine Audit

### Task 2: Classify Legacy UI Code Before Deletion

**Files:**
- Audit: `crates/neoethos-app/src/ui/**`
- Audit: `crates/neoethos-app/src/workspace/**`
- Prefer: `crates/neoethos-app/src/app_services/**`
- Prefer: `crates/neoethos-app/src/server/**`
- Prefer: `crates/neoethos-core/**`
- Prefer: `crates/neoethos-data/**`

- [ ] **Step 1: Build an audit table**

Create a temporary implementation note or issue checklist with columns:

```text
legacy file | category | candidate behavior | existing source of truth | decision | tests needed
```

Categories:

```text
render-only
business-logic
config-writer
broker-auth
risk-safety
data-history
dead-code
unknown
```

- [ ] **Step 2: Mark render-only code for deletion**

Render-only includes egui panels, dock layout, theme tokens, widgets, UI copy, scroll state, selected tabs, and visual chrome.

Expected: These files are not moved anywhere.

- [ ] **Step 3: Audit high-risk areas with multiple checks**

High-risk files must be checked against server/service code before reuse:

```text
crates/neoethos-app/src/ui/wizard/oauth.rs
crates/neoethos-app/src/ui/wizard/summary.rs
crates/neoethos-app/src/ui/wizard/historical.rs
crates/neoethos-app/src/ui/wizard/autonomy_risk.rs
crates/neoethos-app/src/ui/system/bootstrap.rs
crates/neoethos-app/src/ui/system/brokers.rs
crates/neoethos-app/src/ui/trading/execution_panel.rs
```

For each candidate behavior, verify:

```text
1. Is this already implemented in app_services/server/core/data?
2. Does the egui version have known stale defaults, hardcoded values, or old assumptions?
3. Can it be expressed as backend service logic without UI state?
4. Is there a failing test that proves the behavior should exist?
5. Does Flutter need a server endpoint for it, or is it unnecessary?
```

- [ ] **Step 4: Reject direct Flutter migration**

Search for proposed direct ports:

```powershell
rg -n "wizard|egui|Dock|WorkspaceTab|ForexApp|ui::" experiments/forex-flutter-ui/lib
```

Expected: No egui-derived Flutter behavior is introduced.

### Task 3: Add Tests for Any Salvaged Logic

**Files:**
- Modify as needed: `crates/neoethos-app/src/app_services/**`
- Modify as needed: `crates/neoethos-app/src/server/**`
- Test as needed: Rust unit tests colocated in service/server modules

- [ ] **Step 1: Write failing Rust tests first**

For every salvaged behavior from `src/ui/**`, write a test before moving implementation.

Example pattern:

```rust
#[test]
fn validates_canonical_timeframe_before_fetching_history() {
    let err = validate_history_request("EURUSD", "H2").unwrap_err();
    assert!(err.to_string().contains("canonical"));
}
```

Expected: Test fails because the service behavior is missing or incomplete.

- [ ] **Step 2: Implement in backend service layer**

Move only the minimal behavior into the correct non-UI module.

Rules:

```text
broker/auth logic -> app_services
order/risk guard logic -> app_services/trading or neoethos-core
history/data logic -> app_services or neoethos-data
HTTP DTO/wiring -> server
Flutter-only rendering -> nowhere
```

- [ ] **Step 3: Run focused tests**

Run:

```powershell
cargo test -p neoethos-app <test_name>
```

Expected: New test passes.

- [ ] **Step 4: Verify Flutter contract if endpoint changed**

If a route or DTO changed, add or update route tests and inspect `backend_client.dart` parsing.

Run:

```powershell
cargo test -p neoethos-app server::
```

Expected: Route tests pass.

---

## Chunk 3: Remove egui Runtime

### Task 4: Remove the GUI CLI Path

**Files:**
- Modify: `crates/neoethos-app/src/main.rs`

- [ ] **Step 1: Remove egui module declarations**

Remove:

```rust
mod ui;
mod workspace;
```

Expected: Compile errors reveal remaining UI dependencies.

- [ ] **Step 2: Remove egui imports and helpers**

Remove:

```rust
use eframe::egui;
use crate::ui::components::render_ribbon_item;
fn neoethos_icon_data(...)
fn load_neoethos_texture(...)
```

Also remove embedded icon usage if it only supports the egui window.

- [ ] **Step 3: Remove `--gui`**

Remove the `gui: bool` field and its help text from `Args`.

Expected: `neoethos-app` no longer advertises the legacy GUI.

- [ ] **Step 4: Remove `eframe::run_native` dispatch**

Delete the branch that starts legacy egui mode.

Keep this dispatch order:

```text
api-test
reauth
headless
server default
```

- [ ] **Step 5: Remove `ForexApp` and `impl eframe::App`**

Delete the egui app struct and all render-loop methods from `main.rs`.

- [ ] **Step 6: Check compile errors**

Run:

```powershell
cargo check -p neoethos-app --no-default-features
```

Expected: Only remaining errors should identify real non-UI references that still need relocation or deletion.

### Task 5: Remove egui Dependencies

**Files:**
- Modify: `crates/neoethos-app/Cargo.toml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Remove dependency entries**

Remove:

```toml
eframe = "0.31.0"
egui = "0.31.0"
egui_dock = "0.16.0"
```

- [ ] **Step 2: Update stale comments**

Remove or rewrite comments that mention:

```text
egui UI tree
egui AI Helper panel
egui main loop
legacy egui GUI
```

Keep comments about the backend server and Flutter migration.

- [ ] **Step 3: Regenerate lockfile through check**

Run:

```powershell
cargo check -p neoethos-app --no-default-features
```

Expected: `Cargo.lock` drops unused egui-related packages if no other crate depends on them.

- [ ] **Step 4: Verify dependency removal**

Run:

```powershell
cargo tree -p neoethos-app -i egui
```

Expected: No dependency path. If Cargo reports `package ID specification egui did not match`, that is also acceptable after removal.

---

## Chunk 4: Delete Legacy UI Tree

### Task 6: Delete Quarantined UI Files

**Files:**
- Delete: `crates/neoethos-app/src/ui/**`
- Delete: `crates/neoethos-app/src/workspace/**`

- [ ] **Step 1: Confirm audit decisions**

Before deletion, confirm all non-render candidate logic is either:

```text
already represented in app_services/server/core/data
tested and moved
explicitly rejected
```

- [ ] **Step 2: Delete `src/ui/**`**

Delete only after Task 2 and Task 3 are complete.

- [ ] **Step 3: Delete `src/workspace/**`**

Delete workspace dock state and tab viewer code.

- [ ] **Step 4: Run source search**

Run:

```powershell
rg -n "\b(ui::|workspace::|WorkspaceTab|WorkspaceState|ForexApp|egui|eframe|egui_dock)\b" crates/neoethos-app/src crates/neoethos-app/Cargo.toml
```

Expected: No live references. Comments mentioning historical migration are allowed only if they do not imply active support.

- [ ] **Step 5: Compile**

Run:

```powershell
cargo check -p neoethos-app --no-default-features
```

Expected: PASS.

---

## Chunk 5: Server and Flutter Contract Verification

### Task 7: Verify Endpoint Parity

**Files:**
- Read: `experiments/forex-flutter-ui/lib/api/backend_client.dart`
- Read: `crates/neoethos-app/src/server/mod.rs`

- [ ] **Step 1: Extract Flutter paths**

Run:

```powershell
rg -n "_dio\.(get|post)" experiments/forex-flutter-ui/lib/api/backend_client.dart
```

- [ ] **Step 2: Extract Rust routes**

Run:

```powershell
rg -n "route\(" crates/neoethos-app/src/server/mod.rs
```

- [ ] **Step 3: Compare by hand**

Expected route coverage:

```text
/healthz
/account/snapshot
/hardware
/risk
/settings
/engines/status
/engines/discovery/start
/engines/discovery/stop
/engines/training/start
/engines/training/stop
/broker/status
/broker/reauth
/broker/credentials
/broker/symbols
/broker/timeframes
/broker/accounts
/data/bootstrap
/data/fetch
/orders
/orders/cancel
/positions/close
/gemma/status
/gemma/chat
/gemma/news
/intelligence
/chart
```

- [ ] **Step 4: Add route tests for any missing endpoint**

Write tests in the relevant `crates/neoethos-app/src/server/*.rs` module.

Run:

```powershell
cargo test -p neoethos-app server
```

Expected: PASS.

### Task 8: Verify Flutter Still Starts Backend

**Files:**
- Read: `experiments/forex-flutter-ui/lib/startup/backend_supervisor.dart`
- Read: `experiments/forex-flutter-ui/lib/main.dart`

- [ ] **Step 1: Confirm supervisor still runs backend with `--server`**

Inspect `BackendSupervisor.ensureRunning()`.

Expected: It still locates `neoethos-app` and starts it with `--server`.

- [ ] **Step 2: Do not rename backend binary**

Keep `neoethos-app.exe` internal unless a separate binary/crate rename plan is approved.

- [ ] **Step 3: Run Flutter tests**

Run:

```powershell
Set-Location experiments/forex-flutter-ui
flutter analyze
flutter test
```

Expected: PASS.

---

## Chunk 6: User-Facing Branding Cleanup

### Task 9: Standardize Visible Product Name

**Files:**
- Modify: `experiments/forex-flutter-ui/lib/**`
- Modify: `experiments/forex-flutter-ui/pubspec.yaml`
- Modify: `README.md`
- Modify: `packaging/**`
- Modify as needed: `assets/**`

- [ ] **Step 1: Search visible product strings**

Run:

```powershell
rg -n "NeoEthos App|neoethos app|neoethos-app|NeoEthos -|desktop GUI|app.exe|AppDir|product_name|identifier|description" experiments README.md docs packaging crates/neoethos-app/Cargo.toml
```

- [ ] **Step 2: Classify strings**

Classify each match as:

```text
user-facing
internal-binary
package-technical
historical-doc
test-fixture
```

- [ ] **Step 3: Update user-facing strings**

Rules:

```text
Use: neoethos
Avoid: neoethos app
Avoid: NeoEthos App
Avoid visible "app" suffixes in product labels
```

Keep technical identifiers when needed:

```text
crate name: neoethos-app
backend binary: neoethos-app.exe
package paths using neoethos-app
```

- [ ] **Step 4: Verify no unwanted visible labels remain**

Run:

```powershell
rg -n "NeoEthos App|neoethos app|NeoEthos desktop GUI|legacy egui GUI" experiments README.md docs packaging crates/neoethos-app/Cargo.toml
```

Expected: No user-facing occurrences. Internal technical occurrences are documented.

---

## Chunk 7: Final Verification and Commit Strategy

### Task 10: Full Verification

**Files:**
- All modified files

- [ ] **Step 1: Rust check**

Run:

```powershell
cargo check -p neoethos-app --no-default-features
```

Expected: PASS.

- [ ] **Step 2: Rust tests**

Run:

```powershell
cargo test -p neoethos-app
cargo test -p neoethos-core
```

Expected: PASS, or unrelated pre-existing failures documented.

- [ ] **Step 3: Dependency proof**

Run:

```powershell
rg -n "\b(egui|eframe|egui_dock)\b" crates/neoethos-app/src crates/neoethos-app/Cargo.toml
cargo tree -p neoethos-app -i egui
```

Expected: No live egui references and no egui dependency path.

- [ ] **Step 4: Flutter verification**

Run:

```powershell
Set-Location experiments/forex-flutter-ui
flutter analyze
flutter test
```

Expected: PASS.

- [ ] **Step 5: Optional smoke run**

Run backend:

```powershell
cargo run -p neoethos-app -- --server
```

In another shell:

```powershell
Invoke-WebRequest http://127.0.0.1:7423/healthz
```

Expected: HTTP 200 with a JSON body containing version information.

### Task 11: Commit in Small Units

Recommended commits:

```text
test: capture backend behavior before egui removal
refactor: remove legacy egui runtime entry point
refactor: remove egui dependencies and ui tree
chore: standardize user-facing neoethos branding
test: verify flutter backend contract after egui removal
```

Do not mix unrelated dirty worktree changes into these commits.

## Implementation Notes

- Use `ctx7` for any new library/API questions before implementation.
- Do not use web search for Rust/Flutter API syntax unless `ctx7` is unavailable or quota-blocked.
- Prefer server-side tests for behavior and Flutter widget/client tests for rendering/parsing.
- Do not optimize or redesign Flutter screens during egui removal.
- Do not rename the crate or binary unless a separate rename plan is approved.
