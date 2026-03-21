# Canonical Sectioned Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one canonical sectioned log file at `logs/forex-ai.log` that keeps `current` and `previous` runs per subsystem without creating timestamped log sprawl.

**Architecture:** Add a dedicated sectioned-log module in `forex-core`, keep tracing subscriber setup in the existing logging module, and wire the first runtime adopters (`forex-app`, `forex-cli`, `mt5-bridge`) through the shared contract. Use atomic rewrite semantics so only the targeted subsystem section changes while the rest of the canonical file remains intact.

**Tech Stack:** Rust, `tracing`, `tracing-appender`, standard library file IO/rename, workspace unit tests, CLI/app smoke verification

---

## File Structure

Planned ownership:

- Create: `crates/forex-core/src/sectioned_log.rs`
  - section model
  - record model
  - parser/renderer
  - atomic section update
  - file lock and malformed-file recovery

- Modify: `crates/forex-core/src/lib.rs`
  - export `sectioned_log`

- Modify: `crates/forex-core/src/logging.rs`
  - initialize canonical log path
  - keep retained `WorkerGuard`
  - expose helper entrypoints for subsystem updates

- Modify: `crates/forex-app/src/main.rs`
  - replace local tracing init with shared logging setup
  - write `APP` section updates for GUI/headless startup and MT5/local mode outcomes

- Modify: `crates/forex-cli/src/main.rs`
  - write `CLI` section updates for command lifecycle
  - mirror discovery/training outcomes into their subsystem sections

- Modify: `crates/mt5-bridge/src/lib.rs`
  - write `MT5` section updates for initialize success/failure

- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
  - record implementation status after verification

- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
  - add canonical-log verification lane if needed

## Chunk 1: Core Sectioned Log Model

### Task 1: Add a focused sectioned-log module skeleton

**Files:**
- Create: `crates/forex-core/src/sectioned_log.rs`
- Modify: `crates/forex-core/src/lib.rs`

- [ ] **Step 1: Write failing unit tests for section lifecycle behavior**

Add tests for:
- create canonical file from empty state
- update one section while preserving another
- rotate `current -> previous`
- keep only two retained runs

- [ ] **Step 2: Run the focused test target to verify red**

Run: `cargo test -p forex-core sectioned_log -- --nocapture`
Expected: FAIL with missing module/types/functions

- [ ] **Step 3: Implement minimal section and record types**

Add:
- `SubsystemSection`
- `SectionedRunRecord`
- `CanonicalSectionedLog`

Keep the type surface small and explicit.

- [ ] **Step 4: Implement render/parse for the canonical file format**

Implement deterministic:
- section ordering
- `CURRENT`
- `PREVIOUS`

- [ ] **Step 5: Re-run the focused test target**

Run: `cargo test -p forex-core sectioned_log -- --nocapture`
Expected: PASS for the initial lifecycle tests

- [ ] **Step 6: Commit**

```bash
git add crates/forex-core/src/sectioned_log.rs crates/forex-core/src/lib.rs
git commit -m "Add canonical sectioned log core model"
```

## Chunk 2: Atomic Update And Recovery

### Task 2: Implement atomic per-section replacement

**Files:**
- Modify: `crates/forex-core/src/sectioned_log.rs`

- [ ] **Step 1: Write failing tests for atomic section replacement and malformed-file recovery**

Add tests for:
- section rewrite preserves untouched sections
- malformed file rebuilds into valid canonical structure
- recovery records a `SYSTEM` event

- [ ] **Step 2: Run the focused test target to verify red**

Run: `cargo test -p forex-core sectioned_log -- --nocapture`
Expected: FAIL on the new atomic/recovery cases

- [ ] **Step 3: Implement file lock, temporary write, and atomic replace**

Implement one write path that:
- reads current canonical file
- updates only the requested section
- writes temporary file
- renames atomically into place

- [ ] **Step 4: Implement malformed-file recovery behavior**

On parse failure:
- rebuild canonical structure
- write `SYSTEM.current` recovery record
- preserve explicit error information

- [ ] **Step 5: Re-run the focused test target**

Run: `cargo test -p forex-core sectioned_log -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/forex-core/src/sectioned_log.rs
git commit -m "Implement atomic sectioned log updates"
```

## Chunk 3: Shared Logging Integration

### Task 3: Bridge tracing setup to the canonical log contract

**Files:**
- Modify: `crates/forex-core/src/logging.rs`

- [ ] **Step 1: Write failing tests for canonical log writer integration**

Add tests for:
- shared logging setup can initialize canonical path
- section update helper writes to the expected section
- `WorkerGuard` retention still holds

- [ ] **Step 2: Run focused tests to verify red**

Run: `cargo test -p forex-core logging::tests -- --nocapture`
Expected: FAIL for the new canonical section update behavior

- [ ] **Step 3: Implement shared helpers in `logging.rs`**

Add helpers such as:
- canonical log path resolution
- subsystem log update entrypoint
- startup/system event writer

Do not reintroduce per-crate custom logging setup.

- [ ] **Step 4: Re-run the focused tests**

Run: `cargo test -p forex-core logging::tests -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run crate-level lint/test verification**

Run:
- `cargo test -p forex-core -- --nocapture`
- `cargo clippy -p forex-core --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/forex-core/src/logging.rs crates/forex-core/src/sectioned_log.rs crates/forex-core/src/lib.rs
git commit -m "Wire shared logging to canonical sectioned log"
```

## Chunk 4: First Runtime Adopters

### Task 4: Move `forex-app` onto shared logging and section updates

**Files:**
- Modify: `crates/forex-app/src/main.rs`

- [ ] **Step 1: Write failing or missing-behavior tests where feasible**

If direct GUI tests are not practical, add small unit-tested helper functions around:
- local/headless startup event classification
- subsystem log message construction

- [ ] **Step 2: Run the app crate tests**

Run: `cargo test -p forex-app -- --nocapture`
Expected: red if helper tests were added first, otherwise note no direct tests and proceed with runtime smoke verification

- [ ] **Step 3: Replace local tracing init with shared core logging**

Use the shared logging entrypoint instead of `tracing_subscriber::fmt::init()`.

- [ ] **Step 4: Emit `APP` section updates**

Record:
- GUI startup
- headless startup
- local mode selection
- MT5 mode failure summaries

- [ ] **Step 5: Verify the headless runtime path**

Run: `cargo run -p forex-app -- --headless --config config.yaml`
Expected: PASS WITH FINDINGS if local MT5 authorization still fails, but the canonical file must contain updated `APP` and `MT5` sections

- [ ] **Step 6: Commit**

```bash
git add crates/forex-app/src/main.rs
git commit -m "Adopt canonical sectioned logging in forex-app"
```

### Task 5: Move `forex-cli` onto shared logging and subsystem section updates

**Files:**
- Modify: `crates/forex-cli/src/main.rs`

- [ ] **Step 1: Write failing tests for any extracted command-summary helpers**

Prefer small helper tests if command logic needs formatting/section routing.

- [ ] **Step 2: Run the relevant crate tests**

Run: `cargo test -p forex-cli -- --nocapture`
Expected: red for new helper tests if added first

- [ ] **Step 3: Implement CLI section updates**

Record:
- command start
- command success/failure
- mirror discovery results into `DISCOVERY`
- mirror training results into `TRAINING`

- [ ] **Step 4: Verify real command paths**

Run:
- `cargo run -p forex-cli -- load --symbol EURUSD --timeframe M1 --root data`
- `cargo run -p forex-cli -- discover --root data --symbol EURUSD --base M1 --higher M5,M15,H1 --population 10 --generations 1 --candidates 20 --portfolio-size 10`

Expected:
- command success/failure remains explicit
- canonical file updates only the `CLI` and relevant subsystem sections

- [ ] **Step 5: Commit**

```bash
git add crates/forex-cli/src/main.rs
git commit -m "Adopt canonical sectioned logging in forex-cli"
```

### Task 6: Add MT5 section updates from the bridge

**Files:**
- Modify: `crates/mt5-bridge/src/lib.rs`

- [ ] **Step 1: Write failing tests for MT5 section update behavior**

Extend the existing bridge tests to verify that tuple-aware error formatting can be turned into a stable `MT5` section record.

- [ ] **Step 2: Run crate tests to verify red**

Run: `cargo test -p mt5-bridge -- --nocapture`
Expected: FAIL for the new logging behavior test

- [ ] **Step 3: Implement `MT5` section writes**

Record:
- initialize started
- initialize success
- initialize failed with tuple-aware error

- [ ] **Step 4: Re-run crate tests**

Run: `cargo test -p mt5-bridge -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/mt5-bridge/src/lib.rs
git commit -m "Log MT5 bridge events to canonical sectioned log"
```

## Chunk 5: Full Verification And Audit Update

### Task 7: Verify the full supported lane and update the audit artifacts

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

- [ ] **Step 1: Run the full workspace verification**

Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Expected: PASS

- [ ] **Step 2: Inspect the canonical log**

Check:
- only one runtime log file exists at `logs/forex-ai.log`
- sections exist
- `current/previous` retention is correct
- unrelated sections survive targeted updates

- [ ] **Step 3: Update audit artifacts**

Record:
- that the canonical sectioned log contract is implemented
- what runtime paths are first adopters
- what remains for later adoption

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md logs/forex-ai.log
git commit -m "Document canonical sectioned logging rollout"
```

- [ ] **Step 5: Final handoff verification**

Run:
- `git status --short`
- `git log -1 --stat`

Expected: clean worktree and clear final commit summary
