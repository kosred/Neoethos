# Codex Full-Audit Continuation Gap Report

**Date:** 2026-05-16
**Correction:** 2026-05-17 — the project is still in development; no cTrader
live connection has ever been made and no strategy-search/model artifacts are
expected to be ready yet.
**Branch:** `codex/full-audit-continuation`
**Base:** `origin/claude/v0.4.1-full-audit @ 31a218ff`
**Method:** audit claims first, then code reality checks. Subsequent rows record verified fixes made in this continuation branch.

## Command Evidence

```powershell
rg -n "Status:|Gate:|FIXED|COMPLETE|DEFERRED|in flight|scaffold|TODO|FIXME|ship gate|release gate|not yet|remaining|ignored|unimplemented" docs/v0.5_roadmap.md docs/audits -g "*.md"
git log --oneline origin/master..HEAD
rg -n "TODO\(real-data\)" crates -g "*.rs"
rg -n "FIXME\(hardcoded\)" crates -g "*.rs"
rg -n "#\[ignore" crates -g "*.rs"
rg -n "unimplemented!" crates -g "*.rs"
git diff --check origin/claude/v0.4.1-full-audit...HEAD
```

Observed counts:

- `TODO(real-data)`: 23 matches in Rust files.
- `FIXME(hardcoded)`: 8 matches remain in Rust files after shared challenge/runtime/recovery/phase defaults were extracted in this branch (37 at initial audit).
- `#[ignore]`: 23 matches/comments in Rust files.
- `unimplemented!`: 5 matches, all currently under ignored/fixture-gated tests.
- `git diff --check origin/claude/v0.4.1-full-audit...HEAD`: green after the inherited whitespace cleanup in this branch.

## Claim-To-Code Matrix

| Area | Claim | Evidence | Status | Next action |
| --- | --- | --- | --- | --- |
| GUI first-run wizard | v0.4.5 gate requires wizard launches at first run and completes Step 1. `main.rs` comments say actual modal is rendered by `ui::wizard::wizard_ui`. | `crates/forex-app/src/main.rs` now carries `wizard_due` into `ForexApp`, creates a `WizardController` at `Welcome`, and renders `ui::wizard::wizard_ui` while active. Focused tests cover due/not-due controller creation. | **Fixed in this branch** | Follow-up is manual GUI smoke testing of `--wizard` and first-run sentinel behavior. |
| CLI/TUI wizard parity | Spec says `forex-cli wizard` is the TUI counterpart and shares the wizard state machine. | `crates/forex-cli/src/tui/wizard.rs` explicitly says "TUI rendering not yet ported". This branch keeps it as a recognized subcommand but makes the placeholder fail explicitly instead of returning success on TTY. The state machine still lives inside `forex-app`, creating a crate boundary problem. | **Safe placeholder; full parity still missing** | Do not implement full TUI yet. Shared wizard state likely belongs in a shared crate before ratatui pages are real. |
| HALT button and status pill | Older roadmap text says HALT/status pill were in flight and missing, but v0.4.5 gate requires working HALT smoke test. | Later code exists: `main.rs` calls `ui::chrome::halt_button::draw_halt_button` and `status_pill::draw_status_pill`; `TradingSession::trip_manual_halt` blocks new orders and propagates to Risky Mode. Focused tests pass: `cargo test -p forex-app halt --locked` (7/7) and `cargo test -p forex-app status_pill --locked` (7/7). | **Verified implemented; docs stale** | No code change. Optional manual GUI smoke test for button placement and pill visibility. |
| Risky Mode | Roadmap says v0.4.5 ships scaffolding/types only; v0.5 needs backtest harness gates. | `crates/forex-core/src/domain/risky_mode.rs` implements config, manager, kill switches, and unit tests; no release-gating backtest harness found. | **Partial as documented** | Keep as known v0.5 gap. Do not fake a backtest harness without a scoped design. |
| Wizard apply/hash migration | Commit claims migration hash changed from fnv1a64 to `sha2::Sha256`. | `wizard/autonomy_risk.rs` and `wizard/migration.rs` use real `Sha256` with tests. `forex-core` still uses FNV for temporal contract hashes, which is separate existing artifact hash logic. | **Implemented for wizard migration; no duplicate fix yet** | No action unless an audit claim requires all artifact hashes to switch away from FNV. |
| Hardcoded risk/config values | Roadmap says 37 `FIXME(hardcoded)` remain and mostly need config extraction. | Initial matches included duplicate challenge windows, relaxed validation min-days, runtime risk bands, broker lot ceiling, trade caps, recovery bands, and strategy tunables. This branch adds `PropFirmChallengeDefaults`, `PropFirmRuntimeDefaults`, and `PropFirmPhaseRiskDefaults` in `forex-core::domain::prop_firm`, then wires core risk presets plus search challenge/validation defaults to that shared boundary. Recovery drawdown bands/multipliers now share the same runtime defaults in both the order gate and position sizing. 8 `FIXME(hardcoded)` remain: `RiskConfig::default()` account/default strategy tunables and the prop-firm validation consistency ratio. | **Partially fixed in this branch** | The remaining items need real config-shape decisions; do not keep extracting constants just to reduce tags. |
| Real-data fixtures | v0.5 gate requires real cTrader fixtures and unignored tests. | The workspace already has real historical market data under `C:\Users\konst\development\forex-ai\data` (`147` Parquet files and `147` Vortex files, including `symbol=EURUSD/timeframe=M5`). Those files can be used for historical-data/model/search/data-conversion test work. They are not cTrader live broker WebSocket envelopes: no `payloadType: 2188` or `positionUnrealizedPnL` payloads were found under `data/`. Per the 2026-05-17 product-stage correction, missing cTrader live PnL/account/execution fixtures are **future integration gates**, not current blockers, because the app has never connected to cTrader yet. The ignored DQN full-training test remains a future model-training gate because strategy search/model artifacts are not expected to exist yet. | **Split: historical data available; live broker/model gates future** | Use `data/` for local historical fixture work where tests can consume Parquet/Vortex. Do not synthesize broker envelopes or pretend live cTrader/account/PnL captures exist before the first real connection. |
| F-MODELS9-013 swarm horizon | v0.4.5 gate requires previously ignored horizon test green. | The test `load_rejects_or_downgrades_artifact_with_incompatible_horizon` is no longer ignored and passes with `cargo test -p forex-models load_rejects_or_downgrades_artifact_with_incompatible_horizon --locked` (1/1). | **Verified implemented** | No code change. Keep as closed unless broader model tests expose a separate failure. |
| moneyDigits critical fix | v0.4.5 gate requires tests for moneyDigits=2 and 4 across entity money fields. | This branch adds focused moneyDigits=4 coverage for currently parsed account/execution position and deal money fields: swap, commission, mirroringCommission, usedMargin, grossProfit, pnlConversionFee, fees, and net profit. Shared helpers now resolve missing `moneyDigits` consistently and scale unsigned cTrader money fields. Deposit/bonus history entities are not parsed by the current local account/execution parsers. | **Fixed for current parsers; external entity gap remains** | Treat deposits/bonuses as a separate schema/fixture expansion gap, not as duplicate local scaling logic. |
| cTrader TLS provider and pre-close drill-down | The broad `cargo test -p forex-app ctrader_execution --locked` filter exposed a runtime panic in `close_selected_position_surfaces_ctrader_execution_failure`. | Backtrace showed `close_selected_position` hitting production `fetch_orders_by_position_id` before the stubbed execution backend, and rustls 0.23 panicking because both `ring` and `aws-lc-rs` providers were active. This branch installs a process rustls provider before cTrader TLS clients are built and injects the pre-close order-history backend for tests. | **Fixed in this branch** | Keep future cTrader unit tests behind injectable transports/backends; live WSS paths belong in explicit integration/real-data tests. |
| Installer scaffold | v0.4.5 gate requires at least one `.deb`, `.AppImage`, or `.tar.gz` on local CI plus manifest linting. | Workflow and packaging files exist. AppImage build intentionally fails if icon PNG is missing; only `forex-app.png.TODO` exists. Winget/Chocolatey/Scoop/Homebrew contain release-time TODO hashes/URLs. | **Scaffold, not ship-ready** | Keep as packaging gap. Do not fabricate icon or release hashes. Verify workflow metadata later only if packaging becomes selected scope. |
| Diff hygiene | Branch should be mergeable without whitespace noise. | Mechanical whitespace issues from the inherited branch were fixed in `9769128a`; `git diff --check origin/master...HEAD` was green after that cleanup. | **Verified fixed** | Re-run diff check before final handoff or PR. |

## Initial Priority

1. Mechanical `git diff --check` failures are fixed; re-run before final handoff.
2. GUI wizard launch wiring is fixed in this branch. Next, smoke-test `--wizard` manually when a GUI session is available.
3. HALT/status pill and F-MODELS9-013 focused tests are green; treat those roadmap items as stale status, not code gaps.
4. Historical data exists under `data/`; use it for locally testable model/search/data-conversion gaps instead of waiting for separate `tests/fixtures` copies.
5. cTrader TLS provider ambiguity and the pre-close unit-test live-transport leak are fixed; keep future live broker checks in explicit integration tests because no cTrader connection has happened yet.
6. moneyDigits coverage is fixed for currently parsed account/execution entities. Deposits/bonuses remain future cTrader schema/fixture work, not current development blockers.
7. Shared challenge/runtime/recovery/phase defaults now remove the locally provable duplication; the remaining 8 hardcoded tunables need config-shape decisions before code changes.

## Non-Actions

- Do not replace real-data fixture gaps with synthetic broker data.
- Do not treat missing live cTrader broker captures as a current blocker before the first real cTrader connection exists.
- Do not treat missing strategy-search/model artifacts as current blockers while the project is still pre-search/pre-training.
- Do not implement full TUI wizard until the shared state-machine crate boundary is decided.
- Do not rewrite Risky Mode backtesting without a scoped design and acceptance data.
- Do not populate release-time package hashes, URLs, or icons without real release artifacts.
