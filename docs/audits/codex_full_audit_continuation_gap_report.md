# Codex Full-Audit Continuation Gap Report

**Date:** 2026-05-16
**Branch:** `codex/full-audit-continuation`
**Base:** `origin/claude/v0.4.1-full-audit @ 31a218ff`
**Method:** audit claims first, then code reality checks. No production code changes in this report.

## Command Evidence

```powershell
rg -n "Status:|Gate:|FIXED|COMPLETE|DEFERRED|in flight|scaffold|TODO|FIXME|ship gate|release gate|not yet|remaining|ignored|unimplemented" docs/v0.5_roadmap.md docs/audits -g "*.md"
git log --oneline origin/master..HEAD
rg -n "TODO\(real-data\)" crates -g "*.rs"
rg -n "FIXME\(hardcoded\)" crates -g "*.rs"
rg -n "#\[ignore" crates -g "*.rs"
rg -n "unimplemented!" crates -g "*.rs"
git diff --check origin/master...HEAD
```

Observed counts:

- `TODO(real-data)`: 23 matches in Rust files.
- `FIXME(hardcoded)`: 37 matches in Rust files.
- `#[ignore]`: 23 matches/comments in Rust files.
- `unimplemented!`: 5 matches, all currently under ignored/fixture-gated tests.
- `git diff --check origin/master...HEAD`: fails on inherited whitespace/EOF issues.

## Claim-To-Code Matrix

| Area | Claim | Evidence | Status | Next action |
| --- | --- | --- | --- | --- |
| GUI first-run wizard | v0.4.5 gate requires wizard launches at first run and completes Step 1. `main.rs` comments say actual modal is rendered by `ui::wizard::wizard_ui`. | `crates/forex-app/src/main.rs:95` computes and logs `wizard_due`, but `ForexApp` has no wizard controller field, `ForexApp::new` does not receive `wizard_due`, and there is no call to `ui::wizard::wizard_ui`. `crates/forex-app/src/ui/wizard/mod.rs:432` defines the UI entry point. | **Missing wiring** | First code fix candidate. Add a failing test around `ForexApp`/wizard launch state if possible, then wire controller and render path. |
| CLI/TUI wizard parity | Spec says `forex-cli wizard` is the TUI counterpart and shares the wizard state machine. | `crates/forex-cli/src/tui/wizard.rs` explicitly says "TUI rendering not yet ported" and returns a placeholder on TTY. The state machine lives inside `forex-app`, creating a crate boundary problem. | **Partial / placeholder** | Do not implement full TUI yet. Record as larger design gap: shared wizard state likely belongs in a shared crate before ratatui pages are real. |
| HALT button and status pill | Older roadmap text says HALT/status pill were in flight and missing, but v0.4.5 gate requires working HALT smoke test. | Later code exists: `main.rs` calls `ui::chrome::halt_button::draw_halt_button` and `status_pill::draw_status_pill`; `TradingSession::trip_manual_halt` blocks new orders and propagates to Risky Mode. | **Implemented, docs stale, needs focused tests** | Run focused `forex-app` HALT tests after baseline build. If green, update report/status only; if failing, fix with TDD. |
| Risky Mode | Roadmap says v0.4.5 ships scaffolding/types only; v0.5 needs backtest harness gates. | `crates/forex-core/src/domain/risky_mode.rs` implements config, manager, kill switches, and unit tests; no release-gating backtest harness found. | **Partial as documented** | Keep as known v0.5 gap. Do not fake a backtest harness without a scoped design. |
| Wizard apply/hash migration | Commit claims migration hash changed from fnv1a64 to `sha2::Sha256`. | `wizard/autonomy_risk.rs` and `wizard/migration.rs` use real `Sha256` with tests. `forex-core` still uses FNV for temporal contract hashes, which is separate existing artifact hash logic. | **Implemented for wizard migration; no duplicate fix yet** | No action unless an audit claim requires all artifact hashes to switch away from FNV. |
| Hardcoded risk/config values | Roadmap says 37 `FIXME(hardcoded)` remain and mostly need config extraction. | Matches remain in `forex-core/src/config.rs`, `forex-core/src/domain/risk.rs`, `forex-search/src/challenge.rs`, and `forex-search/src/validation.rs`. Several represent duplicate challenge windows, risk bands, trade caps, and strategy tunables. | **Known duplication / config gap** | Second candidate after wizard wiring. Need a small shared config/defaults boundary, not broad refactor. |
| Real-data fixtures | v0.5 gate requires real cTrader fixtures and unignored tests. | `TODO(real-data)` and ignored tests remain in cTrader account/execution/history/live-auth/integration tests, PnL, model tests, search tests, and data conversion. `pnl.rs` has 3 ignored tests with `unimplemented!`. | **External-data gap** | Do not synthesize fake fixtures. Convert placeholders to explicit fixture-contract tests only where local fixture files exist or user supplies captures. |
| F-MODELS9-013 swarm horizon | v0.4.5 gate requires previously ignored horizon test green. | The test `load_rejects_or_downgrades_artifact_with_incompatible_horizon` is no longer ignored and code rejects horizon/history conflicts. | **Likely implemented** | Run focused `cargo test -p forex-models load_rejects_or_downgrades_artifact_with_incompatible_horizon --locked` when test baseline starts. |
| moneyDigits critical fix | v0.4.5 gate requires tests for moneyDigits=2 and 4 across entity money fields. | `ctrader_money.rs` has scale/unscale tests; account parsing uses per-entity `required_money_digits`. Account tests cover some moneyDigits=2 payloads; no clear moneyDigits=4 coverage for all listed entities yet. | **Partial coverage** | Add focused tests if grep confirms missing moneyDigits=4 cases for swap/commission/mirroringCommission/usedMargin/deposits/bonuses. |
| Installer scaffold | v0.4.5 gate requires at least one `.deb`, `.AppImage`, or `.tar.gz` on local CI plus manifest linting. | Workflow and packaging files exist. AppImage build intentionally fails if icon PNG is missing; only `forex-app.png.TODO` exists. Winget/Chocolatey/Scoop/Homebrew contain release-time TODO hashes/URLs. | **Scaffold, not ship-ready** | Keep as packaging gap. Do not fabricate icon or release hashes. Verify workflow metadata later only if packaging becomes selected scope. |
| Diff hygiene | Branch should be mergeable without whitespace noise. | `git diff --check origin/master...HEAD` reports trailing whitespace and blank EOF in proto, trading orders, forex-search challenge/gauntlet, and installer research doc. | **Mechanical failure** | Safe first mechanical fix after report commit; formatting-only, no TDD required. |

## Initial Priority

1. Fix mechanical `git diff --check` failures. This is low-risk and improves mergeability.
2. Prove and fix GUI wizard launch wiring. This directly blocks a v0.4.5 gate and is locally testable.
3. Run focused tests for HALT/status pill and F-MODELS9-013 to separate stale docs from real failures.
4. Investigate `moneyDigits` test coverage gaps and add missing tests before code changes.
5. Triage hardcoded risk/config duplication into a small shared boundary only if tests can lock behavior first.

## Non-Actions

- Do not replace real-data fixture gaps with synthetic broker data.
- Do not implement full TUI wizard until the shared state-machine crate boundary is decided.
- Do not rewrite Risky Mode backtesting without a scoped design and acceptance data.
- Do not populate release-time package hashes, URLs, or icons without real release artifacts.
