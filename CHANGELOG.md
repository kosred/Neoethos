# Changelog

All notable changes to forex-ai are documented here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to semantic versioning.

## [0.4.7] — 2026-05-18 — "Cleanup + Boot-Wire Release"

> Shipping early to surface integration-level bugs that the unit
> tests do not catch — particularly first-run wizard end-to-end,
> Risky Mode boot-time arming, and DXtrade live-session behaviour.

### Added

- **Risky Mode boot-time wire-up.** The wizard's `risky_mode_armed`
  flag is now persisted to `<config_dir>/forex-ai/risky_mode_state.json`
  by `summary.rs::write_risky_mode_state`. At app boot,
  `TradingSession::new_with_persisted_credentials` calls a new
  `auto_arm_risky_mode_from_persisted_state` helper that loads the
  file and calls `enable_risky_mode(RiskyModeConfig::default(),
  starting_bankroll)` when armed. Schema-versioned via the existing
  `HasSchemaVersion` Phase-D4 contract; safe-fallback to disabled
  on every error path (no half-armed sessions).
- New `crates/forex-app/src/app_services/risky_mode_persistence.rs`
  module with 5 unit tests (round-trip, missing-file → None,
  pre-versioning serde compat, malformed-JSON error path,
  future-schema-version fallback).

### Refactored — god-file splits prepared as drafts

A code-health round carved the six largest god files into focused
sibling modules. Each split lives in a `*_split_draft/` directory
next to the active source; the operator activates each one with a
single `Move-Item` after running `cargo check`. Activation docs
in `docs/qa/2026-05-18-*-split-draft.md`.

| File | Pre | Post (max file) | Reduction |
|---|---|---|---|
| `dxtrade.rs` | 2787 | 1369 | 51% |
| `burn_models.rs` | 2634 | 965 | 63% |
| `training_orchestrator.rs` | 4137 | 1946 | 53% |
| `dqn_impl.rs` | 2659 | 1941 | 27% |
| `swarm_impl.rs` | 3397 | 2749 | 19% |
| `deep_models.rs` | 2263 | 1770 | 22% |

### Fixed

- Stale `FIXME(risky-mode-apply)` and `FIXME(wizard-sha256)` comments
  in the wizard now reflect the landed wiring + the existing `sha2`
  workspace dep; references to obsolete "Phase 2B / 2C / 2D" /
  "Agent A / B" scaffolding labels removed from `account_profile.rs`,
  `autonomy_risk.rs`, `summary.rs`, and `migration.rs`.
- Phase C3 dead-code allow-list re-audited: all seven file-level
  `#![allow(dead_code)]` annotations carry current 2026-05-18
  operator-directive justifications (Flutter API consumers pending,
  real-data fixtures pending, spec-complete proto wire format).

### Changed

- Rust workspace crate versions aligned to `0.4.7` so app binaries
  and generated package metadata match the release tag.
- Packaging manifests (chocolatey, scoop, homebrew, portable build
  script) bumped to `0.4.7`. WinGet manifest directory rename
  (`packaging/winget/manifests/k/kosred/forex-ai/0.4.6/`) is the
  one packaging step that has to happen manually on the Windows
  side — the WinGet schema embeds the version in the directory
  path.

### Known issues — to surface via 0.4.7 installation testing

- **Wizard Steps 2-10 + 9.5 end-to-end (task #15)** — individual
  step renderers + the apply writer landed in 0.4.5; the full
  end-to-end Live-mode walk-through is best validated in real use.
- **Full forex-app GUI computer-use smoke test (task #49)** —
  blocked while the operator was away from the machine during the
  prior session; ready to run post-install.
- **God-file splits (six drafts)** — not yet activated; each
  activation needs ~5 min with live `cargo check` per file. The
  active source files remain unchanged so the 0.4.7 binary builds
  as-is from the pre-split layout.

## [0.4.6] — 2026-05-17 — internal bump (no public release)

- Internal version-counter bump after the 0.4.5 audit-fix release.
  No publicly-published packaging artifacts. Folded into 0.4.7 for
  the next public ship.

## [0.4.5] — 2026-05-17 — "Audit Fix Release"

### Added

- First-run wizard scaffold for the v0.5 onboarding surface, including
  Welcome/License, data path, account profile, migration, CLI wizard
  entrypoint, and resumable wizard state.
- v0.4.5 packaging manifests for WinGet, Chocolatey, Scoop, Homebrew,
  AppImage, and the release installer workflows.

### Fixed

- cTrader money scaling now propagates per-entity `moneyDigits` for
  account, margin, commission, deposit, bonus, and mirrored commission
  values instead of relying on unsafe defaults.
- Tree-model local fallback loading rejects or downgrades incompatible
  swarm-horizon artifacts.
- Manual HALT flow now blocks new orders, writes the HALT sentinel, and
  exposes clear/resume behavior through the app chrome.
- Wizard portable migration records skipped cache payloads instead of
  silently dropping skipped-file accounting.
- WinGet `0.4.5` manifest validates cleanly with a single default-locale
  manifest and a concrete release artifact SHA-256.

### Changed

- Rust workspace crate versions are aligned to `0.4.5` so app binaries
  and generated package metadata match the release tag.
- Audit documentation now marks live cTrader connection, strategy search,
  and ready model workflows as future integration work while the project
  is still pre-integration development.

## [0.2.0] — 2026-05-12 — "Smart Discovery + Production Audit"

### Added

- **Smart prop-firm discovery is now the default** ([be64c5cb], [33275fad])
  - `cargo run -p forex-cli -- discover` runs in PropFirm mode out of
    the box: permissive filter floors, FTMO-rule scoring on N random
    60-day windows from history, ranking-based portfolio selection
    (no thresholds to tune), window count auto-derived from dataset
    length. Single opt-out via `FOREX_BOT_DISCOVERY_MODE=strict`.
  - New env knobs (all optional, sane defaults):
    `FOREX_BOT_DISCOVERY_PROP_FIRM_PASS_RATE`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_N_WINDOWS`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_WINDOW_DAYS`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_MAX_DAILY_LOSS_PCT`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_MAX_DD_PCT`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_PROFIT_TARGET_PCT`,
    `FOREX_BOT_DISCOVERY_PROP_FIRM_MIN_TRADING_DAYS`.
- **`FOREX_BOT_DISCOVERY_PERMISSIVE`** ([037ce2a7]) override that
  bypasses the source-level filter floors that previously prevented
  any candidate from surviving.
- **GPU pipeline** ([8c041fe0]) — full `cubecl 0.9` migration with
  `RuntimeCell`-based mutable scalars, libtorch 2.9.0+cu130 link, NVRTC
  CUDA 13.0 support. Verified end-to-end on Hyperstack L40 / driver 595.
- **UI overhaul** ([e1044609], [9b8bfe64]) — design system (warmer
  dark palette, 4-pt spacing grid, 4-level type scale, named
  `ButtonKind` variants, `nav_item` helper); slim 56 px top bar;
  polished sidebar with active-row accent stripe; quieter dock tab
  strip (no more `▼` leaf-collapse buttons).
- **Recalibrated `is_anomalous` filter** ([a0531c48]) — profit gates
  scaled 50× to match a 4-10%/mo target window over a 10y backtest.

### Changed

- **Codex Phases 76-90** ([efbd9b35]) merged. Test-extraction pattern
  (Phase 90) lifted ~3,000 LOC of `#[cfg(test)] mod tests {}` blocks
  out of `trading.rs` and `ensemble.rs` into sibling `_tests.rs`
  files. Same pattern then applied to **9 more god files** in
  [f01bb4aa] (~6,800 LOC moved out): dqn_impl, swarm_impl, exit_agent,
  forex-search/discovery, forex-app/discovery, ctrader_messages,
  ctrader_live_auth, ctrader_execution, ctrader_account.

### Fixed — production bugs caught in audit

- **`broker_persistence.rs` — tests were silently writing to your
  real broker_credentials.toml** ([cbf96976]). When
  `FOREX_AI_BROKER_CREDENTIALS_PATH` pointed at a not-yet-existing
  temp path, `credentials_file_path()` fell through to the user's
  `~/AppData/Roaming/forex-ai/broker_credentials.toml`. Fixed by
  making the env override authoritative (no fallback chain when set).
- **`broker_persistence.rs` — `Mutex` poison cascading** ([cbf96976]).
  When any test panicked while holding `ENV_LOCK`, every subsequent
  env-touching test panicked too. Now uses
  `lock().unwrap_or_else(|p| p.into_inner())` plus an RAII
  `EnvOverrideGuard` that always clears the env on drop.
- **`ctrader_account.rs` + `ctrader_execution.rs` — `money_digits`
  silent fallback** ([70702c0a]). cTrader OpenAPI declares
  `money_digits` as required, but Rust used `Option<u32>` and
  `.unwrap_or(0)` would have made `10_f64.powi(0) = 1.0`, scaling
  every reported balance / equity / commission / P&L **100×**. Now
  emits `tracing::error!` and defaults to `2` (de-facto fiat
  precision) instead of `0`.
- **`forex-models/src/base.rs` — NaN panic in distribution fitting**
  ([a71b6471]). `breakpoints.sort_by(|a,b| a.partial_cmp(b).unwrap())`
  panicked on the first NaN sample. Now sorts NaN to the end and
  drops non-finite values before dedup.
- **`forex-search/src/genetic/evolution_math.rs` — silent flush
  failure** ([a71b6471]). `pending` was cleared after a successful
  `write_all` but before checking `flush()`, dropping unsynced data.
  Now requires both to succeed.
- **`forex-search/src/cubecl_eval.rs` — silent CUDA-device-0
  fallback** ([a71b6471]). Setting
  `FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE` with a typo (`auto`, `all`,
  `GPU0`) would silently run on device 0 instead of the intended one.
  Now emits `tracing::warn!` first.

### Refactored

- **Exponential backoff dedup** ([70702c0a]). `ctrader_backoff_sleep`
  in `ctrader_execution.rs` and `streaming_backoff_sleep` in
  `ctrader_streaming.rs` were byte-for-byte identical. Extracted to
  a single `crates/forex-app/src/app_services/backoff.rs` with proper
  `saturating_*` arithmetic to prevent factor-shift overflow at high
  attempt counts.
- **Branch hygiene**: merged + deleted `claude/happy-gould-23d649`,
  `claude/magical-noyce-5f21ba`, `codex/phases-30-40`,
  `codex/phases-72-75`. Removed 4 stale Claude-Code worktree
  directories. Master is now the single source of truth.

### Test status

- `cargo test --workspace` — **764/764** unit tests pass:
  forex-core 70, forex-data 13, forex-models 338,
  forex-search 114, forex-app 229, forex-cli 2.
  (forex-search needs `--test-threads=1` because of an env-var test
  race; the rest are parallel-clean.)
- `cargo clippy --workspace --all-targets --release` — **0 errors**.
  ~50 warnings remain (mostly `dead_code` from intentional unused
  helpers); none affect correctness.

### Deferred to 0.3 (see [docs/audits/post_release_tech_debt_2026-05-12.md])

- God-file splits for the 5 remaining 90-153 KB production files
  (training_orchestrator, trading, swarm_impl, discovery, dqn_impl).
- 7 medium-severity audit findings around `unwrap_or(false)` /
  `unwrap_or(0)` patterns in cTrader payload parsing.
- 14 dependabot security advisories (2 PRs already open on origin).

[0.4.5]: https://github.com/kosred/forex-ai/releases/tag/v0.4.5
[0.2.0]: https://github.com/kosred/forex-ai/releases/tag/v0.2.0
[a0531c48]: https://github.com/kosred/forex-ai/commit/a0531c48
[037ce2a7]: https://github.com/kosred/forex-ai/commit/037ce2a7
[33275fad]: https://github.com/kosred/forex-ai/commit/33275fad
[be64c5cb]: https://github.com/kosred/forex-ai/commit/be64c5cb
[8c041fe0]: https://github.com/kosred/forex-ai/commit/8c041fe0
[e1044609]: https://github.com/kosred/forex-ai/commit/e1044609
[9b8bfe64]: https://github.com/kosred/forex-ai/commit/9b8bfe64
[efbd9b35]: https://github.com/kosred/forex-ai/commit/efbd9b35
[cbf96976]: https://github.com/kosred/forex-ai/commit/cbf96976
[f01bb4aa]: https://github.com/kosred/forex-ai/commit/f01bb4aa
[a71b6471]: https://github.com/kosred/forex-ai/commit/a71b6471
[70702c0a]: https://github.com/kosred/forex-ai/commit/70702c0a
