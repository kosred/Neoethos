# Changelog

All notable changes to forex-ai are documented here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to semantic versioning.

## [0.4.12] — 2026-05-19 — "Wizard OAuth redirect_uri Matches the cTrader App"

> Patch release after the v0.4.11 wizard walkthrough finally got to
> the cTrader consent page and clicked "Allow access" — only to be
> rejected by id.ctrader.com with the toast
> *"Application authentication failed. Provided application does not
> contain provided URI."* The wizard was advertising
> `http://127.0.0.1:7777/ctrader/callback` as the redirect URI, but
> the developer-registered Open API app on connect.spotware.com has
> only `http://127.0.0.1:43001/callback` in its allowed-redirect list.
> v0.4.12 makes the wizard match the registered URI so the OAuth
> exchange can complete.

### Fixed

- **`WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS`** now leads with `43001`
  (was `[7777, 7878, 8989]`). The legacy ports remain as fallbacks so
  a fork that re-registers them on a different OAuth app still works.
- **`WIZARD_DEFAULT_OAUTH_CALLBACK_PATH`** is now `/callback` (was
  `/ctrader/callback`). Matches the path the cTrader app dashboard has
  registered.
- `default_loopback_ports_match_rfc8252_three_port_fallback` test
  renamed to `default_loopback_ports_lead_with_registered_redirect_port`
  and asserts that index 0 is `43001`.

### Pre-ship gates

- `cargo fmt --all -- --check` — clean.
- `cargo build --release -p forex-app` — 0 errors (4m 03s).
- `cargo packager --release` — produced
  `forex-app_0.4.12_x64-setup.exe` (25.97 MB).

### Artifact

- `forex-app_0.4.12_x64-setup.exe` — 25.97 MB
  - SHA-256: `7062E4E8B9EDDA08B2CBBC10F410CD84E80E25B2D9EBB90304BB170676FAC193`

---

## [0.4.11] — 2026-05-19 — "cTrader Credentials Actually Embedded"

> Patch release after a Phase X1 wizard walkthrough on the v0.4.10
> binary caught a red banner at Step 4 (cTrader Sign-in):
> "Developer build: cTrader app credentials not embedded".
> Phase 0c (2026-05-17) had marked credential embedding as complete,
> but the workspace `.local/forex-ai/broker_credentials.toml` that
> `build.rs` reads at compile time still had empty strings — so the
> v0.4.10 release binary was shipping `EMBEDDED_CTRADER_CLIENT_ID = ""`
> and the OAuth flow could not start. The real values were only in
> `%APPDATA%\forex-ai\broker_credentials.toml` (runtime), which the
> build script does not consult.

### Fixed

- **`.local/forex-ai/broker_credentials.toml` populated with the real
  cTrader Open API app credentials** so `build.rs::emit_embedded_credentials()`
  bakes them into the `EMBEDDED_CTRADER_CLIENT_ID` /
  `EMBEDDED_CTRADER_CLIENT_SECRET` constants. The TOML is `.gitignore`-d
  (`.local/` + `**/broker_credentials.toml`) so the secrets do not leak
  into git history.
- **Wizard Step 4 banner clears on the v0.4.11 binary** —
  `embedded_credentials_present()` returns `true`, the "Sign in to
  your broker" button is wired to a real OAuth flow, and the
  developer-build diagnostic is suppressed.

### Pre-ship gates

- `cargo fmt --all -- --check` — clean.
- `cargo build --release -p forex-app` — 0 errors (3m 33s).
- `cargo packager --release` — produced
  `forex-app_0.4.11_x64-setup.exe` (25.96 MB).

### Artifact

- `forex-app_0.4.11_x64-setup.exe` — 25.96 MB
  - SHA-256: `456809E6AF1ADA460971244FCD33CCA0F1A375B3281030D7A0EBFEE1A256CEBF`

---

## [0.4.10] — 2026-05-19 — "Installer Payload Repair + Gemma Bundle Strategy"

> Patch release after a v0.4.9 binary-walkthrough audit caught the
> installer shipping only the .exe — none of the runtime DLLs, no
> LICENSE, no README, no Gemma-fetch helper. The root cause was the
> `[package.metadata.packager].resources` paths being bare filenames
> instead of crate-relative paths; cargo-packager silently skipped
> every entry that didn't resolve. v0.4.10 repairs the payload, ships
> the user-side AI Helper banner that detects whether the Gemma model
> file is present on disk, and bundles the PowerShell fetcher next to
> the binary so the operator can pull the 5 GB GGUF in one click after
> install instead of cloning the repo.

### Fixed

- **Installer was 20.93 MB and missing 34 MB of payload.** The v0.4.9
  installer carried only `forex-app.exe`. The `resources` array in
  `crates/forex-app/Cargo.toml` listed bare filenames
  (`catboostmodel.dll`, `xgboost.dll`, `config.yaml`) that
  cargo-packager resolves relative to the crate manifest dir — none
  resolved, every entry was silently skipped. v0.4.10 rewrites the
  paths with explicit `../../` prefixes (`../../config.yaml`,
  `../../target/release/catboostmodel.dll`, etc.). Silent-install
  verification confirms the new installer carries:
  `forex-app.exe` (108.9 MB), `catboostmodel.dll` (16.5 MB),
  `xgboost.dll` (18.4 MB), `config.yaml`, `LICENSE`, `README.md`,
  `fetch-gemma-model.ps1`, `uninstall.exe`. Compressed installer:
  **25.96 MB** (was 20.93 MB).

### Added

- **AI Helper panel — "Gemma model not found" banner.** When the
  helper boots and `resolve_or_suggest_model_path()` returns `None`,
  the chat panel now renders a warning frame with the approximate
  model size, the canonical filename, the HuggingFace direct-LFS URL
  and three buttons:
  1. **Copy download URL** — drops the URL into the clipboard.
  2. **Open save folder** — opens `<dirs::data_dir>/forex-ai/models/`
     in Explorer so the operator can drop the GGUF in by hand.
  3. **Run fetch-gemma-model.ps1** — spawns the bundled PowerShell
     helper that streams the GGUF from HuggingFace with a progress
     readout and a disk-space sanity check. The script is shipped
     next to `forex-app.exe` via the installer's `resources`.
- **Bundled `scripts/fetch-gemma-model.ps1`.** Reachable both from the
  in-app button and directly from the Start-menu install dir.
- **`forex-gemma` public constants** for the bundled-model anchors
  (`MODEL_PATH_ENV_VAR`, `BUNDLED_MODEL_FILENAME`,
  `BUNDLED_MODEL_DOWNLOAD_URL`, `BUNDLED_MODEL_APPROX_BYTES`) so the
  UI and the fetch script cannot drift. Pinned alongside the script
  for the lifetime of this minor version.

### Strategy: bundle-vs-download for Gemma

We considered three options for shipping the ~5 GB Gemma 4 E4B
Uncensored GGUF:

1. **Bundle directly in the installer** — installer balloons to
   ~5–6 GB. Hard on the GitHub release asset cap, hostile for users
   on capped connections, painful for every patch release. Rejected.
2. **Download in installer's "post-install" hook** — pulls the file
   during NSIS, blocks the install dialog for ~10 min on a typical
   home connection, and there is no good way to show streaming
   progress from an NSIS macro. Rejected.
3. **Ship a fetch script next to the binary + an in-app banner that
   surfaces it.** Operator runs the install in <30 sec, opens the
   app, sees the banner, clicks "Run fetch-gemma-model.ps1", watches
   PowerShell stream the download with progress. **Selected.** The
   AI Helper panel still works as a chat surface even without the
   model — the topic gate, the read-only tool registry, and the
   audit log are all independent of Gemma's inference path.

### Pre-ship gates

- `cargo fmt --all -- --check` — clean.
- `cargo build --release -p forex-app` — 0 errors, 184 warnings
  (mostly unused-imports from the in-progress trading-mod cleanup),
  52.26 s.
- `cargo packager --release` — produced
  `forex-app_0.4.10_x64-setup.exe`, 25.96 MB.
- Silent-install (`/S /D=<tmp>`) verified all 8 expected files
  present in install dir.

### Known gaps — deferred to v0.5.0

Same as v0.4.9. v0.4.10 is intentionally a payload-fix patch — the
wizard "Broker choice + test connection" steps, the per-panel UI
smoke for all 15 tabs, and full Greek translation all land together
in v0.5.0.

### Artifact

- `forex-app_0.4.10_x64-setup.exe` — 25.96 MB
  - SHA-256: `6737A5FA11FF2CE483E96996F53F82547AE4539C595925137DA59D91901B3046`

---

## [0.4.9] — 2026-05-19 — "Real UI Audit + License-header Fix"

> Patch release that follows v0.4.8 with the bugs surfaced by a real
> walk-through of the binary GUI. v0.4.8 shipped the AI Helper panel
> + proprietary license + NSIS installer; v0.4.9 closes the only
> user-visible regression that audit caught and re-publishes the
> installer.

### Fixed

- **Wizard Step 1 header was stale.** Welcome step rendered
  "Apache License v2.0 / MIT (dual)" above the proprietary license
  text — leftover from the v0.4.7 open-source line. Now reads
  "Proprietary — © 2024-2026 Konstantinos Kokkinos. All rights
  reserved." The LICENSE body itself was already proprietary in
  v0.4.8; only the header label was stale.
- Welcome step's `bundled_license_present` test now accepts a
  `PROPRIETARY` token in addition to the legacy `Apache` / `MIT`
  tokens so the test passes for the new LICENSE without losing the
  guardrail.

### Audit — UI panels static analysis (Items 2 + 7 from operator brief)

Ran `cargo clippy`-style audit over `crates/forex-app/src/ui/**/*.rs`
looking for:

- Stale `MIT` / `Apache` / `open-source` strings — false positives
  only (substring matches on "Limit" / "rate-limited" / "drawdown
  limit"). One real hit fixed (welcome.rs header above).
- TODO / FIXME / `unimplemented!()` in user-visible code — all hits
  are intentional traceability comments pointing at closed gaps
  (`TODO(risky-mode-boot-wire)` — gap closed in v0.4.8;
  `TODO(symbol-universe-canon)` — operator-pin item).
- Hardcoded "Coming soon" / "Placeholder" / "Not implemented" — no
  user-visible occurrences; only test-fixture strings
  (`placeholder-deadbeef`, `placeholder-1234`).
- `panic!` / `.unwrap()` / `.expect()` in UI render paths — all
  occurrences are in non-render auxiliary code (path helpers, test
  setup) or are documented as "must never fail" invariants.

End-to-end click-through with Windows-MCP also exercised:

- Wizard Step 1 → License accept → Continue (worked).
- Wizard advanced past Step 1 cleanly (the v0.4.7 audit had already
  validated Step 4 → 5 cTrader Sign-in → Symbols & Timeframes).
- AI Helper tab loads under the AI Engine group with the welcome
  banner + tool-list hint (validated in v0.4.8).

### Known gaps (deferred to v0.5.0)

The same items as v0.4.8's known-gaps list. v0.4.9 is intentionally
a tight patch — adding the wizard "Broker choice + test connection"
steps + bundling the Gemma GGUF + per-panel UI smoke for all 15
tabs all land together in v0.5.0.

### Artifacts

- `forex-ai-v0.4.9-windows-x86_64-setup.exe` — NSIS installer
  (SHA-256 populated post-build).

## [0.4.8] — 2026-05-19 — "AI Helper + Proprietary License + NSIS Installer"

> Ships the first user-visible Gemma surface, switches the project to a
> proprietary license, and replaces the .zip artifact with a real NSIS
> .exe installer.

### Added

- **AI Helper panel** in the egui workspace under WorkspaceGroup::AiEngine.
  Natural-language read-only console wired to `forex-gemma`:
  - Topic-gate stack (jailbreak-regex via `JailbreakRegexGate`) refuses
    off-topic / injection attempts before Gemma sees them.
  - Keyword router (English + Greek) maps the prompt to one of 10
    read-only `BotTool`s (positions, orders, quote, balance,
    predictions, explain, risk, news, health, log).
  - Result is rendered with a 🛠 prefix; refusals in red ⛔; prose
    fallback through `StubGemmaRuntime` (the real mistral.rs runtime
    lights up in G1 behind the `mistralrs-runtime` feature).
  - Chat scrollback survives tab switches.
  - "Live orders cannot be placed from chat" guardrail visible in the
    panel footer.
- **NSIS Windows installer** via `cargo packager`. The release artifact
  is now `forex-ai-v0.4.8-windows-x86_64-setup.exe` — installs into
  Program Files, registers in Apps & Features for clean uninstall,
  creates Start Menu shortcut. The .zip path is dropped.

### Changed

- **License → Proprietary.** All rights reserved by Konstantinos
  Kokkinos. Personal + demo use OK (subject to the LICENSE terms);
  no redistribution / modification / commercial use without written
  agreement; Greek governing law; commercial-licensing contact
  konstantinoskokkinos1982@gmail.com. The prior MIT-OR-Apache-2.0
  dual grant is **revoked retroactive to v0.4.8**. v0.4.7 and earlier
  binaries remain under MIT-OR-Apache-2.0 per their published LICENSE.
  All Cargo.toml `license` fields updated to `"LicenseRef-Proprietary"`.
- Workspace + Flutter front-end versions bumped to 0.4.8.
- Chocolatey + Scoop + WinGet packaging manifests updated to point at
  the .exe artifact + carry the v0.4.8 SHA-256.

### Verified — pre-ship gates

- `cargo check -p forex-app`: 0 errors (1m 53s)
- `cargo build --release -p forex-app`: 0 errors (3m 18s)
- `cargo fmt --all -- --check`: clean
- `cargo packager --release`: produced `forex-app_0.4.8_x64-setup.exe`
  (20.94 MB)
- GUI smoke: AI Helper tab visible under AI Engine group, 💬 icon,
  chat panel renders with the welcome banner + tool-list hint.

### Known gaps (deferred to v0.4.9)

- The real Gemma runtime (`mistralrs-runtime` feature) — stub returns
  a helpful fallback for now; the GGUF bundling is in the installer
  resources list but the model file itself is fetched separately via
  `scripts/fetch-gemma-model.ps1`.
- Dedicated wizard "Test connection" step (cTrader + DxTrade probe).
- DxTrade panel has the data fields but the dedicated "Test
  connection" button is not yet wired.
- Full Greek translation for every wizard step (currently mixed).
- Comprehensive UI smoke for all 15 workspace tabs (we exercised the
  wizard Steps 1→4→5 + AI Helper + ran Flutter widget tests; the
  per-panel button audit ships in v0.4.9 alongside Flutter parity).

### Artifacts

- `forex-ai-v0.4.8-windows-x86_64-setup.exe` — 20.94 MB
  - SHA-256:
    `E759C4BA7E124250A22D34AD1757403E39ECDF4EF011A5B47C1C8BA138198090`

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
