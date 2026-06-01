# Changelog

All notable changes to NeoEthos are documented here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to semantic versioning.

## [0.4.20] — 2026-06-01

Operator-requested live-desk gaps, plus fixes caught by an exhaustive
click-every-element QA pass. Full notes: `docs/release-0.4.20/RELEASE-NOTES.md`.

### Added
- **Multi-account picker (F-333)** — Settings → App lists every cTrader
  account the OAuth token grants (Demo + Live, with badges) and lets the
  operator pick the active one; the backend promotes the chosen cTID to the
  front of `broker_credentials.toml`.
- **Editable data directory (F-332)** — Settings → Data exposes the data dir
  with an Apply button and a live "✓ N symbols found" readout + inventory,
  so the backend reliably sees the local OHLCV set.
- **Inline buy/sell on the chart (F-334)** — click a Market Watch symbol to
  open its chart; a one-click SELL[bid] · LOTS · BUY[ask] strip sits above the
  candles with a live/stale freshness marker.

### Fixed
- Inline buy/sell never rendered — it was a `Positioned` overlay over a
  `CustomPaint(size: Size.infinite)` in the chart `Stack`; moved to the column
  flow so it always lays out.
- Quick-trade panel vanished on stale ticks (demo majors gap 15–20 s); it now
  stays visible with an amber "stale Ns" marker and an "awaiting price" stub.
- AI Helper chat input sat below the fold (MediaQuery-sized message box);
  pinned the input to the bottom with the message list filling above.
- AI Helper Codex chat verified end-to-end (auth schema + Responses API).
- Live spot stream sends an app heartbeat — no more periodic "Bye" reconnect.

## [Unreleased] — 2026-05-21 — "NeoEthos rebrand"

**Project renamed from `forex-ai` to `NeoEthos`.** New tagline:
*"A disciplined multi-model ML engine for FX strategy research
and risk-aware execution."* The name change removes the generic
"AI" suffix and adopts the Greek root *ethos* (character /
discipline) — a deliberate fit with the risk-aware execution
philosophy.

### Workspace changes

- All 8 crates renamed: `forex-app` → `neoethos-app`, `forex-cli`
  → `neoethos-cli`, `forex-core` → `neoethos-core`, `forex-data`
  → `neoethos-data`, `forex-gemma` → `neoethos-gemma`,
  `forex-models` → `neoethos-models`, `forex-news` →
  `neoethos-news`, `forex-search` → `neoethos-search`.
- Crate directories renamed on disk (`git mv` — preserves history).
- Bundle identifier: `com.forexai.app` → `com.neoethos.app`.
- Display name: `ForexAI` → `NeoEthos`.
- User data dir: `<data-dir>/forex-ai/` → `<data-dir>/neoethos/`.
  **Existing keyring tokens and log files do not migrate** —
  operators need to OAuth again on first run after the rebrand.
  Acceptable because the keyring backend was just rewired in
  task #81 (no prior persisted state in production).
- Stale `crates/forex-app/errors.txt` (4000+ lines of historical
  build errors) deleted.

### Out of scope for this commit

- **Packaging manifests** (winget / chocolatey / scoop / homebrew)
  untouched — those carry historical 0.4.x version refs and will
  be re-cut at the next release under the new name.
- **GitHub repo rename** is a separate manual step on the operator
  side (`kosred/forex-ai` → `kosred/neoethos`). Cargo.toml
  `homepage` / `repository` URLs already point at the new path
  so the rename completes the loop.
- **`experiments/forex-flutter-ui/`** Flutter prototype left
  untouched — it's a parallel sandbox, will get renamed when the
  Flutter migration happens for real.

## [0.4.19] — 2026-05-20 — "First public release"

First publicly-tagged release of neoethos, a pure-Rust forex trading
engine with a native desktop GUI (egui) and a CLI surface for
discovery, training, and backtesting batch jobs.

### Highlights

- **Native desktop UI** (egui/eframe) — chart, watchlist, order
  ticket, news, execution surface, broker setup, runtime status.
- **cTrader broker integration** — OAuth login, account discovery,
  live spot stream, historical trendbars, order execution
  (Market / Limit / Stop), position close, order cancel.
- **DXtrade broker integration** — REST auth + order submission
  (Phase D3.1-D3.4); WebSocket streaming for live quotes.
- **Genetic strategy search** with cTrader-fed datasets. Population
  + generations + archive + novelty + SMC integration. GPU
  acceleration via cubecl.
- **Prop-firm risk gate** — hard pre-trade safety checks:
  daily/total drawdown, mandatory stop-loss, real per-pip account-
  currency value, JPY pip precision, entry-price requirement for
  Market orders with stop-loss.
- **Risky Mode kill-switch tiers** with operator-acknowledged
  initial-stage ruin probability ceiling.
- **News + sentiment** — OpenAI + Perplexity backends with
  explicit `SecretString` opt-in (no env-driven silent activation).
- **Pure-Rust workspace** — no Python at runtime. CI guard
  (`scripts/check_no_python_legacy.sh`) blocks reintroduction.

### Verified at ship

- 553 / 0 unit tests pass in `neoethos-app`.
- 54 / 0 tests pass in `neoethos-data`.
- 5 / 0 tests pass in `neoethos-cli`.
- `cargo check --workspace` clean.
- cTrader OAuth + live spot tested against the demo environment.
