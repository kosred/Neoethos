# Changelog

All notable changes to NeoEthos are documented here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to semantic versioning.

## [0.5.2] — 2026-07-03 — "the honesty release"

The release that makes NeoEthos safe to trust: strategies must survive a
five-sided validation gauntlet before they can touch money, position sizing
is derived from survival mathematics rather than hope, and — new here — many
small machines can pool their compute against the server farms of the big
players. Built and validated on a 6-core mini PC.

### Added
- **Five-gate anti-overfitting validation** — every exported strategy now
  survives walk-forward, **CPCV** (combinatorially purged cross-validation),
  **PBO/CSCV** (López de Prado's probability of backtest overfitting), a
  **permutation test** (no profit allowed on structurally destroyed data),
  and a **parameter-plateau test** (±15% perturbation must keep ≥30% of net).
  The full honesty box, documented in [PRINCIPLES.md](PRINCIPLES.md).
- **Risk-constrained Kelly sizing** (Busseti/Ryu/Boyd) solved on the *full
  empirical R-multiple distribution* — fat left tails (rare catastrophic
  losses) shrink the recommended size automatically, a CVaR-aware sizer with
  no trained model. Surfaced as "RCK risk/trade" in the tail-risk panel.
- **Prop-firm challenge simulator** — first-passage Monte Carlo against
  FTMO-style barriers (+10%/+5% targets vs −10% max / −5% daily loss),
  sweeping risk-per-trade to find the challenge-optimal size and the
  attempts-for-90%-funding budget. `GET /autonomous/challenge`.
- **Federation Phase 0** — SETI@home-style shared discovery with no server:
  any instance can coordinate a work plan; trusted peers run discovery on
  their own machines and submit results into an inbox where every submission
  still passes all local gates before any real money. `/federation/*` +
  Advanced → Federation panel.
- **Auto-cull → automatic re-discovery** — a retired (permanently
  blacklisted) strategy now triggers a fresh discovery on the same
  symbol/timeframe to refill the gap (Settings-gated, default on).
- **Trade-sequence Monte Carlo tail risk** — p95 drawdown, risk-of-ruin and
  time-under-water p95 across thousands of reshuffles of the realized trades.
- **Institutional-footprint feature family** and extended FVG memory (up to
  64 unfilled gaps per side, with magnet-distance/age signals).
- **Live-experience store + offline learnability report** — records the exact
  feature rows live entries acted on and honestly measures whether live
  outcomes carry learnable signal yet (report-only, never blind trust).
- **Session-aware spread recorder**, **news gate** wired into live autopilot,
  and a **live↔backtest parity harness** (`GET /autonomous/parity`).
- **Project site + community pack**: [kosred.github.io/Neoethos](https://kosred.github.io/Neoethos/)
  (bilingual, zero-tracker), PRIVACY.md, PRINCIPLES.md, CONTRIBUTING.md,
  a real-world hardware guide, and GitHub Sponsors support.

### Changed
- **Scoring version 5** — the genetic search now admits **negative indicator
  weights** (contrarian terms are discoverable, not just seed-inherited, with
  a sign-flip mutation), and **Risky mode evolves under its own objective**:
  half-Kelly expected log-growth, the same math its post-GA ranking uses.
  PropFirm/Strict discovery keeps the v4 consistency landscape byte-for-byte.
- **Slippage** folded into every backtest cost; scoring v4 added a
  worst-day penalty so steady monthly income is rewarded over lumpy equity.
- **UI consolidation** — Trade + Positions merged into one cockpit (with
  inline SL/TP editing + trailing); Account + Journal merged into one screen.
- All workspace crates aligned to a single version number (0.5.2).

### Fixed
- Cockpit landing directly showed "No open positions" / a dashed account
  panel — the account stream only pushes on demand; the cockpit now refreshes
  on mount and every 5 s.
- Backtest↔live parity restored for trailing stops and weekend kill zones.

### Deferred (documented, not abandoned)
- **P2P mesh (iroh)** — approved architecture in
  `docs/p2p-mesh-design-2026-07-03.md`, deliberately built as an *isolated
  sidecar* later so it can never destabilise the pinned trading engine.
- **cTrader MCP client** — the official `rmcp` SDK is mature; awaiting stable
  server schemas and a concrete MCP-only workflow.

## [0.5.1] — 2026-07-02

Professional charting and self-defending autopilot.

### Added
- **KLineChart v10** migration — pro indicators, drawing tools and sub-panes;
  zoom to the full history.
- **Auto-cull** — a live strategy that breaks either a consecutive-loss limit
  **or** a rolling-window win-rate floor (default 57% over 10 trades) is
  stopped and **permanently retired** into a blacklist: never re-selected,
  never re-discovered, kept as a record (never deleted).

### Fixed
- Break-even + trailing stop now applied live in the autopilot, in **parity**
  with the discovery backtest.
- **Stop** is now responsive across Discovery, Training and the autopilot
  bar-boundary wait (interruptible, no more "Stop does nothing").
- Trade Journal shows real symbols/pairs and is scoped per account.

## [0.5.0] — 2026-07-01 — "everything works"

End to end: strategy discovery with enforced out-of-sample validation, both
trading modes (Risky multiply / Prop-firm robust), risk-% sizing off the live
balance, market **and** conditional (limit/stop) orders, a bilingual desktop
app, and a live terminal UI — all on one pure-Rust engine. License changed to
**AGPL-3.0-or-later**.

## [0.4.99] — 2026-06-14

Pre-release consolidation: single-process **Tauri** desktop shell linking the
engine crates in-process (Flutter path retired), config unified into one
source of truth edited by both UI and TUI, GPU/CPU compute selectable at
runtime, and a broad defensive-coding + dead-code audit across the workspace.

## [0.4.35] — 2026-06-01

A professional-desk release: a full myfxbook-style trade journal,
tunable strategy-discovery search budget, a settings-persistence fix,
deeper history downloads, and hardened on-disk data — plus a TUI
candlestick chart. All new write paths follow a defensive-coding
standard (no `.unwrap()`/panics on fallible or integration paths;
failures degrade to clear, actionable log messages).

### Added
- **Trade journal / performance analytics (myfxbook-style)** — closed
  trades and an equity curve are persisted (append-only JSONL under
  `<data_dir>/journal/`) and surfaced in a new **Journal** tab on the
  Positions screen. A pure stats engine computes net/gross P&L, profit
  factor, win rate, average win/loss, payoff ratio, expectancy, largest
  win/loss and max consecutive losers, plus equity-derived max drawdown
  (absolute + %), recovery factor and Sharpe. New `GET /journal/trades`
  and `GET /journal/stats` endpoints. The journal is filled automatically
  from live broker deals during the account-refresh heartbeat —
  idempotent on position id, off the main thread, best-effort (a journal
  hiccup never affects trading).
- **Tunable Discovery search budget** — Settings → Discovery exposes and
  persists seven search knobs (population, generations, max-hours,
  max-indicators, portfolio size, correlation threshold, max rows) so the
  search depth can differ between a local box and a VPS.
- **TUI candlestick chart** — a new terminal-UI page renders OHLCV
  candles (Braille canvas) for any local symbol / timeframe.
- **History-download depth readout** — the data bootstrap screen reports
  the oldest bar fetched (date + approximate years of depth) and warns
  when a broker's retention is shallow.

### Fixed
- **Settings did not persist** — handlers wrote a CWD-relative config
  instead of the live per-user `config.yaml` the engine loads; the path
  now resolves to the same `%LOCALAPPDATA%\neoethos\config.yaml`.
- **Truncated history downloads** — the historical-bar chunk ceiling was a
  fixed 100, silently capping long spans; it is now derived from the
  requested span (clamped) so multi-year fetches complete.
- **On-disk data hardening** — the Vortex read/convert path detects
  implausibly small / truncated `.vortex` files and column-length
  mismatches with clear errors instead of surfacing corrupt OHLCV.

### Changed
- Version bumped to 0.4.35 across all crates and the Flutter UI.

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
