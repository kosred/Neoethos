# Changelog

All notable changes to forex-ai are documented here. The format is
loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to semantic versioning.

## [0.4.19] — 2026-05-20 — "First public release"

First publicly-tagged release of forex-ai, a pure-Rust forex trading
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

- 553 / 0 unit tests pass in `forex-app`.
- 54 / 0 tests pass in `forex-data`.
- 5 / 0 tests pass in `forex-cli`.
- `cargo check --workspace` clean.
- cTrader OAuth + live spot tested against the demo environment.
