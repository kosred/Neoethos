<div align="center">

# NeoEthos

### *A new ethos for trading — institutional-grade discipline, in the hands of one person.*

**Pure-Rust trading intelligence: it discovers its own strategies, sizes every trade by the math of survival, and answers to a single, honest goal you set.**

`100% Rust on the hot path` · `cTrader native` · `no Python at runtime` · `your machine, your data, your keys`

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE) · [Latest release](../../releases/latest) · [Build from source](BUILDING.md)

</div>

---

## Why this exists

The serious tools — walk-forward validation, genetic strategy search, Kelly-aware sizing, prop-firm risk gates, a real model ensemble — have always lived behind institutional walls and five-figure subscriptions. The small trader gets a chart and a prayer.

NeoEthos is the refusal of that. It puts the *whole pipeline* — research, discovery, risk, execution — on one person's laptop, in software that is auditable line by line and lies to no one: every number comes from the engine; nothing is invented for the UI.

## What it does

- **Discovers strategies, doesn't ship guesses.** A genetic search breeds and tests strategies across timeframes, then survives them through **mandatory** walk-forward + CPCV out-of-sample gates so what reaches you has held up on data it never trained on.
- **Two honest modes, one master switch.** You pick the goal; *the search, the models, and the risk all re-orient around it.*
- **Sizes for survival.** Position size is a fraction of your *live* balance (it compounds as you grow), derived from each strategy's measured edge and the broker's real per-lot costs — not a fixed, account-blowing percentage.
- **A real model ensemble.** XGBoost, CatBoost, LightGBM, neural nets (Burn), KAN, N-BEATS, TabNet, TiDE and more — native Rust, no GIL, no embedded Python on the hot path.
- **Live, broker-native.** cTrader Open API over OAuth/WebSocket. Market **and** conditional (limit/stop) orders, live P/L, a MyFxbook-style trade journal. Your account, your keys, on your machine.
- **A desktop app *and* a terminal.** A single-process **Tauri + React** desktop app (Greek / English) and a full ratatui TUI with live candlesticks — the same Rust engine underneath.

## The two modes

| | **Risky Mode** | **Prop-Firm Mode** |
|---|---|---|
| **Goal** | Multiply a small balance to a large target, as fast as the edge allows | Pass prop-firm challenges comfortably and bank a steady monthly return |
| **You set** | start → target → horizon (e.g. €100 → €50,000 in 6 months) | the firm preset (FTMO, FundedNext, …) + drawdown caps |
| **The search** | is *pressured* to find strategies that can hit your target in time | optimises for the firm's window-pass rules |
| **The truth** | high risk, deep drawdown — and the odds are computed, not promised | safety and stability first |

You set the goal. The math tells you the truth about it. The bot tries.

> **Risk warning.** Trading leveraged FX carries a substantial risk of loss and is not suitable for everyone. NeoEthos is research/educational software and **not financial advice**. Nothing here is a promise of profit. Use a demo account until *you* have verified an edge; you alone are responsible for any live trading.

## Architecture

Fully pure-Rust — migrated off a Python/Rust hybrid to kill the GIL and earn memory safety end to end. A single Tauri process links the engine crates **in-process** and serves the React UI; there is no separate backend to run.

- **`neoethos-core`** — risk management, portfolio optimisation, the single config the UI/TUI edit
- **`neoethos-data`** — high-speed OHLCV + feature engine, zero look-ahead bias
- **`neoethos-models`** — the native ML ensemble
- **`neoethos-search`** — genetic discovery + target-aware ranking + the OOS validation gates
- **`neoethos-trader`** — live signal generation + the autonomous engine
- **`neoethos-app`** — in-process HTTP API, headless jobs, cTrader transports
- **`neoethos-cli`** — the TUI + batch operator tasks
- **`desktop/`** — the Tauri v2 shell + React/TypeScript UI (`crate neoethos-desktop`)

## Getting started

**Just want to run it?** Grab the Windows installer (`.exe` or `.msi`) from the [latest release](../../releases/latest). Launch it, connect your cTrader account (OAuth, one time), and you're in. Use a **Demo** account first.

**Want to build it yourself?** See **[BUILDING.md](BUILDING.md)** for the full, OS-by-OS guide (toolchains, the desktop build, the TUI, optional GPU acceleration, and cTrader setup). The short version:

```bash
# Prerequisites: Rust (stable, 2024 edition) + Node 20+ + Tauri OS deps.
# Desktop app installer (bundles the frontend + engine into one binary):
cd desktop
npm install
npx tauri build          # → target/release/bundle/{nsis,msi}/...

# Or run it live during development:
npx tauri dev

# Terminal UI (live candlesticks, discovery, logs) — no Node needed:
cargo run --release -p neoethos-cli
```

## Status

**v0.5.0 — "everything works."** End to end: strategy discovery with enforced out-of-sample validation, both trading modes, risk-% sizing off the live balance, market + conditional orders, a bilingual desktop app, and a live TUI. The road ahead: broker-agnostic adapters → a mobile monitor → multi-node deployment.

## License

NeoEthos is licensed under the **GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later) — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

In plain terms: you are free to use, study, modify and self-host it. If you run a **modified** version as a network service, the AGPL requires you to offer that service's users the corresponding source. Third-party components keep their own licenses (see `vendor/` and the dependency manifests).

Copyright © 2024–2026 Konstantinos Kokkinos ("kosred").

<div align="center">

*Built with discipline, in the open, by one person who believes the small trader deserves better tools.*

</div>
