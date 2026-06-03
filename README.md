<div align="center">

# NeoEthos

### *A new ethos for trading — institutional-grade discipline, in the hands of one person.*

**Pure-Rust trading intelligence: it discovers its own strategies, sizes every trade by the math of survival, and answers to a single, honest goal you set.**

`100% Rust on the hot path` · `cTrader / FIX native` · `no Python at runtime` · `your machine, your data, your keys`

</div>

---

## Why this exists

The serious tools — walk-forward validation, genetic strategy search, Kelly-aware sizing, prop-firm risk gates, a real model ensemble — have always lived behind institutional walls and five-figure subscriptions. The small trader gets a chart and a prayer.

NeoEthos is the refusal of that. It puts the *whole pipeline* — research, discovery, risk, execution — on one person's laptop, in software that is auditable line by line and lies to no one (every number comes from the engine; nothing is invented for the UI).

I could have sold this. I'd rather fight to build something larger than me. **So it stays free, for now** — a mission before it's a business. If it helps one person trade with discipline instead of hope, it has already paid for itself.

## What it does

- **Discovers strategies, doesn't ship guesses.** A genetic search breeds and tests strategies across timeframes, then survives them through walk-forward + CPCV gates so what reaches you has held up out-of-sample.
- **Two honest modes, one master switch.** You pick the goal; *the search, the models, and the risk all re-orient around it.*
- **Sizes for survival.** Position size is a fraction of the *live* balance (it compounds as you grow) derived from each strategy's measured edge — not a fixed, account-blowing percentage.
- **A real model ensemble.** XGBoost, CatBoost, LightGBM, neural nets (Burn), KAN, N-BEATS, TabNet, TiDE and more — native Rust, no GIL, no embedded Python on the hot path.
- **Live, broker-native.** cTrader Open API over OAuth/WebSocket (FIX gateways too). Your account, your keys, on your machine.
- **A desktop app *and* a terminal.** A Flutter GUI (Greek / English) and a full ratatui TUI with live candlesticks — same engine underneath.

## The two modes

| | **Risky Mode** | **Prop-Firm Mode** |
|---|---|---|
| **Goal** | Multiply a small balance to a large target, as fast as the edge allows | Pass prop-firm challenges comfortably and bank a steady monthly return |
| **You set** | start → target → horizon (e.g. €100 → €50,000 in 6 months) | the firm preset (FTMO, FundedNext, …) + drawdown caps |
| **The search** | is *pressured* to find strategies that can hit your target in time | optimises for the firm's window-pass rules |
| **The truth** | high risk, deep drawdown — and the odds are computed, not promised | safety and stability first |

You set the goal. The math tells you the truth about it. The bot tries.

## Architecture

Fully pure-Rust — migrated off a Python/Rust hybrid to kill the GIL and earn memory safety end to end.

- **`neoethos-core`** — risk management, portfolio optimisation, the single config that the UI/TUI edit
- **`neoethos-data`** — high-speed OHLCV engine, zero look-ahead bias
- **`neoethos-models`** — the native ML ensemble
- **`neoethos-search`** — genetic discovery + the target-aware ranking
- **`neoethos-app`** — HTTP backend, headless jobs, broker transports
- **`neoethos-cli`** — the TUI + batch operator tasks

## Getting started

**Prerequisites:** [Rust](https://rustup.rs/) 1.80+ · [Flutter](https://flutter.dev) 3.22+ (for the GUI).

```bash
# Build the backend
cargo build --release -p neoethos-app

# Desktop app (spawns the backend for you)
cd experiments/forex-flutter-ui && flutter run -d windows

# Or the terminal UI — live candlesticks, discovery, logs
cargo run --release -p neoethos-cli
```

Prefer a packaged install? Grab the Windows or Linux installer from the [latest release](../../releases/latest).

## Status

**v0.4.40** — the engine works end to end: discovery, both trading modes, risk-aware sizing, a bilingual desktop app, a live TUI. The road ahead: broker-agnostic adapters → a mobile monitor → a multi-node deployment. *v0.5 is "everything works."* We're close.

## License

Free to use, today. This is a gift while it earns the right to be more. The formal terms will follow the mission, not the other way around.

<div align="center">

*Built with discipline, in the open, by one person who believes the small trader deserves better tools.*

</div>
