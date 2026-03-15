# Rust Migration Progress

This document tracks the status of the migration from Python to Rust for the `forex-ai` project.

## Legend
- 🐍 **Python**: 100% Python logic.
- 🔗 **Proxy**: Python file is a thin wrapper/proxy; all core arithmetic/logic is in Rust.
- 🦀 **Rust**: Logic is fully implemented in the `crates/` directory.

---

## Execution Module (`src/forex_bot/execution/`)

| File | Status | Notes |
| :--- | :---: | :--- |
| `risk.py` | 🔗 | Position sizing, drawdown checks, and recovery logic in Rust. |
| `order_execution.py` | 🔗 | Price calculations, leg splitting, and edge evaluation in Rust. |
| `meta_controller.py` | 🔗 | Risk parameter auto-tuning in Rust. |
| `consistency.py` | 🔗 | Performance metrics calculation in Rust. |
| `drift_monitor.py` | 🔗 | ADWIN/Concept drift detection in Rust. |
| `bot.py` | 🐍 | High-level orchestration and MT5 integration. |
| `trading_loop.py` | 🐍 | Main execution loop. |
| `mt5_state_manager.py` | 🐍 | MT5 terminal interface (must stay in Python). |
| `news_service.py` | 🐍 | News filtering and event calendar. |

## Strategy Module (`src/forex_bot/strategy/`)

| File | Status | Notes |
| :--- | :---: | :--- |
| `stop_target.py` | 🔗 | All SL/TP/ATR/Swing logic in Rust. |
| `fast_backtest.py` | 🔗 | Core backtest evaluation engine in Rust. |
| `evo_prop.py` | 🔗 | Evolutionary search, population management, and filtering in Rust. |
| `genetic.py` | 🔗 | All crossover, mutation, and selection logic moved to Rust GA. |
| `discovery.py` | 🔗 | Entry point delegating the discovery cycle to Rust. |
| `discovery_tensor.py` | 🔗 | Tensor-based strategy search and ranking handled by Rust. |

## To-Do List
1. [x] Port `genetic.py` core evolution operators to Rust.
2. [x] Port `evo_prop.py` core search loop and population management to Rust.
3. [x] Port `discovery_tensor.py` logic to Rust (using `burning` or `ndarray`).
