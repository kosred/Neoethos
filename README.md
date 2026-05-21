# NeoEthos

**A disciplined multi-model ML engine for FX strategy research and risk-aware execution.**

100% Pure Rust. Prop-firm compliant. Zero embedded Python on the hot path.

## 🏗️ Architecture
The project has been fully migrated from a hybrid Python/Rust structure to a **Pure Rust** architecture to eliminate GIL bottlenecks and ensure absolute safety.

- **`neoethos-app`**: The main entry point. Supports both a **Native GUI** (Windows/Linux Desktop) and a **Headless Mode** (Linux VPS/Server).
- **`neoethos-core`**: Core logic including Risk Management, Portfolio Optimization, and Configuration.
- **`neoethos-data`**: High-speed OHLCV data engine with zero lookahead bias.
- **`neoethos-models`**: Native Rust machine learning models (XGBoost, Neural Networks via Burn, etc.).
- **`neoethos-search`**: Genetic algorithms and strategy discovery.
- **`neoethos-news`**: Async news aggregation and sentiment analysis.
- **`neoethos-cli`**: Command-line front-end for batch jobs and operator tasks.

Brokers are integrated entirely through native Rust transports (cTrader Open
API over OAuth/WebSocket and FIX gateways). No embedded Python runtime is
required at runtime — the project is 100% pure Rust on the hot path.

## 🚀 Getting Started

### Prerequisites
- [Rust](https://rustup.rs/) (1.80+)

### Build the Executable
```bash
cargo build --release -p neoethos-app
```

### Running the App
- **GUI Mode (Desktop):**
  ```bash
  ./target/release/neoethos-app
  ```
- **Headless Mode (Linux Server):**
  ```bash
  ./target/release/neoethos-app --headless --config config.yaml
  ```

## 📊 Documentation

The CHANGELOG records every release. Per-crate behaviour is documented
inline in the source — start at each crate's `src/lib.rs` or `src/main.rs`.

## ⚖️ License
Proprietary.
