# Forex AI - 100% Pure Rust Edition 🚀

High-frequency, mathematically rigid, prop-firm compliant Forex trading engine.

## 🏗️ Architecture
The project has been fully migrated from a hybrid Python/Rust structure to a **Pure Rust** architecture to eliminate GIL bottlenecks and ensure absolute safety.

- **`forex-app`**: The main entry point. Supports both a **Native GUI** (Windows/Linux Desktop) and a **Headless Mode** (Linux VPS/Server).
- **`forex-core`**: Core logic including Risk Management, Portfolio Optimization, and Configuration.
- **`forex-data`**: High-speed OHLCV data engine with zero lookahead bias.
- **`forex-models`**: Native Rust machine learning models (XGBoost, Neural Networks via Burn, etc.).
- **`forex-search`**: Genetic algorithms and strategy discovery.
- **`forex-news`**: Async news aggregation and sentiment analysis.
- **`forex-cli`**: Command-line front-end for batch jobs and operator tasks.

Brokers are integrated entirely through native Rust transports (cTrader Open
API over OAuth/WebSocket and FIX gateways). No embedded Python runtime is
required at runtime — the project is 100% pure Rust on the hot path.

## 🚀 Getting Started

### Prerequisites
- [Rust](https://rustup.rs/) (1.80+)

### Build the Executable
```bash
cargo build --release -p forex-app
```

### Running the App
- **GUI Mode (Desktop):**
  ```bash
  ./target/release/forex-app
  ```
- **Headless Mode (Linux Server):**
  ```bash
  ./target/release/forex-app --headless --config config.yaml
  ```

## 📊 Documentation
Detailed historical analysis and implementation summaries:
- [HPC_CLOUD_IMPLEMENTATION.md](HPC_CLOUD_IMPLEMENTATION.md)
- [BOTTLENECK_ANALYSIS.md](BOTTLENECK_ANALYSIS.md)
- [MIGRATION_PROGRESS.md](MIGRATION_PROGRESS.md)

## ⚖️ License
Proprietary.
