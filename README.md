# Forex AI - 100% Pure Rust Edition 🚀

High-frequency, mathematically rigid, prop-firm compliant Forex trading engine.

## 🏗️ Architecture
The project has been fully migrated from a hybrid Python/Rust structure to a **Pure Rust** architecture to eliminate GIL bottlenecks and ensure absolute safety.

- **`forex-app`**: The main entry point. Supports both a **Native GUI** (Windows/Linux Desktop) and a **Headless Mode** (Linux VPS/Server).
- **`forex-core`**: Core logic including Risk Management, Portfolio Optimization, and Configuration.
- **`forex-data`**: High-speed OHLCV data engine with zero lookahead bias.
- **`forex-models`**: Native Rust machine learning models (XGBoost, Neural Networks via Burn, etc.).
- **`forex-search`**: Genetic algorithms and strategy discovery.
- **`mt5-bridge`**: Embedded MetaTrader 5 bridge that calls official APIs directly from Rust memory.

## 🚀 Getting Started

### Prerequisites
- [Rust](https://rustup.rs/) (1.80+)
- [MetaTrader 5 Terminal](https://www.metatrader5.com/) installed on your machine.

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
