# Rust-First Library Stack (Research Notes)

## Goal
- Keep Rust as the default execution engine.
- Minimize Python/GIL exposure to orchestration and model wrappers only.
- Replace pandas-heavy dataflow with Rust-native columnar pipelines.

## Recommended Core Stack
- Dataframe + lazy query: Polars (Rust)
  - https://docs.pola.rs/
- Columnar interchange: Apache Arrow / Parquet (Rust crates)
  - https://docs.rs/arrow/latest/arrow/
  - https://docs.rs/parquet/latest/parquet/
- Python bridge with explicit GIL release: PyO3
  - https://pyo3.rs/
  - https://docs.rs/pyo3/latest/pyo3/marker/struct.Python.html#method.detach
- Existing tree models in this repo:
  - `lightgbm3` crate (already integrated)
  - `xgb` crate (already integrated)
  - CatBoost remains optional due Windows linker artifact requirements.

## Deep Learning Options (Rust-native)
- Burn (Rust DL framework):
  - https://docs.rs/crate/burn/latest
- Candle (Rust ML framework by Hugging Face):
  - https://github.com/huggingface/candle

## Execution Rules for This Repo
- `FOREX_BOT_PANDAS_FREE=1` should imply strict Rust tree backend.
- `FOREX_BOT_TREE_BACKEND=rust_strict` should fail fast when Rust bindings are missing.
- Keep Python fallback only when `pandas_free` is disabled.

## Practical Build Modes
- Core tree engine (recommended on Windows, avoids CatBoost linker issue):
  - `cargo build -p forex-bindings --features tree-models-core`
  - `cd crates/forex-bindings && maturin develop --features tree-models-core`
- Full tree engine:
  - `cargo build -p forex-bindings --features tree-models`
  - Requires `catboostmodel.lib` in linker path.
