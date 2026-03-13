# Rust-First Refactor Plan

## Objective
- Move performance-critical data, feature, and validation paths to Rust.
- Keep Python only where mature/critical libraries are still stronger.
- Remove `pandas`-heavy bottlenecks from core training loops.
- Preserve anti-leak guarantees (no look-ahead) and robust OOS validation.

## What Stays in Python (for now)
- LightGBM, XGBoost, CatBoost training/inference adapters:
  - https://lightgbm.readthedocs.io/
  - https://xgboost.readthedocs.io/en/latest/python/index.html
  - https://catboost.ai/en/docs/
- Reason: mature optimization + ecosystem support (callbacks, early stopping, GPU paths).

## Target Runtime Stack (Rust-first)
- Columnar data and query engine:
  - Polars lazy API: https://docs.pola.rs/user-guide/concepts/lazy-api/
  - Polars Rust API: https://docs.pola.rs/api/rust/dev/polars/
  - Arrow / Parquet crates: https://arrow.apache.org/blog/2025/10/30/arrow-rs-57.0.0/
  - DataFusion extension points: https://datafusion.apache.org/library-user-guide/index.html
- Python bridge:
  - PyO3 user guide: https://pyo3.rs/
  - maturin user guide: https://www.maturin.rs/
  - Arrow C Data Interface (zero-copy interchange): https://arrow.apache.org/docs/13.0/format/CDataInterface.html

## Guardrails (No Look-Ahead / Overfit / Outlier Drift)
- Time-ordered CV and forward split:
  - TimeSeriesSplit reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Keep strict forward holdout in strategy discovery and global training.
- Maintain embargo/purge logic where available.
- Use early-stopping consistently:
  - LightGBM early stopping: https://lightgbm.readthedocs.io/en/v3.3.3/pythonapi/lightgbm.early_stopping.html
  - CatBoost overfitting detector: https://catboost.ai/docs/en/features/overfitting-detector-desc
- Outlier-robust scaling baseline:
  - RobustScaler: https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.RobustScaler.html

## Phase Plan

### Phase 0: Freeze Baseline
- Create benchmark matrix: runtime, peak RSS, model metrics, OOS metrics.
- Define "must-not-regress" thresholds.

### Phase 1: Data + Features (Highest ROI)
- Port full training data preparation to Rust/Arrow/Polars.
- Keep Python interface thin (`PreparedDataset` adapter only).
- Avoid Python-side reindex-heavy multi-TF joins where possible.
- Add feature-shape and dtype checks at the Rust boundary.

### Phase 2: Validation + Backtest Core
- Port leakage-sensitive splitting, embargo, and forward evaluation kernels to Rust.
- Keep parity tests against current Python behavior.

### Phase 3: Training Orchestration
- Keep model algorithms in Python where needed.
- Move orchestration, pooling, and dataset streaming to Rust services.
- Keep memory-map and chunked transfer pathways default-on.

### Phase 4: Python Deletion Pass
- Remove dead Python paths only after parity and benchmark gates pass.
- Delete `pandas` paths last, behind feature flags until stable.

## Acceptance Gates (Per Phase)
- Accuracy parity: no statistically significant degradation on forward test.
- Memory: at least 35% lower peak RSS vs baseline for full-symbol run.
- Throughput: at least 2x faster feature stage on same machine.
- Leakage checks: all time-order tests pass.

## Immediate Next Actions
- Done: add float32 downcast path for feature matrices.
- Done: add optional lower-TF drop (`FOREX_BOT_DROP_LOWER_TFS`) to prevent M1 explosion when base TF is higher.
- Done: harden Rust feature tensor orientation handling (Arrow tensor transpose guard).
- Done: add Rust feature profiles for TA-Lib generation (`full/core/compact`) and per-TF caps:
  - `feature_profile`, `htf_feature_profile`
  - `max_features`, `max_htf_features`
- Done: wire Python `FeatureEngineer` to auto-select RAM-safe Rust profiles in `auto` mode.
- Done: auto-enable parallel feature workers in global training when symbol count and RAM budget allow.
- Done: remove Python re-compute of RSI/MACD/ADX in Rust path for `base_signal` fallback:
  - Reuse Rust TA columns (`ta_rsi`, `ta_macd_*_outmacdhist`) directly.
  - New fast-mode guard for huge datasets:
    - `FOREX_BOT_RUST_BASE_SIGNAL_MODE=auto|classic|discovery`
    - `FOREX_BOT_RUST_BASE_SIGNAL_CLASSIC_ROWS` (default `1_000_000`)
- Done: make `parallel_worker` pandas-optional on `FOREX_BOT_PANDAS_FREE=1` path:
  - NumPy memmap loading stays active end-to-end for rust-tree workers.
  - pandas is imported lazily only for legacy/non-rust metadata paths.
- Done: split Rust tree feature bundles to avoid CatBoost linker blockers on constrained setups:
  - New Cargo feature: `tree-models-core` (`lightgbm + xgboost`).
  - Full bundle remains available with `tree-models` (`tree-models-core + catboost`).
- Done: enforce rust-only behavior in pandas-free mode in model registry:
  - `FOREX_BOT_PANDAS_FREE=1` now disables silent Python fallback for Rust tree models.
  - Optional explicit mode: `FOREX_BOT_TREE_BACKEND=rust_strict`.
- Done: enforce rust-only behavior in data loader under Rust profiles:
  - In strict mode (`FOREX_BOT_PANDAS_FREE=1` or `FOREX_BOT_RUNTIME_PROFILE=rust_*`), loader no longer silently falls back to Python frame loading.
- Done: make `data.loader` pandas-lazy:
  - Removed eager module-level pandas import.
  - Added lazy `_pandas_module()` loading and dataframe/datetime duck-typed checks.
  - Strict Rust mode can now import loader without forcing pandas import.
- Done: make `strategy.evo_prop` pandas-lazy:
  - Removed eager pandas import and replaced datetime-index checks with duck-typed helpers.
  - Replaced one timestamp formatting path with stdlib `datetime` conversion (ms -> UTC).
- Done: unify TA-Lib mixer signal conversion:
  - Added `signal_to_numpy` and `signal_shift_prev` in `features.talib_mixer`.
  - Replaced pandas-only signal alignment chains in discovery/pipeline/training/genetic paths.
- Done: runtime profile thread defaults to avoid single-core regressions:
  - `FOREX_BOT_DISCOVERY_CPU_BUDGET`, `FOREX_BOT_PROP_SEARCH_WORKERS`, and BLAS/Rayon thread envs now default from profile-aware CPU budget.
  - `rust_32gb` caps discovery thread budget to a RAM-safe ceiling.
- Done: make `execution.training_service` pandas-lazy:
  - Removed eager module-level pandas import.
  - Added lazy pandas proxy and dataframe/series/datetime duck-typed guards.
  - Replaced pandas `isinstance(...)` checks that previously forced import just for type checks.
- Done: make `features.pipeline` pandas-lazy:
  - Removed eager module-level pandas import.
  - Added lazy pandas proxy and datetime-index duck-typed guards in normalization/session feature steps.
- Done: make `training.trainer` pandas-lazy:
  - Removed eager module-level pandas import.
  - Added lazy pandas proxy and dataframe/series/datetime duck-typed guards in training/eval orchestration checks.
- Done: discovery/search window defaults for Rust runtime profiles:
  - `FOREX_BOT_PROP_SEARCH_TRAIN_YEARS=10`
  - `FOREX_BOT_PROP_HOLDOUT_YEARS=3` and `FOREX_BOT_PROP_HOLDOUT_REQUIRED=1`
  - `FOREX_BOT_PROP_HOLDOUT_FRACTION=0` (calendar holdout takes precedence)
  - `strategy.evo_prop` now trims to recent train-years before holdout split.
- Done: all-timeframes/all-features runtime toggles:
  - `FOREX_BOT_USE_ALL_TIMEFRAMES=1` expands loader + feature-engineer timeframe resolution to canonical TF list.
  - `FOREX_BOT_USE_ALL_FEATURES=1` forces Rust feature extraction profile to `full/full` with unlimited caps.
- Next: move session/news joins and label generation fully to Rust (remove remaining pandas-heavy joins).

## New Runtime Preset (32GB-class machines)
- Keep all symbols, all TFs, but cap feature width aggressively:
  - Preferred single switch: `FOREX_BOT_RUNTIME_PROFILE=rust_32gb`
  - Equivalent grouped defaults include:
    - `FOREX_BOT_RUST_FEATURES=1`
    - `FOREX_BOT_PANDAS_FREE=1`
    - `FOREX_BOT_TREE_BACKEND=rust_strict`
    - `FOREX_BOT_FRAME_IO_BACKEND=polars`
    - `FOREX_BOT_RUST_FEATURE_PROFILE=core`
    - `FOREX_BOT_RUST_HTF_FEATURE_PROFILE=compact`
    - `FOREX_BOT_RUST_MAX_FEATURES=96`
    - `FOREX_BOT_RUST_MAX_HTF_FEATURES=12`
    - `FOREX_BOT_PARALLEL_FEATURES=auto`
    - `FOREX_BOT_FEATURE_WORKER_GB=6`
- Optional extra safety:
  - `FOREX_BOT_DROP_LOWER_TFS=1` when base TF > M1.
  - `FOREX_BOT_RUST_BASE_SIGNAL_MODE=classic` on ultra-low RAM nodes.

## Rust Binding Build Notes
- Core Rust tree build (no CatBoost link dependency):
  - `cargo build -p forex-bindings --features tree-models-core`
  - `cd crates/forex-bindings && maturin develop --features tree-models-core`
- Full Rust tree build (includes CatBoost):
  - `cargo build -p forex-bindings --features tree-models`
  - Requires `catboostmodel.lib` available on Windows linker path.
