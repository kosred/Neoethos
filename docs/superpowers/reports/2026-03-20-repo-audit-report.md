# Repo Audit Report

## Executive Summary

Current stabilized baseline as of `2026-03-21`:

- `cargo test --workspace`: PASS
- `cargo clippy --workspace --all-targets -- -D warnings`: PASS
- `crates/mt5-bridge` now formats the official MetaTrader 5 `last_error()` tuple contract correctly; real headless app startup surfaces `code=-6 description=Terminal: Authorization failed` instead of a PyO3 extraction mismatch
- `crates/forex-data` now fails fast on unreadable discovered/requested timeframes and rejects invalid timestamp columns instead of constructing synthetic zero timestamps

This report still contains the original baseline findings for historical audit traceability. Resolved items are summarized in the stabilization update below.

## Repository Census

Tracked surface:
- `tracked files`: 1081
- `Rust source files`: 106
- `Python files`: 6
- `TOML files`: 14
- `Markdown files`: 15
- `examples/` files: 3
- `scripts/` files: 5
- `tests/` files: 0 tracked files
- `crates/` files: 119 tracked files

## Subsystem Classification Matrix

| subsystem | status | audit treatment |
|---|---|---|
| `crates/forex-app` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/forex-cli` | runtime-critical | full static + runtime + line-by-line audit |
| `crates/mt5-bridge` | runtime-critical | full static + contract + environment audit |
| `crates/forex-core` | runtime-critical | full static + line-by-line audit |
| `crates/forex-data` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-search` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-models` | runtime-critical | full static + runtime-path + line-by-line audit |
| `crates/forex-bindings` | runtime-critical | full static + Python-contract audit |
| `crates/forex-news` | audit-required | static + line-by-line audit, runtime only if reachable from active paths |
| remaining Python files | audit-required | classify as runtime code, bridge, bootstrap, or dead seam |
| top-level config/script/service assets | audit-required | audit as integration assets |
| `tests/` | evidence-only | audit for coverage gaps and stale assumptions |
| `examples/` | evidence-only | audit for coverage gaps and stale assumptions |
| `vendor/` | excluded | inventory only unless a local patch affects behavior |
| `cache/`, `logs/`, `target/`, generated artifacts | excluded | evidence only, not first-party source |

## Active-File Classification Matrix

| file class | status | notes |
|---|---|---|
| runtime-critical crates | `runtime-critical` | direct supported runtime paths |
| shim and binding files | `audit-required bridge` | can affect runtime contracts |
| helper scripts and top-level config | `static-only` | integration assets, not runtime modules |
| examples and tests | `evidence-only` | supporting verification assets |
| vendor and generated output | `excluded` | do not treat as first-party code quality surface |

## Repository Census Details

### Excluded Classes

- `vendor/`: excluded from code-quality findings unless a local patch is implicated
- `cache/`: excluded, evidence-only artifacts only
- `logs/`: excluded, evidence-only artifacts only
- `target/`: excluded build output
- generated local outputs: excluded unless they are evidence for a finding

### Workspace Members

- `crates/forex-search`
- `crates/forex-cli`
- `crates/forex-data`
- `crates/forex-models`
- `crates/forex-core`
- `crates/forex-bindings`
- `crates/forex-app`
- `crates/mt5-bridge`
- `crates/forex-news`

### Binaries And Entrypoints

- `crates/forex-cli/src/main.rs`
- `crates/forex-app/src/main.rs`
- `examples/parallel_training.rs`
- `examples/tree_models_rust_example.rs`
- `examples/true_parallel_training.rs`

### Python-Dependent Crates And Bindings

- `crates/forex-bindings` exports the first-party Python shim at `crates/forex-bindings/forex_bindings/__init__.py`
- `crates/mt5-bridge` still owns the Python-backed MT5 bridge contract
- vendor TA-Lib helper scripts remain tracked but excluded from first-party code-quality findings

### UI Crate

- `crates/forex-app` is the Rust desktop UI crate and the only active UI surface in the tracked workspace

### MT5 Bridge Crate

- `crates/mt5-bridge` is the MT5 integration seam and remains runtime-critical because it owns broker initialization and terminal-info contracts

### Remaining Python And Shim Presence

- `crates/forex-bindings/forex_bindings/__init__.py` is the only tracked first-party Python shim file
- vendor Python helpers tracked under `vendor/talib-sys/dependencies/tmp/ta-lib/swig/src/tools/test_python/`
- vendor Python interface file tracked under `vendor/talib-sys/dependencies/tmp/ta-lib/swig/src/interface/python.py`

### Test And Examples Surface

- `tests/`: 0 tracked files at the current HEAD census
- `examples/`: 3 tracked Rust examples

## Static Verification Findings

### cargo check --workspace

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Warning clusters: 7
- Total warnings observed: 58

#### Warnings by subsystem

- crates/forex-core/src/domain/risk.rs: unused imports in risk module
- crates/forex-core/src/domain/portfolio.rs: unused imports and dead correlation_threshold field
- crates/forex-news/src/openai.rs, crates/forex-news/src/perplexity.rs, crates/forex-news/src/lib.rs: unused imports
- crates/forex-data/src/core/features.rs: unused imports and unused bail import
- crates/forex-search/src/validation.rs: unused imports
- crates/forex-models/src/*: many unused imports, unused variables, and dead model fields across tree/evolution/RL/statistical/anomaly modules
- crates/forex-bindings/src/*: unused imports and deprecated downcast calls
- crates/forex-app/src/main.rs: unused imports and deprecated downcast call


### cargo test --workspace

- Exit code: 1
- Result state: FAIL
- Failing test: `crates/forex-models/src/hardware.rs:377` in `hardware::tests::test_gpu_distribution`
- Root cause: `distribute_gpu_assignment()` uses zero-based modulo arithmetic, but the test contract is explicitly written as one-based model numbering (`Model 1 -> GPU 0`, `Model 2 -> GPU 1`, etc.), so implementation and test disagree on indexing semantics.
- Evidence: `cargo test --workspace` failed with `left: 1 right: 0` at `crates\forex-models\src\hardware.rs:377`
- Recommended fix direction: normalize the index before modulo or change the contract to explicit 0-based indexing and update the test accordingly
### cargo clippy --workspace --all-targets -- -D warnings

- Exit code: 1
- Result state: FAIL
- High-signal clippy errors in `crates/forex-core`:
  - `crates/forex-core/src/domain/risk.rs`: unused imports, `manual_range_contains`, `collapsible_if`, `too_many_arguments`, `manual_clamp`, and `derivable_impls`
  - `crates/forex-core/src/domain/portfolio.rs`: unused imports, dead `correlation_threshold` field, and `needless_range_loop`
  - `crates/forex-core/src/domain/drift_monitor.rs`: `manual_clamp`
  - `crates/forex-core/src/system.rs`: `new_without_default` and `needless_borrows_for_generic_args`
- High-signal clippy errors in `crates/forex-data`:
  - `crates/forex-data/src/core/features.rs`: unused imports and `derivable_impls`
  - `crates/forex-data/src/core/indicators.rs`: `needless_range_loop`
  - `crates/forex-data/src/core/resample.rs`: `collapsible_if`
  - `crates/forex-data/src/lib.rs`: `unnecessary_map_or`
- Evidence: `cargo clippy --workspace --all-targets -- -D warnings` stopped on `forex-core` after surfacing 16 errors there and additional errors in `forex-data`
- Recommended fix direction: clean the shared core/data crates first, then rerun clippy before touching higher-level layers

### cargo build -p forex-cli

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Build completed successfully in `dev` profile
- Repeated warning clusters observed in `crates/forex-core`, `crates/forex-data`, `crates/forex-search`, and `crates/forex-models`
- No new build-only blocker surfaced beyond the existing warning inventory

### cargo build -p forex-app

- Exit code: 0
- Result state: PASS WITH FINDINGS
- Build completed successfully in `dev` profile
- Repeated upstream warning clusters remained visible
- App-local warnings still present in `crates/forex-app/src/main.rs` for unused `Arc` and `Mutex` imports

### Python Contract Probes

- `python -c "import sys; print(sys.version)"`: PASS, Python `3.13.9`
- `python -c "import forex_bindings"`: PASS
- `python -c "import MetaTrader5"`: PASS

### Informational Lanes

- `cargo check -p forex-models`: PASS WITH FINDINGS, repeated warning cluster in `crates/forex-models`
- `cargo check -p forex-search`: PASS WITH FINDINGS, repeated warning cluster in `crates/forex-search` plus shared warnings from `crates/forex-data`
- `baseline-linux`: N/A in the current Windows-only session; retain as review-only lane until a Linux host is available
## Runtime Findings

Runtime probes were executed against the already-built `target/debug` binaries after the static baseline confirmed the binaries link successfully. This avoided repeated `cargo run` rebuild noise while preserving the same runtime code path.

### CLI Data And Feature Paths

- `symbols`: PASS
  - discovered 7 symbols from `data/`
- `timeframes EURUSD`: PASS
  - discovered 21 timeframes for EURUSD
- `load EURUSD M1`: PASS
  - loaded `5267265` rows
- `features EURUSD M1`: PASS
  - produced a feature frame with `5267265` rows and `11` columns
- `prepare EURUSD base=M1 higher=M5,M15,H1`: PASS
  - produced a multitimeframe feature frame with `5267265` rows and `44` columns

### CLI Discovery And Training Paths

- `discover EURUSD ... population=10 generations=1 candidates=20 portfolio-size=10`: PASS WITH FINDINGS
  - command exited `0`
  - result was operationally degraded: `portfolio=0` and `candidates=3`
  - follow-up needed in the line-by-line audit to determine whether this is expected under the minimal smoke configuration or whether the CLI should surface an explicit degraded-state warning/non-zero outcome
- `train EURUSD ...`: PASS
  - completed with `Pure Rust training complete for EURUSD`

### App Startup Paths

- `forex-app --headless --local --config config.yaml`: PASS
  - process started successfully
  - remained alive for a 6-second smoke window
  - required controlled termination
  - emitted no stderr
- `forex-app --local --config config.yaml`: PASS
  - GUI process started successfully
  - remained alive for a 6-second smoke window
  - required controlled termination
  - emitted no stderr

### MT5 Runtime Surface

- MT5 prerequisite probe: PASS WITH FINDINGS
  - `MetaTrader5` imported successfully
  - `initialize()` returned `False`
  - runtime evidence: `last_error = (-6, 'Terminal: Authorization failed')`
- MT5 bridge contract probe: BLOCKED
  - the interactive Trading-tab connect action could not be automated from the current terminal-driven audit session
  - this is environment/tooling blocked, not yet a proven code failure
  - the prerequisite probe already shows the local MT5 environment is not authorized, so a connect attempt is expected to fail unless terminal authorization is fixed

## 2026-03-21 Stabilization Update

### Resolved Since Baseline

- `cargo test --workspace` is now green after fixing the GPU distribution contract in [`crates/forex-models/src/hardware.rs`](C:/Users/konst/development/forex-ai/crates/forex-models/src/hardware.rs)
- `cargo clippy --workspace --all-targets -- -D warnings` is now green after the warning-cleanup tranche across `forex-core`, `forex-data`, `forex-search`, `forex-models`, `forex-bindings`, `forex-app`, `forex-cli`, and `forex-news`
- [`crates/mt5-bridge/src/lib.rs`](C:/Users/konst/development/forex-ai/crates/mt5-bridge/src/lib.rs) no longer extracts `last_error()` as `String`; it now formats the documented `(code, description)` payload with a safe string fallback, backed by unit tests
- [`crates/forex-data/src/lib.rs`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs) no longer swallows unreadable timeframes during dataset assembly and no longer converts invalid timestamp columns into synthetic zeroes; targeted regression tests now pin both contracts
- [`crates/forex-core/src/logging.rs`](C:/Users/konst/development/forex-ai/crates/forex-core/src/logging.rs) and [`crates/forex-core/src/sectioned_log.rs`](C:/Users/konst/development/forex-ai/crates/forex-core/src/sectioned_log.rs) now own one canonical sectioned log file at `logs/forex-ai.log`; the active runtime no longer depends on the legacy per-run file appender path
- [`crates/forex-app/src/main.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs), [`crates/forex-cli/src/main.rs`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs), and [`crates/mt5-bridge/src/lib.rs`](C:/Users/konst/development/forex-ai/crates/mt5-bridge/src/lib.rs) now emit explicit subsystem-scoped records into `APP`, `CLI`, `DISCOVERY`, `TRAINING`, and `MT5` sections instead of creating separate runtime log files or relying on ad-hoc logger setup
- [`crates/forex-app/src/app_services/jobs.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/jobs.rs), [`crates/forex-app/src/app_services/discovery.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/discovery.rs), and [`crates/forex-app/src/app_services/training.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/training.rs) now provide a real app-owned service layer with explicit job state, progress, cancellation, and report contracts
- [`crates/forex-app/src/ui/discovery.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/discovery.rs) and [`crates/forex-app/src/ui/training.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/training.rs) no longer rely on mock progress or log-only placeholders; both tabs now start real backend services and expose `Stop` and `Open Log` actions
- [`crates/forex-app/src/ui/trading.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading.rs) now owns the MT5/local trading-panel routing that previously lived inline in [`crates/forex-app/src/main.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs), preserving the connect/disconnect and canonical `APP` logging contract while shrinking the entrypoint
- [`crates/forex-app/src/ui/hardware.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/hardware.rs), [`crates/forex-app/src/ui/risk.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/risk.rs), and the focused state holders in [`crates/forex-app/src/app_state.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs) now own the remaining operator-panel defaults and slider bounds that previously sat inline in [`crates/forex-app/src/main.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs)
- [`crates/forex-app/src/app_services/jobs.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/jobs.rs), [`crates/forex-app/src/app_services/discovery.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/discovery.rs), [`crates/forex-app/src/app_services/training.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/training.rs), and [`crates/forex-app/src/ui/components.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/components.rs) now expose structured operator reports with `counters`, `highlights`, and `entries`, so the UI can show top discovery metrics and model lists without forcing operators into raw log inspection
- the same app service/report path now keeps a bounded `events` trail per job snapshot, letting the UI surface live operational milestones from discovery and training instead of only the latest summary block
- legacy artifacts such as `logs/forex_bot.log` still exist on disk from earlier runs, but the verified current runtime path updates only `logs/forex-ai.log`

### New Verification Evidence

- `cargo test --workspace` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test -p mt5-bridge` -> PASS
- `cargo test -p forex-app -- --nocapture` -> PASS
- `cargo clippy -p forex-app --all-targets -- -D warnings` -> PASS
- `cargo test -p forex-cli -- --nocapture` -> PASS
- `cargo test -p forex-data -- --nocapture` -> PASS
- `cargo test -p forex-core -- --nocapture` -> PASS
- `cargo clippy -p forex-core --all-targets -- -D warnings` -> PASS
- `cargo run -p forex-app -- --headless --config config.yaml` -> PASS WITH FINDINGS, emitting `MT5 Initialization failed. Last error: code=-6 description=Terminal: Authorization failed`
- `target/debug/forex-app.exe --headless --local --config config.yaml` -> PASS WITH CONTROLLED STOP after a 6-second local-mode smoke window
- `cargo run -p forex-cli -- load --symbol EURUSD --timeframe M1 --root data` -> PASS, `Loaded EURUSD M1 rows: 5267265`
- `cargo run -p forex-cli -- discover --root data --symbol EURUSD --base M1 --higher M5,M15,H1 --population 10 --generations 1 --candidates 20 --portfolio-size 10` -> PASS WITH FINDINGS, now failing explicitly with `Discovery produced an empty portfolio for EURUSD M1 (candidates=4)` instead of exiting `0`
- canonical log inspection after app/CLI reruns -> PASS, with `SYSTEM`, `APP`, `CLI`, `DISCOVERY`, and `MT5` sections updating independently inside `logs/forex-ai.log`

## File-By-File Findings

Initial line-by-line findings from the runtime-critical entrypoints audited so far:

### Critical

#### `crates/forex-models`

- [`crates/forex-models/src/training_orchestrator.rs:48`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L48)
  - the runtime training closure passed into `train_models_parallel()` is still a placeholder that only logs and returns `Ok(())`
  - consequence: the CLI/runtime path can report `Pure Rust training complete` without actually fitting or persisting any model

### Important

#### `crates/forex-app`

Historical baseline findings, now resolved by the Phase 1 app-service-layer tranche:

- [`crates/forex-app/src/ui/discovery.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/discovery.rs)
  - the Discovery tab now starts a real backend discovery service instead of a mock progress loop
- [`crates/forex-app/src/ui/training.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/training.rs)
  - the Training tab now starts a real backend training service instead of a log-only placeholder
- [`crates/forex-app/src/ui/trading.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/trading.rs)
  - the Trading tab logic is now isolated from the app entrypoint and keeps explicit `LocalOnly` / `Disconnected` / `Connected` modes under test instead of mixing MT5 state changes directly into the top-level UI shell
- [`crates/forex-app/src/ui/hardware.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/hardware.rs) and [`crates/forex-app/src/ui/risk.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/ui/risk.rs)
  - the Hardware and Risk tabs now render through focused modules with tested slider bounds, instead of leaving more operator state and control ranges embedded in the main entrypoint
- [`crates/forex-app/src/app_services/jobs.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/jobs.rs), [`crates/forex-app/src/app_services/discovery.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/discovery.rs), and [`crates/forex-app/src/app_services/training.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_services/training.rs)
  - the app service layer now keeps richer operator-facing report sections for both tabs, including best discovery strategy highlights and planned/completed/failed model entries during training
- [`crates/forex-app/src/app_state.rs`](C:/Users/konst/development/forex-ai/crates/forex-app/src/app_state.rs)
  - runtime data-root selection, hardware defaults, and risk defaults are now held in app state and reused by the UI/service layer instead of repeating hardcoded inline assumptions

No new high-severity `forex-app` findings were introduced on the verified Windows lane in this tranche.

#### `crates/mt5-bridge`

- [`crates/mt5-bridge/src/lib.rs:29`](C:/Users/konst/development/forex-ai/crates/mt5-bridge/src/lib.rs#L29)
  - `last_error()` is extracted as `String`, but the official MetaTrader 5 Python docs state that `last_error()` returns an error code and description as a tuple
  - consequence: on failed `initialize()`, the bridge can raise a PyO3 extraction error instead of surfacing the underlying MT5 failure cleanly
  - source: [MQL5 last_error docs](https://www.mql5.com/en/docs/python_metatrader5/mt5lasterror_py)

#### `crates/forex-data`

- [`crates/forex-data/src/lib.rs:93`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L93)
  - `load_symbol_dataset()` silently drops any timeframe that fails to load and still returns `Ok(SymbolDataset { ... })`
  - this can turn missing/corrupt timeframe inputs into partial datasets without any explicit degraded-state signal
- [`crates/forex-data/src/lib.rs:103`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L103)
  - `load_symbol_dataset_with_timeframes()` repeats the same silent-drop behavior for targeted timeframe loads
- [`crates/forex-data/src/lib.rs:116`](C:/Users/konst/development/forex-ai/crates/forex-data/src/lib.rs#L116)
  - timestamp casting failure falls back to a synthetic `0` value vector and then unwraps into integer iteration
  - this is a silent data-corruption path: malformed or unexpected timestamp columns can be converted into apparently valid rows anchored at epoch-zero instead of surfacing a load failure

#### `crates/forex-bindings`

- [`crates/forex-bindings/src/models.rs:71`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/models.rs#L71)
  - the public `MLPModel` binding is explicitly marked as a placeholder and stores a `Gene` rather than a trained model backend
  - it exposes only a constructor, so the API surface suggests an available model while providing no fit/predict contract
- [`crates/forex-bindings/src/models.rs:81`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/models.rs#L81)
  - `GeneticModel` is also exported publicly but only exposes a constructor in this module
  - unless methods are attached elsewhere, this is a skeletal API surface rather than a usable production binding
- [`crates/forex-bindings/src/data.rs:29`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/data.rs#L29)
  - `load_symbol_features()` accepts `max_features`, `max_htf_features`, and `htf_feature_profile`, then explicitly discards them via `_ = ...`
  - the public binding contract therefore advertises feature-budget and HTF-profile control that the implementation does not honor
- [`crates/forex-bindings/src/data.rs:123`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/data.rs#L123)
  - `load_strategy_signals()` is exported publicly but always returns an empty dictionary
  - callers receive a formally successful response from a placeholder bridge with no real signal-loading behavior
- [`crates/forex-bindings/src/data.rs:138`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/data.rs#L138)
  - `count_weekday_trading_days()` always returns `0`, and the exported alignment/news helpers at lines `143` through `169` all return `None`
  - these are live public PyO3 functions registered in [`crates/forex-bindings/src/lib.rs:79`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/lib.rs#L79), so the current bindings surface exposes multiple stubbed contracts as if they were implemented
- [`crates/forex-bindings/src/search.rs:51`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/search.rs#L51)
  - both `search_evolve_ohlcv()` and `search_discovery_ohlcv()` convert gene-serialization failures into `None` entries via `unwrap_or_else(|_| py.None())`
  - that means the binding can silently corrupt result payloads instead of surfacing that Pythonization failed for returned genes
- [`crates/forex-bindings/src/search.rs:181`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/search.rs#L181)
  - the GPU discovery binding always returns `"gpu": true` even though the function contains a `#[cfg(not(feature = "gpu"))]` execution path as well
  - callers cannot trust the returned payload to indicate whether GPU-backed discovery actually happened
- [`crates/forex-bindings/src/validation.rs:42`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/validation.rs#L42)
  - `embargoed_walkforward_backtest_py()` accepts a `settings` argument and then ignores it, always using `BacktestSettings::default()`
  - the Python validation contract therefore exposes configurable backtest settings that never reach the Rust engine
- [`crates/forex-bindings/src/evaluation.rs:532`](C:/Users/konst/development/forex-ai/crates/forex-bindings/src/evaluation.rs#L532)
  - `trade_journal_metrics()` is a public binding that returns hardcoded `sharpe=0.0` and `win_rate=0.0`
  - this is a placeholder analytics contract presented as a real metric endpoint

#### `crates/forex-models`

- [`crates/forex-models/src/training_orchestrator.rs:62`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L62)
  - `get_enabled_models()` is hardcoded to `xgboost` and `genetic`, ignoring the loaded settings surface
  - this means the training runtime contract is currently not configuration-driven
- [`crates/forex-models/src/parallel_trainer.rs:111`](C:/Users/konst/development/forex-ai/crates/forex-models/src/parallel_trainer.rs#L111)
  - `train_models_parallel()` logs individual model failures but still returns `Ok(successes)` without surfacing which models failed or converting all-failed runs into an error
  - any caller that treats `Ok(...)` as a successful training pass can silently accept partial or total model-training failure
- [`crates/forex-models/src/lib.rs:89`](C:/Users/konst/development/forex-ai/crates/forex-models/src/lib.rs#L89)
  - `ONNXInferenceEngine::load_models()` returns `Ok(())` when the models directory or `onnx/` subdirectory is missing
  - the binding caller can therefore treat model loading as successful even when no ONNX assets were available at all
- [`crates/forex-models/src/lib.rs:119`](C:/Users/konst/development/forex-ai/crates/forex-models/src/lib.rs#L119)
  - ONNX inference hardcodes both `with_intra_threads(4)` and `with_inter_threads(4)`
  - this bypasses the repo-wide hardware-adaptive CPU budget and will underutilize large hosts while oversubscribing smaller ones

#### `crates/forex-search`

- [`crates/forex-search/src/validation.rs:63`](C:/Users/konst/development/forex-ai/crates/forex-search/src/validation.rs#L63)
  - `embargoed_walkforward_backtest()` divides by `n_splits` without validating `n_splits > 0`
  - a zero-splits call will panic instead of failing as a normal `Result`
- [`crates/forex-search/src/validation.rs:111`](C:/Users/konst/development/forex-ai/crates/forex-search/src/validation.rs#L111)
  - `prop_compliant` is hardcoded against `0.05` instead of the supplied `max_daily_loss_pct`
  - this makes the returned compliance flag disagree with the actual runtime policy input
- [`crates/forex-search/src/validation.rs:102`](C:/Users/konst/development/forex-ai/crates/forex-search/src/validation.rs#L102)
  - several reported walk-forward fields are still placeholder values (`max_consec_losses`, `consistency_violation`, `trade_limit_violation`, `min_trading_days_ok`)
  - the summary therefore looks richer than the implementation really is
- [`crates/forex-search/src/portfolio.rs:24`](C:/Users/konst/development/forex-ai/crates/forex-search/src/portfolio.rs#L24)
  - `lookback_days` is part of the public optimizer contract but is never used when computing allocations
  - callers can believe they are constraining history while the optimizer actually uses the full return series every time
- [`crates/forex-search/src/discovery.rs:80`](C:/Users/konst/development/forex-ai/crates/forex-search/src/discovery.rs#L80)
  - discovery forces `candidate_count` to a minimum of `100` via `config.candidate_count.max(100)`
  - any caller trying to run a smaller bounded discovery pass cannot actually get the requested limit, so the tuning/runtime contract is not respected
- [`crates/forex-search/src/orchestration.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-search/src/orchestration.rs#L22)
  - `run_batch()` downgrades dataset-load, resample, and feature-prep failures to `info!` logs plus `continue`, then still returns `Ok(())`
  - a batch discovery run can therefore succeed operationally while skipping every symbol or timeframe without surfacing a degraded result to the caller

#### `crates/forex-cli`

- [`crates/forex-cli/src/main.rs:147`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs#L147)
  - `cmd_train()` parses `--root` into `_root` and then ignores it completely
  - the training CLI advertises a data-root flag that never reaches the orchestrator, so runtime behavior depends on the environment variable fallback inside `TrainingOrchestrator` instead
- [`crates/forex-cli/src/main.rs:370`](C:/Users/konst/development/forex-ai/crates/forex-cli/src/main.rs#L370)
  - the printed help for `train` still documents `--higher` and `--horizon`, but `cmd_train()` does not parse either flag
  - the CLI help surface is therefore no longer aligned with the executable command contract

#### `crates/forex-news`

- [`crates/forex-news/src/openai.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-news/src/openai.rs#L22)
  - when `OPENAI_API_KEY` is missing, `analyze_sentiment()` logs a warning and returns a neutral `0.0` score
  - missing provider configuration is therefore downgraded into a seemingly valid sentiment result instead of an explicit unavailable/degraded-state response
- [`crates/forex-news/src/perplexity.rs:22`](C:/Users/konst/development/forex-ai/crates/forex-news/src/perplexity.rs#L22)
  - when `PERPLEXITY_API_KEY` is missing, `search_news()` logs a warning and returns an empty string
  - callers cannot distinguish “provider unavailable” from “provider found no news” without parsing logs

#### `crates/forex-core`

Historical baseline finding, now resolved by the canonical sectioned logging tranche:

- [`crates/forex-core/src/logging.rs:29`](C:/Users/konst/development/forex-ai/crates/forex-core/src/logging.rs#L29)
  - the `WorkerGuard` returned by `tracing_appender::non_blocking` is stored only in a local `_guard` and dropped when `setup_logging()` returns
  - the official tracing-appender docs state that the guard must be held by the entrypoint to preserve the flush guarantee for buffered logs
  - source: [tracing-appender non_blocking docs](https://docs.rs/tracing-appender/latest/tracing_appender/non_blocking/)
- [`crates/forex-app/src/main.rs:43`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L43)
  - the active app entrypoint calls `tracing_subscriber::fmt::init()` directly and never uses `forex-core::logging::setup_logging()`
  - consequence: the file logging/JSON logging policy in `crates/forex-core/src/logging.rs` is not applied to the current desktop app at all
- [`crates/forex-core/src/logging.rs:28`](C:/Users/konst/development/forex-ai/crates/forex-core/src/logging.rs#L28)
  - the comment says file rotation is enabled (`50MB max, 3 backups`), but the implementation uses `tracing_appender::rolling::never(...)`
  - this is a contract/documentation drift issue and will mislead operators during log retention planning

### Minor

- [`crates/forex-app/src/main.rs:5`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L5)
  - unused `Arc` import remains on the active UI entrypoint
- [`crates/forex-app/src/main.rs:6`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L6)
  - unused `Mutex` import remains on the active UI entrypoint

## Contract And Operational Findings

- the highest-risk operational pattern in the current codebase is `false success`: training, discovery, model loading, and provider-backed news paths can all currently return or appear successful while doing placeholder work or operating in degraded mode
- the second recurring pattern is `public contract drift`: the bindings and CLI surfaces expose parameters, flags, or helper functions that are ignored, stubbed, or no longer aligned with runtime behavior
- the third recurring pattern is `operator visibility loss`: important states are downgraded to log lines, placeholder defaults, or dropped guards rather than explicit result states and durable logging
- these three patterns define the first stabilization order more reliably than a crate-by-crate cleanup pass

## Warning Inventory

Pending baseline and runtime sweeps.

## Recommended Fix Tranches

- `Tranche 1`: remove false-success training and parallel failure suppression
- `Tranche 2`: make dataset/discovery degradation explicit and fail fast on corrupt timestamps
- `Tranche 3`: repair MT5 error surfacing and unify logging/observability contracts
- `Tranche 4`: remove or implement placeholder public bindings and CLI/help contract drift
- `Tranche 5`: clear the enforced warning/lint baseline
- detailed breakdown lives in [`docs/superpowers/reports/2026-03-20-stabilization-backlog.md`](C:/Users/konst/development/forex-ai/docs/superpowers/reports/2026-03-20-stabilization-backlog.md)

## Findings Ledger Schema

Each JSON line in `cache/audit/2026-03-20-findings.jsonl` must include:
- `category`
- `severity`
- `lane`
- `command`
- `file`
- `line`
- `summary`
- `evidence`
- `root_cause`
- `recommended_fix`

Allowed `category` values:
- `build breakage`
- `test failure`
- `lint/warning`
- `runtime breakage`
- `correctness bug`
- `contract mismatch`
- `dead or unreachable code`
- `observability gap`
- `performance risk`
- `architectural smell`
