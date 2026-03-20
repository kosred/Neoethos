# Repo Audit Report

## Executive Summary

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

## File-By-File Findings

Initial line-by-line findings from the runtime-critical entrypoints audited so far:

### Critical

#### `crates/forex-models`

- [`crates/forex-models/src/training_orchestrator.rs:48`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L48)
  - the runtime training closure passed into `train_models_parallel()` is still a placeholder that only logs and returns `Ok(())`
  - consequence: the CLI/runtime path can report `Pure Rust training complete` without actually fitting or persisting any model

### Important

#### `crates/forex-app`

- [`crates/forex-app/src/main.rs:298`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L298)
  - the Discovery tab still runs a mock progress loop and never invokes the real Rust discovery backend
  - this explains the current UI/backend disconnect: the screen can show apparent progress without any actual search work
- [`crates/forex-app/src/main.rs:327`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L327)
  - the Training tab button only logs `Training logic triggered.` and does not call any backend training path
  - the current UI therefore cannot truthfully represent training state or failures
- [`crates/forex-app/src/main.rs:71`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L71)
  - local/headless mode hardcodes `"data"` and ignores the supplied config path for symbol discovery
- [`crates/forex-app/src/main.rs:135`](C:/Users/konst/development/forex-ai/crates/forex-app/src/main.rs#L135)
  - GUI local mode repeats the same hardcoded `"data"` assumption during initial symbol enumeration and refresh

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

#### `crates/forex-models`

- [`crates/forex-models/src/training_orchestrator.rs:62`](C:/Users/konst/development/forex-ai/crates/forex-models/src/training_orchestrator.rs#L62)
  - `get_enabled_models()` is hardcoded to `xgboost` and `genetic`, ignoring the loaded settings surface
  - this means the training runtime contract is currently not configuration-driven

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

#### `crates/forex-core`

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

Pending contract audit.

## Warning Inventory

Pending baseline and runtime sweeps.

## Recommended Fix Tranches

Pending stabilization backlog population.

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
