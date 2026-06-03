# NeoEthos environment variables — `NEOETHOS_BOT_*` reference

**Owner**: Κωνσταντίνος · **Status**: living document · **Last update**: 2026-05-29 (F-313)

This file documents every `NEOETHOS_BOT_*` environment variable the backend
consults at startup or while wiring runtime overrides. The list was
historically scattered across `crates/neoethos-core/src/config.rs`,
`crates/neoethos-cli/src/main.rs`, the CatBoost integration tests, and a
handful of feature-pipeline knobs — **F-313 (2026-05-29)** consolidated
the documentation here so the operator has a single place to look when
deciding whether to override defaults.

> **Policy note.** None of these are user-facing — config.yaml is the
> canonical surface for everyday tuning. The env vars exist for:
> (a) CI / smoke tests overriding a single field without authoring a
> whole YAML, (b) headless CLI runs (`neoethos-cli auto-loop`) where a
> wrapper script wants to pin one knob per invocation, (c) niche
> debugging escape hatches. New work should add fields to `Settings`
> + the Advanced Settings knob editor instead of inventing a new env
> var.

## Resolution order

`Settings::with_env_runtime_overrides()` (`config.rs:1093+`) is called
**after** `Settings::from_yaml(...)` — so an env var ALWAYS wins over
config.yaml. The CLI hard-codes `NEOETHOS_BOT_DATA_ROOT` for the in-process
training orchestrator before invoking it (`main.rs:738`), so when you run
`neoethos-cli auto-loop --data-root <path>` the orchestrator sees that
path regardless of what config.yaml says.

## Variables — alphabetical

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `NEOETHOS_BOT_AUTO_ENABLE_RLLIB` | bool (`1`/`true`) | unset | Force-enable RLLib agent path even when `models.use_rllib_agent` is `false` in config.yaml. |
| `NEOETHOS_BOT_BASE_TIMEFRAME` | string (`M5`, `H1`, `D1`, ...) | inherited from config | Override base timeframe for the next Discovery/Training run. |
| `NEOETHOS_BOT_CALIBRATION_METHOD` | `"platt"` / `"isotonic"` / `"none"` | inherited | Override probability-calibration method used by the meta layer. |
| `NEOETHOS_BOT_CALIBRATION_MIN_ROWS` | usize | inherited | Minimum row count before calibration kicks in. Lower it for tiny-dataset smoke tests. |
| `NEOETHOS_BOT_CATBOOST_EXECUTABLE` | path | unset → built-in catboost.exe | Path to a custom catboost CLI binary. Used by `tree_models_integration` tests when the bundled binary isn't viable on the host. |
| `NEOETHOS_BOT_DATA_DIR` | path | (none) | Legacy alias for `NEOETHOS_BOT_DATA_ROOT`. Honoured for back-compat. |
| `NEOETHOS_BOT_DATA_ROOT` | path | inherited from `system.data_dir` | Root of the historical-bars store. The CLI auto-loop sets this to the resolved `--data-root` argument before spawning the in-process trainer (`main.rs:738`). |
| `NEOETHOS_BOT_DEVICE` | `"cpu"` / `"cuda"` / `"vulkan"` | inherited | Compute device used by the deep-timeseries models. Overrides `system.hardware.gpu_*` flags. |
| `NEOETHOS_BOT_ENABLE_GPU_PREFERENCE` | bool | inherited | Toggles the auto-prefer-GPU heuristic in capability resolution. |
| `NEOETHOS_BOT_HIGHER_TFS` | comma-list (`M15,H1,H4`) | inherited | Override the higher-timeframe set used in feature alignment. |
| `NEOETHOS_BOT_LABEL_HORIZON_BARS` | usize | inherited | Bar horizon used when computing the directional label (Classification3). |
| `NEOETHOS_BOT_META_LABEL_MAX_HOLD_BARS` | usize | inherited | Max-hold horizon used when computing the meta-label for the stacker. |
| `NEOETHOS_BOT_ML_MODELS` | comma-list | inherited | Override `models.ml_models`. Useful for smoke tests that only want to train one model. |
| `NEOETHOS_BOT_NORMALIZE_FEATURES` | `1` / unset | unset (disabled) | Opt-in feature normalisation (z-score clip to ±10) in the feature pipeline (`crates/neoethos-data/src/lib.rs:860`). |
| `NEOETHOS_BOT_NUM_TRANSFORMERS` | usize | inherited | Override `models.num_transformers`. |
| `NEOETHOS_BOT_PHASE5_CORE_MODELS` | comma-list | inherited | Override the Phase 5 core model list. |
| `NEOETHOS_BOT_PHASE5_FILTER_META_BLENDER` | bool | inherited | Toggle the Phase-5 filter on `meta_blender`. |
| `NEOETHOS_BOT_PNL_AUDIT_DRIFT_FRACTION` | f64 | inherited | Drift fraction the cTrader PnL audit treats as acceptable (test fixture knob, see `crates/neoethos-app/tests/fixtures/ctrader/unrealized_pnl/README.md`). |
| `NEOETHOS_BOT_PNL_CIRCUIT_BREAKER_FRACTION` | f64 | inherited | Hard ceiling above which the PnL audit raises a circuit-breaker error. |
| `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` | string (`USD`, `EUR`, ...) | inherited | Legacy override for the prop-firm preset account currency. Honoured by both the CLI prop-firm command (`main.rs:414`) and `with_env_runtime_overrides`. |
| `NEOETHOS_BOT_PROP_CONF_THRESHOLD` | f64 | inherited | Confidence threshold the prop-firm gate uses to filter ensemble signals. |
| `NEOETHOS_BOT_PROP_SEARCH_ASYNC` | bool | inherited | Toggle async prop-firm search. |
| `NEOETHOS_BOT_PROP_SEARCH_ASYNC_WAIT` | usize (ms) | inherited | Poll interval for async prop-firm search status. |
| `NEOETHOS_BOT_PROP_SEARCH_DEVICE` | `"cpu"` / `"cuda"` | inherited | Device for prop-firm search. |
| `NEOETHOS_BOT_REGIME_NEUTRAL_MODELS` | comma-list | inherited | Override `models.regime_neutral_models`. |
| `NEOETHOS_BOT_REGIME_RANGE_MODELS` | comma-list | inherited | Override `models.regime_range_models`. |
| `NEOETHOS_BOT_REGIME_ROUTER_ENABLED` | bool | inherited | Force-enable / disable the regime router. |
| `NEOETHOS_BOT_REGIME_ROUTER_MIN_MODELS` | usize | inherited | Minimum loaded models the regime router requires before routing. |
| `NEOETHOS_BOT_REGIME_TREND_MODELS` | comma-list | inherited | Override `models.regime_trend_models`. |
| `NEOETHOS_BOT_RLLIB_NUM_WORKERS` | usize | inherited | Worker count for the RLLib trainer. |
| `NEOETHOS_BOT_SYMBOL` | string (`EURUSD`, `EUR/USD`, ...) | inherited | Override the symbol the next Discovery/Training run targets. |
| `NEOETHOS_BOT_TRAIN_HOLDOUT_PCT` | f64 (`0.0`–`1.0`) | inherited | Hold-out fraction the training orchestrator carves off for the WFA val set. |
| `NEOETHOS_BOT_TREE_DEVICE` | `"cpu"` / `"cuda"` / `"vulkan"` | inherited | Compute device used by the tree models (LightGBM / XGBoost / CatBoost). Independent of `NEOETHOS_BOT_DEVICE` so the operator can keep deep models on GPU while pinning tree models to CPU (or vice versa). |
| `NEOETHOS_BOT_USE_RLLIB_AGENT` | bool | inherited | Override `models.use_rllib_agent`. |

## Deprecation candidates

The following are currently honoured but **planned for removal** once
the matching config.yaml fields land in the Advanced Settings knob
editor surface:

| Variable | Replacement |
|---|---|
| `NEOETHOS_BOT_AUTO_ENABLE_RLLIB` | `models.use_rllib_agent` |
| `NEOETHOS_BOT_PROP_SEARCH_*` | `search.prop_firm.*` |
| `NEOETHOS_BOT_PNL_AUDIT_*` | unified `risk.pnl_audit` block |

Operators should migrate scripts away from these variables; the env-var
fallback survives only for back-compat across the 0.4.x line and will
be removed in 0.5.x.

## Non-`NEOETHOS_BOT_` variables the backend also reads

For completeness (these are documented elsewhere but listed here so the
operator has one search target):

| Variable | Documented in | Purpose |
|---|---|---|
| `PERPLEXITY_API_KEY` | (not committed) | News-blackout LLM gate (optional) |
| `NEOETHOS_LAUNCHED_BY_FLUTTER` | `crates/neoethos-app/src/main.rs` | Set by the Flutter `BackendSupervisor` so the backend can distinguish "spawned by UI" vs "manual run" (#179) |
| `CONFIG_FILE` | `crates/neoethos-app/src/main.rs` | Override the default `%LOCALAPPDATA%\neoethos\config.yaml` lookup path |
| `VULKAN_SDK` / `LIBCLANG_PATH` | `scripts/build-cargo-release.ps1` | Compile-time tooling, not runtime |
