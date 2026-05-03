# Rust Env Flags / Config Debt Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Summary

The project is Rust-first, but current master still contains many `std::env::var` and `FOREX_BOT_*` runtime flags in active Rust code.

For this codebase, hidden environment flags should not be used to change search, backtest, discovery, GPU, model, risk, or live execution semantics. They make runs hard to reproduce and make it impossible to know from a config file or exported portfolio why a strategy was selected.

Rust can technically read environment variables, but in this project they should be limited to infrastructure concerns such as build metadata, logging, CI, and emergency fail-fast switches. Strategy behavior must come from typed config and must be exported into run contracts.

## Current evidence

Repo search for `std::env::var` returned active Rust hits in these areas:

- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/genetic/search_engine.rs`
- `crates/forex-search/src/genetic/smc_indicators.rs`
- `crates/forex-search/src/genetic/strategy_gene.rs`
- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/cubecl_eval.rs`
- `crates/forex-search/src/cubecl_ga.rs`
- `crates/forex-search/src/quality.rs`
- `crates/forex-models/src/genetic.rs`
- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/evolution/*_gpu.rs`
- `crates/forex-models/src/tree_models/*`
- `crates/forex-models/src/runtime/capabilities.rs`
- `crates/forex-app/src/app_services/trading.rs`
- `crates/forex-app/src/app_services/ctrader_execution.rs`
- `crates/forex-core/src/config.rs`
- `crates/forex-core/src/logging.rs`
- `crates/forex-app/build.rs`

Repo search for `FOREX_BOT_` also returned active source hits in search, eval, GPU, model, app-services, core config, and `.env.example`.

## Findings

### 1. Search/discovery behavior is controlled by env flags

Examples already identified in search/discovery:

- prefilter top-k
- prefilter in-sample fraction
- stage-1 funnel percentage
- search seed
- SMC gate schedule
- SMC force ratio / probabilities / min flags
- archive mode and archive thresholds
- survivor/immigrant/selection policy knobs
- novelty weight
- backtest initial equity used by quality screen

**Risk:** two runs with the same `config.yaml`, same data, and same commit can produce different portfolios depending on process environment.

**Severity:** Critical.

**Action:** move these into `DiscoveryConfig`, `SearchConfig`, `SmcSearchConfig`, `BacktestConfig`, and `PortfolioContract`. No hidden env override should affect candidate generation, filtering, ranking, validation, or export.

---

### 2. GPU/HPC behavior is controlled by env flags

Active search hits show env reads in GPU/CUDA paths and fallback paths.

Some flags may be emergency controls, for example fail-fast when GPU was required. But evaluator backend, GPU fallback, CUDA kernels, and CPU/GPU behavior must be recorded in the run contract.

**Risk:** a run may silently use a different backend or kernel path than expected.

**Severity:** High.

**Action:** keep only explicit infrastructure fail-fast flags if absolutely necessary. Everything else should be typed config and exported as effective runtime backend details.

---

### 3. Model/training behavior is controlled by env flags

Active env reads appear in `forex-models`, including genetic model limits, GPU/evolution paths, base/runtime behavior, tree model configuration, and training orchestration.

**Risk:** training behavior is not fully represented by model config/artifact metadata.

**Severity:** High.

**Action:** move model limits, backend choices, and training behavior into typed model config. Export effective settings into `metadata.json` or equivalent training artifact metadata.

---

### 4. Live/app service behavior appears to contain env reads

Search hits include app services such as trading and cTrader execution.

**Risk:** live execution can behave differently from backtest/search if runtime env changes execution details.

**Severity:** High.

**Action:** live execution settings must come from typed risk/execution config only. Backtest/search must use the same execution policy or explicitly record differences.

---

### 5. `.env.example` can preserve Python-era operational habits

The presence of many `FOREX_BOT_*` variables in `.env.example` encourages hidden process-level behavior.

**Risk:** operators tune the system through shell variables instead of versioned config, making results unreproducible.

**Severity:** Medium-High.

**Action:** shrink `.env.example` to infrastructure-only values. Strategy/search/model/risk knobs should move to config files.

## Proposed policy

### Allowed environment variables

Use env vars only for:

- build metadata from build scripts
- CI-only behavior
- logging level/path if not strategy-affecting
- secrets/credentials, never committed or exported
- emergency fail-fast switches, for example requiring GPU instead of silently falling back

### Forbidden environment variables

Do not use env vars for:

- search seed unless copied into typed run contract at startup
- feature selection
- search stage/funnel size
- SMC probabilities or gates
- archive thresholds
- candidate ranking
- backtest initial equity
- spread, commission, pip value, slippage
- validation split sizes
- model training limits
- GPU backend selection, unless exported and explicitly requested in config
- live execution behavior
- risk limits

## Recommended migration plan

1. Create `RuntimeOverrides` / `EffectiveRuntimeConfig` structs.
2. Move all search/discovery env knobs into typed config.
3. Move all SMC env knobs into `SmcSearchConfig` constructed from config, not env.
4. Move all backtest env knobs into `BacktestSettings` / risk config.
5. Move model training env knobs into model config.
6. Restrict `.env.example` to secrets, logging, CI, and infrastructure.
7. Add a CI check that fails on new `std::env::var` in search/model/backtest/live modules unless explicitly allowlisted.
8. Export effective config into every discovery profile, model metadata file, and portfolio contract.
9. Add reproducibility tests: same config + same data + same seed + clean env must produce the same candidates and portfolio.
10. Add a test that clears env vars before search/backtest to guarantee no hidden process-state dependency.

## Immediate cleanup targets

### Critical first pass

- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/genetic/search_engine.rs`
- `crates/forex-search/src/genetic/smc_indicators.rs`
- `crates/forex-search/src/genetic/strategy_gene.rs`
- `crates/forex-search/src/eval.rs`
- `crates/forex-search/src/quality.rs`
- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/lib.rs`

### Second pass

- `crates/forex-models/src/genetic.rs`
- `crates/forex-models/src/base.rs`
- `crates/forex-models/src/training_orchestrator.rs`
- `crates/forex-models/src/evolution/*_gpu.rs`
- `crates/forex-models/src/tree_models/*`
- `crates/forex-app/src/app_services/trading.rs`
- `crates/forex-app/src/app_services/ctrader_execution.rs`

### Allowed/review-only

- `crates/forex-core/src/logging.rs`
- `crates/forex-app/build.rs`
- CI/build metadata paths

## Bottom line

The user expectation is correct: a clean Rust-first trading/search system should not depend on hidden environment flags for strategy behavior. The current repo still has substantial env-driven runtime/config debt. The fix is not to delete the flags blindly, but to migrate them into typed config and exported run contracts, then add CI guardrails so they do not come back.
