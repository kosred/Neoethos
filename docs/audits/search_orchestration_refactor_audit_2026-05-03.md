# Search Orchestration Refactor Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Scope: latest analysis of the search orchestration/API layer and genetic search stack.

## Files inspected

- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/genetic/mod.rs`
- `crates/forex-search/src/genetic/search_engine.rs`
- `crates/forex-search/src/genetic/strategy_gene.rs`
- `crates/forex-search/src/genetic/evolution_math.rs`

Some large files were truncated by the connector, so this audit focuses on the visible orchestration, configuration, evaluation, and gene lifecycle sections.

No production code was changed by this audit.

## Core conclusion

The search stack needs refactoring at the initial orchestration/API layer, not only inside kernels.

The current design can expose the same or similar API names while changing behavior based on compile features, env vars, or fallback paths. This is dangerous for a production trading bot because different execution paths may not mean the same thing.

The target should be explicit, typed modes:

```rust
pub enum SearchEvaluationMode {
    CanonicalCpuBacktest,
    CanonicalCubeClCudaBacktest,
    GpuTensorApproxPresearch,
    HpcIslandApproxPresearch,
    HybridGpuSignalCpuBacktest,
}
```

No caller should have to infer semantics from feature flags, env vars, or runtime backend strings.

## `lib.rs` finding

`crates/forex-search/src/lib.rs` exposes GPU-related discovery modules differently depending on the `gpu` feature.

When `gpu` is enabled, it uses the real `discovery_gpu` module.

When `gpu` is not enabled, it defines a fallback `discovery_gpu` module inside `lib.rs` that keeps a compatible API but runs CPU fallback logic.

This is useful for compilation but risky for semantics.

A function named `run_gpu_discovery` can mean different things depending on build features:

- GPU feature enabled: tensor/tch GPU discovery path.
- GPU feature disabled: CPU fallback implementation under the same module name.

This should be refactored so runtime mode is explicit in the type system and in returned artifacts.

## `discovery.rs` findings

`discovery.rs` is the high-level discovery pipeline.

The visible pipeline does:

```text
trim recent history
-> feature prefilter
-> stage-1 funnel
-> evolve_search_with_progress_and_limits
-> finalize_candidates_with_progress
```

This is the correct place to introduce a typed `DiscoveryExecutionPlan` or `SearchPipelinePlan`.

### Env-driven semantics in `discovery.rs`

The visible code still reads env vars that change search behavior:

- `FOREX_BOT_PREFILTER_TOP_K`
- `FOREX_BOT_FUNNEL_STAGE1_PCT`
- `FOREX_BOT_PREFILTER_INSAMPLE`
- `FOREX_BOT_BACKTEST_INITIAL_EQUITY`

These are not just debug toggles. They affect:

- which features survive prefiltering
- how much history is used in stage-1 search
- whether feature selection leaks into validation data
- final score and PnL scaling

They should become typed config fields, generated at install/first-run and editable through UI.

### Feature prefilter contract issue

`prefilter_features` can change the feature schema. If downstream artifacts still reference original feature names or indices, this can create feature-index/name mismatch.

The discovery result should record:

- original feature schema
- effective feature schema after prefilter
- mapping from original indices to effective indices
- feature schema hash

## `genetic/search_engine.rs` findings

This is the canonical genetic search engine.

Positive finding:

`evaluate_genes_cached` and `evaluate_genes` eventually call:

```rust
crate::eval::evaluate_population_core(...)
```

This means the canonical search path can be anchored around the CPU/CubeCL evaluator contract.

This file is the right place to introduce a `CanonicalSearchEngine` concept.

### Env-driven search policy

The visible code uses many env vars that define core search behavior:

- `FOREX_BOT_SEARCH_SEED`
- `FOREX_BOT_PROP_SMC_GATE_START`
- `FOREX_BOT_PROP_SMC_GATE`
- `FOREX_BOT_PROP_SMC_GATE_END`
- `FOREX_BOT_PROP_SMC_GATE_CURVE`
- `FOREX_BOT_PROP_SMC_GATE_STAGNATION_STEP`
- `FOREX_BOT_PROP_SEEN_RETRY`
- `FOREX_BOT_PROP_ARCHIVE_MODE`
- `FOREX_BOT_PROP_ARCHIVE_MIN_NET`
- `FOREX_BOT_PROP_ARCHIVE_MIN_PF`
- `FOREX_BOT_PROP_ARCHIVE_MIN_SHARPE`
- `FOREX_BOT_PROP_ARCHIVE_CAP`
- `FOREX_BOT_PROP_RANDOM_IMMIGRANTS`
- `FOREX_BOT_PROP_SURVIVOR_FRACTION`
- `FOREX_BOT_PROP_ELITE_FRACTION`
- `FOREX_BOT_PROP_PARENT_SELECTION`
- `FOREX_BOT_PROP_SURVIVOR_SELECTION`
- `FOREX_BOT_PROP_SELECTION_TEMPERATURE`
- `FOREX_BOT_PROP_TOURNAMENT_SIZE`
- `FOREX_BOT_PROP_STAGNATION_GENS`
- `FOREX_BOT_NOVELTY_WEIGHT`

These should be moved into typed policies:

```rust
pub struct EvolutionSearchPolicy { ... }
pub struct SmcGateSchedulePolicy { ... }
pub struct SearchArchivePolicy { ... }
pub struct NoveltySearchPolicy { ... }
pub struct SeenMemoryPolicy { ... }
```

The UI/settings layer can edit these, but the running search should receive an immutable typed plan.

## `strategy_gene.rs` findings

`Gene` currently mixes multiple lifecycle stages:

- raw genome fields
- evaluation metrics
- SMC flags
- stop/target values
- runtime search metadata
- strategy ID
- consistency/slice stats

This works for a prototype but is risky for a production pipeline because approximate, evaluated, validated, and portfolio-selected strategies can all look like the same `Gene` type.

Recommended lifecycle split:

```rust
pub struct CandidateGene { ... }
pub struct EvaluatedGene { gene: CandidateGene, metrics: EvaluationMetrics, evaluation_mode: SearchEvaluationMode }
pub struct ValidatedStrategyGene { evaluated: EvaluatedGene, validation_report: ValidationReport }
pub struct PortfolioStrategy { validated: ValidatedStrategyGene, portfolio_weight: f64 }
```

This prevents approximate GPU presearch candidates from being accidentally treated as final validated strategies.

### `Gene::normalize` determinism issue

`Gene::normalize` still uses `rand::rng()` when it needs to add missing indicators.

This breaks deterministic search even if the outer search has a seeded RNG.

Recommended change:

- make `normalize` deterministic without randomness, or
- add `normalize_with_rng(&mut self, ..., rng: &mut impl Rng)` and ensure callers pass the seeded RNG.

## `strategy_gene.rs` cost model findings

`infer_market_cost_profile` still reads env vars for symbol/account/pip/spread/commission overrides:

- `FOREX_BOT_PROP_SYMBOL`
- `FOREX_BOT_PROP_ACCOUNT_CURRENCY`
- `FOREX_BOT_PROP_PIP_VALUE`
- `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`
- `FOREX_BOT_PROP_PIP_VALUE_PER_LOT`
- `FOREX_BOT_PROP_SPREAD_PIPS`
- `FOREX_BOT_PROP_COMMISSION`

These directly affect PnL, spread, commission, and ranking.

They must become explicit `MarketCostProfile` input from settings/config/UI, not hidden env behavior.

## `evolution_math.rs` findings

Positive finding:

This file already has typed enums for parent and survivor selection:

```rust
ParentSelectionPolicy
SurvivorSelectionPolicy
EvolutionSearchPolicy
```

This is the right style and should be expanded to the rest of search configuration.

### Seen memory env debt

`SeenSignatureMemory::from_env()` still reads env vars for persistence and memory bounds:

- `FOREX_BOT_PROP_SEEN_FLUSH_EVERY`
- `FOREX_BOT_PROP_SEEN_LOAD_MAX`
- `FOREX_BOT_PROP_SEEN_MAX_ENTRIES`
- `FOREX_BOT_PROP_SEEN_FILE`

This should become:

```rust
pub struct SeenMemoryPolicy {
    pub file_path: Option<PathBuf>,
    pub flush_every: usize,
    pub load_max: usize,
    pub max_entries: usize,
}
```

This is especially important for checkpoint/resume and long-term search memory.

## Required refactor direction

### 1. Introduce typed execution plans

Add a top-level immutable search execution plan:

```rust
pub struct SearchPipelinePlan {
    pub mode: SearchEvaluationMode,
    pub evolution: EvolutionSearchPolicy,
    pub smc_gate: SmcGateSchedulePolicy,
    pub archive: SearchArchivePolicy,
    pub novelty: NoveltySearchPolicy,
    pub seen_memory: SeenMemoryPolicy,
    pub market_cost: MarketCostProfile,
    pub feature_schema: FeatureSchemaContract,
    pub runtime: ResolvedRuntimeConfig,
}
```

### 2. Stop hiding semantics behind `run_gpu_discovery`

Rename or split APIs so they say what they do:

```rust
run_gpu_tensor_approx_presearch(...)
run_hpc_island_approx_presearch(...)
run_canonical_search(...)
run_canonical_validation(...)
```

or expose one dispatcher that returns explicit mode provenance.

### 3. Replace env reads with config reads

Env vars may remain for tests/debug/emergency only. Production behavior should come from generated config and UI settings.

### 4. Split `Gene` lifecycle

Do not let approximate and validated results share an indistinguishable type.

### 5. Add artifact provenance

Every discovery result should include:

- search evaluation mode
- exact execution plan hash
- hardware profile ID
- feature schema hash
- original/effective feature mapping
- dataset fingerprint
- CPU/GPU canonical validation status
- runtime backend and device assignment

## Bottom line

The search stack already has strong building blocks. The problem is orchestration ambiguity.

The refactor should start by making execution mode, search policy, feature schema, market cost model, and runtime backend explicit. Only after that should the GPU kernels be scaled further.

Otherwise the bot may become faster while still mixing approximate, canonical, CPU, GPU, and fallback results under the same `Gene` and `run_gpu_discovery` abstractions.
