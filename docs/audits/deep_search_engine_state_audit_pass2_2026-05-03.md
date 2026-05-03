# Deep Search Engine State Audit — Pass 2

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: low-level genetic search state, gene normalization, SMC config/source, seen-memory, genetic model artifact mode, determinism, checkpoint-readiness.

## Summary

This pass re-examines the search stack from lower-level building blocks:

- `Gene`
- `Gene::normalize`
- SMC flags and SMC arrays
- `SeenSignatureMemory`
- `evolve_search_with_progress_impl`
- `GeneticStrategyExpert`

The previous audits remain valid. This pass adds deeper detail: the repo already contains some pieces that can become a real search memory/checkpoint system, but determinism and artifact contracts are not yet tight enough.

## Findings

### 1. `Gene::normalize` can break determinism

`Gene::normalize` repairs invalid/too-small genes. If it needs to add missing indicator terms, it creates its own `rand::rng()` internally.

This means deterministic search can still drift even if the main search loop uses a seeded RNG.

**Risk:** same seed/config/data may not reproduce the exact same candidates if normalization repairs a gene and draws randomness from OS/thread RNG.

**Severity:** High.

**Fix direction:** split normalization into:

- deterministic repair with no randomness, or
- `normalize_with_rng(&mut self, ..., rng: &mut impl Rng)`

The search loop must pass the run RNG into normalization.

---

### 2. `SeenSignatureMemory` is a useful start but not a full search memory

`SeenSignatureMemory` can store canonical gene signatures and optionally persist/load them from a file.

This is valuable because it prevents repeated exploration of duplicate strategy signatures.

However, it stores only signatures, not:

- gene body
- metrics
- regime performance
- validation history
- rejection reasons
- lineage/mutation history
- OOS/forward results

**Risk:** the system can remember that it saw a strategy, but not whether it was good, bad, regime-specific, overfit, rejected, or promising.

**Severity:** Medium-High.

**Fix direction:** keep `SeenSignatureMemory` as a fast duplicate filter, but add a separate `SearchKnowledgeBase` for long-term learning across runs.

---

### 3. SMC arrays have two possible sources of truth

`build_smc_arrays` detects SMC columns in the feature frame. If SMC columns exist, it uses those columns. If not, it derives SMC arrays from OHLCV.

This fallback is useful, but it changes semantics depending on feature availability.

**Risk:** two runs with different feature schemas may use different SMC sources even if the user thinks they are running the same strategy search.

**Severity:** High.

**Fix direction:** export an `smc_source_contract`:

- `feature_columns`
- `ohlcv_derived`
- `mixed_feature_and_derived`

Also export the detected SMC column mapping.

---

### 4. SMC config is env-driven and must become typed

`SmcSearchConfig::from_env` controls force ratio, minimum flags, and probabilities for each SMC flag through env vars.

**Risk:** search universe changes outside typed config.

**Severity:** Critical.

**Fix direction:** make `SmcSearchConfig` part of `SearchConfig` / `DiscoveryConfig` and export it.

---

### 5. `enforce_population_smc_ratio` also breaks determinism

`enforce_population_smc_ratio` creates an internal `rand::rng()` when it must enforce missing SMC flags.

**Risk:** deterministic main search seed does not fully determine population creation.

**Severity:** High.

**Fix direction:** change signature to:

```rust
pub fn enforce_population_smc_ratio(
    genes: &mut [Gene],
    cfg: &SmcSearchConfig,
    rng: &mut impl Rng,
)
```

---

### 6. `GeneticStrategyExpert` has two distinct backend modes

`GeneticStrategyExpert` has:

- `DiscoveryBacked`
- `LabelSearch`

`DiscoveryBacked` runs the discovery pipeline.

`LabelSearch` optimizes agreement with labels using accuracy/macro-F1/coverage/directional precision. It is not equivalent to trading PnL search.

**Risk:** both modes produce `Gene` portfolios, but their `fitness`, `sharpe_ratio`, `win_rate`, `profit_factor`, and `max_drawdown` fields do not have the same meaning.

**Severity:** High.

**Fix direction:** add `GeneOrigin` / `GeneScoreSemantics`:

- `DiscoveryBacktestScore`
- `LabelSearchScore`
- `GpuApproxSearchScore`
- `ImportedWarmStart`

Never rank or export mixed-origin genes without a conversion/validation step.

---

### 7. `GeneticStrategyExpert::LabelSearch` is also non-deterministic

The label-search path creates `let mut rng = rand::rng();` and threads it through generation, selection, crossover, mutation, and immigration.

There does not appear to be a first-class typed seed in model settings for this path.

**Risk:** repeated model training can produce different `genetic_portfolio.json` artifacts without a reproducible seed contract.

**Severity:** High.

**Fix direction:** add model-level `genetic_seed` or shared `search_seed`, and export it inside `GeneticArtifact` and `RuntimeArtifactMetadata`.

---

### 8. `GeneticArtifact` is stronger than raw discovery JSON but still not enough for live strategy runtime

`GeneticArtifact` stores population/generation settings, feature columns, backend mode, selection policies, best fitness, portfolio, and optional runtime metadata.

This is useful for model training/runtime artifacts.

But if used as a live strategy artifact, it still needs:

- feature schema hash
- feature generation config
- data split/validation details
- score semantics
- SMC source contract
- backtest/execution settings
- artifact hash
- exact seed/runtime contract

**Risk:** a genetic model artifact may be more structured than discovery export, but still not sufficient as a live strategy contract.

**Severity:** Medium-High.

**Fix direction:** distinguish `GeneticArtifact` from `LiveStrategyRuntimeArtifact`.

---

### 9. `SearchResult` is too small for checkpointing and auditing

`SearchResult` contains only:

- `genes`
- `metrics`

It does not include:

- generation count reached
- feature names/schema
- archive state
- seen memory
- seed
- score semantics
- effective configs
- SMC source mapping
- backend/evaluator details

**Risk:** after search returns, most context needed for reproducibility and resume is lost.

**Severity:** High.

**Fix direction:** introduce `SearchRunResult` or extend result with `SearchRunStateSnapshot`.

---

## Recommended low-level refactor

### Step 1: Create typed search config

```rust
pub struct SearchConfig {
    pub seed: u64,
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub smc: SmcSearchConfig,
    pub evolution_policy: EvolutionSearchPolicy,
    pub archive_policy: ArchivePolicy,
    pub novelty_weight: f64,
    pub checkpoint: Option<SearchCheckpointConfig>,
}
```

No env reads inside search logic.

### Step 2: Create explicit run state

```rust
pub struct SearchRunState {
    pub generation: usize,
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
    pub archive: Vec<ArchivedGene>,
    pub seen_memory: SeenSignatureMemory,
    pub best_score_seen: f64,
    pub stagnant_generations: usize,
}
```

### Step 3: Make all randomness injectable

Functions that currently use internal randomness must accept `rng`:

- `Gene::normalize`
- `enforce_population_smc_ratio`
- any fallback/repair function that samples indicators or flags
- label-search model path
- HPC island path

### Step 4: Add score semantics

```rust
pub enum GeneScoreSemantics {
    BacktestComposite,
    NetProfit,
    LabelSearchMacroF1,
    GpuApproxReturnFitness,
    ImportedUnvalidated,
}
```

### Step 5: Add SMC source contract

```rust
pub enum SmcSourceContract {
    FeatureColumns { mapping: SmcColumnMapping },
    OhlcvDerived,
    Mixed { mapping: SmcColumnMapping },
}
```

## Tests needed

1. `normalize_with_rng_is_deterministic`
2. `enforce_population_smc_ratio_is_seeded`
3. `same_seed_same_search_result`
4. `label_search_same_seed_same_artifact`
5. `search_result_includes_score_semantics`
6. `smc_source_contract_detects_feature_vs_ohlcv`
7. `seen_signature_memory_prevents_duplicate_warm_start`

## Bottom line

The search system has the building blocks for a serious evolutionary strategy engine, including canonical signatures and seen-memory. But determinism is still leaky, score semantics are overloaded, and search state is not preserved deeply enough. The next refactor should make search a resumable, typed, deterministic state machine.
