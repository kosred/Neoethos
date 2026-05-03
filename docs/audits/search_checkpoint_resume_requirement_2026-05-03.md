# Search Checkpoint / Resume Requirement

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: genetic strategy search resume, long-running discovery, avoiding repeated zero-start search, preserving exploration progress.

## Summary

Strategy search needs a real checkpoint/resume system.

Starting every long-running search from zero wastes compute and can reduce long-term exploration quality. This is especially important for large genetic searches, GPU/HPC runs, and multi-day discovery.

Current evidence shows model training profiles and saved alpha strategy storage exist, but a complete interrupted discovery/search resume mechanism is not yet proven in active code.

The genetic search loop already has the internal state needed for checkpointing. It just needs to be persisted and reloaded safely.

## Why this matters

If every search starts from scratch, then 1,000 runs over 1,000 days may repeatedly rediscover similar early-stage candidates instead of accumulating search knowledge.

A proper resume/checkpoint system allows the search to:

- continue after interruption
- preserve profitable/archive candidates
- preserve diversity memory
- avoid re-evaluating duplicate genes
- explore new areas instead of restarting from the same initial distribution
- recover from power loss, crash, timeout, or manual stop
- compare long-running progress honestly

## Current code evidence

`crates/forex-search/src/genetic/search_engine.rs` already tracks state that is suitable for checkpointing:

- current generation
- current population `genes`
- evaluated metrics
- `profitable_archive`
- archive sequence
- `seen_strategy_ids`
- `SeenSignatureMemory`
- RNG seed / RNG-derived reproducibility contract
- `best_score_seen`
- `stagnant_gens`
- effective `EvaluationConfig`
- SMC gate schedule
- archive mode / thresholds
- survivor and parent selection policy
- novelty weight
- runtime limits

But there is no clear active `SearchCheckpoint`, `save_state`, `load_state`, or `resume_from` path for discovery/search.

## Required artifact: `SearchCheckpoint`

A search checkpoint should be a typed serde artifact, not an ad-hoc JSON dump.

Minimum fields:

```rust
pub struct SearchCheckpoint {
    pub artifact_version: u32,
    pub run_id: String,
    pub search_id: String,
    pub git_commit: String,
    pub symbol: String,
    pub base_timeframe: String,
    pub dataset_fingerprint: String,
    pub feature_schema_hash: String,
    pub effective_feature_names: Vec<String>,
    pub generation_completed: usize,
    pub target_generations: usize,
    pub population_size: usize,
    pub max_indicators: usize,
    pub genes: Vec<Gene>,
    pub gene_metrics: Vec<[f64; 11]>,
    pub profitable_archive: Vec<ArchivedGene>,
    pub seen_signatures: Vec<u64>,
    pub seen_strategy_ids: Vec<String>,
    pub search_seed: u64,
    pub rng_stream_position: Option<u64>,
    pub best_score_seen: f64,
    pub stagnant_generations: usize,
    pub evaluation_config: EvaluationConfig,
    pub search_config_hash: String,
    pub effective_runtime_config: EffectiveRuntimeConfig,
    pub created_at_unix_ms: i64,
    pub updated_at_unix_ms: i64,
}
```

If serializing full RNG state is hard, use deterministic per-generation/per-stage RNG streams derived from:

`search_seed + generation + stream_name + worker_id`

That makes resume deterministic without needing to serialize internal RNG state.

## Required behavior

### 1. Periodic save

The search loop should write a checkpoint:

- after initial population creation
- after each generation
- before returning due to max runtime
- before graceful cancellation
- after final generation as completed state

Use atomic write:

1. write `checkpoint.tmp`
2. fsync if practical
3. rename old checkpoint to `.bak`
4. rename tmp to checkpoint
5. verify load after write in debug/test mode

### 2. Resume mode

Add explicit resume options:

- `resume: false | true`
- `checkpoint_path`
- `resume_policy: exact | extend_generations | reseed_missing | validate_only`

On resume:

- validate artifact version
- validate git commit or allow explicit migration
- validate dataset fingerprint
- validate feature schema hash
- validate config hash
- validate population size / max indicators compatibility
- restore generation, population, archive, seen memory, and scoring state
- continue from `generation_completed + 1`

### 3. Do not confuse checkpoint with portfolio export

A checkpoint is not a live portfolio.

A checkpoint may contain weak, unfinished, or partially evaluated genes. It is only for continuing search.

Final accepted strategies must still go through quality screen, validation, and portfolio artifact export.

## Exploration issue: repeated zero-start search

The user is right that always starting from zero can lead to repeated rediscovery.

However, the fix is not only checkpointing. The system should support three modes:

### A. Exact resume

Continue a run after interruption with the same state.

Use for crash/power loss/timeout.

### B. Warm-start search

Start a new run seeded from previous archive/top candidates, but allow mutation and new data.

Use for daily continuation.

### C. Fresh exploration

Start from scratch with a new seed.

Use for independent search diversity.

Daily workflow should usually be warm-start, not exact resume forever.

## Recommended daily search architecture

For each symbol/timeframe:

1. Load previous `SearchKnowledgeBase`.
2. Seed part of population from previous best/diverse archive.
3. Seed part from recent profitable but not overfit candidates.
4. Seed part from random immigrants.
5. Enforce diversity by canonical gene signature.
6. Run new search on updated data.
7. Save checkpoint each generation.
8. Save final candidate archive.
9. Promote only validated candidates to portfolio artifact.

Suggested population split:

- 30% previous validated/diverse archive
- 20% previous promising candidates
- 20% mutated variants of prior winners
- 30% fresh random immigrants

This avoids both problems:

- not wasting work by starting from zero
- not getting trapped forever in the same local optimum

## Required companion artifact: `SearchKnowledgeBase`

A long-term search memory should be separate from checkpoints.

Checkpoint = exact interrupted run state.

Knowledge base = long-term archive across runs/days.

Recommended fields:

- strategy signature
- gene
- best metrics by dataset window
- validation history
- regimes where it works/fails
- last seen date
- number of times rediscovered
- mutation lineage
- rejection reasons
- OOS/forward-test outcomes

This gives the system a memory of what it has already explored.

## Implementation order

1. Add `SearchCheckpoint` type to `forex-search`.
2. Add `SearchCheckpointWriter` with atomic save/load.
3. Add checkpoint path/config to `DiscoveryConfig` or a new `SearchConfig`.
4. Refactor `evolve_search_with_progress_impl` into a resumable state machine.
5. Save checkpoint at generation boundaries.
6. Add resume validation checks.
7. Add exact-resume test: run 5 generations uninterrupted vs run 2 generations + save + resume 3 generations; results should match when seed/config/data are identical.
8. Add warm-start mode from previous archive.
9. Add long-term `SearchKnowledgeBase`.
10. Only final validated candidates should be exported as portfolio artifacts.

## Tests that must exist

### Exact resume determinism test

Same config, data, and seed:

- run A: 10 generations uninterrupted
- run B: 4 generations, checkpoint, resume 6 generations

Expected:

- same top candidate order
- same archive signatures
- same best score history
- same final metrics within tolerance

### Checkpoint rejection tests

Resume must fail if:

- feature schema hash differs
- dataset fingerprint differs
- config hash differs
- artifact version unsupported
- population/gene dimensions incompatible

### Warm-start exploration test

A warm-start run should include previous archive candidates, but also guarantee fresh random immigrants and diversity limits.

## Bottom line

The project should add search checkpoints. Training/runtime checkpoints are not enough. A long-running genetic strategy search needs both exact resume checkpoints and a long-term search knowledge base, otherwise it wastes compute and repeatedly re-explores similar regions.
