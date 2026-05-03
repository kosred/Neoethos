# Forex Search Functional Audit

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: full functional/refactor audit of `crates/forex-search`.

Files inspected:

- `crates/forex-search/Cargo.toml`
- `crates/forex-search/src/lib.rs`
- `crates/forex-search/src/discovery.rs`
- `crates/forex-search/src/eval.rs`
- `crates/forex-search/src/cubecl_eval.rs`
- `crates/forex-search/src/cubecl_ga.rs`
- `crates/forex-search/src/discovery_gpu.rs`
- `crates/forex-search/src/hpc_gpu_discovery.rs`
- `crates/forex-search/src/hpc.rs`
- `crates/forex-search/src/quality.rs`
- `crates/forex-search/src/challenge.rs`
- `crates/forex-search/src/stop_target.rs`
- `crates/forex-search/src/genetic/mod.rs`
- `crates/forex-search/src/genetic/search_engine.rs`
- `crates/forex-search/src/genetic/strategy_gene.rs`
- `crates/forex-search/src/genetic/evolution_math.rs`
- `crates/forex-search/src/genetic/smc_indicators.rs`

No production code was changed by this audit.

## Scope limitation

Several large files were partially truncated by the GitHub connector:

- `discovery.rs`
- `eval.rs`
- `cubecl_eval.rs`
- `discovery_gpu.rs`
- `hpc_gpu_discovery.rs`
- `search_engine.rs`
- `stop_target.rs`

Targeted symbol searches were used to cross-check the truncated areas. This should be treated as a near-complete functional/refactor audit of the search crate, not a literal every-line proof of the truncated tails.

## Core conclusion

`forex-search` contains the most important correctness and performance logic in the bot:

```text
features -> genes -> signals -> backtest -> validation -> portfolio
```

The crate has valuable custom CPU/GPU work, especially CubeCL kernels. These should not be deleted.

The main problem is semantic fragmentation:

- CPU evaluator and CUDA evaluator are not fully equivalent.
- Approximate GPU presearch and canonical backtest search are not clearly separated in types/artifacts.
- Hyperstack/HPC paths are hardware-specific instead of scheduler-driven.
- Search, SMC, quality, backtest, and runtime behavior still read many env vars directly.
- `Gene` mixes candidate, evaluated, validated, and portfolio-stage responsibilities.
- Metrics are passed as implicit `[f64; 11]` arrays.

The target should be:

```text
one canonical search/evaluation contract
+ GPU-first backend kernels
+ explicit approximate-vs-canonical modes
+ typed policies
+ deterministic RNG
+ provenance-rich artifacts
```

## Feature flags / `Cargo.toml`

The GPU path is feature-gated:

```toml
gpu = ["tch", "cubecl", "half", "rand_distr"]
```

This matters because behavior changes significantly depending on whether `gpu` is enabled.

The system should not hide this behind identical function names. Runtime artifacts should record:

- crate features used
- backend used
- precision used
- canonical vs approximate mode
- fallback/degraded reason

## `lib.rs`

`lib.rs` defines the public module surface and contains a large inline fallback `discovery_gpu` module when `gpu` is not enabled.

Problem:

The same public name can mean different semantics depending on build features.

Example:

```text
run_gpu_discovery with gpu feature = GPU/tch path
run_gpu_discovery without gpu feature = CPU fallback path
```

Target:

```rust
SearchExecutionMode::CanonicalCpu
SearchExecutionMode::CanonicalCubeClCuda
SearchExecutionMode::GpuTensorApproxPresearch
SearchExecutionMode::CpuFallback
```

The fallback module should move out of `lib.rs` or be replaced by explicit runtime mode handling.

## `discovery.rs`

`discovery.rs` is the high-level discovery orchestration layer.

It contains:

- `DiscoveryConfig`
- `DiscoveryResult`
- `DiscoveryRunProfile`
- progress events
- recent-history trimming
- feature prefiltering
- stage-1 funnel
- GA invocation
- candidate finalization
- quality screen
- trade logging
- portfolio selection
- export/profile logic
- CPCV/walk-forward settings

Positive:

`DiscoveryConfig::from_settings` already maps many typed settings from `forex_core::Settings`.

Problem:

The file still reads production semantics directly from env vars, including:

- `FOREX_BOT_PREFILTER_TOP_K`
- `FOREX_BOT_PREFILTER_INSAMPLE`
- `FOREX_BOT_FUNNEL_STAGE1_PCT`
- `FOREX_BOT_BACKTEST_INITIAL_EQUITY`

These should be typed fields in resolved search/evaluation/validation config.

### Feature prefilter issue

`prefilter_features` ranks features by correlation with one-bar forward returns over an in-sample prefix. Restricting to an in-sample prefix is good.

But prefiltering changes feature columns. The final artifact must preserve:

- original feature schema
- effective feature schema
- original-to-effective mapping
- feature schema hash
- prefilter policy hash

Otherwise a saved gene using index `17` may not be safely interpretable later.

### Stage-1 funnel issue

Stage-1 uses a recent fraction of rows.

This is useful for speed, but it must be recorded as an explicit funnel policy. The artifact must say whether a candidate was:

```text
Stage1Candidate
FullHistoryCandidate
ValidatedStrategyGene
PortfolioStrategy
```

### Quality screen integration

`discovery.rs` uses:

- `StrategyQualityAnalyzer`
- `simulate_trades_core`
- `LoggedStrategyTrades`

This means `quality.rs` is part of final candidate screening, not just a helper.

Problem:

The quality-screen path still reads initial balance from env. This should be part of `BacktestPolicy` / `ValidationPolicy`.

### Regime robustness issue

`validate_regime_robustness` exists and checks PnL by regime columns. This is useful, but it depends on feature names and assumes specific regime column names.

This should move to validation with a typed feature schema:

```rust
RegimeRobustnessPolicy
RegimeRobustnessReport
```

## `eval.rs`

`eval.rs` is currently the canonical CPU evaluator and fallback orchestrator.

It contains:

- `BacktestSettings`
- `fast_evaluate_strategy_core`
- `simulate_trades_core`
- CPU signal synthesis
- CPU/GPU fallback orchestration
- metric array generation

Positive:

The CPU evaluator has a correct causal entry rule:

```rust
signals[i - 1]
```

The comment explicitly says the trade acts on the prior bar signal and fills at the current bar close. This should be the canonical rule.

Problem:

`eval.rs` owns too many concepts:

- signal synthesis
- backtest semantics
- trade simulation
- GPU fallback policy
- metric aggregation
- env-driven settings

Suggested split:

```text
signal/synthesis.rs
evaluation/backtest_settings.rs
evaluation/canonical_backtest.rs
evaluation/trade_simulation.rs
evaluation/metrics.rs
evaluation/runtime.rs
```

### Env debt

`BacktestSettings` reads:

- `FOREX_BOT_BACKTEST_INITIAL_EQUITY`
- `FOREX_BOT_BACKTEST_MAX_MONTH_BUCKETS`
- `FOREX_BOT_RUST_THREADS`

These should become typed config.

### Metric schema issue

Metrics are returned as `[f64; 11]`.

This is unsafe. It should become:

```rust
EvaluationMetrics {
    net_profit,
    sharpe,
    peak_equity,
    max_drawdown,
    win_rate,
    profit_factor,
    expectancy,
    reserved,
    trade_count,
    consistency,
    max_daily_drawdown,
}
```

or equivalent.

## `cubecl_eval.rs`

This file contains valuable custom CubeCL/CUDA kernels:

- `synthesize_signals_kernel`
- `backtest_population_kernel`

These should be preserved.

### P0 correctness issue: CUDA current-bar entry

The CUDA backtest kernel enters with:

```rust
signals_flat[signal_base + i]
```

The CPU canonical evaluator enters with:

```rust
signals[i - 1]
```

This means the full CUDA backtest can use same-bar signals while the CPU evaluator uses prior-bar signals. That is a critical CPU/GPU semantic mismatch and can create lookahead in GPU search results.

Required fix:

The CUDA backtest kernel must follow the canonical prior-bar rule.

Required test:

```text
same data + same signals + same settings
CPU canonical evaluator == CubeCL CUDA evaluator within tolerance
```

### Timestamp issue

`timestamp_delta_ms` subtracts raw timestamps and treats the delta as milliseconds.

If data timestamps are ns/us/sec, gap detection is wrong. This must use the timestamp unit contract from `forex-data`.

### Env/runtime issue

The file reads env for:

- CUDA kernel enable/disable
- CUDA device id
- kernel units
- precision

These should come from scheduler/runtime config.

## `cubecl_ga.rs`

This file contains the custom CUDA reproduction kernel:

- `blend_mutate_kernel`

This should be preserved.

Positive:

Most random decisions happen on the caller side via `&mut R`, so the path can be deterministic with a seeded RNG.

Problem:

The file still reads env vars:

- `FOREX_BOT_SEARCH_CUDA_REPRO_KERNEL`
- `FOREX_BOT_SEARCH_GPU_KERNEL_UNITS`

These should come from scheduler/runtime config.

Target role:

```rust
SearchReproductionKernel::CubeClCuda
```

## `discovery_gpu.rs`

This is a tensor/tch GPU discovery path.

Important design note in the file says it uses returns-based fitness and does not model:

- SL/TP
- spread
- commission

Therefore it is not equivalent to the CPU GA / CubeCL canonical backtest path.

Target classification:

```rust
GpuTensorApproxPresearch
```

It must not produce final validated strategies without canonical validation afterward.

Positive:

`GpuDiscoveryConfig` now has optional seed. `make_rng` can produce deterministic genomes/segments when seed is supplied.

Problems:

- fallback/fail-fast still reads `FOREX_BOT_REQUIRE_GPU`
- result type does not strongly distinguish approximate candidates from validated strategies
- `timeframes` are synthetic `tf_0`, `tf_1`, etc.
- feature schema/provenance is weak

Required output type:

```rust
ApproxSearchCandidateBatch
```

followed by canonical validation:

```rust
ValidatedStrategyGene
```

## `hpc_gpu_discovery.rs`

This is an HPC/island model path optimized for 8xA6000 / NVLink / Hyperstack style hardware.

It contains useful concepts:

- island model
- elite migration
- per-GPU population shards
- topology-aware execution
- CUDA reproduction kernel reuse

But the file should not remain as a special production path.

Problems:

- hardcoded HPC mode requirement
- relies on `hpc.rs`
- assumes NVLink pair topology
- unseeded `rand::rng()` in island initialization, selection, evolution, and segment building
- returns approximate `GpuDiscoveryResult`
- visible code does not show final canonical validation despite comments mentioning multi-fidelity screening

Target:

Move useful concepts into generic scheduler:

```rust
TopologyAwareIslandScheduler
IslandMigrationPolicy
GpuWorkShard
PeerLinkTopology
```

Then retire the Hyperstack-specific file.

## `hpc.rs`

This file is explicitly Hyperstack N3 specific.

It assumes:

- 8x RTX A6000
- 48GB VRAM per GPU
- 252 physical cores / 504 logical threads
- 464GB RAM
- two NUMA sockets
- hardcoded CPU ranges
- hardcoded NVLink pairs
- fixed chunk/population sizing

This confirms that it should not be the future scheduler foundation.

What to preserve:

- hardware detection idea
- VRAM-aware planning
- CPU/GPU topology concept
- NUMA affinity idea
- peer-link concept
- chunk/population sizing concept

What to replace it with:

```text
runtime/hardware_profile.rs
runtime/hardware_probe.rs
runtime/topology.rs
runtime/device_assignment.rs
runtime/scheduler.rs
runtime/work_unit.rs
```

After migration, this file can be removed or kept only as a preset/test fixture.

## `quality.rs`

`quality.rs` is useful and should be preserved as validation logic.

It calculates:

- win rate
- profit factor
- Sharpe
- Sortino
- Calmar
- total return
- drawdown
- expectancy
- Kelly
- statistical significance
- monthly consistency
- trades/month
- Monte Carlo 95% worst drawdown
- risk of ruin
- quality score
- edge flag

Problems:

- env-driven monthly policy:
  - `FOREX_BOT_PROP_MIN_TRADES_PER_MONTH`
  - `FOREX_BOT_TRADING_DAYS_PER_MONTH`
- unseeded Monte Carlo via `rand::rng()`
- no explicit validation provenance

Target:

```rust
QualityValidationPolicy
MonteCarloValidationPolicy
StrategyQualityReport
```

## `challenge.rs`

`challenge.rs` is small and typed, but overlaps with `forex-core/src/domain/risk.rs`.

Important issue:

`optimize_risk_allocation` ends with:

```rust
optimal_risk.clamp(0.001, 0.015)
```

This can force a minimum risk even when daily/total drawdown room is exhausted.

For prop/challenge accounts, exhausted risk room should return zero risk or block trading.

Target:

Move/merge challenge rules into core domain risk and validation policy. Retire `search/challenge.rs` after migration.

## `stop_target.rs`

This file is useful but too broad.

It contains:

- volatility estimators
- expected shortfall
- Hurst estimation
- ADX/regime inference
- ATR
- swing/structure distances
- stop/target inference
- RR policy

Target split:

```text
risk/stop_target/settings.rs
risk/stop_target/volatility.rs
risk/stop_target/tail_risk.rs
risk/stop_target/regime.rs
risk/stop_target/structure.rs
risk/stop_target/infer.rs
```

It should consume a typed stop-target policy from config, not hardcoded/default settings hidden in search.

## `genetic/mod.rs`

This is a clean small re-export module. Keep this style.

## `genetic/strategy_gene.rs`

`Gene` currently mixes multiple lifecycle stages:

- candidate genome
- strategy identifier
- SMC flags
- SL/TP settings
- evaluation metrics
- generation metadata
- consistency/slice metrics

Target split:

```rust
CandidateGene
EvaluatedGene
ValidatedStrategyGene
PortfolioStrategy
```

Problem:

`Gene::normalize` uses `rand::rng()` internally. This breaks deterministic search even when the outer GA uses a seed.

Required fix:

Pass RNG from caller or use a deterministic repair policy.

Env debt:

Cost model and SMC evaluation defaults read env vars. These should move to typed resolved config.

## `genetic/evolution_math.rs`

Positive:

- selection policies are typed
- parent/survivor selection mostly accepts caller RNG
- mutation/crossover mostly accepts caller RNG
- signature hash / seen memory concept is useful

Problems:

- `SeenSignatureMemory::from_env()` is env-driven
- `unique_candidate_or_retry` calls `Gene::normalize`, which currently has unseeded RNG
- metrics remain implicit `[f64; 11]`

Target:

```rust
SearchMemoryPolicy
SeenSignatureStore
EvaluationMetrics
```

## `genetic/smc_indicators.rs`

This file bridges feature/data SMC into search SMC gates.

Positive:

- `derive_smc_arrays` is deterministic from OHLCV
- `build_smc_arrays` can use feature columns when present
- most random SMC functions accept caller RNG

Problems:

- `SmcSearchConfig::from_env()` is env-driven
- `enforce_population_smc_ratio` uses internal `rand::rng()`
- `find_feature_column` uses fuzzy `contains`, which can match the wrong feature column
- source provenance is missing: derived OHLCV vs feature columns vs mixed

Target:

```rust
SmcSearchPolicy
SmcFeatureSource
SmcColumnMapping
SmcGatePolicy
```

## `genetic/search_engine.rs`

This is the canonical GA loop.

Positive:

- `EvalDataCache` avoids recomputing stable arrays
- seeded RNG exists via `FOREX_BOT_SEARCH_SEED`
- evaluation goes through `evaluate_population_core`
- archive logic exists
- novelty search exists
- SMC gate schedule exists

Problems:

- search policy is mostly env-driven
- `month_day_indices` assumes timestamp milliseconds
- `SmcSearchConfig::from_env()` and `SeenSignatureMemory::from_env()` are used directly
- archive mode/cap/thresholds are env-driven
- novelty weight is env-driven
- metrics are implicit arrays
- search checkpoint/resume is not a complete state object

Target:

```rust
SearchPolicy
SearchState
SearchCheckpoint
SearchArchivePolicy
NoveltyPolicy
SmcGateSchedule
SearchProgressEvent
```

## Required canonical mode separation

The repo should explicitly distinguish:

```rust
CanonicalCpuBacktest
CanonicalCubeClCudaBacktest
HybridCudaSignalCpuBacktest
GpuTensorApproxPresearch
HpcIslandApproxPresearch
```

Only canonical modes should produce final validated strategy artifacts.

Approximate modes may produce candidates, but every candidate must go through canonical validation before portfolio/live use.

## Required GPU-first scheduler integration

Search should not decide device/precision/kernel toggles internally.

Target flow:

```text
ResolvedRuntimeConfig
-> Scheduler
-> WorkUnit assignment
-> BackendKernel execution
-> Artifact provenance
```

Search work units:

```rust
SearchCandidateBatchWorkUnit
SignalSynthesisBatchWorkUnit
BacktestPopulationWorkUnit
SearchReproductionBatchWorkUnit
ApproxPresearchBatchWorkUnit
ValidationCandidateBatchWorkUnit
```

## P0 findings

1. CUDA full backtest uses current-bar signal while CPU canonical uses prior-bar signal.
2. Approximate GPU/HPC presearch can be confused with canonical discovery unless artifact types are separated.
3. Timestamp unit assumptions remain in search/eval/month/day/gap logic.
4. Feature prefilter changes column schema without enough artifact mapping.
5. `Gene::normalize` and some SMC/HPC paths use unseeded RNG.
6. Hyperstack-specific logic should be replaced by generic scheduler.

## P1 findings

1. Env vars still control search/eval/quality/runtime behavior.
2. `[f64; 11]` metrics should become typed.
3. `Gene` should be split by lifecycle stage.
4. `quality.rs` should become typed validation policy/report.
5. `challenge.rs` should merge into core domain risk/validation.
6. `stop_target.rs` should be split into smaller risk modules.
7. Search checkpoint/resume should capture full state, not only final artifacts.

## Required tests

Add tests for:

- CPU vs CubeCL CUDA backtest parity
- prior-bar signal timing in CUDA
- gap detection with ms/ns/us timestamps
- signal synthesis CPU vs CUDA parity
- GPU reproduction deterministic with seeded RNG
- `Gene::normalize` deterministic repair
- SMC population ratio deterministic with seed
- SMC column mapping exact schema match
- approximate presearch candidates must be canonical-validated before export
- feature prefilter original/effective schema mapping
- quality Monte Carlo deterministic with seed
- challenge risk returns zero/block when drawdown room is exhausted

## Proposed target structure

```text
forex-search/src/
  lib.rs
  search/
    mod.rs
    config.rs
    orchestrator.rs
    state.rs
    checkpoint.rs
    archive.rs
    progress.rs
  genetic/
    mod.rs
    candidate.rs
    lifecycle.rs
    selection.rs
    mutation.rs
    seen_memory.rs
    smc_policy.rs
  signal/
    mod.rs
    synthesis.rs
    smc_gate.rs
  evaluation/
    mod.rs
    settings.rs
    metrics.rs
    cpu_backtest.rs
    trade_simulation.rs
    parity.rs
  kernels/
    mod.rs
    cubecl_eval.rs
    cubecl_reproduction.rs
  presearch/
    mod.rs
    tensor_gpu.rs
    island.rs
  validation/
    mod.rs
    quality.rs
    challenge.rs
    regime.rs
  risk/
    stop_target/
      mod.rs
      volatility.rs
      tail_risk.rs
      regime.rs
      structure.rs
      infer.rs
```

## Bottom line

`forex-search` has strong pieces, including valuable custom GPU kernels. The next step is not deletion or rewrite.

The next step is to make semantics explicit:

```text
candidate generation -> signal synthesis -> canonical backtest -> validation -> portfolio
```

with typed policies, deterministic RNG, CPU/GPU parity tests, and clear separation between approximate GPU presearch and canonical validated strategies.

The first production fix should be the CUDA current-bar vs CPU prior-bar signal mismatch.
