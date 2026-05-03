# Architecture Unification / Duplicate-Code Cleanup Audit

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: CPU/GPU unification, duplicate evaluator/search/training logic, backend abstraction, artifact contracts, and simplification roadmap.

## Summary

The main architectural problem is not lack of GPU code. The main problem is that several parts of the system implement similar concepts through separate paths with different semantics.

This creates drift between:

- CPU evaluator
- CubeCL CUDA evaluator
- tensor GPU presearch
- HPC island discovery
- search scoring
- trade simulation
- validation
- model genetic label-search
- discovery-backed genetic search
- training artifacts
- runtime artifacts

The system needs consolidation around one canonical contract, not more parallel implementations.

## Core principle

There should be one canonical definition of:

- feature schema
- timestamp unit
- signal timing
- execution/backtest contract
- metrics
- risk diagnostics
- candidate score semantics
- artifact schema
- backend/device/precision contract

CPU and GPU should be execution backends for the same contract, not separate strategy engines.

## Current architectural smell

### 1. Multiple evaluators with overlapping responsibility

The repo has at least:

- CPU fast evaluator
- CPU trade simulator
- CubeCL CUDA evaluator
- tensor returns-based GPU discovery
- HPC island returns-based discovery
- label-search genetic scoring

These are not all equivalent. Some use trading/backtest semantics, others use classification or approximate return fitness.

**Problem:** same word, `fitness`, can mean different things.

**Required cleanup:** introduce explicit evaluation modes and score semantics.

---

### 2. GPU code is currently too separate from CPU semantics

GPU should not be a separate strategy model.

The right design is:

```text
CanonicalContract
    -> CpuBackend
    -> GpuBackend
    -> MultiGpuBackend
```

Not:

```text
CpuSearch
GpuSearch
HpcSearch
LabelSearch
all produce Gene but with different meaning
```

**Required cleanup:** make GPU consume the same `ExecutionContract`, `FeatureSchema`, and `BacktestMetrics` types as CPU.

---

### 3. Search and model genetic paths overlap

`forex-search` owns evolutionary strategy discovery.

`forex-models/src/genetic.rs` has `GeneticStrategyExpert`, which can run:

- `DiscoveryBacked`
- `LabelSearch`

Both can produce gene portfolios.

**Problem:** model genetic code and search genetic code overlap but do not fully share score semantics, seeds, artifacts, and checkpoint contracts.

**Required cleanup:** define one shared `GeneCandidate` / `ValidatedStrategyGene` lifecycle.

---

### 4. Artifacts are split by subsystem, not by lifecycle stage

Current artifacts include:

- discovery portfolio JSON
- genetic model artifact
- runtime metadata
- training profile
- ONNX export status
- optimization reports

But they are not unified by lifecycle stage.

**Required cleanup:** classify artifacts by intent:

- `SearchCheckpoint`
- `SearchCandidateArchive`
- `ValidatedStrategyPortfolio`
- `TrainingRuntimeProfile`
- `ModelArtifact`
- `RuntimePredictionArtifact`
- `LiveStrategyRuntimeArtifact`

Do not let one artifact type pretend to be another.

---

### 5. Environment variables spread behavior across modules

Env vars currently control behavior in search, eval, model base, GPU, live/app, and tree models.

**Problem:** hidden runtime behavior makes reproducibility impossible.

**Required cleanup:** env vars should only be used at process boundary, then converted once into typed config.

---

## Recommended target architecture

### Layer 1: Data contract

```rust
pub struct MarketDataFrame {
    pub symbol: String,
    pub timeframe: String,
    pub timestamps: Vec<i64>,
    pub timestamp_unit: TimestampUnit,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}
```

### Layer 2: Feature contract

```rust
pub struct FeatureMatrix {
    pub data: Array2<f32>,
    pub schema: FeatureSchema,
}
```

### Layer 3: Signal contract

```rust
pub trait SignalEngine {
    fn synthesize_signals(&self, features: &FeatureMatrix, gene: &Gene) -> SignalSeries;
}
```

### Layer 4: Execution contract

```rust
pub struct ExecutionContract {
    pub timestamp_unit: TimestampUnit,
    pub initial_equity: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub max_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub kill_zones_enabled: bool,
    pub intrabar_fill_policy: IntrabarFillPolicy,
    pub final_position_policy: FinalPositionPolicy,
}
```

### Layer 5: Backend abstraction

```rust
pub trait EvaluationBackend {
    fn backend_id(&self) -> BackendId;
    fn supports(&self, contract: &ExecutionContract) -> BackendSupport;
    fn evaluate_population(
        &self,
        market: &MarketDataFrame,
        features: &FeatureMatrix,
        genes: &[Gene],
        contract: &ExecutionContract,
    ) -> anyhow::Result<Vec<BacktestResult>>;
}
```

CPU and GPU both implement this trait.

GPU must reject unsupported contracts instead of silently ignoring features.

### Layer 6: Search engine

The search engine should not care whether evaluation runs on CPU or GPU.

```rust
SearchEngine<E: EvaluationBackend>
```

or runtime dynamic dispatch:

```rust
Box<dyn EvaluationBackend>
```

The search engine receives:

- `SearchConfig`
- `FeatureMatrix`
- `MarketDataFrame`
- `ExecutionContract`
- backend

and emits:

- `SearchCheckpoint`
- `CandidateArchive`
- `ValidatedStrategyPortfolio`

## What should be removed or renamed

### Rename approximate paths

`discovery_gpu.rs` and `hpc_gpu_discovery.rs` should not sound like final discovery engines unless they implement full canonical contract.

Suggested names:

- `gpu_approx_presearch.rs`
- `hpc_approx_presearch.rs`

### Remove duplicated score meanings

Do not let `Gene.fitness` mean:

- net profit
- composite score
- macro F1
- approximate tensor returns

without a `GeneScoreSemantics` enum.

### Stop saving weak artifacts as if final

A candidate from approximate GPU/HPC or label search must not be exported as final strategy without canonical validation.

## Why CPU can accelerate to GPU with less chaos

Most of the real work should be backend-neutral:

- feature matrix preparation
- gene normalization
- signal synthesis contract
- execution contract
- metric aggregation
- artifact export
- checkpointing
- validation gates

Only the inner numeric loops need backend-specific acceleration.

That means the project should avoid duplicating whole pipelines for GPU. Instead, move to:

```text
Same pipeline
Same contract
Different backend kernels
```

## Recommended implementation order

1. Define `BacktestMetrics` typed struct.
2. Define `ExecutionContract` typed struct.
3. Define `EvaluationBackend` trait.
4. Wrap current CPU evaluator as `CpuEvaluationBackend`.
5. Wrap CubeCL as `CubeClEvaluationBackend` but make it reject unsupported contracts.
6. Move `fast_evaluate_strategy_core` and `simulate_trades_core` toward one shared simulation state machine.
7. Rename tensor/HPC discovery as approximate presearch.
8. Require canonical validation after approximate presearch.
9. Add `GeneScoreSemantics`.
10. Add artifact lifecycle names and schema hashes.
11. Move env access to config loading only.
12. Add CPU/GPU/multi-GPU parity test matrix.

## Minimal practical cleanup path

The fastest safe path is not a rewrite. It is staged extraction:

### Stage A: Types first

Add typed structs/enums without changing behavior:

- `TimestampUnit`
- `BacktestMetrics`
- `ExecutionContract`
- `GeneScoreSemantics`
- `BackendId`
- `BackendSupport`

### Stage B: Adapters

Create adapter wrappers around existing CPU/GPU evaluators.

Do not rewrite algorithms yet.

### Stage C: Parity tests

Add small deterministic fixtures.

### Stage D: Replace direct calls

Make search call the backend trait instead of calling CPU/GPU functions directly.

### Stage E: Delete or rename duplicate paths

Only after tests prove equivalence.

## Bottom line

The user’s diagnosis is correct: the project needs unification and cleanup more than more GPU-specific code. With a single canonical contract and backend abstraction, the CPU path can become the reference implementation and GPU can accelerate the same logic instead of creating a parallel universe.
