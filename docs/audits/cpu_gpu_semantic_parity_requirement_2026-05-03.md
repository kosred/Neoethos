# CPU / GPU Semantic Parity Requirement

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Principle

Everything that exists in the canonical CPU evaluator/search/backtest contract must exist in the canonical GPU evaluator with the same semantics.

GPU acceleration is allowed to be faster. It is not allowed to silently use a different trading model.

If a GPU path cannot implement the full CPU contract, it must be labeled as approximate presearch and must not be used for final strategy acceptance.

## Required rule

For every canonical CPU behavior, there must be a GPU equivalent:

- signal timing
- prior-bar causal entry
- SL/TP handling
- intrabar SL/TP ordering policy
- spread model
- commission model
- pip size
- pip value per lot
- min hold
- max hold
- max trades per day
- gap handling
- timestamp unit handling
- day/month bucket logic
- kill-zone/session/weekend behavior
- trailing stop behavior
- final open-position policy
- initial equity
- daily drawdown
- monthly consistency
- profit factor
- expectancy
- trade count
- max daily drawdown
- prop/risk diagnostics if used for acceptance

If the CPU evaluator gains a new field, gate, policy, or behavior, the GPU evaluator must either implement it or explicitly refuse to run under that contract.

## Current mismatch examples from audit

### 1. Signal timing mismatch

CPU evaluator uses causal prior-bar entry:

```rust
signals[i - 1]
```

The CubeCL full CUDA backtest kernel currently appears to use:

```rust
signals_flat[signal_base + i]
```

This can create same-bar lookahead in GPU ranking.

### 2. Session/kill-zone mismatch

`simulate_trades_core` contains kill-zone/weekend entry blocking and force-exit behavior.

The CubeCL backtest kernel does not appear to implement the same full session contract.

### 3. Timestamp unit mismatch

CPU and validation paths already have mixed millisecond/nanosecond/day-key risk.

CUDA `timestamp_delta_ms` subtracts raw timestamps and treats the result as milliseconds.

### 4. Approximate GPU discovery mismatch

`discovery_gpu.rs` and `hpc_gpu_discovery.rs` use returns-based tensor fitness. They do not model the canonical SL/TP/spread/commission contract.

Therefore they must be considered approximate presearch, not canonical strategy acceptance.

## Required architecture

Introduce a single canonical execution contract:

```rust
pub struct ExecutionContract {
    pub initial_equity: f64,
    pub timestamp_unit: TimestampUnit,
    pub pip_size: f64,
    pub pip_value_per_lot: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub min_hold_bars: usize,
    pub max_hold_bars: usize,
    pub max_trades_per_day: usize,
    pub gap_threshold_ms: i64,
    pub kill_zones_enabled: bool,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub intrabar_fill_policy: IntrabarFillPolicy,
    pub final_position_policy: FinalPositionPolicy,
}
```

Then require both CPU and GPU to accept the same contract.

## Backend behavior

### Canonical CPU

The CPU evaluator is the reference implementation.

### Canonical GPU

The GPU evaluator must match CPU metrics within explicit tolerances.

Allowed numeric tolerance should be small and documented, for example:

- net profit absolute tolerance: configurable by pip precision and population size
- trade count: exact match required
- win rate: exact or near-exact if trade count matches
- max drawdown: tolerance-bound
- daily/monthly bucket counts: exact match required

### Approximate GPU

Tensor/HPC approximate search may exist, but must be labeled:

- `GpuApproxPresearch`
- `HpcIslandApproxPresearch`

Approximate results cannot be exported as final portfolio strategies. They must go through canonical CPU/GPU validation first.

## Required compile/runtime guard

If `ExecutionContract` requests a behavior not implemented by GPU, the GPU evaluator must fail closed:

```text
GpuUnsupportedContract: kill_zones_enabled requires CPU evaluator or GPU session kernel support
```

It must not silently ignore unsupported CPU behavior.

## Required tests

### CPU/GPU parity tests

1. prior-bar signal timing must match
2. SL/TP trade count must match
3. spread/commission PnL must match
4. max hold/min hold must match
5. max trades per day must match
6. gap threshold must match
7. timestamp unit conversion must match
8. trailing stop behavior must match
9. final open-position policy must match
10. intrabar SL/TP policy must match
11. kill-zone/session behavior must match or GPU must reject contract
12. monthly consistency must match
13. daily drawdown must match
14. archive ranking must not change between CPU and GPU for the same metrics

### Approximate path tests

1. approximate GPU discovery cannot produce final accepted portfolio directly
2. approximate GPU candidate must be revalidated by canonical evaluator
3. artifact must record that candidate came from approximate presearch

## Required artifact fields

Every search/evaluation artifact must record:

- requested backend
- actual backend
- requested precision
- actual precision
- fallback status
- CPU/GPU parity status
- execution contract hash
- timestamp unit
- evaluator version
- unsupported GPU features, if any

## Implementation order

1. Make CPU the reference contract.
2. Add typed `ExecutionContract`.
3. Replace implicit metric array with typed `BacktestMetrics`.
4. Fix CUDA signal timing to prior-bar entry.
5. Normalize timestamps before CPU/GPU evaluation.
6. Add GPU contract rejection for unsupported features.
7. Add CPU/GPU parity fixtures.
8. Only enable full CUDA evaluator for contracts it fully supports.
9. Label tensor/HPC paths as approximate presearch.
10. Require canonical validation before final portfolio export.

## Bottom line

The user's requirement is correct: if the CPU search/backtest contract has a feature, the canonical GPU path must implement the same feature in the same way. Otherwise the GPU is not a faster evaluator; it is a different evaluator. Different evaluators may be useful for exploration, but they must never be confused with final canonical strategy validation.
