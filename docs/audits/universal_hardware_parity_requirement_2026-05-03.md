# Universal Hardware Parity Requirement

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Principle

The same canonical search/backtest/evaluation contract must produce the same strategy decisions and equivalent metrics everywhere:

- CPU single-thread
- CPU multi-thread / Rayon
- single GPU
- multi-GPU
- CubeCL CUDA
- tch CUDA
- HPC island mode
- any future accelerator backend

Hardware may change speed. Hardware must not change trading semantics.

## Rule

If two backends receive the same:

- data
- feature matrix
- timestamps
- feature schema
- search config
- execution contract
- seed
- gene population
- validation splits

then they must produce the same:

- signals
- entries
- exits
- trade count
- PnL
- drawdown
- daily/monthly ledgers
- risk diagnostics
- ranking order within documented numeric tolerance

If exact parity is not implemented, the backend must not be called canonical. It must be labeled approximate and must be prevented from final strategy acceptance.

## CPU is the reference

The CPU evaluator should be treated as the reference implementation.

Every GPU/HPC/accelerator backend must prove parity against CPU using fixture tests.

## Multi-GPU parity

Multi-GPU execution must be equivalent to single-device canonical execution.

Splitting work across cards must not change:

- random streams
- segment selection
- candidate order
- archive order
- tie-breaking
- selection pressure
- floating-point reduction semantics beyond documented tolerance

If multi-GPU execution changes the candidate ranking, that difference must be detected and reported.

## RNG requirement

No backend may use hidden OS randomness for canonical search.

All random streams must derive from typed config seed:

```text
base_seed + backend_id + device_id + generation + stream_name + worker_id
```

This applies to:

- CPU search
- label search
- SMC flag repair
- gene normalization
- GPU genome initialization
- island initialization
- parent/survivor selection
- mutation
- crossover
- random immigrants
- segment/window selection
- warm-start sampling

## Backend rejection rule

If a backend cannot implement a feature in the canonical contract, it must fail closed.

Examples:

```text
GpuUnsupportedContract: kill_zones_enabled requires GPU session kernel support
GpuUnsupportedContract: intrabar_fill_policy=RejectAmbiguousBars is not implemented
GpuUnsupportedContract: timestamp_unit=Nanoseconds not normalized before CUDA gap logic
```

It must never silently ignore a CPU behavior.

## Required artifact fields

Every run artifact must include:

- reference_backend
- requested_backend
- actual_backend
- device_ids
- backend_count
- requested_precision
- actual_precision
- fallback_status
- parity_mode
- parity_test_suite_version
- execution_contract_hash
- feature_schema_hash
- timestamp_unit
- seed
- deterministic_stream_contract
- unsupported_features
- approximate_backend_warning, if applicable

## Canonical versus approximate modes

### Canonical mode

A canonical backend can produce final accepted strategies only if it passes CPU parity tests for the requested contract.

### Approximate mode

Approximate backends may be used only for exploration, presearch, warm-start, or candidate generation.

Approximate outputs must be converted and revalidated by canonical evaluation before export.

## Required test matrix

Minimum backend parity matrix:

| Test | CPU | CPU Rayon | Single GPU | Multi-GPU | HPC |
|---|---:|---:|---:|---:|---:|
| signal synthesis | required | required | required | required | required if canonical |
| prior-bar entry | required | required | required | required | required if canonical |
| SL/TP | required | required | required | required | required if canonical |
| spread/commission | required | required | required | required | required if canonical |
| max trades/day | required | required | required | required | required if canonical |
| gap handling | required | required | required | required | required if canonical |
| kill-zones | required | required | required or reject | required or reject | required or reject |
| monthly ledger | required | required | required | required | required if canonical |
| daily drawdown | required | required | required | required | required if canonical |
| ranking stability | required | required | required | required | required if canonical |

## Bottom line

The user requirement is correct and should be non-negotiable: the system must not produce different trading conclusions depending on whether it ran on CPU, one GPU, multiple GPUs, or an HPC island setup. Hardware is an implementation detail, not a different trading strategy.
