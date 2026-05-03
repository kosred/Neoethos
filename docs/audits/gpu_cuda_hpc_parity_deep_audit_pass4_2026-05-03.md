# GPU / CUDA / HPC Parity Deep Audit — Pass 4

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: CubeCL CUDA evaluator, tensor GPU discovery, HPC island discovery, CPU/GPU parity, precision/fallback semantics, deterministic hardware search.

## Summary

The repo contains multiple GPU-related paths, but they do not all mean the same thing.

There are at least three separate GPU/HPC concepts:

1. CubeCL CUDA evaluator inside the normal population evaluator.
2. Tensor/tch GPU discovery in `discovery_gpu.rs`.
3. HPC island-model GPU discovery in `hpc_gpu_discovery.rs`.

Only the CubeCL evaluator attempts to match the CPU backtest evaluator. The tensor GPU and HPC island discovery paths are approximate returns-based search engines and should be treated as presearch/warm-start generators unless followed by canonical validation.

## Findings

### 1. CubeCL full backtest kernel appears to use same-bar signal entry

In the CUDA backtest kernel, the entry branch reads:

```rust
let s = signals_flat[signal_base + i];
```

The CPU fast evaluator and trade simulator use causal prior-bar entry:

```rust
let s = signals[i - 1];
```

**Risk:** full CUDA backtest can use same-bar information while CPU uses prior-bar causal signals. This can create lookahead and CPU/GPU ranking mismatch.

**Severity:** Critical.

**Fix direction:** CUDA full backtest kernel must use `signals_flat[signal_base + i - 1]` for entry, matching CPU. Add parity tests.

---

### 2. CubeCL evaluator does not encode full session/kill-zone behavior

The CubeCL kernel parameters include max hold, min hold, max trades/day, gap threshold, timestamps, trailing, spread, commission, and pip value.

It does not appear to include the full kill-zone/session/weekend logic found in `simulate_trades_core`.

**Risk:** GPU backtest can disagree with the trade simulator and validation/logging path when session rules matter.

**Severity:** High.

**Fix direction:** either add session/kill-zone parity to CUDA kernel or mark CUDA kernel as limited-contract evaluator unless session features are disabled.

---

### 3. CUDA timestamp delta assumes timestamp unit compatibility

`timestamp_delta_ms` subtracts raw timestamp values and treats the result as milliseconds.

If upstream timestamps are nanoseconds, the CUDA gap logic sees huge deltas, saturates to i32, and can force incorrect gap behavior.

**Risk:** GPU backtest gap behavior changes depending on timestamp unit.

**Severity:** High.

**Fix direction:** timestamp unit must be explicit. Convert timestamps to milliseconds before passing to CUDA backtest, or pass a timestamp-unit enum and normalize once.

---

### 4. CUDA evaluator behavior is env-driven

CubeCL evaluator behavior is controlled by env flags such as:

- `FOREX_BOT_SEARCH_EVAL_CUDA_KERNEL`
- `FOREX_BOT_SEARCH_BACKTEST_CUDA_KERNEL`
- `FOREX_BOT_SEARCH_EVAL_PRECISION`
- `FOREX_BOT_TRAIN_PRECISION`
- `FOREX_TRAIN_PRECISION`
- kernel unit flags
- CUDA device env flag

**Risk:** backend/precision/kernel behavior can change without typed config or export.

**Severity:** High.

**Fix direction:** make evaluator backend, device, precision, and kernel toggles typed config and export them in the search/evaluation contract.

---

### 5. No clear CUDA-vs-CPU parity tests were found

Repo search did not reveal dedicated tests proving that `try_evaluate_population_cuda` matches CPU evaluation across:

- causal signal shift
- SL/TP
- max hold
- min hold
- max trades/day
- gap threshold
- spread/commission
- monthly consistency
- daily drawdown

**Risk:** GPU path can silently diverge and change strategy ranking.

**Severity:** Critical.

**Fix direction:** add parity tests with fixed small datasets and known signals/genes. GPU tests can be conditional, but CPU parity fixtures must exist.

---

### 6. Tensor GPU discovery is approximate presearch, not canonical search

`discovery_gpu.rs` explicitly states its fitness is returns-based:

- `action * (close_next - open_next) / open_next`
- flat 0.0002 cost

It does not model SL/TP, spread, commission, or the full backtest contract.

**Risk:** tensor GPU output can be mistaken for final strategy candidates.

**Severity:** High.

**Fix direction:** label results as `GpuApproxPresearch`. They must go through canonical CPU/CubeCL evaluator before promotion.

---

### 7. HPC island discovery is also approximate presearch

`hpc_gpu_discovery.rs` advertises multi-fidelity screening in comments, but the visible `run_island_model_discovery` returns top elites/genomes from island evolution.

The fitness path uses tensor returns/Sortino/consistency/penalties, not the canonical SL/TP evaluator.

**Risk:** HPC island results can be over-trusted as final validated strategies.

**Severity:** High.

**Fix direction:** treat HPC island output as warm-start candidates only. Add an explicit canonical validation stage before any export/promotion.

---

### 8. HPC island discovery is not fully deterministic

The HPC island path uses internal `rand::rng()` in:

- island population initialization
- elite selection
- evolution generation
- historical segment selection

This bypasses `GpuDiscoveryConfig.seed`.

**Risk:** same config/data/seed can produce different HPC results.

**Severity:** High.

**Fix direction:** derive per-island RNGs from `base_seed + island_id + generation + stream_name`. Pass RNG explicitly everywhere.

---

### 9. GPU discovery exports genomes, not canonical genes

Tensor GPU/HPC discovery returns `genomes: Vec<Vec<f32>>`, not `Gene` objects with indicator indices, weights, SMC flags, SL/TP, and evaluation contract.

**Risk:** conversion from genome to live/discovery gene semantics is not explicit.

**Severity:** Medium-High.

**Fix direction:** introduce `GpuGenomeCandidate` and a conversion/validation path:

`GpuGenomeCandidate -> CandidateGene -> CanonicalEvaluation -> DiscoveryPortfolioArtifact`

---

### 10. Precision fallback can alter behavior

CUDA signal kernel can attempt bf16 depending on requested precision and fall back to fp32. Tensor paths use f32.

**Risk:** precision behavior is not fully reflected in exported artifacts or candidate ranking reports.

**Severity:** Medium-High.

**Fix direction:** export actual precision used per stage, not only requested precision.

## Required terminology

Use these labels consistently:

- `CanonicalCpuEvaluator`
- `CanonicalCubeClCudaEvaluator`
- `GpuApproxPresearch`
- `HpcIslandApproxPresearch`
- `GpuGenomeCandidate`
- `CandidateGene`
- `ValidatedStrategyGene`

## Recommended implementation order

1. Fix CUDA signal shift: use prior-bar signal in full backtest kernel.
2. Add CPU-vs-CUDA parity tests for simple deterministic fixtures.
3. Add timestamp unit normalization before CUDA gap logic.
4. Add typed `EvaluatorBackendConfig` and remove env-driven evaluator behavior.
5. Export actual backend, device, precision, and fallback status per search stage.
6. Rename or label tensor/HPC results as approximate presearch.
7. Add canonical validation stage for every GPU/HPC candidate before export.
8. Make HPC island RNG deterministic with per-island seeded streams.
9. Add GPU genome-to-gene conversion contract.
10. Only canonical evaluated genes should enter portfolio export.

## Required tests

1. `cuda_backtest_uses_prior_bar_signal`
2. `cuda_cpu_parity_basic_sl_tp`
3. `cuda_cpu_parity_max_trades_per_day`
4. `cuda_cpu_parity_gap_threshold_ms`
5. `timestamp_unit_normalized_before_cuda_gap_logic`
6. `gpu_approx_presearch_requires_canonical_validation`
7. `hpc_island_same_seed_same_elites`
8. `actual_gpu_backend_and_precision_exported`

## Bottom line

GPU acceleration is valuable, but only if the semantics are controlled. The project should separate approximate GPU exploration from canonical evaluation. The CubeCL CUDA evaluator must match CPU exactly before it is trusted for search ranking. Tensor GPU/HPC discovery should be used as presearch/warm-start unless every candidate is revalidated under the canonical evaluator.
