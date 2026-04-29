# CPU/GPU strategy search optimization notes

This branch starts the search optimization work by removing mathematically redundant genetic candidates before hashing and evaluation.

## Immediate change

`Gene::normalize` now canonicalizes the linear part of each gene:

- indicator indices are clamped into range
- indicator terms are sorted
- duplicate indicator indices are merged by summing their weights
- invalid or non-finite weights are repaired
- invalid thresholds and stop/target values are repaired

This reduces wasted CPU search without removing genuinely different strategies. Equivalent rules now hash the same way, so `SeenSignatureMemory` becomes more effective.

## Next CPU-side steps

1. Normalize every crossover/mutation child before archive insertion.
2. Deduplicate archives by `gene_signature_hash`, not `strategy_id`.
3. Separate actual money metrics from selection scores.
4. Add explicit `execution_lag_bars` to backtesting.
5. Move feature prefiltering inside train folds only.
6. Bound novelty/diversity work so large searches do not become O(n^2).
7. Use staged Monte Carlo validation: small screening first, large perturbation count only for survivors.

## GPU parity target

The fast GPU path should become a stage-1 candidate generator, not final proof. GPU candidates should be converted into the same canonical strategy representation and then validated through the same SMC, SL/TP, spread, commission, timestamp, walk-forward and stress-test logic as CPU/Gene strategies.

## Runtime reporting target

Every search run should report requested backend, actual backend, GPU feature availability, CUDA device count, whether GPU was used, fallback reason, signal backend, backtest backend and final validation backend.
