# 400 × 5 experimental GA completion — matched selector A/B

Date: 2026-10-26. Isolated experimental branch only.

## Execution

- **SUCCESS** GitHub Actions: https://github.com/kosred/Neoethos/actions/runs/40250386021
- Log artifact: `ga-active-percentile-400x5-research` (run artifact 12620244359).
- Actual genetic loop: `crates/neoethos-search/src/genetic/search_engine.rs`.
- Actual input-metric-to-economic-fitness `apply_metrics` (Risky V6, goal context; no trading data).
- Tested 400 individuals x 5 full generations for the opt-in percentile and old default selector with the SAME seed (20261026), 240 feature slots, <=12 terms/gene, novelty weight 0.20, same test-double evaluator.
- Tests verify 5 progress callbacks, 400 candidate evaluation slots per generation, final archive replay and gene/metric lengths. 9 evaluator calls in each (5 generations plus 4 archive replay waves).
- Unit tests: 3 passed, 0 failed, including no-trader sentinel, all-silent population, ties and invalid fitness cases.

## Matched summary

| Criterion | Existing all-finite min/max + novelty | Experimental active percentile + novelty |
|---|---:|---:|
| Success | PASS | PASS |
| Population | 400 | 400 |
| Generations completed | 5 | 5 |
| Generation evaluation slots | 2,000 | 2,000 |
| Total evaluator batches incl. final archive replay | 9 | 9 |
| Returned final archive + population distinct candidates | 1,304 | 1,350 |
| Returned economically active by TEST-DOUBLE trade counts | 1,227 | 1,277 |
| Returned positive net by SYNTHETIC test-double fitness | 1,159 | 1,239 |
| Evolution execution reported seconds | 0.122 | 0.122 |
| Opt-in feature | Off | On |

The experimental selection yields 80 more positive synthetic-metric observations in this **one-seed functionality fixture**, but these are artificial candidate metrics by design. The number is not FX P&L, edge, realized R expectancy, predictive accuracy or an independent time-series backtest. A positive fixture metric is not a profitable tradable strategy. Scores 0.98 vs 0.999 are on different normalization scales and are NOT comparable as profitability.

## Real-market route remained sealed

A separate attempt to run the public `evolve_search_with_progress_and_limits` with the shipped 100-row cTrader EURUSD M1 test fixture exited before the first generation with:

    population evaluation requires a run-bound sealed device route; refusing detached CPU/GPU dispatch

This failure is a correct, intended safety check. It was NOT removed or bypassed. It means this branch does **not** yet have a fully authenticated broker-truth historical evaluation session, and a 100-bar embedded fixture cannot substitute for the required provenance. The complete source-to-broker-price pipeline, CUDA parity, and full discovery promotion funnel remain untested under the new selector.

## Implementation

- Feature opt-in: `neoethos-search/experimental-active-percentile`, default OFF.
- Real host genetic selection routes to `genetic/experimental_percentile.rs` only when compiled with the feature.
- Eligibility requires a finite fitness and trades >=1. Nontraders retain -100. No-trader all-population case stays -100. Exact-score ties get the same midpoint percentile. Nonfinite metrics are excluded.
- Default build retains the pre-existing normalizer unchanged.
- No production or remote broker orders, no risk-limit change, no final chronological OOS exposure.

## Next meaningful evidence

Run baseline/feature selection with the **same certified broker-sourced Stage 1** identity, risk sizing, spread, commissions and swaps. Record realized candidate-level net R, win/loss payoff, PF, drawdown and behaviorally distinct entries. Compare at least 3 seeds and multiple embargoed *inner-training* folds before changing the default. Audit native CUDA normalization independently before claiming CPU/CUDA parity.

**Verdict:** Host GA completes 400 x 5 with the new selector; no historical trading edge certified.
