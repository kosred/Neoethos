# Scoring selection-pressure research — October 2026

Scope: branch exp/pre-oos-funnel-ci-20261025 only. No production scorer changes, no changes to master, no final OOS.

## Evidence

- Successful, executable, exact Rust scoring probe: https://github.com/kosred/Neoethos/actions/runs/40245356620
- Extended novelty-normalization regression: https://github.com/kosred/Neoethos/actions/runs/40245497847
- Reproducible source: crates/neoethos-gpu-contracts/examples/scoring_landscape_probe.rs
- Prior **historical** analysis of actual July EURUSD M15 22,269-quality-survivor population: docs/search-objective-audit-2026-08-08.md. Treat this as historical analysis, not present-day live verified edge.

## Exact PropFirm V4 sensitivity, other metric slots frozen

| Isolated delta | Fitness delta |
|---|---:|
| +$100 net profit | +0.0007500 |
| +1 percentage point peak drawdown | -0.1500000 |
| +1 percentage point worst-day drawdown | -0.1000000 |
| Profit factor from 0.9999 to 1.0001 | +0.2500325 |

These are mathematical sensitivity derivatives, not economically consistent executed candidate tapes. Because the absolute net-profit divisor is 20,000 and coefficient is 0.15, moving from $20 to $120 net barely matters next to the drawdown terms. The PF=1.0 discontinuity arises from the below-one penalty `-0.25/pf` versus the above-one reward `0.15 * (pf-1)*0.5`. The jump can change selection over a tiny PF perturbation.

Counterfactual weight-grid: when reward was multiplied 10x AND peak/worst-day penalties reduced to weights 4/2, the higher-profit constructed candidate beat the smoother one. At unchanged 15/10 DD weights, even 20x net weighting did not reverse the illustrative ranking. None of these weights were installed into the trading system.

## Risky V5 and V6 are different and must not be incorrectly treated as PropFirm

- Legacy Risky V5 half-Kelly objective rewards estimated log-growth from win rate, profit factor and trade count. In the isolated sensitivity fixture changing only positive net-profit dollars from $20 to $2,000 left the fitness bit-identical; it uses the *negative-net edge gradient* only. Actual net changes normally co-vary with other metrics, but this exposes its independent sensitivity.
- Risky V6 growth-goal fitness does reward actual realized net-profit ratio; with 10,000 simulated starting equity and fixed 180-day span, +$20 generated score 0.0006428986, and +$2,000 score 0.0578144647. It is drawdown-agnostic unless peak DD >=100%, a deliberate distinction; separate risk gates remain required.
- The actual route chooses V6 only if the goal is installed. Do not assume a particular run's actual objective without its sealed run profile.

## Host-GA novelty sentinel inversion — exact numerical reproduction

Source: crates/neoethos-search/src/genetic/search_engine.rs, late scoring loop with `min_fit` / `max_fit` over *all finite* scored candidates, and `scored[i].0 = (1-novelty_weight)*normalized_fitness + novelty_weight*normalized_novelty`.

Fixture: better-profit V6 fitness 0.0578144647 / novelty 0.1; lower-profit V6 fitness 0.0006428986 / novelty 0.9; one zero-trade sentinel with fitness -100, and configured novelty weight 0.2. These are synthetic metric counterfactuals, not a historical price backtest.

| Population normalization | Better-profit blended | Lower-profit blended | Winner |
|---|---:|---:|---|
| Including zero-trade -100 in min-max | 0.8222222222 | 0.9995428917 | Lower profit / greater novelty |
| Excluding -100 from fitness min-max (same novelty) | 0.8222222222 | 0.2000000000 | Greater profit |

**Mechanism:** -100 makes the fitness denominator ~100 rather than ~0.057, destroying economically meaningful differences among viable genes, while novelty maintains its full 20% range. This result is specific to the host-GA normalization logic and the fixture. It does not demonstrate live profitability or prove that GPU-native follows that path.

## Follow-up engineering priorities

1. For a **branch-only experimental** version, compute fitness normalization bounds from finite, actually trading candidates, while retaining the no-trade penalty outside the normalization. Test no-trade-only, mixed, negative-profit and nonfinite cases. Keep original baseline intact for A/B.
2. Audit native CUDA scoring/rejection/novelty path separately: it uses different kernels and V2 economic rejection. Do not assume CPU/GPU parity; do not silently promote a host-only fix.
3. Evaluate a continuous profit-factor function across PF=1.0, preserving no-edge penalties and safety gates.
4. Compare PropFirm candidate selection with profitability expressed relative to the actual account equity instead of a fixed $20,000 divisor, and freeze drawdown eligibility separately from ranking. Do not simply delete safety penalties.
5. Reuse the historical payoff-aware measurement script `scripts/measure_selection_objective.py` if the full July `.quality.json` file is supplied; unavailable in public GitHub checkout. Do not fabricate portfolio outcomes.
6. Only then run a new *pre-OOS* budgeted GA on broker-authoritative datasets, with matched seeds/epochs/costs. Final OOS remains untouched.

**Verdict:** Concrete host-ranking inversion and PropFirm fitness discontinuity reproduced; no proven broker-real edge yet.
