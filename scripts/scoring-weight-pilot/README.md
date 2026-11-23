# scoring-weight-pilot

Offline Bayesian-style scoring-weight search over existing exit-ladder artifacts.
**This is a scoring-knob search, NOT a gate change.** No Rust code was modified,
no cargo was run, no new backtest/build was produced, and nothing in this pilot
changes production policy, gates, costs or sizing. Both input artifacts are
ResearchOnly saved-candidate counterfactual replays (authorization_issued: false,
promotion-ineligible); their own warnings state variant settings are diagnostic only.

## Inputs
- C:/Users/konst/AppData/Local/neoethos/experiments/eurusd-m15-weekly-20261021-v7-exit-ladder3/exit-policy.json
- C:/Users/konst/AppData/Local/neoethos/experiments/eurusd-m5-risky2-ladder3b/exit-policy.json

Variant pool: exit_policy_variants.candidates[*].variants[*] (label + metrics +
ledger.monthly per variant; two strategy candidates per dataset, 366 exit-policy
variants each).

## Score

    S = w1*monthly_win_rate + w2*median_month_pnl_pct*100 + w3*avg_win_over_avg_loss
      - w4*max(0, -(worst_month_pnl_pct*100) - 2.0) - w5*max_drawdown

weights w1..w5 in [0,1], normalized to sum 1. The artifacts store no avg-win/avg-loss
field, so avg_win_over_avg_loss is derived per variant as
profit_factor * (1 - win_rate) / win_rate.

## Hard constraints (primary run)

    worst_month_pnl_pct >= -0.03
    monthly_win_rate  >=  0.55
    net_profit        >   0

## Method

scikit-optimize is not installed in .git/dukascopy-venv, so the search uses a
Gaussian-process-free random + local search (pure Python stdlib, deterministic seed):
3000 normalized random weight vectors plus the five pure-knob vertices, refined by
local hill-climbing from the 40 best starts. Because S is linear in w and selection
is an argmax over a finite variant set, the joint objective is piecewise-linear
convex, so its maximum over the weight simplex sits at a vertex; the search
evaluates all five vertices and verifies this.

## Key results (full detail in report.json)

Strict gates, top vector w = (0.0, 0.0, 1.0, 0.0, 0.0):
- eurusd_m15_weekly: - max_hold_96_bars_approx_24h_m15 (gene_782139_19): S=0.995, monthly_win_rate=0.5980392156862745, worst_month=-2.23%, aw/al=0.994956, net_profit=1247.3373
- eurusd_m5_risky2: - no variant passes this scenario's gates

Sensitivity (m5 worst-month gate disabled, w4 penalty still active), top vector w = (0.0, 1.0, 0.0, 0.0, 0.0):
- eurusd_m15_weekly: - profit_lock_adaptive_be0.5_give0.2_lock2 (gene_782139_19): S=0.2608, monthly_win_rate=0.6274509803921569, worst_month=-2.87%, aw/al=0.786183, net_profit=1298.6002
- eurusd_m5_risky2: - fixed_sl20_rr3_hold96_trailfalse (gene_489395_7): S=25.9911, monthly_win_rate=0.7087378640776699, worst_month=-1202.8%, aw/al=1.127107, net_profit=92524.3237

Headline findings:
- STRICT GATES: eurusd_m5_risky2 has ZERO feasible variants under the requested hard gates. Among variants passing monthly_win_rate>=0.55 and net_profit>0, the best worst month is -26.93%. No scoring-weight vector can make m5 satisfy a -3% worst-month gate; the gate itself is the binding limit, not the weights.
- STRICT GATES: the m15 feasible pool is 21 variants, all from candidate gene_782139_19 (baseline candidate gene_953632_18 near-misses with best worst month -3.06%).
- Strict optimum: w=(0.0, 0.0, 1.0, 0.0, 0.0) selects m15 max_hold_96_bars_approx_24h_m15 (strategy gene_782139_19; S=0.995; mwr=0.5980392156862745, worst=-2.23%, aw/al=0.994956, net=1247.3373).
- Sensitivity (m5 worst-month gate disabled, w4 penalty still active): joint optimum w=(0.0, 1.0, 0.0, 0.0, 0.0) selects m15 profit_lock_adaptive_be0.5_give0.2_lock2 (aw/al=0.786183, net=1298.6002) AND m5 fixed_sl20_rr3_hold96_trailfalse / gene_489395_7 (aw/al=1.127107, mwr=0.7087378640776699, worst=-1202.8%, net=92524.3237).
- Scale note: median_month_pnl_pct*100 and aw/al dominate S magnitude; max_drawdown is stored as a fraction (0..1), so w5 needs ~100x relative weight to matter. The optimizer therefore concentrates weight on aw/al (and median-month pnl in the relaxed run).
- Sensitivity caution: with the m5 worst-month gate disabled, the formula selects a blowup-profile replay (m5 fixed_sl20_rr3_hold96_trailfalse: worst month -1202.8%, max drawdown 93.27%) - evidence that the -3% worst-month gate, not the scoring weights, is what keeps such profiles out.
- Offline only: both artifacts are ResearchOnly counterfactual replays (authorization_issued=false); this pilot does not change any gate, policy, cost or sizing.

## Re-run

    .git/dukascopy-venv/Scripts/python.exe scripts/scoring-weight-pilot/search_scoring_weights.py

No extra dependencies. Outputs: report.json and this README next to the script.

## Provenance & ledger

Results were appended to issue neo-671 in the Beads ledger as offline pilot
evidence. No gate was changed.
