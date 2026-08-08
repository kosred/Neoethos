#!/usr/bin/env python3
"""Measure what the discovery pipeline selected vs what a payoff-explicit
net-expectancy objective would have selected, from a run's artifacts.

Companion to docs/search-objective-audit-2026-08-08.md (slice 4).
Read-only: streams the (potentially multi-GB, pretty-printed)
<SYMBOL_TF>.quality.json line-wise and never loads it whole.

Usage:
    python scripts/measure_selection_objective.py EURUSD_M15
    python scripts/measure_selection_objective.py EURUSD_M15 --store D:\\path\\to\\cache\\discovery

Definitions:
    payoff  = pf * (1 - wr) / wr        (derived; old artifacts lack payoff_ratio)
    E[R]    = wr * payoff - (1 - wr)    (net expectancy per trade, in R)
    conf    = min(1, sqrt(trades) / 10)
    proposed score = E[R] * conf        (NO volume term — that is the point)
"""

import argparse
import json
import math
import os
import sys

KEYS = {
    "strategy_id",
    "total_trades",
    "win_rate",
    "profit_factor",
    "payoff_ratio",
    "max_drawdown_pct",
    "expectancy",
    "avg_monthly_return_pct",
    "trades_per_month",
    "quality_score",
    "net_profit",
    "monthly_win_rate",
}


def stream_records(path):
    """Yield the scalar prefix of each StrategyMetrics record.

    The artifact is a pretty-printed JSON array; every scalar field sits on
    its own line and precedes the record's huge equity_curve array, so a
    line-wise scan recovers everything needed without parsing 4 GB of JSON.
    """
    rec = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith('"strategy_id"'):
                if rec is not None:
                    yield rec
                rec = {}
            if rec is None or not s.startswith('"'):
                continue
            ci = s.find('":')
            if ci < 0:
                continue
            key = s[1:ci]
            if key not in KEYS:
                continue
            val = s[ci + 2 :].strip().rstrip(",")
            if key == "strategy_id":
                rec[key] = val.strip('"')
            else:
                try:
                    rec[key] = float(val)
                except ValueError:
                    rec[key] = float("nan")
        if rec is not None:
            yield rec


def derived_payoff(rec):
    p = rec.get("win_rate", float("nan"))
    pf = rec.get("profit_factor", float("nan"))
    direct = rec.get("payoff_ratio", float("nan"))
    if math.isfinite(direct) and direct > 0.0:
        return direct
    if math.isfinite(p) and math.isfinite(pf) and 0.0 < p < 1.0:
        return pf * (1.0 - p) / p
    return float("nan")


def r_expectancy(wr, payoff):
    return wr * payoff - (1.0 - wr)


def confidence(n):
    return min(1.0, math.sqrt(max(n, 0.0)) / 10.0)


def proposed_score(rec):
    wr = rec.get("win_rate", float("nan"))
    payoff = derived_payoff(rec)
    if not (math.isfinite(wr) and math.isfinite(payoff)):
        return float("-inf")
    return r_expectancy(wr, payoff) * confidence(rec.get("total_trades", 0.0))


def q(vals, p):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(p * (len(v) - 1))))]


def describe(name, recs):
    wrs = [r["win_rate"] for r in recs if math.isfinite(r.get("win_rate", float("nan")))]
    pays = [derived_payoff(r) for r in recs if math.isfinite(derived_payoff(r))]
    tpm = [
        r["trades_per_month"]
        for r in recs
        if math.isfinite(r.get("trades_per_month", float("nan")))
    ]
    nets = [r["net_profit"] for r in recs if math.isfinite(r.get("net_profit", float("nan")))]
    exps = [
        r_expectancy(r["win_rate"], derived_payoff(r))
        for r in recs
        if math.isfinite(r.get("win_rate", float("nan"))) and math.isfinite(derived_payoff(r))
    ]
    print(f"--- {name} (n={len(recs)}) ---")
    print(f"  win_rate   med {q(wrs, 0.5):.3f}  p25 {q(wrs, 0.25):.3f}  p75 {q(wrs, 0.75):.3f}")
    below = sum(1 for x in pays if x < 1.0) / max(1, len(pays))
    print(
        f"  payoff     med {q(pays, 0.5):.3f}  p25 {q(pays, 0.25):.3f}  p75 {q(pays, 0.75):.3f}"
        f"  share<1 {100 * below:.1f}%"
    )
    pos = sum(1 for e in exps if e > 0) / max(1, len(exps))
    print(
        f"  E[R]/trade med {q(exps, 0.5):+.4f}  p25 {q(exps, 0.25):+.4f}"
        f"  p75 {q(exps, 0.75):+.4f}  share>0 {100 * pos:.1f}%"
    )
    print(f"  trades/mo  med {q(tpm, 0.5):.1f}  (~{q(tpm, 0.5) / 21.0:.1f}/day)")
    print(f"  net        med {q(nets, 0.5):,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("combo", help="e.g. EURUSD_M15")
    ap.add_argument(
        "--store",
        default=os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "neoethos", "cache", "discovery"
        ),
    )
    args = ap.parse_args()

    quality = os.path.join(args.store, f"{args.combo}.quality.json")
    portfolio = os.path.join(args.store, f"{args.combo}.live_portfolio.json")
    for p in (quality, portfolio):
        if not os.path.isfile(p):
            sys.exit(f"missing artifact: {p}")

    with open(portfolio, "r", encoding="utf-8") as f:
        port = json.load(f)
    sel_ids = {g["strategy_id"] for g in port["genes"]}
    print(f"{port['symbol']} {port['base_tf']}: {len(sel_ids)} exported genes")

    pool = list(stream_records(quality))
    print(f"quality pool: {len(pool)} records")

    selected = [r for r in pool if r.get("strategy_id") in sel_ids]
    ranked = sorted(pool, key=proposed_score, reverse=True)
    top_n = ranked[: len(sel_ids)]
    top_ids = {r["strategy_id"] for r in top_n}

    describe("quality pool", pool)
    describe("actually selected", selected)
    describe("proposed top-N (E[R]*conf, payoff explicit)", top_n)
    print(
        f"overlap proposed vs actual: {len(top_ids & sel_ids)} / {len(sel_ids)}"
    )

    fin = [r for r in pool if math.isfinite(derived_payoff(r))]
    for floor in (1.2, 1.3, 1.5, 2.0):
        surv = sum(1 for r in fin if derived_payoff(r) >= floor)
        sel_surv = sum(1 for r in selected if derived_payoff(r) >= floor)
        print(
            f"payoff floor {floor}: pool survivors {surv}/{len(fin)}"
            f" ({100 * surv / max(1, len(fin)):.1f}%), of actual selection {sel_surv}/{len(selected)}"
        )


if __name__ == "__main__":
    main()
