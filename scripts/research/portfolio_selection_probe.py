#!/usr/bin/env python3
"""Exploratory portfolio-gating counterexample and 100k structured-candidate census.

Synthetic-only tests. NO market quotes, NO GA/SMC Rust evaluation, NO OOS claim.
`python3 scripts/research/portfolio_selection_probe.py --test --stress 100000`
The static capital-sleeve model below is deliberately NOT a substitute for
QuoteValidatedOuterHoldoutReplaySetV3 or the broker's execution economics.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import unittest

FAMILIES = (
    "trend_pullback", "mean_reversion", "breakout", "momentum", "counter_trend"
)

@dataclass(frozen=True)
class Result:
    return_pct: float
    max_drawdown_pct: float
    worst_day_pct: float
    target_passed: bool
    prop_passed: bool

def replay(returns, weights, target_pct=8.0, max_daily_loss_pct=5.0,
           max_drawdown_pct=10.0):
    """Rebalanced, time-aligned dedicated sleeves; weights share capital <= 1."""
    if not returns or len(weights) != len(returns):
        raise ValueError("one weight per sleeve")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights)>1.000000000001:
        raise ValueError("aggregate allocation cannot exceed 100%")
    n = len(returns[0])
    if not n or any(len(series) != n for series in returns):
        raise ValueError("series must share the same chronological calendar")
    if any(not math.isfinite(r) or r < -1 for xs in returns for r in xs):
        raise ValueError("invalid individual net daily return")
    equity = peak = 1.0
    worst_day = max_dd = 0.0
    day_breach = False
    for day in range(n):
        r = sum(weight * xs[day] for weight, xs in zip(weights, returns))
        worst_day = min(worst_day, r)
        day_breach |= r <= -max_daily_loss_pct / 100
        equity *= 1 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    ret = (equity - 1) * 100
    reached = ret >= target_pct
    return Result(ret, max_dd*100, worst_day*100, reached,
                  reached and not day_breach and max_dd*100 < max_drawdown_pct)

def structured_candidates(n):
    """20,000 deterministic unique role combinations per family, not trades.

    One family has 8 regimes * 10 setups * 10 triggers * 25 exit variants.
    This enumerates role slots, not real VectorTA or SMC rule feasibility.
    """
    if not (0 <= n <= 100_000):
        raise ValueError("prototype's bounded budget is 0..100000")
    quotas = [n // 5 + (i < n % 5) for i in range(5)]
    for family, quota in zip(FAMILIES, quotas):
        for i in range(quota):
            yield (family, i % 8, i // 8 % 10, i // 80 % 10, i // 800 % 25)

class TestProbe(unittest.TestCase):
    def test_two_individually_failing_can_pass_portfolio_goal(self):
        a = [-.06, .10, -.06, .10]
        b = [.10, -.06, .10, -.06]
        self.assertFalse(replay([a], [1]).prop_passed)
        self.assertFalse(replay([b], [1]).prop_passed)
        together = replay([a, b], [.5, .5])
        self.assertTrue(together.prop_passed)
        self.assertAlmostEqual(together.return_pct, (1.02**4 - 1)*100)
        self.assertAlmostEqual(together.max_drawdown_pct, 0.0)

    def test_capital_not_double_counted(self):
        xs = [.05]*4
        self.assertAlmostEqual(replay([xs, xs], [.5, .5]).return_pct,
                               replay([xs], [1]).return_pct)
        with self.assertRaises(ValueError):
            replay([xs, xs], [1.0, 1.0])

    def test_correlated_losses_are_not_diversification(self):
        xs = [-.06, .10, -.06, .10]
        self.assertAlmostEqual(replay([xs, xs], [.5, .5]).max_drawdown_pct,
                               replay([xs], [1]).max_drawdown_pct)

    def test_two_negative_sleeves_are_not_magically_profitable(self):
        self.assertLess(replay([[-.02]*4, [-.01]*4], [.5, .5]).return_pct, 0.0)

    def test_mismatched_calendar_rejected(self):
        with self.assertRaises(ValueError):
            replay([[.01, .02], [.01]], [.5, .5])

    def test_one_hundred_thousand_unique_cover_all_five_families(self):
        xs = list(structured_candidates(100_000))
        self.assertEqual(len(xs), 100_000)
        self.assertEqual(len(set(xs)), 100_000)
        self.assertEqual(Counter(row[0] for row in xs),
                         {family:20_000 for family in FAMILIES})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--stress", type=int, default=0)
    args = p.parse_args()
    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestProbe))
        if not result.wasSuccessful():
            raise SystemExit(1)
    if args.stress:
        rows = list(structured_candidates(args.stress))
        if len(set(rows)) != args.stress:
            raise SystemExit("duplicate candidate signatures")
        print(json.dumps({
            "structured_candidate_signatures": args.stress,
            "per_family": dict(Counter(x[0] for x in rows)),
            "duplicates": 0,
            "executed_market_backtests": 0,
            "warning": "Synthetic candidate coverage only; no trading profit evidence."
        }, indent=2))

if __name__ == "__main__":
    main()
