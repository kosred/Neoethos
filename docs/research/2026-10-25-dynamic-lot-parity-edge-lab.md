# Dynamic lot research and stage-parity audit (2026-10-25)

Scope: isolated branch exp/pre-oos-funnel-ci-20261025. No edits to master, final OOS, or broker orders.

## Execution evidence

- Actions run: https://github.com/kosred/Neoethos/actions/runs/40241481755
- SHA-pinned real EURUSD M5 data: 1,054,644 total rows, first 843,715 selected only, remaining 210,929 final-OOS rows never read as features or scored.
- 306 deterministic candidate rules covering six signal families, three windows (25/50/100% of training pool), dynamic USD lot sizes, next-bar-open fills, 0.01 lot broker grid and 30x notional limit.
- Risky proxy: 100 USD initial, 10% risk-budget ceiling. Best 25/50%-window volatility-expansion candidate showed +170.64%/+185.90% on its training arm but -38.80% internal forward at 0.4 pip spread plus 0.2 pip extra assumed round-trip fee. Even zero-cost internal forward was negative.
- PropFirm proxy: 10,000 USD and 1% risk-budget ceiling with simplified daily entry/drawdown gates. No evaluated candidate qualified in training.
- Risky at initial 20 USD: no executed trades for chosen volatility-expansion rule, 232 would-be entries below broker min due to the 30x notional cap. This is a feasibility finding, not permission to remove risk controls.
- Full numbers, per-fold returns, and size/cap diagnostics live in the attached Actions artifact eurusd-dynamic-edge-lab-results.

## Existing production path discrepancies

- live_trading.rs risk_based_lots computes lots using account balance, risk fraction, actual entry SL, currency-specific pip value, broker min/max/step, operator max lot, and separate 30x notional ceiling. The request schema still displays a default 0.01-lot field; this is not proof live autonomous entries use fixed lots.
- eval.rs BacktestSettings has risk-based sizing enabled by default, but fast_evaluate_strategy_core uses it only with nonempty per-bar confidences. With empty confidence vector, it uses a legacy fixed 1.0 lot.
- discovery.rs quality-stage Monte Carlo profitability perturbations explicitly use fixed 1 lot (legacy parity with simulate_trades_core).
- eval.rs WF parity tests document one legacy fixed-1-lot population lane and the analogous single-gene empty-confidence route. validation.rs supports other confidence-aware paths; not all walk-forward evaluation can be described as fixed-lot.
- discovery.rs portfolio CPCV includes confidence-scaled sizing. Money-valued rankings/results may therefore be incomparable unless the effective position-sizing contract for each gate is established.

## Correct research follow-up

1. Trace same candidate, equity, confidence, stop, commission, broker lot grid and lot size end to end through GA, quality, WF, CPCV and live.
2. Distinguish fixed-lot shape-only robustness checks from money-valued account-risk gates. Do not compare nominal PnL across unlike sizing models.
3. Fix economically inconsistent model boundaries only after preserving CPU/GPU parity and fail-closed broker constraints. Never lift caps to manufacture profit.
4. Expand true strategy architectures and evaluate with nested, chronological training-only WF. Already-observed internal-forward windows must not be treated as pristine if used for repeated research tuning.
5. Public OHLCV lacks synchronized Bid/Ask and broker fees. No positive broker-real edge, prop-firm compliance or trading permission is established.
