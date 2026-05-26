# Risky Mode Compounding — Research Notes

**Last updated**: 2026-05-25 (operator-validated, Kelly-aligned remediation)

This document is the authoritative source for the §-referenced design
decisions throughout `crates/neoethos-core/src/domain/risky_mode.rs`.
Each section number below corresponds to an inline reference in the
code.

---

## §4.1 — Bankroll framing ($20 → $50,000 default)

**Operator directive**: turn a tiny starting balance ($20-$100) into a
larger one ($50K-$100K+) via aggressive geometric compounding, then
flip to normal-pace trading. Two configurable ends:

- `starting_capital_usd` — default `$20`. Operator may raise to `$100`
  or higher via the wizard's AutonomyRisk step.
- `target_capital_usd` — default `$50,000`. Operator may set to
  `$100,000` or any positive value larger than starting capital.

**Why $20 not $100 as the canonical default**: the lower the starting
balance, the more honest about the ruin probability. A $20 floor makes
the "you will almost certainly lose this" acknowledgement land harder.

---

## §4.2 — Logarithmic stage table

The bankroll range is tiled into `stages` of geometric span: each
stage covers a doubling of bankroll (default `stage_doubling_factor =
2.0` → ~11 stages from $20 to $50,000).

Each `RiskyStage` carries:
- `bankroll_lower_usd` / `bankroll_upper_usd` (half-open `[lo, hi)`)
- `risk_per_trade_fraction` (taper from `MAX` at stage 0 to `MIN` at
  the last stage — see §7.1)
- `daily_loss_cap_fraction` (taper 0.80 → 0.50)
- `weekly_drawdown_cap_fraction` (taper 0.95 → 0.60)
- `max_concurrent_positions` (default 1 — single-position scalping)

---

## §4.6.2 — Volatility-σ pause kill switch

When the rolling 30-day ATR exceeds 3σ above its rolling mean, the
per-stage kill switch pauses new entries until ATR normalizes. The
threshold is operator-tunable via `RiskyModeConfig::volatility_sigma_pause`
(default `DEFAULT_VOLATILITY_SIGMA_PAUSE = 3.0`).

**Why 3σ**: empirically the high-vol tail of FX M1 ATR distribution
fattens above 3σ. Trading inside the tail with 30-50% risk per trade
accelerates ruin disproportionately to expected return.

---

## §4.6.3 — News blackout filter

Required when `RiskyModeConfig::require_news_blackout` is true (default
`true`). The bot refuses to open new positions inside the high-impact
news blackout window (USD CPI, NFP, FOMC, etc.) sourced from the
ForexFactory XML feed via `neoethos_app::news_filter`.

**Rationale**: a 30-50% per-trade risk through a news spike is a
direct gateway to ruin. The blackout filter is the cheapest defense.

---

## §4.6.4 — Regime filter

Required when `RiskyModeConfig::require_regime_filter` is true (default
`true`). The bot only enters in a regime the producer's classifier
(rule-based `stop_target::infer_regime` and/or HMM-based
`RegimeHmmExpert`) confirms is tradable.

**Rationale**: high-risk entries during transitional/range regimes
have disproportionately worse expected value. Filtering to trend
regimes lifts the effective win rate AFTER costs.

---

## §6.3 — "Ruined" definition

The Brownian-barrier estimator in
`RiskyModeManager::current_ruin_probability_estimate` defines "ruined"
as `bankroll ≤ $1`. The probability is computed as
`P(ruin) ≈ exp(−2 · μ_log · ln(B / $1) / σ²_log)` where `μ_log` and
`σ²_log` are the per-trade log-return mean and variance under the
stage's `risk_per_trade_fraction` and the operator's
`(expected_win_rate, expected_reward_to_risk)` pair.

---

## §6.4 — 99% ruin probability acknowledgement ceiling

`MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY = 0.99` — the operator must
sign this acknowledgement via the wizard's AutonomyRisk step before
Risky Mode can be enabled. The §7.1 default parameters
(`expected_win_rate=0.52`, `expected_reward_to_risk=1.5`, `f=0.30`)
sit right around the edge of positive expected log-growth — most
attempts WILL lose the starting balance.

Operators who empirically demonstrate a stronger edge (e.g.
`win_rate=0.55`, `RR=2.0`) drop the initial ruin estimate to ~0.5%
(see Kelly analysis table below).

---

## §7.1 — 30-50 % per-trade risk band + Kelly analysis (2026-05-25)

The operator-signed risk fraction band is `[0.30, 0.50]` per
`RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION` / `MAX`. Anything below 0.30
degenerates into "FTMO with a different name"; anything above 0.50
makes the daily kill switch the only thing preventing total loss.

### Kelly criterion analysis

For the operator's typical strong-edge configuration
(`win_rate p = 0.55`, `reward_to_risk RR = 2.0`):

```
Kelly fraction f* = (p · RR − (1 − p)) / RR
                  = (0.55 · 2.0 − 0.45) / 2.0
                  = 0.65 / 2.0
                  = 0.325
```

The signed band 0.30-0.50 thus brackets Kelly: lower edge slightly
sub-Kelly, upper edge over-Kelly. Mathematically, over-Kelly is
dominated by variance without commensurate expected-growth gain.

### Comparison for $100 → $100K target

| risk_f | Position | Expected days | Conservative | Ruin |
|--------|----------|--------------:|-------------:|-----:|
| 0.50 | over-Kelly | 7d | 11d | 12% |
| 0.40 (old default) | over-Kelly | 8d | 12d | **5.6%** |
| **0.30 (Kelly-aligned, new default 2026-05-25)** | **at-Kelly** | **8d** | **10d** | **0.48%** ✅ |
| 0.20 | sub-Kelly | 9d | 11d | 0.004% |
| 0.10 | very sub-Kelly | 14d | 16d | ~0% |

**Conclusion**: lowering the default from 0.40 to 0.30 gives 12× lower
ruin probability with identical expected speed and a *faster*
conservative-case (10d vs 12d). The geometric-kick rationale for the
0.50 ceiling stays valid for the *first stage* (small bankrolls
$20-$40 where psychological tolerance of 50% loss is highest), and
the stage-table tapers from 0.50 at stage 0 down to 0.30 at the
final stage. The new default is the static fallback for code paths
that don't iterate the stage table.

---

## §7.2 — f64 numeric convention

All bankroll, price, PnL, and risk-fraction values throughout
`risky_mode.rs` are `f64`. f64 carries ~15-16 decimal digits of
mantissa — enough to keep cents accurate at the $50,000-target scale.
The earlier f32 build is retired.

---

## §10.3 — Paper-trading-first ladder

`RiskyModeConfig::allow_live_broker` defaults to `false`. The
operator must explicitly flip this AFTER demonstrating paper-trading
stability for the configured horizon (research §10.3 promotion
ladder, not yet fully wired — `#[allow(dead_code)]` markers in
`trading/mod.rs` track the wiring points).

---

## §10.5 — Don't surface optimistic projections when math says
target is unreachable

`RiskyModeManager::estimated_days_to_target` returns `Option<u32>`
and yields `None` whenever the expected per-trade log-growth is
non-positive (`mu_log ≤ 0`). The wizard's days-to-target estimate
must display a "target unreachable with current edge" message rather
than fabricating an optimistic number when the math says ruin is the
expected outcome.

`time_to_target_scenarios()` returns the full triple
(`best_case_days`, `expected_days`, `conservative_days`,
`ruin_probability`) so operators see VARIANCE, not just a single
deterministic projection.

---

## References

- **`base.rs::ExpertModel`** — canonical 3-class output convention
  (`[neutral, buy, sell]`) consumed by HMM regime filter (§4.6.4)
- **`RegimeHmmExpert`** (`neoethos-models::forecasting::hmm_regime`) —
  soft-posterior 3-state HMM, synergizes with `f=0.30` to drop ruin
  to ~0.001% for the same time-to-target
- **`docs/audit/AUDIT-FINDINGS.md`** Kelly analysis table — concrete
  numbers comparing 0.10 / 0.16 / 0.20 / 0.30 / 0.40 / 0.50 across
  multiple target multiples
- **Operator directive 2026-05-17** — 20-pip-challenge fixed framing
  retired in favour of adaptive scalping + net-profit-after-expenses
  optimization
