//! Honest goal report — the "no fake results" output side.
//!
//! The operator's Risky goal is to reach a target (100 EUR -> 50k) within a
//! horizon and then graduate to a steady-income mode. A backtest's headline net
//! profit answers none of the questions that actually matter for that: *how
//! likely* is the target, *how long* would it take, and *how often does the
//! account blow up first*. Worse, at high risk the mean terminal balance is
//! dragged up by a few lucky paths while the median path is ruined — so a single
//! "expected 50k" number is the exact fiction the operator asked to avoid.
//!
//! This module Monte-Carlos the selected portfolio's REAL per-trade R-multiples
//! (each already net of the broker costs Decision D charges) across a sweep of
//! risk levels, and reports, per risk level: P(reach target), P(ruin), median
//! AND mean terminal balance, and the median time-to-target. The risk that
//! MAXIMISES P(reach target) is surfaced — for a positive-edge strategy that is
//! moderate (near-Kelly), never the maximum, so this is also how the sizing
//! ceiling gets an honest recommendation instead of a guess.
//!
//! R-multiples are size-independent (pnl / risk_amount), so re-applying an
//! arbitrary per-trade risk fraction `f` as `equity *= 1 + f*R` is exact: `f` is
//! the fraction of equity risked, `R` the outcome in units of that risk.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Below this fraction of the starting balance the account is treated as blown
/// (a 100 EUR account at 2 EUR cannot realistically recover to 50k).
const RUIN_FLOOR_FRACTION: f64 = 0.02;

/// Monte-Carlo paths per risk level. Large enough that P(reach) is stable to
/// ~0.3% and cheap: paths short-circuit at target or ruin.
const DEFAULT_PATHS: usize = 20_000;

/// The risk-per-trade fractions swept by default. Spans conservative to the
/// operator's 30% ceiling so the frontier shows where P(reach) peaks and where
/// the "20-pip challenge" 30% setting actually lands.
pub const DEFAULT_RISK_LEVELS: &[f64] = &[0.05, 0.10, 0.15, 0.20, 0.30];

/// One risk level's simulated outcome distribution.
#[derive(Debug, Clone)]
pub struct RiskOutcome {
    pub risk_fraction: f64,
    /// Fraction of paths that reached the target within the horizon.
    pub p_reach_target: f64,
    /// Fraction of paths that hit the ruin floor.
    pub p_ruin: f64,
    /// Median terminal balance across ALL paths (the typical outcome).
    pub median_terminal: f64,
    /// Mean terminal balance — for high risk this is lottery-inflated above the
    /// median; the gap between them is the tell.
    pub mean_terminal: f64,
    /// Median days to first reach the target, among reaching paths. `None` when
    /// fewer than half the paths reached (no meaningful "typical" time).
    pub median_days_to_target: Option<f64>,
    pub reached_paths: usize,
    pub paths: usize,
}

/// The full report: the frontier plus the risk level that maximised P(reach).
#[derive(Debug, Clone)]
pub struct GoalReport {
    pub start_balance: f64,
    pub target_balance: f64,
    pub horizon_days: f64,
    pub trades_per_day: f64,
    pub trades_in_horizon: usize,
    /// How many real trades fed the bootstrap (0 => the report is not meaningful).
    pub n_trades_sampled: usize,
    pub avg_r_multiple: f64,
    pub frontier: Vec<RiskOutcome>,
    /// The risk fraction on the frontier with the highest P(reach target).
    pub best_risk_fraction: f64,
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Simulate `n_paths` bootstrap paths at one risk fraction.
fn simulate_one_risk(
    r_multiples: &[f64],
    start: f64,
    target: f64,
    trades_in_horizon: usize,
    risk_fraction: f64,
    n_paths: usize,
    trades_per_day: f64,
    seed: u64,
) -> RiskOutcome {
    let ruin_floor = start * RUIN_FLOOR_FRACTION;
    let n = r_multiples.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reached = 0usize;
    let mut ruined = 0usize;
    let mut terminals: Vec<f64> = Vec::with_capacity(n_paths);
    let mut days_to_target: Vec<f64> = Vec::new();

    for _ in 0..n_paths {
        let mut equity = start;
        let mut reached_at: Option<usize> = None;
        for t in 0..trades_in_horizon {
            let r = r_multiples[rng.random_range(0..n)];
            // `1 + f*R`: an R below -1/f wipes the account; clamp the factor at
            // 0 so a single catastrophic outlier cannot make equity negative.
            equity *= (1.0 + risk_fraction * r).max(0.0);
            if equity <= ruin_floor {
                equity = 0.0;
                ruined += 1;
                break;
            }
            if equity >= target {
                reached_at = Some(t + 1);
                break;
            }
        }
        if let Some(tt) = reached_at {
            reached += 1;
            days_to_target.push(tt as f64 / trades_per_day.max(1e-9));
        }
        terminals.push(equity);
    }

    terminals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    days_to_target.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_terminal = terminals.iter().sum::<f64>() / n_paths.max(1) as f64;

    RiskOutcome {
        risk_fraction,
        p_reach_target: reached as f64 / n_paths.max(1) as f64,
        p_ruin: ruined as f64 / n_paths.max(1) as f64,
        median_terminal: percentile_sorted(&terminals, 0.5),
        mean_terminal,
        median_days_to_target: if reached * 2 >= n_paths {
            Some(percentile_sorted(&days_to_target, 0.5))
        } else {
            None
        },
        reached_paths: reached,
        paths: n_paths,
    }
}

/// Build the goal report from a portfolio's realized per-trade R-multiples.
///
/// `r_multiples` must be net of costs (Decision D). `seed` makes the report
/// reproducible (slice 5): the same inputs always produce the same frontier.
pub fn build_report(
    r_multiples: &[f64],
    start_balance: f64,
    target_balance: f64,
    horizon_days: f64,
    trades_per_day: f64,
    risk_levels: &[f64],
    seed: u64,
) -> GoalReport {
    let trades_in_horizon = (trades_per_day.max(0.0) * horizon_days.max(0.0)).round() as usize;
    let avg_r = if r_multiples.is_empty() {
        0.0
    } else {
        r_multiples.iter().sum::<f64>() / r_multiples.len() as f64
    };

    let frontier: Vec<RiskOutcome> = if r_multiples.is_empty() || trades_in_horizon == 0 {
        Vec::new()
    } else {
        risk_levels
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                simulate_one_risk(
                    r_multiples,
                    start_balance,
                    target_balance,
                    trades_in_horizon,
                    f,
                    DEFAULT_PATHS,
                    trades_per_day,
                    // Per-level seed so levels are independent yet reproducible.
                    seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                )
            })
            .collect()
    };

    let best_risk_fraction = frontier
        .iter()
        .max_by(|a, b| {
            a.p_reach_target
                .partial_cmp(&b.p_reach_target)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|o| o.risk_fraction)
        .unwrap_or(0.0);

    GoalReport {
        start_balance,
        target_balance,
        horizon_days,
        trades_per_day,
        trades_in_horizon,
        n_trades_sampled: r_multiples.len(),
        avg_r_multiple: avg_r,
        frontier,
        best_risk_fraction,
    }
}

impl GoalReport {
    /// Human-readable multi-line summary for the discovery log / CLI. Leads with
    /// the honest headline: the risk that maximises P(reach), and the fact that
    /// the operator's 30% ceiling is usually NOT it.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        if self.n_trades_sampled == 0 || self.frontier.is_empty() {
            return "GOAL REPORT: not enough trades to simulate — no honest \
                    projection possible (the portfolio produced no usable trades)."
                .to_string();
        }
        let _ = writeln!(
            s,
            "GOAL REPORT — reach {:.0} from {:.0} within {:.0} days (~{:.1} trades/day, \
             {} real trades bootstrapped, avg {:.3} R/trade net of costs)",
            self.target_balance,
            self.start_balance,
            self.horizon_days,
            self.trades_per_day,
            self.n_trades_sampled,
            self.avg_r_multiple,
        );
        let _ = writeln!(
            s,
            "  risk/trade  P(reach)  P(ruin)   median-end     mean-end   median-time",
        );
        for o in &self.frontier {
            let time = match o.median_days_to_target {
                Some(d) => format!("{:.0} days", d),
                None => "> horizon".to_string(),
            };
            let star = if (o.risk_fraction - self.best_risk_fraction).abs() < 1e-9 {
                " <= max P(reach)"
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "  {:>6.0}%     {:>6.1}%   {:>6.1}%  {:>11.0}  {:>11.0}   {:>11}{}",
                o.risk_fraction * 100.0,
                o.p_reach_target * 100.0,
                o.p_ruin * 100.0,
                o.median_terminal,
                o.mean_terminal,
                time,
                star,
            );
        }
        let _ = writeln!(
            s,
            "  Honest read: P(reach) peaks at {:.0}% risk; higher risk trades a \
             lower median (often ruin) for a fatter but rarer lottery tail. \
             'mean-end' far above 'median-end' means the average is lottery-driven.",
            self.best_risk_fraction * 100.0,
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 2RR edge at ~45% win rate: wins = +2R, losses = -1R.
    fn edge_2r_45pct() -> Vec<f64> {
        let mut v = Vec::new();
        for _ in 0..45 {
            v.push(2.0);
        }
        for _ in 0..55 {
            v.push(-1.0);
        }
        v
    }

    #[test]
    fn a_positive_edge_reaches_the_target_more_often_at_moderate_risk_than_at_max() {
        // The core honest claim: for a positive edge, P(reach) is NOT monotonic
        // in risk — it peaks at a moderate level and falls at extreme risk where
        // ruin dominates. This is the whole reason 30% fixed is wrong.
        let r = edge_2r_45pct();
        let rep = build_report(&r, 100.0, 50_000.0, 180.0, 2.0, DEFAULT_RISK_LEVELS, 42);
        assert_eq!(rep.frontier.len(), DEFAULT_RISK_LEVELS.len());
        let p_at = |f: f64| {
            rep.frontier
                .iter()
                .find(|o| (o.risk_fraction - f).abs() < 1e-9)
                .unwrap()
        };
        // 30% over-bets a 2R/45% edge (Kelly ~17.5%) into the ruin zone: its
        // P(ruin) must exceed a moderate level's, and it must not be the best.
        assert!(
            p_at(0.30).p_ruin > p_at(0.10).p_ruin,
            "30% must ruin more often than 10%: {} vs {}",
            p_at(0.30).p_ruin,
            p_at(0.10).p_ruin
        );
        assert!(
            rep.best_risk_fraction < 0.30,
            "the P(reach)-maximising risk must be below the 30% ceiling, got {}",
            rep.best_risk_fraction
        );
    }

    #[test]
    fn a_negative_edge_basically_never_reaches_and_mostly_ruins() {
        // Losing system: wins +1R at 40%, losses -1R at 60% => negative EV.
        let mut r = Vec::new();
        for _ in 0..40 {
            r.push(1.0);
        }
        for _ in 0..60 {
            r.push(-1.0);
        }
        let rep = build_report(&r, 100.0, 50_000.0, 180.0, 2.0, DEFAULT_RISK_LEVELS, 7);
        for o in &rep.frontier {
            assert!(
                o.p_reach_target < 0.05,
                "a negative-edge system must almost never reach 500x at {}% risk, got {}",
                o.risk_fraction * 100.0,
                o.p_reach_target
            );
        }
    }

    #[test]
    fn the_report_is_reproducible_for_the_same_seed() {
        let r = edge_2r_45pct();
        let a = build_report(&r, 100.0, 50_000.0, 180.0, 2.0, DEFAULT_RISK_LEVELS, 99);
        let b = build_report(&r, 100.0, 50_000.0, 180.0, 2.0, DEFAULT_RISK_LEVELS, 99);
        for (x, y) in a.frontier.iter().zip(b.frontier.iter()) {
            assert_eq!(x.reached_paths, y.reached_paths);
            assert_eq!(x.p_ruin, y.p_ruin);
        }
    }

    #[test]
    fn no_trades_yields_an_honest_empty_report_not_a_panic() {
        let rep = build_report(&[], 100.0, 50_000.0, 180.0, 2.0, DEFAULT_RISK_LEVELS, 1);
        assert_eq!(rep.n_trades_sampled, 0);
        assert!(rep.frontier.is_empty());
        assert!(rep.render().contains("not enough trades"));
    }
}
