//! Four canonical scoring formulas, each composed from the shared
//! ingredients in [`super::ingredients`].
//!
//! Phase A: every named function preserves the exact magic-constant
//! weight table of its predecessor so the GA's fitness landscape stays
//! byte-for-byte identical. The migration is STRUCTURAL — old
//! functions in `evolution_math.rs` / `quality.rs` / `regime_labels.rs`
//! / `diversity.rs` become `#[deprecated]` re-exports that call the
//! named functions here. Behavioural unification (collapsing the four
//! weight tables into one) is Phase C, gated by `scoring_version`
//! bump to 2.
//!
//! ## Weight tables (preserved from legacy callers, Phase A)
//!
//! | Function | Sharpe | Consistency | DD penalty | PF | Win-rate | Net | Expectancy |
//! |----------|--------|-------------|-----------|----|----|----|--|
//! | `ga_fitness` (v3) | 0.10 × conf₁₀ | 0.10 | subtract `dd*15→5` | 0.15 (GA shape) | 0.10 | 0.15 (net÷20k) | — (+0.45 × monthly-hit-rate, slot 7) |
//! | `quality_score` | 0.25 × conf₁₀ | 0.15 | subtract `dd*8→3` | 0.20 (smooth shape) | 0.10 | 0.20 | 0.10 |
//! | `window_score` | 0.25 × conf₈ | 0.15 | subtract `dd*8→3` | 0.20 (smooth shape) | 0.10 | 0.20 | 0.10 |
//! | `archive_score` | 0.25 × conf₁₀ | 0.15 | subtract `dd*8→3` | 0.20 (smooth shape) | 0.10 | 0.20 | 0.10 |
//!
//! Phase C unification will pick ONE table (the operator's research
//! input drives the choice) and delete the others.

use super::ingredients::{
    consistency_component, drawdown_penalty_window, expectancy_component, net_component,
    profit_factor_component, sharpe_component, trades_confidence, trades_confidence_window,
    win_rate_component,
};

// ---------------------------------------------------------------------------
// Scoring version
// ---------------------------------------------------------------------------

/// Typed wrapper around the scoring-formula schema version.
///
/// Per the operator-approved migration plan (doctrine §3 → §4.4),
/// persisted `DiscoveryRunProfile` artifacts carry this version so
/// that:
/// - Old artifacts (`scoring_version=1`) still deserialize after the
///   Phase-C weight-table unification.
/// - Discovery runs that produced their archive under the old formula
///   are clearly tagged so the operator knows whether top-of-archive
///   genomes are directly comparable to a new run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ScoringVersion(pub u32);

/// Current scoring-formula version.
///
/// 2026-06-06: bumped to `3` — `ga_fitness` is now CONSISTENT-monthly-return
/// oriented. v2 rewarded total net (compounding → lumpy genes that failed the
/// prop-firm window-consistency gate); v3's dominant reward is the fraction of
/// months hitting the operator's ≥4%/month bar (`metrics[7]`, the same consistency
/// the gate checks). (v1 = Sharpe-only; v2 = total-net.) Runs before this are NOT
/// directly comparable (different fitness landscape); old artifacts still deserialize.
///
/// 2026-07-02: bumped to `5` — two landscape changes land together:
/// (a) the weight search space admits NEGATIVE indicator weights (contrarian
/// terms, previously only reachable via seed inheritance and lost on first
/// mutation), and (b) Risky/growth mode gets its OWN objective
/// [`ga_fitness_growth`] — expected Kelly log-growth over the evaluation
/// window — instead of borrowing the prop-firm consistency formula. PropFirm /
/// Strict discovery still uses [`ga_fitness`] (v4 math, unchanged).
/// v6: Risky discovery binds the configured goal/deadline to measured realized
/// balance growth. The v5 function remains only for explicitly legacy callers.
pub const SCORING_VERSION_CURRENT: ScoringVersion = ScoringVersion(6);

pub use neoethos_gpu_contracts::resident_search_scoring_v2::RiskyGrowthGoal;

/// Risky GA v6: continuous squared relative log-shortfall of the observed
/// realized-balance growth pace against the requested deadline.
///
/// `pace = log(1 + net / actual_initial_equity) * horizon / observed_days`.
/// `score = 1 - max(1 - pace / log(target / start), 0)^2`.
/// No hypothetical Kelly bet replaces the confidence-sized simulation. There
/// is no arbitrary drawdown weight or additional trade-frequency gate here;
/// existing validation/risk constraints still apply. Zero trades retain the
/// explicit exploration penalty, and an observed wipeout is never rewarded.
///
/// This is a screening PACE PROXY, not a probability or evidence that a goal
/// was achieved. It extrapolates a realized balance rate (not marked wealth),
/// ignores terminal open PnL, and must not be presented as a replay at the
/// reference starting capital. Above-goal pace saturates; below goal the
/// continuous gradient remains. Target changes need not reorder candidates
/// whose only difference is return, but do change selection pressure.
pub fn ga_fitness_goal(
    metrics: &[f64; 11],
    initial_equity: f64,
    span_days: f64,
    goal: RiskyGrowthGoal,
) -> f64 {
    neoethos_gpu_contracts::resident_search_scoring_v2::score_risky_ga_fitness_goal_v6(
        metrics,
        initial_equity,
        span_days,
        goal,
    )
}

// ---------------------------------------------------------------------------
// ga_fitness — was `genetic::evolution_math::score_from_metrics`
// ---------------------------------------------------------------------------

/// GA fitness — the value the genetic algorithm MAXIMISES per genome.
///
/// Matches `genetic::evolution_math::score_from_metrics` byte-for-byte.
/// The mapping in `metrics: &[f64; 11]` is the canonical
/// `[BacktestMetrics::to_metric_array]` order (see `eval.rs` lines
/// 165-200 and `BACKTEST_METRICS_MONTHLY_TARGET_HIT_RATE_INDEX`):
///
/// ```text
///   metrics[0] = net_profit      metrics[5] = profit_factor
///   metrics[1] = sharpe          metrics[6] = expectancy
///   metrics[2] = peak_equity     metrics[7] = monthly_target_hit_rate
///   metrics[3] = max_drawdown    metrics[8] = trade_count
///   metrics[4] = win_rate        metrics[9] = consistency
///                                metrics[10] = max_daily_drawdown
/// ```
///
/// Sentinel: returns `f64::NEG_INFINITY` when Sharpe is non-finite OR
/// trade_count < 1 → caller (the GA selection step) treats this as a
/// "do not propagate" marker.
pub fn ga_fitness(metrics: &[f64; 11]) -> f64 {
    neoethos_gpu_contracts::resident_search_scoring_v2::score_prop_firm_ga_fitness_v4(metrics)
}

// ---------------------------------------------------------------------------
// ga_fitness_growth — Risky-mode objective (scoring_version 5)
// ---------------------------------------------------------------------------

/// GA fitness for Risky / capital-multiplication discovery: expected
/// **Kelly log-growth over the evaluation window**, from the gene's own
/// measured `(win_rate, profit_factor, trades)`.
///
/// Rationale (operator + Curupira/first-passage analysis, 2026-07-02): the
/// post-GA Risky ranking already scores candidates by half-Kelly log-growth
/// scaled to the operator's horizon (`discovery.rs::calculate_income_score`),
/// but the population it ranks was EVOLVED under the prop-firm consistency
/// objective — the GA never searched for fast compounders. This is the same
/// math moved INTO the search. Horizon scaling is deliberately absent: genes
/// in one run share the evaluation window, so total window growth
/// (`g_trade × trades`) orders them identically and needs no span input.
///
/// Shares the [`ga_fitness`] guards: non-finite Sharpe → `NEG_INFINITY`
/// (metrics unusable), zero trades → −100.0 (graduated, not −∞, per GA Fix B).
/// Genes WITHOUT an edge (pf ≤ 1) all have zero growth — a flat plateau the
/// GA cannot climb — so a small ≤ 0 "edge gradient" (distance below pf=1 /
/// wr=50% / net=0) slopes the landscape toward edge; any real edge
/// (growth > 0) dominates it by construction.
///
/// The drawdown-tolerance difference is intentional: Risky mode is
/// drawdown-agnostic BY DESIGN (its survival constraints live in the
/// `risky_mode` domain manager + the risky WF filter, not the fitness), so
/// unlike v4 there is no DD or worst-day penalty here.
pub fn ga_fitness_growth(metrics: &[f64; 11]) -> f64 {
    neoethos_gpu_contracts::resident_search_scoring_v2::score_risky_ga_fitness_growth_v5(metrics)
}

// ---------------------------------------------------------------------------
// archive_score — was `genetic::diversity::archive_quality_score`
// ---------------------------------------------------------------------------

/// Diversity-archive ranking score — what survives across generations
/// in the GA's hall-of-fame buffer.
///
/// Phase A: matches `genetic::diversity::archive_quality_score`
/// behaviourally. Uses the "smooth PF" + net-profit shape (NOT the GA
/// shape), because the archive should prefer genomes that earned real
/// money + had smooth equity curves, not just high Sharpe.
pub fn archive_score(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let expectancy = metrics[6];
    let trades = metrics[8];
    let consistency = metrics[9];

    if !sharpe.is_finite() || trades < 1.0 {
        return f64::NEG_INFINITY;
    }

    let conf = trades_confidence(trades);
    let net_c = net_component(net) * 0.20;
    let sh = sharpe_component(sharpe, conf) * 0.25;
    let pf = profit_factor_component(profit_factor) * 0.20;
    let cons = consistency_component(consistency) * 0.15;
    let wr = win_rate_component(win_rate) * 0.10;
    let exp = expectancy_component(expectancy) * 0.10;
    let dd = drawdown_penalty_window(max_dd);

    net_c + sh + pf + cons + wr + exp - dd
}

// ---------------------------------------------------------------------------
// window_score — was `genetic::regime_labels::window_quality_score`
// ---------------------------------------------------------------------------

/// Per-regime-window scoring during regime labelling.
///
/// Uses the smaller-sample confidence multiplier (`/8.0`) because
/// per-window trade counts are smaller than full-backtest counts.
/// Otherwise identical to `archive_score`.
pub fn window_score(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let expectancy = metrics[6];
    let trades = metrics[8];
    let consistency = metrics[9];

    let conf = trades_confidence_window(trades);
    let net_c = net_component(net) * 0.20;
    let sh = sharpe_component(sharpe, conf) * 0.25;
    let pf = profit_factor_component(profit_factor) * 0.20;
    let cons = consistency_component(consistency) * 0.15;
    let wr = win_rate_component(win_rate) * 0.10;
    let exp = expectancy_component(expectancy) * 0.10;
    let dd = drawdown_penalty_window(max_dd);

    net_c + sh + pf + cons + wr + exp - dd
}

// ---------------------------------------------------------------------------
// quality_score — was `quality.rs::score_strategy`
// ---------------------------------------------------------------------------

/// Post-GA quality gate score — used by `StrategyQualityAnalyzer`
/// downstream of the GA to filter genomes before promotion.
///
/// Phase A: simplest possible delegation — `quality.rs::score_strategy`
/// is a heavy function that combines metrics + gates; this named
/// wrapper covers the bare numeric-score portion. The gates (min
/// Sharpe, min consistency, min trades-per-month) stay in `quality.rs`
/// for now because they need access to the `StrategyQualityAnalyzer`
/// configuration. Phase B migrates them.
pub fn quality_score(metrics: &[f64; 11]) -> f64 {
    // Phase-A behaviour: identical to archive_score. quality.rs's
    // legacy `score_strategy` has the same shape modulo a slightly
    // different consistency clamp (clamps at 0.9 vs 1.0) — that
    // edge-case difference is preserved in the legacy function's
    // `#[deprecated]` shim; this canonical version uses 1.0.
    archive_score(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goal() -> RiskyGrowthGoal {
        RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        }
    }

    #[test]
    fn goal_pace_uses_actual_net_and_sizing_not_hypothetical_kelly() {
        let lost_money = metrics(-2_000.0, 1.0, 0.5, 0.6, 2.0, -20.0, 100.0, 0.5);
        let earned_money = metrics(2_000.0, 1.0, 0.5, 0.6, 2.0, 20.0, 100.0, 0.5);
        assert!(
            (ga_fitness_growth(&lost_money) - ga_fitness_growth(&earned_money) + 0.001).abs()
                < 1e-12
        );
        assert!(ga_fitness_goal(&lost_money, 10_000.0, 180.0, goal()) < 0.0);
        assert!(ga_fitness_goal(&earned_money, 10_000.0, 180.0, goal()) > 0.0);
        let mut different_summary = earned_money;
        different_summary[4] = 0.3;
        different_summary[5] = 1.1;
        assert_eq!(
            ga_fitness_goal(&earned_money, 10_000.0, 180.0, goal()),
            ga_fitness_goal(&different_summary, 10_000.0, 180.0, goal())
        );
    }

    #[test]
    fn goal_pace_binds_span_deadline_and_reference_target() {
        let row = metrics(10_000.0, 1.0, 0.2, 0.6, 2.0, 100.0, 100.0, 0.5);
        let base = ga_fitness_goal(&row, 10_000.0, 180.0, goal());
        assert!(ga_fitness_goal(&row, 10_000.0, 90.0, goal()) > base);
        assert!(
            ga_fitness_goal(
                &row,
                10_000.0,
                180.0,
                RiskyGrowthGoal {
                    horizon_days: 90.0,
                    ..goal()
                }
            ) < base
        );
        assert!(
            ga_fitness_goal(
                &row,
                10_000.0,
                180.0,
                RiskyGrowthGoal {
                    target_balance: 100_000.0,
                    ..goal()
                }
            ) < base
        );
        // The same actual growth rate on differently sized measured windows.
        let mut longer = row;
        longer[0] = 30_000.0; // 4x over twice the time of 2x.
        assert!((ga_fitness_goal(&longer, 10_000.0, 360.0, goal()) - base).abs() < 1e-14);
    }

    #[test]
    fn goal_pace_has_continuous_gradient_below_target_and_saturates_above_it() {
        let mut row = metrics(100.0, 1.0, 0.2, 0.6, 2.0, 1.0, 100.0, 0.5);
        let first = ga_fitness_goal(&row, 100.0, 180.0, goal());
        row[0] = 1_000.0;
        let second = ga_fitness_goal(&row, 100.0, 180.0, goal());
        assert!(first > 0.0 && first < second && second < 1.0);
        row[0] = 49_900.0;
        assert_eq!(ga_fitness_goal(&row, 100.0, 180.0, goal()), 1.0);
        row[0] = 500_000.0;
        assert_eq!(ga_fitness_goal(&row, 100.0, 180.0, goal()), 1.0);
    }

    #[test]
    fn goal_pace_rejects_bad_context_and_bankruptcy_without_mutating_metrics() {
        let row = metrics(100.0, 1.0, 0.2, 0.6, 2.0, 1.0, 100.0, 0.5);
        for span in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                ga_fitness_goal(&row, 100.0, span, goal()),
                f64::NEG_INFINITY
            );
        }
        for initial in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                ga_fitness_goal(&row, initial, 180.0, goal()),
                f64::NEG_INFINITY
            );
        }
        for invalid in [
            RiskyGrowthGoal {
                start_balance: 0.0,
                ..goal()
            },
            RiskyGrowthGoal {
                target_balance: 100.0,
                ..goal()
            },
            RiskyGrowthGoal {
                horizon_days: 0.0,
                ..goal()
            },
            RiskyGrowthGoal {
                target_balance: f64::INFINITY,
                ..goal()
            },
        ] {
            assert!(invalid.validate().is_err());
            assert_eq!(
                ga_fitness_goal(&row, 100.0, 180.0, invalid),
                f64::NEG_INFINITY
            );
        }
        let mut wiped = row;
        wiped[0] = -100.0;
        assert_eq!(
            ga_fitness_goal(&wiped, 100.0, 180.0, goal()),
            f64::NEG_INFINITY
        );
        wiped = row;
        wiped[3] = 1.0;
        assert_eq!(
            ga_fitness_goal(&wiped, 100.0, 180.0, goal()),
            f64::NEG_INFINITY
        );
        let mut no_trades = row;
        no_trades[8] = 0.0;
        assert_eq!(ga_fitness_goal(&no_trades, 100.0, 180.0, goal()), -100.0);
    }

    #[test]
    fn goal_pace_uses_actual_simulation_capital_and_preserves_reference_ratio() {
        let row = metrics(1_000.0, 1.0, 0.2, 0.6, 2.0, 10.0, 100.0, 0.5);
        let base = ga_fitness_goal(&row, 10_000.0, 180.0, goal());
        assert!(ga_fitness_goal(&row, 100.0, 180.0, goal()) > base);
        let equivalent_ratio = RiskyGrowthGoal {
            start_balance: 10_000.0,
            target_balance: 5_000_000.0,
            ..goal()
        };
        assert!((ga_fitness_goal(&row, 10_000.0, 180.0, equivalent_ratio) - base).abs() < 1e-14);
    }

    /// Helper: build a canonical `[f64; 11]` from named fields.
    fn metrics(
        net: f64,
        sharpe: f64,
        max_dd: f64,
        win_rate: f64,
        pf: f64,
        expectancy: f64,
        trades: f64,
        consistency: f64,
    ) -> [f64; 11] {
        [
            net,
            sharpe,
            0.0,
            max_dd,
            win_rate,
            pf,
            expectancy,
            0.0,
            trades,
            consistency,
            0.0,
        ]
    }

    #[test]
    fn ga_fitness_returns_strong_negative_finite_for_zero_trades() {
        // GA Fix B (taskdoc #274): zero-trade is no longer
        // NEG_INFINITY — it's a strong but finite penalty so the GA
        // has a gradient to escape the "vacuous DD<=4%" reward-hack.
        let m = metrics(100.0, 2.0, 0.05, 0.6, 1.8, 12.0, 0.0, 0.7);
        let s = ga_fitness(&m);
        assert!(
            s.is_finite() && s <= -50.0,
            "zero-trade fitness must be strongly negative and finite, got {}",
            s
        );
        // And any healthy trading strategy must beat it by a wide margin.
        let healthy = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        assert!(
            ga_fitness(&healthy) > s + 50.0,
            "trading strategy must comfortably beat zero-trade penalty"
        );
    }

    #[test]
    fn ga_fitness_returns_neg_infinity_for_nan_sharpe() {
        let m = metrics(100.0, f64::NAN, 0.05, 0.6, 1.8, 12.0, 50.0, 0.7);
        assert_eq!(ga_fitness(&m), f64::NEG_INFINITY);
    }

    #[test]
    fn ga_fitness_finite_for_healthy_genome() {
        // scoring_version 3: "healthy" now requires CONSISTENT monthly return, not just
        // high total net. The dominant reward is metrics[7] = monthly_target_hit_rate.
        // A genuinely healthy genome hits the >=4% bar in most months: hit_rate=0.70,
        // net=20000, sharpe=2.0, dd=0.05, wr=0.60, pf=1.8, trades=100, consistency=0.70.
        //   hit=0.70*0.45=0.315; ret=(20000/20000)*0.15=0.15; sh=2.0*0.10=0.20;
        //   cons=0.70*0.10=0.07; pf=0.40*0.15=0.06; wr=0.30*0.10=0.03; dd=0.05*15=0.75
        //   total = (0.315+0.15+0.20+0.07+0.06+0.03)*1.0 - 0.75 = +0.075 > 0.
        let mut m = metrics(20_000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        m[7] = 0.70; // monthly_target_hit_rate: hits >=4% in 70% of months
        let s = ga_fitness(&m);
        assert!(s.is_finite());
        assert!(
            s > 0.0,
            "healthy (consistent-monthly-return, low-DD) genome must score positive, got {}",
            s
        );
    }

    #[test]
    fn ga_fitness_penalises_drawdown() {
        let base = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        let heavy_dd = metrics(1000.0, 2.0, 0.30, 0.60, 1.8, 12.0, 100.0, 0.70);
        assert!(
            ga_fitness(&base) > ga_fitness(&heavy_dd),
            "heavier drawdown should score lower"
        );
    }

    #[test]
    fn ga_fitness_penalises_catastrophic_days_scoring_v4() {
        // scoring_version 4 (steady income): two otherwise-identical genes —
        // the one whose worst DAY was a 4% hit must rank strictly below the
        // one that never had a day worse than 0.5%. Weight 10.0 ⇒ delta 0.35.
        let mut calm = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        calm[10] = 0.005;
        let mut violent = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        violent[10] = 0.04;
        let (c, v) = (ga_fitness(&calm), ga_fitness(&violent));
        assert!(
            c > v && (c - v - 0.35).abs() < 1e-9,
            "worst-day penalty must separate them by exactly 0.35: {c} vs {v}"
        );
    }

    #[test]
    fn ga_fitness_growth_prefers_fast_compounder_over_consistent_grinder() {
        // scoring_version 5: under the GROWTH objective, a 60% WR / PF 2.0 gene
        // must outrank a 60% WR / PF 1.2 gene with identical activity — even
        // though under the prop-firm formula their gap is much narrower.
        let compounder = metrics(5000.0, 2.0, 0.10, 0.60, 2.0, 12.0, 200.0, 0.60);
        let grinder = metrics(5000.0, 2.0, 0.05, 0.60, 1.2, 12.0, 200.0, 0.60);
        let (c, g) = (ga_fitness_growth(&compounder), ga_fitness_growth(&grinder));
        assert!(
            c.is_finite() && g.is_finite() && c > g * 2.0 && c > 0.0 && g > 0.0,
            "growth objective must decisively prefer the compounder: {c} vs {g}"
        );
    }

    #[test]
    fn ga_fitness_growth_no_edge_scores_negative_with_gradient() {
        // pf <= 1 has zero Kelly growth; the edge-gradient must (a) be negative
        // and (b) SLOPE toward the edge — closer-to-edge scores higher.
        let far = metrics(-2000.0, 0.5, 0.20, 0.40, 0.7, 12.0, 100.0, 0.30);
        let near = metrics(-200.0, 0.8, 0.10, 0.48, 0.95, 12.0, 100.0, 0.40);
        let (f, n) = (ga_fitness_growth(&far), ga_fitness_growth(&near));
        assert!(
            f < 0.0 && n < 0.0,
            "no-edge genes must score negative: {f}, {n}"
        );
        assert!(
            n > f,
            "closer-to-edge must score higher: near {n} vs far {f}"
        );
    }

    #[test]
    fn ga_fitness_growth_shares_hard_guards() {
        let nan_sharpe = metrics(100.0, f64::NAN, 0.05, 0.6, 1.8, 12.0, 50.0, 0.7);
        assert_eq!(ga_fitness_growth(&nan_sharpe), f64::NEG_INFINITY);
        let zero_trades = metrics(100.0, 2.0, 0.05, 0.6, 1.8, 12.0, 0.0, 0.7);
        assert_eq!(ga_fitness_growth(&zero_trades), -100.0);
    }

    #[test]
    fn ga_fitness_growth_all_wins_tiny_sample_does_not_zero_out() {
        // p is capped at 0.99 so an all-wins 3-trade gene keeps a positive rr
        // and a positive (small) growth instead of collapsing to exactly 0.
        let lucky = metrics(300.0, 3.0, 0.0, 1.0, 10.0, 12.0, 3.0, 1.0);
        let s = ga_fitness_growth(&lucky);
        assert!(
            s > 0.0 && s.is_finite(),
            "all-wins gene must score >0, got {s}"
        );
    }

    #[test]
    fn archive_score_finite_for_healthy_genome() {
        let m = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        let s = archive_score(&m);
        assert!(s.is_finite());
        assert!(s > 0.0);
    }

    #[test]
    fn window_score_uses_smaller_confidence_divisor() {
        // Same metrics, comparing window_score vs archive_score: the
        // window-side ÷8 confidence means a 64-trade window saturates
        // the multiplier, while archive's ÷10 needs 100 trades.
        let m = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 64.0, 0.70);
        let arch = archive_score(&m);
        let win = window_score(&m);
        assert!(
            win >= arch,
            "window-side saturates confidence faster → score must be ≥ archive: {} vs {}",
            win,
            arch
        );
    }

    #[test]
    fn quality_score_delegates_to_archive_for_phase_a() {
        let m = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        assert_eq!(quality_score(&m), archive_score(&m));
    }

    #[test]
    fn ga_fitness_matches_legacy_score_from_metrics_pin() {
        // PIN the Phase-A behaviour-preservation contract. If the
        // weight table moves in ga_fitness, this test breaks LOUDLY
        // so a Phase-C unification doesn't silently change the GA
        // fitness landscape without the scoring_version bump.
        //
        // GA Fix B (2026-05-26, taskdoc #274): added activity multiplier
        // `(0.3 + 0.7 * activity)` to positive components. For trades=100
        // the activity clamps to 1.0 → multiplier = 1.0 → math unchanged.
        // The 0.335 pin therefore SURVIVES the graduated-fitness fix
        // because the healthy-genome case sits in the saturated region.
        //
        // scoring_version 3 (2026-06-06, consistent-monthly-return GA): dominant reward
        // is metrics[7]=monthly_target_hit_rate (×0.45); net demoted to ÷20k×0.15;
        // Sharpe 0.20→0.10, consistency 0.15→0.10, PF 0.20→0.15.
        // Pin genome has monthly_hit=0 (the `metrics` helper leaves slot 7 = 0):
        //   net=1000, sharpe=2, dd=0.05, wr=0.60, pf=1.8, trades=100, consistency=0.70.
        //   activity_mult = 1.0 ; conf = 1.0
        //   hit  = 0.0 * 0.45 = 0.0
        //   ret  = (1000/20000).clamp(±2) = 0.05 → * 0.15 = 0.0075
        //   sh   = 2.0 (clamped) * 1.0 = 2.0 → * 0.10 = 0.20
        //   cons = 0.70 → * 0.10 = 0.07
        //   pf   = (1.8 - 1.0) * 0.5 = 0.40 → * 0.15 = 0.06
        //   wr   = (0.60 - 0.45) * 2.0 = 0.30 → * 0.10 = 0.03
        //   dd   = 0.05 * 15.0 = 0.75
        //   total = (0.0 + 0.0075 + 0.20 + 0.07 + 0.06 + 0.03) * 1.0 - 0.75 = -0.3825
        // NOTE: a genome with ZERO consistency (never hits 4%/month) scores NEGATIVE
        // even at a positive net — exactly the lumpy case v3 is built to reject.
        let m = metrics(1000.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        let s = ga_fitness(&m);
        assert!(
            (s - (-0.3825)).abs() < 1e-5,
            "GA fitness pin (scoring_version 3) broken: expected -0.3825, got {}",
            s
        );
    }

    #[test]
    fn ga_fitness_low_trade_count_receives_reduced_positive_score() {
        // GA Fix B (taskdoc #274): a candidate with only 5 trades
        // should score lower than the same Sharpe/PF/etc. with 100
        // trades. The activity multiplier ramps from 0.3 (no trades)
        // to 1.0 (>=30 trades).
        let low = metrics(100.0, 2.0, 0.05, 0.60, 1.8, 12.0, 5.0, 0.70);
        let high = metrics(100.0, 2.0, 0.05, 0.60, 1.8, 12.0, 100.0, 0.70);
        let s_low = ga_fitness(&low);
        let s_high = ga_fitness(&high);
        assert!(s_low.is_finite() && s_high.is_finite());
        assert!(
            s_low < s_high,
            "low-trade (5) candidate must score lower than high-trade (100): {} vs {}",
            s_low,
            s_high
        );
    }
}
