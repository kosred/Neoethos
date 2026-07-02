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
    consistency_component, drawdown_penalty, drawdown_penalty_window, expectancy_component,
    ga_pf_component, net_component, profit_factor_component, sharpe_component, trades_confidence,
    trades_confidence_window, win_rate_component,
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
/// months hitting the operator's ≥4%/month bar (metrics[7], the same consistency
/// the gate checks). (v1 = Sharpe-only; v2 = total-net.) Runs before this are NOT
/// directly comparable (different fitness landscape); old artifacts still deserialize.
pub const SCORING_VERSION_CURRENT: ScoringVersion = ScoringVersion(4);

// ---------------------------------------------------------------------------
// ga_fitness — was `genetic::evolution_math::score_from_metrics`
// ---------------------------------------------------------------------------

/// GA fitness — the value the genetic algorithm MAXIMISES per genome.
///
/// Matches `genetic::evolution_math::score_from_metrics` byte-for-byte.
/// The mapping in `metrics: &[f64; 11]` is the canonical
/// `[BacktestMetrics::to_metric_array]` order (see `eval.rs` lines
/// 165-200 and `BACKTEST_METRICS_RESERVED_INDEX_7`):
///
/// ```text
///   metrics[0] = net_profit      metrics[5] = profit_factor
///   metrics[1] = sharpe          metrics[6] = expectancy
///   metrics[2] = peak_equity     metrics[7] = RESERVED (always 0.0)
///   metrics[3] = max_drawdown    metrics[8] = trade_count
///   metrics[4] = win_rate        metrics[9] = consistency
///                                metrics[10] = max_daily_drawdown
/// ```
///
/// Sentinel: returns `f64::NEG_INFINITY` when Sharpe is non-finite OR
/// trade_count < 1 → caller (the GA selection step) treats this as a
/// "do not propagate" marker.
pub fn ga_fitness(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    // slot 7 (scoring_version 3): monthly_target_hit_rate — fraction of months hitting
    // the operator's >=4% bar. The CONSISTENT-monthly-return signal (see eval.rs).
    let monthly_hit = metrics[7];
    let trades = metrics[8];
    let consistency = metrics[9];

    // Sharpe non-finite is still a hard reject — the metric is unusable
    // and any score derived from it is meaningless.
    if !sharpe.is_finite() {
        return f64::NEG_INFINITY;
    }

    let trades_f = trades;

    // **2026-05-26 (GA Fix B — graduated fitness, taskdoc #274)**: the
    // Python prototype this engine replaced learned to NEVER trade
    // because `max_dd == 0` trivially satisfies the operator's
    // `max_dd <= 4%` filter. Returning `f64::NEG_INFINITY` for the
    // zero-trade case makes 0-trade and 1-trade candidates indistinguishable,
    // denying the GA any fitness gradient to escape this reward-hack —
    // every Discovery cycle ends with an empty archive.
    //
    // The fix is a graduated penalty:
    //   1. Zero-trade returns a STRONGLY-NEGATIVE-BUT-FINITE score
    //      (-100.0) so any trading strategy beats it but mutation can
    //      still reach trade-firing genes through small fitness deltas.
    //   2. Trade-count up to ~30/month earns an `activity` multiplier
    //      [0.033, 1.0] that scales the positive-fitness components.
    //      30 trades/month is a typical pro-trader pace.
    //   3. The drawdown PENALTY is unscaled — we still want the GA to
    //      avoid blowups, just not via the "do nothing" loophole.
    //
    // Side effect: the previous Phase-A pin test
    // (`ga_fitness_matches_legacy_score_from_metrics_pin`) updates from
    // 0.335 to ~0.335 because 100 trades / 30 = 3.33 → activity clamped
    // to 1.0 → multiplier (0.3 + 0.7 * 1.0) = 1.0 → math unchanged.
    if trades_f < 1.0 {
        return -100.0;
    }

    // Activity bonus: 1 trade/month = 0.033, 30+/month = 1.0. We can't
    // know the actual run-month-count from a `[f64;11]` here, so we use
    // a coarse proxy: trade-count itself capped at 30. This is fine
    // because the GA's purpose at this level is to PREFER trade-firing
    // genes — once the population has them, downstream gating
    // (passes_filter, quality_score) applies stricter trades_per_month
    // checks.
    let activity = (trades_f / 30.0).clamp(0.0, 1.0);
    let activity_mult = 0.3 + 0.7 * activity;

    let conf = trades_confidence(trades);
    // **2026-06-06 (scoring_version 3 — CONSISTENT-monthly-return GA)**:
    // v2 rewarded TOTAL net (net/12k), but compounding made that reward LUMPY — the GA
    // converged on genes with huge AGGREGATE return concentrated in a few periods
    // (in-sample Sharpe 6-13) that ALL failed the prop-firm window-consistency gate
    // (best gene passed only ~11% of 60-day windows vs the 65% floor). High total net
    // ≠ a temporally-stable edge.
    //
    // The DOMINANT reward is now `monthly_hit` (metrics[7]) = the fraction of months
    // that hit the operator's >=4%/month bar — the SAME consistency the prop-firm gate
    // checks, so the GA now searches FOR what survives the gate. `net` is kept as a
    // smaller magnitude bonus (so among equally-consistent genes the GA still prefers
    // bigger returns). `sharpe`/`consistency` are demoted: both are monthly mean/std
    // (consistency = sharpe/3.46 clamped), which a few big months inflate — they do
    // NOT predict the window pass-rate, which is exactly why v2 over-rewarded lumpy
    // genes. The DD penalty stays UNSCALED so blow-ups are still rejected.
    let hit = monthly_hit.clamp(0.0, 1.0) * 0.45;
    let ret = (net / 20_000.0).clamp(-2.0, 2.0) * 0.15;
    let sh = sharpe_component(sharpe, conf) * 0.10;
    let cons = consistency_component(consistency) * 0.10;
    let pf = ga_pf_component(profit_factor)
        * if profit_factor >= 1.0 { 0.15 } else { 0.25 };
    let wr = win_rate_component(win_rate) * 0.10;
    let dd = drawdown_penalty(max_dd);

    // **2026-07-02 (scoring_version 4 — STEADY-INCOME worst-period penalty)**:
    // the operator's product goal is a stable MONTHLY income, and the missing
    // half of "stable" was the DOWNSIDE: v3 rewarded frequent >=4% months
    // (slot 7) but nothing punished the occasional CATASTROPHIC period that
    // sets a small account back months. `max_daily_drawdown` (slot 10) was
    // computed by the evaluator and IGNORED by the GA. Penalize it now —
    // catastrophic months are built from catastrophic days, so the daily
    // granularity is the stricter, earlier signal. Weight 10.0 (vs overall-DD's
    // 15.0): a 3% worst-day costs 0.30 — decisive between otherwise-equal
    // genes, not dominant over the whole objective. Like the DD penalty it is
    // NOT activity-scaled: a blow-up day disqualifies regardless of activity.
    let max_daily_dd = metrics[10];
    let daily_dd_pen = max_daily_dd.clamp(0.0, 1.0) * 10.0;

    // Positive components scaled by activity so low-trade candidates (1–5 trades
    // over the whole window) cannot win on noise; the DD penalties are NOT scaled —
    // full weight, rejects blow-ups even when return is high.
    (hit + ret + sh + cons + pf + wr) * activity_mult - dd - daily_dd_pen
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
            net, sharpe, 0.0, max_dd, win_rate, pf, expectancy, 0.0, trades, consistency, 0.0,
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
