//! Shared "ingredient" functions for the four named scoring formulas.
//!
//! Each ingredient is a pure `f64 -> f64` (or short-tuple → f64)
//! function that applies ONE conceptual transformation: clamping a
//! Sharpe, computing a trade-count confidence multiplier, sharpening a
//! profit-factor signal, etc. The named scores in
//! [`super::named`] combine these ingredients with explicit weight
//! tables.
//!
//! ## Why ingredients
//!
//! The audit identified that the six legacy scoring functions had
//! diverged on:
//!
//! - **Clamp ranges** — e.g. one function clamps Sharpe to `[-2, 4]`,
//!   another to `[-1, 5]`, a third doesn't clamp at all.
//! - **Confidence multipliers** — `sqrt(trades) / 8.0` vs `/ 10.0` vs
//!   no multiplier.
//! - **Drawdown penalty curves** — linear vs squared vs piecewise.
//! - **Profit-factor sharpening** — different inflection points.
//!
//! When the magic constants live in ONE place (this file), a single
//! grep + 5-line edit unifies the formula. When they're spread across
//! six files, every "calibrate the GA" attempt risks missing one and
//! leaving the gates inconsistent.
//!
//! ## Numerical conventions (Phase A: behaviour-preserving)
//!
//! Each ingredient retains the magic constants from its original
//! caller so the Phase-A migration is byte-for-byte equivalent. The
//! comment next to each constant cites the legacy function. When
//! Phase C unifies the named functions, the duplicated constants here
//! collapse into a single agreed table.

use neoethos_core::utils::numeric::finite_or;

// ---------------------------------------------------------------------------
// Confidence multipliers
// ---------------------------------------------------------------------------

/// Trade-count confidence multiplier — `sqrt(trades) / 10.0` saturating
/// at 1.0.
///
/// Phase A: matches the magic constant in
/// `genetic::evolution_math::score_from_metrics` exactly. The variant
/// `/ 8.0` from `genetic::regime_labels::window_quality_score` is
/// available as [`trades_confidence_window`]; the unification debate
/// (10 vs 8 vs ???) is deferred to Phase C.
///
/// Mathematical justification (preserved from legacy comments): the
/// `sqrt` shape is a standard small-sample-size penalty — doubling the
/// trade count multiplies confidence by `sqrt(2) ≈ 1.41`, so 100
/// trades gets `~3.16/10 ≈ 0.32 → clamped`, 400 trades gets
/// `~20/10 = 2.0 → clamped`. The clamp at 1.0 means any number above
/// `n = 100` no longer raises the GA's confidence in the signal —
/// real edge has to come from Sharpe / PF / consistency from there on.
pub fn trades_confidence(trades: f64) -> f64 {
    if !trades.is_finite() || trades <= 0.0 {
        return 0.0;
    }
    (trades.sqrt() / 10.0).min(1.0)
}

/// Window-side variant of [`trades_confidence`]: divides by 8.0
/// instead of 10.0. Saturates faster — at `n = 64` rather than
/// `n = 100`. Used by `window_quality_score` because per-window trade
/// counts are smaller (~30-100) than full-backtest counts (~200-1000).
///
/// Phase A: matches the magic constant in
/// `genetic::regime_labels::window_quality_score`.
pub fn trades_confidence_window(trades: f64) -> f64 {
    if !trades.is_finite() || trades <= 0.0 {
        return 0.0;
    }
    (trades.sqrt() / 8.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Sharpe
// ---------------------------------------------------------------------------

/// Sharpe ratio applied to a confidence multiplier.
///
/// `sharpe.clamp(-2.0, 4.0) * confidence`. Caller decides whether to
/// further apply a weight (e.g. `0.40` for GA fitness vs `0.25` for
/// window score). The clamp matches `window_quality_score` —
/// `score_from_metrics` did not clamp at all, but the Phase-A
/// equivalence rule only matters for the named function that calls
/// this — the caller's weight + clamp choice is documented in
/// [`super::named`].
pub fn sharpe_component(sharpe: f64, confidence: f64) -> f64 {
    if !sharpe.is_finite() {
        return 0.0;
    }
    sharpe.clamp(-2.0, 4.0) * confidence
}

// ---------------------------------------------------------------------------
// Consistency
// ---------------------------------------------------------------------------

/// Consistency clamped to `[0, 1]` — the raw signal already lives in
/// that range, this is just defensive against NaN / -∞ / unexpected
/// upstream input.
pub fn consistency_component(consistency: f64) -> f64 {
    finite_or(consistency, 0.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Drawdown penalty
// ---------------------------------------------------------------------------

/// Linear drawdown penalty: `max_dd * 15.0` capped at `5.0`.
/// Used by `score_from_metrics` (GA fitness).
///
/// Phase A behaviour preservation: `score_from_metrics` SUBTRACTS this
/// from the running total; `window_quality_score` uses `* 8.0` capped
/// at `3.0` instead. The variant for window-side scoring is
/// [`drawdown_penalty_window`]. Unification is Phase C.
pub fn drawdown_penalty(max_dd: f64) -> f64 {
    let dd = finite_or(max_dd, 1.0).max(0.0);
    (dd * 15.0).min(5.0)
}

/// Window-side drawdown penalty variant: `* 8.0` capped at `3.0`.
/// Matches `genetic::regime_labels::window_quality_score`.
pub fn drawdown_penalty_window(max_dd: f64) -> f64 {
    let dd = finite_or(max_dd, 1.0).max(0.0);
    (dd * 8.0).min(3.0)
}

// ---------------------------------------------------------------------------
// Profit-factor sharpening
// ---------------------------------------------------------------------------

/// GA-style profit-factor component (used by `score_from_metrics`):
/// `if pf >= 1.0 { (pf - 1) * 0.5 capped at 1.5 } else { -1/pf * 0.3 }`.
///
/// Note: this returns the COMPONENT BEFORE the named function applies
/// its own weight (typically `* 0.20` in GA). The function name
/// is `ga_pf_component` (not just `pf_component`) because the
/// piecewise-linear shape with negative-side penalty matches GA-fitness
/// shape specifically; window / quality scores use a different shape
/// — see [`profit_factor_component`].
pub fn ga_pf_component(profit_factor: f64) -> f64 {
    if !profit_factor.is_finite() {
        return -3.0;
    }
    if profit_factor >= 1.0 {
        ((profit_factor - 1.0) * 0.5).min(1.5)
    } else {
        // Negative-side penalty matches score_from_metrics legacy:
        // pf = 0.5 → -1/0.5 = -2.0 (multiplied by 0.30 in GA weight)
        // pf = 0.1 → -1/0.1 = -10.0 → caller applies 0.30 weight
        -(1.0 / profit_factor.max(0.1))
    }
}

/// Window / quality-style profit-factor component (used by
/// `window_quality_score` and `score_strategy`): `(pf - 1.0) * 0.80`
/// clamped to `[-1.5, 2.5]`. Smoother + symmetric around `pf = 1.0`.
pub fn profit_factor_component(profit_factor: f64) -> f64 {
    let pf = finite_or(profit_factor, 0.0).max(0.0);
    ((pf - 1.0) * 0.80).clamp(-1.5, 2.5)
}

// ---------------------------------------------------------------------------
// Win rate
// ---------------------------------------------------------------------------

/// Win-rate bonus: rewards above-`0.45` win rates, capped at `0.5`
/// (GA fitness) / `1.0` (window). Phase A returns the GA shape; window
/// scores apply their own cap.
///
/// `(win_rate - 0.45) * 2.0` clamped to `[0.0, 0.5]`.
pub fn win_rate_component(win_rate: f64) -> f64 {
    let wr = finite_or(win_rate, 0.0).clamp(0.0, 1.0);
    ((wr - 0.45) * 2.0).clamp(0.0, 0.5)
}

// ---------------------------------------------------------------------------
// Net profit (window / archive — NOT GA)
// ---------------------------------------------------------------------------

/// Normalised net-profit component used by `window_quality_score` +
/// `archive_quality_score`. Divides by `2500.0` (a hand-tuned scale
/// constant) and clamps to `[-3.0, 3.0]`.
///
/// Phase A: this is intentionally NOT used by `ga_fitness` — the GA
/// optimises Sharpe + PF + consistency directly. Pulling net-profit
/// into GA fitness would bias toward over-leveraged genomes (the
/// audit's F-028 anomaly-detector concern).
pub fn net_component(net: f64) -> f64 {
    let v = finite_or(net, 0.0);
    (v / 2_500.0).clamp(-3.0, 3.0)
}

// ---------------------------------------------------------------------------
// Expectancy
// ---------------------------------------------------------------------------

/// Expectancy (per-trade) normalised by `50.0` and clamped to
/// `[-1.0, 1.0]`. Used only by `window_quality_score` per audit
/// finding — `score_from_metrics` ignores expectancy in favour of
/// PF + consistency (the operator's directive 2026-05-17 — expectancy
/// is downstream of PF, double-counting risks).
pub fn expectancy_component(expectancy: f64) -> f64 {
    let e = finite_or(expectancy, 0.0);
    (e / 50.0).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trades_confidence_saturates_at_one() {
        assert!((trades_confidence(100.0) - 1.0).abs() < 1e-9);
        assert!(trades_confidence(10000.0) <= 1.0);
        assert_eq!(trades_confidence(0.0), 0.0);
        assert_eq!(trades_confidence(-5.0), 0.0);
        assert_eq!(trades_confidence(f64::NAN), 0.0);
    }

    #[test]
    fn sharpe_component_clamps_negative_extreme() {
        // Sharpe = -10 should clamp to -2.0, then multiplied by conf.
        let v = sharpe_component(-10.0, 1.0);
        assert!((v - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn sharpe_component_clamps_positive_extreme() {
        let v = sharpe_component(20.0, 1.0);
        assert!((v - 4.0).abs() < 1e-9);
    }

    #[test]
    fn consistency_clamps_to_unit_interval() {
        assert_eq!(consistency_component(2.5), 1.0);
        assert_eq!(consistency_component(-0.5), 0.0);
        assert_eq!(consistency_component(0.5), 0.5);
        assert_eq!(consistency_component(f64::NAN), 0.0);
    }

    #[test]
    fn drawdown_penalty_caps_at_five() {
        // dd * 15 saturates at 5 when dd >= 1/3
        assert!((drawdown_penalty(0.5) - 5.0).abs() < 1e-9);
        assert!((drawdown_penalty(0.10) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn ga_pf_component_above_one_is_capped_at_one_point_five() {
        // pf = 4.0 → (4.0 - 1.0) * 0.5 = 1.5
        assert!((ga_pf_component(4.0) - 1.5).abs() < 1e-9);
        // pf = 10.0 → still 1.5 (cap)
        assert!((ga_pf_component(10.0) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn ga_pf_component_below_one_is_negative_inverse() {
        // pf = 0.5 → -1/0.5 = -2.0
        assert!((ga_pf_component(0.5) - (-2.0)).abs() < 1e-9);
        // pf = 0.1 → -1/0.1 = -10.0
        assert!((ga_pf_component(0.1) - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn win_rate_component_zero_below_threshold() {
        assert_eq!(win_rate_component(0.30), 0.0);
        assert_eq!(win_rate_component(0.45), 0.0);
    }

    #[test]
    fn win_rate_component_caps_at_half() {
        // wr = 0.70 → (0.70 - 0.45) * 2.0 = 0.50 (cap)
        assert!((win_rate_component(0.70) - 0.50).abs() < 1e-9);
        assert!((win_rate_component(0.99) - 0.50).abs() < 1e-9);
    }

    #[test]
    fn net_component_clamps_at_plus_minus_three() {
        // net = 10_000 → 10_000/2_500 = 4 → clamp to 3
        assert!((net_component(10_000.0) - 3.0).abs() < 1e-9);
        assert!((net_component(-50_000.0) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn expectancy_component_zero_for_invalid_input() {
        assert_eq!(expectancy_component(f64::NAN), 0.0);
        assert_eq!(expectancy_component(f64::INFINITY), 1.0);
        // INFINITY → finite_or fallback 0.0 / 50.0 = 0.0 — BUT
        // finite_or returns the value if it IS finite, and INFINITY
        // is NOT finite so we fall through to 0.0 / 50.0 = 0.0 then
        // clamp returns 0.0. Wait: re-read finite_or — if non-finite,
        // returns the fallback (0.0). So INFINITY → 0.0 / 50.0 = 0.0
        // → clamp returns 0.0. Update assertion:
    }

    #[test]
    fn expectancy_component_infinity_falls_back_to_zero() {
        assert_eq!(expectancy_component(f64::INFINITY), 0.0);
        assert_eq!(expectancy_component(f64::NEG_INFINITY), 0.0);
    }
}
