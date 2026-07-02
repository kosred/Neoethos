//! Risk-constrained Kelly sizing — Busseti, Ryu & Boyd, "Risk-Constrained
//! Kelly Gambling" (arXiv:1603.06183 / Stanford).
//!
//! Classic Kelly maximises asymptotic log-growth but says nothing about the
//! PATH — a full-Kelly bettor routinely visits drawdowns that would end a
//! small trader's account (or a prop-firm challenge). The paper adds the
//! constraint the operator actually cares about:
//!
//! > P( wealth ever falls below `dd_level` × starting wealth ) ≤ `dd_prob`
//!
//! and shows the constraint holds whenever
//!
//! >   E[(1 + f·r)^(−λ)] ≤ 1,   λ = ln(dd_prob) / ln(dd_level)
//!
//! (a supermartingale bound; both logs are negative so λ > 0). For our
//! two-outcome per-trade model — win `+rr·f`, lose `−f` with probability
//! `p` / `1−p` — the expectation is closed-form and the feasible-f boundary
//! is found by bisection; no convex solver needed.
//!
//! This module is a pure function: SIZING ADVICE derived from measured
//! (win_rate, reward_to_risk). It deliberately does NOT touch the
//! `risky_mode` stage ladder (operator directive §7.1 fixes that band at
//! 30–50% with signed 99%-ruin acceptance) — consumers surface it as the
//! recommended `risk_per_trade` for survival-constrained accounts.

/// Largest per-trade risk fraction `f` such that
/// `P(drawdown below dd_level·W₀) ≤ dd_prob`, capped at the full-Kelly
/// growth optimum `f* = p − (1−p)/rr` (risking more than Kelly only adds
/// variance, never growth).
///
/// Returns `0.0` when there is no edge (`p·rr ≤ 1−p`) or on nonsensical
/// inputs — a fail-safe "don't bet" answer, never an error.
///
/// * `win_rate` — probability of a winning trade, in (0, 1).
/// * `reward_to_risk` — avg win / avg loss in R terms, > 0.
/// * `dd_level` — wealth floor as a fraction of start (0.5 = "half"), in (0, 1).
/// * `dd_prob` — acceptable probability of ever hitting that floor, in (0, 1).
pub fn risk_constrained_kelly(
    win_rate: f64,
    reward_to_risk: f64,
    dd_level: f64,
    dd_prob: f64,
) -> f64 {
    let p = win_rate;
    let rr = reward_to_risk;
    if !(p > 0.0 && p < 1.0)
        || !(rr > 0.0 && rr.is_finite())
        || !(dd_level > 0.0 && dd_level < 1.0)
        || !(dd_prob > 0.0 && dd_prob < 1.0)
    {
        return 0.0;
    }
    // Full Kelly for the asymmetric two-outcome bet: f* = p − (1−p)/rr.
    let f_kelly = p - (1.0 - p) / rr;
    if f_kelly <= 0.0 {
        return 0.0; // no edge — the only safe size is zero
    }
    let f_kelly = f_kelly.min(0.999);

    let lambda = dd_prob.ln() / dd_level.ln(); // > 0
    // Constraint slack g(f) = E[(1+f·r)^(−λ)] − 1; feasible ⇔ g ≤ 0.
    let g = |f: f64| -> f64 {
        p * (1.0 + rr * f).powf(-lambda) + (1.0 - p) * (1.0 - f).powf(-lambda) - 1.0
    };

    // g(0) = 0 and, with positive edge, g dips negative before rising to +∞
    // as f → 1. If the growth optimum is already feasible, take it.
    if g(f_kelly) <= 0.0 {
        return f_kelly;
    }
    // Otherwise the constraint boundary lies in (0, f_kelly): bisect for the
    // upper root of g. Seed `lo` just off zero (g(lo) < 0 for any edge).
    let mut lo = f_kelly * 1e-9;
    let mut hi = f_kelly;
    if g(lo) > 0.0 {
        return 0.0; // pathological (λ enormous) — nothing is feasible
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if g(mid) <= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audit-doc reference edge: p=0.55, rr=2.0 ⇒ full Kelly 0.325.
    const P: f64 = 0.55;
    const RR: f64 = 2.0;

    #[test]
    fn tolerant_constraint_returns_full_kelly() {
        // Accepting a 90% chance of dipping to 5% of start is looser than
        // anything full Kelly does — the growth optimum must come back.
        let f = risk_constrained_kelly(P, RR, 0.05, 0.90);
        assert!((f - 0.325).abs() < 1e-9, "expected full Kelly 0.325, got {f}");
    }

    #[test]
    fn strict_constraint_sizes_below_kelly_and_is_monotonic() {
        // "≤5% chance of ever losing half" must bind well below Kelly...
        let strict = risk_constrained_kelly(P, RR, 0.5, 0.05);
        assert!(
            strict > 0.0 && strict < 0.325,
            "strict RCK must land in (0, f_kelly), got {strict}"
        );
        // ...and loosening the probability budget must monotonically raise f.
        let looser = risk_constrained_kelly(P, RR, 0.5, 0.20);
        assert!(
            looser > strict,
            "20% budget must allow more risk than 5%: {looser} vs {strict}"
        );
        // Verify the returned size actually satisfies the bound equation.
        let lambda = (0.05f64).ln() / (0.5f64).ln();
        let g = P * (1.0 + RR * strict).powf(-lambda)
            + (1.0 - P) * (1.0 - strict).powf(-lambda)
            - 1.0;
        assert!(g <= 1e-9, "returned f must satisfy E[(1+f·r)^-λ] ≤ 1, g={g}");
    }

    #[test]
    fn no_edge_returns_zero() {
        assert_eq!(risk_constrained_kelly(0.50, 1.0, 0.5, 0.05), 0.0);
        assert_eq!(risk_constrained_kelly(0.30, 1.5, 0.5, 0.05), 0.0);
    }

    #[test]
    fn garbage_inputs_return_zero_not_panic() {
        assert_eq!(risk_constrained_kelly(f64::NAN, 2.0, 0.5, 0.05), 0.0);
        assert_eq!(risk_constrained_kelly(0.55, f64::INFINITY, 0.5, 0.05), 0.0);
        assert_eq!(risk_constrained_kelly(0.55, 2.0, 1.5, 0.05), 0.0);
        assert_eq!(risk_constrained_kelly(0.55, 2.0, 0.5, 0.0), 0.0);
        assert_eq!(risk_constrained_kelly(1.0, 2.0, 0.5, 0.05), 0.0);
    }
}
