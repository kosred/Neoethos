//! Backend-independent Search objective authority.
//!
//! CUDA and CPU oracles bind the semantic identity below. The functions keep
//! the current named-search formulas in one dependency that has no GPU runtime.

pub const PROPFIRM_GA_FITNESS_V4_SEMANTICS: &str = concat!(
    "neoethos.search.objective.propfirm-v4;",
    "slots=net,sharpe,peak,max-dd,win-rate,pf,expectancy,monthly-hit,trades,consistency,max-daily-dd;",
    "zero-trades=-100;nonfinite-sharpe=-inf"
);

pub const RISKY_GA_FITNESS_GROWTH_V5_SEMANTICS: &str = concat!(
    "neoethos.search.objective.risky-growth-v5;",
    "half-kelly;fraction-cap=0.25;pf-cap=10;win-rate-cap=0.99;",
    "zero-trades=-100;nonfinite-sharpe=-inf"
);

pub const RISKY_GA_FITNESS_GOAL_V6_SEMANTICS: &str = concat!(
    "neoethos.search.objective.risky-growth-goal-v6;",
    "realized-net/actual-initial-equity;ln1p;observed-calendar-days;",
    "reference-log-target-minus-log-start;deadline;one-minus-squared-relative-shortfall;",
    "zero-trades=-100;invalid-context-or-metrics-or-wipeout=-inf;pace-proxy-not-goal-proof"
);

/// Checked resident transport interpretation. Unlike the legacy scalar API,
/// this distinguishes an economically unusable candidate from corrupt inputs
/// or arithmetic. The negative-infinity monthly Sharpe marker is authenticated
/// by the sealed producer: only finite nonpositive month-start equity emits it;
/// any non-finite producer arithmetic emits NaN instead.
pub const RESIDENT_ECONOMIC_REJECTION_V2_SEMANTICS: &str = concat!(
    "neoethos.resident-economic-rejection.v2;",
    "monthly-sharpe=-inf-only-finite-nonpositive-month-equity-and-dd>=1;",
    "producer-arithmetic-fault=nan;other-ten-metrics-finite;",
    "v6-finite-wipeout=-inf;arithmetic-overflow=fault;",
    "rejected-score-keeps-negative-infinity;rank-key=1;fault-key=0;",
    "finite-only-fitness-normalization;all-candidates-novelty;archive-rules-unchanged"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentEconomicRejectReasonV2 {
    NonPositiveMonthlyEquity,
    DrawdownAtOrAboveOne,
    NonPositiveRelativeEquity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentScoringFaultV2 {
    InvalidContext,
    NonFiniteMetric { slot: usize },
    InconsistentMonthlyEquityMarker,
    InvalidMetricDomain,
    NonFiniteArithmetic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidentScoringOutcomeV2 {
    Finite(f64),
    EconomicReject(ResidentEconomicRejectReasonV2),
    Fault(ResidentScoringFaultV2),
}

/// Only use with current sealed producer semantics, never to authenticate an
/// arbitrary external metric array. Shape/identity/producer provenance remain
/// the caller's responsibility; this function does not grant archive admission.
pub fn classify_resident_metrics_v2(
    metrics: &[f64; 11],
) -> Result<Option<ResidentEconomicRejectReasonV2>, ResidentScoringFaultV2> {
    for (slot, value) in metrics.iter().enumerate() {
        if !value.is_finite() && !(slot == 1 && *value == f64::NEG_INFINITY) {
            return Err(ResidentScoringFaultV2::NonFiniteMetric { slot });
        }
    }
    if metrics[1] == f64::NEG_INFINITY {
        if metrics[3] < 1.0 {
            return Err(ResidentScoringFaultV2::InconsistentMonthlyEquityMarker);
        }
        return Ok(Some(
            ResidentEconomicRejectReasonV2::NonPositiveMonthlyEquity,
        ));
    }
    Ok(None)
}

fn checked_resident_legacy_score_v2(
    metrics: &[f64; 11],
    score: impl FnOnce(&[f64; 11]) -> f64,
) -> ResidentScoringOutcomeV2 {
    match classify_resident_metrics_v2(metrics) {
        Err(fault) => ResidentScoringOutcomeV2::Fault(fault),
        Ok(Some(reason)) => ResidentScoringOutcomeV2::EconomicReject(reason),
        Ok(None) => match score(metrics) {
            value if value.is_finite() => ResidentScoringOutcomeV2::Finite(value),
            _ => ResidentScoringOutcomeV2::Fault(ResidentScoringFaultV2::NonFiniteArithmetic),
        },
    }
}

pub fn checked_resident_prop_firm_score_v4(metrics: &[f64; 11]) -> ResidentScoringOutcomeV2 {
    checked_resident_legacy_score_v2(metrics, score_prop_firm_ga_fitness_v4)
}

pub fn checked_resident_growth_score_v5(metrics: &[f64; 11]) -> ResidentScoringOutcomeV2 {
    checked_resident_legacy_score_v2(metrics, score_risky_ga_fitness_growth_v5)
}

pub fn checked_resident_goal_score_v6(
    metrics: &[f64; 11],
    initial_equity: f64,
    span_days: f64,
    goal: RiskyGrowthGoal,
) -> ResidentScoringOutcomeV2 {
    use ResidentScoringOutcomeV2::{EconomicReject, Fault, Finite};
    if goal.validate().is_err()
        || !initial_equity.is_finite()
        || initial_equity <= 0.0
        || !span_days.is_finite()
        || span_days <= 0.0
    {
        return Fault(ResidentScoringFaultV2::InvalidContext);
    }
    let metric_reason = match classify_resident_metrics_v2(metrics) {
        Ok(reason) => reason,
        Err(fault) => return Fault(fault),
    };
    if metrics[3] < 0.0 || metrics[8] < 0.0 {
        return Fault(ResidentScoringFaultV2::InvalidMetricDomain);
    }
    if let Some(reason) = metric_reason {
        return EconomicReject(reason);
    }
    if metrics[3] >= 1.0 {
        return EconomicReject(ResidentEconomicRejectReasonV2::DrawdownAtOrAboveOne);
    }
    // Preserve the scalar's order: an overflowing ratio is not evidence of a
    // finite measured wipeout. No pace or square arithmetic runs after rejection.
    let relative_net = metrics[0] / initial_equity;
    if !relative_net.is_finite() {
        return Fault(ResidentScoringFaultV2::NonFiniteArithmetic);
    }
    if relative_net <= -1.0 {
        return EconomicReject(ResidentEconomicRejectReasonV2::NonPositiveRelativeEquity);
    }
    let score = score_risky_ga_fitness_goal_v6(metrics, initial_equity, span_days, goal);
    if score.is_finite() {
        Finite(score)
    } else {
        Fault(ResidentScoringFaultV2::NonFiniteArithmetic)
    }
}

/// Reference capital ratio and calendar deadline, in the same currency.
/// This does not change the actual simulation capital or authorize risk sizing.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskyGrowthGoal {
    pub start_balance: f64,
    pub target_balance: f64,
    pub horizon_days: f64,
}

impl RiskyGrowthGoal {
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.start_balance.is_finite()
            || self.start_balance <= 0.0
            || !self.target_balance.is_finite()
            || self.target_balance <= self.start_balance
            || !self.horizon_days.is_finite()
            || self.horizon_days <= 0.0
            || !(self.target_balance.ln() - self.start_balance.ln()).is_finite()
            || self.target_balance.ln() - self.start_balance.ln() <= 0.0
        {
            return Err("growth goal requires finite positive capital/deadline and target > start");
        }
        Ok(())
    }
}

/// Risky v6 realized-balance pace proxy, not a goal-achievement probability or
/// a replay at the reference starting capital. The actual simulation equity
/// and observed calendar span are required; terminal open PnL is not included.
/// Arithmetic and rejection order are the canonical CPU named-score contract.
pub fn score_risky_ga_fitness_goal_v6(
    metrics: &[f64; 11],
    initial_equity: f64,
    span_days: f64,
    goal: RiskyGrowthGoal,
) -> f64 {
    if goal.validate().is_err()
        || !initial_equity.is_finite()
        || initial_equity <= 0.0
        || !span_days.is_finite()
        || span_days <= 0.0
        || !metrics[0].is_finite()
        || !metrics[1].is_finite()
        || !metrics[3].is_finite()
        || !(0.0..1.0).contains(&metrics[3])
        || !metrics[8].is_finite()
        || metrics[8] < 0.0
    {
        return f64::NEG_INFINITY;
    }
    let relative_net = metrics[0] / initial_equity;
    if !relative_net.is_finite() || relative_net <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if metrics[8] < 1.0 {
        return -100.0;
    }
    let required_log_growth = goal.target_balance.ln() - goal.start_balance.ln();
    let relative_pace =
        relative_net.ln_1p() / required_log_growth * (goal.horizon_days / span_days);
    if !relative_pace.is_finite() {
        return f64::NEG_INFINITY;
    }
    let shortfall = (1.0 - relative_pace).max(0.0);
    1.0 - shortfall * shortfall
}

/// Canonical half-Kelly ceiling used by the Risky v5 search objective.
/// Live sizing consumes the same function so a promoted strategy cannot be
/// searched at one risk fraction and traded at an unrelated hard-coded one.
pub const RISKY_GROWTH_V5_HALF_KELLY_CAP: f64 = 0.25;

/// Resolve the Risky v5 half-Kelly fraction from measured win rate and profit
/// factor. Inputs are bounded exactly as they are in the GPU/CPU scoring
/// contract; no edge (`PF <= 1`) yields zero rather than an invented bet.
#[inline]
pub fn risky_growth_v5_half_kelly_fraction(win_rate: f64, profit_factor: f64) -> f64 {
    let p = win_rate.clamp(0.0, 0.99);
    let pf = profit_factor.clamp(0.0, 10.0);
    let full_kelly = if pf > 1.0 && p > 0.0 {
        p * (pf - 1.0) / pf
    } else {
        0.0
    };
    (full_kelly * 0.5).clamp(0.0, RISKY_GROWTH_V5_HALF_KELLY_CAP)
}

#[inline]
fn trades_confidence(trades: f64) -> f64 {
    (trades.sqrt() / 10.0).min(1.0)
}

#[inline]
fn ga_pf_component(profit_factor: f64) -> f64 {
    if profit_factor >= 1.0 {
        ((profit_factor - 1.0) * 0.5).min(1.5)
    } else {
        -(1.0 / profit_factor.max(0.1))
    }
}

/// Current PropFirm GA objective, scoring version 4.
pub fn score_prop_firm_ga_fitness_v4(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let max_drawdown = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let monthly_hit = metrics[7];
    let trades = metrics[8];
    let consistency = metrics[9];
    let max_daily_drawdown = metrics[10];
    if !sharpe.is_finite() {
        return f64::NEG_INFINITY;
    }
    if trades < 1.0 {
        return -100.0;
    }
    let activity_multiplier = 0.3 + 0.7 * (trades / 30.0).clamp(0.0, 1.0);
    let confidence = trades_confidence(trades);
    let hit = monthly_hit.clamp(0.0, 1.0) * 0.45;
    let net_return = (net / 20_000.0).clamp(-2.0, 2.0) * 0.15;
    let sharpe_score = sharpe.clamp(-2.0, 4.0) * confidence * 0.10;
    let consistency_score = consistency.clamp(0.0, 1.0) * 0.10;
    let profit_factor_score =
        ga_pf_component(profit_factor) * if profit_factor >= 1.0 { 0.15 } else { 0.25 };
    let win_rate_score = ((win_rate.clamp(0.0, 1.0) - 0.45) * 2.0).clamp(0.0, 0.5) * 0.10;
    let drawdown = (max_drawdown.max(0.0) * 15.0).min(5.0);
    let daily_drawdown = max_daily_drawdown.clamp(0.0, 1.0) * 10.0;
    (hit + net_return + sharpe_score + consistency_score + profit_factor_score + win_rate_score)
        * activity_multiplier
        - drawdown
        - daily_drawdown
}

/// Current Risky GA half-Kelly objective, scoring version 5.
pub fn score_risky_ga_fitness_growth_v5(metrics: &[f64; 11]) -> f64 {
    let net = metrics[0];
    let sharpe = metrics[1];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let trades = metrics[8];
    if !sharpe.is_finite() {
        return f64::NEG_INFINITY;
    }
    if trades < 1.0 {
        return -100.0;
    }
    let p = win_rate.clamp(0.0, 0.99);
    let pf = profit_factor.clamp(0.0, 10.0);
    let f = risky_growth_v5_half_kelly_fraction(p, pf);
    let rr = if p > 0.0 { pf * (1.0 - p) / p } else { 0.0 };
    let growth_per_trade = if f > 0.0 && rr > 0.0 {
        p * (1.0 + rr * f).ln() + (1.0 - p) * (1.0 - f).ln()
    } else {
        0.0
    };
    let edge_gradient = (pf - 1.0).clamp(-1.0, 0.0) * 0.05
        + (p - 0.5).clamp(-0.5, 0.0) * 0.05
        + (net / 20_000.0).clamp(-2.0, 0.0) * 0.01;
    growth_per_trade * trades * 10.0 + edge_gradient
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checked_fixture() -> ([f64; 11], RiskyGrowthGoal) {
        let mut metrics = [0.0; 11];
        metrics[8] = 1.0;
        (
            metrics,
            RiskyGrowthGoal {
                start_balance: 1.0,
                target_balance: 2.0,
                horizon_days: 1.0,
            },
        )
    }

    #[test]
    fn checked_economic_goal_has_independent_finite_and_wipeout_vectors() {
        use ResidentScoringOutcomeV2::{EconomicReject, Finite};
        let (mut metrics, goal) = checked_fixture();
        for (net, expected) in [(0.0, 0.0), (100.0, 1.0), (-50.0, -3.0)] {
            metrics[0] = net;
            let Finite(score) = checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal) else {
                panic!("finite measured return must produce a finite score")
            };
            assert!((score - expected).abs() < 1.0e-12);
            assert_eq!(
                score.to_bits(),
                score_risky_ga_fitness_goal_v6(&metrics, 100.0, 1.0, goal).to_bits()
            );
        }
        metrics[0] = -100.0;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            EconomicReject(ResidentEconomicRejectReasonV2::NonPositiveRelativeEquity)
        );
        metrics[0] = -99.999_999_999_999_99;
        assert!(matches!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            Finite(_)
        ));
        metrics[0] = 10.0;
        for drawdown in [1.0, 1.5] {
            metrics[3] = drawdown;
            assert_eq!(
                checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
                EconomicReject(ResidentEconomicRejectReasonV2::DrawdownAtOrAboveOne)
            );
            // V4/V5 do not gain V6's drawdown rejection or an archive exclusion.
            assert!(matches!(
                checked_resident_prop_firm_score_v4(&metrics),
                Finite(_)
            ));
            assert!(matches!(
                checked_resident_growth_score_v5(&metrics),
                Finite(_)
            ));
        }
        metrics[3] = 0.0;
        metrics[8] = 0.0;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            Finite(-100.0)
        );
    }

    #[test]
    fn checked_economic_marker_never_launders_other_nonfinite_slots() {
        use ResidentScoringOutcomeV2::{EconomicReject, Fault};
        let (mut metrics, goal) = checked_fixture();
        metrics[1] = f64::NEG_INFINITY;
        assert_eq!(
            classify_resident_metrics_v2(&metrics),
            Err(ResidentScoringFaultV2::InconsistentMonthlyEquityMarker)
        );
        metrics[3] = 1.0;
        let expected = EconomicReject(ResidentEconomicRejectReasonV2::NonPositiveMonthlyEquity);
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            expected
        );
        assert_eq!(checked_resident_prop_firm_score_v4(&metrics), expected);
        assert_eq!(checked_resident_growth_score_v5(&metrics), expected);
        for slot in 0..11 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                if slot == 1 && invalid == f64::NEG_INFINITY {
                    continue;
                }
                let mut corrupted = metrics;
                corrupted[slot] = invalid;
                assert_eq!(
                    checked_resident_goal_score_v6(&corrupted, 100.0, 1.0, goal),
                    Fault(ResidentScoringFaultV2::NonFiniteMetric { slot })
                );
            }
        }
    }

    #[test]
    fn checked_economic_context_domain_and_arithmetic_faults_take_precedence() {
        use ResidentScoringOutcomeV2::{Fault, Finite};
        let (mut metrics, goal) = checked_fixture();
        metrics[1] = f64::NEG_INFINITY;
        metrics[3] = 1.5;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 0.0, 1.0, goal),
            Fault(ResidentScoringFaultV2::InvalidContext)
        );
        metrics[8] = -1.0;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            Fault(ResidentScoringFaultV2::InvalidMetricDomain)
        );
        metrics[1] = 0.0;
        metrics[8] = 1.0;
        metrics[3] = -0.1;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            Fault(ResidentScoringFaultV2::InvalidMetricDomain)
        );
        metrics[3] = 0.0;
        metrics[0] = f64::MAX;
        metrics[8] = 0.0;
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, f64::MIN_POSITIVE, 1.0, goal),
            Fault(ResidentScoringFaultV2::NonFiniteArithmetic)
        );
        metrics[0] = -50.0;
        metrics[8] = 1.0;
        let overflow_goal = RiskyGrowthGoal {
            horizon_days: 1.0e200,
            ..goal
        };
        // Relative pace is finite, but its squared shortfall overflows. The
        // unchanged scalar returns -infinity; the checked path must NOT label
        // that result an economically measured wipeout.
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 1.0, overflow_goal),
            f64::NEG_INFINITY
        );
        assert_eq!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, overflow_goal),
            Fault(ResidentScoringFaultV2::NonFiniteArithmetic)
        );
        assert!(matches!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            Finite(_)
        ));
    }

    #[test]
    fn checked_economic_api_does_not_change_legacy_scalar_behavior() {
        let (mut metrics, goal) = checked_fixture();
        metrics[6] = f64::NAN; // Not an ingredient of any named scalar.
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 1.0, goal),
            0.0
        );
        assert!(score_prop_firm_ga_fitness_v4(&metrics).is_finite());
        assert!(score_risky_ga_fitness_growth_v5(&metrics).is_finite());
        assert!(matches!(
            checked_resident_goal_score_v6(&metrics, 100.0, 1.0, goal),
            ResidentScoringOutcomeV2::Fault(ResidentScoringFaultV2::NonFiniteMetric { slot: 6 })
        ));
    }

    #[test]
    fn canonical_objectives_keep_named_pins() {
        let mut metrics = [0.0; 11];
        metrics[0] = 1_000.0;
        metrics[1] = 2.0;
        metrics[3] = 0.05;
        metrics[4] = 0.60;
        metrics[5] = 1.8;
        metrics[8] = 100.0;
        metrics[9] = 0.70;
        assert!((score_prop_firm_ga_fitness_v4(&metrics) - -0.3825).abs() < 1.0e-9);
        assert!(score_risky_ga_fitness_growth_v5(&metrics).is_finite());
    }

    #[test]
    fn canonical_goal_v6_keeps_realized_growth_and_context_semantics() {
        let goal = RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        };
        let mut metrics = [0.0; 11];
        metrics[0] = 1_000.0;
        metrics[1] = 1.0;
        metrics[3] = 0.2;
        metrics[8] = 30.0;
        let score = score_risky_ga_fitness_goal_v6(&metrics, 10_000.0, 180.0, goal);
        assert!(score > 0.0 && score < 1.0);
        assert!(score_risky_ga_fitness_goal_v6(&metrics, 1_000.0, 180.0, goal) > score);
        assert!(score_risky_ga_fitness_goal_v6(&metrics, 10_000.0, 90.0, goal) > score);
        metrics[0] = -1_000.0;
        assert!(score_risky_ga_fitness_goal_v6(&metrics, 10_000.0, 180.0, goal) < 0.0);
        metrics[0] = 49_900.0;
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 180.0, goal),
            1.0
        );
        // Context fields are required even for an empty candidate; no default
        // equity, duration or target can turn an invalid context into a score.
        metrics[8] = 0.0;
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 180.0, goal),
            -100.0
        );
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 0.0, goal),
            f64::NEG_INFINITY
        );
        metrics[0] = -100.0;
        assert_eq!(
            score_risky_ga_fitness_goal_v6(&metrics, 100.0, 180.0, goal),
            f64::NEG_INFINITY
        );
    }
}
