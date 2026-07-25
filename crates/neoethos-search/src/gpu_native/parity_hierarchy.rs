//! Causal twelve-level parity hierarchy for CPU/GPU engine validation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum ParityLevel {
    ScoreBeforeThreshold = 1,
    SignalDirection = 2,
    Confidence = 3,
    EntryEvents = 4,
    ExitBarAndReason = 5,
    AcceptedTradeSequence = 6,
    PositionSizeAndCosts = 7,
    EquityAfterTrade = 8,
    CalendarAndPropFirmState = 9,
    FinalMetrics = 10,
    ValidationVerdict = 11,
    SurvivorOrdering = 12,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FloatTolerance {
    pub absolute: f64,
    pub relative: f64,
    pub max_ulps: u64,
}

impl FloatTolerance {
    pub const EXACT: Self = Self {
        absolute: 0.0,
        relative: 0.0,
        max_ulps: 0,
    };

    pub const CANONICAL_F32: Self = Self {
        absolute: 1.0e-6,
        relative: 1.0e-6,
        max_ulps: 8,
    };

    pub const CANONICAL_F64: Self = Self {
        absolute: 1.0e-10,
        relative: 1.0e-9,
        max_ulps: 16,
    };

    pub fn matches_f64(self, expected: f64, actual: f64) -> bool {
        if expected.to_bits() == actual.to_bits() {
            return true;
        }
        if !expected.is_finite() || !actual.is_finite() {
            return false;
        }
        let delta = (expected - actual).abs();
        if delta <= self.absolute {
            return true;
        }
        let scale = expected.abs().max(actual.abs()).max(1.0);
        if delta <= self.relative * scale {
            return true;
        }
        ulp_distance_f64(expected, actual) <= self.max_ulps
    }

    pub fn matches_f32(self, expected: f32, actual: f32) -> bool {
        self.matches_f64(expected as f64, actual as f64)
            || ulp_distance_f32(expected, actual) as u64 <= self.max_ulps
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryEventTrace {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub bar: u32,
    pub direction: i8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExitEventTrace {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub bar: u32,
    pub reason: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeTrace {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub entry_bar: u32,
    pub exit_bar: u32,
    pub direction: i8,
    pub exit_reason: u32,
    pub size: f64,
    pub costs: f64,
    pub pnl: f64,
    pub equity_after: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskStateTrace {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub bar: u32,
    pub day_id: i64,
    pub month_id: i64,
    pub equity: f64,
    pub peak_equity: f64,
    pub day_start_equity: f64,
    pub month_start_equity: f64,
    pub flags: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParityTrace {
    pub scores_before_threshold: Vec<f64>,
    pub signals: Vec<i8>,
    pub confidences: Vec<f32>,
    pub entries: Vec<EntryEventTrace>,
    pub exits: Vec<ExitEventTrace>,
    pub accepted_trades: Vec<TradeTrace>,
    pub risk_states: Vec<RiskStateTrace>,
    pub final_metrics: Vec<[f64; 11]>,
    pub validation_verdicts: Vec<bool>,
    pub survivor_order: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParityPolicy {
    pub score: FloatTolerance,
    pub confidence: FloatTolerance,
    pub size_and_costs: FloatTolerance,
    pub pnl_and_equity: FloatTolerance,
    pub risk_state: FloatTolerance,
    pub metrics: FloatTolerance,
}

impl Default for ParityPolicy {
    fn default() -> Self {
        Self {
            score: FloatTolerance::CANONICAL_F64,
            confidence: FloatTolerance::CANONICAL_F32,
            size_and_costs: FloatTolerance::CANONICAL_F64,
            pnl_and_equity: FloatTolerance::CANONICAL_F64,
            risk_state: FloatTolerance::CANONICAL_F64,
            metrics: FloatTolerance::CANONICAL_F64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityDivergence {
    pub level: ParityLevel,
    pub index: usize,
    pub field: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceComparisonReport {
    pub reference_backend: String,
    pub candidate_backend: String,
    pub first_divergence: Option<ParityDivergence>,
}

impl TraceComparisonReport {
    pub fn is_match(&self) -> bool {
        self.first_divergence.is_none()
    }
}

pub fn compare_traces(
    reference_backend: impl Into<String>,
    candidate_backend: impl Into<String>,
    reference: &ParityTrace,
    candidate: &ParityTrace,
    policy: ParityPolicy,
) -> TraceComparisonReport {
    let first_divergence = compare_float_slice(
        ParityLevel::ScoreBeforeThreshold,
        "score",
        &reference.scores_before_threshold,
        &candidate.scores_before_threshold,
        policy.score,
    )
    .or_else(|| {
        compare_exact_slice(
            ParityLevel::SignalDirection,
            "signal",
            &reference.signals,
            &candidate.signals,
        )
    })
    .or_else(|| {
        compare_f32_slice(
            ParityLevel::Confidence,
            "confidence",
            &reference.confidences,
            &candidate.confidences,
            policy.confidence,
        )
    })
    .or_else(|| {
        compare_exact_slice(
            ParityLevel::EntryEvents,
            "entry",
            &reference.entries,
            &candidate.entries,
        )
    })
    .or_else(|| {
        compare_exact_slice(
            ParityLevel::ExitBarAndReason,
            "exit",
            &reference.exits,
            &candidate.exits,
        )
    })
    .or_else(|| compare_trade_identity(&reference.accepted_trades, &candidate.accepted_trades))
    .or_else(|| {
        compare_trade_values(
            &reference.accepted_trades,
            &candidate.accepted_trades,
            policy,
        )
    })
    .or_else(|| {
        compare_trade_equity(
            &reference.accepted_trades,
            &candidate.accepted_trades,
            policy.pnl_and_equity,
        )
    })
    .or_else(|| {
        compare_risk_states(
            &reference.risk_states,
            &candidate.risk_states,
            policy.risk_state,
        )
    })
    .or_else(|| {
        compare_metrics(
            &reference.final_metrics,
            &candidate.final_metrics,
            policy.metrics,
        )
    })
    .or_else(|| {
        compare_exact_slice(
            ParityLevel::ValidationVerdict,
            "verdict",
            &reference.validation_verdicts,
            &candidate.validation_verdicts,
        )
    })
    .or_else(|| {
        compare_exact_slice(
            ParityLevel::SurvivorOrdering,
            "survivor",
            &reference.survivor_order,
            &candidate.survivor_order,
        )
    });

    TraceComparisonReport {
        reference_backend: reference_backend.into(),
        candidate_backend: candidate_backend.into(),
        first_divergence,
    }
}

fn compare_float_slice(
    level: ParityLevel,
    field: &str,
    expected: &[f64],
    actual: &[f64],
    tolerance: FloatTolerance,
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(level, field, expected.len(), actual.len());
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (expected, actual))| {
            (!tolerance.matches_f64(*expected, *actual))
                .then(|| divergence(level, index, field, expected, actual))
        })
}

fn compare_f32_slice(
    level: ParityLevel,
    field: &str,
    expected: &[f32],
    actual: &[f32],
    tolerance: FloatTolerance,
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(level, field, expected.len(), actual.len());
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (expected, actual))| {
            (!tolerance.matches_f32(*expected, *actual))
                .then(|| divergence(level, index, field, expected, actual))
        })
}

fn compare_exact_slice<T: PartialEq + core::fmt::Debug>(
    level: ParityLevel,
    field: &str,
    expected: &[T],
    actual: &[T],
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(level, field, expected.len(), actual.len());
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (expected, actual))| {
            (expected != actual).then(|| divergence(level, index, field, expected, actual))
        })
}

fn compare_trade_identity(
    expected: &[TradeTrace],
    actual: &[TradeTrace],
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(
            ParityLevel::AcceptedTradeSequence,
            "trade",
            expected.len(),
            actual.len(),
        );
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (e, a))| {
            let expected_id = (
                e.candidate_id,
                e.scenario_id,
                e.entry_bar,
                e.exit_bar,
                e.direction,
                e.exit_reason,
            );
            let actual_id = (
                a.candidate_id,
                a.scenario_id,
                a.entry_bar,
                a.exit_bar,
                a.direction,
                a.exit_reason,
            );
            (expected_id != actual_id).then(|| {
                divergence(
                    ParityLevel::AcceptedTradeSequence,
                    index,
                    "trade_identity",
                    &expected_id,
                    &actual_id,
                )
            })
        })
}

fn compare_trade_values(
    expected: &[TradeTrace],
    actual: &[TradeTrace],
    policy: ParityPolicy,
) -> Option<ParityDivergence> {
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (e, a))| {
            if !policy.size_and_costs.matches_f64(e.size, a.size) {
                Some(divergence(
                    ParityLevel::PositionSizeAndCosts,
                    index,
                    "size",
                    &e.size,
                    &a.size,
                ))
            } else if !policy.size_and_costs.matches_f64(e.costs, a.costs) {
                Some(divergence(
                    ParityLevel::PositionSizeAndCosts,
                    index,
                    "costs",
                    &e.costs,
                    &a.costs,
                ))
            } else if !policy.pnl_and_equity.matches_f64(e.pnl, a.pnl) {
                Some(divergence(
                    ParityLevel::PositionSizeAndCosts,
                    index,
                    "pnl",
                    &e.pnl,
                    &a.pnl,
                ))
            } else {
                None
            }
        })
}

fn compare_trade_equity(
    expected: &[TradeTrace],
    actual: &[TradeTrace],
    tolerance: FloatTolerance,
) -> Option<ParityDivergence> {
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (e, a))| {
            (!tolerance.matches_f64(e.equity_after, a.equity_after)).then(|| {
                divergence(
                    ParityLevel::EquityAfterTrade,
                    index,
                    "equity_after",
                    &e.equity_after,
                    &a.equity_after,
                )
            })
        })
}

fn compare_risk_states(
    expected: &[RiskStateTrace],
    actual: &[RiskStateTrace],
    tolerance: FloatTolerance,
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(
            ParityLevel::CalendarAndPropFirmState,
            "risk_state",
            expected.len(),
            actual.len(),
        );
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(index, (e, a))| {
            let expected_identity = (
                e.candidate_id,
                e.scenario_id,
                e.bar,
                e.day_id,
                e.month_id,
                e.flags,
            );
            let actual_identity = (
                a.candidate_id,
                a.scenario_id,
                a.bar,
                a.day_id,
                a.month_id,
                a.flags,
            );
            if expected_identity != actual_identity {
                return Some(divergence(
                    ParityLevel::CalendarAndPropFirmState,
                    index,
                    "risk_state_identity",
                    &expected_identity,
                    &actual_identity,
                ));
            }
            for (field, expected, actual) in [
                ("equity", e.equity, a.equity),
                ("peak_equity", e.peak_equity, a.peak_equity),
                ("day_start_equity", e.day_start_equity, a.day_start_equity),
                (
                    "month_start_equity",
                    e.month_start_equity,
                    a.month_start_equity,
                ),
            ] {
                if !tolerance.matches_f64(expected, actual) {
                    return Some(divergence(
                        ParityLevel::CalendarAndPropFirmState,
                        index,
                        field,
                        &expected,
                        &actual,
                    ));
                }
            }
            None
        })
}

fn compare_metrics(
    expected: &[[f64; 11]],
    actual: &[[f64; 11]],
    tolerance: FloatTolerance,
) -> Option<ParityDivergence> {
    if expected.len() != actual.len() {
        return length_mismatch(
            ParityLevel::FinalMetrics,
            "metrics",
            expected.len(),
            actual.len(),
        );
    }
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find_map(|(row, (e, a))| {
            e.iter().zip(a).enumerate().find_map(|(column, (e, a))| {
                (!tolerance.matches_f64(*e, *a)).then(|| {
                    divergence(
                        ParityLevel::FinalMetrics,
                        row,
                        &format!("metric[{column}]"),
                        e,
                        a,
                    )
                })
            })
        })
}

fn length_mismatch(
    level: ParityLevel,
    field: &str,
    expected: usize,
    actual: usize,
) -> Option<ParityDivergence> {
    Some(divergence(
        level,
        0,
        &format!("{field}.len"),
        &expected,
        &actual,
    ))
}

fn divergence(
    level: ParityLevel,
    index: usize,
    field: &str,
    expected: &impl core::fmt::Debug,
    actual: &impl core::fmt::Debug,
) -> ParityDivergence {
    ParityDivergence {
        level,
        index,
        field: field.to_owned(),
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    ordered_f32(a).abs_diff(ordered_f32(b))
}

fn ordered_f32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    ordered_f64(a).abs_diff(ordered_f64(b))
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_first_causal_divergence() {
        let reference = ParityTrace {
            scores_before_threshold: vec![0.5],
            signals: vec![1],
            survivor_order: vec![10, 20],
            ..ParityTrace::default()
        };
        let candidate = ParityTrace {
            scores_before_threshold: vec![0.5],
            signals: vec![-1],
            survivor_order: vec![20, 10],
            ..ParityTrace::default()
        };
        let report = compare_traces(
            "cpu",
            "gpu",
            &reference,
            &candidate,
            ParityPolicy::default(),
        );
        assert_eq!(
            report.first_divergence.unwrap().level,
            ParityLevel::SignalDirection
        );
    }

    #[test]
    fn tolerates_declared_float_noise_but_not_order_changes() {
        let reference = ParityTrace {
            scores_before_threshold: vec![1.0],
            survivor_order: vec![1, 2],
            ..ParityTrace::default()
        };
        let candidate = ParityTrace {
            scores_before_threshold: vec![1.0 + 1.0e-12],
            survivor_order: vec![2, 1],
            ..ParityTrace::default()
        };
        let report = compare_traces(
            "cpu",
            "gpu",
            &reference,
            &candidate,
            ParityPolicy::default(),
        );
        assert_eq!(
            report.first_divergence.unwrap().level,
            ParityLevel::SurvivorOrdering
        );
    }

    #[test]
    fn exact_discrete_trace_matches() {
        let trace = ParityTrace {
            entries: vec![EntryEventTrace {
                candidate_id: 1,
                scenario_id: 2,
                bar: 3,
                direction: 1,
            }],
            exits: vec![ExitEventTrace {
                candidate_id: 1,
                scenario_id: 2,
                bar: 4,
                reason: 5,
            }],
            ..ParityTrace::default()
        };
        assert!(compare_traces("a", "b", &trace, &trace, ParityPolicy::default()).is_match());
    }

    #[test]
    fn levels_four_through_nine_report_the_first_causal_field() {
        let reference = ParityTrace {
            entries: vec![EntryEventTrace {
                candidate_id: 1,
                scenario_id: 2,
                bar: 3,
                direction: 1,
            }],
            exits: vec![ExitEventTrace {
                candidate_id: 1,
                scenario_id: 2,
                bar: 4,
                reason: 1,
            }],
            accepted_trades: vec![TradeTrace {
                candidate_id: 1,
                scenario_id: 2,
                entry_bar: 3,
                exit_bar: 4,
                direction: 1,
                exit_reason: 1,
                size: 2.0,
                costs: 3.0,
                pnl: -10.0,
                equity_after: 99_990.0,
            }],
            risk_states: vec![RiskStateTrace {
                candidate_id: 1,
                scenario_id: 2,
                bar: 4,
                day_id: 5,
                month_id: 6,
                equity: 99_990.0,
                peak_equity: 100_000.0,
                day_start_equity: 100_000.0,
                month_start_equity: 100_000.0,
                flags: 0,
            }],
            ..ParityTrace::default()
        };

        let mut candidate = reference.clone();
        candidate.entries[0].bar = 9;
        assert_first_level(&reference, &candidate, ParityLevel::EntryEvents);

        candidate = reference.clone();
        candidate.exits[0].reason = 2;
        assert_first_level(&reference, &candidate, ParityLevel::ExitBarAndReason);

        candidate = reference.clone();
        candidate.accepted_trades[0].direction = -1;
        assert_first_level(&reference, &candidate, ParityLevel::AcceptedTradeSequence);

        candidate = reference.clone();
        candidate.accepted_trades[0].costs = 4.0;
        assert_first_level(&reference, &candidate, ParityLevel::PositionSizeAndCosts);

        candidate = reference.clone();
        candidate.accepted_trades[0].equity_after = 99_980.0;
        assert_first_level(&reference, &candidate, ParityLevel::EquityAfterTrade);

        candidate = reference.clone();
        candidate.risk_states[0].flags = 1;
        assert_first_level(
            &reference,
            &candidate,
            ParityLevel::CalendarAndPropFirmState,
        );
    }

    fn assert_first_level(reference: &ParityTrace, candidate: &ParityTrace, expected: ParityLevel) {
        let report = compare_traces(
            "reference",
            "candidate",
            reference,
            candidate,
            ParityPolicy::default(),
        );
        assert_eq!(report.first_divergence.unwrap().level, expected);
    }
}
