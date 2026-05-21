use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParityExecutionSemantics {
    Canonical,
    Approximate,
    Degraded,
}

impl ParityExecutionSemantics {
    pub fn is_canonical(self) -> bool {
        matches!(self, Self::Canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityComparisonReport {
    pub reference_backend: String,
    pub candidate_backend: String,
    pub semantics: ParityExecutionSemantics,
    pub tolerance: f64,
    pub max_abs_delta: f64,
    pub mismatches: Vec<ParityMismatch>,
}

impl ParityComparisonReport {
    pub fn is_within_tolerance(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityMismatch {
    pub row: usize,
    pub column: usize,
    pub expected: f64,
    pub actual: f64,
    pub abs_delta: f64,
}

pub fn compare_metric_matrices(
    reference_backend: impl Into<String>,
    candidate_backend: impl Into<String>,
    reference: &[[f64; 11]],
    candidate: &[[f64; 11]],
    tolerance: f64,
    semantics: ParityExecutionSemantics,
) -> Result<ParityComparisonReport> {
    if reference.len() != candidate.len() {
        bail!(
            "parity fixture length mismatch: reference={} candidate={}",
            reference.len(),
            candidate.len()
        );
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        bail!("parity tolerance must be finite and non-negative");
    }

    let mut max_abs_delta = 0.0_f64;
    let mut mismatches = Vec::new();
    for (row, (expected_row, actual_row)) in reference.iter().zip(candidate.iter()).enumerate() {
        for (column, (expected, actual)) in expected_row.iter().zip(actual_row.iter()).enumerate() {
            let abs_delta = if expected.is_finite() && actual.is_finite() {
                (expected - actual).abs()
            } else if expected.to_bits() == actual.to_bits() {
                0.0
            } else {
                f64::INFINITY
            };
            max_abs_delta = max_abs_delta.max(abs_delta);
            if abs_delta > tolerance {
                mismatches.push(ParityMismatch {
                    row,
                    column,
                    expected: *expected,
                    actual: *actual,
                    abs_delta,
                });
            }
        }
    }

    let report = ParityComparisonReport {
        reference_backend: reference_backend.into(),
        candidate_backend: candidate_backend.into(),
        semantics,
        tolerance,
        max_abs_delta,
        mismatches,
    };

    if !report.is_within_tolerance() && semantics.is_canonical() {
        bail!(
            "canonical parity mismatch between {} and {}: {} cells exceeded tolerance {} (max delta {})",
            report.reference_backend,
            report.candidate_backend,
            report.mismatches.len(),
            report.tolerance,
            report.max_abs_delta
        );
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{BacktestSettings, fast_evaluate_strategy_core};
    use crate::genetic::{
        EvaluationConfig, Gene, evaluate_genes, month_day_indices, signals_for_gene,
    };
    use neoethos_data::{FeatureFrame, Ohlcv};
    use ndarray::arr2;

    // TODO(real-data): synthetic feature/OHLCV fixture. Replace with a
    // cTrader historical sample (e.g. EURUSD M1 12 bars from a fixed
    // timestamp range) so parity is asserted against broker-shaped data
    // rather than hand-tuned sequences.
    fn fixture_frame() -> FeatureFrame {
        let timestamps = (0..12)
            .map(|idx| 1_700_000_000_000_i64 + (idx as i64) * 60_000)
            .collect();
        FeatureFrame {
            timestamps,
            names: vec!["momentum".to_string(), "reversion".to_string()],
            data: arr2(&[
                [0.10, -0.10],
                [0.35, -0.20],
                [0.60, -0.30],
                [0.20, -0.10],
                [-0.20, 0.25],
                [-0.45, 0.50],
                [-0.15, 0.20],
                [0.40, -0.35],
                [0.55, -0.45],
                [0.05, -0.05],
                [-0.50, 0.55],
                [-0.65, 0.70],
            ]),
        }
    }

    fn fixture_ohlcv(frame: &FeatureFrame) -> Ohlcv {
        Ohlcv {
            timestamp: Some(frame.timestamps.clone()),
            open: vec![
                1.1000, 1.1010, 1.1020, 1.1040, 1.1030, 1.1010, 1.0990, 1.1005, 1.1025, 1.1035,
                1.1015, 1.0995,
            ],
            high: vec![
                1.1010, 1.1025, 1.1050, 1.1050, 1.1040, 1.1020, 1.1000, 1.1030, 1.1040, 1.1040,
                1.1020, 1.1000,
            ],
            low: vec![
                1.0990, 1.1005, 1.1015, 1.1025, 1.1005, 1.0985, 1.0980, 1.1000, 1.1010, 1.1005,
                1.0985, 1.0975,
            ],
            close: vec![
                1.1000, 1.1015, 1.1040, 1.1030, 1.1010, 1.0990, 1.1005, 1.1025, 1.1035, 1.1015,
                1.0995, 1.0985,
            ],
            volume: None,
        }
    }

    fn fixture_genes() -> Vec<Gene> {
        vec![
            Gene {
                indices: vec![0, 1],
                weights: vec![0.8, -0.4],
                long_threshold: 0.35,
                short_threshold: -0.35,
                tp_pips: 25.0,
                sl_pips: 18.0,
                strategy_id: "fixture-trend".to_string(),
                ..Gene::default()
            },
            Gene {
                indices: vec![0, 1],
                weights: vec![-0.3, 0.9],
                long_threshold: 0.25,
                short_threshold: -0.25,
                tp_pips: 20.0,
                sl_pips: 15.0,
                strategy_id: "fixture-reversion".to_string(),
                ..Gene::default()
            },
        ]
    }

    fn eval_config() -> EvaluationConfig {
        EvaluationConfig {
            max_hold_bars: 3,
            trailing_enabled: false,
            pip_value: 0.0001,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10.0,
            smc_gate_threshold: 0.0,
            ..EvaluationConfig::default()
        }
    }

    fn backtest_settings(config: &EvaluationConfig) -> BacktestSettings {
        BacktestSettings {
            max_hold_bars: config.max_hold_bars,
            trailing_enabled: config.trailing_enabled,
            trailing_atr_multiplier: config.trailing_atr_multiplier,
            trailing_be_trigger_r: config.trailing_be_trigger_r,
            pip_value: config.pip_value,
            spread_pips: config.spread_pips,
            commission_per_trade: config.commission_per_trade,
            pip_value_per_lot: config.pip_value_per_lot,
            ..BacktestSettings::default()
        }
    }

    #[test]
    fn population_evaluator_matches_scalar_cpu_fixture() {
        let frame = fixture_frame();
        let ohlcv = fixture_ohlcv(&frame);
        let genes = fixture_genes();
        let config = eval_config();
        let candidate = evaluate_genes(&frame, &ohlcv, &genes, &config)
            .expect("population evaluator should score fixture genes");

        let (_months, days) = month_day_indices(&frame.timestamps);
        let months = vec![0_i64; frame.timestamps.len()];
        let settings = backtest_settings(&config);
        let reference: Vec<[f64; 11]> = genes
            .iter()
            .map(|gene| {
                let signals = signals_for_gene(&frame, gene);
                fast_evaluate_strategy_core(
                    &ohlcv.close,
                    &ohlcv.high,
                    &ohlcv.low,
                    &signals,
                    &months,
                    &days,
                    &frame.timestamps,
                    &BacktestSettings {
                        sl_pips: gene.sl_pips,
                        tp_pips: gene.tp_pips,
                        ..settings.clone()
                    },
                )
            })
            .collect();

        let report = compare_metric_matrices(
            "scalar_cpu_reference",
            "population_evaluator_candidate",
            &reference,
            &candidate,
            1e-9,
            ParityExecutionSemantics::Canonical,
        )
        .expect("canonical fixture should match exactly");
        assert!(report.is_within_tolerance());
        assert_eq!(report.max_abs_delta, 0.0);
    }

    #[test]
    fn canonical_parity_rejects_silent_mismatch() {
        let reference = vec![[1.0_f64; 11]];
        let mut candidate = reference.clone();
        candidate[0][3] += 0.25;

        let err = compare_metric_matrices(
            "cpu",
            "gpu",
            &reference,
            &candidate,
            1e-12,
            ParityExecutionSemantics::Canonical,
        )
        .expect_err("canonical mismatch must not be silent");
        assert!(err.to_string().contains("canonical parity mismatch"));
    }

    #[test]
    fn approximate_or_degraded_parity_must_be_explicitly_marked() {
        let reference = vec![[1.0_f64; 11]];
        let mut candidate = reference.clone();
        candidate[0][0] += 0.01;

        let approximate = compare_metric_matrices(
            "cpu",
            "surrogate_gpu",
            &reference,
            &candidate,
            1e-12,
            ParityExecutionSemantics::Approximate,
        )
        .expect("approximate mismatch should be reported, not rejected");
        assert!(!approximate.is_within_tolerance());
        assert_eq!(approximate.mismatches.len(), 1);

        let degraded = compare_metric_matrices(
            "cpu",
            "cpu_fallback",
            &reference,
            &candidate,
            1e-12,
            ParityExecutionSemantics::Degraded,
        )
        .expect("degraded mismatch should be reported, not rejected");
        assert_eq!(degraded.semantics, ParityExecutionSemantics::Degraded);
        assert!(!degraded.mismatches.is_empty());
    }
}
