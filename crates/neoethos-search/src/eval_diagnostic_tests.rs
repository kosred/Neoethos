use super::*;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct DiagnosticWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for DiagnosticWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn capture_diagnostics<T>(work: impl FnOnce() -> T) -> (T, String) {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = DiagnosticWriter(Arc::clone(&bytes));
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let value = tracing::subscriber::with_default(subscriber, work);
    let logs = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    (value, logs)
}

#[test]
fn invalid_monthly_return_inputs_keep_the_rejection_sentinel_and_correct_diagnostic() {
    let cases = [
        (vec![1.0], vec![]),
        (vec![1.0], vec![0.0]),
        (vec![1.0], vec![-1.0]),
        (vec![f64::NAN], vec![100.0]),
        (vec![1.0], vec![f64::INFINITY]),
        (vec![f64::MAX], vec![f64::MIN_POSITIVE]),
        (vec![f64::MAX, f64::MAX], vec![1.0, 1.0]),
    ];
    for (pnl, equity) in cases {
        let sharpe = completed_month_return_sharpe_v1(&pnl, &equity);
        assert_eq!(sharpe, INVALID_MONTHLY_RETURN_SHARPE_V1);
        assert_eq!(sanitize_sharpe_v1(sharpe), f64::NEG_INFINITY);
        let (_, logs) = capture_diagnostics(|| {
            report_non_finite_candidate_metrics_v1(2, &["sharpe"], sharpe);
        });
        assert!(
            logs.contains("reason=\"invalid_completed_month_return\""),
            "{logs}"
        );
        assert!(logs.contains("candidate rejected"), "{logs}");
        assert!(!logs.contains("broker financials missing"), "{logs}");
    }
}

#[test]
fn finite_financial_inputs_can_reject_monthly_sharpe_without_missing_broker_data() {
    // Deliberately insolvent, fixed-one-lot formula fixture: two finite 200-unit
    // charges exhaust a 100-unit account. This is not a production account or
    // broker-authority fixture, and does not alter bankruptcy/sizing policy.
    let settings = BacktestSettings {
        initial_equity_override: Some(100.0),
        pip_value: 0.0001,
        pip_value_per_lot: 10.0,
        spread_pips: 0.0,
        commission_per_trade: 200.0,
        swap_long_pips_per_day: 0.0,
        swap_short_pips_per_day: 0.0,
        pnl_conversion_fee_rate: 0.0,
        max_hold_bars: 1,
        min_hold_bars: 0,
        gap_threshold_ms: 0,
        kill_zones_enabled: false,
        trailing_enabled: false,
        risk_based_sizing: false,
        ..BacktestSettings::default()
    };
    let (metrics, logs) = capture_diagnostics(|| {
        fast_evaluate_strategy_core(
            &[1.0; 6],
            &[1.0; 6],
            &[1.0; 6],
            &[1, 0, 1, 0, 0, 0],
            &[],
            &[0, 0, 0, 1, 1, 2],
            &[0; 6],
            &[],
            &settings,
        )
    });
    assert_eq!(metrics[0], -400.0);
    assert_eq!(metrics[8], 2.0);
    assert_eq!(metrics[1], INVALID_MONTHLY_RETURN_SHARPE_V1);
    assert!(
        metrics
            .iter()
            .enumerate()
            .all(|(index, value)| index == 1 || value.is_finite())
    );
    assert_eq!(crate::scoring::ga_fitness(&metrics), f64::NEG_INFINITY);
    assert_eq!(
        crate::scoring::ga_fitness_growth(&metrics),
        f64::NEG_INFINITY
    );
    assert!(logs.contains("non_finite_metrics=[\"sharpe\"]"), "{logs}");
    assert!(
        logs.contains("reason=\"invalid_completed_month_return\""),
        "{logs}"
    );
    assert!(!logs.contains("broker financials missing"), "{logs}");
}

#[test]
fn other_nonfinite_metrics_do_not_guess_a_monthly_return_or_broker_cause() {
    for (names, sharpe) in [
        (
            vec!["net_profit", "sharpe"],
            INVALID_MONTHLY_RETURN_SHARPE_V1,
        ),
        (vec!["sharpe"], f64::NAN),
        (vec!["sharpe"], f64::INFINITY),
    ] {
        let (_, logs) = capture_diagnostics(|| {
            report_non_finite_candidate_metrics_v1(2, &names, sharpe);
        });
        assert!(
            logs.contains("reason=\"non_finite_evaluation_metrics\""),
            "{logs}"
        );
        assert!(!logs.contains("invalid_completed_month_return"), "{logs}");
        assert!(!logs.contains("broker financials missing"), "{logs}");
    }
}
