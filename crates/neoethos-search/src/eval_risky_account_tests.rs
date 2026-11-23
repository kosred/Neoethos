use super::*;

fn settings() -> BacktestSettings {
    BacktestSettings {
        risky_account_experiment: Some(RiskyAccountExperimentV1 {
            target_balance: 50_000.0,
            continuous_exact_stop_risk: false,
            lot_constraints: None,
            fixed_leverage: None,
        }),
        initial_equity_override: Some(100.0),
        sl_pips: 4.0,
        tp_pips: 5.2,
        pip_value: 0.125,
        pip_value_per_lot: 1.0,
        spread_pips: 0.0,
        commission_per_trade: 0.0,
        swap_long_pips_per_day: 0.0,
        swap_short_pips_per_day: 0.0,
        pnl_conversion_fee_rate: 0.0,
        risk_per_trade_min: 3.0 / 13.0,
        risk_per_trade_max: 3.0 / 13.0,
        max_hold_bars: 0,
        min_hold_bars: 0,
        gap_threshold_ms: 0,
        ..Default::default()
    }
}

fn replay(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    settings: &BacktestSettings,
) -> ([f64; 11], Vec<Trade>) {
    let timestamps = (0..close.len())
        .map(|i| 1_735_689_600_000 + i as i64 * 60_000)
        .collect::<Vec<_>>();
    let confidence = vec![1.0; close.len()];
    let calendar = vec![0; close.len()];
    let result = evaluate_strategy_with_confidence_and_ledger_core(
        close,
        high,
        low,
        signals,
        &confidence,
        &calendar,
        &calendar,
        &timestamps,
        settings,
        None,
        None,
    )
    .unwrap();
    let fast = fast_evaluate_strategy_core(
        close,
        high,
        low,
        signals,
        &confidence,
        &calendar,
        &calendar,
        &timestamps,
        settings,
    );
    assert_eq!(
        fast, result.0,
        "fitness and detailed ledger must use identical accounting"
    );
    result
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "got {actual}, expected {expected}"
    );
}

fn trailing_settings() -> BacktestSettings {
    let mut policy = settings();
    policy.trailing_enabled = true;
    policy.trailing_be_trigger_r = 1.0;
    policy.trailing_atr_multiplier = 1.0;
    policy.trailing_min_lock_pips = 2.0;
    policy.tp_pips = 80.0;
    policy
}

#[test]
fn risky_trailing_cannot_arm_beyond_the_observed_close() {
    for side in [1, -1] {
        for retracement in [0.125, 0.25] {
            let mut policy = trailing_settings();
            policy.max_hold_bars = 2;
            let sign = side as f64;
            let close = [100.0, 100.0, 100.0 + sign * retracement, 100.0];
            let (high, low) = if side == 1 {
                (
                    [100.0, 100.0, 100.75, 100.125],
                    [100.0, 100.0, 99.875, 99.875],
                )
            } else {
                (
                    [100.0, 100.0, 100.125, 100.125],
                    [100.0, 100.0, 99.25, 99.875],
                )
            };
            let (metrics, trades) = replay(&close, &high, &low, &[side, 0, 0, 0], &policy);
            assert_eq!(trades.len(), 1);
            // At the decision close the desired stop is already crossed (or
            // equal to market). No executable next-bar amend was established.
            // The final bar never trades at that desired +2-pip stop either.
            // Only the explicit two-bar timeout at the entry price settles it;
            // this replay does not liquidate an open trade merely at EOF.
            near(trades[0].pnl, 0.0);
            near(metrics[0], 0.0);
            assert_ne!(trades[0].exit_reason, Some(TradeExitReason::TrailingStop));

            let mut legacy = policy.clone();
            legacy.risky_account_experiment = None;
            let (old, _) = replay(&close, &high, &low, &[side, 0, 0, 0], &legacy);
            near(old[0], 150.0 / 13.0);
        }
    }
}

#[test]
fn risky_valid_closed_bar_trail_still_protects_the_next_bar() {
    for side in [1, -1] {
        let policy = trailing_settings();
        let sign = side as f64;
        let close = [100.0, 100.0, 100.0 + sign * 0.375, 100.0];
        let (high, low) = if side == 1 {
            (
                [100.0, 100.0, 100.75, 100.375],
                [100.0, 100.0, 99.875, 99.875],
            )
        } else {
            (
                [100.0, 100.0, 100.125, 100.125],
                [100.0, 100.0, 99.25, 99.625],
            )
        };
        let (metrics, trades) = replay(&close, &high, &low, &[side, 0, 0, 0], &policy);
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_reason, Some(TradeExitReason::TrailingStop));
        // +2 pips with 100*(3/13)/4 lots, independently of the ratchet formula.
        near(trades[0].pnl, 150.0 / 13.0);
        near(metrics[0], 150.0 / 13.0);
    }
}

#[test]
fn risky_refused_ratchet_keeps_its_previous_accepted_stop() {
    for side in [1, -1] {
        let sign = side as f64;
        let close = [0.0, 0.0, 0.375, 0.5, 0.25].map(|x| 100.0 + sign * x);
        let upper = [0.0, 0.0, 0.75, 1.0, 0.375];
        let lower = [0.0, 0.0, -0.125, 0.375, 0.125];
        let (high, low) = if side == 1 {
            (upper.map(|x| 100.0 + x), lower.map(|x| 100.0 + x))
        } else {
            (lower.map(|x| 100.0 - x), upper.map(|x| 100.0 - x))
        };
        let (metrics, trades) = replay(
            &close,
            &high,
            &low,
            &[side, 0, 0, 0, 0],
            &trailing_settings(),
        );
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_reason, Some(TradeExitReason::TrailingStop));
        near(metrics[0], 150.0 / 13.0);
    }
}

#[test]
fn risky_stop_excludes_after_exit_extremes_for_long_and_short() {
    for side in [1, -1] {
        let mut high = [100.0; 3];
        let mut low = high;
        if side == 1 {
            high[2] = 100.125;
            low[2] = 97.5;
        } else {
            high[2] = 102.5;
            low[2] = 99.875;
        }
        let (metrics, trades) = replay(&[100.0; 3], &high, &low, &[side, 0, 0], &settings());
        assert_eq!(trades.len(), 1);
        near(trades[0].pnl, -300.0 / 13.0);
        near(trades[0].mae, 300.0 / 13.0);
        near(trades[0].mfe, 0.0);
        near(metrics[3], 3.0 / 13.0);
    }
}

#[test]
fn risky_target_caps_excursion_and_uses_adverse_before_the_exit() {
    let (metrics, trades) = replay(
        &[100.0; 3],
        &[100.0, 100.0, 103.0],
        &[100.0, 100.0, 99.75],
        &[1, 0, 0],
        &settings(),
    );
    near(trades[0].pnl, 30.0);
    near(trades[0].mfe, 30.0);
    near(trades[0].mae, 150.0 / 13.0);
    near(metrics[3], 1.5 / 13.0);
}

#[test]
fn risky_dual_touch_retains_stop_first_without_post_exit_peaks() {
    let (metrics, trades) = replay(
        &[100.0; 3],
        &[100.0, 100.0, 103.0],
        &[100.0, 100.0, 97.5],
        &[1, 0, 0],
        &settings(),
    );
    assert_eq!(trades[0].exit_reason, Some(TradeExitReason::Stop));
    near(trades[0].mae, 300.0 / 13.0);
    near(trades[0].mfe, 0.0);
    near(metrics[3], 3.0 / 13.0);
}

#[test]
fn risky_first_realized_target_is_absorbing() {
    let mut settings = settings();
    settings
        .risky_account_experiment
        .as_mut()
        .unwrap()
        .target_balance = 130.0;
    let (metrics, trades) = replay(
        &[100.0; 5],
        &[100.0, 100.0, 100.75, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0, 97.5],
        &[1, 0, 1, 0, 0],
        &settings,
    );
    assert_eq!(
        trades.len(),
        1,
        "a later signal cannot spend a banked target"
    );
    near(metrics[0], 30.0);
    near(100.0 + metrics[0], 130.0);
}

#[test]
fn legacy_whole_bar_observation_stays_explicitly_unchanged() {
    let mut settings = settings();
    settings.risky_account_experiment = None;
    let (metrics, trades) = replay(
        &[100.0; 3],
        &[100.0, 100.0, 100.125],
        &[100.0, 100.0, 97.5],
        &[1, 0, 0],
        &settings,
    );
    near(trades[0].mae, 1500.0 / 13.0);
    // Legacy credits the candle's +5.7692 peak before its -115.3846 trough.
    near(metrics[3], 63.0 / 55.0);
}

#[test]
fn risky_true_gap_loss_is_uncapped_and_realized_ruin_is_absorbing() {
    let mut settings = settings();
    settings.gap_threshold_ms = 60_000;
    let (metrics, trades) = replay(
        &[100.0, 100.0, 97.5, 100.0, 200.0],
        &[100.0, 100.0, 98.0, 101.0, 200.0],
        &[100.0, 100.0, 90.0, 99.0, 100.0],
        &[1, 0, 1, 0, 0],
        &settings,
    );
    assert_eq!(trades.len(), 1);
    assert_eq!(trades[0].exit_reason, Some(TradeExitReason::DataGap));
    near(metrics[0], -1500.0 / 13.0);
    near(trades[0].mae, 1500.0 / 13.0);
    near(metrics[3], 15.0 / 13.0);
    assert!(
        100.0 + metrics[0] < 0.0,
        "do not hide modeled negative cash"
    );
}

#[test]
fn risky_target_requires_realized_cash_after_costs() {
    let mut settings = settings();
    settings
        .risky_account_experiment
        .as_mut()
        .unwrap()
        .target_balance = 130.0;
    settings.commission_per_trade = 1.0;
    let (_, trades) = replay(
        &[100.0; 5],
        &[100.0, 100.0, 100.75, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0, 97.5],
        &[1, 0, 1, 0, 0],
        &settings,
    );
    assert_eq!(trades.len(), 2, "gross target touch did not bank net 130");
    near(trades[0].pnl, 315.0 / 13.0);
}

#[test]
fn risky_policy_validates_target_and_cannot_disappear_in_device_conversion() {
    let mut settings = settings();
    settings
        .risky_account_experiment
        .as_mut()
        .unwrap()
        .lot_constraints =
        Some(crate::QuoteReplayLotConstraintsV3::new(1.0, 10.0, 1.0, "a".repeat(64)).unwrap());
    let dto = crate::gpu_native::snapshot_fixture::SnapshotSettingsDto::from_settings(&settings);
    assert_eq!(
        dto.to_settings().risky_account_experiment,
        settings.risky_account_experiment
    );
    assert!(matches!(
        crate::gpu_native::prototype_population_oracle::population_settings_for_settings(&settings),
        Err(crate::gpu_native::prototype_population_oracle::PopulationOracleError::UnsupportedRiskyAccountExperiment)
    ));
    for target in [0.0, 100.0, f64::NAN, f64::INFINITY] {
        settings
            .risky_account_experiment
            .as_mut()
            .unwrap()
            .target_balance = target;
        assert!(validate_sizing_confidences(0, &[], &settings).is_err());
    }
}

#[test]
fn risky_explicit_lot_grid_applies_to_direct_confidence_and_ledger() {
    // Pure accounting fixture, not captured broker quantities. Requested
    // size is 75/13 lots; only the admitted quantity may drive costs/PnL/DD.
    for (min, max, step, expected_trades, expected_loss) in [
        (1.0, 10.0, 1.0, 1, 20.0),
        (6.0, 10.0, 1.0, 0, 0.0),
        (0.5, 3.0, 0.5, 1, 12.0),
    ] {
        let mut settings = settings();
        settings
            .risky_account_experiment
            .as_mut()
            .unwrap()
            .lot_constraints =
            Some(crate::QuoteReplayLotConstraintsV3::new(min, max, step, "a".repeat(64)).unwrap());
        let (metrics, trades) = replay(
            &[100.0; 3],
            &[100.0, 100.0, 100.125],
            &[100.0, 100.0, 97.5],
            &[1, 0, 0],
            &settings,
        );
        assert_eq!(trades.len(), expected_trades);
        near(metrics[0], -expected_loss);
        near(metrics[3], expected_loss / 100.0);
    }
}

#[test]
fn risky_lot_grid_is_optional_on_old_wire_and_rejects_unchecked_metadata() {
    let legacy = r#"{"target_balance":50000.0}"#;
    let policy: RiskyAccountExperimentV1 = serde_json::from_str(legacy).unwrap();
    assert_eq!(serde_json::to_string(&policy).unwrap(), legacy);
    for (min, max, step, source) in [
        (0.0, 2.0, 1.0, "a".repeat(64)),
        (2.0, 1.0, 1.0, "a".repeat(64)),
        (1.0, 2.0, 0.0, "a".repeat(64)),
        (1.0, 2.0, 1.0, "unbound".to_owned()),
    ] {
        let wire = serde_json::json!({"target_balance":50000.0,
            "lot_constraints":{"min_lots":min,"max_lots":max,"lot_step":step,
                "source_artifact_sha256":source}});
        let untrusted: RiskyAccountExperimentV1 = serde_json::from_value(wire).unwrap();
        assert!(untrusted.validate(100.0, (0.30, 0.30)).is_err());
    }
}
