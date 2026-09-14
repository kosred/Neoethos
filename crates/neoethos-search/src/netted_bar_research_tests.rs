use super::*;

// Deliberately hand-calculated account-money fixtures. This private constructor
// tests arithmetic, not canonical data authority or broker execution evidence.
fn settings() -> BacktestSettings {
    BacktestSettings {
        initial_equity_override: Some(1_000.0),
        pip_value: 1.0,
        pip_value_per_lot: 10.0,
        sl_pips: 2.0,
        tp_pips: 4.0,
        spread_pips: 0.0,
        commission_per_trade: 2.0,
        risk_based_sizing: true,
        risk_per_trade_min: 0.01,
        risk_per_trade_max: 0.03,
        high_quality_confidence: 1.0,
        max_hold_bars: 0,
        gap_threshold_ms: 0,
        kill_zones_enabled: false,
        trailing_enabled: false,
        session_spread_profile: None,
        swap_long_pips_per_day: 0.0,
        swap_short_pips_per_day: 0.0,
        pnl_conversion_fee_rate: 0.0,
        ..Default::default()
    }
}

fn prepared(close: &[f64], high: &[f64], low: &[f64]) -> PreparedNettedCanonicalBarResearchV1 {
    PreparedNettedCanonicalBarResearchV1::from_checked_parts(
        close.to_vec(),
        high.to_vec(),
        low.to_vec(),
        (0..close.len())
            .map(|row| 1_704_067_200_000 + row as i64 * 60_000)
            .collect(),
        settings(),
        "USD".into(),
        "private-numerical-fixture-not-authority".into(),
        "private-explicit-screening-cost-fixture".into(),
    )
    .unwrap()
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "actual={actual:.15} expected={expected:.15}"
    );
}

#[test]
fn varying_brackets_confidence_and_ml_once_compound_hand_calculated_account_pnl() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 103.0],
        &[100.0; 5],
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 1, 0, 0],
        confidences: &[0.5, 0.0, 1.0, 0.0, 0.0],
        sl_pips: &[2.0, 0.0, 1.0, 0.0, 0.0],
        tp_pips: &[4.0, 0.0, 3.0, 0.0, 0.0],
        ml_multipliers: &[0.5, 0.0, 0.25, 0.0, 0.0],
    };
    let out = run.evaluate(tape, None).unwrap();
    // First: 1000 * (1% + (3%-1%)*.5) / (2*10) * .5 = .5 lots.
    // (4 pips*10 - 2 commission)*.5 = 19. Second uses ACTUAL 1019 equity:
    // 1019*.03/(1*10)*.25 = .76425 lots; (3*10-2)*.76425 = 21.399.
    assert_eq!(out.closed_trades.len(), 2);
    near(out.closed_trades[0].pnl, 19.0);
    near(out.closed_trades[1].pnl, 21.399);
    near(out.closed_trades[0].r_multiple, 1.9);
    near(out.closed_trades[1].r_multiple, 2.8);
    near(out.ending_realized_balance, 1040.399);
    near(out.metrics.net_profit, 40.399);
    assert_eq!(out.closed_trades[0].entry_time, run.timestamps[1]);
    assert_eq!(out.closed_trades[1].entry_time, run.timestamps[3]);
    assert!(out.terminal_open.is_none());
    assert!(!out.promotion_eligible);
    assert!(out.execution_basis.contains("ohlc_screening"));
}

#[test]
fn below_min_is_non_entry_and_next_signal_can_occupy_the_position() {
    let mut run = prepared(&[100.0; 4], &[100.0, 100.0, 100.0, 104.0], &[100.0; 4]);
    // Keep the requested 1.0 lot exact: this case tests occupancy rather than
    // a floating-point value immediately below a broker grid boundary.
    run.settings.risk_per_trade_min = 0.02;
    run.settings.risk_per_trade_max = 0.02;
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 1, 0, 0],
                confidences: &[0.5; 4],
                sl_pips: &[2.0; 4],
                tp_pips: &[4.0; 4],
                ml_multipliers: &[0.25, 1.0, 0.0, 0.0],
            },
            Some(BarReplayLotGridV1::new(0.5, 100.0, 0.5).unwrap()),
        )
        .unwrap();
    assert_eq!(out.below_min_entries, 1);
    assert_eq!(out.closed_trades.len(), 1);
    assert_eq!(out.closed_trades[0].entry_time, run.timestamps[2]);
    near(out.closed_trades[0].pnl, 38.0);
    near(out.ending_realized_balance, 1038.0);
    assert!(out.terminal_open.is_none());
}

#[test]
fn terminal_position_is_retained_without_fabricated_close_or_realized_profit() {
    let run = prepared(
        &[100.0, 100.0, 101.0],
        &[100.0, 100.0, 101.0],
        &[100.0, 100.0, 101.0],
    );
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 0, 0],
                confidences: &[0.5; 3],
                sl_pips: &[10.0; 3],
                tp_pips: &[20.0; 3],
                ml_multipliers: &[0.5, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    assert!(out.closed_trades.is_empty());
    assert_eq!(out.metrics.trade_count, 0);
    near(out.metrics.net_profit, 0.0);
    near(out.ending_realized_balance, 1000.0);
    let open = out.terminal_open.unwrap();
    assert_eq!(open.direction, 1);
    assert_eq!(open.entry_bar_index, 1);
    assert_eq!(open.entry_timestamp_ms, run.timestamps[1]);
    assert_eq!(open.mark_timestamp_ms, run.timestamps[2]);
    near(open.lots, 0.1);
    near(open.stop_pips, 10.0);
    near(open.target_pips, 20.0);
    assert!(open.active_trailing_stop_price.is_none());
    near(open.gross_unrealized_account, 1.0);
    near(open.pending_round_trip_commission_account, 0.2);
    near(open.marked_equity_before_pending_costs, 1001.0);
}

#[test]
fn future_tape_cannot_change_a_prior_entry_and_ml_veto_cannot_disable_protection() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 100.0],
        &[100.0; 5],
    );
    let baseline = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, 0, 0, 0, 0],
                confidences: &[0.5; 5],
                sl_pips: &[2.0; 5],
                tp_pips: &[4.0; 5],
                ml_multipliers: &[1.0, 0.0, 0.0, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    let changed_future = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1, -1, -1, -1, -1],
                confidences: &[0.5, 1.0, 1.0, 1.0, 1.0],
                sl_pips: &[2.0, 8.0, 8.0, 8.0, 8.0],
                tp_pips: &[4.0, 16.0, 16.0, 16.0, 16.0],
                ml_multipliers: &[1.0, 0.0, 0.0, 0.0, 0.0],
            },
            None,
        )
        .unwrap();
    assert_eq!(baseline.closed_trades.len(), 1);
    assert_eq!(changed_future.closed_trades.len(), 1);
    assert_eq!(
        serde_json::to_value(&baseline.closed_trades).unwrap(),
        serde_json::to_value(&changed_future.closed_trades).unwrap()
    );
    near(changed_future.closed_trades[0].pnl, 38.0);
    assert_eq!(
        changed_future.closed_trades[0].exit_time,
        Some(run.timestamps[2])
    );
    assert!(changed_future.terminal_open.is_none());
}

#[test]
fn all_one_multipliers_and_fixed_brackets_match_legacy_core_bit_for_bit() {
    let run = prepared(
        &[100.0; 5],
        &[100.0, 100.0, 104.0, 100.0, 104.0],
        &[100.0; 5],
    );
    let signals = [1, 0, 1, 0, 0];
    let confidences = [0.5; 5];
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &signals,
                confidences: &confidences,
                sl_pips: &[2.0; 5],
                tp_pips: &[4.0; 5],
                ml_multipliers: &[1.0; 5],
            },
            None,
        )
        .unwrap();
    let (legacy_metrics, legacy_trades) =
        crate::eval::evaluate_strategy_with_confidence_and_ledger_core(
            &run.close,
            &run.high,
            &run.low,
            &signals,
            &confidences,
            &run.months,
            &run.days,
            &run.timestamps,
            &run.settings,
        )
        .unwrap();
    assert_eq!(
        out.metrics.to_metric_array().map(f64::to_bits),
        legacy_metrics.map(f64::to_bits)
    );
    assert_eq!(
        serde_json::to_value(&out.closed_trades).unwrap(),
        serde_json::to_value(&legacy_trades).unwrap()
    );
}

#[test]
fn malformed_tapes_and_bracket_overflow_fail_before_statistics() {
    let run = prepared(&[100.0; 3], &[100.0; 3], &[100.0; 3]);
    let signals = [1, 0, 0];
    let confidence = [0.5; 3];
    let stops = [2.0; 3];
    let targets = [4.0; 3];
    let multipliers = [1.0; 3];
    let tape = NettedBarDecisionTapeV1 {
        signals: &signals,
        confidences: &confidence,
        sl_pips: &stops,
        tp_pips: &targets,
        ml_multipliers: &multipliers,
    };
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                confidences: &[0.5],
                ..tape
            },
            None
        )
        .is_err()
    );
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(
            run.evaluate(
                NettedBarDecisionTapeV1 {
                    ml_multipliers: &[value; 3],
                    ..tape
                },
                None
            )
            .is_err()
        );
    }
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                sl_pips: &[f64::MAX; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                tp_pips: &[f64::MAX; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[2, 0, 0],
                ..tape
            },
            None
        )
        .is_err()
    );
    assert!(
        run.evaluate(
            NettedBarDecisionTapeV1 {
                sl_pips: &[0.0; 3],
                ..tape
            },
            None
        )
        .is_err()
    );
}

#[test]
fn broker_grid_never_rounds_up_to_force_a_trade_and_zero_risk_is_not_below_min() {
    let grid = BarReplayLotGridV1::new(0.5, 2.0, 0.25).unwrap();
    assert_eq!(grid.normalize_down(0.49), None);
    assert_eq!(grid.normalize_down(0.99), Some(0.75));
    assert_eq!(grid.normalize_down(3.0), Some(2.0));
    assert_eq!(grid.normalize_down(f64::INFINITY), None);
    assert!(BarReplayLotGridV1::new(0.0, 2.0, 0.25).is_err());
    let mut run = prepared(&[100.0; 3], &[100.0; 3], &[100.0; 3]);
    run.settings.risk_per_trade_min = 0.0;
    run.settings.risk_per_trade_max = 0.0;
    let out = run
        .evaluate(
            NettedBarDecisionTapeV1 {
                signals: &[1; 3],
                confidences: &[0.5; 3],
                sl_pips: &[2.0; 3],
                tp_pips: &[4.0; 3],
                ml_multipliers: &[1.0; 3],
            },
            Some(grid),
        )
        .unwrap();
    assert_eq!(out.below_min_entries, 0);
    assert!(out.closed_trades.is_empty());
    assert!(out.terminal_open.is_none());
}

#[test]
fn public_preparation_evaluates_exact_published_canonical_holdout_and_rejects_rebinding() {
    use crate::data_selection::CanonicalSearchInput;
    use crate::{CanonicalSearchEvaluatedWindowV1, CanonicalTrendbarResearchCostAssumptionsV2};
    use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
    use neoethos_data::{
        BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
        CanonicalTimeframe, CanonicalVolumeRef, FeatureCellValidity, FeatureColumnF64,
        FeatureFrame, publish_canonical_ohlcv_generation,
    };
    use neoethos_feature_contracts::{
        DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    };

    // Actual captured fixture prices, persisted and reopened by the real
    // canonical generation path. The explicitly external fixture identity is
    // not broker-truth or live-account evidence.
    let root = tempfile::tempdir().unwrap();
    let anchor = CanonicalDatasetIdentity::external(
        "netted-public-canonical-fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    let source = ProducerProvenanceEnvelopeV1::new(
        "neoethos.netted-public-canonical-fixture.v1",
        anchor.canonical_bytes(),
    )
    .unwrap();
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root.path(),
        identity: &anchor,
        expected_generation: None,
        provenance: &source,
        ohlcv: &bars,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 128,
    })
    .unwrap();
    let dataset =
        neoethos_data::load_dataset_for_identity_with_timeframes(root.path(), &anchor, &["M1"])
            .unwrap();
    let base = dataset.canonical_frame("M1").unwrap();
    let source_id = "source:netted-public-canonical-fixture";
    let plan = FeaturePlanV1::new(
        vec![
            FeatureNodeV1::source(
                source_id,
                anchor.clone(),
                "neoethos.netted-public-fixture.close.v1",
                1,
                vec![FeatureOutputV1::f64("close", 1).unwrap()],
                [19; 32],
            )
            .unwrap(),
        ],
        vec!["close".into()],
    )
    .unwrap();
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![base.source_binding(source_id).unwrap()],
    )
    .unwrap();
    let features = FeatureFrame::from_columns(
        base.ohlcv().timestamp.clone().unwrap(),
        vec![
            FeatureColumnF64::new(
                "close",
                base.ohlcv().close.clone(),
                vec![FeatureCellValidity::Valid; base.len()],
            )
            .unwrap(),
        ],
        plan,
        provenance,
    )
    .unwrap();
    let input =
        CanonicalSearchInput::from_prepared_canonical_frame(anchor, base, features).unwrap();
    let run = input.as_run_input().unwrap();
    let scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::Holdout,
        &run,
        20..28,
    )
    .unwrap();
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        run.receipt().clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "explicit-public-connection-test-costs",
            assumption_source_sha256: &"c".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.5,
            slippage_pips_per_fill_assumption: 0.5,
            commission_account_per_lot_per_fill_assumption: 7.0,
            swap_long_pips_per_day: -0.25,
            swap_short_pips_per_day: 0.1,
            pnl_conversion_fee_rate: 0.01,
        },
    )
    .unwrap();
    // The published symbol rate above is nonzero, but EURUSD price P&L is
    // already USD. The sealed account-specific assumption must charge no
    // currency-conversion fee when quote and account currencies match.
    assert_eq!(contract.pnl_conversion_fee_rate(), 0.0);
    let evaluation = EvaluationConfig {
        symbol: "EURUSD".into(),
        account_currency: "USD".into(),
        initial_equity: 1_000.0,
        max_hold_bars: 1,
        trailing_enabled: false,
        trailing_atr_multiplier: 1.0,
        trailing_be_trigger_r: 1.0,
        trailing_min_lock_pips: 0.0,
        pip_value: 0.0001,
        pip_value_per_lot: 10.0,
        spread_pips: 2.5,
        commission_per_trade: 14.0,
        swap_long_pips_per_day: -0.25,
        swap_short_pips_per_day: 0.1,
        pnl_conversion_fee_rate: contract.pnl_conversion_fee_rate(),
        kill_zones_enabled: false,
        session_spread_pips: None,
        risk_per_trade_min: 0.02,
        risk_per_trade_max: 0.02,
        high_quality_confidence: 1.0,
        smc_gate_threshold: 0.0,
        smc_weight_ob: 1.0,
        smc_weight_fvg: 1.0,
        smc_weight_liq: 1.0,
        smc_weight_mtf: 1.0,
        smc_weight_premium: 1.0,
        smc_weight_inducement: 1.0,
        smc_weight_bos: 1.0,
        smc_weight_choch: 1.0,
        smc_weight_eqh: 1.0,
        smc_weight_eql: 1.0,
        smc_weight_displacement: 1.0,
        growth_objective: false,
        growth_goal: None,
    };
    contract.validate_evaluation_costs(&evaluation).unwrap();
    let prepared =
        PreparedNettedCanonicalBarResearchV1::from_input(&run, &scope, &contract, &evaluation)
            .unwrap();
    assert_eq!(
        prepared.timestamps(),
        &bars.timestamp.as_ref().unwrap()[20..28]
    );
    let tape = NettedBarDecisionTapeV1 {
        signals: &[1, 0, 0, 0, 0, 0, 0, 0],
        confidences: &[0.5; 8],
        sl_pips: &[100.0; 8],
        tp_pips: &[200.0; 8],
        ml_multipliers: &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let out = prepared.evaluate(tape, None).unwrap();
    // This tests explicit *bar-assumption* accounting, not broker settlement.
    // Size =1000*.02/(100*10)*.5=.01 lots.
    // Two half-spreads sum to2.5 pips; two fill commissions sum to14/lot.
    // Carry is prorated elapsed days. No USD-to-USD conversion takes place:
    // neither a price loss nor commission/swap may receive the old 1% discount.
    let elapsed_days = (bars.timestamp.as_ref().unwrap()[22] - bars.timestamp.as_ref().unwrap()[21])
        as f64
        / 86_400_000.0;
    let price_gross_account = ((bars.close[22] - bars.close[21]) / 0.0001 * 10.0 - 25.0) * 0.01;
    let commission_account = 14.0 * 0.01;
    let swap_account = -0.25 * elapsed_days * 10.0 * 0.01;
    let expected = price_gross_account - commission_account + swap_account;
    assert_eq!(out.closed_trades.len(), 1);
    assert_eq!(
        out.closed_trades[0].entry_time,
        bars.timestamp.as_ref().unwrap()[21]
    );
    assert_eq!(
        out.closed_trades[0].exit_time,
        Some(bars.timestamp.as_ref().unwrap()[22])
    );
    near(out.closed_trades[0].pnl, expected);
    near(out.ending_realized_balance, 1000.0 + expected);
    assert_eq!(out.scope_identity_sha256, scope.identity_sha256().unwrap());
    assert_eq!(
        out.cost_contract_identity_sha256,
        contract.identity_sha256().unwrap()
    );
    assert_eq!(out.account_currency, "USD");
    assert!(out.terminal_open.is_none());
    assert!(!out.promotion_eligible);

    let wrong_role = CanonicalSearchArtifactScopeV2::from_run_input_range(
        CanonicalSearchWindowRoleV1::InSample,
        &run,
        20..28,
    )
    .unwrap();
    assert!(PreparedNettedCanonicalBarResearchV1::from_input(
        &run, &wrong_role, &contract, &evaluation,
    ).is_err());
    let wrong_rows = CanonicalSearchArtifactScopeV2::new(
        run.receipt().clone(),
        CanonicalSearchEvaluatedWindowV1::new(
            CanonicalSearchWindowRoleV1::Holdout,
            21,
            29,
            bars.timestamp.as_ref().unwrap()[20],
            bars.timestamp.as_ref().unwrap()[27],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(PreparedNettedCanonicalBarResearchV1::from_input(
        &run, &wrong_rows, &contract, &evaluation,
    ).is_err());
    for (field, field_name) in [
        "account_currency",
        "symbol",
        "pip_value",
        "pip_value_per_lot",
        "spread_pips",
        "commission_per_trade",
        "swap_long_pips_per_day",
        "swap_short_pips_per_day",
        "pnl_conversion_fee_rate",
    ]
    .iter()
    .enumerate()
    {
        let mut changed = evaluation.clone();
        match field {
            0 => changed.account_currency = "EUR".into(),
            1 => changed.symbol = "GBPUSD".into(),
            2 => changed.pip_value = 0.01,
            3 => changed.pip_value_per_lot = 20.0,
            4 => changed.spread_pips += 0.1,
            5 => changed.commission_per_trade += 0.1,
            6 => changed.swap_long_pips_per_day += 0.1,
            7 => changed.swap_short_pips_per_day += 0.1,
            _ => changed.pnl_conversion_fee_rate = f64::EPSILON,
        }
        let expected_error = contract
            .validate_evaluation_costs(&changed)
            .unwrap_err()
            .to_string();
        assert!(expected_error.contains(&format!(": {field_name} (evaluation ")));
        let preparation_error =
            PreparedNettedCanonicalBarResearchV1::from_input(&run, &scope, &contract, &changed)
                .err()
                .expect("financial field must not detach from its contract");
        assert_eq!(preparation_error.to_string(), expected_error);
        contract.validate_evaluation_costs(&evaluation).unwrap();
    }
    // Preparation owns only the bounded held-out data: subsequent evaluation
    // remains possible after both source dataset and feature cube are gone.
    drop(run);
    drop(input);
    drop(dataset);
    let detached = prepared.evaluate(tape, None).unwrap();
    assert_eq!(detached.metrics, out.metrics);
    near(detached.ending_realized_balance, 1000.0 + expected);
}
