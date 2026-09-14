use super::*;
use chrono::{TimeZone, Utc};

fn settings() -> BacktestSettings {
    BacktestSettings {
        pip_value: 1.0,
        pip_value_per_lot: 1.0,
        spread_pips: 0.0,
        commission_per_trade: 0.0,
        swap_long_pips_per_day: 0.0,
        swap_short_pips_per_day: 0.0,
        pnl_conversion_fee_rate: 0.0,
        sl_pips: 2.0,
        tp_pips: 10.0,
        trailing_enabled: false,
        risk_based_sizing: false,
        kill_zones_enabled: true,
        ..BacktestSettings::default()
    }
}

fn timestamp(day: u32, hour: u32, minute: u32) -> i64 {
    Utc.with_ymd_and_hms(2026, 9, day, hour, minute, 0)
        .single()
        .unwrap()
        .timestamp_millis()
}

fn assert_ledger(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    times: &[i64],
    signals: &[i8],
    settings: &BacktestSettings,
    expected_pnls: &[f64],
) -> Vec<Trade> {
    let days = times
        .iter()
        .map(|ts| ts.div_euclid(86_400_000))
        .collect::<Vec<_>>();
    let metrics = fast_evaluate_strategy_core(
        close,
        high,
        low,
        signals,
        &[],
        &vec![0; close.len()],
        &days,
        times,
        settings,
    );
    let trades = simulate_trades_core(close, high, low, times, signals, settings);
    assert_eq!(
        metrics[8] as usize,
        expected_pnls.len(),
        "fitness trade count"
    );
    assert_eq!(trades.len(), expected_pnls.len(), "ledger trade count");
    for (trade, expected) in trades.iter().zip(expected_pnls) {
        assert!(
            (trade.pnl - expected).abs() < 1e-9,
            "ledger PnL {} != {expected}",
            trade.pnl
        );
    }
    let expected_net = expected_pnls.iter().sum::<f64>();
    assert!(
        (metrics[0] - expected_net).abs() < 1e-9,
        "fitness net {} != {expected_net}",
        metrics[0]
    );
    trades
}

#[test]
fn friday_close_is_priced_by_fitness_and_the_trade_ledger() {
    let times = [
        timestamp(4, 19, 50),
        timestamp(4, 19, 55),
        timestamp(4, 20, 0),
    ];
    let trades = assert_ledger(
        &[100.0, 100.0, 105.0],
        &[100.0, 100.0, 106.0],
        &[100.0, 100.0, 99.0],
        &times,
        &[1, 0, 0],
        &settings(),
        &[5.0],
    );
    assert_eq!(trades[0].exit_time, Some(times[2]));
}

#[test]
fn intrabar_protection_precedes_the_friday_close_of_bar_exit() {
    for side in [-1, 1] {
        let times = [
            timestamp(4, 19, 50),
            timestamp(4, 19, 55),
            timestamp(4, 20, 0),
        ];
        let (high, low) = if side == 1 {
            (106.0, 97.0)
        } else {
            (103.0, 94.0)
        };
        assert_ledger(
            &[100.0, 100.0, 100.0 + f64::from(side) * 5.0],
            &[100.0, 100.0, high],
            &[100.0, 100.0, low],
            &times,
            &[side, 0, 0],
            &settings(),
            &[-2.0],
        );
    }
}

#[test]
fn monday_and_friday_entry_blackouts_apply_to_fitness_too() {
    for (day, hour) in [(7, 0), (4, 20)] {
        let times = [
            timestamp(day, hour, 0),
            timestamp(day, hour, 5),
            timestamp(day, hour, 10),
        ];
        assert_ledger(
            &[100.0; 3],
            &[100.0; 3],
            &[100.0, 100.0, 97.0],
            &times,
            &[1, 0, 0],
            &settings(),
            &[],
        );
    }
}

#[test]
fn disabled_session_policy_does_not_invent_a_friday_exit() {
    let times = [
        timestamp(4, 19, 50),
        timestamp(4, 19, 55),
        timestamp(4, 20, 0),
    ];
    let mut settings = settings();
    settings.kill_zones_enabled = false;
    assert_ledger(
        &[100.0, 100.0, 105.0],
        &[100.0, 100.0, 106.0],
        &[100.0, 100.0, 99.0],
        &times,
        &[1, 0, 0],
        &settings,
        &[],
    );
}

#[test]
fn a_forced_gap_close_cannot_reenter_using_the_pre_gap_signal() {
    let start = timestamp(1, 12, 0);
    let times = [
        start,
        start + 300_000,
        start + 5 * 86_400_000,
        start + 5 * 86_400_000 + 300_000,
    ];
    let mut settings = settings();
    settings.kill_zones_enabled = false;
    assert_ledger(
        &[100.0, 100.0, 105.0, 105.0],
        &[100.0, 100.0, 106.0, 106.0],
        &[100.0, 100.0, 104.0, 100.0],
        &times,
        &[1, 1, 0, 0],
        &settings,
        &[5.0],
    );
}

#[test]
fn recording_does_not_change_risk_sized_metrics_or_compounding() {
    let start = timestamp(1, 12, 0);
    let times = (0..5).map(|i| start + i * 300_000).collect::<Vec<_>>();
    let close = [100.0, 100.0, 110.0, 100.0, 102.0];
    let high = [100.0, 100.0, 110.0, 100.0, 102.0];
    let low = [100.0, 100.0, 100.0, 100.0, 100.0];
    let signals = [1, 0, -1, 0, 0];
    let confidence = [1.0; 5];
    let mut settings = settings();
    settings.risk_based_sizing = true;
    settings.risk_per_trade_min = 0.01;
    settings.risk_per_trade_max = 0.01;
    settings.pip_value_per_lot = 100.0;
    let months = [0; 5];
    let days = [times[0] / 86_400_000; 5];
    let fast = fast_evaluate_strategy_core(
        &close,
        &high,
        &low,
        &signals,
        &confidence,
        &months,
        &days,
        &times,
        &settings,
    );
    let mut trades = Vec::new();
    let recorded = evaluate_strategy_with_ledger_core::<true>(
        &close,
        &high,
        &low,
        &signals,
        &confidence,
        &months,
        &days,
        &times,
        &settings,
        &mut trades,
        None,
    );
    assert_eq!(fast.map(f64::to_bits), recorded.map(f64::to_bits));
    assert_eq!(trades.len(), 2);
    let first_profit = settings.initial_equity() * 0.01 * 5.0;
    let next_loss = -(settings.initial_equity() + first_profit) * 0.01;
    assert!((trades[0].pnl - first_profit).abs() < 1e-9);
    assert!((trades[1].pnl - next_loss).abs() < 1e-9);
    assert!((trades[0].r_multiple - 5.0).abs() < 1e-12);
    assert!((trades[1].r_multiple + 1.0).abs() < 1e-12);
    assert!((trades[0].mfe - first_profit).abs() < 1e-9);
    assert!((trades[1].mae + next_loss).abs() < 1e-9);
}

#[test]
fn trailing_can_reach_a_payoff_above_the_retired_empirical_cap() {
    let start = timestamp(1, 12, 0);
    let times = (0..5).map(|i| start + i * 300_000).collect::<Vec<_>>();
    let mut settings = settings();
    settings.trailing_enabled = true;
    settings.trailing_be_trigger_r = 1.0;
    settings.trailing_atr_multiplier = 1.0;
    let trades = assert_ledger(
        &[100.0, 100.0, 110.0, 100.0, 98.0],
        &[100.0, 100.0, 110.0, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0, 98.0],
        &times,
        &[1, 0, 1, 0, 0],
        &settings,
        &[10.0, -2.0],
    );
    let realized_payoff = trades[0].pnl / -trades[1].pnl;
    assert_eq!(realized_payoff, 5.0);
    assert!(realized_payoff > crate::run_identity::MEASURED_TRAILING_PAYOFF_CEILING);
    let inputs = crate::run_identity::PayoffCeilingInputs {
        sl_min_pips: 2.0,
        sl_max_pips: 2.0,
        tp_min_pips: 10.0,
        tp_max_pips: 10.0,
        initializer_rr_min: 5.0,
        initializer_rr_max: 5.0,
        atr_pips: None,
        cost_pips_round_trip: 0.0,
        trailing_enabled: true,
        trailing_be_trigger_r: 1.0,
        trailing_give_back_r: 1.0,
        trailing_min_lock_pips: 2.0,
    };
    assert!(crate::run_identity::assert_payoff_floor_reachable(2.0, &inputs).is_ok());
}

#[test]
fn fixed_lot_costs_swaps_and_session_spreads_reconcile_to_one_ledger() {
    let start = timestamp(1, 6, 50);
    let times = (0..30).map(|i| start + i * 3_600_000).collect::<Vec<_>>();
    let close = vec![100.0; 30];
    let high = vec![100.5; 30];
    let low = vec![99.5; 30];
    let signals = vec![1; 30];
    let days = times.iter().map(|ts| ts / 86_400_000).collect::<Vec<_>>();
    let mut settings = settings();
    settings.max_hold_bars = 3;
    settings.spread_pips = 0.2;
    settings.commission_per_trade = 0.3;
    settings.swap_long_pips_per_day = -0.5;
    settings.pnl_conversion_fee_rate = 0.01;
    settings.session_spread_profile = Some(SessionSpreadProfile {
        asian_pips: 0.4,
        overlap_pips: 0.2,
        late_ny_pips: 0.3,
    });
    let metrics = fast_evaluate_strategy_core(
        &close,
        &high,
        &low,
        &signals,
        &[],
        &vec![0; 30],
        &days,
        &times,
        &settings,
    );
    let trades = simulate_trades_core(&close, &high, &low, &times, &signals, &settings);
    assert!(!trades.is_empty());
    for trade in &trades {
        // All prices are flat. Both half-spreads, one commission, fractional
        // three-hour carry and the price-gross conversion debit determine PnL.
        // The conversion debit does not discount commission or signed carry.
        let profile = settings.session_spread_profile.unwrap();
        let spread = (profile.spread_pips_at(trade.entry_time)
            + profile.spread_pips_at(trade.exit_time.unwrap()))
            * 0.5;
        let expected = -spread - 0.3 - 0.5 * 3.0 / 24.0 - spread * 0.01;
        assert!((trade.pnl - expected).abs() < 1e-12);
    }
    assert_eq!(metrics[8] as usize, trades.len());
    assert!((metrics[0] - trades.iter().map(|t| t.pnl).sum::<f64>()).abs() < 1e-9);
}

#[test]
fn account_risk_ledger_prices_signal_confidence_compounding_and_all_costs() {
    let times = [
        timestamp(1, 12, 0),
        timestamp(1, 12, 5),
        timestamp(2, 12, 5),
        timestamp(2, 12, 10),
        timestamp(3, 12, 10),
    ];
    let close = [100.0, 100.0, 121.0, 100.0, 109.0];
    let high = [100.0, 100.0, 121.0, 100.0, 109.0];
    let low = [100.0, 100.0, 100.0, 100.0, 99.0];
    let signals = [1, 0, -1, 0, 0];
    // The fill bar has the opposite confidence to its signal bar. Only the
    // prior signal is known when sizing; using the fill confidence changes lots.
    let confidence = [0.0, 1.0, 1.0, 0.0, 0.0];
    let mut policy = settings();
    policy.initial_equity_override = Some(10_000.0);
    policy.kill_zones_enabled = false;
    policy.risk_based_sizing = true;
    policy.risk_per_trade_min = 0.01;
    policy.risk_per_trade_max = 0.02;
    policy.high_quality_confidence = 1.0;
    policy.sl_pips = 10.0;
    policy.tp_pips = 20.0;
    policy.pip_value_per_lot = 100.0;
    policy.spread_pips = 2.0;
    policy.commission_per_trade = 4.0;
    policy.swap_long_pips_per_day = -0.5;
    policy.swap_short_pips_per_day = 0.25;
    policy.pnl_conversion_fee_rate = 0.01;
    let days: Vec<_> = times.iter().map(|ts| ts.div_euclid(86_400_000)).collect();
    let (metrics, trades) = evaluate_strategy_with_confidence_and_ledger_core(
        &close,
        &high,
        &low,
        &signals,
        &confidence,
        &[0; 5],
        &days,
        &times,
        &policy,
    )
    .expect("complete account-risk replay");
    // Long entry=101, TP=121. Risk cash=100, stop cash/lot=1000 => 0.1 lot.
    // Gross=2000/lot, exit half spread=100, commission=4, one-day swap=-50.
    let first_lots = 100.0 / 1_000.0;
    let first_net = (2_000.0 - 100.0 - 4.0 - 50.0) * first_lots - 1_900.0 * first_lots * 0.01;
    // Short entry=99, SL=109. Next risk uses the realized, net-of-cost balance.
    // One-day positive short swap adds 25/lot; the fee still debits the price loss.
    let second_risk = (10_000.0 + first_net) * 0.02;
    let second_lots = second_risk / 1_000.0;
    let second_net = (-1_000.0 - 100.0 - 4.0 + 25.0) * second_lots - 1_100.0 * second_lots * 0.01;
    assert_eq!(trades.len(), 2);
    for (trade, (net, risk_cash)) in trades
        .iter()
        .zip([(first_net, 100.0), (second_net, second_risk)])
    {
        assert!((trade.pnl - net).abs() < 1e-9, "{} != {net}", trade.pnl);
        assert!((trade.pnl_pct.unwrap() - net / 10_000.0).abs() < 1e-12);
        assert!((trade.r_multiple - net / risk_cash).abs() < 1e-12);
        assert_eq!(trade.duration_hours, Some(24.0));
    }
    assert!((metrics[0] - first_net - second_net).abs() < 1e-9);
    assert_eq!(metrics[8], 2.0);
    assert!((trades[0].mfe - 2_000.0 * first_lots).abs() < 1e-9);
    assert!((trades[1].mae - 1_000.0 * second_lots).abs() < 1e-9);
}

#[test]
fn account_risk_replay_rejects_missing_partial_and_invalid_confidence() {
    let mut policy = settings();
    policy.risk_based_sizing = true;
    for confidence in [
        vec![],
        vec![0.5; 2],
        vec![0.5, f64::NAN, 0.5],
        vec![0.5, 1.01, 0.5],
    ] {
        assert!(
            simulate_trades_with_confidence_core(
                &[100.0; 3],
                &[100.0; 3],
                &[100.0; 3],
                &[1, 2, 3],
                &[1, 0, 0],
                &confidence,
                &policy,
            )
            .is_err()
        );
    }
    policy.initial_equity_override = Some(f64::NAN);
    assert!(validate_sizing_confidences(3, &[0.5; 3], &policy).is_err());
    policy.initial_equity_override = Some(10_000.0);
    policy.risk_based_sizing = false;
    assert!(
        validate_sizing_confidences(3, &[], &policy).is_ok(),
        "fixed-lot is explicit"
    );
}

#[test]
fn parallel_population_ledger_retains_each_runs_balance_and_synthesized_confidence() {
    let cases = [10_000.0, 25_000.0];
    let results: Vec<_> = cases
        .par_iter()
        .map(|&balance| {
            let mut policy = settings();
            policy.initial_equity_override = Some(balance);
            policy.kill_zones_enabled = false;
            policy.risk_based_sizing = true;
            policy.risk_per_trade_min = 0.01;
            policy.risk_per_trade_max = 0.02;
            policy.high_quality_confidence = 1.0;
            policy.pip_value_per_lot = 100.0;
            let indicators = ndarray::arr2(&[[1.0, 0.0, -2.0, 0.0, 0.0]]);
            let times = [1, 2, 3, 4, 5];
            validation_backtest_population_cpu_core::<true>(PopulationEvalInputs {
                close: &[100.0, 100.0, 120.0, 100.0, 110.0],
                high: &[100.0, 100.0, 120.0, 100.0, 110.0],
                low: &[100.0; 5],
                indicators: indicators.view(),
                gene_offsets: &[0, 1],
                gene_indices: &[0],
                gene_weights: &[1.0],
                long_thr: &[1.0],
                short_thr: &[-1.0],
                month_idx: &[0; 5],
                day_idx: &[0; 5],
                timestamps: &times,
                sl_pips: &[10.0],
                tp_pips: &[20.0],
                stop_vol_mult: &[],
                smc_data: &[[0; 11]; 5],
                gene_smc_flags: &[[0; 11]],
                gate_threshold: 0.0,
                weights: &[0.0; 11],
                settings: &policy,
            })
            .expect("CPU mathematical population fixture")
        })
        .collect();
    for (balance, rows) in cases.into_iter().zip(results) {
        // score 1 equals the long threshold => confidence 0 => 1% risk, +2R.
        // score -2 is one below the short threshold across a gap of 2 =>
        // confidence 0.5 => 1.5% risk on the compounded balance, then -1R.
        let first = balance * 0.01 * 2.0;
        let second = -(balance + first) * 0.015;
        assert_eq!(rows.len(), 1);
        let (metrics, trades) = &rows[0];
        assert_eq!(trades.len(), 2);
        assert!((trades[0].pnl - first).abs() < 1e-9);
        assert!((trades[1].pnl - second).abs() < 1e-9);
        assert!((metrics[0] - first - second).abs() < 1e-9);
    }
}
