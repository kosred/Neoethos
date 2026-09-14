use super::*;
use crate::canonical_trendbar_research::{
    CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchExecutionContractV3,
    install_canonical_trendbar_research_execution_v3,
};

thread_local! {
    pub(super) static LEGACY_COST_RESOLUTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Reference the previous consumer, including the superseded monetary lookup,
// so this optimization proves every field, not just a few final cost values.
// The additive recorded goal follows the current run binding in both paths.
fn prior_evaluation_config(
    config: &DiscoveryConfig,
    price: Option<f64>,
    contract: Option<&CanonicalTrendbarResearchExecutionContractV3>,
) -> EvaluationConfig {
    let mut evaluation = EvaluationConfig::for_symbol(
        &config.evaluation_symbol,
        &config.evaluation_account_currency,
        price,
        Some(config.evaluation_spread_pips),
        Some(config.evaluation_commission_per_trade),
    );
    evaluation.kill_zones_enabled = config.kill_zones_enabled;
    evaluation.session_spread_pips = config.session_spread_pips;
    evaluation.risk_per_trade_min = config.risk_per_trade_min;
    evaluation.risk_per_trade_max = config.risk_per_trade_max;
    evaluation.high_quality_confidence = config.high_quality_confidence;
    evaluation.initial_equity = config.initial_balance;
    evaluation.swap_long_pips_per_day = config.swap_long_pips_per_day;
    evaluation.swap_short_pips_per_day = config.swap_short_pips_per_day;
    evaluation.pnl_conversion_fee_rate = config.pnl_conversion_fee_rate;
    evaluation.growth_objective = matches!(config.mode, DiscoveryMode::Risky);
    evaluation.growth_goal =
        evaluation
            .growth_objective
            .then_some(crate::scoring::RiskyGrowthGoal {
                start_balance: config.risky_start_balance,
                target_balance: config.risky_target_balance,
                horizon_days: config.risky_horizon_days,
            });
    if let Some(contract) = contract {
        evaluation.session_spread_pips = None;
        if config.evaluation_symbol == contract.symbol()
            && config.evaluation_account_currency == contract.account_currency()
        {
            evaluation.pip_value = contract.pip_size();
            evaluation.pip_value_per_lot = contract.pip_value_per_lot();
            evaluation.spread_pips = contract.screening_spread_and_slippage_round_trip_pips();
            evaluation.commission_per_trade = contract.round_trip_commission_account_per_lot();
            evaluation.swap_long_pips_per_day = contract.swap_long_pips_per_day();
            evaluation.swap_short_pips_per_day = contract.swap_short_pips_per_day();
            evaluation.pnl_conversion_fee_rate = contract.pnl_conversion_fee_rate();
        } else {
            evaluation.pip_value = f64::NAN;
            evaluation.pip_value_per_lot = f64::NAN;
            evaluation.spread_pips = f64::NAN;
            evaluation.commission_per_trade = f64::NAN;
            evaluation.swap_long_pips_per_day = f64::NAN;
            evaluation.swap_short_pips_per_day = f64::NAN;
            evaluation.pnl_conversion_fee_rate = f64::NAN;
        }
    }
    evaluation
}

fn assert_same_evaluation_config(actual: &EvaluationConfig, expected: &EvaluationConfig) {
    macro_rules! compare_fields {
        (values: [$($value:ident),* $(,)?], floats: [$($float:ident),* $(,)?]) => {
            let EvaluationConfig { $($value,)* $($float,)* } = actual;
            $(assert_eq!($value, &expected.$value, stringify!($value));)*
            $(assert_eq!($float.to_bits(), expected.$float.to_bits(), stringify!($float));)*
        };
    }
    // No `..`: adding an evaluation field requires extending this proof.
    compare_fields! {
        values: [symbol, account_currency, max_hold_bars, trailing_enabled,
            kill_zones_enabled, session_spread_pips, growth_objective, growth_goal],
        floats: [initial_equity, trailing_atr_multiplier, trailing_be_trigger_r,
            trailing_min_lock_pips, pip_value, spread_pips, commission_per_trade,
            pip_value_per_lot, swap_long_pips_per_day, swap_short_pips_per_day,
            pnl_conversion_fee_rate, risk_per_trade_min, risk_per_trade_max,
            high_quality_confidence, smc_gate_threshold, smc_weight_ob, smc_weight_fvg,
            smc_weight_liq, smc_weight_mtf, smc_weight_premium, smc_weight_inducement,
            smc_weight_bos, smc_weight_choch, smc_weight_eqh, smc_weight_eql,
            smc_weight_displacement]
    }
}

fn settings_with_spread(spread: f64, slippage_per_fill: f64) -> neoethos_core::Settings {
    let mut settings = neoethos_core::Settings::default();
    settings.system.symbol = "EURUSD".to_owned();
    settings.system.account_currency = "USD".to_owned();
    settings.risk.backtest_spread_pips = spread;
    settings.risk.slippage_pips = slippage_per_fill;
    settings
}

fn research_contract() -> CanonicalTrendbarResearchExecutionContractV3 {
    research_contract_with_financials(7.0, 0.0, 0.0)
}

fn research_contract_with_financials(
    commission_per_fill: f64,
    swap_long: f64,
    swap_short: f64,
) -> CanonicalTrendbarResearchExecutionContractV3 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0].dataset_identity();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &features)
        .expect("receipt for the cost-formula fixture");
    CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.discovery-cost-consistency.v1",
            assumption_source_sha256: &"a".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.5,
            slippage_pips_per_fill_assumption: 0.5,
            commission_account_per_lot_per_fill_assumption: commission_per_fill,
            swap_long_pips_per_day: swap_long,
            swap_short_pips_per_day: swap_short,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .expect("explicit research-only cost assumptions")
}

#[derive(Clone)]
struct DiagnosticWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

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
    let bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let writer = DiagnosticWriter(std::sync::Arc::clone(&bytes));
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

fn assert_same_resolved_discovery_config(actual: &DiscoveryConfig, expected: &DiscoveryConfig) {
    // DiscoveryConfig intentionally has no wire format. Destructure without
    // `..` so a newly added resolved field cannot silently escape this check.
    // Scalar costs/risk values are bit-exact, not rounded JSON or Debug text.
    macro_rules! compare_fields {
        (values: [$($value:ident),* $(,)?],
         floats: [$($float:ident),* $(,)?],
         serialized: [$($serialized:ident),* $(,)?]) => {
            let DiscoveryConfig {
                $($value,)*
                $($float,)*
                $($serialized,)*
            } = actual;
            $(assert_eq!($value, &expected.$value, stringify!($value));)*
            $(assert_eq!(
                $float.to_bits(), expected.$float.to_bits(), stringify!($float)
            );)*
            // These two nested types already define Serialize, but not
            // PartialEq. Compare every field in their existing typed format.
            $(assert_eq!(
                serde_json::to_value($serialized).unwrap(),
                serde_json::to_value(&expected.$serialized).unwrap(),
                stringify!($serialized)
            );)*
        };
    }
    compare_fields! {
        values: [
            timeframe_label, evaluation_symbol, evaluation_account_currency,
            session_spread_pips, cost_band_pips, kill_zones_enabled,
            population, population_auto, generations, max_indicators,
            candidate_count, portfolio_size, max_rows, max_rows_by_timeframe,
            target_profile, walkforward_splits, embargo_minutes, enable_cpcv,
            cpcv_n_splits, cpcv_n_test_groups, cpcv_max_rows,
            risky_risk_band, prop_firm_risk_band, higher_timeframes,
            runtime_overrides, mc_runs, mc_min_profitable, adaptive_thresholds,
            mode, prop_firm_gate_params, require_walkforward_for_export,
            discovery_ledger_enabled, discovery_ledger_cache_dir,
            discovery_ledger_archive_top_n,
        ],
        floats: [
            evaluation_spread_pips, evaluation_commission_per_trade,
            swap_long_pips_per_day, swap_short_pips_per_day,
            pnl_conversion_fee_rate, max_hours, corr_threshold,
            min_trades_per_day, cpcv_embargo_pct, cpcv_purge_pct, cpcv_min_phi,
            max_pbo, initial_balance, risk_per_trade_min, risk_per_trade_max,
            high_quality_confidence, max_regime_loss_pct,
            sensitivity_spread_pips, sensitivity_commission_per_lot,
            risky_start_balance, risky_target_balance, risky_horizon_days,
            prop_firm_min_pass_rate,
        ],
        serialized: [filtering, prop_firm_gate]
    }
}

#[test]
fn research_constructor_logs_only_its_final_sealed_costs_without_changing_resolution() {
    let mut settings = settings_with_spread(1.5, 0.5);
    settings.models.prop_search_sensitivity_commission_per_lot = 1.0;
    let contract = research_contract_with_financials(4.5, -2.445, -0.105);
    let identity = contract.identity_sha256().unwrap();
    let (actual, logs) = capture_diagnostics(|| {
        DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(&settings, &contract)
            .expect("valid receipt-bound research constructor")
    });
    let mut previous_resolution = DiscoveryConfig::from_settings(&settings);
    apply_research_contract_to_discovery_config(&mut previous_resolution, &contract);
    assert_same_resolved_discovery_config(&actual, &previous_resolution);
    assert_eq!(actual.evaluation_spread_pips.to_bits(), 2.5_f64.to_bits());
    assert_eq!(
        actual.evaluation_commission_per_trade.to_bits(),
        9.0_f64.to_bits()
    );
    assert_eq!(
        actual.swap_long_pips_per_day.to_bits(),
        (-2.445_f64).to_bits()
    );
    assert_eq!(
        actual.swap_short_pips_per_day.to_bits(),
        (-0.105_f64).to_bits()
    );
    assert!(actual.session_spread_pips.is_none());
    assert_eq!(contract.identity_sha256().unwrap(), identity);
    for expected in [
        "classification=\"research_only\"",
        "full_spread_pips=1.5",
        "entry_slippage_pips=0.5",
        "exit_slippage_pips=0.5",
        "total_spread_and_slippage_round_trip_pips=2.5",
        "commission_account_per_lot_per_fill=4.5",
        "commission_account_per_lot_round_trip=9.0",
        "swap_long_pips_per_day=-2.445",
        "swap_short_pips_per_day=-0.105",
        "not broker/live financial authority",
    ] {
        assert!(logs.contains(expected), "missing {expected}: {logs}");
    }
    for superseded in [
        "session spread curve REFUSED",
        "no session spread curve configured",
        "Decision D:",
        "commission resolved to a ROUND TRIP charge",
        "resolved_commission=14.0",
        "models.prop_search_sensitivity_commission_per_lot is BELOW",
    ] {
        assert!(
            !logs.contains(superseded),
            "superseded cost diagnostic: {logs}"
        );
    }
    assert_eq!(
        logs.matches("resolved sealed canonical research cost assumptions")
            .count(),
        1
    );
}

#[test]
fn research_diagnostic_scope_preserves_contract_checks_and_broker_authority_refusal() {
    let settings = settings_with_spread(1.5, 0.5);
    let contract = research_contract();
    let mut wrong_symbol = settings.clone();
    wrong_symbol.system.symbol = "GBPUSD".to_owned();
    let (result, logs) = capture_diagnostics(|| {
        DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(&wrong_symbol, &contract)
    });
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not match settings symbol")
    );
    assert!(!logs.contains("resolved sealed canonical research cost assumptions"));
    DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(&settings, &contract)
        .expect("research costs do not need a broker execution capability");
    assert!(settings.risk.session_spread_pips().is_err());
    let error = DiscoveryConfig::try_from_settings(&settings)
        .expect_err("research constructor must not install broker financial authority");
    assert!(
        error
            .to_string()
            .contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1")
    );
}

#[test]
fn uniform_session_curve_matches_flat_two_fill_slippage_at_every_hour() {
    let settings = settings_with_spread(1.5, 0.5);
    let flat = DiscoveryConfig::from_settings(&settings);
    // Test the numerical adapter without granting broker financial authority:
    // RiskConfig::session_spread_pips correctly refuses it on this host.
    let mut session = flat.clone();
    session.session_spread_pips = Some(
        [1.5; 3]
            .map(|spread| screening_spread_and_slippage_pips(spread, settings.risk.slippage_pips)),
    );
    let template = PopulationTemplateResolver::new(&session, Some(1.0)).template(&Gene::default());
    let profile = template.session_spread_profile.expect("configured curve");

    // A full quoted width plus half a pip at EACH fill: 1.5 + 0.5 + 0.5.
    assert_eq!(flat.evaluation_spread_pips.to_bits(), 2.5_f64.to_bits());
    for hour in 0..24 {
        let timestamp = 1_735_689_600_000 + hour * 3_600_000;
        assert_eq!(
            profile.spread_pips_at(timestamp).to_bits(),
            2.5_f64.to_bits(),
            "uniform spread changed the round-trip charge at UTC hour {hour}",
        );
    }
}

#[test]
fn nonuniform_session_curve_charges_both_fills_in_every_bucket() {
    let curve = [2.0, 1.25, 1.75].map(|spread| screening_spread_and_slippage_pips(spread, 0.375));
    assert_eq!(
        curve.map(f64::to_bits),
        [2.75_f64, 2.0, 2.5].map(f64::to_bits),
    );
}

#[test]
fn binding_research_costs_removes_only_the_unsealed_cost_curve() {
    let contract = research_contract();
    let identity = contract.identity_sha256().expect("contract identity");
    let mut config = DiscoveryConfig {
        session_spread_pips: Some([0.25, 0.5, 0.75]),
        evaluation_spread_pips: 99.0,
        evaluation_commission_per_trade: 99.0,
        risk_per_trade_min: 0.01,
        risk_per_trade_max: 0.07,
        population: 512,
        generations: 256,
        ..DiscoveryConfig::default()
    };
    apply_research_contract_to_discovery_config(&mut config, &contract);

    assert_eq!(config.evaluation_spread_pips.to_bits(), 2.5_f64.to_bits());
    assert_eq!(
        config.evaluation_commission_per_trade.to_bits(),
        14.0_f64.to_bits()
    );
    assert!(
        config.session_spread_pips.is_none(),
        "the V3 contract seals scalar costs, not this ambient curve"
    );
    assert_eq!(config.risk_per_trade_min, 0.01);
    assert_eq!(config.risk_per_trade_max, 0.07);
    assert_eq!(config.population, 512);
    assert_eq!(config.generations, 256);
    assert_eq!(contract.identity_sha256().unwrap(), identity);
    #[cfg(feature = "gpu-cuda")]
    for mode in [
        DiscoveryMode::Risky,
        DiscoveryMode::PropFirm,
        DiscoveryMode::Strict,
    ] {
        config.mode = mode;
        config.risky_start_balance = 100.0;
        config.risky_target_balance = 50_000.0;
        config.risky_horizon_days = 180.0;
        let expected =
            matches!(mode, DiscoveryMode::Risky).then_some(crate::scoring::RiskyGrowthGoal {
                start_balance: 100.0,
                target_balance: 50_000.0,
                horizon_days: 180.0,
            });
        let explicit = crate::resident_population_auto_sizing_receipt_v2::evaluation_config_from_canonical_trendbar_contract_v2(
            &config, &contract,
        ).expect("shared explicit-cost preparation must retain the recorded goal");
        assert_eq!(explicit.growth_goal, expected);
        assert_eq!(config.evaluation_config(None).growth_goal, expected);
    }
}

#[test]
fn active_research_contract_cannot_be_overridden_by_an_ambient_cost_curve() {
    // The research authority is process-wide. Run its numerical consumer test
    // in a separate copy of this test executable, so ordinary parallel tests
    // cannot observe the temporary research scope or have their costs changed.
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "discovery::cost_consistency_tests::active_research_costs_worker",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .expect("isolated research-cost test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated cost consumer failed:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("1 passed"),
        "the isolated worker did not run: {stdout}"
    );
}

#[test]
#[ignore = "invoked in an isolated process by the active-research contract test"]
fn active_research_costs_worker() {
    assert!(crate::historical_evaluation_authority::active_research_contract_v1().is_none());
    let unbound = DiscoveryConfig::default();
    // Compare steady-state diagnostics, excluding one-time metadata initialization.
    let _ = prior_evaluation_config(&unbound, Some(1.0), None);
    let (prior, prior_logs) =
        capture_diagnostics(|| prior_evaluation_config(&unbound, Some(1.0), None));
    LEGACY_COST_RESOLUTIONS.with(|count| count.set(0));
    let (actual, actual_logs) = capture_diagnostics(|| unbound.evaluation_config(Some(1.0)));
    assert_eq!(LEGACY_COST_RESOLUTIONS.with(std::cell::Cell::get), 1);
    assert_same_evaluation_config(&actual, &prior);
    assert_eq!(
        actual_logs, prior_logs,
        "missing-authority legacy diagnostics must remain"
    );
    let contract = research_contract();
    let identity = contract.identity_sha256().expect("contract identity");
    let _scope = install_canonical_trendbar_research_execution_v3(&contract)
        .expect("isolated process owns the research scope");

    for curve in [None, Some([0.25, 0.5, 0.75]), Some([8.0, 9.0, 10.0])] {
        let mut config = DiscoveryConfig {
            evaluation_symbol: "EURUSD".to_owned(),
            evaluation_account_currency: "USD".to_owned(),
            session_spread_pips: curve,
            evaluation_spread_pips: 99.0,
            evaluation_commission_per_trade: 99.0,
            ..DiscoveryConfig::default()
        };
        // Check both the active-authority consumer and the normal pre-bound
        // path. Neither may charge ambient costs under the same sealed contract.
        for prebound in [false, true] {
            if prebound {
                apply_research_contract_to_discovery_config(&mut config, &contract);
            }
            let prior = prior_evaluation_config(&config, Some(1.0), Some(&contract));
            LEGACY_COST_RESOLUTIONS.with(|count| count.set(0));
            let (evaluation, logs) = capture_diagnostics(|| config.evaluation_config(Some(1.0)));
            assert_eq!(
                LEGACY_COST_RESOLUTIONS.with(std::cell::Cell::get),
                0,
                "valid sealed costs must not resolve ambient broker/FX costs"
            );
            assert!(!logs.contains("neoethos_search::cost_model"), "{logs}");
            assert_same_evaluation_config(&evaluation, &prior);
            let mut settings =
                PopulationTemplateResolver::new(&config, Some(1.0)).template(&Gene::default());
            settings.min_hold_bars = 0;
            settings.max_hold_bars = 1;
            settings.trailing_enabled = false;
            settings.kill_zones_enabled = false;
            settings.max_trades_per_day = 0;
            settings.risk_based_sizing = false;

            for hour in 0..24 {
                let start = 1_735_689_600_000 + hour * 3_600_000;
                let timestamps = [start, start + 60_000, start + 120_000];
                for side in [-1_i8, 1] {
                    let trades = simulate_trades_core(
                        &[1.0; 3],
                        &[1.0; 3],
                        &[1.0; 3],
                        &timestamps,
                        &[side, 0, 0],
                        &settings,
                    );
                    assert_eq!(trades.len(), 1, "one entry and one timed exit");
                    // Zero price movement, one standard lot: 15 spread + 5
                    // entry slippage + 5 exit slippage + 7 + 7 commission = 39.
                    assert!(
                        (trades[0].pnl + 39.0).abs() < 1e-8,
                        "curve={curve:?}, prebound={prebound}, hour={hour}, side={side}: charged {}, expected -39",
                        trades[0].pnl,
                    );
                }
            }
            assert!(evaluation.session_spread_pips.is_none());
            assert!(settings.session_spread_profile.is_none());
        }
    }

    for (symbol, account) in [("GBPUSD", "USD"), ("EURUSD", "GBP")] {
        let config = DiscoveryConfig {
            evaluation_symbol: symbol.to_owned(),
            evaluation_account_currency: account.to_owned(),
            session_spread_pips: Some([0.25, 0.5, 0.75]),
            ..DiscoveryConfig::default()
        };
        let (prior, legacy_logs) =
            capture_diagnostics(|| prior_evaluation_config(&config, Some(1.0), Some(&contract)));
        LEGACY_COST_RESOLUTIONS.with(|count| count.set(0));
        let (mismatch, logs) = capture_diagnostics(|| config.evaluation_config(Some(1.0)));
        assert_eq!(LEGACY_COST_RESOLUTIONS.with(std::cell::Cell::get), 1);
        assert_same_evaluation_config(&mismatch, &prior);
        for line in legacy_logs.lines() {
            assert!(
                logs.contains(line),
                "missing legacy diagnostic {line}: {logs}"
            );
        }
        assert!(logs.contains("does not match discovery config"), "{logs}");
        assert!(mismatch.pip_value.is_nan());
        assert!(mismatch.spread_pips.is_nan());
        assert!(mismatch.session_spread_pips.is_none());
    }
    assert_eq!(contract.identity_sha256().unwrap(), identity);
}
