use super::*;

fn signal_fixture() -> (FeatureFrame, Ohlcv) {
    let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    let rows = ohlcv.close.len();
    let data = ndarray::Array2::from_shape_fn((rows, 3), |(row, col)| match col {
        0 => {
            if row % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        }
        1 => -1.0,
        _ => 1.0,
    });
    let features = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
        ohlcv
            .timestamp
            .clone()
            .expect("canonical fixture timestamps"),
        vec![
            "signal".to_owned(),
            "smc_ob".to_owned(),
            "smc_fvg".to_owned(),
        ],
        data,
    )
    .expect("controlled signal/SMC fixture on canonical timestamps");
    (features, ohlcv)
}

fn gene() -> Gene {
    Gene {
        strategy_id: "locked-smc".to_owned(),
        indices: vec![0],
        weights: vec![1.0],
        long_threshold: 0.5,
        short_threshold: -0.5,
        use_ob: true,
        ..Gene::default()
    }
}

fn evaluation(threshold: f64) -> EvaluationConfig {
    EvaluationConfig {
        smc_gate_threshold: threshold,
        smc_weight_ob: 1.0,
        smc_weight_fvg: 1.0,
        ..EvaluationConfig::default()
    }
}

fn holdout_fixture() -> (
    CanonicalSearchInputReceiptV2,
    CanonicalSearchArtifactScopeV2,
    FeatureFrame,
    Ohlcv,
) {
    scoped_fixture(false)
}

fn scoped_fixture(
    calibration: bool,
) -> (
    CanonicalSearchInputReceiptV2,
    CanonicalSearchArtifactScopeV2,
    FeatureFrame,
    Ohlcv,
) {
    let (features, ohlcv) = signal_fixture();
    let anchor = features.provenance().bindings()[0].dataset_identity();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &features)
        .expect("receipt for the controlled signal fixture");
    let input = CanonicalSearchRunInputV2::new_for_test_values(receipt.clone(), &features, &ohlcv)
        .expect("exact fixture input");
    let split = CanonicalDiscoveryRunInputs::with_holdout(&input).expect("selection/holdout split");
    let holdout = if calibration {
        split.calibration().expect("separate selection calibration")
    } else {
        split.holdout().expect("with_holdout includes the suffix")
    };
    (
        receipt,
        holdout.scope().clone(),
        holdout.features().clone(),
        holdout.ohlcv().clone(),
    )
}

fn expected_ob_signals(rows: usize) -> Vec<i8> {
    (0..rows)
        .map(|row| if row % 2 == 0 { 0 } else { -1 })
        .collect()
}

#[test]
fn quote_holdout_signals_honor_the_locked_genes_smc_gate() {
    let (features, ohlcv) = signal_fixture();
    let signals = locked_holdout_signals(&[gene()], &features, &ohlcv, &evaluation(0.75))
        .expect("valid locked signal inputs");
    // OB is bearish on every row. It rejects each raw long, while agreeing
    // with each raw short. These expected directions do not use another
    // implementation of the signal generator as their oracle.
    let expected: Vec<i8> = (0..features.n_samples())
        .map(|row| if row % 2 == 0 { 0 } else { -1 })
        .collect();
    assert_eq!(signals, vec![expected]);
}

#[test]
fn quote_holdout_signals_use_the_final_gate_not_an_ungated_threshold() {
    let (features, ohlcv) = signal_fixture();
    let mut locked_gene = gene();
    locked_gene.use_fvg = true;
    // Exactly one of OB/FVG agrees with either direction. A 1.5 gate must
    // reject both; a final 0.75 gate admits both. The gene itself is unchanged.
    let strict = locked_holdout_signals(
        std::slice::from_ref(&locked_gene),
        &features,
        &ohlcv,
        &evaluation(1.5),
    )
    .expect("valid strict gate");
    let relaxed = locked_holdout_signals(&[locked_gene], &features, &ohlcv, &evaluation(0.75))
        .expect("valid final gate");
    assert_eq!(strict, vec![vec![0; features.n_samples()]]);
    let expected: Vec<i8> = (0..features.n_samples())
        .map(|row| if row % 2 == 0 { 1 } else { -1 })
        .collect();
    assert_eq!(relaxed, vec![expected]);
}

#[test]
fn locked_holdout_signals_reject_incomplete_or_misaligned_ohlcv_rows() {
    let (features, ohlcv) = signal_fixture();
    for mutation in 0..4 {
        let mut invalid = ohlcv.clone();
        match mutation {
            0 => {
                invalid.open.pop();
            }
            1 => invalid.timestamp = None,
            2 => {
                invalid.timestamp.as_mut().unwrap().pop();
            }
            _ => invalid.timestamp.as_mut().unwrap()[features.n_samples() / 2] += 1,
        }
        let error = locked_holdout_signals(&[gene()], &features, &invalid, &evaluation(0.75))
            .expect_err("one missing row or an interior timestamp mismatch must fail closed");
        assert!(
            error
                .to_string()
                .contains("same complete OHLCV rows and timestamps")
        );
    }
}

#[test]
fn locked_holdout_signals_reject_invalid_final_smc_gates() {
    let (features, ohlcv) = signal_fixture();
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
        let error = locked_holdout_signals(&[gene()], &features, &ohlcv, &evaluation(threshold))
            .expect_err("invalid final gate must not silently admit raw signals");
        assert!(
            error
                .to_string()
                .contains("finite non-negative final SMC gate")
        );
    }
}

#[test]
fn parallel_locked_signals_keep_the_locked_portfolio_order() {
    let (features, ohlcv) = signal_fixture();
    let portfolio: Vec<_> = (0..24)
        .map(|index| Gene {
            strategy_id: format!("ordered-{index}"),
            use_ob: index % 3 == 0,
            use_fvg: index % 3 == 1,
            ..gene()
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let signals = pool.install(|| {
        locked_holdout_signals(&portfolio, &features, &ohlcv, &evaluation(0.75)).unwrap()
    });
    let expected: Vec<Vec<i8>> = (0..portfolio.len())
        .map(|index| {
            (0..features.n_samples())
                .map(|row| match (index % 3, row % 2) {
                    (0, 0) | (1, 1) => 0,
                    (_, 0) => 1,
                    _ => -1,
                })
                .collect()
        })
        .collect();
    assert_eq!(signals, expected);
}

#[test]
fn prepared_holdout_projects_effective_features_and_pins_the_final_gate() {
    let (_, scope, features, ohlcv) = holdout_fixture();
    let portfolio = [Gene {
        indices: vec![1],
        ..gene()
    }];
    let config = DiscoveryConfig::default();
    let hash = "fnv64:0123456789abcdef";
    let final_gate = 0.001;
    assert_ne!(
        config.evaluation_config(None).smc_gate_threshold,
        final_gate
    );
    let effective_names = [
        "smc_fvg".to_owned(),
        "signal".to_owned(),
        "smc_ob".to_owned(),
    ];
    let prepared = PreparedLockedHoldoutResearch::new(
        &portfolio,
        &effective_names,
        &features,
        &ohlcv,
        &scope,
        hash,
        &config,
        final_gate,
    );
    assert!(prepared.is_ok());
    let prepared = prepared.unwrap();
    assert_eq!(prepared.evaluation.smc_gate_threshold, final_gate);
    assert_eq!(
        prepared.ordered_signals,
        vec![expected_ob_signals(features.n_samples())]
    );
}

#[test]
fn shared_holdout_consumers_reuse_one_smc_build_and_preserve_artifacts() {
    // The numerical research contract is process-wide. Keep it isolated from
    // other tests and from the desktop. No Search, broker request, or order runs.
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "discovery::holdout_signal_tests::shared_holdout_consumers_worker",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .expect("isolated holdout consumer test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated holdout consumer failed:\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("1 passed"),
        "isolated worker did not run: {stdout}"
    );
    print!("{stdout}");
}

#[test]
#[ignore = "invoked in a separate process by the shared-holdout consumer test"]
fn shared_holdout_consumers_worker() {
    use crate::canonical_trendbar_research::{
        CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchExecutionContractV3,
        install_canonical_trendbar_research_execution_v3,
    };
    use crate::genetic::search_engine::SMC_GATE_BUILD_CALLS;

    let (receipt, scope, features, ohlcv) = holdout_fixture();
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.shared-holdout-signals.v1",
            assumption_source_sha256: &"a".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 0.0,
            slippage_pips_per_fill_assumption: 0.0,
            commission_account_per_lot_per_fill_assumption: 0.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .expect("explicit numerical-fixture cost assumptions, not observed broker fills");
    let _scope = install_canonical_trendbar_research_execution_v3(&contract)
        .expect("isolated research execution scope");
    let mut config = DiscoveryConfig {
        initial_balance: 10_000.0,
        kill_zones_enabled: false,
        ..DiscoveryConfig::default()
    };
    apply_research_contract_to_discovery_config(&mut config, &contract);
    let hash = "fnv64:0123456789abcdef";
    let portfolio: Vec<_> = (0..6)
        .map(|index| Gene {
            strategy_id: format!("shared-{index}"),
            sl_pips: 0.5 + index as f64 * 0.1,
            tp_pips: 0.5 + index as f64 * 0.25,
            ..gene()
        })
        .collect();
    let rules = PropFirmRiskRules::default();
    let final_gate = 0.001;
    // Count the existing thread-local instrumentation on one worker. A
    // separate three-worker test above checks indexed output ordering.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (forward, prop) = pool.install(|| {
        let before = SMC_GATE_BUILD_CALLS.with(|counter| counter.get());
        let prepared = PreparedLockedHoldoutResearch::new(
            &portfolio,
            &features.names,
            &features,
            &ohlcv,
            &scope,
            hash,
            &config,
            final_gate,
        )
        .expect("shared preparation");
        assert_eq!(
            prepared.ordered_signals,
            vec![expected_ob_signals(features.n_samples()); portfolio.len()]
        );
        let (forward, prop) = rayon::join(
            || prepared.forward_test_artifacts(),
            || prepared.prop_firm_artifacts(rules),
        );
        assert_eq!(
            SMC_GATE_BUILD_CALLS.with(|counter| counter.get()) - before,
            1,
            "both consumers must reuse the single prepared SMC cache"
        );
        (
            forward.expect("forward-test artifacts"),
            prop.expect("prop-firm artifacts"),
        )
    });
    assert_eq!(forward.len(), portfolio.len());
    assert_eq!(prop.len(), portfolio.len());

    // Reference the pre-existing numerical consumers directly, with written
    // expected signals, so this verifies wiring/ordering rather than claiming
    // these small OHLC results are independent proof of financial accuracy.
    let resolver = GeneEvalSettingsResolver::for_slice(
        &config,
        portfolio.iter(),
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
    )
    .unwrap();
    let signals = expected_ob_signals(features.n_samples());
    // score=-1, short threshold=-0.5 and threshold gap=1 give confidence=0.5;
    // the SMC-blocked long rows have no confidence-bearing entry.
    let confidences: Vec<f64> = signals
        .iter()
        .map(|&s| if s == 0 { 0.0 } else { 0.5 })
        .collect();
    let (months, days) = month_day_indices(&features.timestamps);
    for ((gene, actual_forward), actual_prop) in portfolio.iter().zip(&forward).zip(&prop) {
        let settings = resolver.settings_for_gene(gene);
        let expected_forward = compute_forward_test_summary(ForwardTestInput {
            close: &ohlcv.close,
            high: &ohlcv.high,
            low: &ohlcv.low,
            signals: &signals,
            confidences: &confidences,
            months: &months,
            days: &days,
            timestamps: &features.timestamps,
            settings: &settings,
        })
        .unwrap();
        let trades = crate::eval::simulate_trades_with_confidence_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            &features.timestamps,
            &signals,
            &confidences,
            &settings,
        )
        .expect("risk-sized holdout ledger");
        assert!(
            !trades.is_empty(),
            "this must exercise actual numerical trades, not zero-trade artifacts"
        );
        let expected_prop = compute_prop_firm_risk_summary(PropFirmRiskInput {
            trades: &trades,
            initial_balance: config.initial_balance,
            rules,
        });
        let expected_forward =
            ForwardTestValidationArtifactFile::new(scope.clone(), hash, gene, expected_forward)
                .unwrap();
        let expected_prop =
            PropFirmRiskValidationArtifactFile::new(scope.clone(), hash, gene, expected_prop)
                .unwrap();
        assert_eq!(
            serde_json::to_value(actual_forward).unwrap(),
            serde_json::to_value(expected_forward).unwrap()
        );
        assert_eq!(
            serde_json::to_value(actual_prop).unwrap(),
            serde_json::to_value(expected_prop).unwrap()
        );
        assert_eq!(actual_forward.scope(), &scope);
        assert_eq!(actual_prop.scope(), &scope);
        assert!(actual_prop.summary().trades_observed > 0);
    }
    assert!(
        neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
            .is_err(),
        "a research test must not open the legacy financial gate"
    );
    // The broad research pool uses the same actual producer/CPU forward
    // calculation, but it retains scalar summaries and only on calibration.
    let (_, calibration_scope, calibration_features, calibration_ohlcv) = scoped_fixture(true);
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let cohort = parallel_pool
        .install(|| {
            evaluate_selection_calibration_cohort(
                &portfolio,
                &portfolio,
                &calibration_features.names,
                &calibration_features,
                &calibration_ohlcv,
                &calibration_scope,
                hash,
                &config,
                final_gate,
                None,
            )
        })
        .expect("all six real calibration replays, no portfolio capacity argument");
    assert_eq!(cohort.trials.len(), portfolio.len());
    let reference = PreparedLockedHoldoutResearch::new(
        &portfolio,
        &calibration_features.names,
        &calibration_features,
        &calibration_ohlcv,
        &calibration_scope,
        hash,
        &config,
        final_gate,
    )
    .unwrap()
    .forward_test_artifacts()
    .unwrap();
    for (index, (trial, expected)) in cohort.trials.iter().zip(&reference).enumerate() {
        assert_eq!(trial.candidate_archive_index, index);
        trial
            .strategy_identity
            .validate_against(&portfolio[index])
            .unwrap();
        if expected
            .summary()
            .metrics
            .to_metric_array()
            .iter()
            .all(|value| value.is_finite())
        {
            assert_eq!(trial.summary.as_ref(), Some(expected.summary()));
        } else {
            assert!(trial.summary.is_none());
            assert!(
                trial
                    .rejection_reason
                    .as_ref()
                    .unwrap()
                    .contains("canonical slots")
            );
        }
        assert!(
            expected.summary().metrics.trade_count > 0,
            "actual CPU trades, not placeholder rows"
        );
        assert_eq!(
            trial.profitable_for_selection,
            crate::live_portfolio::LiveSizingEvidenceV1::validate_calibration_metrics(
                &portfolio[index].strategy_id,
                &expected.summary().metrics,
            )
            .is_ok()
        );
    }
    assert!(
        evaluate_selection_calibration_cohort(
            &portfolio,
            &portfolio,
            &features.names,
            &features,
            &ohlcv,
            &scope,
            hash,
            &config,
            final_gate,
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("never the final holdout")
    );
    let positive_genes = cohort
        .trials
        .iter()
        .filter(|trial| trial.profitable_for_selection)
        .map(|trial| portfolio[trial.candidate_archive_index].clone())
        .collect::<Vec<_>>();
    let selected =
        selected_calibration_artifacts(&positive_genes, &calibration_scope, hash, &cohort).unwrap();
    assert_eq!(selected.len(), positive_genes.len());
    for (gene, artifact) in positive_genes.iter().zip(&selected) {
        artifact
            .validate_against(&calibration_scope, hash, gene)
            .unwrap();
    }
    let encoded = serde_json::to_string(&cohort).unwrap();
    assert!(
        !encoded.contains("feature_plan_canonical_bytes"),
        "pool summaries must not duplicate receipts"
    );
    // Explicit transport/reconstruction cases, separate from the real CPU
    // replay above: finite winners, finite losers and invalid metrics must
    // remain distinguishable in a typed profile roundtrip.
    let positive_summary = crate::validation::ForwardTestSummary {
        bars: calibration_features.n_samples(),
        metrics: BacktestMetrics::from_metric_array([
            20.0, 1.0, 10_020.0, 0.01, 0.6, 1.5, 2.0, 0.5, 10.0, 0.8, 0.005,
        ]),
        span_days: ranking_window_span_days(&calibration_features.timestamps),
    };
    let mut transport = cohort.clone();
    transport.trials = vec![selection_calibration_trial(
        0,
        ValidationStrategyIdentityV2::from_gene(&portfolio[0]).unwrap(),
        positive_summary.clone(),
    )];
    let reconstructed =
        selected_calibration_artifacts(&portfolio[..1], &calibration_scope, hash, &transport)
            .unwrap();
    assert_eq!(reconstructed.len(), 1);
    assert_eq!(reconstructed[0].summary(), &positive_summary);
    reconstructed[0]
        .validate_against(&calibration_scope, hash, &portfolio[0])
        .unwrap();
    let mut invalid_summary = positive_summary.clone();
    invalid_summary.metrics.sharpe = f64::NEG_INFINITY;
    let invalid_trial = selection_calibration_trial(
        0,
        ValidationStrategyIdentityV2::from_gene(&portfolio[0]).unwrap(),
        invalid_summary,
    );
    assert!(invalid_trial.summary.is_none());
    assert!(!invalid_trial.profitable_for_selection);
    assert!(
        invalid_trial
            .rejection_reason
            .as_ref()
            .unwrap()
            .contains("[1]")
    );
    transport.trials = vec![invalid_trial];
    let mut profile = crate::funnel_profile::FunnelProfile::new("EURUSD", "M1");
    profile.selection_calibration_cohort = Some(transport);
    let bytes = serde_json::to_vec(&profile).unwrap();
    let restored: crate::funnel_profile::FunnelProfile = serde_json::from_slice(&bytes).unwrap();
    let restored = restored.selection_calibration_cohort.unwrap();
    assert!(restored.trials[0].summary.is_none());
    assert!(
        selected_calibration_artifacts(&portfolio[..1], &calibration_scope, hash, &restored)
            .is_err()
    );
    let mut losing_summary = positive_summary;
    losing_summary.metrics.net_profit = -20.0;
    losing_summary.metrics.expectancy = -2.0;
    let losing_trial = selection_calibration_trial(
        0,
        ValidationStrategyIdentityV2::from_gene(&portfolio[0]).unwrap(),
        losing_summary.clone(),
    );
    assert_eq!(losing_trial.summary.as_ref(), Some(&losing_summary));
    assert!(!losing_trial.profitable_for_selection);
    let mut substituted = portfolio.clone();
    substituted[0].weights[0] += 0.25;
    assert!(
        evaluate_selection_calibration_cohort(
            &substituted,
            &portfolio,
            &calibration_features.names,
            &calibration_features,
            &calibration_ohlcv,
            &calibration_scope,
            hash,
            &config,
            final_gate,
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("exact_gene_hash")
    );
    assert!(selected_calibration_artifacts(&[], &scope, hash, &cohort).is_err());
    let before_cancel = SMC_GATE_BUILD_CALLS.with(|counter| counter.get());
    crate::genetic::search_engine::set_search_cancel(Some(std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(true),
    )));
    let cancelled = evaluate_selection_calibration_cohort(
        &portfolio,
        &portfolio,
        &calibration_features.names,
        &calibration_features,
        &calibration_ohlcv,
        &calibration_scope,
        hash,
        &config,
        final_gate,
        None,
    );
    crate::genetic::search_engine::set_search_cancel(None);
    assert!(cancelled.unwrap_err().to_string().contains("cancel"));
    assert_eq!(
        SMC_GATE_BUILD_CALLS.with(|counter| counter.get()),
        before_cancel
    );
    println!(
        "selection calibration pool: {} exact CPU summaries; {} profitable; final window explicitly refused",
        cohort.trials.len(),
        positive_genes.len()
    );
    println!(
        "shared holdout: {} rows, {} strategies, 1 SMC build, {} forward and {} risk artifacts; first strategy numerical trades: {}",
        features.n_samples(),
        portfolio.len(),
        forward.len(),
        prop.len(),
        prop[0].summary().trades_observed
    );
}
