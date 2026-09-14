// Private manual transport fixture, included only inside live_portfolio::tests.
// The evaluation policy and selection summaries are explicitly synthetic.
// This exercises the actual V6 writer, not Discovery or financial performance.

#[test]
#[ignore = "manual policy2 fixture producer only; writes a new temporary directory, never repository artifacts"]
fn export_candidate_policy2_fixture_to_temporary_directory() {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    assert_eq!(features.names.len(), 2);
    let authority = report_fixture_scalar_authority(sample_discovery_authority(&features));
    let mut result = sample_discovery_result_for_authority(
        vec![Gene {
            strategy_id: "synthetic-candidate-install-fixture-not-traded".to_owned(),
            indices: vec![0],
            weights: vec![1.0],
            sl_pips: 10.0,
            tp_pips: 20.0,
            ..Default::default()
        }],
        authority,
    );
    let mut policy = sample_sealed_evaluation_policy();
    let mut evaluation = policy.sealed_evaluation_config().unwrap();
    // The general roundtrip sample deliberately retains a nonzero rate. This
    // executable research fixture instead seals its applicable USD-to-USD fee.
    assert_eq!(evaluation.symbol, "EURUSD");
    evaluation.pnl_conversion_fee_rate =
        neoethos_core::research_conversion_fee::effective_conversion_fee_rate_v1(
            evaluation.pnl_conversion_fee_rate,
            "USD",
            &evaluation.account_currency,
        )
        .unwrap();
    let smc_gate_disabled = policy.sealed_smc_gate_disabled().unwrap();
    let adaptive_stops = policy.sealed_adaptive_stops_policy().unwrap().clone();
    policy.sealed_evaluation_v1 = Some(
        SealedEvaluationPolicyV1::from_evaluation(&evaluation, smc_gate_disabled, &adaptive_stops)
            .unwrap(),
    );
    policy.identity_hash = policy.computed_identity_hash().unwrap();
    policy.validate().unwrap();
    assert_eq!(
        policy
            .sealed_evaluation_config()
            .unwrap()
            .pnl_conversion_fee_rate,
        0.0,
    );
    result.effective_smc_gate_threshold = policy
        .sealed_evaluation_config()
        .unwrap()
        .smc_gate_threshold;
    let mut funnel = crate::funnel_profile::FunnelProfile::new("EURUSD", "M1");
    funnel.attach_live_trading_policy_v1(policy).unwrap();
    result.funnel_profile = Some(funnel);

    let directory = std::env::temp_dir().join(format!(
        "neoethos-candidate-policy2-fixture-{}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("create a new owned fixture output, never overwrite");
    let path = directory.join("synthetic-policy2.live_portfolio.json");
    save_live_portfolio_json(&path, &result).unwrap();
    let reopened = load_live_portfolio_json(&path).unwrap();
    assert_eq!(reopened.schema_version, 6);
    assert_eq!(reopened.live_trading_policy.schema_version, 2);
    assert_eq!(
        reopened
            .live_trading_policy
            .sealed_evaluation_config()
            .unwrap()
            .pnl_conversion_fee_rate,
        0.0,
    );
    assert_eq!(
        reopened
            .live_trading_policy
            .sealed_smc_gate_disabled()
            .unwrap(),
        smc_gate_disabled,
    );
    assert_eq!(
        reopened
            .live_trading_policy
            .sealed_adaptive_stops_policy()
            .unwrap(),
        &adaptive_stops,
    );
    assert_eq!(reopened.search_scope.evaluated_window().row_end(), 80);
    assert_eq!(
        reopened.final_holdout_scope.evaluated_window().row_start(),
        90
    );
    assert!(std::fs::metadata(&path).unwrap().len() < 256 * 1024);
    println!("CANDIDATE_POLICY2_FIXTURE={}", path.display());
    println!(
        "Fixture only: manually instantiated policy/selection evidence, not a Discovery run or broker execution."
    );
}
