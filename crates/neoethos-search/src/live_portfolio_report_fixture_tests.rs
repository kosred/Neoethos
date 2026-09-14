// Included only inside live_portfolio::tests: reuse its private typed V6
// constructors without exporting synthetic evidence to application callers.

include!("live_portfolio_candidate_fixture_tests.rs");

fn report_golden_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test_fixtures/strategy_report_v6")
}

fn report_fixture_scalar_authority(
    (receipt, selection, calibration, holdout): SampleDiscoveryAuthority,
) -> SampleDiscoveryAuthority {
    // Transport-only synthetic metadata, never a claim that scalar kernels ran.
    // Leave all real/runtime receipt constructors and lane validation untouched.
    assert!(matches!(
        receipt.validate().unwrap().scope(),
        CanonicalDatasetScope::External { source_namespace }
            if source_namespace == "embedded-ctrader-fixture-unverified"
    ));
    for scope in [&selection, &calibration, &holdout] {
        scope.validate_against_receipt(&receipt).unwrap();
    }
    let mut wire: serde_json::Value =
        serde_json::from_slice(&receipt.to_json_bytes().unwrap()).unwrap();
    wire["feature_execution"]["compute_policy"] = "cpu_only".into();
    wire["feature_execution"]["selected_lane"] = "cpu_scalar".into();
    let pinned = crate::data_selection::CanonicalSearchInputReceiptV2::from_json_bytes(
        &serde_json::to_vec(&wire).unwrap(),
    )
    .unwrap();
    let rebind = |scope: &CanonicalSearchArtifactScopeV2| {
        CanonicalSearchArtifactScopeV2::new(pinned.clone(), scope.evaluated_window().clone())
            .unwrap()
    };
    let scopes = (rebind(&selection), rebind(&calibration), rebind(&holdout));
    (pinned, scopes.0, scopes.1, scopes.2)
}

fn report_fixture_research() -> crate::CanonicalTrendbarResearchDiscoveryResultV3 {
    use crate::canonical_trendbar_research::{
        CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchDiscoveryResultV3,
        CanonicalTrendbarResearchExecutionContractV3,
    };

    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let timestamps = &features.timestamps;
    let genes = vec![
        Gene {
            strategy_id: "report-selected-a".to_owned(),
            indices: vec![0],
            weights: vec![1.0],
            sl_pips: 10.0,
            tp_pips: 20.0,
            ..Gene::default()
        },
        Gene {
            strategy_id: "report-selected-b".to_owned(),
            indices: vec![0],
            weights: vec![0.5],
            sl_pips: 20.0,
            tp_pips: 40.0,
            ..Gene::default()
        },
    ];
    let authority = report_fixture_scalar_authority(sample_discovery_authority(&features));
    let mut result = sample_discovery_result_for_authority(genes, authority);
    let funnel = result.funnel_profile.as_mut().unwrap();
    // The profile contains wall-clock fields even though the input is static.
    // Do not let those fields silently change the complete research identity.
    funnel.started_at = "2025-01-13T09:00:00Z".to_owned();
    funnel.finished_at = "2025-01-13T09:01:00Z".to_owned();
    funnel.mode = "risky".to_owned();
    funnel.outcome = "synthetic-report-fixture-not-trading-evidence".to_owned();

    // These PnLs are deliberate adversarial display inputs, NOT executed fills.
    // Every row lies inside the genuine fixture's typed 0..80 IS window.
    let logged = |strategy_id: &str, pnls: &[f64]| crate::discovery::LoggedStrategyTrades {
        strategy_id: strategy_id.to_owned(),
        opportunistic: false,
        trades: pnls
            .iter()
            .enumerate()
            .map(|(index, &pnl)| crate::quality::Trade {
                entry_time: timestamps[2 * index + 1],
                exit_time: Some(timestamps[2 * index + 2]),
                pnl,
                pnl_pct: Some(pnl / 1_000.0),
                ..Default::default()
            })
            .collect(),
    };
    result.logged_trades = vec![
        logged("report-selected-a", &[100.0, -50.0]),
        logged(
            "report-not-selected",
            &[9_000.0, -1.0, 9_000.0, -1.0, 9_000.0, -1.0],
        ),
        logged("report-selected-b", &[100.0, -50.0, 25.0, 25.0]),
    ];
    let analyzer = crate::quality::StrategyQualityAnalyzer {
        min_trades_per_month: Some(1),
        ..Default::default()
    };
    assert_eq!(
        crate::quality::current_quality_runtime_overrides()
            .trading_days_per_month
            .to_bits(),
        21.0_f64.to_bits(),
        "golden fixture generation requires its pinned 21-trading-day quality convention"
    );
    // Keep the actual quality producer and its deterministic seeded bootstrap;
    // reverse row order so the consumer cannot accidentally join by position.
    result.quality_metrics = result
        .logged_trades
        .iter()
        .rev()
        .map(|row| analyzer.analyze_strategy(&row.strategy_id, &row.trades, 1_000.0))
        .collect();
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        result.search_input_receipt.clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "synthetic-report-fixture-not-broker-financial-evidence",
            assumption_source_sha256: &"a".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.5,
            slippage_pips_per_fill_assumption: 0.0,
            commission_account_per_lot_per_fill_assumption: 0.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .unwrap();
    CanonicalTrendbarResearchDiscoveryResultV3::new(contract, result).unwrap()
}

fn write_report_fixture_pair(directory: &std::path::Path) -> Vec<std::path::PathBuf> {
    let research = report_fixture_research();
    research.validate().unwrap();
    let base = format!("{}.research", research.evidence_identity_sha256());
    let research_path = directory.join(format!("{base}.json"));
    let portfolio_path = directory.join(format!("{base}.live_portfolio.json"));
    crate::artifact_io::write_json_atomic(&research_path, &research).unwrap();
    save_live_portfolio_json(&portfolio_path, research.discovery_result()).unwrap();
    let restored = load_live_portfolio_json(&portfolio_path).unwrap();
    assert_eq!(restored.schema_version, 6);
    assert_eq!(restored.genes.len(), 2);
    assert_eq!(restored.genes[1].strategy_id, "report-selected-b");
    assert_eq!(restored.search_scope.evaluated_window().row_end(), 80);
    assert_eq!(
        restored.sizing_evidence[0]
            .forward_test
            .scope()
            .evaluated_window()
            .row_end(),
        90
    );
    assert_eq!(
        restored.final_holdout_scope.evaluated_window().row_start(),
        90
    );
    for path in [&research_path, &portfolio_path] {
        assert!(
            std::fs::metadata(path).unwrap().len() < 256 * 1024,
            "report fixture must remain a small typed transport regression"
        );
    }
    vec![research_path, portfolio_path]
}

#[test]
fn report_fixture_receipt_and_scopes_are_independent_of_host_cpu_lane() {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let original = sample_discovery_authority(&features);
    let expected = report_fixture_scalar_authority(original.clone());
    for lane in [
        "cpu_scalar",
        "cpu_avx2_fma",
        "cpu_avx512f_dq_vl_bw_avx2_fma",
    ] {
        let mut wire: serde_json::Value =
            serde_json::from_slice(&original.0.to_json_bytes().unwrap()).unwrap();
        wire["feature_execution"]["compute_policy"] = "auto".into();
        wire["feature_execution"]["selected_lane"] = lane.into();
        let receipt = crate::data_selection::CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&wire).unwrap(),
        )
        .unwrap();
        let rebind = |scope: &CanonicalSearchArtifactScopeV2| {
            CanonicalSearchArtifactScopeV2::new(receipt.clone(), scope.evaluated_window().clone())
                .unwrap()
        };
        let scopes = (
            rebind(&original.1),
            rebind(&original.2),
            rebind(&original.3),
        );
        assert_eq!(
            report_fixture_scalar_authority((receipt, scopes.0, scopes.1, scopes.2)),
            expected,
            "complete receipt and all scope identities must be stable for {lane}"
        );
    }
}

#[test]
fn report_golden_pair_matches_the_real_search_writers_byte_for_byte() {
    let temporary = tempfile::tempdir().unwrap();
    let actual = write_report_fixture_pair(temporary.path());
    let golden = report_golden_directory();
    for path in actual {
        let recorded = golden.join(path.file_name().unwrap());
        let bytes = std::fs::read(&recorded).unwrap_or_else(|error| {
            panic!("missing genuine report golden {}: {error}; explicitly run regenerate_report_golden_pair_to_temporary_directory, review and install its output", recorded.display())
        });
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "the actual Search writer changed {}; review and explicitly regenerate, never silently bless new bytes",
            recorded.display()
        );
    }
}

#[test]
#[ignore = "manual fixture regeneration only; writes a new temporary directory, never checked-in artifacts"]
fn regenerate_report_golden_pair_to_temporary_directory() {
    let directory = std::env::temp_dir().join(format!(
        "neoethos-strategy-report-v6-golden-{}",
        std::process::id()
    ));
    // Fail if this exact output already exists; do not overwrite older evidence.
    std::fs::create_dir(&directory).unwrap();
    for path in write_report_fixture_pair(&directory) {
        println!("REPORT_GOLDEN_OUTPUT={}", path.display());
    }
}
