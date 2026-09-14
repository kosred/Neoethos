use super::*;

const JAN: i64 = 1_704_067_200_000;
const FEB: i64 = 1_706_745_600_000;
const NEXT_JAN: i64 = 1_735_689_600_000;

fn gene(id: &str, stop: f64) -> Gene {
    Gene {
        strategy_id: id.to_owned(),
        sl_pips: stop,
        ..Gene::default()
    }
}

fn trade(exit: i64, pnl: f64) -> ReportTrade {
    ReportTrade {
        entry_time: exit - 60_000,
        exit_time: Some(exit),
        pnl,
        pnl_pct: Some(pnl / 1_000.0),
    }
}

fn logged(id: &str, pnls: &[f64]) -> ReportTrades {
    ReportTrades {
        strategy_id: id.to_owned(),
        trades: pnls
            .iter()
            .enumerate()
            .map(|(i, pnl)| trade(JAN + (i as i64 + 1) * 60_000, *pnl))
            .collect(),
    }
}

fn quality(id: &str, pnls: &[f64], sharpe: f64) -> ReportQuality {
    let net_profit = pnls.iter().sum();
    ReportQuality {
        strategy_id: id.to_owned(),
        total_trades: pnls.len(),
        initial_capital: 1_000.0,
        net_profit,
        total_return_pct: net_profit / 1_000.0,
        win_rate: Some(0.5),
        profit_factor: Some(1.2),
        sharpe_ratio: Some(sharpe),
    }
}

#[test]
fn selected_strategy_keeps_its_own_trade_and_quality_rows_not_nonselected_outlier() {
    let selected = vec![gene("selected-a", 10.0), gene("selected-b", 20.0)];
    let logs = vec![
        logged("selected-b", &[30.0, -10.0]),
        logged("not-selected", &[9_000.0; 4]),
        logged("selected-a", &[50.0]),
    ];
    let quality = vec![
        quality("not-selected", &[9_000.0; 4], 99.0),
        quality("selected-a", &[50.0], 1.0),
        quality("selected-b", &[30.0, -10.0], 2.0),
    ];
    let bound = bind_diagnostics(&selected, &logs, &quality).unwrap();
    assert_eq!(bound.gene.strategy_id, "selected-b");
    assert_eq!(bound.trades.len(), 2);
    assert_eq!(bound.quality.sharpe_ratio, Some(2.0));
    let curve = diagnostic_curve(bound.trades).unwrap();
    assert_eq!(
        curve.equity, 1_020.0,
        "not selected-a + selected-b and not the outlier"
    );
}

#[test]
fn missing_or_ambiguous_selected_evidence_is_unavailable() {
    let selected = vec![gene("a", 10.0)];
    assert!(bind_diagnostics(&selected, &[], &[quality("a", &[10.0], 1.0)]).is_err());
    assert!(bind_diagnostics(&selected, &[logged("a", &[10.0])], &[]).is_err());
    assert!(
        bind_diagnostics(
            &selected,
            &[logged("a", &[10.0]), logged("a", &[20.0])],
            &[quality("a", &[10.0], 1.0)]
        )
        .is_err()
    );
    assert!(
        bind_diagnostics(
            &selected,
            &[logged("a", &[10.0])],
            &[quality("a", &[10.0], 1.0), quality("a", &[10.0], 2.0)]
        )
        .is_err()
    );
    assert!(
        bind_diagnostics(
            &[gene("a", 10.0), gene("a", 20.0)],
            &[logged("a", &[10.0])],
            &[quality("a", &[10.0], 1.0)]
        )
        .is_err()
    );
    assert!(
        bind_diagnostics(
            &[gene("", 10.0)],
            &[logged("", &[10.0])],
            &[quality("", &[10.0], 1.0)]
        )
        .is_err()
    );
}

#[test]
fn full_selected_gene_identity_rejects_same_display_id_with_different_rule() {
    let selected = vec![gene("same-display-id", 10.0)];
    validate_selected_genes(&selected, &selected).unwrap();
    assert!(validate_selected_genes(&selected, &[gene("same-display-id", 20.0)]).is_err());
    assert!(
        validate_selected_genes(&selected, &[selected[0].clone(), selected[0].clone()]).is_err()
    );
    assert!(validate_selected_genes(&selected, &[]).is_err());
}

#[test]
fn initial_capital_returns_are_added_and_month_closes_before_next_trade() {
    // Initial 1000: January +100-50 =>1050; February +200 =>1250.
    // February return is 200/1050=19.0476%, not another January gain.
    // Next January -50 =>1200, a new-year return of -50/1250=-4%.
    let trades = vec![
        trade(JAN + 60_000, 100.0),
        trade(JAN + 120_000, -50.0),
        trade(FEB, 200.0),
        trade(NEXT_JAN, -50.0),
    ];
    let curve = diagnostic_curve(&trades).unwrap();
    assert_eq!(
        curve
            .monthly
            .iter()
            .map(|m| (&*m.month, m.balance, m.return_pct, m.trades))
            .collect::<Vec<_>>(),
        vec![
            ("2024-01", 1050.0, 5.0, 2),
            ("2024-02", 1250.0, 19.05, 1),
            ("2025-01", 1200.0, -4.0, 1)
        ]
    );
    assert_eq!(
        curve
            .yearly
            .iter()
            .map(|y| (&*y.month, y.balance, y.return_pct, y.trades))
            .collect::<Vec<_>>(),
        vec![("2024", 1250.0, 25.0, 3), ("2025", 1200.0, -4.0, 1)]
    );
    assert_eq!(curve.equity, 1200.0);
    assert!((curve.max_dd - 50.0 / 1100.0).abs() < 1e-12);
    let two_tens =
        diagnostic_curve(&[trade(JAN + 60_000, 100.0), trade(JAN + 120_000, 100.0)]).unwrap();
    assert_eq!(
        two_tens.equity, 1200.0,
        "two initial-capital 10% amounts are not 1210"
    );
    assert_eq!(two_tens.monthly[0].return_pct, 20.0);
}

#[test]
fn monthly_realized_pnl_is_ordered_by_exit_not_entry_or_input_order() {
    let mut late_exit = trade(FEB + 60_000, -100.0);
    late_exit.entry_time = JAN;
    let earlier_exit = trade(JAN + 3 * 86_400_000, 200.0);
    let curve = diagnostic_curve(&[late_exit, earlier_exit]).unwrap();
    assert_eq!(curve.monthly[0].balance, 1200.0);
    assert_eq!(curve.monthly[1].balance, 1100.0);
    assert_eq!(curve.monthly[1].return_pct, -8.33);
    assert!((curve.max_dd - 100.0 / 1200.0).abs() < 1e-12);
}

#[test]
fn malformed_missing_or_crosswired_numeric_fields_are_not_filled_from_other_rows() {
    let incomplete = r#"[{"strategy_id":"a","total_trades":1,"initial_capital":1000,"net_profit":10},{"strategy_id":"b","total_trades":1,"initial_capital":1000,"net_profit":20,"total_return_pct":0.02}]"#;
    assert!(serde_json::from_str::<Vec<ReportQuality>>(incomplete).is_err());
    let q = quality("a", &[10.0], 1.0);
    let mut t = trade(JAN + 60_000, 10.0);
    t.pnl_pct = Some(1.0); // percent-point units instead of fraction
    assert!(validate_trade_units(&[t], &q).is_err());
    let mut t = trade(JAN + 60_000, 10.0);
    t.pnl_pct = None;
    assert!(validate_trade_units(&[t], &q).is_err());
    let mut t = trade(JAN + 60_000, 10.0);
    t.exit_time = None;
    assert!(validate_trade_units(&[t], &q).is_err());
    assert!(
        validate_trade_units(&[trade(JAN + 60_000, 10.0)], &quality("a", &[20.0], 1.0)).is_err()
    );
    assert!(diagnostic_curve(&[trade(JAN + 60_000, f64::INFINITY)]).is_err());
}

#[test]
fn current_research_inventory_is_scoped_and_unvalidated_artifacts_are_unavailable() {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let root =
        std::env::temp_dir().join(format!("neoethos-report-{}-{sequence}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let research = root.join("discovery").join("research");
    std::fs::create_dir_all(&research).unwrap();
    let base = format!("{}.research", "a".repeat(64));
    std::fs::write(research.join(format!("{base}.json")), b"{}").unwrap();
    std::fs::create_dir(root.join("unrelated")).unwrap();
    std::fs::write(
        root.join("unrelated").join("EURUSD_M5.json.trades.json"),
        b"[]",
    )
    .unwrap();
    let (entries, unavailable) = inventory(&root);
    assert!(unavailable.is_empty());
    assert_eq!(entries, vec![(RESEARCH_DIR.to_owned(), base.clone())]);
    assert!(
        build(&root, RESEARCH_DIR, &base, true).is_err(),
        "a filename is not a validated selected portfolio"
    );
    assert!(report_directory(&root, "auto_loop_x/../../outside").is_err());
    assert!(!plain_segment("C:outside"));
    assert!(!plain_segment("../outside"));
    std::fs::remove_dir_all(&root).unwrap();
}

/// Copies only the pair emitted by Search's private, explicitly invoked fixture
/// producer. No hand-written positive artifact or test authority is introduced
/// into the App. All adversarial edits below affect this test-owned copy only.
struct SavedResearchFixture {
    root: std::path::PathBuf,
    directory: std::path::PathBuf,
    base: String,
}

impl SavedResearchFixture {
    fn copy_genuine_writer_pair() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../neoethos-search/test_fixtures/strategy_report_v6");
        let mut research_paths = std::fs::read_dir(&golden)
            .expect(
                "install the reviewed genuine Search writer fixture pair before running App tests",
            )
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".research.json"))
            })
            .collect::<Vec<_>>();
        research_paths.sort();
        assert_eq!(
            research_paths.len(),
            1,
            "one unambiguous writer fixture pair"
        );
        let research_path = &research_paths[0];
        let base = research_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix(".json")
            .unwrap()
            .to_owned();
        let root = std::env::temp_dir().join(format!(
            "neoethos-saved-report-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let fixture = Self {
            directory: root.join(RESEARCH_DIR),
            root,
            base,
        };
        std::fs::create_dir_all(&fixture.directory).unwrap();
        for suffix in [".json", ".live_portfolio.json"] {
            let name = format!("{}{suffix}", fixture.base);
            let original = golden.join(&name);
            let copied = fixture.directory.join(name);
            std::fs::copy(&original, &copied).unwrap();
            assert_eq!(
                std::fs::read(&original).unwrap(),
                std::fs::read(&copied).unwrap()
            );
        }
        fixture
    }

    fn research_path(&self) -> std::path::PathBuf {
        self.directory.join(format!("{}.json", self.base))
    }

    fn portfolio_path(&self) -> std::path::PathBuf {
        self.directory
            .join(format!("{}.live_portfolio.json", self.base))
    }

    fn mutate_file(&self, portfolio: bool, mutation: fn(&mut serde_json::Value)) {
        let path = if portfolio {
            self.portfolio_path()
        } else {
            self.research_path()
        };
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        mutation(&mut value);
        std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
    }

    fn assert_unavailable(&self, label: &str) {
        assert!(
            build(&self.root, RESEARCH_DIR, &self.base, true).is_err(),
            "accepted {label}"
        );
        let list = scan_reports(&self.root);
        assert_eq!(list.count, 0, "listed {label}");
        assert!(list.strategies.is_empty());
        assert_eq!(list.unavailable.len(), 1, "{label}: {:?}", list.unavailable);
        assert!(list.unavailable[0].contains(&self.base));
    }
}

impl Drop for SavedResearchFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn recorded_mode_accepts_search_enum_and_legacy_spellings_without_inventing_missing_values() {
    for (recorded, expected) in [
        ("Risky", "risky"),
        ("risky", "risky"),
        ("PropFirm", "prop_firm"),
        ("prop_firm", "prop_firm"),
        ("Strict", "strict"),
        ("strict", "strict"),
        ("", "unknown"),
        ("future-mode", "unknown"),
    ] {
        assert_eq!(canonical_recorded_mode(recorded), expected);
    }
    let fixture = SavedResearchFixture::copy_genuine_writer_pair();
    fixture.mutate_file(false, |value| {
        value["discovery_result"]["funnel_profile"]["mode"] = "Risky".into();
    });
    assert_eq!(
        build(&fixture.root, RESEARCH_DIR, &fixture.base, true)
            .unwrap()
            .head
            .mode,
        "risky"
    );
    let list = scan_reports(&fixture.root);
    assert_eq!(list.count, 1);
    assert_eq!(list.strategies[0].mode, "risky");
    fixture.mutate_file(false, |value| {
        value["discovery_result"]["funnel_profile"] = serde_json::Value::Null;
    });
    assert_eq!(
        build(&fixture.root, RESEARCH_DIR, &fixture.base, true)
            .unwrap()
            .head
            .mode,
        "unknown"
    );
}

#[test]
fn recorded_evaluation_reopens_saved_policy_values_and_refuses_tampering() {
    // Reuse the existing genuine Search/Models transport fixture's small policy,
    // not its synthetic performance or a policy reconstructed from Settings.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/combined-research-candidate-v1/training-handoff.json");
    let handoff: serde_json::Value = read_json(&path).unwrap();
    let body: serde_json::Value = serde_json::from_str(
        handoff["locked_portfolio"]["canonical_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let policy: LiveTradingPolicyV1 =
        serde_json::from_value(body["live_trading_policy"].clone()).unwrap();
    let recorded = recorded_evaluation(&policy).unwrap().unwrap();
    assert_eq!(recorded.policy_identity_hash, policy.identity_hash);
    assert_eq!(recorded.account_currency, "USD");
    assert_eq!(recorded.initial_capital, 12_345.67);
    assert_eq!(recorded.risk_per_trade_min, 0.003);
    assert_eq!(recorded.risk_per_trade_max, 0.023);
    assert_eq!(recorded.high_quality_confidence, 0.73);
    assert_eq!(recorded.confidence_basis, "signal_threshold_margin");
    assert!(recorded.growth_goal.is_none());
    let wire = serde_json::to_value(&recorded).unwrap();
    assert_eq!(wire["riskPerTradeMax"], 0.023);
    assert_eq!(wire["initialCapital"], 12_345.67);
    assert!(wire["growthGoal"].is_null());
    let goal = RecordedGrowthGoal::from(neoethos_search::scoring::RiskyGrowthGoal {
        start_balance: 100.0,
        target_balance: 50_000.0,
        horizon_days: 180.0,
    });
    let goal_wire = serde_json::to_value(goal).unwrap();
    assert_eq!(goal_wire["referenceStartBalance"], 100.0);
    assert_eq!(goal_wire["targetBalance"], 50_000.0);
    assert_eq!(goal_wire["horizonDays"], 180.0);
    assert_eq!(recorded.initial_capital, 12_345.67);

    let mut tampered = body["live_trading_policy"].clone();
    tampered["sealed_evaluation_v1"]["risk_per_trade_max"] = 0.3.into();
    let tampered: LiveTradingPolicyV1 = serde_json::from_value(tampered).unwrap();
    assert!(recorded_evaluation(&tampered).is_err());
    let mut incomplete = policy;
    incomplete.schema_version = 1;
    assert!(recorded_evaluation(&incomplete).is_err());
}

#[test]
fn genuine_search_v6_writer_pair_reaches_report_inventory_list_and_details_without_oos_claims() {
    let fixture = SavedResearchFixture::copy_genuine_writer_pair();
    let (artifact, trades, quality, mode) = load_inputs(&fixture.directory, &fixture.base).unwrap();
    assert_eq!(artifact.schema_version, 6);
    assert_eq!(mode, "risky");
    assert_eq!(
        artifact
            .genes
            .iter()
            .map(|g| g.strategy_id.as_str())
            .collect::<Vec<_>>(),
        vec!["report-selected-a", "report-selected-b"]
    );
    let calibration = artifact.sizing_evidence[0]
        .forward_test
        .scope()
        .evaluated_window();
    let final_window = artifact.final_holdout_scope.evaluated_window();
    assert_eq!((calibration.row_start(), calibration.row_end()), (80, 90));
    assert_eq!(
        (final_window.row_start(), final_window.row_end()),
        (90, 100)
    );
    assert!(
        artifact
            .sizing_evidence
            .iter()
            .all(|row| row.oos_metrics().net_profit > 0.0)
    );
    let research: serde_json::Value = read_json(&fixture.research_path()).unwrap();
    assert_eq!(
        research["discovery_result"]["validation_gates"]["walkforward_passed"],
        true
    );
    assert_eq!(
        research["discovery_result"]["validation_gates"]["cpcv_passed"],
        true
    );
    assert_eq!(
        trades.len(),
        3,
        "the nonselected outlier must be present in the actual saved diagnostics"
    );
    assert!(
        quality
            .iter()
            .any(|row| row.strategy_id == "report-not-selected" && row.net_profit > 20_000.0)
    );
    let (entries, unavailable) = inventory(&fixture.root);
    assert_eq!(
        entries,
        vec![(RESEARCH_DIR.to_owned(), fixture.base.clone())]
    );
    assert!(unavailable.is_empty());
    let list = scan_reports(&fixture.root);
    assert_eq!(list.count, 1);
    assert!(list.unavailable.is_empty(), "{:?}", list.unavailable);
    let report = build(&fixture.root, RESEARCH_DIR, &fixture.base, true).unwrap();
    let expected_identity = ValidationStrategyIdentityV2::from_gene(&artifact.genes[1]).unwrap();
    let expected_quality = quality
        .iter()
        .find(|row| row.strategy_id == "report-selected-b")
        .unwrap();
    for head in [&list.strategies[0], &report.head] {
        assert_eq!(head.strategy_id, "report-selected-b");
        assert_eq!(head.exact_gene_hash, expected_identity.exact_gene_hash());
        assert_eq!(head.trades, 4);
        assert_eq!(
            head.final_from_1000, 1_100.0,
            "neither sum of standalone strategies nor the outlier"
        );
        assert_eq!(head.max_dd_pct, 4.55);
        assert_eq!(head.sharpe, expected_quality.sharpe_ratio);
        assert_eq!(head.profit_factor, expected_quality.profit_factor);
        assert_eq!(head.win_rate, Some(0.75));
        assert_eq!(head.mode, "risky");
        assert!(
            head.recorded_evaluation.is_none(),
            "legacy policy is not filled from today's settings"
        );
        assert_eq!(head.diagnostic_initial_capital, 1_000.0);
        assert_eq!(head.symbol, "EURUSD");
        assert_eq!(head.timeframe, "M1");
        assert_eq!(
            (
                head.cpcv_passed,
                head.walkforward_passed,
                head.validation_complete
            ),
            (None, None, None),
            "passed selection/calibration fixtures must never become independent final OOS claims"
        );
        assert!(
            head.flags
                .iter()
                .any(|flag| flag.contains("Unsealed IS diagnostics")
                    && flag.contains("not portfolio PnL"))
        );
    }
    assert_eq!(report.monthly.len(), 1);
    assert_eq!(
        (
            report.monthly[0].month.as_str(),
            report.monthly[0].balance,
            report.monthly[0].return_pct,
            report.monthly[0].trades
        ),
        ("2025-01", 1_100.0, 10.0, 4)
    );
    assert_eq!(
        report.yearly[0].return_pct, 10.0,
        "a 0.10 return fraction displays as 10%, not 0.1%"
    );
    let wire = serde_json::to_value(&report).unwrap();
    assert_eq!(wire["strategyId"], "report-selected-b");
    assert!(wire["recordedEvaluation"].is_null());
    assert_eq!(wire["diagnosticInitialCapital"], 1_000.0);
    assert!(
        wire["walkforwardPassed"].is_null()
            && wire["cpcvPassed"].is_null()
            && wire["validationComplete"].is_null()
    );
}

#[test]
fn genuine_saved_report_refuses_rebound_ids_scope_configuration_and_receipt() {
    let mutations: &[(&str, bool, fn(&mut serde_json::Value))] = &[
        ("same display ID with changed exact gene", false, |v| {
            v["discovery_result"]["portfolio"][1]["sl_pips"] = 99.0.into();
        }),
        ("ambiguous selected ID", false, |v| {
            let duplicate = v["discovery_result"]["portfolio"][1].clone();
            v["discovery_result"]["portfolio"]
                .as_array_mut()
                .unwrap()
                .push(duplicate);
        }),
        ("missing selected ID", false, |v| {
            v["discovery_result"]["portfolio"]
                .as_array_mut()
                .unwrap()
                .remove(1);
        }),
        ("selection scope substitution", false, |v| {
            v["discovery_result"]["selection_scope"] =
                v["discovery_result"]["calibration_scope"].clone();
        }),
        ("configuration substitution", false, |v| {
            v["discovery_result"]["search_config_hash"] = "fnv64:ffffffffffffffff".into();
        }),
        ("feature ordering substitution", false, |v| {
            v["discovery_result"]["effective_feature_names"][0] = "range_pips".into();
        }),
        ("receipt identity substitution", false, |v| {
            v["execution_contract"]["input_receipt_sha256"] = "0".repeat(64).into();
        }),
        ("research evidence filename mismatch", false, |v| {
            v["evidence_identity_sha256"] = "0".repeat(64).into();
        }),
        ("missing V6 final scope", true, |v| {
            v["portfolio"]
                .as_object_mut()
                .unwrap()
                .remove("final_holdout_scope");
        }),
        ("changed compact portfolio exact gene", true, |v| {
            v["portfolio"]["genes"][1]["weights"][0] = 0.75.into();
        }),
    ];
    for (label, portfolio, mutation) in mutations {
        let fixture = SavedResearchFixture::copy_genuine_writer_pair();
        fixture.mutate_file(*portfolio, *mutation);
        fixture.assert_unavailable(label);
    }
}

#[test]
fn genuine_saved_report_refuses_missing_duplicate_or_out_of_scope_diagnostic_rows() {
    let mutations: &[(&str, fn(&mut serde_json::Value))] = &[
        ("missing selected trade row", |v| {
            v["discovery_result"]["logged_trades"]
                .as_array_mut()
                .unwrap()
                .retain(|row| row["strategy_id"] != "report-selected-b");
        }),
        ("duplicate selected trade row", |v| {
            let duplicate = v["discovery_result"]["logged_trades"][2].clone();
            v["discovery_result"]["logged_trades"]
                .as_array_mut()
                .unwrap()
                .push(duplicate);
        }),
        ("missing selected quality row", |v| {
            v["discovery_result"]["quality_metrics"]
                .as_array_mut()
                .unwrap()
                .retain(|row| row["strategy_id"] != "report-selected-b");
        }),
        ("duplicate selected quality row", |v| {
            let duplicate = v["discovery_result"]["quality_metrics"][0].clone();
            v["discovery_result"]["quality_metrics"]
                .as_array_mut()
                .unwrap()
                .push(duplicate);
        }),
        ("trade outside IS", |v| {
            let final_end =
                v["discovery_result"]["holdout_scope"]["evaluated_window"]["timestamp_end_ms"]
                    .clone();
            v["discovery_result"]["logged_trades"][2]["trades"][0]["exit_time"] = final_end;
        }),
    ];
    for (label, mutation) in mutations {
        let fixture = SavedResearchFixture::copy_genuine_writer_pair();
        fixture.mutate_file(false, *mutation);
        fixture.assert_unavailable(label);
    }
    let fixture = SavedResearchFixture::copy_genuine_writer_pair();
    std::fs::remove_file(fixture.portfolio_path()).unwrap();
    fixture.assert_unavailable("missing adjacent validated V6 portfolio");
}

#[tokio::test]
async fn report_http_distinguishes_incomplete_diagnostics_missing_files_and_identity_conflicts() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let fixture = SavedResearchFixture::copy_genuine_writer_pair();
    let mut settings = Settings::default();
    settings.system.cache_dir = fixture.root.clone();
    let config_path = fixture.root.join("config.yaml");
    std::fs::write(&config_path, serde_yaml_ng::to_string(&settings).unwrap()).unwrap();
    // Exercise the production HTTP reader with an operation-local config path;
    // do not replace process-global configuration used by parallel App tests.
    let router = axum::Router::new()
        .route(
            "/strategy/report",
            axum::routing::get(
                |State(config): State<PathBuf>, Query(query): Query<ReportQuery>| async move {
                    report_from_config(&config, query).await
                },
            ),
        )
        .with_state(config_path);
    let uri = format!("/strategy/report?dir={RESEARCH_DIR}&base={}", fixture.base);
    let request = |uri: &str| Request::builder().uri(uri).body(Body::empty()).unwrap();

    let response = router.clone().oneshot(request(&uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["strategyId"], "report-selected-b");
    assert_eq!(body["trades"], 4);
    assert_eq!(body["mode"], "risky");
    assert!(
        body["recordedEvaluation"].is_null(),
        "genuine legacy policy must not be backfilled"
    );
    assert_eq!(body["diagnosticInitialCapital"], 1000.0);

    let response = router
        .clone()
        .oneshot(request(&format!("{uri}&strategy_id=other")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    assert!(
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains("Selected strategy identity differs")
    );

    let original_research = std::fs::read(fixture.research_path()).unwrap();
    fixture.mutate_file(false, |value| {
        let row = value["discovery_result"]["logged_trades"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|row| row["strategy_id"] == "report-selected-b")
            .unwrap();
        row["trades"][0]["pnl_pct"] = 42.0.into();
    });
    let response = router.clone().oneshot(request(&uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        body["detail"],
        "diagnostic pnl_pct does not equal pnl / initial_capital"
    );
    assert!(body.get("monthly").is_none());
    std::fs::write(fixture.research_path(), original_research).unwrap();

    fixture.mutate_file(false, |value| {
        value["discovery_result"]["logged_trades"]
            .as_array_mut()
            .unwrap()
            .retain(|row| row["strategy_id"] != "report-selected-b");
    });
    let response = router.clone().oneshot(request(&uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"], "Saved strategy diagnostics are unavailable.");
    assert_eq!(
        body["detail"],
        "selected strategy 'report-selected-b' has no diagnostic trades"
    );
    assert!(body.get("monthly").is_none());
    assert!(
        !String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains(fixture.root.to_str().unwrap())
    );

    let missing_uri = format!(
        "/strategy/report?dir={RESEARCH_DIR}&base={}.research",
        "0".repeat(64)
    );
    let response = router.clone().oneshot(request(&missing_uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"], "Saved strategy report was not found.");

    let legacy_dir = fixture.root.join("auto_loop_http");
    std::fs::create_dir(&legacy_dir).unwrap();
    let legacy_uri = "/strategy/report?dir=auto_loop_http&base=missing";
    let response = router.clone().oneshot(request(legacy_uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"], "Saved strategy report was not found.");
    for name in [
        "missing.live_portfolio.json",
        "missing.json.live_portfolio.json",
    ] {
        std::fs::copy(fixture.portfolio_path(), legacy_dir.join(name)).unwrap();
    }
    let response = router.clone().oneshot(request(legacy_uri)).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"], "Saved strategy report is invalid.");

    for uri in [
        "/strategy/report?dir=..&base=bad",
        "/strategy/report?dir=discovery/research&base=bad.research",
    ] {
        let response = router.clone().oneshot(request(uri)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
