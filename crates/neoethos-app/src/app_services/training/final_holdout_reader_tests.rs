use super::super::tests::TestDirectory;
use super::*;
use crate::app_services::training::final_holdout::{FinalHoldoutUse, begin};
use neoethos_search::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchEvaluatedWindowV1,
    CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};

struct Fixture {
    root: TestDirectory,
    context: VerifiedContext,
    scope: CanonicalSearchArtifactScopeV2,
}

impl Fixture {
    fn new() -> Self {
        let root = TestDirectory::new();
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            features.provenance().bindings()[0].dataset_identity(),
            &features,
        )
        .unwrap();
        let scope = CanonicalSearchArtifactScopeV2::new(
            receipt,
            CanonicalSearchEvaluatedWindowV1::new(
                CanonicalSearchWindowRoleV1::Holdout,
                90,
                100,
                features.timestamps[90],
                features.timestamps[99],
            )
            .unwrap(),
        )
        .unwrap();
        let window = scope.evaluated_window();
        // Only this private reader seam receives synthetic candidate identities.
        // It proves journal/report binding, not training or installed-model proof.
        let context = VerifiedContext {
            handoff: "a".repeat(64),
            portfolio: "b".repeat(64),
            model: Some(VerifiedModelContext {
                candidate_tree: "c".repeat(64),
                model_input: "d".repeat(64),
            }),
            raw_scope: raw_scope_material(&scope).unwrap(),
            final_window: serde_json::to_value(window).unwrap(),
            final_scope_identity: scope.identity_sha256().unwrap(),
            cost_identity: "e".repeat(64),
            symbol: "EURUSD".to_owned(),
            base_timeframe: "M1".to_owned(),
            account_currency: "EUR".to_owned(),
            row_start: 90,
            row_end: 100,
            timestamp_start_ms: features.timestamps[90],
            timestamp_end_ms: features.timestamps[99],
            training_cutoff_ms: features.timestamps[80],
        };
        std::fs::create_dir_all(root.path().join("models/candidates")).unwrap();
        Self {
            root,
            context,
            scope,
        }
    }
    fn data(&self) -> PathBuf {
        self.root.path().join("data")
    }
    fn candidates(&self) -> PathBuf {
        self.root.path().join("models/candidates")
    }
    fn lock(&self) -> Value {
        let model = self.context.model.as_ref().unwrap();
        serde_json::json!({
            "protocol": PROTOCOL, "training_handoff": self.context.handoff,
            "locked_portfolio_identity_sha256": self.context.portfolio,
            "candidate_tree_sha256": model.candidate_tree, "model_input_sha256": model.model_input,
            "model_inference_settings": {"blend_gate_floor":0.34,"blend_veto_below":0.15,"live_ml_gate":true,"fixture":"synthetic journal reader only"},
            "blend_mode":"ml_scale","blend_gate_floor":0.34,"blend_veto_below":0.15,"model_history_rows":256,
            "account_policy":"archived_search_risk_fractional_lots_v1"
        })
    }
    fn start(&self) -> FinalHoldoutUse {
        begin(&self.data(), &self.scope, self.lock()).unwrap()
    }
    fn report(&self, usage: &FinalHoldoutUse, net: f64) -> Value {
        synthetic_report(&self.context, usage, net)
    }
    fn save(&self, usage: &FinalHoldoutUse, report: &Value) -> PathBuf {
        let path = usage.report_path(
            &self.candidates(),
            crate::app_services::training::combined::REPORT_SUFFIX,
        );
        usage.complete(&path, report).unwrap();
        path
    }
    fn read(&self) -> CombinedResearchReportsDto {
        read_verified_reports(&self.data(), &self.candidates(), &self.context).unwrap()
    }
}

// Synthetic financial values exercise display/binding only; no kernel,
// inference, trading profitability or financial admission is claimed.
fn synthetic_report(context: &VerifiedContext, usage: &FinalHoldoutUse, net: f64) -> Value {
    let account = neoethos_search::eval::NettedBarResearchResultV1 {
        execution_basis: "canonical_cpu_ohlc_screening_prior_bar_signal_next_close_fill; scalar_account_pip_and_cost_assumptions; closed_trade_realized_pnl; open_gross_mark_before_pending_costs",
        promotion_eligible: false,
        scope_identity_sha256: context.final_scope_identity.clone(),
        cost_contract_identity_sha256: context.cost_identity.clone(),
        account_currency: context.account_currency.clone(),
        metrics: neoethos_search::eval::BacktestMetrics {
            net_profit: net,
            sharpe: f64::NAN,
            peak_equity: 1025.0,
            max_drawdown: 0.1,
            win_rate: if net > 0.0 { 1.0 } else { 0.0 },
            profit_factor: f64::INFINITY,
            expectancy: net,
            monthly_target_hit_rate: 1.0,
            trade_count: 1,
            consistency: 1.0,
            max_daily_drawdown: 0.1,
        },
        closed_trades: vec![neoethos_search::quality::Trade {
            entry_time: context.timestamp_start_ms,
            exit_time: Some(context.timestamp_end_ms),
            pnl: net,
            pnl_pct: Some(net / 1000.0),
            duration_hours: Some(
                (context.timestamp_end_ms - context.timestamp_start_ms) as f64 / 3_600_000.0,
            ),
            mfe: net.max(0.0),
            mae: net.min(0.0),
            r_multiple: net / 10.0,
        }],
        ending_realized_balance: 1000.0 + net,
        terminal_open: None,
        below_min_entries: 0,
        lot_grid: None,
    };
    serde_json::json!({
            "schema":PROTOCOL,"training_handoff":context.handoff,"locked_portfolio_identity_sha256":context.portfolio,"candidate_tree_sha256":context.model.as_ref().map(|model| &model.candidate_tree),"model_input_sha256":context.model.as_ref().map(|model| &model.model_input),
            "symbol":context.symbol,"base_timeframe":context.base_timeframe,"rows":context.row_end-context.row_start,"timestamp_start_ms":context.timestamp_start_ms,"timestamp_end_ms":context.timestamp_end_ms,"model_history_rows":256,"inference_workers":1,"invalid_model_signal_rows":2,
            "blend_mode":"ml_scale","blend_gate_floor":0.34,"blend_veto_below":0.15,"configured_live_ml_gate":true,
            "holdout_use":if usage.first_recorded_use { FIRST_USE } else { REUSED },"historical_exposure":EXPOSURE,
            "raw_final_scope_sha256":usage.raw_scope_sha256,"locked_final_inputs_sha256":usage.locked_inputs_sha256,"first_locked_final_inputs_sha256":usage.first_locked_inputs_sha256,
            "final_scope_window":context.final_window,"training_cutoff_ms":context.training_cutoff_ms,
            "sizing_basis":"archived_search_confidence_risk_band_not_live_Risky_or_PropFirm_account_simulation",
            "exit_basis":"canonical_bar_brackets_trailing_time_and_session_policy_not_live_reversal_or_supervisor_actions",
            "volume_basis":"fractional_research_lots_broker_grid_not_attested","promotion_eligible":false,"gene_only":account,"combined":account,
    })
}

fn strategy_lock(context: &VerifiedContext) -> Value {
    serde_json::json!({
        "protocol": STRATEGY_PROTOCOL,
        "training_handoff": context.handoff,
        "locked_portfolio_identity_sha256": context.portfolio,
        "account_policy": "archived_search_risk_fractional_lots_v1",
    })
}

fn synthetic_strategy_report(
    context: &VerifiedContext,
    usage: &FinalHoldoutUse,
    net: f64,
) -> Value {
    let mut report = synthetic_report(context, usage, net);
    report["schema"] = serde_json::json!(STRATEGY_PROTOCOL);
    for key in [
        "candidate_tree_sha256",
        "model_input_sha256",
        "model_history_rows",
        "inference_workers",
        "invalid_model_signal_rows",
        "blend_mode",
        "blend_gate_floor",
        "blend_veto_below",
        "configured_live_ml_gate",
        "combined",
    ] {
        report.as_object_mut().unwrap().remove(key);
    }
    report
}

/// Real saved handoff and journal, explicitly synthetic account result. No
/// installed model tree/manifest/input, training, inference or held-out read.
pub(crate) fn install_saved_strategy_research_test_fixture(root: &Path) -> Result<String> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/combined-research-candidate-v1/training-handoff.json");
    let handoff: PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(
        &neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
            &source,
            neoethos_models::MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64,
        )?,
    )?;
    let data = root.join("data");
    let identity = crate::app_services::training::handoff::publish(&data, &handoff)?;
    let context = verified_handoff_context(&handoff)?;
    let portfolio = handoff.locked_portfolio().deserialize_live_portfolio()?;
    let usage = begin(
        &data,
        &portfolio.final_holdout_scope,
        strategy_lock(&context),
    )?;
    let candidates = root.join("models/candidates");
    std::fs::create_dir_all(&candidates)?;
    usage.complete(
        &usage.report_path(&candidates, STRATEGY_SUFFIX),
        &synthetic_strategy_report(&context, &usage, -20.0),
    )?;
    Ok(identity)
}

#[test]
fn public_strategy_reader_needs_only_saved_handoff_not_models_or_final_data() {
    let root = TestDirectory::new();
    let identity = install_saved_strategy_research_test_fixture(root.path()).unwrap();
    let candidates = root.path().join("models/candidates");
    assert_eq!(std::fs::read_dir(&candidates).unwrap().count(), 1);
    let before = snapshot(root.path());
    let reports =
        read_saved_final_research_reports(&root.path().join("data"), &candidates, &identity)
            .unwrap();
    assert_eq!(reports.status, "completed_results");
    assert!(reports.unavailable.is_empty(), "{:?}", reports.unavailable);
    assert_eq!(reports.reports.len(), 1);
    let report = &reports.reports[0];
    assert_eq!(report.evaluation_mode, "strategy_only");
    assert_eq!(report.gene_only.net_profit, Some(-20.0));
    assert_eq!(report.gene_only.sharpe, None);
    assert_eq!(report.gene_only.win_rate, Some(0.0));
    assert_eq!(report.gene_only.profit_factor, None);
    assert_eq!(report.gene_only.expectancy, Some(-20.0));
    assert!(!report.promotion_eligible);
    assert!(report.model_inference_settings.is_null());
    let wire = serde_json::to_value(report).unwrap();
    for key in [
        "combined",
        "blendMode",
        "blendGateFloor",
        "blendVetoBelow",
        "modelHistoryRows",
        "invalidModelSignalRows",
        "configuredLiveMlGate",
    ] {
        assert!(wire.get(key).unwrap().is_null(), "{key}: {wire}");
    }
    assert_eq!(
        before,
        snapshot(root.path()),
        "read must not install models or consume final data"
    );
}

#[test]
fn incomplete_strategy_attempt_needs_no_model_directory_and_retry_stays_reused() {
    let mut fixture = Fixture::new();
    fixture.context.model = None;
    let candidates = fixture.root.path().join("not-created-candidates");
    let first = begin(
        &fixture.data(),
        &fixture.scope,
        strategy_lock(&fixture.context),
    )
    .unwrap();
    let before = snapshot(fixture.root.path());
    let pending = read_reports(&fixture.data(), &candidates, &fixture.context, true, || {
        panic!("an incomplete strategy attempt must not inspect models")
    })
    .unwrap();
    assert_eq!(pending.status, "candidate_not_ready");
    assert!(pending.reports.is_empty() && pending.unavailable.is_empty());
    assert!(!candidates.exists());
    assert_eq!(before, snapshot(fixture.root.path()));

    let retry = begin(
        &fixture.data(),
        &fixture.scope,
        strategy_lock(&fixture.context),
    )
    .unwrap();
    assert!(!retry.first_recorded_use);
    std::fs::create_dir(&candidates).unwrap();
    retry
        .complete(
            &retry.report_path(&candidates, STRATEGY_SUFFIX),
            &synthetic_strategy_report(&fixture.context, &retry, 1.0),
        )
        .unwrap();
    let result = read_reports(&fixture.data(), &candidates, &fixture.context, true, || {
        panic!("strategy completion must not inspect models")
    })
    .unwrap();
    assert!(result.unavailable.is_empty(), "{:?}", result.unavailable);
    assert_eq!(result.reports.len(), 1);
    assert_eq!(result.reports[0].holdout_use, REUSED);
    assert_eq!(
        result.reports[0]
            .first_locked_final_inputs_sha256
            .as_deref(),
        Some(first.locked_inputs_sha256.as_str())
    );
}

#[test]
fn mixed_modes_share_first_use_and_combined_wrapper_cannot_return_strategy_policy() {
    for strategy_first in [true, false] {
        let fixture = Fixture::new();
        for strategy in [strategy_first, !strategy_first] {
            let usage = begin(
                &fixture.data(),
                &fixture.scope,
                if strategy {
                    strategy_lock(&fixture.context)
                } else {
                    fixture.lock()
                },
            )
            .unwrap();
            let report = if strategy {
                synthetic_strategy_report(&fixture.context, &usage, -20.0)
            } else {
                fixture.report(&usage, 25.0)
            };
            usage
                .complete(
                    &usage.report_path(
                        &fixture.candidates(),
                        if strategy {
                            STRATEGY_SUFFIX
                        } else {
                            crate::app_services::training::combined::REPORT_SUFFIX
                        },
                    ),
                    &report,
                )
                .unwrap();
        }
        let before = snapshot(fixture.root.path());
        let reports = read_reports(
            &fixture.data(),
            &fixture.candidates(),
            &fixture.context,
            true,
            || panic!("already verified model context must not reload"),
        )
        .unwrap();
        assert!(reports.unavailable.is_empty(), "{:?}", reports.unavailable);
        assert_eq!(reports.reports.len(), 2);
        assert_eq!(
            reports
                .reports
                .iter()
                .filter(|r| r.holdout_use == FIRST_USE)
                .count(),
            1
        );
        assert_eq!(
            reports
                .reports
                .iter()
                .filter(|r| r.holdout_use == REUSED)
                .count(),
            1
        );
        let combined_only = fixture.read();
        assert_eq!(combined_only.reports.len(), 1);
        assert_eq!(combined_only.reports[0].evaluation_mode, "train_models");
        assert!(!combined_only.reports[0].model_inference_settings.is_null());
        assert_eq!(before, snapshot(fixture.root.path()));
    }
}

#[test]
fn failed_model_verification_is_cached_once_without_hiding_strategy_attempts() {
    let mut fixture = Fixture::new();
    let strategy = begin(
        &fixture.data(),
        &fixture.scope,
        strategy_lock(&fixture.context),
    )
    .unwrap();
    strategy
        .complete(
            &strategy.report_path(&fixture.candidates(), STRATEGY_SUFFIX),
            &synthetic_strategy_report(&fixture.context, &strategy, -20.0),
        )
        .unwrap();
    for net in [10.0, 20.0] {
        let usage = fixture.start();
        fixture.save(&usage, &fixture.report(&usage, net));
    }
    fixture.context.model = None;
    let calls = std::cell::Cell::new(0);
    let reports = read_reports(
        &fixture.data(),
        &fixture.candidates(),
        &fixture.context,
        true,
        || {
            calls.set(calls.get() + 1);
            anyhow::bail!("synthetic missing model evidence")
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(reports.reports.len(), 1);
    assert_eq!(reports.reports[0].evaluation_mode, "strategy_only");
    assert_eq!(reports.unavailable.len(), 2);
    assert!(
        reports
            .unavailable
            .iter()
            .all(|r| r.reason.contains("synthetic missing model evidence"))
    );
}

#[test]
fn strategy_scope_policy_model_fields_and_promotion_tamper_are_unavailable() {
    for (pointer, replacement) in [
        (
            "/final_scope_window/role",
            serde_json::json!("selection_validation"),
        ),
        (
            "/gene_only/cost_contract_identity_sha256",
            serde_json::json!("f".repeat(64)),
        ),
        ("/gene_only/account_currency", serde_json::json!("USD")),
        ("/promotion_eligible", serde_json::json!(true)),
        ("/schema", serde_json::json!(PROTOCOL)),
        ("/combined", Value::Null),
    ] {
        let fixture = Fixture::new();
        let usage = begin(
            &fixture.data(),
            &fixture.scope,
            strategy_lock(&fixture.context),
        )
        .unwrap();
        let mut report = synthetic_strategy_report(&fixture.context, &usage, 1.0);
        if pointer == "/combined" {
            report["combined"] = replacement;
        } else {
            *report.pointer_mut(pointer).unwrap() = replacement;
        }
        usage
            .complete(
                &usage.report_path(&fixture.candidates(), STRATEGY_SUFFIX),
                &report,
            )
            .unwrap();
        let result = read_reports(
            &fixture.data(),
            &fixture.candidates(),
            &fixture.context,
            true,
            || panic!("strategy does not read models"),
        )
        .unwrap();
        assert!(result.reports.is_empty(), "{pointer}");
        assert_eq!(result.unavailable.len(), 1, "{pointer}");
    }
}

fn snapshot(path: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut result = Vec::new();
    if path.is_dir() {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                result.extend(snapshot(&path));
            } else {
                result.push((path.clone(), std::fs::read(path).unwrap()));
            }
        }
    }
    result.sort_by(|left, right| left.0.cmp(&right.0));
    result
}

/// Shared by the real HTTP child-harness test. Copies only the reviewed output
/// of the private Search policy producer and actual Models candidate installer.
/// Model bytes and report PnLs are synthetic; no inference or profitability proof.
pub(crate) fn install_saved_research_test_fixture(root: &Path) -> Result<String> {
    let bundle =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/combined-research-candidate-v1");
    let handoff: neoethos_models::PromotionCandidateTrainingHandoffV1 = serde_json::from_slice(
        &neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
            &bundle.join("training-handoff.json"),
            neoethos_models::MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64,
        ).context("install reviewed candidate fixture with the documented private generators before running this contract")?,
    )?;
    let manifest_bytes =
        neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
            &bundle.join("manifest.json"),
            1024 * 1024,
        )?;
    let manifest: PromotionCandidateTrainingManifestV1 = serde_json::from_slice(&manifest_bytes)?;
    let identity = handoff.identity_sha256()?;
    ensure_sha(manifest.candidate_relative_dir())?;
    let candidates = root.join("models/candidates");
    let destination = candidates.join(manifest.candidate_relative_dir());
    ensure!(
        !destination.try_exists()?,
        "fixture candidate destination must be fresh"
    );
    let mut files = 0;
    let mut bytes = 0;
    copy_fixture_tree(
        &bundle.join(manifest.candidate_relative_dir()),
        &destination,
        0,
        &mut files,
        &mut bytes,
    )?;
    let data = root.join("data");
    crate::app_services::training::handoff::publish(&data, &handoff)?;
    super::super::write_new(
        &candidates.join(format!("{identity}.manifest.json")),
        &manifest,
    )?;
    let context = verified_context(&candidates, &identity, &manifest)?;
    let model = context
        .model
        .as_ref()
        .context("fixture model context absent")?;
    let portfolio = handoff.locked_portfolio().deserialize_live_portfolio()?;
    let mut settings = neoethos_core::Settings::default();
    settings.models.blend_gate_floor = 0.34;
    settings.models.blend_veto_below = 0.15;
    settings.models.live_ml_gate = true;
    let locked = serde_json::json!({
        "protocol":PROTOCOL,"training_handoff":identity,"locked_portfolio_identity_sha256":context.portfolio,
        "candidate_tree_sha256":model.candidate_tree,"model_input_sha256":model.model_input,
        "model_inference_settings":settings.models,"blend_mode":"ml_scale","blend_gate_floor":0.34,
        "blend_veto_below":0.15,"model_history_rows":256,"account_policy":"archived_search_risk_fractional_lots_v1",
    });
    for net in [-20.0, 25.0] {
        let usage = begin(&data, &portfolio.final_holdout_scope, locked.clone())?;
        let report = synthetic_report(&context, &usage, net);
        let path = usage.report_path(
            &candidates,
            crate::app_services::training::combined::REPORT_SUFFIX,
        );
        usage.complete(&path, &report)?;
    }
    Ok(identity)
}

fn copy_fixture_tree(
    source: &Path,
    destination: &Path,
    depth: usize,
    files: &mut usize,
    bytes: &mut u64,
) -> Result<()> {
    ensure!(depth <= 8, "reviewed test fixture tree is too deep");
    ensure_physical(source, true)?;
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let target = destination.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.is_dir() {
            copy_fixture_tree(&path, &target, depth + 1, files, bytes)?;
        } else {
            ensure_physical(&path, false)?;
            *files += 1;
            *bytes = bytes
                .checked_add(metadata.len())
                .context("fixture byte count overflow")?;
            ensure!(
                *files <= 32 && *bytes <= 8 * 1024 * 1024,
                "reviewed tiny candidate fixture exceeds copy bound"
            );
            ensure!(
                !target.try_exists()?,
                "fixture copy must not overwrite a file"
            );
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

#[test]
fn public_saved_reader_reopens_genuine_installed_candidate_and_detects_tree_tamper() {
    let root = TestDirectory::new();
    let identity = install_saved_research_test_fixture(root.path()).unwrap();
    let candidates = root.path().join("models/candidates");
    let data = root.path().join("data");
    let manifest: PromotionCandidateTrainingManifestV1 = serde_json::from_slice(
        &std::fs::read(candidates.join(format!("{identity}.manifest.json"))).unwrap(),
    )
    .unwrap();
    let before = snapshot(root.path());
    let report =
        read_saved_combined_research_reports(&data, &candidates, &identity, &manifest).unwrap();
    assert!(report.unavailable.is_empty(), "{:?}", report.unavailable);
    assert_eq!(report.reports.len(), 2);
    assert_eq!(report.training_handoff, identity);
    assert!(
        report
            .reports
            .iter()
            .all(|r| r.account_currency == "USD" && !r.promotion_eligible)
    );
    assert!(
        report
            .reports
            .iter()
            .any(|r| r.gene_only.net_profit == Some(-20.0))
    );
    assert!(
        report
            .reports
            .iter()
            .any(|r| r.combined.as_ref().unwrap().net_profit == Some(25.0))
    );
    assert_eq!(before, snapshot(root.path()));
    let tree = candidates.join(manifest.candidate_relative_dir());
    let entry = snapshot(&tree)
        .into_iter()
        .find(|(path, _)| {
            path.file_name().and_then(|n| n.to_str()) == Some("model_feature_input.v1.json")
        })
        .unwrap();
    let mut changed = entry.1;
    changed.push(b' ');
    std::fs::write(entry.0, changed).unwrap();
    let error =
        read_saved_combined_research_reports(&data, &candidates, &identity, &manifest).unwrap_err();
    assert!(format!("{error:#}").contains("installed candidate tree hash/count/bytes changed"));
}

#[test]
fn journal_writer_to_reader_keeps_all_attempts_and_does_not_mutate_or_rank_profit() {
    let fixture = Fixture::new();
    let first = fixture.start();
    let first_report = fixture.report(&first, -20.0);
    let path = fixture.save(&first, &first_report);
    // Hash identity is semantic JSON, not its whitespace/raw-byte digest.
    std::fs::write(path, serde_json::to_vec_pretty(&first_report).unwrap()).unwrap();
    let second = fixture.start();
    fixture.save(&second, &fixture.report(&second, 25.0));
    let before = snapshot(fixture.root.path());
    let result = fixture.read();
    assert_eq!(result.status, "completed_results");
    assert!(result.unavailable.is_empty(), "{:?}", result.unavailable);
    assert_eq!(result.reports.len(), 2);
    assert_eq!(
        result
            .reports
            .iter()
            .filter(|r| r.holdout_use == FIRST_USE)
            .count(),
        1
    );
    assert_eq!(
        result
            .reports
            .iter()
            .filter(|r| r.holdout_use == REUSED)
            .count(),
        1
    );
    assert!(
        result
            .reports
            .iter()
            .any(|r| r.gene_only.net_profit == Some(-20.0))
    );
    assert!(
        result
            .reports
            .iter()
            .any(|r| r.gene_only.net_profit == Some(25.0))
    );
    assert!(
        result
            .reports
            .windows(2)
            .all(|pair| pair[0].report_id <= pair[1].report_id)
    );
    for report in &result.reports {
        assert_eq!(
            report.model_inference_settings,
            fixture.lock()["model_inference_settings"]
        );
        assert!(
            serde_json::to_value(report)
                .unwrap()
                .get("modelInferenceSettings")
                .is_none(),
            "internal policy binding is not an extra UI readiness field"
        );
        assert!(!report.promotion_eligible);
        assert_eq!(report.historical_exposure, EXPOSURE);
        assert_eq!(
            (report.row_start, report.row_end, report.rows),
            (90, 100, 10)
        );
        assert_eq!(report.account_currency, "EUR");
        assert_eq!(
            report.combined.as_ref().unwrap().max_drawdown_fraction,
            Some(0.10)
        );
        assert_eq!(report.evaluation_mode, "train_models");
    }
    assert_eq!(before, snapshot(fixture.root.path()));
}

#[test]
fn saved_account_metrics_reach_wire_without_recalculation_or_account_mixup() {
    let fixture = Fixture::new();
    let usage = fixture.start();
    let mut report = fixture.report(&usage, -20.0);
    // Synthetic display values deliberately differ between the two accounts.
    // This is projection/binding evidence, not a validation of financial math.
    for (field, gene, combined) in [
        ("sharpe", -1.25, 0.75),
        ("win_rate", 0.25, 0.75),
        ("profit_factor", 0.6, 1.4),
        ("expectancy", -20.0, 12.5),
    ] {
        report["gene_only"]["metrics"][field] = serde_json::json!(gene);
        report["combined"]["metrics"][field] = serde_json::json!(combined);
    }
    fixture.save(&usage, &report);
    let before = snapshot(fixture.root.path());
    let result = fixture.read();
    assert!(result.unavailable.is_empty(), "{:?}", result.unavailable);
    assert_eq!(result.reports.len(), 1);
    let wire = serde_json::to_value(&result.reports[0]).unwrap();
    for (field, gene, combined) in [
        ("sharpe", -1.25, 0.75),
        ("winRate", 0.25, 0.75),
        ("profitFactor", 0.6, 1.4),
        ("expectancy", -20.0, 12.5),
    ] {
        assert_eq!(wire["geneOnly"][field], serde_json::json!(gene), "{field}");
        assert_eq!(
            wire["combined"][field],
            serde_json::json!(combined),
            "{field}"
        );
    }
    assert_eq!(wire["promotionEligible"], false);
    assert_eq!(before, snapshot(fixture.root.path()));
}

#[test]
fn null_metrics_are_unknown_and_terminal_mark_is_not_closed_profit() {
    let fixture = Fixture::new();
    let usage = fixture.start();
    let mut report = fixture.report(&usage, 25.0);
    report["combined"]["metrics"]["net_profit"] = Value::Null;
    report["combined"]["metrics"]["max_drawdown"] = Value::Null;
    for field in ["sharpe", "win_rate", "profit_factor", "expectancy"] {
        report["combined"]["metrics"][field] = Value::Null;
        report["gene_only"]["metrics"][field] = serde_json::json!(0.0);
    }
    report["combined"]["terminal_open"] = serde_json::json!({
        "direction":1,"entry_bar_index":8,"entry_timestamp_ms":fixture.context.timestamp_end_ms-60_000,
        "modeled_entry_price":1.1,"lots":0.1,"stop_pips":10.0,"target_pips":20.0,"active_trailing_stop_price":null,
        "mark_timestamp_ms":fixture.context.timestamp_end_ms,"mark_close_price":1.2,
        "gross_unrealized_account":12.0,"pending_round_trip_commission_account":0.9,"marked_equity_before_pending_costs":1037.0
    });
    fixture.save(&usage, &report);
    let result = fixture.read();
    assert!(result.unavailable.is_empty(), "{:?}", result.unavailable);
    let combined = result.reports[0].combined.as_ref().unwrap();
    assert_eq!(combined.net_profit, None);
    assert_eq!(combined.max_drawdown_fraction, None);
    let wire = serde_json::to_value(&result.reports[0]).unwrap();
    for field in ["sharpe", "winRate", "profitFactor", "expectancy"] {
        assert!(wire["combined"][field].is_null(), "{field}");
        assert_eq!(wire["geneOnly"][field], serde_json::json!(0.0), "{field}");
    }
    assert_eq!(combined.ending_realized_balance, 1025.0);
    assert_eq!(combined.gross_unrealized_account, Some(12.0));
    assert_eq!(combined.pending_round_trip_commission_account, Some(0.9));
    assert!(combined.terminal_open);
}

#[test]
fn incomplete_first_marker_never_upgrades_reuse_even_if_marker_later_finishes() {
    let fixture = Fixture::new();
    let raw_hash = hash_material(
        b"neoethos.final-raw-window.v1\0",
        &fixture.context.raw_scope,
    )
    .unwrap();
    let directory = fixture.data().join("final_holdout_uses").join(&raw_hash);
    std::fs::create_dir_all(&directory).unwrap();
    let marker = directory.join("first-start.json");
    std::fs::write(&marker, b"").unwrap();
    let usage = fixture.start();
    assert!(!usage.first_recorded_use);
    assert_eq!(usage.first_locked_inputs_sha256, None);
    fixture.save(&usage, &fixture.report(&usage, 1.0));
    for finish_first in [false, true] {
        if finish_first {
            std::fs::write(
                &marker,
                serde_json::to_vec(&FirstUse {
                    schema: JOURNAL_SCHEMA.to_owned(),
                    raw_scope_sha256: raw_hash.clone(),
                    locked_inputs_sha256: "f".repeat(64),
                })
                .unwrap(),
            )
            .unwrap();
        }
        let result = fixture.read();
        assert!(result.unavailable.is_empty(), "{:?}", result.unavailable);
        assert_eq!(result.reports[0].holdout_use, REUSED);
        assert_eq!(result.reports[0].first_locked_final_inputs_sha256, None);
    }
}

#[test]
fn missing_report_is_not_no_completed_result_and_reading_never_completes_an_attempt() {
    let fixture = Fixture::new();
    assert_eq!(fixture.read().status, "no_completed_result");
    let usage = fixture.start();
    assert_eq!(fixture.read().status, "no_completed_result");
    let path = fixture.save(&usage, &fixture.report(&usage, 2.0));
    std::fs::remove_file(path).unwrap();
    let before = snapshot(fixture.root.path());
    let result = fixture.read();
    assert_eq!(result.status, "completed_results");
    assert!(result.reports.is_empty());
    assert_eq!(result.unavailable.len(), 1);
    assert_eq!(before, snapshot(fixture.root.path()));
}

#[test]
fn saved_semantic_tamper_and_rebound_policy_scope_and_identity_are_unavailable() {
    let changes: &[(&str, Value)] = &[
        ("/combined/metrics/net_profit", serde_json::json!(9999.0)),
        ("/training_handoff", serde_json::json!("f".repeat(64))),
        ("/model_input_sha256", serde_json::json!("f".repeat(64))),
        ("/candidate_tree_sha256", serde_json::json!("f".repeat(64))),
        ("/blend_gate_floor", serde_json::json!(0.8)),
        ("/configured_live_ml_gate", serde_json::json!(false)),
        (
            "/final_scope_window/role",
            serde_json::json!("selection_validation"),
        ),
        ("/training_cutoff_ms", serde_json::json!(0)),
        ("/promotion_eligible", serde_json::json!(true)),
        (
            "/combined/scope_identity_sha256",
            serde_json::json!("f".repeat(64)),
        ),
        (
            "/combined/cost_contract_identity_sha256",
            serde_json::json!("f".repeat(64)),
        ),
        ("/combined/account_currency", serde_json::json!("USD")),
    ];
    for (index, (pointer, replacement)) in changes.iter().enumerate() {
        let fixture = Fixture::new();
        let usage = fixture.start();
        let mut report = fixture.report(&usage, 25.0);
        if index == 0 {
            let path = fixture.save(&usage, &report);
            *report.pointer_mut(pointer).unwrap() = replacement.clone();
            std::fs::write(path, serde_json::to_vec(&report).unwrap()).unwrap();
        } else {
            *report.pointer_mut(pointer).unwrap() = replacement.clone();
            // A genuine completion hash of a wrong-context report must still
            // fail semantic binding, not merely the byte/hash check.
            fixture.save(&usage, &report);
        }
        let result = fixture.read();
        assert!(result.reports.is_empty(), "unexpectedly accepted {pointer}");
        assert_eq!(result.unavailable.len(), 1, "{pointer}");
    }
}

#[test]
fn changed_locked_settings_and_redirected_completion_are_unavailable() {
    for change_start in [true, false] {
        let fixture = Fixture::new();
        let usage = fixture.start();
        fixture.save(&usage, &fixture.report(&usage, 25.0));
        let path = usage.directory.join(format!(
            "{}.{}.json",
            usage.attempt,
            if change_start { "start" } else { "completed" }
        ));
        let mut wire: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if change_start {
            wire["locked_inputs"]["model_inference_settings"]["blend_gate_floor"] =
                serde_json::json!(0.9);
        } else {
            wire["report_path"] = serde_json::json!(fixture.root.path().join("outside.json"));
        }
        std::fs::write(path, serde_json::to_vec(&wire).unwrap()).unwrap();
        if change_start {
            let error =
                read_verified_reports(&fixture.data(), &fixture.candidates(), &fixture.context)
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("hash mismatch before attempt attribution")
            );
            continue;
        }
        let result = fixture.read();
        assert!(result.reports.is_empty());
        assert_eq!(result.unavailable.len(), 1);
    }
}

#[test]
fn duplicate_completed_first_claims_are_not_presented_as_independent_fresh_uses() {
    let fixture = Fixture::new();
    let first = fixture.start();
    fixture.save(&first, &fixture.report(&first, 1.0));
    let mut second = fixture.start();
    let path = second
        .directory
        .join(format!("{}.start.json", second.attempt));
    let mut wire: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    wire["first_recorded_use"] = serde_json::json!(true);
    std::fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    second.first_recorded_use = true;
    fixture.save(&second, &fixture.report(&second, 2.0));
    let error = read_verified_reports(&fixture.data(), &fixture.candidates(), &fixture.context)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("multiple completed attempts claim")
    );
}

#[test]
fn display_byte_limits_and_parent_traversal_fail_explicitly() {
    let root = TestDirectory::new();
    let path = root.path().join("bytes.json");
    std::fs::write(&path, b"12345").unwrap();
    let mut budget = ReadBudget::default();
    assert!(
        budget
            .read(&path, 4)
            .unwrap_err()
            .to_string()
            .contains("per-file display byte limit")
    );
    budget.consumed = MAX_REQUEST_BYTES - 4;
    assert!(
        budget
            .read(&path, 5)
            .unwrap_err()
            .to_string()
            .contains("total display byte limit")
    );
    assert!(absolute_lexical(Path::new("../outside.json")).is_err());
}

#[cfg(windows)]
#[test]
fn ordinary_verbatim_disk_paths_match_without_changing_device_or_literal_semantics() {
    assert_eq!(
        absolute_lexical(Path::new(r"\\?\C:\work\models\candidate.json")).unwrap(),
        absolute_lexical(Path::new(r"C:\work\models\candidate.json")).unwrap()
    );
    for refused in [
        r"\\?\C:\work\..\secret.json",
        r"\\?\C:\work\literal/name.json",
        r"\\?\C:\work\trailing.\report.json",
        r"\\?\C:\work\NUL",
    ] {
        assert!(absolute_lexical(Path::new(refused)).is_err(), "{refused}");
    }
    assert_ne!(
        absolute_lexical(Path::new(r"\\?\UNC\server\share\report.json")).unwrap(),
        absolute_lexical(Path::new(r"\\server\share\report.json")).unwrap()
    );
    assert_ne!(
        absolute_lexical(Path::new(r"\\.\C:\work\report.json")).unwrap(),
        absolute_lexical(Path::new(r"C:\work\report.json")).unwrap()
    );
}
