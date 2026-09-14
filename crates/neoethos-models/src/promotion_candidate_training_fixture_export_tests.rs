//! Manual, private bridge fixture for App saved-report/HTTP tests. The real
//! installer seals deliberately unloadable model bytes; no training or inference
//! happens here. Never use this output as promotion or execution evidence.

use super::*;
use crate::runtime::feature_input::{ModelFeatureInputV1, write_model_feature_input_v1};
use neoethos_search::data_selection::CanonicalSearchArtifactScopeV2;
use neoethos_search::live_portfolio::{LivePortfolioArtifact, LiveSizingEvidenceV1};
use neoethos_search::validation::ForwardTestValidationArtifactFile;
use std::io::Write;

fn write_new(path: &Path, bytes: &[u8]) {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("fixture output must not replace an existing file");
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

#[test]
#[ignore = "manual fixture producer only; requires NEOETHOS_TEST_POLICY2_PORTFOLIO and writes one new temporary bundle"]
fn export_installed_candidate_fixture_to_temporary_directory() {
    let source = PathBuf::from(
        std::env::var_os("NEOETHOS_TEST_POLICY2_PORTFOLIO")
            .expect("first run Search export_candidate_policy2_fixture_to_temporary_directory"),
    );
    assert!(fs::metadata(&source).unwrap().len() < 256 * 1024);
    let mut portfolio = neoethos_search::live_portfolio::load_live_portfolio_json(&source)
        .expect("read genuine private Search policy2 writer fixture");
    assert_eq!(portfolio.schema_version, 6);
    assert_eq!(portfolio.live_trading_policy.schema_version, 2);
    assert_eq!(portfolio.genes.len(), 1);
    assert_eq!(portfolio.sizing_evidence.len(), portfolio.genes.len());
    assert_eq!(
        portfolio.genes[0].strategy_id,
        "synthetic-candidate-install-fixture-not-traded"
    );
    let original = portfolio.search_scope.receipt();
    assert!(matches!(
        original.validate().unwrap().scope(),
        neoethos_data::CanonicalDatasetScope::External { source_namespace }
            if source_namespace == "embedded-ctrader-fixture-unverified"
    ));
    // Existing Models fixtures use explicit synthetic g1 generation labels.
    // Only the unverified fixture is canonicalized; no production hashes are
    // invented and all affected scopes/evidence are rebuilt by constructors.
    let mut receipt_wire = serde_json::to_value(exact_receipt()).unwrap();
    receipt_wire["feature_execution"]["compute_policy"] = "cpu_only".into();
    receipt_wire["feature_execution"]["selected_lane"] = "cpu_scalar".into();
    let receipt =
        CanonicalSearchInputReceiptV2::from_json_bytes(&serde_json::to_vec(&receipt_wire).unwrap())
            .unwrap();
    assert_eq!(
        original.feature_plan_identity(),
        receipt.feature_plan_identity()
    );
    let rebind = |scope: &CanonicalSearchArtifactScopeV2| {
        CanonicalSearchArtifactScopeV2::new(receipt.clone(), scope.evaluated_window().clone())
            .unwrap()
    };
    portfolio.search_scope = rebind(&portfolio.search_scope);
    portfolio.final_holdout_scope = rebind(&portfolio.final_holdout_scope);
    portfolio.sizing_evidence = portfolio
        .sizing_evidence
        .iter()
        .zip(&portfolio.genes)
        .map(|(evidence, gene)| LiveSizingEvidenceV1 {
            forward_test: ForwardTestValidationArtifactFile::new(
                rebind(evidence.forward_test.scope()),
                &portfolio.search_config_hash,
                gene,
                evidence.forward_test.summary().clone(),
            )
            .unwrap(),
        })
        .collect();
    portfolio.validate().unwrap();

    let evaluation = portfolio
        .live_trading_policy
        .sealed_evaluation_config()
        .unwrap();
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: &portfolio.symbol,
            account_currency: &evaluation.account_currency,
            assumption_source_id: "synthetic-installed-candidate-transport-fixture-not-broker-truth",
            assumption_source_sha256: &"5".repeat(64),
            pip_size: evaluation.pip_value,
            pip_value_per_lot: evaluation.pip_value_per_lot,
            full_spread_pips_assumption: evaluation.spread_pips,
            slippage_pips_per_fill_assumption: 0.0,
            commission_account_per_lot_per_fill_assumption: evaluation.commission_per_trade / 2.0,
            swap_long_pips_per_day: evaluation.swap_long_pips_per_day,
            swap_short_pips_per_day: evaluation.swap_short_pips_per_day,
            pnl_conversion_fee_rate: evaluation.pnl_conversion_fee_rate,
        },
    )
    .unwrap();
    let mut settings = Settings::default();
    settings.system.enable_gpu_preference = "cpu".to_owned();
    settings.models.label_horizon_bars = 3;
    settings.models.ml_models = vec!["bayes_logit".to_owned()];
    settings.models.phase5_core_models.clear();
    settings.models.regime_router_enabled = false;
    settings.models.phase5_filter_meta_blender = false;
    settings.models.calibration_enabled = false;
    settings.models.use_sac_agent = false;
    settings.models.use_rl_agent = false;
    settings.models.use_neuroevolution = false;
    let selected = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
        exact_series(&receipt),
        contract,
        &portfolio,
        &settings,
    )
    .unwrap();
    let names = selected.training_config().planned_models().to_vec();
    assert_eq!(
        names.len(),
        2,
        "keep the fixture to one requested model plus regime"
    );
    let requested = names.iter().map(String::as_str).collect::<Vec<_>>();
    let handoff_bytes = selected.to_json_bytes().unwrap();
    let handoff_identity = selected.identity_sha256().unwrap();
    let root = TestRoot::new("saved-research-export");
    let staging = root.staging("saved-research");
    write_model_tree(&staging, &requested);

    let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    assert_eq!(raw.names.len(), 2);
    let fit_end = raw
        .timestamps
        .partition_point(|timestamp| *timestamp < selected.oos_cutoff_ms())
        .checked_sub(selected.purge_bars())
        .unwrap();
    assert_eq!(fit_end, 77);
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
        &raw,
        0..fit_end,
        None,
    )
    .unwrap();
    let input = ModelFeatureInputV1::from_training_frame(
        frame.provenance().bindings()[0].dataset_identity(),
        &frame,
        Some(selected.oos_cutoff_ms()),
        selected.purge_bars(),
    )
    .unwrap();
    let mut input_wire = serde_json::to_value(input).unwrap();
    input_wire["producer_receipt"]["source_bindings"] =
        serde_json::to_value(receipt.source_bindings()).unwrap();
    input_wire["producer_receipt"]["feature_execution"]["compute_policy"] = "cpu_only".into();
    input_wire["producer_receipt"]["feature_execution"]["selected_lane"] = "cpu_scalar".into();
    let input =
        ModelFeatureInputV1::from_json_bytes(&serde_json::to_vec(&input_wire).unwrap()).unwrap();
    input.validate_for_handoff(&selected).unwrap();
    write_model_feature_input_v1(&staging, &input).unwrap();

    let manifest = installed_manifest(install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        selected,
        &complete_summary(&requested),
    ));
    manifest.verify_installed(root.path()).unwrap();
    let reopened = manifest.reopen_handoff(root.path()).unwrap();
    assert_eq!(reopened.identity_sha256().unwrap(), handoff_identity);
    let restored: LivePortfolioArtifact = reopened
        .locked_portfolio()
        .deserialize_live_portfolio()
        .unwrap();
    restored
        .live_trading_policy
        .sealed_evaluation_config()
        .unwrap();
    write_new(&root.path().join("training-handoff.json"), &handoff_bytes);
    write_new(
        &root.path().join("manifest.json"),
        &serde_json::to_vec(&manifest).unwrap(),
    );
    write_new(&root.path().join("README.md"), format!(
        "# Synthetic installed candidate transport fixture\n\nGenerated by the private Models ignored test; actual V6 decoding, handoff constructor, model-input writer and atomic candidate installer. Two embedded features, 100 source bars, training/calibration/final 80/10/10, model fit 0..77.\n\nModel payloads and evaluation/selection evidence are synthetic and deliberately not inference-ready. No training, inference, final evaluation or broker execution occurred. Never use as financial/promotion evidence.\n\ntraining-handoff.json: exact serialized handoff {handoff_identity}\nmanifest.json: actual installer manifest\n{}/: actual content-addressed installed tree including handoff evidence, two dummy model files/profiles and model_feature_input.v1.json.\n\nApp tests copy the bundle to a new owned root and generate synthetic report/journal fixtures separately. No test may silently regenerate or bless fixture bytes.\n",
        manifest.candidate_relative_dir(),
    ).as_bytes());
    println!("INSTALLED_CANDIDATE_FIXTURE_ROOT={}", root.path().display());
    println!("INSTALLED_CANDIDATE_FIXTURE_HANDOFF={handoff_identity}");
    println!(
        "INSTALLED_CANDIDATE_FIXTURE_TREE={}",
        manifest.candidate_relative_dir()
    );
    // Explicit ignored producer intentionally retains ONLY the newly owned temp
    // directory for reviewed copying; failed production cleans it via TestRoot.
    std::mem::forget(root);
}
