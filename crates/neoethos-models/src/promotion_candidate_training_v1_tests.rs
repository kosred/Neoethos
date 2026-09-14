use super::promotion_candidate_training_v1::{
    MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1, PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1,
    PromotionCandidateBrokerAuthorityIdentityV1, PromotionCandidateLockedPortfolioV1,
    PromotionCandidateTrainingConfigIdentityV1, PromotionCandidateTrainingHandoffV1,
    PromotionCandidateTrainingRefusalCodeV1, PromotionCandidateTrainingRefusalV1,
    PromotionCandidateTrainingTerminalV1, install_promotion_candidate_model_tree_v1,
    resolve_promotion_candidate_training_config_identity_v1,
};
use crate::{ModelTrainingFailure, TrainingRunSummary};
use neoethos_core::Settings;
use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe,
    SelectedDatasetGenerationV1,
};
use neoethos_search::{
    CanonicalSearchInputReceiptV2, CanonicalTrendbarResearchCostAssumptionsV2,
    CanonicalTrendbarResearchExecutionContractV3,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

#[path = "promotion_candidate_training_fixture_export_tests.rs"]
mod fixture_export;

static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

struct TestRoot(PathBuf);

impl TestRoot {
    fn new(label: &str) -> Self {
        let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-promotion-candidate-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated candidate root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn staging(&self, label: &str) -> PathBuf {
        self.0
            .join(format!(".promotion-candidate.tmp-v1-test-{label}"))
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!(
                "ERROR promotion-candidate test cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

pub(crate) fn exact_receipt() -> CanonicalSearchInputReceiptV2 {
    exact_receipt_for_frame(&neoethos_data::test_fixtures::ctrader_sample_feature_frame())
}

fn exact_receipt_for_frame(
    features: &neoethos_data::FeatureFrame,
) -> CanonicalSearchInputReceiptV2 {
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, features)
        .expect("build fixture search receipt");
    let mut wire = serde_json::to_value(receipt).expect("encode fixture receipt");
    wire["source_bindings"][0]["generation_id"] =
        serde_json::Value::String(format!("g1-{}.vortex", "1".repeat(64)));
    wire["source_bindings"][0]["manifest_sha256"] = serde_json::Value::String("2".repeat(64));
    wire["source_bindings"][0]["vortex_sha256"] = serde_json::Value::String("1".repeat(64));
    CanonicalSearchInputReceiptV2::from_json_bytes(
        &serde_json::to_vec(&wire).expect("encode canonicalized fixture receipt"),
    )
    .expect("decode canonicalized fixture receipt")
}

pub(crate) fn exact_series(
    receipt: &CanonicalSearchInputReceiptV2,
) -> CanonicalDatasetSeriesReceiptV1 {
    let direct = receipt
        .source_bindings()
        .iter()
        .map(|binding| {
            SelectedDatasetGenerationV1::new(
                CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
                    .expect("decode fixture identity"),
                binding.generation_id(),
                binding.manifest_sha256(),
            )
            .expect("build selected fixture generation")
        })
        .collect::<Vec<_>>();
    let anchor_identity = receipt.validate().expect("validate fixture receipt");
    let anchor = direct
        .iter()
        .find(|selected| selected.identity() == &anchor_identity)
        .expect("fixture series contains anchor")
        .clone();
    CanonicalDatasetSeriesReceiptV1::new(anchor, direct).expect("build fixture series")
}

fn cutoff_after_receipt(receipt: &CanonicalSearchInputReceiptV2) -> i64 {
    receipt
        .source_bindings()
        .iter()
        .flat_map(|binding| binding.segments())
        .map(|segment| segment.timestamp_end_ms())
        .max()
        .expect("fixture receipt has a segment")
        .checked_add(1)
        .expect("fixture cutoff does not overflow")
}

fn config(planned_models: &[&str]) -> PromotionCandidateTrainingConfigIdentityV1 {
    PromotionCandidateTrainingConfigIdentityV1::checked_new(
        "3".repeat(64),
        "4".repeat(64),
        planned_models
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
    )
    .expect("valid fixture training config identity")
}

fn locked_portfolio() -> PromotionCandidateLockedPortfolioV1 {
    PromotionCandidateLockedPortfolioV1::from_serializable(&serde_json::json!({
        "schema": "neoethos.autoresearch.promotion-portfolio.v5",
        "session_id": "fixture-session",
        "sweep": 3,
        "slot": 7,
        "config_hash": "6".repeat(64),
        "batch_bindings": [{
            "ordinal": 0,
            "cursor": 0,
            "genes": [{"generation": 0, "strategy_id": "fixture-finalist"}]
        }]
    }))
    .expect("valid bounded locked portfolio")
}

pub(crate) fn handoff(planned_models: &[&str]) -> PromotionCandidateTrainingHandoffV1 {
    handoff_with_config(config(planned_models), 7)
}

fn handoff_with_config(
    config: PromotionCandidateTrainingConfigIdentityV1,
    purge_bars: usize,
) -> PromotionCandidateTrainingHandoffV1 {
    try_handoff_with_config(config, purge_bars).expect("valid promotion-candidate handoff")
}

fn try_handoff_with_config(
    config: PromotionCandidateTrainingConfigIdentityV1,
    purge_bars: usize,
) -> Result<PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingRefusalV1> {
    try_handoff_with_receipt(config, purge_bars, exact_receipt())
}

fn try_handoff_with_receipt(
    config: PromotionCandidateTrainingConfigIdentityV1,
    purge_bars: usize,
    receipt: CanonicalSearchInputReceiptV2,
) -> Result<PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingRefusalV1> {
    let series = exact_series(&receipt);
    let cutoff = cutoff_after_receipt(&receipt);
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: series.anchor().identity().symbol_name(),
            account_currency: "USD",
            assumption_source_id: "neoethos.test.promotion-candidate.v1",
            assumption_source_sha256: &"5".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.0,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .expect("valid fixture screening contract");
    PromotionCandidateTrainingHandoffV1::checked_new(
        series,
        CanonicalTimeframe::M1,
        receipt,
        contract,
        locked_portfolio(),
        cutoff,
        purge_bars,
        PromotionCandidateBrokerAuthorityIdentityV1::checked_new("7".repeat(64))
            .expect("valid broker authority identity"),
        config,
    )
}

fn complete_summary(models: &[&str]) -> TrainingRunSummary {
    TrainingRunSummary {
        planned_models: models.iter().map(|name| (*name).to_owned()).collect(),
        completed_models: models.iter().map(|name| (*name).to_owned()).collect(),
        failed_models: Vec::new(),
    }
}

fn write_model_tree(staging: &Path, models: &[&str]) {
    for (ordinal, model) in models.iter().enumerate() {
        let model_dir = staging.join("EURUSD").join("M1").join(model);
        fs::create_dir_all(&model_dir).expect("create fixture model dir");
        fs::write(
            model_dir.join("model.bin"),
            format!("deterministic-model-{ordinal}").as_bytes(),
        )
        .expect("write fixture model");
        fs::write(
            model_dir.join("training_profile.json"),
            format!(r#"{{"model":"{model}","cutoff":1700000000000}}"#).as_bytes(),
        )
        .expect("write fixture profile");
    }
    write_fixture_model_input(staging, 7);
}

fn write_fixture_model_input(staging: &Path, purge_bars: usize) {
    let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
        &raw,
        0..raw.n_samples().checked_sub(purge_bars).unwrap(),
        None,
    )
    .unwrap();
    write_fixture_model_input_for_frame(staging, purge_bars, &frame, &exact_receipt());
}

fn write_fixture_model_input_for_frame(
    staging: &Path,
    purge_bars: usize,
    frame: &neoethos_data::FeatureFrame,
    receipt: &CanonicalSearchInputReceiptV2,
) -> crate::runtime::feature_input::ModelFeatureInputV1 {
    use crate::runtime::feature_input::{ModelFeatureInputV1, write_model_feature_input_v1};
    let anchor = frame.provenance().bindings()[0].dataset_identity();
    let contract = ModelFeatureInputV1::from_training_frame(
        anchor,
        frame,
        Some(cutoff_after_receipt(receipt)),
        purge_bars,
    )
    .unwrap();
    // Match the explicitly synthetic canonical generation labels used by this
    // install fixture; actual producer fit and row/cutoff proof stay unchanged.
    let mut wire = serde_json::to_value(contract).unwrap();
    wire["producer_receipt"]["source_bindings"] =
        serde_json::to_value(receipt.source_bindings()).unwrap();
    let contract =
        ModelFeatureInputV1::from_json_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap();
    write_model_feature_input_v1(staging, &contract).unwrap();
    contract
}

fn installed_manifest(
    terminal: PromotionCandidateTrainingTerminalV1,
) -> super::promotion_candidate_training_v1::PromotionCandidateTrainingManifestV1 {
    match terminal {
        PromotionCandidateTrainingTerminalV1::Installed(manifest) => manifest,
        other => panic!("expected Installed terminal, got {other:?}"),
    }
}

#[test]
fn candidate_install_requires_its_own_exact_model_preprocessing_contract() {
    let root = TestRoot::new("missing-model-input");
    let staging = root.staging("missing");
    write_model_tree(&staging, &["alpha"]);
    let contract_path = staging.join("EURUSD/M1/model_feature_input.v1.json");
    fs::remove_file(&contract_path).unwrap();
    let terminal = install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        handoff(&["alpha"]),
        &complete_summary(&["alpha"]),
    );
    let PromotionCandidateTrainingTerminalV1::Refused(error) = terminal else {
        panic!("missing preprocessing must not install a candidate");
    };
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch
    );
    assert!(error.detail().contains("model preprocessing"));
    assert!(!staging.exists());

    let staging = root.staging("wrong-purge");
    write_model_tree(&staging, &["alpha"]);
    write_fixture_model_input(&staging, 6);
    let terminal = install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        handoff(&["alpha"]),
        &complete_summary(&["alpha"]),
    );
    let PromotionCandidateTrainingTerminalV1::Refused(error) = terminal else {
        panic!("a different fitted training boundary must not install");
    };
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch
    );
    assert!(error.detail().contains("cutoff/purge"));
}

#[test]
fn candidate_ensemble_reopens_the_selected_tree_before_any_legacy_model_loading() {
    let root = TestRoot::new("ensemble-tamper");
    let staging = root.staging("tamper");
    write_model_tree(&staging, &["alpha"]);
    let manifest = installed_manifest(install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        handoff(&["alpha"]),
        &complete_summary(&["alpha"]),
    ));
    fs::write(
        root.path()
            .join(manifest.candidate_relative_dir())
            .join("EURUSD/M1/alpha/model.bin"),
        b"changed model",
    )
    .unwrap();
    let result = crate::ensemble_inference::bootstrap::build_ensemble_for_candidate(
        root.path(),
        &manifest,
        &Settings::default(),
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("changed candidate must not load"),
    };
    assert!(
        format!("{error:#}").contains("installed candidate tree hash/count/bytes changed"),
        "{error:#}"
    );
    let inference = crate::ensemble_inference::bootstrap::build_ensemble_for_candidate_inference(
        root.path(),
        &manifest,
        &Settings::default(),
    );
    let error = match inference {
        Err(error) => error,
        Ok(_) => panic!("inference must reject a changed candidate before loading"),
    };
    assert!(
        format!("{error:#}").contains("installed candidate tree hash/count/bytes changed"),
        "{error:#}"
    );
}

#[test]
fn candidate_ensemble_rejects_saved_but_unloadable_models_instead_of_using_a_partial_set() {
    let mut settings = Settings::default();
    // This 100-row serialization fixture needs a nonempty purged fit before it
    // can reach the deliberately unloadable model bytes under test.
    settings.models.label_horizon_bars = 7;
    settings.models.ml_models = vec!["bayes_logit".into()];
    settings.models.phase5_core_models.clear();
    settings.models.regime_router_enabled = false;
    settings.models.phase5_filter_meta_blender = false;
    settings.models.calibration_enabled = false;
    settings.models.use_sac_agent = false;
    settings.models.use_rl_agent = false;
    settings.models.use_neuroevolution = false;
    let config = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let names = config.planned_models().to_vec();
    assert_eq!(
        names.len(),
        2,
        "the fixture should request only bayes_logit and the regime model: {names:?}"
    );
    let requested = names.iter().map(String::as_str).collect::<Vec<_>>();
    let root = TestRoot::new("ensemble-unloadable");
    let staging = root.staging("unloadable");
    write_model_tree(&staging, &requested);
    write_fixture_model_input(
        &staging,
        crate::training_orchestrator::effective_label_horizon_bars_v1(&settings),
    );
    let manifest = installed_manifest(install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        handoff_with_config(
            config,
            crate::training_orchestrator::effective_label_horizon_bars_v1(&settings),
        ),
        &complete_summary(&requested),
    ));
    // Correct tree identity does not imply correct model serialization. The
    // actual registered loaders must reject these deliberate fixture bytes.
    let result = crate::ensemble_inference::bootstrap::build_ensemble_for_candidate(
        root.path(),
        &manifest,
        &settings,
    );
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("unloadable models must not become an ensemble"),
    };
    assert!(
        format!("{error:#}").contains("selected candidate inference inventory"),
        "{error:#}"
    );
    let mut relocated = settings.clone();
    relocated.system.data_dir = root.path().join("relocated-data");
    relocated.system.cache_dir = root.path().join("relocated-cache");
    let inference = crate::ensemble_inference::bootstrap::build_ensemble_for_candidate_inference(
        root.path(),
        &manifest,
        &relocated,
    );
    let error = match inference {
        Err(error) => error,
        Ok(_) => panic!("relocation must not make unloadable models an ensemble"),
    };
    assert!(
        format!("{error:#}").contains("selected candidate inference inventory"),
        "inference must reach the same strict registered-loader checks: {error:#}"
    );
    manifest
        .verify_installed(root.path())
        .expect("loading must leave the saved candidate untouched");
    let input_path = root
        .path()
        .join(manifest.candidate_relative_dir())
        .join("EURUSD/M1")
        .join(crate::runtime::feature_input::MODEL_FEATURE_INPUT_FILE_V1);
    fs::remove_file(input_path).unwrap();
    let missing_input =
        crate::ensemble_inference::bootstrap::build_ensemble_for_candidate_inference(
            root.path(),
            &manifest,
            &relocated,
        );
    let error = match missing_input {
        Err(error) => error,
        Ok(_) => panic!("inference must not reconstruct a missing saved input recipe"),
    };
    assert!(
        format!("{error:#}").contains("installed candidate tree hash/count/bytes changed"),
        "the saved model-owned input must remain mandatory after relocation: {error:#}"
    );
}

#[test]
fn handoff_rejects_generation_drift_and_any_search_row_at_the_oos_cutoff() {
    let receipt = exact_receipt();
    let mut series_wire = serde_json::to_value(exact_series(&receipt)).expect("encode series");
    series_wire["anchor"]["generation_id"] =
        serde_json::Value::String(format!("g1-{}.vortex", "8".repeat(64)));
    series_wire["direct_timeframes"][0]["generation_id"] =
        serde_json::Value::String(format!("g1-{}.vortex", "8".repeat(64)));
    let drifted_series = CanonicalDatasetSeriesReceiptV1::from_json_bytes(
        &serde_json::to_vec(&series_wire).expect("encode drifted series"),
    )
    .expect("drifted series is internally valid");
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt.clone(),
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.promotion-candidate.v1",
            assumption_source_sha256: &"5".repeat(64),
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.0,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .expect("valid fixture contract");
    let cutoff = cutoff_after_receipt(&receipt);
    let error = PromotionCandidateTrainingHandoffV1::checked_new(
        drifted_series,
        CanonicalTimeframe::M1,
        receipt.clone(),
        contract.clone(),
        locked_portfolio(),
        cutoff,
        7,
        PromotionCandidateBrokerAuthorityIdentityV1::checked_new("7".repeat(64)).unwrap(),
        config(&["alpha"]),
    )
    .expect_err("search receipt must not bind a replacement current generation");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch
    );

    let error = PromotionCandidateTrainingHandoffV1::checked_new(
        exact_series(&receipt),
        CanonicalTimeframe::M1,
        receipt,
        contract,
        locked_portfolio(),
        cutoff - 1,
        7,
        PromotionCandidateBrokerAuthorityIdentityV1::checked_new("7".repeat(64)).unwrap(),
        config(&["alpha"]),
    )
    .expect_err("a search row at the cutoff leaks into final OOS");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
    );
}

#[test]
fn handoff_identity_is_deterministic_and_runtime_or_model_config_drift_is_refused() {
    let first = handoff(&["alpha", "beta"]);
    let second = handoff(&["alpha", "beta"]);
    assert_eq!(
        first.identity_sha256().expect("hash first handoff"),
        second.identity_sha256().expect("hash equivalent handoff")
    );
    assert_eq!(
        first.locked_portfolio().identity_sha256(),
        neoethos_search::canonical_locked_portfolio_identity_sha256_v1(&serde_json::json!({
            "schema": "neoethos.autoresearch.promotion-portfolio.v5",
            "session_id": "fixture-session",
            "sweep": 3,
            "slot": 7,
            "config_hash": "6".repeat(64),
            "batch_bindings": [{
                "ordinal": 0,
                "cursor": 0,
                "genes": [{"generation": 0, "strategy_id": "fixture-finalist"}]
            }]
        }))
        .expect("hash exact fixture portfolio")
    );

    let runtime_drift = PromotionCandidateTrainingConfigIdentityV1::checked_new(
        "8".repeat(64),
        "4".repeat(64),
        vec!["alpha".into(), "beta".into()],
    )
    .unwrap();
    let error = first
        .validate_against_config_identity_v1(&runtime_drift)
        .expect_err("runtime-plan drift must refuse training");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch
    );

    let model_drift = PromotionCandidateTrainingConfigIdentityV1::checked_new(
        "3".repeat(64),
        "9".repeat(64),
        vec!["alpha".into(), "beta".into()],
    )
    .unwrap();
    let error = second
        .validate_against_config_identity_v1(&model_drift)
        .expect_err("model-plan drift must refuse training");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch
    );
}

#[test]
fn settings_resolution_matches_the_handoff_then_refuses_purge_config_drift() {
    let mut settings = Settings::default();
    let resolved = resolve_promotion_candidate_training_config_identity_v1(&settings)
        .expect("default configured model plan must be sealable");
    let sealed = handoff_with_config(
        resolved,
        crate::training_orchestrator::effective_label_horizon_bars_v1(&settings),
    );
    sealed
        .validate_against_settings_v1(&settings)
        .expect("unchanged effective training settings must match their handoff");
    sealed
        .validate_against_settings_v1(&settings)
        .expect("revalidation must reuse the sealed plan without volatile reprobe drift");

    let mut runtime_drift = settings.clone();
    runtime_drift.system.enable_gpu_preference =
        if runtime_drift.system.enable_gpu_preference == "cpu" {
            "auto".to_owned()
        } else {
            "cpu".to_owned()
        };
    let error = sealed
        .validate_against_settings_v1(&runtime_drift)
        .expect_err("retained runtime selection drift must refuse before training");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch
    );

    settings.models.label_horizon_bars += 1;
    let error = sealed
        .validate_against_settings_v1(&settings)
        .expect_err("purge drift after sealing must refuse before training");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch
    );

    let error = try_handoff_with_config(config(&["alpha"]), 1_000_001)
        .expect_err("purge values above the explicit cap must be refused");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff
    );
}

fn inference_settings_fixture() -> Settings {
    let mut settings = Settings::default();
    settings.system.enable_gpu_preference = "cpu".to_owned();
    settings.models.label_horizon_bars = 7;
    settings.models.ml_models = vec!["bayes_logit".into()];
    settings.models.phase5_core_models.clear();
    settings.models.regime_router_enabled = false;
    settings.models.phase5_filter_meta_blender = false;
    settings.models.calibration_enabled = false;
    settings.models.use_sac_agent = false;
    settings.models.use_rl_agent = false;
    settings.models.use_neuroevolution = false;
    settings
}

fn write_trained_cpu_fixture_sidecars(
    dir: &Path,
    settings: &Settings,
    name: &str,
    model_type: crate::parallel_trainer::ModelType,
    payload: &crate::parallel_trainer::TrainingPayload,
) {
    use crate::parallel_trainer::ModelConfig;
    use crate::runtime::capabilities::{CapabilityState, ModelFamily};
    use crate::runtime::profile::{
        TRAINING_RUNTIME_PROFILE_FILE_NAME, write_training_runtime_profile,
    };
    use crate::runtime::training_artifact::{
        write_model_runtime_artifact_contract_sidecar,
        write_training_model_artifact_contract_sidecar,
    };
    let frame = &payload.frame;
    let anchor = frame.provenance().bindings()[0].dataset_identity();
    let options = frame.feature_build_options().unwrap();
    let mut profile = crate::runtime::profile::tests::sample_profile();
    profile.model_name = name.to_owned();
    profile.capability_family = ModelFamily::Meta;
    profile.symbol = anchor.symbol_name().to_owned();
    profile.base_timeframe = anchor.timeframe().as_str().to_owned();
    profile.feature_count = frame.n_features();
    profile.dataset_rows = frame.n_samples();
    profile.row_budget_applied = None;
    profile.label_horizon_bars = settings.models.label_horizon_bars;
    profile.effective_label_horizon_bars = settings.models.label_horizon_bars;
    profile.meta_label_max_hold_bars = settings.risk.meta_label_max_hold_bars;
    profile.label_use_triple_barrier = false;
    profile.higher_timeframes = options.higher_tfs.clone();
    profile.multi_resolution_enabled = !options.higher_tfs.is_empty();
    profile.base_features_prefixed = options.prefix_base_features;
    profile.l1_feature_selection_enabled = false;
    profile.requested_backend = Some("cpu".into());
    profile.requested_device = Some("cpu".into());
    profile.planned_backend = Some("cpu".into());
    profile.planned_device = Some("cpu".into());
    profile.planned_precision = Some("fp64".into());
    profile.planned_cpu_threads = Some(1);
    profile.planned_batch_size = None;
    profile.planned_memory_budget_gb = None;
    profile.train_years = 0;
    profile.val_years = 0;
    profile.requested_hpo_backend = "none".into();
    profile.requested_hpo_trials = 1;
    profile.holdout_pct = 0.0;
    profile.embargo_minutes = 0;
    profile.notes = vec![
        "Test-only direct CPU fit on synthetic features/labels; no HPO, market or OOS claim. The expert's own metadata records its internal fit split.".into(),
    ];
    let config = ModelConfig {
        name: name.to_owned(),
        model_type,
        capability_family: ModelFamily::Meta,
        capability_state: CapabilityState::Implemented,
        params: Default::default(),
    };
    write_training_runtime_profile(&dir.join(TRAINING_RUNTIME_PROFILE_FILE_NAME), &profile)
        .unwrap();
    write_training_model_artifact_contract_sidecar(dir, settings, &config, payload, &profile)
        .unwrap();
    write_model_runtime_artifact_contract_sidecar(dir, settings, &config, payload, &profile)
        .unwrap();
}

#[test]
fn trained_cpu_candidate_inference_preserves_predictions_and_input_after_relocation() {
    use crate::base::ExpertModel as TrainingModel;
    use crate::ensemble_inference::EnsemblePredictor;
    use crate::ensemble_inference::bootstrap::build_ensemble_for_candidate_inference;
    use crate::forecasting::hmm_regime::{HmmRegimeConfig, RegimeHmmExpert};
    use crate::parallel_trainer::{ModelType, TrainingPayload};
    use crate::statistical::bayesian_impl::BayesianLogitExpert;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};

    // Loader/serialization proof only: this existing generic diagnostic handoff
    // is not an App V6 candidate, untouched OOS result or deployment permission.
    const ROWS: usize = 700;
    const PURGE: usize = 3;
    const FIT_ROWS: usize = ROWS - PURGE;
    const PROBE_ROWS: usize = 32;
    let root = TestRoot::new("trained-cpu-inference");
    let relocated_root = TestRoot::new("trained-cpu-inference-relocated");
    let staging = root.staging("real-cpu-models");
    let mut settings = inference_settings_fixture();
    settings.models.label_horizon_bars = PURGE;
    settings.system.hardware.cpu_budget = Some(1);
    settings.system.data_dir = root.path().join("data");
    settings.system.cache_dir = root.path().join("cache");
    let config = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let planned = config.planned_models().to_vec();
    let mut inventory = planned.clone();
    inventory.sort();
    assert_eq!(inventory, ["bayes_logit", "hmm_regime"]);
    let requested: Vec<_> = planned.iter().map(String::as_str).collect();

    // Reuse the existing HMM adapter's bounded numerical fixture, with its two
    // actual required column names and an explicit canonical test-time grid.
    let columns = [
        (
            "quant_log_return",
            (0..ROWS)
                .map(|row| ((row % 7) as f64 - 3.0) * 0.0005)
                .collect(),
        ),
        (
            "quant_log_volatility",
            (0..ROWS).map(|row| -7.0 + (row % 5) as f64 * 0.1).collect(),
        ),
    ]
    .into_iter()
    .map(|(name, values)| {
        FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; ROWS]).unwrap()
    })
    .collect();
    let raw = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        neoethos_data::test_fixtures::canonical_test_timestamps(ROWS),
        columns,
    )
    .unwrap();
    let receipt = exact_receipt_for_frame(&raw);
    let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
        &raw,
        0..FIT_ROWS,
        None,
    )
    .unwrap();
    let input = write_fixture_model_input_for_frame(&staging, PURGE, &frame, &receipt);
    let handoff = try_handoff_with_receipt(config, PURGE, receipt).unwrap();
    input.validate_for_handoff(&handoff).unwrap();
    let handoff_bytes = handoff.to_json_bytes().unwrap();
    let input_bytes = input.to_json_bytes().unwrap();
    let payload = TrainingPayload::from_frame(
        frame.row_window(0, FIT_ROWS).unwrap(),
        (0..FIT_ROWS).map(|row| (row % 3) as i32 - 1).collect(),
    )
    .unwrap();
    assert_eq!(payload.frame.n_samples(), FIT_ROWS);
    assert_eq!(payload.frame.n_features(), 2);
    let width = WorkerLimit::new(1).unwrap();
    let broker = CpuPermitBroker::new(width);
    let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
    assert_eq!(
        crate::statistical::common::statistical_device_policy("bayes_logit"),
        "cpu"
    );

    let mut bayes = BayesianLogitExpert::new();
    assert_eq!(bayes.epochs, 250);
    bayes.fit(&payload.frame, &payload.labels, &lease).unwrap();
    let observations =
        RegimeHmmExpert::training_observations_from_feature_frame(&payload.frame, &lease).unwrap();
    let hmm_config = HmmRegimeConfig::default();
    assert_eq!(hmm_config.min_training_bars, 500);
    assert_eq!(hmm_config.max_em_iterations, 50);
    let hmm = lease
        .scope(|| RegimeHmmExpert::train(&observations, payload.frame.names.clone(), hmm_config))
        .unwrap();
    let probe = frame.row_window(ROWS - PROBE_ROWS, ROWS).unwrap();
    let expected_bayes = bayes.predict_proba(&probe, &lease).unwrap();
    let expected_hmm = hmm.predict_feature_frame(&probe, &lease).unwrap();
    assert_eq!(expected_bayes.dim(), (PROBE_ROWS, 3));
    assert_eq!(expected_hmm.probabilities.dim(), (PROBE_ROWS, 3));
    assert!(expected_hmm.validity.iter().all(|value| value.is_valid()));

    let bayes_dir = staging.join("EURUSD/M1/bayes_logit");
    let hmm_dir = staging.join("EURUSD/M1/hmm_regime");
    bayes.save(&bayes_dir).unwrap();
    hmm.save_to_path(&hmm_dir).unwrap();
    for (dir, name, kind) in [
        (&bayes_dir, "bayes_logit", ModelType::BayesianLogit),
        (&hmm_dir, "hmm_regime", ModelType::HmmRegime),
    ] {
        write_trained_cpu_fixture_sidecars(dir, &settings, name, kind, &payload);
    }
    let manifest = installed_manifest(install_promotion_candidate_model_tree_v1(
        root.path(),
        &staging,
        handoff,
        &complete_summary(&requested),
    ));
    manifest.verify_installed(root.path()).unwrap();
    let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
    let tree_name = manifest.candidate_relative_dir().to_owned();

    // The same fitted state is replayed, not estimated again on prediction rows.
    let replayed = Arc::new(
        neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
            &raw,
            0..FIT_ROWS,
            input.producer_receipt().normalization_fitted_state(),
        )
        .unwrap(),
    );
    for pass in 0..2 {
        let (candidate_root, current_settings) = if pass == 0 {
            (root.path(), settings.clone())
        } else {
            let destination = relocated_root.path().join(&tree_name);
            assert!(!destination.exists());
            // Move only this test's owned installed tree, retaining every byte.
            fs::rename(root.path().join(&tree_name), &destination).unwrap();
            let mut moved = settings.clone();
            moved.system.data_dir = relocated_root.path().join("data");
            moved.system.cache_dir = relocated_root.path().join("cache");
            moved.system.hardware.cpu_budget = Some(2);
            (relocated_root.path(), moved)
        };
        let sealed = manifest.reopen_handoff(candidate_root).unwrap();
        assert_eq!(sealed.to_json_bytes().unwrap(), handoff_bytes);
        if pass == 1 {
            assert_eq!(
                sealed
                    .validate_against_settings_v1(&current_settings)
                    .unwrap_err()
                    .code(),
                PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
            );
        }
        let ensemble =
            build_ensemble_for_candidate_inference(candidate_root, &manifest, &current_settings)
                .unwrap();
        let outcome = ensemble.load_outcome();
        let mut loaded = outcome.loaded_names();
        loaded.sort_unstable();
        assert_eq!(loaded, ["bayes_logit", "hmm_regime"]);
        assert!(outcome.missing.is_empty() && outcome.degraded.is_empty());
        assert_eq!(
            ensemble
                .model_feature_input()
                .unwrap()
                .to_json_bytes()
                .unwrap(),
            input_bytes
        );

        // Compare every registered expert against its pre-save prediction, not
        // only two ensemble runs that might share the same loading defect.
        for expert in &outcome.loaded {
            let expected = match expert.name() {
                "bayes_logit" => &expected_bayes,
                "hmm_regime" => &expected_hmm.probabilities,
                other => panic!("unexpected installed expert {other}"),
            };
            let actual = expert.predict(&probe, &lease).unwrap();
            assert_eq!(actual.len(), PROBE_ROWS);
            for (row, prediction) in actual.iter().enumerate() {
                prediction.validate().unwrap();
                assert!(prediction.validity.is_valid());
                assert_eq!(prediction.values.len(), 3);
                assert!((prediction.values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                for class in 0..3 {
                    assert!(
                        (prediction.values[class] - expected[(row, class)]).abs() < 1e-12,
                        "{} prediction changed after save/install/relocation at {row}/{class}",
                        expert.name(),
                    );
                }
            }
        }
        let decisions = ensemble.predict_with_roles(&probe, &lease).unwrap();
        assert_eq!(decisions.len(), PROBE_ROWS);
        for (row, decision) in decisions.iter().enumerate() {
            assert!(decision.validity.is_valid());
            assert!(
                decision
                    .dir_probs
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            );
            assert!((decision.dir_probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for class in 0..3 {
                assert!((decision.dir_probs[class] - expected_bayes[(row, class)]).abs() < 1e-12);
            }
            let posterior = &expected_hmm.probabilities;
            let trend_mass = (posterior[(row, 1)] + posterior[(row, 2)]).clamp(0.0, 1.0);
            let side = if expected_bayes[(row, 1)] >= expected_bayes[(row, 2)] {
                1
            } else {
                2
            };
            let expected_gate = (trend_mass * posterior[(row, side)]).clamp(0.0, 1.0);
            assert!(
                decision.regime_gate.is_finite() && (0.0..=1.0).contains(&decision.regime_gate)
            );
            assert!((decision.regime_gate - expected_gate).abs() < 1e-12);
            assert_eq!(decision.anomaly_scale, 1.0);
        }
        let bound = ensemble.bind_model_features(&replayed).unwrap();
        assert_eq!(
            bound.last_row(ROWS, PROBE_ROWS, &lease).unwrap(),
            decisions[PROBE_ROWS - 1]
        );
        manifest.verify_installed(candidate_root).unwrap();
        assert_eq!(serde_json::to_vec(&manifest).unwrap(), manifest_bytes);
    }
    assert!(!root.path().join(tree_name).exists());
}

#[test]
fn inference_settings_allow_runtime_relocation_without_changing_training_identity() {
    let settings = inference_settings_fixture();
    let resolved = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let sealed = handoff_with_config(resolved, 7);
    let before = sealed.to_json_bytes().unwrap();
    sealed.validate_against_settings_v1(&settings).unwrap();
    sealed.validate_inference_settings_v1(&settings).unwrap();

    let mut relocated = settings.clone();
    relocated.system.data_dir = PathBuf::from("relocated-inference-data");
    relocated.system.cache_dir = PathBuf::from("relocated-inference-cache");
    relocated.system.hardware.cpu_budget = Some(1);
    assert_eq!(
        sealed
            .validate_against_settings_v1(&relocated)
            .unwrap_err()
            .code(),
        PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
        "training and combined research must retain exact runtime binding"
    );
    sealed.validate_inference_settings_v1(&relocated).unwrap();
    sealed.validate_inference_settings_v1(&relocated).unwrap();
    assert_eq!(
        sealed.to_json_bytes().unwrap(),
        before,
        "inference compatibility must not reseal the original training provenance"
    );
}

#[test]
fn inference_settings_reject_model_parameters_inventory_and_purge_drift() {
    let settings = inference_settings_fixture();
    let resolved = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let sealed = handoff_with_config(resolved, 7);

    let mut parameters = settings.clone();
    parameters
        .models
        .model_param_overrides
        .entry("bayes_logit".to_owned())
        .or_default()
        .insert("__hpo_trials".to_owned(), "123456".to_owned());
    let mut inventory = settings.clone();
    inventory.models.ml_models.push("lightgbm".to_owned());
    let mut purge = settings.clone();
    purge.models.label_horizon_bars = 8;
    for (kind, changed) in [
        ("parameters", parameters),
        ("inventory", inventory),
        ("purge", purge),
    ] {
        assert_eq!(
            sealed
                .validate_inference_settings_v1(&changed)
                .unwrap_err()
                .code(),
            PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
            "inference must not accept changed {kind}"
        );
    }
}

#[test]
fn inference_settings_require_full_handoff_validity_and_the_sealed_hardware_plan() {
    let settings = inference_settings_fixture();
    let legacy = handoff_with_config(config(&["alpha"]), 7);
    assert_eq!(
        legacy
            .validate_inference_settings_v1(&settings)
            .unwrap_err()
            .code(),
        PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
        "missing original hardware plan must never be replaced by a current probe"
    );
    let resolved = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let sealed = handoff_with_config(resolved, 7);
    let mut wire = serde_json::to_value(&sealed).unwrap();
    wire["purge_bars"] = 0.into();
    let invalid: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(wire).unwrap();
    assert_eq!(
        invalid
            .validate_inference_settings_v1(&settings)
            .unwrap_err()
            .code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
        "inference settings cannot bypass structural handoff validation"
    );
}

#[test]
fn zero_horizon_handoff_requires_effective_purge_and_refuses_fallback_drift() {
    let mut settings = Settings::default();
    settings.models.label_horizon_bars = 0;
    settings.risk.meta_label_max_hold_bars = 3;
    let resolved = resolve_promotion_candidate_training_config_identity_v1(&settings).unwrap();
    let sealed = handoff_with_config(resolved, 3);
    sealed.validate_against_settings_v1(&settings).unwrap();
    assert_eq!(sealed.purge_bars(), 3);

    let mut no_purge = serde_json::to_value(&sealed).unwrap();
    no_purge["purge_bars"] = 0.into();
    let no_purge: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(no_purge).unwrap();
    assert_eq!(
        no_purge.identity_sha256().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
        "legacy zero-purge authority must not be silently upgraded"
    );
    assert_eq!(
        try_handoff_with_config(config(&["alpha"]), 0)
            .unwrap_err()
            .code(),
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff
    );

    settings.risk.meta_label_max_hold_bars = 4;
    assert_eq!(
        sealed
            .validate_against_settings_v1(&settings)
            .unwrap_err()
            .code(),
        PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch
    );
}

#[test]
fn locked_portfolio_payload_is_bounded_before_it_can_enter_a_handoff() {
    let oversized = "x".repeat(MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 + 1);
    let error = PromotionCandidateLockedPortfolioV1::from_serializable(&oversized)
        .expect_err("oversized locked portfolio must fail before handoff allocation");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
    );
}

#[test]
fn partial_or_failed_model_inventory_is_refused_before_any_candidate_is_visible() {
    let root = TestRoot::new("partial");
    let partial_staging = root.staging("partial");
    write_model_tree(&partial_staging, &["alpha"]);
    let terminal = install_promotion_candidate_model_tree_v1(
        root.path(),
        &partial_staging,
        handoff(&["alpha", "beta"]),
        &complete_summary(&["alpha"]),
    );
    assert!(matches!(
        terminal,
        PromotionCandidateTrainingTerminalV1::Refused(refusal)
            if refusal.code() == PromotionCandidateTrainingRefusalCodeV1::ModelInventoryIncomplete
    ));
    assert!(
        !partial_staging.exists(),
        "refused partial staging must be cleaned"
    );
    assert_eq!(
        fs::read_dir(root.path()).unwrap().count(),
        0,
        "a partial model set must never expose a candidate directory"
    );

    let failed_staging = root.staging("failed");
    write_model_tree(&failed_staging, &["alpha", "beta"]);
    let failed = TrainingRunSummary {
        planned_models: vec!["alpha".into(), "beta".into()],
        completed_models: vec!["alpha".into()],
        failed_models: vec![ModelTrainingFailure {
            name: "beta".into(),
            error: "fixture failure".into(),
        }],
    };
    let terminal = install_promotion_candidate_model_tree_v1(
        root.path(),
        &failed_staging,
        handoff(&["alpha", "beta"]),
        &failed,
    );
    assert!(matches!(
        terminal,
        PromotionCandidateTrainingTerminalV1::Refused(refusal)
            if refusal.code() == PromotionCandidateTrainingRefusalCodeV1::ModelTrainingFailed
    ));
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
}

#[test]
fn deterministic_no_replace_install_reopens_every_file_and_detects_tree_mutation() {
    let root = TestRoot::new("deterministic");
    let first_staging = root.staging("first");
    write_model_tree(&first_staging, &["alpha", "beta"]);
    let first = installed_manifest(install_promotion_candidate_model_tree_v1(
        root.path(),
        &first_staging,
        handoff(&["alpha", "beta"]),
        &complete_summary(&["alpha", "beta"]),
    ));
    first
        .verify_installed(root.path())
        .expect("fresh candidate tree must reopen exactly");
    assert_eq!(first.model_artifacts().len(), 2);
    assert_eq!(
        first.candidate_relative_dir(),
        first.candidate_tree_sha256(),
        "candidate directory must be the exact installed-tree content address"
    );
    let first_tree = first.candidate_tree_sha256().to_owned();
    let candidate_dir = root.path().join(first.candidate_relative_dir());
    assert!(
        candidate_dir
            .join(PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1)
            .is_file(),
        "the installed tree must contain its exact handoff evidence"
    );
    let reopened_handoff = first
        .reopen_handoff(root.path())
        .expect("combined OOS must be able to reopen the exact move-only handoff");
    assert_eq!(
        reopened_handoff.identity_sha256().unwrap(),
        handoff(&["alpha", "beta"]).identity_sha256().unwrap()
    );

    let second_staging = root.staging("second");
    write_model_tree(&second_staging, &["alpha", "beta"]);
    let second = match install_promotion_candidate_model_tree_v1(
        root.path(),
        &second_staging,
        handoff(&["alpha", "beta"]),
        &complete_summary(&["alpha", "beta"]),
    ) {
        PromotionCandidateTrainingTerminalV1::ExistingIdentical(manifest) => manifest,
        other => panic!("expected ExistingIdentical terminal, got {other:?}"),
    };
    assert_eq!(second.candidate_tree_sha256(), first_tree);
    assert!(!second_staging.exists());

    let model_path = candidate_dir.join("EURUSD/M1/alpha/model.bin");
    let original = fs::read(&model_path).expect("read installed model");
    fs::write(&model_path, vec![b'X'; original.len()]).expect("mutate installed model in place");
    let error = first
        .verify_installed(root.path())
        .expect_err("same-length model mutation must invalidate the manifest");
    assert_eq!(
        error.code(),
        PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged
    );

    let third_staging = root.staging("third");
    write_model_tree(&third_staging, &["alpha", "beta"]);
    let terminal = install_promotion_candidate_model_tree_v1(
        root.path(),
        &third_staging,
        handoff(&["alpha", "beta"]),
        &complete_summary(&["alpha", "beta"]),
    );
    assert!(matches!(
        terminal,
        PromotionCandidateTrainingTerminalV1::Refused(refusal)
            if refusal.code() == PromotionCandidateTrainingRefusalCodeV1::CandidateIdentityCollision
    ));
    assert_eq!(
        fs::read(&model_path).unwrap(),
        vec![b'X'; original.len()],
        "no-replace collision handling must never overwrite the existing candidate"
    );
}

#[test]
fn concurrent_identical_install_has_exactly_one_installer_and_never_replaces() {
    let root = TestRoot::new("concurrent");
    let first_staging = root.staging("race-first");
    let second_staging = root.staging("race-second");
    write_model_tree(&first_staging, &["alpha", "beta"]);
    write_model_tree(&second_staging, &["alpha", "beta"]);

    let barrier = Arc::new(Barrier::new(2));
    let launches = [first_staging, second_staging]
        .into_iter()
        .map(|staging| {
            let root = root.path().to_path_buf();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                install_promotion_candidate_model_tree_v1(
                    &root,
                    &staging,
                    handoff(&["alpha", "beta"]),
                    &complete_summary(&["alpha", "beta"]),
                )
            })
        })
        .collect::<Vec<_>>();
    let terminals = launches
        .into_iter()
        .map(|thread| thread.join().expect("installer thread must not panic"))
        .collect::<Vec<_>>();

    assert_eq!(
        terminals
            .iter()
            .filter(|terminal| matches!(
                terminal,
                PromotionCandidateTrainingTerminalV1::Installed(_)
            ))
            .count(),
        1,
        "atomic no-replace publication must elect exactly one installer"
    );
    assert_eq!(
        terminals
            .iter()
            .filter(|terminal| {
                matches!(
                    terminal,
                    PromotionCandidateTrainingTerminalV1::ExistingIdentical(_)
                )
            })
            .count(),
        1,
        "the losing identical publisher must verify and reuse the winner"
    );
    assert_eq!(
        fs::read_dir(root.path()).unwrap().count(),
        1,
        "only the one content-addressed candidate directory may remain"
    );
}

#[test]
fn handoff_and_manifest_are_bounded_non_clone_contracts() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = [
        crate_root.join("src/promotion_candidate_training_v1.rs"),
        crate_root.join("src/promotion_candidate_training_v1/install.rs"),
    ]
    .into_iter()
    .map(|path| fs::read_to_string(&path).expect("read promotion-candidate source"))
    .collect::<Vec<_>>()
    .join("\n");
    for type_name in [
        "PromotionCandidateTrainingHandoffV1",
        "PromotionCandidateTrainingManifestV1",
    ] {
        let declaration = format!("pub struct {type_name}");
        let offset = source.find(&declaration).expect("type declaration exists");
        let prefix = &source[..offset];
        let attributes = prefix.rsplit_once("\n\n").map_or(prefix, |(_, tail)| tail);
        assert!(
            !attributes.contains("Clone")
                && !source.contains(&format!("impl Clone for {type_name}")),
            "{type_name} must remain move-only"
        );
    }
    for required in [
        "MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1",
        "MAX_PROMOTION_CANDIDATE_MODEL_TREE_BYTES_V1",
        "MAX_PROMOTION_CANDIDATE_MODEL_FILE_COUNT_V1",
        "renameat2",
        "RENAME_NOREPLACE",
        "MoveFileExW",
        "verify_installed",
    ] {
        assert!(source.contains(required), "source omits `{required}`");
    }
    for forbidden in [
        "write_dir_with_backup",
        "fs::rename(staging",
        "JobState::Degraded",
    ] {
        assert!(
            !source.contains(forbidden),
            "source contains forbidden `{forbidden}`"
        );
    }
}

pub(crate) fn discovery_fixture_portfolio(
    receipt: CanonicalSearchInputReceiptV2,
    effective_feature_names: Vec<String>,
    normalize_features: bool,
) -> (
    neoethos_search::live_portfolio::LivePortfolioArtifact,
    Settings,
) {
    use neoethos_search::data_selection::{
        CanonicalSearchArtifactScopeV2, CanonicalSearchEvaluatedWindowV1,
        CanonicalSearchWindowRoleV1,
    };
    use neoethos_search::live_portfolio::{
        LivePortfolioArtifact, LiveSizingEvidenceV1, LiveTradingPolicyV1,
    };
    use neoethos_search::validation::{ForwardTestSummary, ForwardTestValidationArtifactFile};

    let timestamps = neoethos_data::test_fixtures::ctrader_sample_ohlcv()
        .timestamp
        .unwrap();
    let selection = CanonicalSearchArtifactScopeV2::new(
        receipt.clone(),
        CanonicalSearchEvaluatedWindowV1::new(
            CanonicalSearchWindowRoleV1::InSample,
            0,
            80,
            timestamps[0],
            timestamps[79],
        )
        .unwrap(),
    )
    .unwrap();
    let calibration = CanonicalSearchArtifactScopeV2::new(
        receipt.clone(),
        CanonicalSearchEvaluatedWindowV1::new(
            CanonicalSearchWindowRoleV1::SelectionValidation,
            80,
            90,
            timestamps[80],
            timestamps[89],
        )
        .unwrap(),
    )
    .unwrap();
    let final_holdout_scope = CanonicalSearchArtifactScopeV2::new(
        receipt.clone(),
        CanonicalSearchEvaluatedWindowV1::new(
            CanonicalSearchWindowRoleV1::Holdout,
            90,
            100,
            timestamps[90],
            timestamps[99],
        )
        .unwrap(),
    )
    .unwrap();
    // Explicit fixture policy, hashed with the production JSON hash function.
    // Field order is part of that existing contract.
    #[derive(serde::Serialize)]
    struct PolicyBody {
        kind: &'static str,
        schema_version: u16,
        source_search_config_hash: &'static str,
        source_resolved_config_hash: &'static str,
        trailing_enabled: bool,
        trailing_be_trigger_r: f64,
        trailing_stop_multiplier: f64,
        trailing_min_lock_pips: f64,
        kill_zones_enabled: bool,
        baseline_spread_pips: f64,
        session_spread_pips: Option<[f64; 3]>,
    }
    let body = PolicyBody {
        kind: "neoethos.live-trading-policy-identity.v1",
        schema_version: 1,
        source_search_config_hash: "fnv64:0123456789abcdef",
        source_resolved_config_hash: "fnv64:fedcba9876543210",
        trailing_enabled: false,
        trailing_be_trigger_r: 1.0,
        trailing_stop_multiplier: 1.0,
        trailing_min_lock_pips: 0.0,
        kill_zones_enabled: false,
        baseline_spread_pips: 1.0,
        session_spread_pips: None,
    };
    let hash = neoethos_core::storage::json::stable_json_hash(&body).unwrap();
    let mut wire = serde_json::to_value(&body).unwrap();
    wire.as_object_mut().unwrap().remove("kind");
    wire["identity_hash"] = hash.into();
    let policy: LiveTradingPolicyV1 = serde_json::from_value(wire).unwrap();
    let gene = neoethos_search::Gene {
        strategy_id: "handoff-fixture".to_owned(),
        indices: vec![0],
        weights: vec![1.0],
        ..Default::default()
    };
    let evidence = ForwardTestValidationArtifactFile::new(
        calibration.clone(),
        body.source_search_config_hash,
        &gene,
        ForwardTestSummary {
            bars: 10,
            metrics: neoethos_search::eval::BacktestMetrics::from_metric_array([
                1.0, 1.0, 100_001.0, 0.01, 0.55, 1.5, 1.0, 0.5, 1.0, 0.8, 0.005,
            ]),
            span_days: 1.0,
        },
    )
    .unwrap();
    let higher_tfs = receipt
        .source_bindings()
        .iter()
        .filter(|binding| binding.dataset_identity() != receipt.anchor_dataset_identity())
        .map(|binding| {
            CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
                .unwrap()
                .timeframe()
                .as_str()
                .to_owned()
        })
        .collect();
    let portfolio = LivePortfolioArtifact {
        schema_version: 6,
        search_scope: selection,
        final_holdout_scope,
        search_config_hash: body.source_search_config_hash.to_owned(),
        live_trading_policy: policy,
        symbol: "EURUSD".to_owned(),
        base_tf: "M1".to_owned(),
        higher_tfs,
        effective_feature_names,
        normalize_features,
        cost_band: vec![(
            gene.strategy_id.clone(),
            neoethos_search::discovery::CostBandVerdict::Unmeasured,
        )],
        genes: vec![gene],
        sizing_evidence: vec![LiveSizingEvidenceV1 {
            forward_test: evidence,
        }],
    };
    portfolio.validate().unwrap();
    let mut settings = Settings::default();
    settings.models.label_horizon_bars = 0;
    settings.risk.meta_label_max_hold_bars = 3;
    (portfolio, settings)
}

#[test]
fn discovery_handoff_accepts_the_full_receipt_but_never_moves_the_training_cutoff() {
    let receipt = exact_receipt();
    let (portfolio, settings) =
        discovery_fixture_portfolio(receipt.clone(), vec!["close_minus_open".to_owned()], false);
    let timestamps = neoethos_data::test_fixtures::ctrader_sample_ohlcv()
        .timestamp
        .unwrap();
    let contract = handoff(&["alpha"]).screening_contract().clone();
    let selected = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
        exact_series(&receipt),
        contract,
        &portfolio,
        &settings,
    )
    .expect("full source receipt with exact training/calibration/final scopes is a valid training input");
    assert_eq!(selected.oos_cutoff_ms(), timestamps[80]);
    assert_eq!(
        portfolio.sizing_evidence[0]
            .forward_test
            .scope()
            .evaluated_window()
            .timestamp_start_ms(),
        timestamps[80]
    );
    assert_eq!(
        portfolio
            .final_holdout_scope
            .evaluated_window()
            .timestamp_start_ms(),
        timestamps[90]
    );
    assert!(
        selected.oos_cutoff_ms()
            < portfolio
                .final_holdout_scope
                .evaluated_window()
                .timestamp_start_ms()
    );
    assert_eq!(selected.purge_bars(), 3);
    assert_eq!(selected.search_input_receipt(), &receipt);
    selected.validate_against_settings_v1(&settings).unwrap();

    let original = serde_json::to_value(&selected).unwrap();
    let mut final_cutoff = original.clone();
    final_cutoff["oos_cutoff_ms"] = timestamps[90].into();
    let final_cutoff: PromotionCandidateTrainingHandoffV1 =
        serde_json::from_value(final_cutoff).unwrap();
    assert_eq!(
        final_cutoff.identity_sha256().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage,
        "final-test reservation must not permit refitting on the calibration interval"
    );
    let mut late_cutoff = original.clone();
    late_cutoff["oos_cutoff_ms"] = (timestamps[80] + 1).into();
    let late: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(late_cutoff).unwrap();
    assert_eq!(
        late.identity_sha256().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
    );
    let mut missing_split = original.clone();
    missing_split
        .as_object_mut()
        .unwrap()
        .remove("discovery_holdout_scope");
    let missing: PromotionCandidateTrainingHandoffV1 =
        serde_json::from_value(missing_split).unwrap();
    assert_eq!(
        missing.identity_sha256().unwrap_err().code(),
        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage
    );
    let decoded: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(original).unwrap();
    assert_eq!(
        decoded.identity_sha256().unwrap(),
        selected.identity_sha256().unwrap()
    );
}
