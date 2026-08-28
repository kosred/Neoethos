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

fn exact_receipt() -> CanonicalSearchInputReceiptV2 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
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

fn exact_series(receipt: &CanonicalSearchInputReceiptV2) -> CanonicalDatasetSeriesReceiptV1 {
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

fn handoff(planned_models: &[&str]) -> PromotionCandidateTrainingHandoffV1 {
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
    let receipt = exact_receipt();
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
    let sealed = handoff_with_config(resolved, settings.models.label_horizon_bars);
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
