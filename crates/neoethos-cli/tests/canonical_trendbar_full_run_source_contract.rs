use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-cli"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

#[test]
fn model_artifact_read_trait_is_compiled_only_with_the_full_gpu_run() {
    let source = read("src/canonical_full_run.rs");
    assert!(
        source.contains("#[cfg(feature = \"gpu-nvidia-full\")]\nuse std::io::Read;"),
        "the model-artifact Read trait must not enter the default CLI build",
    );
    assert_eq!(
        source.matches("use std::io::Read;").count(),
        1,
        "canonical full run must retain one feature-gated Read import",
    );
}

#[test]
fn cli_exposes_one_exact_matrix_bound_full_search_and_training_command() {
    let main = read("src/main.rs");
    let module = read("src/canonical_full_run.rs");

    assert!(main.contains("mod canonical_full_run;"));
    assert!(
        main.contains("\"canonical-full-run\" => canonical_full_run::run("),
        "CLI does not dispatch the exact full-run command"
    );

    for required in [
        "--authority-root",
        "--data-root",
        "--plan-sha256",
        "--matrix-sha256",
        "--symbol",
        "--base-timeframe",
        "--cost-assumptions",
        "--broker-symbol-contract",
        "--settings-source",
        "--models-dir",
        "--out",
        "--receipt-out",
        "CanonicalTrendbarAcquisitionStoreV1",
        "open_plan",
        "open_matrix",
        "pin_exact_canonical_series_v1",
        "into_cpu_dataset_after_no_physical_gpu_v1",
        "prepare_canonical_discovery_run_input_v3",
        "CanonicalSearchInput::from_prepared_canonical_frame",
        "canonical_discovery_normalization_training_rows",
        "normalization_training_rows",
        "CanonicalTrendbarResearchExecutionContractV3::new",
        "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
        "research_contract: neoethos_search::CanonicalTrendbarResearchExecutionContractV3",
        "discovery_result: neoethos_search::DiscoveryResult",
        "train_canonical_series_with_progress",
        "HistoricalResearchArtifactClassV1::ResearchOnly",
        "HistoricalResearchPromotionEligibilityV1::NotPromotionEligible",
        "CanonicalFullRunReceiptV1",
        "artifact_sha256",
        "receipt_path",
        "ensure_distinct_output_targets",
    ] {
        assert!(
            module.contains(required),
            "canonical full-run command is missing `{required}`"
        );
    }
}

#[test]
fn cli_exposes_receipt_bound_training_without_replaying_discovery_or_broker_truth() {
    let main = read("src/main.rs");
    let source = read("src/canonical_full_run.rs");

    assert!(
        main.contains("\"canonical-train\" => canonical_full_run::train_receipt_bound("),
        "CLI does not dispatch the exact receipt-bound training command"
    );
    for required in [
        "const CANONICAL_TRAIN_REQUIRED_FLAGS",
        "pub fn train_receipt_bound(",
        "--input-receipt",
        "--oos-from-ms",
        "CanonicalSearchInputReceiptV2::from_json_bytes",
        "validate_input_receipt_against_series",
        "validate_against_receipt",
        "preflight_configured_nvidia_training",
        "train_canonical_series_receipt_with_progress",
        "CanonicalTrainingArtifactWireV1",
        "HistoricalResearchArtifactClassV1::ResearchOnly",
        "HistoricalResearchPromotionEligibilityV1::NotPromotionEligible",
        "authorization_issued: false",
        "publish_canonical_training_artifact",
    ] {
        assert!(
            source.contains(required),
            "receipt-bound canonical training is missing `{required}`"
        );
    }
}

#[test]
fn cli_builds_screening_cost_envelope_from_settings_broker_contract_and_direct_d1_bases() {
    let main = read("src/main.rs");
    let source = read("src/canonical_full_run.rs");

    assert!(
        main.contains("\"canonical-cost-build\" => canonical_full_run::build_cost_assumptions("),
        "CLI does not dispatch the screening cost-envelope builder"
    );
    for required in [
        "const COST_BUILD_REQUIRED_FLAGS",
        "pub fn build_cost_assumptions(",
        "--authority-root",
        "--data-root",
        "--plan-sha256",
        "--matrix-sha256",
        "--symbol",
        "--basis-timeframe",
        "--broker-symbol-contract",
        "--settings-source",
        "--out",
        "resolve_exact_conversion_basis",
        "load_final_direct_basis",
        "generation_sha256",
        "derive_commission_account_per_lot_per_fill_assumption",
        "validate_broker_symbol_contract",
        "validate_costs",
        "write_json_atomic",
        "cost_assumption_sha256",
    ] {
        assert!(
            source.contains(required),
            "screening cost-envelope builder is missing `{required}`"
        );
    }
    for forbidden in ["resample", "build_get_tick_data_request", "capture_tick"] {
        assert!(
            !source.contains(forbidden),
            "screening cost-envelope builder contains forbidden `{forbidden}` route"
        );
    }
}

#[test]
fn cli_rejects_every_unknown_duplicate_or_unpaired_argument_before_opening_evidence() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "const FULL_RUN_REQUIRED_FLAGS",
        "validate_exact_args(args)?;",
        "args.len() == FULL_RUN_REQUIRED_FLAGS.len() * 2",
        "args.chunks_exact(2)",
        "FULL_RUN_REQUIRED_FLAGS.contains(&flag)",
        "seen.insert(flag)",
        "!value.starts_with(\"--\")",
    ] {
        assert!(
            source.contains(required),
            "strict canonical full-run argument parser is missing `{required}`"
        );
    }
}

#[test]
fn full_run_requires_the_combined_native_and_burn_nvidia_feature() {
    let manifest = read("Cargo.toml");
    let source = read("src/canonical_full_run.rs");

    assert!(
        manifest.contains("gpu-nvidia-full = [")
            && manifest.contains("\"gpu-nvidia\"")
            && manifest.contains("\"neoethos-models/burn-cuda-backend\""),
        "CLI has no one-feature full NVIDIA search-and-training build"
    );
    assert!(
        source.contains("cfg!(feature = \"gpu-nvidia-full\")")
            && source.contains("canonical-full-run requires the complete NVIDIA CUDA feature"),
        "canonical-full-run can start without the complete native + Burn CUDA surface"
    );
}

#[test]
fn cli_full_run_has_no_current_derived_quote_or_tick_path() {
    let source = read("src/canonical_full_run.rs");
    for forbidden in [
        "load_symbol_dataset(",
        "load_canonical_timeframe(",
        "open_current",
        "current_generation",
        "discover_",
        "resample",
        "tick",
        "BidAsk",
        "BrokerFinancialTruthCapability",
        ".session_spread_pips()",
        "permit_issued=true",
        "HistoricalResearchPromotionEligibilityV1::PromotionEligible",
    ] {
        assert!(
            !source.contains(forbidden),
            "canonical full-run command contains forbidden path `{forbidden}`"
        );
    }
}

#[test]
fn cli_full_run_requires_canonical_timeframes_and_exact_financial_assumption_bytes() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "CanonicalTimeframe",
        "FeatureBuildOptions",
        "resolve_higher_timeframes(base_timeframe.as_str())",
        "deny_unknown_fields",
        "cost_assumption_bytes",
        "Sha256",
        "assumption_source_sha256",
        "assumption_source_id",
        "source_environment",
        "source_server",
        "source_account_id",
        "source_components",
        "validate_settings_source",
        "validate_broker_symbol_contract",
        "ConfigSource::EnvConfigFile",
        "settings.provenance()",
        ".path()",
        "Settings::from_yaml",
        "serde_json::to_value(settings)",
        "payloadType",
        "ctidTraderAccountId",
        "pipPosition",
        "swapCalculationType",
        "pnlConversionFeeRate",
        "validate_source_component_bytes",
        "settings.risk.backtest_spread_pips",
        "settings.risk.slippage_pips",
        "settings.risk.commission_per_lot",
        "settings.risk.commission_per_lot_is_per_side",
        "full_spread_pips_assumption",
        "slippage_pips_per_fill_assumption",
        "commission_account_per_lot_per_fill_assumption",
        "ensure_no_session_spread_curve",
        "costs.source_account_id == plan.account_id()",
        "pip_value_quote_per_lot",
        "pip_value_conversion",
        "load_exact_canonical_timeframe",
        "PipValueConversionOperationV1::Divide",
        "PipValueConversionOperationV1::Multiply",
        "PipValueConversionOperationV1::Identity",
        "generation_sha256",
        "ensure_unique_series",
        "installed_process_budget",
        "CpuPermitRequest::local",
    ] {
        assert!(
            source.contains(required),
            "canonical full-run validation is missing `{required}`"
        );
    }
}

#[test]
fn cli_full_run_requires_holdout_and_locks_training_before_its_first_bar() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "training_oos_from_ms",
        "canonical full run requires a holdout scope before model training",
        "holdout_scope.evaluated_window().timestamp_start_ms()",
        ".with_oos_lock_from_ms(training_oos_from_ms)",
        "training_label_round_trip_cost_pips",
        "contract.screening_round_trip_cost_pips()",
        "&contract",
    ] {
        assert!(
            source.contains(required),
            "canonical full-run OOS boundary is missing `{required}`"
        );
    }
}

#[test]
fn cli_recomputes_screening_pip_value_and_per_fill_commission_from_bound_evidence() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "commission_symbol_price_basis",
        "CommissionSymbolPriceBasisWireV1",
        "load_exact_basis_close",
        "lotSize",
        "commissionType",
        "preciseTradingCommissionRate",
        "derive_commission_account_per_lot_per_fill_assumption",
        "expected_pip_value_quote_per_lot",
        "resolved_settings: neoethos_core::Settings",
        "cost_assumption_exact_utf8",
        "settings_source_exact_utf8",
        "broker_symbol_contract_exact_utf8",
    ] {
        assert!(
            source.contains(required),
            "canonical full-run economic evidence is missing `{required}`"
        );
    }
}

#[test]
fn canonical_cost_wire_is_a_versioned_fail_closed_screening_envelope() {
    let source = read("src/canonical_full_run.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("canonical full-run production source");

    for required in [
        "const SCREENING_COST_SCHEMA_V2: &str = \"neoethos.canonical-trendbar-screening-cost-envelope.v2\";",
        "struct ScreeningCostEnvelopeWireV2",
        "full_spread_pips_assumption: f64",
        "slippage_pips_per_fill_assumption: f64",
        "commission_account_per_lot_per_fill_assumption: f64",
        "neoethos.canonical-d1-screening-cost-assumptions.v2",
        "screening_cost_envelope_v2_rejects_legacy_v1_wire",
        "HistoricalResearchArtifactClassV1::ResearchOnly",
        "HistoricalResearchPromotionEligibilityV1::NotPromotionEligible",
    ] {
        assert!(
            source.contains(required),
            "canonical screening-cost envelope is missing `{required}`"
        );
    }

    for forbidden in [
        "CostAssumptionWireV1",
        "COST_SCHEMA_V1",
        "neoethos.exact-broker-canonical-d1-costs.v1",
        "round_trip_commission_per_trade",
    ] {
        assert!(
            !production.contains(forbidden),
            "legacy or falsely exact cost wire remains active through `{forbidden}`"
        );
    }
}

#[test]
fn canonical_screening_envelope_refuses_invalid_assumptions_instead_of_clamping_them() {
    let source = read("src/canonical_full_run.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("canonical full-run production source");

    assert!(
        production.contains("fn require_non_negative_screening_assumption("),
        "canonical screening assumptions have no fail-closed scalar validator"
    );
    for forbidden in [
        "full_spread_pips_assumption: settings.risk.backtest_spread_pips.max(0.0)",
        "slippage_pips_per_fill_assumption: settings.risk.slippage_pips.max(0.0)",
    ] {
        assert!(
            !production.contains(forbidden),
            "canonical screening assumption is silently clamped through `{forbidden}`"
        );
    }
}

#[test]
fn final_evidence_hash_binds_every_completed_model_artifact_tree() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "ModelArtifactEvidenceWireV1",
        "model_artifacts: Vec<ModelArtifactEvidenceWireV1>",
        "hash_model_artifact_tree",
        "model_artifact_evidence",
        "validate_model_artifact_evidence_unchanged",
        "tree_sha256",
        "file_count",
        "total_bytes",
        "fs::symlink_metadata",
        "file_type().is_symlink()",
    ] {
        assert!(
            source.contains(required),
            "final evidence does not bind completed model artifacts through `{required}`"
        );
    }
}

#[test]
fn training_pipeline_errors_publish_reopenable_failure_evidence_before_returning() {
    let source = read("src/canonical_full_run.rs");
    for required in [
        "match orchestrator.train_canonical_series_with_progress(",
        "Err(error) =>",
        "__training_pipeline__",
        "publish_full_run_artifact",
        "exact evidence was written",
    ] {
        assert!(
            source.contains(required),
            "training pipeline failure evidence is missing `{required}`"
        );
    }
}
