use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-models"))
        .parent()
        .and_then(|path| path.parent())
        .expect("models manifest must be below the workspace root")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing function marker {marker:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has an opening brace");
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {marker:?} has no closing brace")
}

#[test]
fn data_opens_every_series_member_by_exact_generation_receipt_only() {
    let source = read("crates/neoethos-data/src/lib.rs");
    let body = function_body(&source, "pub fn load_exact_dataset_series_receipt(");

    for required in [
        "CanonicalDatasetSeriesReceiptV1",
        "series.validate()",
        "series.direct_timeframes()",
        "load_exact_canonical_timeframe",
        "selected.identity().timeframe()",
        "selected.identity().symbol_name()",
        "SymbolDataset",
    ] {
        assert!(
            body.contains(required),
            "exact series loader is missing `{required}`"
        );
    }

    for forbidden in [
        "load_canonical_timeframe(",
        "open_current_dataset_generation",
        "discover_",
        "resample",
        "load_symbol_dataset(",
    ] {
        assert!(
            !body.contains(forbidden),
            "exact series loader contains forbidden current/discovery/derived path `{forbidden}`"
        );
    }
}

#[test]
fn training_has_a_receipt_bound_entrypoint_and_shared_dataset_execution_body() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let entry = function_body(&source, "pub fn train_canonical_series_with_progress<R>(");

    for required in [
        "&CanonicalDatasetSeriesReceiptV1",
        "CanonicalTimeframe",
        "series.validate()",
        "load_exact_dataset_series_receipt",
        "series.anchor().identity().symbol_name()",
        "train_dataset_with_progress",
    ] {
        assert!(
            entry.contains(required),
            "receipt-bound training entry is missing `{required}`"
        );
    }

    for forbidden in [
        "load_symbol_dataset(",
        "load_canonical_timeframe(",
        "current_generation",
        "discover_",
        "resample",
    ] {
        assert!(
            !entry.contains(forbidden),
            "receipt-bound training entry contains forbidden loader `{forbidden}`"
        );
    }

    let shared = function_body(&source, "fn train_dataset_with_progress<R>(");
    assert!(
        shared.contains("prepare_multitimeframe_features_with_options")
            && shared.contains("train_models_parallel_with_progress"),
        "the exact-series entry does not converge on the real full training pipeline"
    );
}

#[test]
fn training_accepts_an_exact_search_receipt_without_requiring_the_search_feature_frame() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let entry = function_body(
        &source,
        "pub fn train_canonical_series_receipt_with_progress<R>(",
    );

    for required in [
        "&CanonicalDatasetSeriesReceiptV1",
        "&neoethos_search::CanonicalSearchInputReceiptV2",
        "screening_contract.validate_against_receipt(input_receipt)?",
        "series.validate()",
        "selected_feature_timeframes",
        "load_exact_dataset_series_receipt",
        "TrainingLabelEconomics::CanonicalTrendbarScreeningV2",
        "train_dataset_with_progress",
    ] {
        assert!(
            entry.contains(required),
            "receipt-only canonical training entry is missing `{required}`"
        );
    }
    for forbidden in [
        "CanonicalSearchInput::from_",
        "load_symbol_dataset(",
        "load_canonical_timeframe(",
        "current_generation",
        "discover_",
        "resample",
        "BrokerFinancialTruth",
    ] {
        assert!(
            !entry.contains(forbidden),
            "receipt-only canonical training reaches forbidden route `{forbidden}`"
        );
    }
}

#[test]
fn configured_nvidia_preflight_checks_only_the_exact_plan_and_rejects_every_cpu_substitution() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let preflight = function_body(
        &source,
        "pub fn preflight_configured_nvidia_training(&self)",
    );

    for required in [
        "self.create_dispatch_plan()?",
        "self.validate_dispatch_plan(&dispatch_plan)?",
        "AcceleratorBackend::Cuda",
        "self.validate_nvidia_model_config_v1(&config)?",
    ] {
        assert!(
            preflight.contains(required),
            "configured NVIDIA preflight is missing `{required}`"
        );
    }
    assert!(
        !preflight.contains("DEFAULT_BOOTSTRAP_EXPERT_NAMES")
            && !preflight.contains("missing production ensemble voters"),
        "configured training incorrectly requires models that are absent from its exact plan"
    );
    for forbidden in [
        "CudaDevicePolicy::Cpu",
        "CPU-only model",
        "configured_cuda_models",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "configured NVIDIA preflight still admits partial GPU execution via `{forbidden}`"
        );
    }
}

#[test]
fn canonical_training_never_accepts_noncanonical_or_unselected_feature_timeframes() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let entry = function_body(&source, "pub fn train_canonical_series_with_progress<R>(");
    let selector = function_body(&source, "fn selected_feature_timeframes(");

    assert!(
        entry.contains("required_timeframes")
            && entry.contains("direct_timeframes()")
            && entry.contains("selected_feature_timeframes"),
        "canonical training does not prove every requested feature timeframe is in the receipt"
    );
    assert!(
        selector.contains("CanonicalTimeframe")
            && selector.contains("parse")
            && selector.contains("self.settings.system.resolve_higher_timeframes(base_tf)"),
        "training timeframe selection still accepts arbitrary strings such as H2"
    );
    assert!(
        !selector.contains("resample"),
        "training timeframe selection contains a derived-timeframe path"
    );
}

#[test]
fn persisted_training_profile_uses_the_same_canonical_timeframe_resolver_as_execution() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let profile = function_body(&source, "fn training_profile_higher_timeframes(");

    assert!(
        profile.contains("settings.system.resolve_higher_timeframes(base_tf)"),
        "persisted training profile does not use the execution-time canonical resolver"
    );
    for forbidden in [
        "multi_resolution_timeframes",
        "higher_timeframes.iter()",
        "eq_ignore_ascii_case",
    ] {
        assert!(
            !profile.contains(forbidden),
            "persisted training profile reimplements timeframe selection via `{forbidden}`"
        );
    }
}

#[test]
fn full_nvidia_training_refuses_the_explicit_cpu_bayes_boundary_until_its_gpu_route_exists() {
    let statistical = read("crates/neoethos-models/src/statistical/common.rs");
    let policy = function_body(&statistical, "pub fn statistical_device_policy(");
    for required in [
        "current_model_device_overrides()",
        "per_model.contains_key(model_name)",
        "requested_runtime_device_policy(model_name)",
        "configured_statistical_device()",
    ] {
        assert!(
            policy.contains(required),
            "statistical device policy cannot express the exact per-model split via `{required}`"
        );
    }

    let config = read("config.yaml");
    for required in [
        "  enable_gpu: true",
        "  enable_gpu_preference: gpu",
        "  device: gpu:0",
        "  statistical_device: gpu:0",
        "  tree_runtime:",
        "    gpu_only: true",
        "    lightgbm_gpu: true",
        "  model_param_overrides:",
        "    bayes_logit:",
        "      device: cpu",
    ] {
        assert!(
            config.contains(required),
            "full-run config does not pin the exact GPU/CPU model policy `{required}`"
        );
    }

    let models = read("crates/neoethos-models/src/training_orchestrator.rs");
    let validator = function_body(&models, "fn validate_nvidia_model_config_v1(");
    assert!(
        validator.contains("supports_nvidia_cuda_for_model")
            && validator.contains("CudaDevicePolicy::Gpu { ordinal: 0 }")
            && !validator.contains("CudaDevicePolicy::Cpu"),
        "the explicit bayes CPU boundary must make full-GPU preflight fail, not become a fallback"
    );
}

#[test]
fn burn_exit_and_sac_training_consume_the_exact_planned_device_policy() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");

    for (marker, next_marker, model) in [
        (
            "ModelType::ExitAgent => {",
            "ModelType::SacAgent => {",
            "exit_agent",
        ),
        ("ModelType::SacAgent => {", "ModelType::Dqn => {", "sac"),
    ] {
        let start = source
            .find(marker)
            .unwrap_or_else(|| panic!("missing {model} training arm"));
        let end = source[start..]
            .find(next_marker)
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("missing boundary after {model} training arm"));
        let arm = &source[start..end];

        assert!(
            arm.contains(".with_device_policy(")
                && arm.contains("parse_string_param(&config.params, \"device\")")
                && arm.contains("unwrap_or_else(|| \"auto\".to_string())")
                && arm.contains(")?"),
            "{model} training ignores the exact device selected by the dispatch/hardware plan"
        );
    }
}

#[test]
fn full_run_preflights_the_complete_training_dispatch_on_exact_cuda_zero() {
    let models = read("crates/neoethos-models/src/training_orchestrator.rs");
    let preflight = function_body(&models, "pub fn preflight_full_nvidia_cuda_training(");

    for required in [
        "self.create_dispatch_plan()?",
        "self.validate_dispatch_plan(&dispatch_plan)?",
        "self.build_training_configs_with_hardware_plan(&dispatch_plan, &hardware_plan)?",
        "DEFAULT_BOOTSTRAP_EXPERT_NAMES",
        "canonical_model_name(&config.name)",
        ".difference(&planned_canonical)",
        ".difference(&required_voters)",
        "missing_voters.is_empty()",
        "unconsumed_models.is_empty()",
        "self.hardware_execution_plan()",
        "AcceleratorBackend::Cuda",
        "WorkloadKind::StrategySearch",
        "WorkloadKind::TreeTraining",
        "WorkloadKind::DeepTraining",
        "WorkloadKind::RlTraining",
        "self.validate_nvidia_model_config_v1(&config)?",
    ] {
        assert!(
            preflight.contains(required),
            "full NVIDIA training preflight is missing `{required}`"
        );
    }

    let strict_model = function_body(&models, "fn validate_nvidia_model_config_v1(");
    for required in [
        "supports_nvidia_cuda_for_model",
        "full_nvidia_device_policy_for_config",
        "CudaDevicePolicy::Gpu { ordinal: 0 }",
    ] {
        assert!(
            strict_model.contains(required),
            "strict full-GPU model authority is missing `{required}`"
        );
    }

    let cpu_pin = function_body(&models, "fn pin_cpu_only_model_device(");
    for required in [
        "model_requires_cuda_in_full_nvidia_run",
        "\"device\"",
        "\"cpu\"",
        "\"__planned_backend\"",
        "\"__planned_device\"",
    ] {
        assert!(
            cpu_pin.contains(required),
            "CPU-only model planning is not explicit about `{required}`"
        );
    }

    let cli = read("crates/neoethos-cli/src/canonical_full_run.rs");
    let run = function_body(&cli, "pub fn run(");
    let preflight_at = run
        .find("preflight_full_nvidia_cuda_training")
        .expect("canonical full run must preflight training");
    let evidence_at = run
        .find("store.open_plan")
        .expect("canonical full run must open its plan");
    let search_at = [
        "run_prepared_canonical_trendbar_research_with_cpu_training_handoff_v3",
        "run_canonical_trendbar_gpu_only_compact_v1",
    ]
    .into_iter()
    .filter_map(|entrypoint| run.find(entrypoint))
    .min()
    .expect("canonical full run must execute its prepared discovery entrypoint");
    assert!(
        preflight_at < evidence_at && preflight_at < search_at,
        "training dispatch/device failures must be detected before evidence loading and expensive search"
    );
}

#[test]
fn sac_enablement_does_not_auto_train_the_unconsumed_exit_agent() {
    let models = read("crates/neoethos-models/src/training_orchestrator.rs");
    let dispatch = function_body(&models, "fn create_dispatch_plan(&self)");

    assert!(
        dispatch.contains("self.settings.models.ml_models.clone()"),
        "explicitly requested models, including a future explicit exit_agent request, must remain supported"
    );
    assert!(
        dispatch.contains("if self.settings.models.use_sac_agent")
            && dispatch.contains("requested_models.push(\"sac\".to_string())"),
        "SAC enablement must still request the production SAC entry voter"
    );
    assert!(
        !dispatch.contains("requested_models.push(\"exit_agent\".to_string())"),
        "SAC enablement still auto-trains exit_agent even though no production path consumes it"
    );
    assert!(
        !dispatch.contains("exit_agent queued for training"),
        "the automatic wasted-training warning should disappear with the automatic wasted training"
    );

    let bootstrap = read("crates/neoethos-models/src/ensemble_inference/bootstrap.rs");
    assert!(
        bootstrap.contains("`exit_agent` — F-318 (no production exit-side consumer)"),
        "the no-consumer authority for excluding automatic exit-agent training drifted"
    );
}

#[test]
fn property_search_does_not_auto_train_a_second_unconsumed_genetic_expert() {
    let models = read("crates/neoethos-models/src/training_orchestrator.rs");
    let dispatch = function_body(&models, "fn create_dispatch_plan(&self)");

    assert!(
        dispatch.contains("self.settings.models.ml_models.clone()"),
        "an explicitly requested genetic expert must remain available for direct development use"
    );
    assert!(
        !dispatch.contains("requested_models.push(\"genetic\".to_string())"),
        "prop_search_enabled still auto-trains a genetic model artifact that no production loader consumes"
    );

    let bootstrap = read("crates/neoethos-models/src/ensemble_inference/bootstrap.rs");
    assert!(
        bootstrap.contains("`genetic` — the strategy DISCOVERER")
            && bootstrap.contains("search-only exemption applies to it alone"),
        "the strategy-search-only authority for excluding automatic genetic-model training drifted"
    );
}

#[test]
fn exact_training_normalization_is_fit_only_to_the_purged_pre_holdout_rows() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let shared = function_body(&source, "fn train_dataset_with_progress<R>(");
    for required in [
        "oos_training_boundary",
        "normalization_training_rows",
        "drop_columns_without_normalization_training_support: true",
        "0..keep",
        "label_horizon_bars",
        "t < cutoff",
    ] {
        assert!(
            shared.contains(required),
            "exact training normalization boundary is missing `{required}`"
        );
    }
}

#[test]
fn dense_model_training_projects_one_receipted_valid_suffix_without_imputation() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let projection = function_body(&source, "fn project_dense_training_suffix(");
    let shared = function_body(&source, "fn train_dataset_with_progress<R>(");

    for required in [
        "DENSE_TRAINING_MIN_ROWS",
        "DENSE_TRAINING_REQUIRED_COLUMNS",
        "trailing_valid_run",
        "checked_mul",
        "frame.select_columns(&kept_columns)?",
        "row_window(row_start, frame.n_samples())?",
    ] {
        assert!(
            projection.contains(required),
            "dense training suffix projection is missing `{required}`"
        );
    }
    for required in ["quant_log_return", "quant_log_volatility"] {
        assert!(
            source.contains(required),
            "dense training required-column authority is missing `{required}`"
        );
    }
    for forbidden in ["fill", "imput", "unwrap_or", "f64::NAN"] {
        assert!(
            !projection.contains(forbidden),
            "dense training suffix projection contains forbidden fallback `{forbidden}`"
        );
    }
    assert!(
        shared.contains("project_dense_training_suffix(&frame)?")
            && !shared.contains("fully_valid_row_indices(&frame)?"),
        "training still intersects every partially-valid feature row instead of projecting a dense suffix"
    );
}

#[test]
fn canonical_training_consumes_the_versioned_screening_cost_contract_without_scalar_fallback() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let entry = function_body(&source, "pub fn train_canonical_series_with_progress<R>(");
    let shared = function_body(&source, "fn train_dataset_with_progress<R>(");
    let screening_labels = function_body(&source, "fn derive_labels_with_screening_costs(");

    for required in [
        "search_input: neoethos_search::data_selection::CanonicalSearchInput",
        "screening_contract: &neoethos_search::CanonicalTrendbarResearchExecutionContractV3",
        "screening_contract.validate_against_input(&search_input)?",
        "TrainingLabelEconomics::CanonicalTrendbarScreeningV2",
        "screening_contract.pip_size()",
        "screening_contract.screening_round_trip_cost_pips()",
    ] {
        assert!(
            entry.contains(required),
            "receipt-bound training entry is missing screening-cost contract fact `{required}`"
        );
    }
    assert!(
        shared.contains("TrainingLabelEconomics::CanonicalTrendbarScreeningV2")
            && shared.contains("derive_labels_with_screening_costs(")
            && shared.contains("round_trip_cost_pips"),
        "shared training body does not route canonical labels through V2 screening costs"
    );
    assert!(
        screening_labels.contains(
            "derive_labels_unchecked_test_oracle(ohlcv, symbol, pip_size, round_trip_cost_pips)",
        ),
        "screening-cost label helper does not reach the real label implementation"
    );
    for forbidden in [
        "exact_round_trip_cost_pips: f64",
        "derive_labels_with_exact_broker_costs",
        "TrainingLabelEconomics::ExactCanonicalTrendbar",
        "get_symbol_metadata",
        "pip_size(symbol)",
        "unwrap_or",
    ] {
        assert!(
            !entry.contains(forbidden) && !screening_labels.contains(forbidden),
            "canonical screening labels retain untyped or fallback path `{forbidden}`"
        );
    }
}

#[test]
fn canonical_training_refuses_a_same_symbol_contract_from_a_different_search_payload() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let entry = function_body(&source, "pub fn train_canonical_series_with_progress<R>(");

    let owned_input = "search_input: neoethos_search::data_selection::CanonicalSearchInput";
    let scoped_input = "let search_input = search_input.as_run_input()?";
    assert!(
        entry.contains(owned_input) && !entry.contains("search_input: &"),
        "canonical training must own the exact CanonicalSearch V2 payload so it can release the search frame before rebuilding training features"
    );
    assert!(
        entry.contains(scoped_input),
        "canonical training must derive the receipt-bound run input from the owned search payload"
    );
    assert!(
        entry.contains("screening_contract.validate_against_input(&search_input)?"),
        "canonical training must refuse a valid same-symbol contract bound to different feature bits/provenance"
    );
    assert!(
        !entry.contains("screening_contract.symbol() == symbol"),
        "symbol equality is not a substitute for the exact CanonicalSearchInputReceiptV2"
    );

    let validation_offset = entry
        .find("screening_contract.validate_against_input(&search_input)?")
        .expect("exact search-payload validation must be present");
    let cost_offset = entry
        .find("screening_contract.screening_round_trip_cost_pips()")
        .expect("validated screening cost must feed canonical training labels");
    let drop_offset = entry
        .find("drop(search_input);")
        .expect("owned search features must be released before training rebuilds its frame");
    let rebuild_offset = entry
        .find("load_exact_dataset_series_receipt")
        .expect("canonical training must reopen the exact series receipt");
    assert!(
        validation_offset < cost_offset
            && cost_offset < drop_offset
            && drop_offset < rebuild_offset,
        "training must validate exact V2 payload, extract costs, release search memory, then rebuild training features"
    );
}

#[test]
fn settings_only_training_refuses_to_omit_account_currency_commission() {
    let source = read("crates/neoethos-models/src/training_orchestrator.rs");
    let configured = function_body(&source, "fn configured_label_round_trip_cost_pips(");
    assert!(
        configured.contains("settings-only training cannot convert account-currency commission")
            && configured.contains("commission_account_per_lot.is_finite()")
            && configured.contains("full_spread_pips.is_finite() && full_spread_pips >= 0.0")
            && configured
                .contains("slippage_pips_per_fill.is_finite() && slippage_pips_per_fill >= 0.0",)
            && configured.contains("full_spread_pips + 2.0 * slippage_pips_per_fill")
            && !configured.contains(".max(0.0)"),
        "settings-only training still clamps invalid assumptions, silently omits commission, or counts only one slippage fill"
    );
}
