const DISCOVERY: &str = include_str!("../src/app_services/discovery.rs");
const ENGINES_CONTROL: &str = include_str!("../src/server/engines_control.rs");
const TYPED_EXECUTION: &str = include_str!("../src/server/engines_control/typed_execution_v1.rs");
const HEADLESS: &str = include_str!("../src/main.rs");
const ENTRYPOINTS: &str = include_str!("../src/app_services/entrypoints.rs");
const VALIDATION: &str = include_str!("../src/app_services/validation.rs");
const SUPERVISOR: &str = include_str!("../src/app_services/supervisor.rs");
const FEDERATION: &str = include_str!("../src/app_services/federation.rs");
const REDISCOVERY: &str = include_str!("../src/app_services/rediscovery.rs");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, rest) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source marker {start:?}"));
    rest.split_once(end)
        .unwrap_or_else(|| panic!("missing source marker {end:?} after {start:?}"))
        .0
}

#[test]
fn discovery_request_owns_one_pinned_input_without_a_reopen_selector() {
    let request = section(
        DISCOVERY,
        "pub struct DiscoveryRequest {",
        "\n}\n\nimpl DiscoveryRequest",
    );

    assert!(request.contains("pub pinned_input: Arc<PinnedDiscoveryInput>"));
    assert!(request.contains("pub settings_source: Arc<DiscoverySettingsSource>"));
    assert!(!request.contains("pub dataset_identity:"));
    assert!(!request.contains("pub symbol:"));
    assert!(!request.contains("pub base_tf:"));
}

#[test]
fn discovery_worker_consumes_the_pre_pinned_dataset_without_reopening_current() {
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    assert!(worker.contains("Arc::clone(&request.pinned_input)"));
    assert!(worker.contains("take_pinned_series_v1()"));
    assert!(worker.contains("prepare_canonical_discovery_run_input_v3"));
    assert!(worker.contains("into_cpu_dataset_after_no_physical_gpu_v1"));
    assert!(!worker.contains("load_dataset_for_identity"));
    assert!(!worker.contains("load_canonical_timeframe"));
    assert!(!worker.contains("load_exact_canonical_timeframe"));
    assert!(!DISCOVERY.contains("load_symbol_dataset"));
    assert!(!DISCOVERY.contains("ensure_timeframes_with_resample"));
}

#[test]
fn discovery_cpu_workers_use_the_shared_admission_owner_and_exact_pool() {
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    assert!(worker.contains("execution: Arc<AppExecutionState>"));
    assert_eq!(worker.matches("admit_discovery_cpu_stage(").count(), 2);
    assert_eq!(worker.matches("spawn_discovery_cpu_stage(").count(), 2);
    assert!(!worker.contains("tokio::task::spawn_blocking("));
    let admission = section(
        DISCOVERY,
        "async fn admit_discovery_cpu_stage(",
        "fn spawn_discovery_cpu_stage<",
    );
    assert!(admission.contains("execution.admission_snapshot().cpu.installed_limit"));
    assert!(admission.contains("CpuPermitRequest::local(width)"));
    assert!(!admission.contains("available_permits"));
    assert!(admission.contains("cancel.is_requested()"));
    let executor = section(
        DISCOVERY,
        "fn spawn_discovery_cpu_stage<",
        "pub fn start_discovery_job(",
    );
    assert!(executor.contains("lease.execute_with_scope(execution.executor()"));
    assert!(executor.contains("scope.require_current_pool()?"));
    assert!(executor.contains("cancel.is_requested()"));
    let launch = section(
        TYPED_EXECUTION,
        "async fn run_discovery_job_v1(",
        "async fn run_training_intent_v1(",
    );
    assert!(launch.contains("state.execution_state()"));
    assert!(launch.contains("TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable("));
    assert!(launch.contains("start_discovery_job(request, execution, tx)"));
    assert!(!launch.contains("AppExecutionState::new"));
}

#[test]
fn every_requested_timeframe_is_pinned_as_a_direct_generation_before_launch() {
    let pin = section(
        DISCOVERY,
        "pub fn pin_discovery_input(",
        "\n/// Background jobs do not have",
    );
    assert!(pin.contains("SelectedDatasetGenerationV1"));
    assert!(pin.contains("CanonicalDatasetSeriesReceiptV1"));
    assert!(pin.contains("pin_exact_canonical_series_v1"));
    assert!(!pin.contains("load_exact_canonical_timeframe"));
    assert!(pin.contains("DatasetDiscovery::scan_metadata"));
    assert!(DISCOVERY.contains("required_direct_timeframes"));
    assert!(DISCOVERY.contains("validate_direct_timeframe_artifacts"));
    assert!(DISCOVERY.contains("require_direct_timeframes"));
    assert!(!DISCOVERY.contains("FrameDerivationV1"));
    assert!(!DISCOVERY.contains("download_missing_ctrader_timeframes"));
    assert!(!DISCOVERY.contains("may_auto_fetch_broker_history"));
    assert!(!DISCOVERY.contains("verify_downloaded_identity"));
}

#[test]
fn discovery_requires_only_the_timeframes_the_feature_plan_consumes() {
    let required = section(
        DISCOVERY,
        "fn required_direct_timeframes(",
        "\n}\n\nfn validate_direct_timeframe_artifacts",
    );
    assert!(required.contains("request.dataset_identity().timeframe()"));
    assert!(required.contains("for label in &request.higher_tfs"));
    assert!(
        !required.contains("REQUIRED_DIRECT_TIMEFRAMES"),
        "discovery must not download an unrelated fixed timeframe bundle"
    );
}

#[test]
fn feature_timeframes_and_search_temporal_hash_have_one_validated_truth() {
    assert!(DISCOVERY.contains("duplicate higher timeframe"));
    assert!(DISCOVERY.contains("must be strictly above base"));
    assert!(DISCOVERY.contains("config.higher_timeframes == higher"));
    assert!(!DISCOVERY.contains("request.config.higher_timeframes = request.higher_tfs.clone()"));
}

#[test]
fn selected_run_settings_are_bound_once_without_mutating_the_owned_source() {
    let prepare = section(
        TYPED_EXECUTION,
        "async fn prepare_discovery_request_v1(",
        "\nfn classify_discovery_pin_error_v1(",
    );
    assert!(prepare.contains("DiscoverySettingsSource::load(&config_path)"));
    assert!(prepare.contains("let settings = settings_source.settings();"));
    assert!(prepare.contains("settings_source,"));
    let resolve = section(
        DISCOVERY,
        "fn resolve_research_config(",
        "\n    fn feature_build_options(",
    );
    let financial_resolution = resolve
        .find("DiscoveryConfig::try_from_settings_for_canonical_trendbar_research")
        .unwrap();
    assert!(resolve.contains("let mut run_settings = self.settings_source.settings().clone();"));
    for binding in [
        "run_settings.system.symbol = self.symbol().to_owned()",
        "run_settings.system.base_timeframe = self.base_tf().to_owned()",
        "run_settings.system.higher_timeframes = self.canonical_higher_timeframes()?",
    ] {
        assert!(
            resolve
                .find(binding)
                .is_some_and(|position| position < financial_resolution),
            "missing selected-run binding before financial resolution: {binding}",
        );
    }
    assert!(!resolve.contains("apply_mode_overrides"));
    assert_eq!(resolve.matches("self.overrides.apply(").count(), 1);
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    assert!(worker.contains("search_request.execution_config()"));
    assert!(!worker.contains("apply_mode_overrides"));
}

#[test]
fn both_cpu_factories_use_explicit_options_before_building_features() {
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    assert_eq!(
        worker
            .matches("prepare_cpu_discovery_features_with_control(")
            .count(),
        1
    );
    assert_eq!(
        worker
            .matches("prepare_cpu_discovery_batch_with_control(")
            .count(),
        1
    );
    assert!(!worker.contains("prepare_multitimeframe_features("));
    let producer = section(
        DISCOVERY,
        "fn prepare_cpu_discovery_features_with_control(",
        "\nstruct PreparedDiscoveryScreeningCosts",
    );
    let options = producer
        .find("request.feature_build_options(base_rows)")
        .unwrap();
    let build = producer
        .find("prepare_multitimeframe_features_with_control(")
        .unwrap();
    assert!(
        options < build,
        "reject an invalid IS range before the feature build"
    );
    assert!(producer.contains("dataset: SymbolDataset"));
    assert!(!producer.contains("dataset.canonical_frame("));
    let handoff = producer
        .find("dataset.into_canonical_frame(request.base_tf())")
        .unwrap();
    assert!(
        build < handoff,
        "move the original base only after borrowed feature computation"
    );
    assert!(producer.contains("from_prepared_canonical_frame_with_control("));
    assert!(producer.contains("control: &FeatureBuildControl"));
    assert!(producer.contains("control.checkpoint()?"));
    let batched = section(
        DISCOVERY,
        "fn prepare_cpu_discovery_batch_with_control(",
        "\n#[derive(Default)]\nstruct DiscoveryWorkingSetProgress",
    );
    assert!(
        batched
            .find("request.feature_build_options(base_rows)")
            .unwrap()
            < batched
                .find("prepare_multitimeframe_features_batch_with_options_and_control(")
                .unwrap()
    );
    assert!(batched.contains("dataset: &SymbolDataset"));
    assert!(batched.contains("Some(batch),"));
    assert!(batched.contains("from_prepared_canonical_frame_with_control("));
    assert!(DISCOVERY.contains("validate_exact_file_settings(&settings, path, &exact_bytes)"));
}

#[test]
fn desktop_financial_configuration_is_staged_after_the_real_feature_receipt() {
    let prepare = section(
        TYPED_EXECUTION,
        "async fn prepare_discovery_request_v1(",
        "\nfn classify_discovery_pin_error_v1(",
    );
    assert!(!prepare.contains("DiscoveryConfig::try_from_settings("));
    assert!(prepare.contains("config: None"));
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    let features = worker
        .rfind("prepare_cpu_discovery_batch_with_control(")
        .unwrap();
    let receipt = worker
        .find("let receipt = prepared_input.receipt()?")
        .unwrap();
    let contract = worker.find("seal_discovery_research_contract(").unwrap();
    let config = worker.find("resolve_research_config(").unwrap();
    let completed_cpu_stage = worker.find("feature_handle.await").unwrap();
    let resolved_request = worker.find("request.config = Some(config)").unwrap();
    assert!(features < receipt && receipt < contract && contract < config);
    assert!(
        config < completed_cpu_stage && completed_cpu_stage < resolved_request,
        "receipt hashing and financial resolution must complete inside the admitted feature worker"
    );
    assert!(worker.contains("prepare_discovery_screening_costs("));
    assert!(DISCOVERY.contains("build_screening_cost_envelope_v2("));
    assert!(
        worker.contains("run_canonical_trendbar_research_discovery_with_holdout_and_progress(")
    );
    assert!(
        worker.contains("run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3(")
    );
    assert!(!worker.contains("run_discovery_cycle_with_holdout_and_progress("));
}

#[test]
fn discovery_http_contract_and_preflight_are_exact_identity_bound() {
    let body = section(
        ENGINES_CONTROL,
        "pub struct StartJobBody {",
        "\n}\n\nfn resolve_discovery_selection",
    );
    assert!(body.contains("pub dataset_selection: Option<SelectedDatasetGenerationV1>"));
    assert!(!body.contains("deserialize_optional_dataset_identity"));

    let handler = section(
        ENGINES_CONTROL,
        "pub async fn discovery_start(",
        "\n}\n\npub async fn discovery_stop",
    );
    assert!(handler.contains("TypedDiscoveryDatasetPolicyV1::Exact(dataset_selection.clone())"));
    assert!(handler.contains("start_typed_discovery_execution_v1"));
    assert!(handler.contains("admitted != dataset_selection"));
    assert!(TYPED_EXECUTION.contains("pin_discovery_input"));
    assert!(TYPED_EXECUTION.contains("ExactDatasetGenerationConflict"));
    let admission_errors = section(
        ENGINES_CONTROL,
        "fn typed_legacy_admission_error_response_v1(",
        "\n}\n\n#[cfg(test)]",
    );
    assert!(admission_errors.contains("TypedLegacyExecutionAdmissionErrorV1::Conflict(_)"));
    assert!(admission_errors.contains("StatusCode::CONFLICT"));
    assert!(!handler.contains("preflight_discovery_data_root"));
}

#[test]
fn an_explicit_empty_http_higher_timeframe_set_never_becomes_a_settings_fallback() {
    let handler = section(
        ENGINES_CONTROL,
        "pub async fn discovery_start(",
        "\n}\n\npub async fn discovery_stop",
    );
    let higher_timeframes = section(
        handler,
        "let higher_timeframes = match &body.higher_tfs {",
        "\n    let overrides =",
    );
    assert!(higher_timeframes.contains("Some(labels) =>"));
    assert!(
        higher_timeframes
            .contains("Ok(timeframes) => TypedHigherTimeframePolicyV1::Exact(timeframes)")
    );
    assert!(higher_timeframes.contains("None => TypedHigherTimeframePolicyV1::Configured"));
    assert!(!handler.contains("body.higher_tfs.filter"));
}

#[test]
fn discovery_never_mutates_or_rebinds_data_during_a_run() {
    let worker = section(
        DISCOVERY,
        "pub fn start_discovery_job(",
        "\n/// On-disk contract between Discovery output",
    );
    for forbidden in [
        "download_history",
        "BrokerHistoryTarget",
        "fetching_direct_timeframes",
        "fetching_history",
        "reload",
    ] {
        assert!(
            !worker.contains(forbidden),
            "discovery worker still contains mutation/rebind path {forbidden}"
        );
    }
    assert!(DISCOVERY.contains("acquisition required"));
}

#[test]
fn every_app_background_discovery_caller_passes_an_exact_identity() {
    assert!(HEADLESS.contains("run_headless_execution_pipeline_v1"));
    assert!(ENTRYPOINTS.contains("start_typed_discovery_execution_v1"));
    assert!(ENTRYPOINTS.contains("TypedDiscoveryDatasetPolicyV1::Current"));
    // Interactive Supervisor actions carry the inventory's exact selection
    // through the UI handler, whose typed admission is checked above. They must
    // not resolve a different CURRENT generation as background jobs do.
    let supervisor_execute = section(
        SUPERVISOR,
        "async fn execute(",
        "\n        SupervisorAction::StopDiscovery",
    );
    let supervisor_discovery = supervisor_execute
        .split_once("SupervisorAction::StartDiscovery { dataset_selection } => {")
        .expect("missing exact Supervisor Discovery action")
        .1;
    assert!(supervisor_discovery.contains("engines_control::discovery_start("));
    assert!(supervisor_discovery.contains("dataset_selection: Some(dataset_selection)"));
    assert!(supervisor_discovery.contains("action_response(response).await?"));
    assert!(!SUPERVISOR.contains("TypedDiscoveryDatasetPolicyV1::Current"));
    assert!(!supervisor_discovery.contains("start_typed_discovery_execution_v1"));
    for (name, source) in [
        ("validation", VALIDATION),
        ("federation", FEDERATION),
        ("rediscovery", REDISCOVERY),
    ] {
        assert!(
            source.contains("start_typed_discovery_execution_v1"),
            "{name} still bypasses the shared typed Discovery boundary"
        );
        assert!(
            source.contains("TypedDiscoveryDatasetPolicyV1::Current"),
            "{name} still starts background Discovery without current-generation pin policy"
        );
    }
    assert!(TYPED_EXECUTION.contains("resolve_unique_background_dataset_identity"));
    assert!(TYPED_EXECUTION.contains("pin_current_discovery_input"));
}
