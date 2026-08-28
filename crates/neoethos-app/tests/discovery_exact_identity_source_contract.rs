const DISCOVERY: &str = include_str!("../src/app_services/discovery.rs");
const ENGINES_CONTROL: &str = include_str!("../src/server/engines_control.rs");
const HEADLESS: &str = include_str!("../src/main.rs");
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
    assert!(DISCOVERY.contains("request.config.higher_timeframes = request.higher_tfs.clone()"));
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
    assert!(handler.contains("pin_discovery_input"));
    assert!(handler.contains("ExactDatasetGenerationConflict"));
    assert!(handler.contains("StatusCode::CONFLICT"));
    assert!(!handler.contains("preflight_discovery_data_root"));
}

#[test]
fn an_explicit_empty_http_higher_timeframe_set_never_becomes_a_settings_fallback() {
    let handler = section(
        ENGINES_CONTROL,
        "pub async fn discovery_start(",
        "\n}\n\npub async fn discovery_stop",
    );
    assert!(handler.contains("match body.higher_tfs {"));
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
    for (name, source) in [
        ("headless", HEADLESS),
        ("validation", VALIDATION),
        ("supervisor", SUPERVISOR),
        ("federation", FEDERATION),
        ("rediscovery", REDISCOVERY),
    ] {
        assert!(
            source.contains("resolve_unique_background_dataset_identity"),
            "{name} still bypasses strict unique background identity resolution"
        );
        if matches!(name, "headless" | "validation") {
            assert!(
                source.contains("pin_current_discovery_input"),
                "{name} still starts discovery without pinning exact generations"
            );
        }
    }
}
