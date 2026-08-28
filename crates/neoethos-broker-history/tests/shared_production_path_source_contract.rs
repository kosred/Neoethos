const APP_BROKER_API: &str = include_str!("../../neoethos-app/src/app_services/broker_api.rs");
const APP_DATA_CONTROL: &str = include_str!("../../neoethos-app/src/server/data_control.rs");
const APP_MANIFEST: &str = include_str!("../../neoethos-app/Cargo.toml");
const SHARED_SERVICE: &str = include_str!("../src/service.rs");
const CLI_SERVICE: &str = include_str!("../src/cli.rs");
const CLI_MAIN: &str = include_str!("../src/bin/neoethos-historical-fetch.rs");
const MANIFEST: &str = include_str!("../Cargo.toml");

#[test]
fn app_and_cli_reach_one_authoritative_capture_service() {
    assert!(APP_BROKER_API.contains("neoethos_broker_history::capture_historical_generation"));
    assert!(CLI_MAIN.contains("neoethos_broker_history::cli::{"));
    assert!(CLI_MAIN.contains("let receipt = execute(cli, budget)?"));
    assert!(CLI_SERVICE.contains("capture_historical_generation"));
    assert!(SHARED_SERVICE.contains("pub fn capture_historical_generation"));
    assert_eq!(
        SHARED_SERVICE
            .matches("CTraderAuthenticatedHistoricalSession::connect")
            .count(),
        1
    );

    assert!(
        !APP_BROKER_API.contains("fn download_history_blocking_inner"),
        "the superseded app-local page loop still exists"
    );
    assert!(
        !APP_BROKER_API.contains("fn publish_resolved_broker_history"),
        "the superseded app-local publisher wrapper still exists"
    );
}

#[test]
fn app_http_status_stop_and_capture_share_the_authoritative_run_registry() {
    assert!(APP_MANIFEST.contains("neoethos-broker-history"));
    for required in [
        "begin_process_historical_capture",
        "cancel_process_historical_capture",
        "process_historical_capture_status",
        "is_historical_capture_cancelled",
    ] {
        assert!(
            APP_DATA_CONTROL.contains(required),
            "app HTTP path does not use shared historical service API {required}"
        );
    }
    for superseded in [
        "begin_process_historical_fetch_queued",
        "cancel_process_historical_fetch",
        "process_historical_fetch_status",
        "is_historical_request_cancelled",
    ] {
        assert!(
            !APP_DATA_CONTROL.contains(superseded),
            "app HTTP path still uses its superseded local registry API {superseded}"
        );
    }
    let stable_wire_sources = format!("{APP_DATA_CONTROL}\n{SHARED_SERVICE}");
    for stable_wire_marker in [
        "FETCH_CANCELLED",
        "STALE_DATASET_RECEIPT",
        "DATASET_PUBLICATION_CONFLICT",
        "BROKER_IDENTITY_MISMATCH",
        "BROKER_DATASET_ALREADY_EXISTS",
        "PublicationInProgress",
        "StaleRun",
        "NoActiveFetch",
    ] {
        assert!(
            stable_wire_sources.contains(stable_wire_marker),
            "app HTTP behavior lost stable marker {stable_wire_marker}"
        );
    }
}

#[test]
fn success_receipt_is_derived_from_the_returned_publication_manifest() {
    assert!(
        SHARED_SERVICE
            .contains("SelectedDatasetGenerationV1::from_manifest(publication.manifest())")
    );
    for forbidden in [
        "read_current_manifest",
        "load_current",
        "current_generation",
        "resample",
        "download_history_blocking(",
    ] {
        assert!(
            !SHARED_SERVICE.contains(forbidden),
            "shared service contains forbidden fallback/legacy marker {forbidden}"
        );
    }
}

#[test]
fn lean_crate_has_no_model_search_app_or_tauri_dependency() {
    for forbidden in [
        "neoethos-app",
        "neoethos-models",
        "neoethos-search",
        "tauri",
    ] {
        assert!(
            !MANIFEST.contains(forbidden),
            "historical broker crate depends on forbidden graph edge {forbidden}"
        );
    }
}

#[test]
fn cli_prints_receipt_bytes_and_keeps_diagnostics_off_stdout() {
    assert!(CLI_MAIN.contains("render_receipt_stdout"));
    assert!(CLI_MAIN.contains("std::io::stdout().write_all"));
    assert!(!CLI_MAIN.contains("serde_json::json!"));
    assert!(!CLI_MAIN.contains("println!"));
}
