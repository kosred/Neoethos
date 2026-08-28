const BROKER_API: &str = include_str!("../src/app_services/broker_api.rs");
const BOOTSTRAP_WRITER: &str = include_str!("../src/app_services/bootstrap_writer.rs");
const DATA_CONTROL: &str = include_str!("../src/server/data_control.rs");
const SHARED_SERVICE: &str = include_str!("../../neoethos-broker-history/src/service.rs");

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing {start}"));
    let tail = &source[start..];
    let end = tail.find(end).unwrap_or_else(|| panic!("missing {end}"));
    &tail[..end]
}

fn shared_capture() -> &'static str {
    between(
        SHARED_SERVICE,
        "pub(crate) fn capture_with_connector_and_publication_hook",
        "pub fn capture_historical_generation",
    )
}

#[test]
fn app_adapter_delegates_and_the_superseded_capture_path_is_deleted() {
    let adapter = between(
        BROKER_API,
        "pub fn download_history_blocking",
        "pub fn fetch_recent_chart_bars_blocking",
    );
    assert!(adapter.contains("neoethos_broker_history::capture_historical_generation"));
    assert!(adapter.contains("HistoricalCaptureTarget::SelectedGeneration(selected.clone())"));
    assert!(adapter.contains("HistoricalCaptureTarget::NewIdentity"));
    assert!(adapter.contains("outcome.selected_generation.identity()"));
    assert!(adapter.contains("outcome.selected_generation.generation_id()"));

    for retired in [
        "download_history_blocking_inner",
        "publish_resolved_broker_history",
        "BrokerHistoryTarget",
        "BrokerHistoryCasBase",
        "historical_bars_into_chunk",
    ] {
        assert!(
            !BROKER_API.contains(retired),
            "superseded app-local history marker remains: {retired}"
        );
    }
    assert!(!BOOTSTRAP_WRITER.contains("BrokerHistoryCasBase"));
}

#[test]
fn shared_target_uses_typed_identity_and_occupied_root_conflicts() {
    let target = between(
        SHARED_SERVICE,
        "pub enum HistoricalCaptureTarget",
        "pub struct HistoricalCaptureRequest",
    );
    assert!(target.contains("NewIdentity"));
    assert!(target.contains("SelectedGeneration(SelectedDatasetGenerationV1)"));
    assert!(target.contains("BrokerHistoryConflict::IdentityMismatch"));
    assert!(target.contains("BrokerHistoryConflict::DatasetRootOccupied"));
    assert!(target.contains("std::fs::symlink_metadata"));
    assert!(target.contains("std::io::ErrorKind::NotFound"));
    assert!(!target.contains("dataset_root.exists()"));
}

#[test]
fn selected_generation_is_verified_and_pinned_before_network_capture() {
    let capture = shared_capture();
    let assertion = capture
        .find("let _selected_generation_lease")
        .expect("selected request preflight");
    let exact_open = capture
        .find("neoethos_data::open_exact_dataset_generation")
        .expect("selected receipt is verified and pinned");
    let connect = capture
        .find(".connect_authenticated(")
        .expect("one authenticated persistent connection");
    let resolved_identity = capture
        .find(".validate_resolved_identity")
        .expect("resolved broker identity validation");
    let first_page = capture
        .find("while cursor_to > request.from_ms")
        .expect("bounded direct-page loop");

    assert!(
        assertion < exact_open
            && exact_open < connect
            && connect < resolved_identity
            && resolved_identity < first_page
    );
    let preflight = &capture[assertion..connect];
    assert!(preflight.contains("BrokerHistoryConflict::IdentityMismatch"));
    assert!(!preflight.contains("bail!("));
}

#[test]
fn broker_capture_uses_the_exact_direct_timeframe_without_resampling() {
    let capture = shared_capture();
    assert!(capture.contains("timeframe: request.timeframe"));
    assert!(capture.contains("HistoricalPageRequest"));
    assert!(capture.contains("page.timeframe != request.timeframe"));
    assert!(capture.contains("CanonicalDatasetIdentity::ctrader"));
    assert!(!capture.to_ascii_lowercase().contains("resample"));
    assert!(!capture.to_ascii_lowercase().contains("synthesi"));
}

#[test]
fn selected_refresh_rechecks_the_complete_receipt_at_final_publication() {
    let publisher = between(
        SHARED_SERVICE,
        "fn publish_history",
        "pub(crate) fn capture_with_connector_and_publication_hook",
    );
    let cas = publisher
        .find("request.target.expected_generation_for(identity)?")
        .expect("typed exact generation resolution");
    let request = publisher
        .find("let publication_request = BrokerTrendbarStreamRequest")
        .expect("canonical publisher request");
    let exact = publisher
        .find("publish_broker_trendbar_chunks_exact(publication_request, selected)")
        .expect("complete receipt reaches final publication");

    assert!(cas < request && request < exact);
    assert!(BOOTSTRAP_WRITER.contains("publish_canonical_ohlcv_stream_exact(request, selected)"));
}

#[test]
fn fetch_wire_requires_complete_receipt_and_maps_stable_typed_conflicts() {
    let body = between(
        DATA_CONTROL,
        "pub struct FetchBody",
        "pub struct FetchOutcomeDto",
    );
    assert!(body.contains("dataset_selection"));
    assert!(body.contains("SelectedDatasetGenerationV1"));
    assert!(!body.contains("expected_generation"));

    let mapper = between(
        DATA_CONTROL,
        "fn cancelled_fetch_response",
        "// ─── GET /broker/timeframes",
    );
    assert!(mapper.contains("is_historical_capture_cancelled(err.as_ref())"));
    assert!(mapper.contains("\"code\": \"FETCH_CANCELLED\""));
    assert!(mapper.contains("ExactDatasetGenerationConflict"));
    assert!(mapper.contains("PublicationConflict"));
    assert!(mapper.contains("BrokerHistoryConflict"));
    assert!(mapper.contains("\"code\": conflict_code"));
    assert!(SHARED_SERVICE.contains("\"BROKER_IDENTITY_MISMATCH\""));
    assert!(SHARED_SERVICE.contains("\"BROKER_DATASET_ALREADY_EXISTS\""));
}

#[test]
fn blocked_historical_payload_maps_both_transport_types_to_the_same_typed_429() {
    let mapper = between(
        DATA_CONTROL,
        "fn broker_gateway_error",
        "// ─── GET /broker/timeframes",
    );
    assert!(mapper.contains("downcast_ref::<CTraderBlockedPayloadError>()"));
    assert!(
        mapper.contains("neoethos_broker_history::ctrader_messages::CTraderBlockedPayloadError")
    );
    assert!(mapper.contains("StatusCode::TOO_MANY_REQUESTS"));
    assert!(mapper.contains("\"code\": \"BLOCKED_PAYLOAD_TYPE\""));
    assert!(mapper.contains("\"retryAfterSeconds\": retry_after_seconds"));
    assert!(mapper.contains("axum::http::header::RETRY_AFTER"));
}
