const BROKER_API: &str = include_str!("../src/app_services/broker_api.rs");
const BOOTSTRAP_WRITER: &str = include_str!("../src/app_services/bootstrap_writer.rs");
const CTRADER_DATA: &str = include_str!("../src/app_services/ctrader_data.rs");
const APP_SERVICES_MOD: &str = include_str!("../src/app_services/mod.rs");
const SHARED_SERVICE: &str = include_str!("../../neoethos-broker-history/src/service.rs");

fn download_path() -> &'static str {
    let start = SHARED_SERVICE
        .find("pub(crate) fn capture_with_connector_and_publication_hook")
        .expect("shared history capture entrypoint");
    let tail = &SHARED_SERVICE[start..];
    let end = tail
        .find("pub fn capture_historical_generation")
        .expect("shared capture boundary");
    &tail[..end]
}

#[test]
fn broker_history_never_collects_complete_pages_or_columns_in_memory() {
    let download = download_path();
    for retired in [
        "Vec<Vec<HistoricalBar>>",
        "let mut all_bars",
        "all_bars.extend",
        "bars_to_normalized",
        "let normalized =",
    ] {
        assert!(
            !download.contains(retired),
            "retired full-history buffer remains in production: {retired}"
        );
    }
    assert!(download.contains("CanonicalOhlcvReverseSpool::create"));
    assert!(download.contains("historical_bars_into_chunk("));
    assert!(download.contains(".push_latest(chunk)?"));
    assert!(download.contains("spool.into_oldest_first()"));
}

#[test]
fn exact_identity_cas_is_resolved_before_chunk_publication() {
    let helper_start = SHARED_SERVICE
        .find("fn publish_history")
        .expect("shared publication helper");
    let helper_tail = &SHARED_SERVICE[helper_start..];
    let helper_end = helper_tail
        .find("pub(crate) fn capture_with_connector_and_publication_hook")
        .expect("publication helper boundary");
    let helper = &helper_tail[..helper_end];
    let cas = helper
        .find("request.target.expected_generation_for(identity)?")
        .expect("typed CAS resolution");
    let publication = helper
        .find("let publication_request = BrokerTrendbarStreamRequest")
        .expect("streaming broker publisher");
    assert!(cas < publication);
    assert!(!helper.contains("publish_broker_trendbars"));
}

#[test]
fn every_broker_page_keeps_and_checks_wire_identity_before_spooling() {
    let result_start = CTRADER_DATA
        .find("pub struct CTraderHistoricalBarsFetchResult")
        .expect("broker page result");
    let result_tail = &CTRADER_DATA[result_start..];
    let result_end = result_tail
        .find("#[derive(Debug, Deserialize)]")
        .expect("broker page result boundary");
    let result = &result_tail[..result_end];
    assert!(result.contains("pub symbol_id: i64"));
    assert!(result.contains("pub timeframe: neoethos_core::CanonicalTimeframe"));

    let bridge_start = CTRADER_DATA
        .find("impl CTraderPersistentHistoricalWire for ProductionCTraderPersistentHistoricalWire")
        .expect("persistent broker history wire");
    let bridge_tail = &CTRADER_DATA[bridge_start..];
    let bridge_end = bridge_tail
        .find("pub(crate) struct CTraderAuthenticatedHistoricalSession")
        .expect("broker history bridge boundary");
    let bridge = &bridge_tail[..bridge_end];
    let parsed = bridge
        .find("parse_trendbars_response")
        .expect("wire response parsing");
    let validated = bridge
        .find("validate_identity")
        .expect("wire response identity validation");
    let returned = bridge
        .find("Ok(result)")
        .expect("validated broker page return");
    assert!(parsed < validated && validated < returned);

    let download = download_path();
    let fetched = download
        .find(".next_page(HistoricalPageRequest")
        .expect("one bounded broker page fetch");
    let revalidated = download[fetched..]
        .find("page.symbol_id != resolved.symbol_id || page.timeframe != request.timeframe")
        .map(|offset| fetched + offset)
        .expect("download-side page identity validation");
    let spool_created = download
        .find("CanonicalOhlcvReverseSpool::create")
        .expect("bounded Vortex spool creation");
    let spooled = download
        .find(".push_latest(chunk)?")
        .expect("bounded Vortex page spool");
    assert!(fetched < revalidated && revalidated < spool_created && spool_created < spooled);
    let lower = download.to_ascii_lowercase();
    assert!(!lower.contains("resampl"));
    assert!(!lower.contains("synthesi"));
}

#[test]
fn bootstrap_writer_has_only_the_owned_chunk_publication_path() {
    assert!(BOOTSTRAP_WRITER.contains("pub fn publish_broker_trendbar_chunks"));
    assert!(BOOTSTRAP_WRITER.contains("CanonicalOhlcvStreamPublishRequest"));
    assert!(BOOTSTRAP_WRITER.contains("CanonicalVolumeChunk::Int64"));
    assert!(!BOOTSTRAP_WRITER.contains("pub fn publish_broker_trendbars"));
    assert!(!BOOTSTRAP_WRITER.contains("normalized_bars_to_ohlcv"));
}

#[test]
fn app_has_only_the_shared_capture_adapter_and_no_retired_normalized_path() {
    assert!(BROKER_API.contains("neoethos_broker_history::capture_historical_generation"));
    assert!(!BROKER_API.contains("fn download_history_blocking_inner"));
    assert!(!APP_SERVICES_MOD.contains("mod ctrader_bootstrap"));
    assert!(!BROKER_API.contains("NormalizedBar"));
    assert!(!BOOTSTRAP_WRITER.contains("NormalizedBar"));
}
