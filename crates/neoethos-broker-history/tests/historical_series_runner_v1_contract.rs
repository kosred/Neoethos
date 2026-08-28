use anyhow::{Result, bail};
use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_history::{
    CANONICAL_TRENDBAR_SERIES_FROM_MS_V1, CanonicalTrendbarAcquisitionPlanV1,
    CanonicalTrendbarAcquisitionRunStageV1, CanonicalTrendbarAcquisitionStoreV1,
    CanonicalTrendbarSymbolV1, HistoricalCaptureRequest, HistoricalCaptureTarget,
    resume_canonical_trendbar_acquisition_v1_with,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
};
use std::cell::RefCell;

const SERVER: &str = "demo.ctraderapi.com";
const ACCOUNT_ID: i64 = 42;
const TO_MS_EXCLUSIVE: i64 = 1_767_225_600_000;

#[test]
fn production_runner_reuses_the_authenticated_symbol_session_instead_of_reconnecting_per_cell() {
    let source = include_str!("../src/historical_series_runner_v1.rs");
    let production = source
        .split("pub fn run_production_canonical_trendbar_acquisition_v1(")
        .nth(1)
        .expect("production runner source");

    assert!(production.contains("HistoricalSeriesCapture::new("));
    assert!(production.contains("capture_historical_series_generation("));
    assert!(!production.contains("capture_historical_generation("));
}

fn symbols() -> Vec<CanonicalTrendbarSymbolV1> {
    vec![
        CanonicalTrendbarSymbolV1::new(1, "EUR/USD").expect("EUR/USD"),
        CanonicalTrendbarSymbolV1::new(2, "USD/JPY").expect("USD/JPY"),
    ]
}

fn plan() -> CanonicalTrendbarAcquisitionPlanV1 {
    CanonicalTrendbarAcquisitionPlanV1::new(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
        TO_MS_EXCLUSIVE,
        symbols(),
        vec![CanonicalTimeframe::M1, CanonicalTimeframe::H1],
    )
    .expect("exact acquisition plan")
}

fn publish_request(
    data_root: &std::path::Path,
    request: &HistoricalCaptureRequest,
    requested_from_ms: i64,
) -> SelectedDatasetGenerationV1 {
    let symbol = symbols()
        .into_iter()
        .find(|symbol| symbol.symbol_name() == request.symbol)
        .expect("planned symbol");
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        symbol.symbol_id(),
        symbol.symbol_name(),
        request.timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("exact dataset identity");
    let step = request
        .timeframe
        .fixed_duration_ms()
        .expect("fixed test timeframe");
    let first = CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + step;
    let second = first + step;
    let chunk = CanonicalOhlcvChunk {
        timestamp_ms: vec![first, second],
        open: vec![1.10, 1.11],
        high: vec![1.12, 1.13],
        low: vec![1.09, 1.10],
        close: vec![1.11, 1.12],
        volume: CanonicalVolumeChunk::Int64(vec![10, 11]),
    };
    let published = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: data_root,
        identity: &identity,
        expected_generation: None,
        requested_from_ms,
        requested_to_ms: request.to_ms,
        retrieved_unix_ms: u64::try_from(TO_MS_EXCLUSIVE).expect("positive retrieval time"),
        returned_from_ms: first,
        returned_to_ms: second,
        row_count: 2,
        chunks: vec![Ok::<_, anyhow::Error>(chunk)],
    })
    .expect("publish captured cell");
    SelectedDatasetGenerationV1::from_manifest(published.manifest()).expect("selected generation")
}

#[test]
fn full_run_captures_exact_plan_order_and_publishes_matrix() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;
    let observed = RefCell::new(Vec::new());

    let outcome = resume_canonical_trendbar_acquisition_v1_with(
        data.path(),
        &store,
        &plan_receipt,
        None,
        |request| {
            assert_eq!(request.from_ms, CANONICAL_TRENDBAR_SERIES_FROM_MS_V1);
            assert_eq!(request.to_ms, TO_MS_EXCLUSIVE);
            assert!(matches!(
                request.target,
                HistoricalCaptureTarget::NewIdentity
            ));
            observed
                .borrow_mut()
                .push((request.symbol.clone(), request.timeframe));
            Ok(publish_request(data.path(), &request, request.from_ms))
        },
    )?;

    assert_eq!(outcome.completed_cells(), 4);
    assert_eq!(outcome.total_cells(), 4);
    assert_eq!(
        observed.into_inner(),
        vec![
            ("EUR/USD".to_owned(), CanonicalTimeframe::M1),
            ("EUR/USD".to_owned(), CanonicalTimeframe::H1),
            ("USD/JPY".to_owned(), CanonicalTimeframe::M1),
            ("USD/JPY".to_owned(), CanonicalTimeframe::H1),
        ]
    );
    let matrix = store.open_matrix(data.path(), &plan_receipt, outcome.matrix_receipt())?;
    assert_eq!(matrix.series().len(), 2);
    Ok(())
}

#[test]
fn failure_returns_last_checkpoint_and_resume_skips_its_exact_prefix() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;
    let calls = RefCell::new(0_usize);

    let failure = resume_canonical_trendbar_acquisition_v1_with(
        data.path(),
        &store,
        &plan_receipt,
        None,
        |request| {
            let call = *calls.borrow();
            *calls.borrow_mut() += 1;
            if call == 1 {
                bail!("synthetic interruption after one durable cell");
            }
            Ok(publish_request(data.path(), &request, request.from_ms))
        },
    )
    .expect_err("second cell must interrupt the first run");
    assert_eq!(
        failure.stage(),
        CanonicalTrendbarAcquisitionRunStageV1::Capture
    );
    let checkpoint_receipt = failure
        .last_checkpoint_receipt()
        .expect("one completed cell must already be resumable")
        .clone();
    let checkpoint = store.open_checkpoint(data.path(), &plan_receipt, &checkpoint_receipt)?;
    assert_eq!(checkpoint.completed_cells().len(), 1);

    let resumed = RefCell::new(Vec::new());
    let outcome = resume_canonical_trendbar_acquisition_v1_with(
        data.path(),
        &store,
        &plan_receipt,
        Some(&checkpoint_receipt),
        |request| {
            resumed
                .borrow_mut()
                .push((request.symbol.clone(), request.timeframe));
            Ok(publish_request(data.path(), &request, request.from_ms))
        },
    )?;
    assert_eq!(outcome.completed_cells(), 4);
    assert_eq!(resumed.borrow().len(), 3);
    assert_eq!(
        resumed.borrow()[0],
        ("EUR/USD".to_owned(), CanonicalTimeframe::H1)
    );
    Ok(())
}

#[test]
fn wrong_window_generation_is_refused_before_checkpoint_or_matrix() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;

    let failure = resume_canonical_trendbar_acquisition_v1_with(
        data.path(),
        &store,
        &plan_receipt,
        None,
        |request| Ok(publish_request(data.path(), &request, request.from_ms + 1)),
    )
    .expect_err("different requested window must fail closed");
    assert_eq!(
        failure.stage(),
        CanonicalTrendbarAcquisitionRunStageV1::Checkpoint
    );
    assert!(failure.last_checkpoint_receipt().is_none());
    assert_eq!(
        fs_entries_with_prefix(authority.path(), "ctc1-")?,
        0,
        "no checkpoint may be published for an unbound generation"
    );
    assert_eq!(fs_entries_with_prefix(authority.path(), "ctm1-")?, 0);
    Ok(())
}

fn fs_entries_with_prefix(root: &std::path::Path, prefix: &str) -> Result<usize> {
    Ok(std::fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .count())
}
