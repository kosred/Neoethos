use super::ctrader_historical_admission::{
    CTraderIoBoundaryError, CTraderIoPhase, HistoricalRequestCancelled,
};
use super::ctrader_messages::{
    CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE, CTraderCannotRouteRequestError,
    ctrader_historical_session_error_from_response,
};
use super::service::{
    AuthenticatedHistoricalSessionRequest, BrokerHistoryConflict, HistoricalCaptureRequest,
    HistoricalCaptureStatus, HistoricalCaptureTarget, HistoricalCredentials,
    HistoricalFetchCancelResult, HistoricalFetchStartFailure, HistoricalPage,
    HistoricalPageRequest, HistoricalSeriesCapture, HistoricalSession, HistoricalSessionConnector,
    ProductionPublication, ResolvedHistoricalSymbol, begin_process_historical_capture,
    cancel_process_historical_capture, capture_with_connector_and_publication_hook,
    connect_historical_session_with_route_retry, is_historical_capture_cancelled,
    process_historical_capture_status,
};
use super::{HistoricalFetchCancelOutcome, HistoricalFetchRegistry};
use anyhow::{Result, anyhow};
use neoethos_core::CanonicalTimeframe;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity,
    SelectedDatasetGenerationV1, load_canonical_timeframe,
};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const START_2016_MS: i64 = 1_451_606_400_000;
const SERVER: &str = "demo.ctraderapi.com";
const ACCOUNT_ID: i64 = 42;
const SYMBOL_ID: i64 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Connect,
    ApplicationAuth,
    AccountAuth,
    SymbolsList,
    SymbolDetail,
    Page {
        client_msg_id: String,
        timeframe: CanonicalTimeframe,
        from_timestamp_ms: i64,
        to_timestamp_ms: i64,
        count: Option<u32>,
    },
}

struct FakeConnector {
    events: Arc<Mutex<Vec<Event>>>,
    pages: Mutex<VecDeque<HistoricalPage>>,
    after_first_page: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
    fail_page_with_wrapped_cancellation: bool,
}

struct FakeSession {
    events: Arc<Mutex<Vec<Event>>>,
    pages: VecDeque<HistoricalPage>,
    resolved: ResolvedHistoricalSymbol,
    after_first_page: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
    next_page_sequence: u64,
    fail_page_with_wrapped_cancellation: bool,
}

impl HistoricalSession for FakeSession {
    fn resolved_symbol(&self) -> &ResolvedHistoricalSymbol {
        &self.resolved
    }

    fn next_page(&mut self, request: HistoricalPageRequest) -> Result<HistoricalPage> {
        self.next_page_sequence += 1;
        self.events.lock().expect("event lock").push(Event::Page {
            client_msg_id: format!("history-trendbars-{}", self.next_page_sequence),
            timeframe: request.timeframe,
            from_timestamp_ms: request.from_timestamp_ms,
            to_timestamp_ms: request.to_timestamp_ms,
            count: request.count,
        });
        if self.fail_page_with_wrapped_cancellation {
            self.fail_page_with_wrapped_cancellation = false;
            let boundary = CTraderIoBoundaryError::Cancelled {
                phase: CTraderIoPhase::ResponseRead,
            };
            let io_error = std::io::Error::new(std::io::ErrorKind::Interrupted, boundary);
            return Err(anyhow::Error::new(tungstenite::Error::Io(io_error))
                .context("fake persistent-session response read"));
        }
        let page = self
            .pages
            .pop_front()
            .ok_or_else(|| anyhow!("unexpected extra historical page"))?;
        if let Some(cancel) = self
            .after_first_page
            .lock()
            .expect("page callback lock")
            .take()
        {
            cancel();
        }
        Ok(page)
    }
}

impl HistoricalSessionConnector for FakeConnector {
    type Session = FakeSession;

    fn connect_authenticated(
        &self,
        request: &AuthenticatedHistoricalSessionRequest,
        _cancellation: &super::ctrader_historical_admission::HistoricalRequestCancellation,
    ) -> Result<Self::Session> {
        let mut events = self.events.lock().expect("event lock");
        events.extend([
            Event::Connect,
            Event::ApplicationAuth,
            Event::AccountAuth,
            Event::SymbolsList,
            Event::SymbolDetail,
        ]);
        drop(events);
        Ok(FakeSession {
            events: Arc::clone(&self.events),
            pages: self.pages.lock().expect("page lock").clone(),
            after_first_page: Arc::clone(&self.after_first_page),
            resolved: ResolvedHistoricalSymbol {
                environment: request.environment,
                server: request.server.clone(),
                account_id: request.account_id,
                symbol_id: SYMBOL_ID,
                symbol_name: "EURUSD".to_owned(),
            },
            next_page_sequence: 0,
            fail_page_with_wrapped_cancellation: self.fail_page_with_wrapped_cancellation,
        })
    }
}

fn canonical_identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical cTrader identity")
}

fn selected(discriminator: u64) -> SelectedDatasetGenerationV1 {
    SelectedDatasetGenerationV1::new(
        canonical_identity(),
        format!("g1-{discriminator:064x}.vortex"),
        format!("{:064x}", discriminator + 1_000),
    )
    .expect("synthetic exact receipt")
}

fn request(root: &Path, target: HistoricalCaptureTarget) -> HistoricalCaptureRequest {
    HistoricalCaptureRequest {
        symbol: "EURUSD".to_owned(),
        timeframe: CanonicalTimeframe::M1,
        from_ms: START_2016_MS,
        to_ms: START_2016_MS + 6 * 24 * 60 * 60 * 1_000,
        data_root: root.to_path_buf(),
        target,
    }
}

fn credentials() -> HistoricalCredentials {
    HistoricalCredentials {
        client_id: "test-client".to_owned(),
        client_secret: "test-secret".to_owned(),
        access_token: "test-token".to_owned(),
        environment: super::service::BrokerEnvironment::Demo,
        account_id: ACCOUNT_ID,
    }
}

#[test]
fn one_run_connects_authenticates_and_resolves_once_then_streams_unique_direct_pages() {
    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([
            HistoricalPage::fixture_m1(START_2016_MS + 3 * 24 * 60 * 60 * 1_000, 2),
            HistoricalPage::fixture_m1(START_2016_MS, 2),
        ])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut publication_seen = None;

    let outcome = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |publication: &ProductionPublication| {
            publication_seen = Some(
                SelectedDatasetGenerationV1::from_manifest(publication.manifest())
                    .expect("publication receipt"),
            );
            Ok(())
        },
    )
    .expect("bounded capture");

    let events = events.lock().expect("event lock");
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::Connect)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::ApplicationAuth)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::AccountAuth)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::SymbolsList)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::SymbolDetail)
            .count(),
        1
    );
    let page_events = events
        .iter()
        .filter_map(|event| match event {
            Event::Page {
                client_msg_id,
                timeframe,
                ..
            } => Some((client_msg_id, timeframe)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(page_events.len(), 2);
    assert_ne!(page_events[0].0, page_events[1].0);
    assert!(
        page_events
            .iter()
            .all(|(_, timeframe)| **timeframe == CanonicalTimeframe::M1)
    );
    assert_eq!(
        outcome.selected_generation,
        publication_seen.expect("hook receipt")
    );
    assert_eq!(
        outcome.selected_generation.identity(),
        &canonical_identity()
    );
}

#[test]
fn bit_identical_adjacent_broker_duplicate_is_collapsed_without_resampling() {
    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let first = super::service::HistoricalBar {
        timestamp_ms: START_2016_MS,
        open: 1.1000,
        high: 1.1003,
        low: 1.0997,
        close: 1.1001,
        volume: Some(10),
    };
    let second = super::service::HistoricalBar {
        timestamp_ms: START_2016_MS + 60_000,
        open: 1.1001,
        high: 1.1004,
        low: 1.0998,
        close: 1.1002,
        volume: Some(11),
    };
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([HistoricalPage {
            symbol_id: SYMBOL_ID,
            timeframe: CanonicalTimeframe::M1,
            bars: vec![first.clone(), first, second],
            has_more: false,
        }])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut capture_request = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    capture_request.to_ms = START_2016_MS + 60 * 60 * 1_000;

    let outcome = capture_with_connector_and_publication_hook(
        capture_request,
        credentials(),
        &active,
        &connector,
        |_| Ok(()),
    )
    .expect("an exact duplicate wire row must not create a second canonical candle");

    assert_eq!(outcome.bar_count, 2);
    let reopened =
        neoethos_data::load_exact_canonical_timeframe(root.path(), &outcome.selected_generation)
            .expect("reopen exact generation");
    assert_eq!(
        reopened.ohlcv().timestamp.as_deref(),
        Some([START_2016_MS, START_2016_MS + 60_000].as_slice())
    );
}

#[test]
fn same_timestamp_with_different_broker_payload_fails_before_publication() {
    let root = tempfile::tempdir().expect("data root");
    let first = super::service::HistoricalBar {
        timestamp_ms: START_2016_MS,
        open: 1.1000,
        high: 1.1003,
        low: 1.0997,
        close: 1.1001,
        volume: Some(10),
    };
    let mut conflicting = first.clone();
    conflicting.close = 1.1002;
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::from([HistoricalPage {
            symbol_id: SYMBOL_ID,
            timeframe: CanonicalTimeframe::M1,
            bars: vec![first, conflicting],
            has_more: false,
        }])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut capture_request = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    capture_request.to_ms = START_2016_MS + 60 * 60 * 1_000;

    let error = capture_with_connector_and_publication_hook(
        capture_request,
        credentials(),
        &active,
        &connector,
        |_| panic!("conflicting duplicate must not publish"),
    )
    .expect_err("same timestamp with a different payload must fail closed");

    assert!(
        error
            .to_string()
            .contains("same timestamp with different OHLCV"),
        "unexpected conflict error: {error:#}"
    );
}

#[test]
fn series_capture_reuses_one_authenticated_symbol_session_across_canonical_timeframes() {
    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([
            HistoricalPage::fixture_m1(START_2016_MS, 1),
            HistoricalPage {
                symbol_id: SYMBOL_ID,
                timeframe: CanonicalTimeframe::H1,
                bars: vec![super::service::HistoricalBar {
                    timestamp_ms: START_2016_MS,
                    open: 1.1000,
                    high: 1.1003,
                    low: 1.0997,
                    close: 1.1001,
                    volume: Some(10),
                }],
                has_more: false,
            },
        ])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut series = HistoricalSeriesCapture::new(credentials(), connector);
    let mut m1 = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    m1.to_ms = START_2016_MS + 60 * 60 * 1_000;
    let mut h1 = m1.clone();
    h1.timeframe = CanonicalTimeframe::H1;

    series
        .capture_with_publication_hook(m1, &active, |_| Ok(()))
        .expect("M1 generation");
    series
        .capture_with_publication_hook(h1, &active, |_| Ok(()))
        .expect("H1 generation");

    let events = events.lock().expect("event lock");
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::Connect)
            .count(),
        1,
        "one symbol must use one authenticated socket across every canonical timeframe"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Page { .. }))
            .count(),
        2
    );
}

#[test]
fn cannot_route_connection_errors_retry_the_same_exact_session_request_with_bounded_backoff() {
    let cancellation = super::ctrader_historical_admission::HistoricalRequestCancellation::new();
    let mut attempts = 0_usize;
    let mut waits = Vec::new();

    let connected = connect_historical_session_with_route_retry(
        &cancellation,
        || {
            attempts += 1;
            if attempts < 3 {
                let response = serde_json::json!({
                    "clientMsgId": "history-application-auth",
                    "payloadType": CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
                    "payload": {
                        "errorCode": "CANT_ROUTE_REQUEST",
                        "description": "Cannot route request"
                    }
                })
                .to_string();
                return Err(ctrader_historical_session_error_from_response(&response)?
                    .context("same exact historical application-auth request"));
            }
            Ok("connected")
        },
        |delay| {
            waits.push(delay);
            Ok(())
        },
    )
    .expect("transient CANT_ROUTE_REQUEST should reconnect in-place");

    assert_eq!(connected, "connected");
    assert_eq!(attempts, 3);
    assert_eq!(waits, vec![Duration::from_secs(1), Duration::from_secs(2)]);
    assert!(
        ctrader_historical_session_error_from_response(
            &serde_json::json!({
                "clientMsgId": "history-application-auth",
                "payloadType": CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
                "payload": {
                    "errorCode": "CANT_ROUTE_REQUEST",
                    "description": "Cannot route request"
                }
            })
            .to_string()
        )
        .expect("typed route error")
        .downcast_ref::<CTraderCannotRouteRequestError>()
        .is_some()
    );
}

#[test]
fn weekly_capture_ignores_one_leading_containing_bar_without_resampling() {
    const DAY_MS: i64 = 24 * 60 * 60 * 1_000;

    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let leading_open_ms = START_2016_MS - 4 * DAY_MS;
    let first_in_range_open_ms = START_2016_MS + 3 * DAY_MS;
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([HistoricalPage {
            symbol_id: SYMBOL_ID,
            timeframe: CanonicalTimeframe::W1,
            bars: vec![
                super::service::HistoricalBar {
                    timestamp_ms: leading_open_ms,
                    open: 1.0900,
                    high: 1.1100,
                    low: 1.0800,
                    close: 1.1000,
                    volume: Some(90),
                },
                super::service::HistoricalBar {
                    timestamp_ms: first_in_range_open_ms,
                    open: 1.1000,
                    high: 1.1200,
                    low: 1.0900,
                    close: 1.1100,
                    volume: Some(100),
                },
            ],
            has_more: false,
        }])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut capture_request = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    capture_request.timeframe = CanonicalTimeframe::W1;
    capture_request.to_ms = START_2016_MS + 7 * DAY_MS;

    let outcome = capture_with_connector_and_publication_hook(
        capture_request,
        credentials(),
        &active,
        &connector,
        |_| Ok(()),
    )
    .expect("leading broker overlap must not replace the exact bar-open window");

    assert_eq!(outcome.bar_count, 1);
    assert_eq!(outcome.oldest_ms, first_in_range_open_ms);
    let reopened = load_canonical_timeframe(root.path(), outcome.selected_generation.identity())
        .expect("reopen exact weekly generation");
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![first_in_range_open_ms])
    );
    assert_eq!(reopened.ohlcv().open, vec![1.1000]);
}

#[test]
fn fixed_timeframe_capture_ignores_one_native_leading_containing_bar_without_reanchoring() {
    const MINUTE_MS: i64 = 60_000;

    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let leading_open_ms = START_2016_MS;
    let requested_from_ms = START_2016_MS + MINUTE_MS;
    let first_in_range_open_ms = START_2016_MS + 3 * MINUTE_MS;
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([HistoricalPage {
            symbol_id: SYMBOL_ID,
            timeframe: CanonicalTimeframe::M3,
            bars: vec![
                super::service::HistoricalBar {
                    timestamp_ms: leading_open_ms,
                    open: 1.0900,
                    high: 1.1100,
                    low: 1.0800,
                    close: 1.1000,
                    volume: Some(90),
                },
                super::service::HistoricalBar {
                    timestamp_ms: first_in_range_open_ms,
                    open: 1.1000,
                    high: 1.1200,
                    low: 1.0900,
                    close: 1.1100,
                    volume: Some(100),
                },
            ],
            has_more: false,
        }])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut capture_request = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    capture_request.timeframe = CanonicalTimeframe::M3;
    capture_request.from_ms = requested_from_ms;
    capture_request.to_ms = START_2016_MS + 10 * MINUTE_MS;

    let outcome = capture_with_connector_and_publication_hook(
        capture_request,
        credentials(),
        &active,
        &connector,
        |_| Ok(()),
    )
    .expect("one native M3 containing bar may precede a non-aligned half-open request");

    assert_eq!(outcome.bar_count, 1);
    assert_eq!(outcome.oldest_ms, first_in_range_open_ms);
    let reopened = load_canonical_timeframe(root.path(), outcome.selected_generation.identity())
        .expect("reopen exact M3 generation");
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![first_in_range_open_ms])
    );
    assert_eq!(reopened.ohlcv().open, vec![1.1000]);
}

#[test]
fn adjacent_fixed_timeframe_chunks_compare_filtered_half_open_boundaries() {
    const MINUTE_MS: i64 = 60_000;
    const M3_CHUNK_MS: i64 = 4_320 * 3 * MINUTE_MS;

    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let requested_from_ms = START_2016_MS + MINUTE_MS;
    let logical_boundary_ms = requested_from_ms + 3 * MINUTE_MS;
    let shared_containing_open_ms = logical_boundary_ms - MINUTE_MS;
    let first_newer_open_ms = shared_containing_open_ms + 3 * MINUTE_MS;
    let bar = |timestamp_ms, open| super::service::HistoricalBar {
        timestamp_ms,
        open,
        high: open + 0.0003,
        low: open - 0.0003,
        close: open + 0.0001,
        volume: Some(10),
    };
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([
            HistoricalPage {
                symbol_id: SYMBOL_ID,
                timeframe: CanonicalTimeframe::M3,
                bars: vec![
                    bar(shared_containing_open_ms, 1.1000),
                    bar(first_newer_open_ms, 1.1001),
                ],
                has_more: false,
            },
            HistoricalPage {
                symbol_id: SYMBOL_ID,
                timeframe: CanonicalTimeframe::M3,
                bars: vec![
                    bar(START_2016_MS, 1.0999),
                    bar(shared_containing_open_ms, 1.1000),
                ],
                has_more: false,
            },
        ])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut capture_request = request(root.path(), HistoricalCaptureTarget::NewIdentity);
    capture_request.timeframe = CanonicalTimeframe::M3;
    capture_request.from_ms = requested_from_ms;
    capture_request.to_ms = requested_from_ms + M3_CHUNK_MS + 3 * MINUTE_MS;

    let outcome = capture_with_connector_and_publication_hook(
        capture_request,
        credentials(),
        &active,
        &connector,
        |_| Ok(()),
    )
    .expect("filtered half-open chunks must not overlap on their shared containing bar");

    assert_eq!(outcome.bar_count, 2);
    let reopened =
        neoethos_data::load_exact_canonical_timeframe(root.path(), &outcome.selected_generation)
            .expect("reopen exact M3 generation");
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![shared_containing_open_ms, first_newer_open_ms])
    );
}

#[test]
fn explicit_has_more_reissues_same_logical_window_with_strictly_older_exclusive_to_on_same_session_and_publishes_all_rows()
 {
    const DAY_MS: i64 = 24 * 60 * 60 * 1_000;

    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let first_page_oldest_ms = START_2016_MS + 5 * DAY_MS;
    let logical_chunk_from_ms = START_2016_MS + 3 * DAY_MS;

    let mut incomplete_newest_page = HistoricalPage::fixture_m1(first_page_oldest_ms, 2);
    incomplete_newest_page.has_more = true;
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::from([
            incomplete_newest_page,
            HistoricalPage::fixture_m1(logical_chunk_from_ms, 2),
            HistoricalPage::fixture_m1(START_2016_MS, 2),
        ])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");

    let outcome = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |_| Ok(()),
    )
    .expect("explicit hasMore pages must be traversed to a terminal response");

    let events = events.lock().expect("event lock");
    assert_eq!(
        events
            .iter()
            .filter(|event| **event == Event::Connect)
            .count(),
        1,
        "all subpages must reuse the one authenticated session"
    );
    let page_requests = events
        .iter()
        .filter_map(|event| match event {
            Event::Page {
                client_msg_id,
                timeframe,
                from_timestamp_ms,
                to_timestamp_ms,
                count,
            } => Some((
                client_msg_id,
                *timeframe,
                *from_timestamp_ms,
                *to_timestamp_ms,
                *count,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(page_requests.len(), 3);
    assert_eq!(page_requests[0].1, CanonicalTimeframe::M1);
    assert_eq!(page_requests[0].2, logical_chunk_from_ms);
    assert_eq!(page_requests[1].2, logical_chunk_from_ms);
    assert_eq!(page_requests[1].3, first_page_oldest_ms - 1);
    assert_eq!(page_requests[0].4, None);
    assert_eq!(page_requests[1].4, None);
    assert_ne!(page_requests[0].0, page_requests[1].0);
    drop(events);

    assert_eq!(outcome.bar_count, 6);
    assert_eq!(outcome.oldest_ms, START_2016_MS);
    let reopened =
        neoethos_data::load_exact_canonical_timeframe(root.path(), &outcome.selected_generation)
            .expect("reopen exact published generation");
    assert_eq!(reopened.len(), 6);
    assert_eq!(
        reopened.ohlcv().timestamp.as_deref(),
        Some(
            [
                START_2016_MS,
                START_2016_MS + 60_000,
                logical_chunk_from_ms,
                logical_chunk_from_ms + 60_000,
                first_page_oldest_ms,
                first_page_oldest_ms + 60_000,
            ]
            .as_slice()
        )
    );
}

#[test]
fn has_more_with_an_empty_page_fails_before_publication() {
    const DAY_MS: i64 = 24 * 60 * 60 * 1_000;

    let root = tempfile::tempdir().expect("data root");
    let mut empty_page = HistoricalPage::fixture_m1(START_2016_MS + 3 * DAY_MS, 0);
    empty_page.has_more = true;
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::from([empty_page])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");

    let error = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |_| panic!("an empty hasMore page must never publish"),
    )
    .expect_err("an empty hasMore page must fail closed");

    assert!(
        error
            .to_string()
            .contains("hasMore with an empty EURUSD M1 page"),
        "unexpected empty hasMore error: {error:#}"
    );
}

#[test]
fn has_more_without_a_strictly_older_cursor_fails_before_publication() {
    const DAY_MS: i64 = 24 * 60 * 60 * 1_000;

    let root = tempfile::tempdir().expect("data root");
    let logical_chunk_from_ms = START_2016_MS + 3 * DAY_MS;
    let mut non_progressing_page = HistoricalPage::fixture_m1(logical_chunk_from_ms, 2);
    non_progressing_page.has_more = true;
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::from([non_progressing_page])),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");

    let error = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |_| panic!("a non-progressing hasMore page must never publish"),
    )
    .expect_err("a non-progressing hasMore page must fail closed");

    assert!(
        error
            .to_string()
            .contains("hasMore without a strictly older EURUSD M1 cursor"),
        "unexpected non-progressing hasMore error: {error:#}"
    );
}

#[test]
fn stale_exact_binding_fails_before_connect_and_never_publishes() {
    let root = tempfile::tempdir().expect("data root");
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::new()),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let result = capture_with_connector_and_publication_hook(
        request(
            root.path(),
            HistoricalCaptureTarget::SelectedGeneration(selected(9)),
        ),
        credentials(),
        &active,
        &connector,
        |_| panic!("stale selection must not reach publication"),
    );

    assert!(result.is_err());
    assert!(connector.events.lock().expect("event lock").is_empty());
}

#[test]
fn selected_request_identity_mismatch_is_typed_before_connect() {
    let root = tempfile::tempdir().expect("data root");
    let events = Arc::new(Mutex::new(Vec::new()));
    let connector = FakeConnector {
        events: Arc::clone(&events),
        pages: Mutex::new(VecDeque::new()),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: false,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");
    let mut mismatched = request(
        root.path(),
        HistoricalCaptureTarget::SelectedGeneration(selected(10)),
    );
    mismatched.symbol = "GBPUSD".to_owned();

    let error = capture_with_connector_and_publication_hook(
        mismatched,
        credentials(),
        &active,
        &connector,
        |_| panic!("identity mismatch must not publish"),
    )
    .expect_err("request identity mismatch must fail closed");
    let conflict = error
        .downcast_ref::<BrokerHistoryConflict>()
        .expect("request identity mismatch remains a typed conflict");
    assert_eq!(conflict.response_code(), "BROKER_IDENTITY_MISMATCH");
    assert!(events.lock().expect("event lock").is_empty());
}

#[test]
fn accepted_cancel_after_a_page_prevents_publication_and_releases_the_run() {
    let root = tempfile::tempdir().expect("data root");
    let registry = Arc::new(HistoricalFetchRegistry::new());
    let active = registry.try_start().expect("active run");
    let run_id = active.run_id();
    let cancellation_registry = Arc::clone(&registry);
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::from([
            HistoricalPage::fixture_m1(START_2016_MS + 3 * 24 * 60 * 60 * 1_000, 2),
            HistoricalPage::fixture_m1(START_2016_MS, 2),
        ])),
        after_first_page: Arc::new(Mutex::new(Some(Box::new(move || {
            assert_eq!(
                cancellation_registry.cancel_run(run_id),
                HistoricalFetchCancelOutcome::Cancelled { run_id }
            );
        })))),
        fail_page_with_wrapped_cancellation: false,
    };
    let result = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |_| panic!("accepted cancellation must not publish"),
    );
    assert!(result.is_err());
    drop(active);
    registry.try_start().expect("RAII released the run");
}

#[test]
fn wrapped_transport_cancellation_is_normalized_to_typed_fetch_cancellation() {
    let root = tempfile::tempdir().expect("data root");
    let connector = FakeConnector {
        events: Arc::new(Mutex::new(Vec::new())),
        pages: Mutex::new(VecDeque::new()),
        after_first_page: Arc::new(Mutex::new(None)),
        fail_page_with_wrapped_cancellation: true,
    };
    let registry = HistoricalFetchRegistry::new();
    let active = registry.try_start().expect("active run");

    let error = capture_with_connector_and_publication_hook(
        request(root.path(), HistoricalCaptureTarget::NewIdentity),
        credentials(),
        &active,
        &connector,
        |_| panic!("transport cancellation must not publish"),
    )
    .expect_err("wrapped response-read cancellation must fail the capture");

    assert!(
        error.downcast_ref::<HistoricalRequestCancelled>().is_some(),
        "wrapped transport cancellation was not normalized: {error:#}"
    );
}

#[test]
fn process_status_maps_the_exact_active_run_phase_and_releases_with_raii() {
    assert_eq!(process_historical_capture_status(), None);
    let active = begin_process_historical_capture().expect("process historical capture");
    let run_id = active.run_id();
    assert_eq!(
        process_historical_capture_status(),
        Some(HistoricalCaptureStatus {
            run_id,
            phase: "capturing",
        })
    );
    assert!(matches!(
        begin_process_historical_capture(),
        Err(HistoricalFetchStartFailure::AlreadyActive { active_run_id })
            if active_run_id == run_id
    ));

    assert_eq!(
        cancel_process_historical_capture(run_id),
        HistoricalFetchCancelResult::Cancelled { run_id }
    );
    let wrapped_cancel = anyhow!(HistoricalRequestCancelled).context("shared app boundary");
    assert!(is_historical_capture_cancelled(wrapped_cancel.as_ref()));
    assert_eq!(
        process_historical_capture_status(),
        Some(HistoricalCaptureStatus {
            run_id,
            phase: "cancellation_requested",
        })
    );

    drop(active);
    assert_eq!(process_historical_capture_status(), None);
}
