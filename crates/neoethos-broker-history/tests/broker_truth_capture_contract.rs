use std::cell::Cell;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use neoethos_broker_history::broker_truth_capture::{
    BrokerEvidenceRowKindV1, BrokerFinancialTruthCaptureErrorCodeV1,
    BrokerFinancialTruthCaptureRequestV1, CapturedBrokerEvidencePairV1,
    CapturedBrokerEvidenceRowV1, CapturedQuoteSideV1, CapturedTickPageV1, CapturedTickV1,
    ExactBrokerTruthCaptureSessionV1, ExactConversionLegCaptureRequestV1,
    ExactConversionRouteCaptureRequestV1, ExactQuoteCaptureRequestV1, ExactQuoteInstrumentV1,
    ExactQuoteSynchronizationCaptureRequestV1, capture_and_publish_broker_financial_truth_v1,
};
use neoethos_broker_history::ctrader_messages::{
    CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
};
use neoethos_broker_truth::{
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthBundleStoreV1, EvidenceWindowV1, QuoteSideV1,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};

const ACCOUNT_ID: i64 = 7;
const PRIMARY_SYMBOL_ID: i64 = 42;
const CONVERSION_SYMBOL_ID: i64 = 43;
const FROM_MS: i64 = 1_700_000_000_000;
const TO_MS: i64 = FROM_MS + 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuoteFault {
    None,
    MissingPages,
    Truncated,
    Overlapping,
    OutOfOrder,
    WrongAccount,
    WrongWindow,
}

struct ScriptedSession {
    calls: Rc<Cell<usize>>,
    quote_requests: Vec<ExactQuoteCaptureRequestV1>,
    quote_fault: QuoteFault,
    fail_conversion_ask: bool,
    omit_symbol_contract: bool,
    omit_unrealized_pnl: bool,
    omit_close_deal: bool,
}

impl ScriptedSession {
    fn complete() -> Self {
        Self {
            calls: Rc::new(Cell::new(0)),
            quote_requests: Vec::new(),
            quote_fault: QuoteFault::None,
            fail_conversion_ask: false,
            omit_symbol_contract: false,
            omit_unrealized_pnl: false,
            omit_close_deal: false,
        }
    }

    fn record_call(&self) {
        self.calls.set(self.calls.get() + 1);
    }

    fn evidence_pair(
        &self,
        label: &str,
        synchronization_symbol_id: Option<i64>,
    ) -> CapturedBrokerEvidencePairV1 {
        let window = EvidenceWindowV1::new(FROM_MS, TO_MS).expect("scripted evidence window");
        let (raw_specs, decoded_specs) = match label {
            "symbols" => (
                vec![
                    (
                        Some(PRIMARY_SYMBOL_ID),
                        None,
                        BrokerEvidenceRowKindV1::SymbolResponse,
                        None,
                    ),
                    (
                        Some(CONVERSION_SYMBOL_ID),
                        None,
                        BrokerEvidenceRowKindV1::SymbolResponse,
                        None,
                    ),
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::AccountAssetResponse,
                        None,
                    ),
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::TraderAccountResponse,
                        None,
                    ),
                ],
                vec![
                    (
                        Some(PRIMARY_SYMBOL_ID),
                        None,
                        BrokerEvidenceRowKindV1::SymbolContract,
                        None,
                    ),
                    (
                        Some(CONVERSION_SYMBOL_ID),
                        None,
                        BrokerEvidenceRowKindV1::SymbolContract,
                        None,
                    ),
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::AccountAssetContract,
                        None,
                    ),
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::TraderAccountContract,
                        None,
                    ),
                ],
            ),
            "synchronization" => (
                vec![
                    (
                        synchronization_symbol_id,
                        Some(QuoteSideV1::Bid),
                        BrokerEvidenceRowKindV1::QuoteSessionObservation,
                        Some(window),
                    ),
                    (
                        synchronization_symbol_id,
                        Some(QuoteSideV1::Ask),
                        BrokerEvidenceRowKindV1::QuoteSessionObservation,
                        Some(window),
                    ),
                ],
                vec![(
                    synchronization_symbol_id,
                    None,
                    BrokerEvidenceRowKindV1::QuoteReplayRule,
                    Some(window),
                )],
            ),
            "pnl" => (
                vec![(
                    None,
                    None,
                    BrokerEvidenceRowKindV1::PositionUnrealizedPnlResponse,
                    None,
                )],
                vec![(
                    None,
                    None,
                    BrokerEvidenceRowKindV1::PositionUnrealizedPnl,
                    None,
                )],
            ),
            "deals" => (
                vec![
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::OpenPositionReconcileResponse,
                        Some(window),
                    ),
                    (
                        None,
                        None,
                        BrokerEvidenceRowKindV1::DealResponse,
                        Some(window),
                    ),
                ],
                vec![(
                    None,
                    None,
                    BrokerEvidenceRowKindV1::CloseDealReconciliation,
                    Some(window),
                )],
            ),
            unexpected => panic!("unexpected scripted evidence label {unexpected}"),
        };
        let make_rows = |representation: &str,
                         specs: Vec<(
            Option<i64>,
            Option<QuoteSideV1>,
            BrokerEvidenceRowKindV1,
            Option<EvidenceWindowV1>,
        )>| {
            let correlation = synchronization_symbol_id
                .map(|symbol_id| format!("symbol-{symbol_id}"))
                .unwrap_or_else(|| "account".to_owned());
            specs
                .into_iter()
                .enumerate()
                .map(|(sequence, (symbol_id, quote_side, kind, requested_window))| {
                let payload_type = match kind {
                    BrokerEvidenceRowKindV1::QuoteSessionObservation
                    | BrokerEvidenceRowKindV1::QuoteReplayRule => {
                        CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::SymbolResponse
                    | BrokerEvidenceRowKindV1::SymbolContract => {
                        CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::AccountAssetResponse
                    | BrokerEvidenceRowKindV1::AccountAssetContract => {
                        CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::TraderAccountResponse
                    | BrokerEvidenceRowKindV1::TraderAccountContract => {
                        CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::PositionUnrealizedPnlResponse
                    | BrokerEvidenceRowKindV1::PositionUnrealizedPnl => {
                        CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::OpenPositionReconcileResponse => {
                        CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE
                    }
                    BrokerEvidenceRowKindV1::DealResponse
                    | BrokerEvidenceRowKindV1::CloseDealReconciliation => {
                        CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE
                    }
                };
                CapturedBrokerEvidenceRowV1::new(
                    sequence as u64,
                    ACCOUNT_ID,
                    symbol_id,
                    quote_side,
                    kind,
                    requested_window,
                        format!("{label}-{correlation}-{representation}-{sequence}"),
                    payload_type,
                    format!(
                        "{{\"accountId\":{ACCOUNT_ID},\"label\":\"{label}\",\"representation\":\"{representation}\",\"sequence\":{sequence}}}"
                    ),
                )
            })
                .collect::<Vec<_>>()
        };
        CapturedBrokerEvidencePairV1::new(
            make_rows("raw", raw_specs),
            make_rows("decoded", decoded_specs),
        )
    }
}

impl ExactBrokerTruthCaptureSessionV1 for ScriptedSession {
    fn capture_quote_side(
        &mut self,
        request: &ExactQuoteCaptureRequestV1,
    ) -> Result<CapturedQuoteSideV1> {
        self.record_call();
        self.quote_requests.push(request.clone());
        if self.fail_conversion_ask
            && request.instrument().symbol_id() == CONVERSION_SYMBOL_ID
            && request.side() == QuoteSideV1::Ask
        {
            return Err(anyhow!("scripted conversion Ask failure"));
        }
        if self.quote_fault == QuoteFault::MissingPages {
            return Ok(CapturedQuoteSideV1::new(Vec::new()));
        }

        let first_boundary = FROM_MS + 500;
        let page_account_id = if self.quote_fault == QuoteFault::WrongAccount {
            ACCOUNT_ID + 1
        } else {
            ACCOUNT_ID
        };
        let first_window = if self.quote_fault == QuoteFault::WrongWindow {
            EvidenceWindowV1::new(FROM_MS + 1, TO_MS).expect("different valid window")
        } else {
            request.window()
        };
        let side_label = match request.side() {
            QuoteSideV1::Bid => "bid",
            QuoteSideV1::Ask => "ask",
        };
        let newer_client_msg_id = format!(
            "symbol-{}-{side_label}-page-newer",
            request.instrument().symbol_id()
        );
        let older_client_msg_id = format!(
            "symbol-{}-{side_label}-page-older",
            request.instrument().symbol_id()
        );
        let mut newer = CapturedTickPageV1::new(
            page_account_id,
            request.instrument().symbol_id(),
            request.side(),
            newer_client_msg_id.clone(),
            first_window,
            format!(
                "{{\"clientMsgId\":\"{newer_client_msg_id}\",\"payloadType\":{CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE},\"payload\":{{\"ctidTraderAccountId\":{ACCOUNT_ID},\"hasMore\":true}},\"capturedRequestSide\":\"{:?}\"}}",
                request.side()
            ),
            vec![
                CapturedTickV1::new(first_boundary, 1.000_5),
                CapturedTickV1::new(FROM_MS + 750, 1.000_75),
            ],
            true,
        );
        let older_window = EvidenceWindowV1::new(FROM_MS, first_boundary)
            .expect("valid scripted older page window");
        let mut older = CapturedTickPageV1::new(
            page_account_id,
            request.instrument().symbol_id(),
            request.side(),
            older_client_msg_id.clone(),
            older_window,
            format!(
                "{{\"clientMsgId\":\"{older_client_msg_id}\",\"payloadType\":{CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE},\"payload\":{{\"ctidTraderAccountId\":{ACCOUNT_ID},\"hasMore\":false}},\"capturedRequestSide\":\"{:?}\"}}",
                request.side()
            ),
            vec![
                CapturedTickV1::new(FROM_MS, 1.0),
                CapturedTickV1::new(FROM_MS + 250, 1.000_25),
            ],
            false,
        );

        match self.quote_fault {
            QuoteFault::None
            | QuoteFault::MissingPages
            | QuoteFault::WrongAccount
            | QuoteFault::WrongWindow => {}
            QuoteFault::Truncated => older.set_has_more_for_untrusted_capture(true),
            QuoteFault::Overlapping => older.replace_ticks_for_untrusted_capture(vec![
                CapturedTickV1::new(first_boundary, 1.000_5),
                CapturedTickV1::new(FROM_MS + 600, 1.000_6),
            ]),
            QuoteFault::OutOfOrder => newer.replace_ticks_for_untrusted_capture(vec![
                CapturedTickV1::new(FROM_MS + 750, 1.000_75),
                CapturedTickV1::new(first_boundary, 1.000_5),
            ]),
        }
        Ok(CapturedQuoteSideV1::new(vec![newer, older]))
    }

    fn capture_quote_synchronization(
        &mut self,
        _request: &ExactQuoteSynchronizationCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1> {
        self.record_call();
        Ok(self.evidence_pair("synchronization", Some(_request.instrument().symbol_id())))
    }

    fn capture_symbol_contracts(
        &mut self,
        _request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1> {
        self.record_call();
        if self.omit_symbol_contract {
            return Ok(CapturedBrokerEvidencePairV1::new(Vec::new(), Vec::new()));
        }
        Ok(self.evidence_pair("symbols", None))
    }

    fn capture_position_unrealized_pnl(
        &mut self,
        _request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1> {
        self.record_call();
        if self.omit_unrealized_pnl {
            return Ok(CapturedBrokerEvidencePairV1::new(Vec::new(), Vec::new()));
        }
        Ok(self.evidence_pair("pnl", None))
    }

    fn capture_close_deal_reconciliation(
        &mut self,
        _request: &BrokerFinancialTruthCaptureRequestV1,
    ) -> Result<CapturedBrokerEvidencePairV1> {
        self.record_call();
        if self.omit_close_deal {
            return Ok(CapturedBrokerEvidencePairV1::new(Vec::new(), Vec::new()));
        }
        Ok(self.evidence_pair("deals", None))
    }
}

fn request() -> BrokerFinancialTruthCaptureRequestV1 {
    let window = EvidenceWindowV1::new(FROM_MS, TO_MS).expect("valid evidence window");
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical cTrader identity");
    let binding = BrokerFinancialTruthBindingV1::new(
        &identity,
        "11".repeat(32),
        window,
        1,
        "EUR",
        2,
        "USD",
        3,
        "GBP",
    )
    .expect("exact broker-truth binding");
    let primary = ExactQuoteInstrumentV1::new(PRIMARY_SYMBOL_ID, "EURUSD", 1, "EUR", 2, "USD")
        .expect("primary instrument");
    let conversion_instrument =
        ExactQuoteInstrumentV1::new(CONVERSION_SYMBOL_ID, "USDGBP", 2, "USD", 3, "GBP")
            .expect("conversion instrument");
    let conversion_leg =
        ExactConversionLegCaptureRequestV1::new(2, "USD", 3, "GBP", conversion_instrument)
            .expect("conversion leg");
    let conversion_route = ExactConversionRouteCaptureRequestV1::new(
        "primary_pnl_settlement",
        2,
        "USD",
        3,
        "GBP",
        vec![conversion_leg],
    )
    .expect("conversion route");
    BrokerFinancialTruthCaptureRequestV1::new(ACCOUNT_ID, binding, primary, vec![conversion_route])
        .expect("complete capture request")
}

fn store_is_empty(store: &BrokerFinancialTruthBundleStoreV1) -> bool {
    !store.root().exists()
        || std::fs::read_dir(store.root())
            .expect("read evidence store")
            .next()
            .is_none()
}

#[test]
fn producer_retains_explicit_bid_ask_pages_and_every_required_conversion_leg() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
    let mut session = ScriptedSession::complete();
    let request = request();
    let publication_started = Cell::new(false);

    let receipt = capture_and_publish_broker_financial_truth_v1(
        &mut session,
        &request,
        temp.path().join("capture-work"),
        &store,
        || false,
        || {
            publication_started.set(true);
            Ok(())
        },
    )
    .expect("complete structural capture publishes one immutable bundle");
    assert!(publication_started.get());

    let verified = store
        .open_exact(&receipt, request.binding())
        .expect("exact receipt reopens exact capture binding");
    assert_eq!(session.quote_requests.len(), 4);
    assert_eq!(session.quote_requests[0].side(), QuoteSideV1::Bid);
    assert_eq!(session.quote_requests[1].side(), QuoteSideV1::Ask);
    assert_eq!(session.quote_requests[2].side(), QuoteSideV1::Bid);
    assert_eq!(session.quote_requests[3].side(), QuoteSideV1::Ask);
    assert_eq!(
        session.quote_requests[0].instrument().symbol_id(),
        PRIMARY_SYMBOL_ID
    );
    assert_eq!(
        session.quote_requests[2].instrument().symbol_id(),
        CONVERSION_SYMBOL_ID
    );
    assert!(
        session
            .quote_requests
            .iter()
            .all(|quote| quote.account_id() == ACCOUNT_ID && quote.window() == request.window())
    );

    let manifest = verified.manifest();
    assert_eq!(
        manifest
            .primary_quotes()
            .bid()
            .capture()
            .raw_envelopes()
            .row_count(),
        2
    );
    assert_eq!(
        manifest
            .primary_quotes()
            .bid()
            .capture()
            .decoded_records()
            .row_count(),
        4
    );
    assert_eq!(
        manifest
            .primary_quotes()
            .ask()
            .capture()
            .raw_envelopes()
            .row_count(),
        2
    );
    assert_eq!(manifest.conversion_routes().len(), 1);
    assert_eq!(manifest.conversion_routes()[0].legs().len(), 1);
    assert_eq!(
        neoethos_data::core::vortex_io::read_vortex_row_count(
            verified.artifact_path(manifest.primary_quotes().bid().capture().raw_envelopes())
        )
        .expect("read retained raw page table"),
        2
    );
}

#[test]
fn producer_refuses_missing_truncated_overlapping_and_out_of_order_quote_pages() {
    for (fault, expected) in [
        (
            QuoteFault::MissingPages,
            BrokerFinancialTruthCaptureErrorCodeV1::MissingQuotePages,
        ),
        (
            QuoteFault::Truncated,
            BrokerFinancialTruthCaptureErrorCodeV1::TruncatedQuotePages,
        ),
        (
            QuoteFault::Overlapping,
            BrokerFinancialTruthCaptureErrorCodeV1::OverlappingQuotePages,
        ),
        (
            QuoteFault::OutOfOrder,
            BrokerFinancialTruthCaptureErrorCodeV1::OutOfOrderQuoteRows,
        ),
        (
            QuoteFault::WrongAccount,
            BrokerFinancialTruthCaptureErrorCodeV1::EvidenceAccountMismatch,
        ),
        (
            QuoteFault::WrongWindow,
            BrokerFinancialTruthCaptureErrorCodeV1::InvalidQuotePage,
        ),
    ] {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
        let mut session = ScriptedSession {
            quote_fault: fault,
            ..ScriptedSession::complete()
        };
        let error = capture_and_publish_broker_financial_truth_v1(
            &mut session,
            &request(),
            temp.path().join("capture-work"),
            &store,
            || false,
            || Ok(()),
        )
        .expect_err("invalid page provenance must fail before publication");
        assert_eq!(error.code(), expected, "fault {fault:?}");
        assert!(store_is_empty(&store), "fault {fault:?} published a bundle");
    }
}

#[test]
fn conversion_capture_failure_leaves_no_published_bundle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
    let mut session = ScriptedSession {
        fail_conversion_ask: true,
        ..ScriptedSession::complete()
    };

    let error = capture_and_publish_broker_financial_truth_v1(
        &mut session,
        &request(),
        temp.path().join("capture-work"),
        &store,
        || false,
        || Ok(()),
    )
    .expect_err("missing required conversion Ask must fail closed");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthCaptureErrorCodeV1::CaptureFailed
    );
    assert!(store_is_empty(&store));
}

#[test]
fn every_symbol_pnl_and_close_deal_family_is_required_before_publication() {
    for (omit, expected) in [
        (
            "symbol",
            BrokerFinancialTruthCaptureErrorCodeV1::MissingSymbolContracts,
        ),
        (
            "pnl",
            BrokerFinancialTruthCaptureErrorCodeV1::MissingUnrealizedPnl,
        ),
        (
            "deal",
            BrokerFinancialTruthCaptureErrorCodeV1::MissingCloseDealReconciliation,
        ),
    ] {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
        let mut session = ScriptedSession {
            omit_symbol_contract: omit == "symbol",
            omit_unrealized_pnl: omit == "pnl",
            omit_close_deal: omit == "deal",
            ..ScriptedSession::complete()
        };
        let error = capture_and_publish_broker_financial_truth_v1(
            &mut session,
            &request(),
            temp.path().join("capture-work"),
            &store,
            || false,
            || Ok(()),
        )
        .expect_err("an omitted financial evidence family must fail closed");
        assert_eq!(error.code(), expected, "omitted family {omit}");
        assert!(store_is_empty(&store));
    }
}

#[test]
fn cancellation_observed_after_quote_capture_still_prevents_publication() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
    let mut session = ScriptedSession::complete();
    let calls = Rc::clone(&session.calls);
    let publication_started = Cell::new(false);

    let error = capture_and_publish_broker_financial_truth_v1(
        &mut session,
        &request(),
        temp.path().join("capture-work"),
        &store,
        || calls.get() >= 2,
        || {
            publication_started.set(true);
            Ok(())
        },
    )
    .expect_err("observed cancellation must stop before encoding/publication");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthCaptureErrorCodeV1::Cancelled
    );
    assert_eq!(session.quote_requests.len(), 2);
    assert!(!publication_started.get());
    assert!(store_is_empty(&store));
}

#[test]
fn producer_is_receipt_only_and_cannot_open_a_second_transport_or_capability() {
    let capture_source = include_str!("../src/broker_truth_capture.rs");
    let vortex_source = include_str!("../src/broker_truth_vortex.rs");

    assert!(capture_source.contains("ExactBrokerTruthCaptureSessionV1"));
    assert!(capture_source.contains("let _publication_permit = begin_publication()?"));
    assert!(!capture_source.contains("ProductionCTraderOpenApiTransport::new"));
    assert!(!capture_source.contains("current_broker_financial_truth_capability_v1"));
    assert!(!capture_source.contains("std::env"));
    assert!(!capture_source.contains("OnceLock"));
    assert!(vortex_source.contains("neoethos_data::core::vortex_io::write_vortex_array"));
    assert!(!vortex_source.contains("BrokerFinancialTruthCapabilityV1"));
}
