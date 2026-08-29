use std::collections::HashSet;

use anyhow::Result;
use neoethos_broker_history::broker_truth_capture::{
    BrokerEvidenceRowKindV2, BrokerFinancialTruthCaptureErrorCodeV2,
    BrokerFinancialTruthCaptureRequestV2, CapturedBrokerEvidencePairV2,
    CapturedBrokerEvidenceRowV2, CapturedQuoteSynchronizationV2,
    ExactConversionRouteCaptureRequestV2, ExactQuoteInstrumentV2,
    capture_and_publish_broker_financial_truth_v2,
};
use neoethos_broker_history::broker_truth_ctrader::{
    CTraderBrokerTruthAdapterV2, CTraderBrokerTruthSameSessionV2,
    ReviewedCTraderQuoteSynchronizationV2,
};
use neoethos_broker_history::ctrader_messages::{
    CTRADER_OA_ASSET_LIST_REQUEST_PAYLOAD_TYPE, CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_REQUEST_PAYLOAD_TYPE,
    CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE, CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE, CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE, CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
    CTraderOpenApiJsonMessage,
};
use neoethos_broker_truth::{
    BrokerFinancialTruthBindingV1, BrokerFinancialTruthBundleStoreV1, EvidenceWindowV1,
    MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2, QuoteSideV1, ReviewedQuoteReplayRuleIdentityV2,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use serde_json::{Value, json};

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const FROM_MS: i64 = 1_700_000_000_000;
const TO_MS: i64 = FROM_MS + MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2 + 1_000;
const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickScript {
    Terminal,
    Paged,
    EmptyHasMore,
    NonProgress,
    OutOfWindow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DealScript {
    Terminal,
    Paged,
    EmptyHasMore,
}

struct ScriptedSameSession {
    requests: Vec<CTraderOpenApiJsonMessage>,
    first_tick_windows: HashSet<(i32, i64)>,
    tick_script: TickScript,
    deal_script: DealScript,
    deal_requests: usize,
}

impl ScriptedSameSession {
    fn new(tick_script: TickScript, deal_script: DealScript) -> Self {
        Self {
            requests: Vec::new(),
            first_tick_windows: HashSet::new(),
            tick_script,
            deal_script,
            deal_requests: 0,
        }
    }

    fn response(message: &CTraderOpenApiJsonMessage, payload_type: u32, payload: Value) -> String {
        json!({
            "clientMsgId": message.client_msg_id,
            "payloadType": payload_type,
            "payload": payload,
        })
        .to_string()
    }

    fn tick_response(&mut self, message: &CTraderOpenApiJsonMessage) -> String {
        let side = message.payload["type"].as_i64().expect("tick side") as i32;
        let from = message.payload["fromTimestamp"]
            .as_i64()
            .expect("tick from");
        let to = message.payload["toTimestamp"].as_i64().expect("tick to");
        let first = self.first_tick_windows.insert((side, from));
        let (ticks, has_more) = match self.tick_script {
            TickScript::Terminal => (vec![json!({"timestamp": from, "tick": 110_000})], false),
            TickScript::Paged if first => {
                let midpoint = from + (to - from) / 2;
                (vec![json!({"timestamp": midpoint, "tick": 110_000})], true)
            }
            TickScript::Paged => (vec![json!({"timestamp": from, "tick": 110_000})], false),
            TickScript::EmptyHasMore => (Vec::new(), true),
            TickScript::NonProgress => (vec![json!({"timestamp": from, "tick": 110_000})], true),
            TickScript::OutOfWindow => (vec![json!({"timestamp": to, "tick": 110_000})], false),
        };
        Self::response(
            message,
            CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
            json!({
                "ctidTraderAccountId": ACCOUNT_ID,
                "hasMore": has_more,
                "tickData": ticks,
            }),
        )
    }

    fn deal_response(&mut self, message: &CTraderOpenApiJsonMessage) -> String {
        let from = message.payload["fromTimestamp"]
            .as_i64()
            .expect("deal from");
        self.deal_requests += 1;
        let (deals, has_more) = match self.deal_script {
            DealScript::Terminal => (Vec::new(), false),
            DealScript::Paged if self.deal_requests == 1 => (
                vec![
                    json!({"dealId": 1, "executionTimestamp": from + 100}),
                    json!({"dealId": 2, "executionTimestamp": from + 200}),
                ],
                true,
            ),
            DealScript::Paged => (
                vec![json!({"dealId": 3, "executionTimestamp": from + 50})],
                false,
            ),
            DealScript::EmptyHasMore => (Vec::new(), true),
        };
        Self::response(
            message,
            CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
            json!({
                "ctidTraderAccountId": ACCOUNT_ID,
                "deal": deals,
                "hasMore": has_more,
            }),
        )
    }
}

impl CTraderBrokerTruthSameSessionV2 for ScriptedSameSession {
    fn exchange_same_session(&mut self, message: &CTraderOpenApiJsonMessage) -> Result<String> {
        let response = match message.payload_type {
            CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
                json!({
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "symbol": [{
                        "symbolId": SYMBOL_ID,
                        "symbolName": "EURUSD",
                        "enabled": true,
                        "baseAssetId": 1,
                        "quoteAssetId": 2,
                    }],
                }),
            ),
            CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
                json!({
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "symbol": [{"symbolId": SYMBOL_ID, "digits": 5, "pipPosition": 4}],
                }),
            ),
            CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE => self.tick_response(message),
            CTRADER_OA_ASSET_LIST_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,
                json!({
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "asset": [
                        {"assetId": 1, "name": "EUR", "digits": 2},
                        {"assetId": 2, "name": "USD", "digits": 2},
                    ],
                }),
            ),
            CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
                json!({
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "trader": {"balance": 100_000, "moneyDigits": 2, "depositAssetId": 2},
                }),
            ),
            CTRADER_OA_GET_POSITION_UNREALIZED_PNL_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_GET_POSITION_UNREALIZED_PNL_RESPONSE_PAYLOAD_TYPE,
                json!({
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "moneyDigits": 2,
                    "positionUnrealizedPnL": [],
                }),
            ),
            CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE => Self::response(
                message,
                CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
                json!({"ctidTraderAccountId": ACCOUNT_ID, "position": [], "order": []}),
            ),
            CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE => self.deal_response(message),
            unexpected => panic!("unexpected request payload type {unexpected}"),
        };
        self.requests.push(message.clone());
        Ok(response)
    }
}

fn request() -> BrokerFinancialTruthCaptureRequestV2 {
    let window = EvidenceWindowV1::new(FROM_MS, TO_MS).expect("valid long evidence window");
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        SYMBOL_ID,
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
        2,
        "USD",
    )
    .expect("exact broker truth binding");
    let primary = ExactQuoteInstrumentV2::new(SYMBOL_ID, "EURUSD", 1, "EUR", 2, "USD")
        .expect("primary instrument");
    let settlement = ExactConversionRouteCaptureRequestV2::new(
        "primary_pnl_settlement",
        2,
        "USD",
        2,
        "USD",
        Vec::new(),
    )
    .expect("identity settlement route");
    BrokerFinancialTruthCaptureRequestV2::new(ACCOUNT_ID, binding, primary, vec![settlement])
        .expect("complete capture request")
}

fn reviewed_synchronization(
    request: &BrokerFinancialTruthCaptureRequestV2,
) -> ReviewedCTraderQuoteSynchronizationV2 {
    ReviewedCTraderQuoteSynchronizationV2::new(
        ACCOUNT_ID,
        request.primary_instrument().clone(),
        request.window(),
        synchronization_capture(request),
    )
    .expect("exact keyed reviewed synchronization")
}

fn synchronization_capture(
    request: &BrokerFinancialTruthCaptureRequestV2,
) -> CapturedQuoteSynchronizationV2 {
    let window = request.window();
    let raw = [QuoteSideV1::Bid, QuoteSideV1::Ask]
        .into_iter()
        .enumerate()
        .map(|(sequence, side)| {
            let side_label = match side {
                QuoteSideV1::Bid => "bid",
                QuoteSideV1::Ask => "ask",
            };
            CapturedBrokerEvidenceRowV2::new(
                sequence as u64,
                ACCOUNT_ID,
                Some(SYMBOL_ID),
                Some(side),
                BrokerEvidenceRowKindV2::QuoteSessionObservation,
                Some(window),
                format!("review-observation-{side_label}"),
                CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
                json!({"reviewedRawObservation": side_label}).to_string(),
            )
        })
        .collect::<Vec<_>>();
    let decoded = vec![CapturedBrokerEvidenceRowV2::new(
        0,
        ACCOUNT_ID,
        Some(SYMBOL_ID),
        None,
        BrokerEvidenceRowKindV2::QuoteReplayRule,
        Some(window),
        "review-observation-bid",
        CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
        json!({"reviewedReplayRule": "exact-v2"}).to_string(),
    )];
    let identity = ReviewedQuoteReplayRuleIdentityV2::new(SHA_A, SHA_B, SHA_C)
        .expect("immutable reviewed replay identity");
    CapturedQuoteSynchronizationV2::new(identity, CapturedBrokerEvidencePairV2::new(raw, decoded))
}

fn run_capture(
    tick_script: TickScript,
    deal_script: DealScript,
) -> (
    Result<neoethos_broker_truth::BrokerFinancialTruthBundleReceiptV2, String>,
    ScriptedSameSession,
) {
    let request = request();
    let reviewed = reviewed_synchronization(&request);
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BrokerFinancialTruthBundleStoreV1::new(temp.path().join("store"));
    let mut wire = ScriptedSameSession::new(tick_script, deal_script);
    let result = {
        let mut adapter = CTraderBrokerTruthAdapterV2::new(
            &mut wire,
            &request,
            "adapter-contract",
            100,
            true,
            vec![reviewed],
        )
        .expect("valid exact adapter inputs");
        capture_and_publish_broker_financial_truth_v2(
            &mut adapter,
            &request,
            temp.path().join("capture-work"),
            &store,
            || false,
            || Ok(()),
        )
        .map_err(|error| format!("{:?}: {}", error.code(), error.detail()))
    };
    (result, wire)
}

#[test]
fn adapter_uses_one_session_and_exact_v2_quote_and_deal_boundaries() {
    let (result, wire) = run_capture(TickScript::Paged, DealScript::Paged);
    result.expect("complete same-session broker evidence capture");

    let ticks = wire
        .requests
        .iter()
        .filter(|request| request.payload_type == CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE)
        .collect::<Vec<_>>();
    assert_eq!(ticks.len(), 8, "two sides x two chunks x two pages");
    for request in &ticks {
        let from = request.payload["fromTimestamp"].as_i64().expect("from");
        let to = request.payload["toTimestamp"].as_i64().expect("to");
        assert!(to > from);
        assert!(to - from <= MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2);
        assert!(matches!(request.payload["type"].as_i64(), Some(1 | 2)));
    }
    for side in [1_i64, 2_i64] {
        let side_requests = ticks
            .iter()
            .filter(|request| request.payload["type"].as_i64() == Some(side))
            .collect::<Vec<_>>();
        for pair in side_requests.chunks_exact(2) {
            let first_from = pair[0].payload["fromTimestamp"].as_i64().expect("from");
            let first_to = pair[0].payload["toTimestamp"].as_i64().expect("to");
            let second_to = pair[1].payload["toTimestamp"].as_i64().expect("to");
            assert_eq!(second_to, first_from + (first_to - first_from) / 2);
        }
    }

    let deals = wire
        .requests
        .iter()
        .filter(|request| request.payload_type == CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE)
        .collect::<Vec<_>>();
    assert_eq!(deals.len(), 2);
    assert_eq!(deals[0].payload["maxRows"].as_i64(), Some(100));
    assert_eq!(
        deals[1].payload["toTimestamp"].as_i64(),
        Some(FROM_MS + 100)
    );

    let client_ids = wire
        .requests
        .iter()
        .map(|request| request.client_msg_id.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(client_ids.len(), wire.requests.len());
}

#[test]
fn adapter_requires_the_exact_reviewed_replay_input_before_any_exchange() {
    let request = request();
    let mut wire = ScriptedSameSession::new(TickScript::Terminal, DealScript::Terminal);
    let error = CTraderBrokerTruthAdapterV2::new(
        &mut wire,
        &request,
        "missing-review",
        100,
        true,
        Vec::new(),
    )
    .err()
    .expect("reviewed replay evidence is mandatory and has no default");
    assert!(error.to_string().contains("reviewed"));
    assert!(wire.requests.is_empty());

    let wrong_window = EvidenceWindowV1::new(FROM_MS, TO_MS - 1).expect("different window");
    let mismatched = ReviewedCTraderQuoteSynchronizationV2::new(
        ACCOUNT_ID,
        request.primary_instrument().clone(),
        wrong_window,
        synchronization_capture(&request),
    )
    .expect("structurally valid but differently keyed review input");
    let error = CTraderBrokerTruthAdapterV2::new(
        &mut wire,
        &request,
        "mismatched-review",
        100,
        true,
        vec![mismatched],
    )
    .err()
    .expect("reviewed replay evidence for another window must fail closed");
    assert!(error.to_string().contains("account/instrument/window"));
    assert!(wire.requests.is_empty());
}

#[test]
fn incomplete_quote_and_deal_pages_fail_closed() {
    for (tick_script, deal_script, expected) in [
        (
            TickScript::EmptyHasMore,
            DealScript::Terminal,
            BrokerFinancialTruthCaptureErrorCodeV2::CaptureFailed,
        ),
        (
            TickScript::NonProgress,
            DealScript::Terminal,
            BrokerFinancialTruthCaptureErrorCodeV2::CaptureFailed,
        ),
        (
            TickScript::OutOfWindow,
            DealScript::Terminal,
            BrokerFinancialTruthCaptureErrorCodeV2::CaptureFailed,
        ),
        (
            TickScript::Terminal,
            DealScript::EmptyHasMore,
            BrokerFinancialTruthCaptureErrorCodeV2::CaptureFailed,
        ),
    ] {
        let (result, _) = run_capture(tick_script, deal_script);
        let error = result.expect_err("incomplete/non-progress evidence must not publish");
        assert!(error.starts_with(&format!("{expected:?}:")), "{error}");
    }
}

#[test]
fn adapter_source_has_no_transport_constructor_semantic_permit_or_fallback() {
    let source = include_str!("../src/broker_truth_ctrader.rs");
    let lib_source = include_str!("../src/lib.rs");

    assert!(
        source.contains("impl CTraderBrokerTruthSameSessionV2 for ProductionCTraderOpenApiSession")
    );
    assert!(source.contains("MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2"));
    assert!(source.contains("parse_symbols_list_response"));
    assert!(source.contains("ReviewedCTraderQuoteSynchronizationV2"));
    assert!(lib_source.contains("pub mod broker_truth_ctrader"));
    for forbidden in [
        "ProductionCTraderOpenApiTransport::new",
        "send_sequence",
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
        "current_broker_financial_truth",
        "OnceLock",
        "std::env",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden adapter token {forbidden}"
        );
    }
}
