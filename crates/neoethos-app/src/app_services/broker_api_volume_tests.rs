//! Deterministic transport tests of the actual history/margin API consumers.
//! Hand-built protocol fixtures prove units and identity joins, not device,
//! live broker, historical-cost, or trading acceptance.
use super::*;
use crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::Mutex;

struct WireTransport {
    responses: Mutex<VecDeque<Vec<String>>>,
    sent: Mutex<Vec<Vec<CTraderOpenApiJsonMessage>>>,
}

impl WireTransport {
    fn new(responses: Vec<Vec<String>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            sent: Mutex::new(Vec::new()),
        }
    }
}

impl CTraderOpenApiTransport for WireTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        self.sent.lock().unwrap().push(messages.to_vec());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("unexpected transport call"))
    }
}

fn credentials() -> ResolvedCreds {
    ResolvedCreds {
        client_id: "fixture-client".into(),
        client_secret: "fixture-secret".into(),
        access_token: "fixture-token".into(),
        account_id_str: "42".into(),
        environment: CTraderEnvironment::Demo,
        env_label: "Demo",
    }
}

fn reply(payload_type: u32, payload: Value) -> String {
    json!({"payloadType":payload_type,"payload":payload}).to_string()
}

fn authenticated(payload_type: u32, payload: Value) -> Vec<String> {
    vec![
        reply(2101, json!({})),
        reply(2103, json!({"ctidTraderAccountId":42})),
        reply(payload_type, payload),
    ]
}

fn order(id: i64, symbol: i64, raw: i64, executed: Option<i64>) -> Value {
    json!({"orderId":id,"orderType":2,"orderStatus":2,"executedVolume":executed,
        "tradeData":{"symbolId":symbol,"volume":raw,"tradeSide":1}})
}

fn history(account: i64, orders: Vec<Value>) -> Vec<String> {
    authenticated(
        2176,
        json!({"ctidTraderAccountId":account,"hasMore":true,"order":orders}),
    )
}

fn symbol(id: i64, lot_size: Option<i64>) -> Value {
    json!({"symbolId":id,"digits":5,"pipPosition":4,"lotSize":lot_size})
}

fn metadata(account: i64, symbols: Vec<Value>) -> Vec<String> {
    authenticated(
        2117,
        json!({"ctidTraderAccountId":account,"symbol":symbols}),
    )
}

fn margin(account: i64, volumes: &[i64]) -> Vec<String> {
    authenticated(
        2140,
        json!({"ctidTraderAccountId":account,"moneyDigits":4,
        "margin":volumes.iter().map(|v| json!({"volume":v,"buyMargin":123456,"sellMargin":234567})).collect::<Vec<_>>()}),
    )
}

#[test]
fn history_consumer_joins_symbol_specific_lots_and_preserves_partial_fills() {
    let transport = WireTransport::new(vec![
        history(
            42,
            vec![
                order(1, 7, 1_000_000, Some(600_000)),
                order(2, 8, 2_500, None),
                order(3, 7, 100_000, Some(0)),
            ],
        ),
        metadata(
            42,
            vec![symbol(8, Some(10_000)), symbol(7, Some(10_000_000))],
        ),
    ]);
    let bundle =
        fetch_broker_order_history_with_transport(&transport, &credentials(), 1, 2).unwrap();
    assert!(bundle.has_more);
    assert!(bundle.lot_size_error.is_none());
    assert!(bundle.lot_size_observed_at_unix_ms.is_some());
    assert_eq!(bundle.orders[0].volume_units, 10_000.0);
    assert_eq!(bundle.orders[0].volume_lots, Some(0.1));
    assert_eq!(bundle.orders[0].executed_volume_units, Some(6_000.0));
    assert_eq!(bundle.orders[0].executed_volume_lots, Some(0.06));
    assert_eq!(bundle.orders[1].volume_units, 25.0);
    assert_eq!(bundle.orders[1].volume_lots, Some(0.25));
    assert_eq!(bundle.orders[1].executed_volume_lots, None);
    assert_eq!(bundle.orders[2].executed_volume_lots, Some(0.0));
    let sent = transport.sent.lock().unwrap();
    assert_eq!(
        sent.len(),
        2,
        "one batched metadata read, not one per order"
    );
    assert_eq!(sent[1][2].payload["symbolId"], json!([7, 8]));
    assert_eq!(sent[1][2].payload["ctidTraderAccountId"], 42);
}

#[test]
fn history_refuses_a_different_account_before_metadata_lookup() {
    let transport = WireTransport::new(vec![history(43, vec![order(1, 7, 1_000_000, None)])]);
    let error =
        fetch_broker_order_history_with_transport(&transport, &credentials(), 1, 2).unwrap_err();
    assert!(error.to_string().contains("account 43"));
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[test]
fn history_keeps_units_but_never_lots_when_metadata_is_incomplete_or_mismatched() {
    for (account, rows) in [
        (43, vec![symbol(7, Some(10_000_000))]),
        (42, vec![symbol(9, Some(10_000_000))]),
        (
            42,
            vec![symbol(7, Some(10_000_000)), symbol(7, Some(10_000_000))],
        ),
        (42, vec![]),
        (42, vec![symbol(7, None)]),
        (42, vec![symbol(7, Some(0))]),
    ] {
        let transport = WireTransport::new(vec![
            history(42, vec![order(1, 7, 1_000_000, None)]),
            metadata(account, rows),
        ]);
        let bundle =
            fetch_broker_order_history_with_transport(&transport, &credentials(), 1, 2).unwrap();
        assert_eq!(bundle.orders.len(), 1);
        assert_eq!(bundle.orders[0].volume_units, 10_000.0);
        assert_eq!(bundle.orders[0].volume_lots, None);
        assert_eq!(bundle.orders[0].lot_size_raw_centi_units, None);
        assert!(bundle.lot_size_error.is_some());
        assert!(bundle.lot_size_observed_at_unix_ms.is_none());
    }
}

#[test]
fn empty_history_does_not_fetch_or_invent_lot_sizes() {
    let transport = WireTransport::new(vec![history(42, vec![])]);
    let bundle =
        fetch_broker_order_history_with_transport(&transport, &credentials(), 1, 2).unwrap();
    assert!(bundle.orders.is_empty());
    assert!(
        bundle.has_more,
        "an empty truncated response is not completeness proof"
    );
    assert!(bundle.lot_size_observed_at_unix_ms.is_none());
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[test]
fn margin_consumer_binds_requested_symbol_volume_and_current_contract() {
    let transport = WireTransport::new(vec![
        margin(42, &[1_000_000]),
        metadata(42, vec![symbol(7, Some(10_000_000))]),
    ]);
    let bundle =
        fetch_broker_expected_margin_with_transport(&transport, &credentials(), 7, &[1_000_000])
            .unwrap();
    assert_eq!(bundle.symbol_id, Some(7));
    assert!(bundle.lot_size_error.is_none());
    assert_eq!(bundle.entries[0].volume_raw_centi_units, 1_000_000);
    assert_eq!(bundle.entries[0].volume_units, 10_000.0);
    assert_eq!(bundle.entries[0].volume_lots, Some(0.1));
    assert_eq!(bundle.entries[0].buy_margin, 12.3456);
    assert_eq!(bundle.entries[0].sell_margin, 23.4567);
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent[0][2].payload["symbolId"], 7);
    assert_eq!(sent[0][2].payload["volume"], json!([1_000_000]));
    assert_eq!(sent[1][2].payload["symbolId"], json!([7]));
}

#[test]
fn margin_refuses_foreign_missing_duplicate_and_unrequested_volume_rows() {
    for (account, rows) in [
        (43, vec![1_000_000]),
        (42, vec![]),
        (42, vec![1_000_000, 1_000_000]),
        (42, vec![100_000]),
    ] {
        let transport = WireTransport::new(vec![margin(account, &rows)]);
        assert!(
            fetch_broker_expected_margin_with_transport(
                &transport,
                &credentials(),
                7,
                &[1_000_000]
            )
            .is_err()
        );
        assert_eq!(transport.sent.lock().unwrap().len(), 1);
    }
}

#[test]
fn margin_missing_lot_size_keeps_the_requested_units_and_broker_money() {
    let transport = WireTransport::new(vec![
        margin(42, &[1_000_000]),
        metadata(42, vec![symbol(7, None)]),
    ]);
    let bundle =
        fetch_broker_expected_margin_with_transport(&transport, &credentials(), 7, &[1_000_000])
            .unwrap();
    assert_eq!(bundle.entries[0].volume_units, 10_000.0);
    assert_eq!(bundle.entries[0].volume_lots, None);
    assert_eq!(bundle.entries[0].buy_margin, 12.3456);
    assert!(bundle.lot_size_error.unwrap().contains("omitted lotSize"));
}

#[test]
fn lot_size_lookup_batches_all_symbols_without_truncating_the_request() {
    let ids: BTreeSet<i64> = (1..=53).collect();
    let transport = WireTransport::new(vec![
        metadata(
            42,
            (1..=50).map(|id| symbol(id, Some(10_000_000))).collect(),
        ),
        metadata(42, (51..=53).map(|id| symbol(id, Some(10_000))).collect()),
    ]);
    let sizes = fetch_current_broker_lot_sizes_with_transport(&transport, &credentials(), 42, &ids)
        .unwrap();
    assert_eq!(sizes.len(), 53);
    let sent = transport.sent.lock().unwrap();
    assert_eq!(sent[0][2].payload["symbolId"].as_array().unwrap().len(), 50);
    assert_eq!(sent[1][2].payload["symbolId"], json!([51, 52, 53]));
}

#[test]
fn invalid_margin_inputs_fail_before_credentials_or_network() {
    for (symbol_id, volumes) in [
        (0, vec![1]),
        (7, vec![]),
        (7, vec![0]),
        (7, vec![-1]),
        (7, vec![1, 1]),
        (7, vec![MAX_EXACT_BROKER_VOLUME + 1]),
    ] {
        let transport = WireTransport::new(vec![]);
        assert!(
            fetch_broker_expected_margin_with_transport(
                &transport,
                &credentials(),
                symbol_id,
                &volumes
            )
            .is_err()
        );
        assert!(transport.sent.lock().unwrap().is_empty());
    }
}
