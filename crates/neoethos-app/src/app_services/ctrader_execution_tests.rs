// TODO(real-data): every hand-written JSON string fed to StubTransport
// in this file (payloadType 2101/2103/2126 etc.) is a model of what we
// think the cTrader server returns. Replace each with a captured
// response from the demo Open API endpoint for the same symbol /
// execution-type / payload-type so behaviour is asserted against real
// broker bytes and not a hand-rolled fixture.
use super::*;

use crate::app_services::ctrader_messages::{
    CTraderClosePositionRequest, CTraderOrderType, CTraderTimeInForce,
};

#[derive(Clone)]
struct StubTransport {
    responses: Arc<Mutex<Vec<anyhow::Result<String>>>>,
    sent_batches: Arc<Mutex<Vec<Vec<CTraderOpenApiJsonMessage>>>>,
}

impl StubTransport {
    fn with_responses(responses: Vec<anyhow::Result<String>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            sent_batches: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn sent_batches(&self) -> Vec<Vec<CTraderOpenApiJsonMessage>> {
        self.sent_batches
            .lock()
            .expect("sent batches lock poisoned")
            .clone()
    }
}

impl CTraderOpenApiTransport for StubTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        self.sent_batches
            .lock()
            .expect("sent batches lock poisoned")
            .push(messages.to_vec());
        self.responses
            .lock()
            .expect("responses lock poisoned")
            .drain(..)
            .collect()
    }
}

fn sample_runtime_request(request: CTraderExecutionRequest) -> CTraderExecutionRuntimeRequest {
    CTraderExecutionRuntimeRequest {
        client_id: "client".to_string(),
        client_secret: "secret".to_string(),
        access_token: "token".to_string(),
        environment: CTraderEnvironment::Demo,
        account_id: request.account_id().to_string(),
        request,
    }
}

#[test]
fn execution_event_maps_filled_outcome_with_realized_pnl() {
    let response = r#"{
        "payloadType": 2126,
        "payload": {
            "ctidTraderAccountId": 712345,
            "executionType": 3,
            "order": {
                "orderId": 8001,
                "tradeData": {
                    "symbolId": 14,
                    "volume": 10000000,
                    "tradeSide": 1,
                    "openTimestamp": 1710000000000
                },
                "orderType": 1,
                "executionPrice": 1.09876
            },
            "position": {
                "positionId": 9001,
                "tradeData": {
                    "symbolId": 14,
                    "volume": 10000000,
                    "tradeSide": 1,
                    "openTimestamp": 1710000000000
                },
                "price": 1.09876
            },
            "deal": {
                "dealId": 3001,
                "orderId": 8001,
                "positionId": 9001,
                "volume": 10000000,
                "filledVolume": 10000000,
                "symbolId": 14,
                "executionTimestamp": 1710000201000,
                "executionPrice": 1.099,
                "tradeSide": 1,
                "commission": -40,
                "moneyDigits": 2,
                "closePositionDetail": {
                    "grossProfit": 1250,
                    "swap": -15,
                    "commission": -40,
                    "pnlConversionFee": -10,
                    "moneyDigits": 2
                }
            }
        }
    }"#;

    let outcome = parse_execution_outcome(response).expect("filled execution should parse");

    assert_eq!(outcome.status, CTraderExecutionStatus::Filled);
    assert_eq!(outcome.account_id, 712345);
    assert_eq!(outcome.symbol_id, Some(14));
    assert_eq!(outcome.order_id, Some(8001));
    assert_eq!(outcome.position_id, Some(9001));
    assert_eq!(outcome.deal_id, Some(3001));
    assert_eq!(outcome.filled_volume_raw_centi_units, Some(10_000_000));
    assert!(outcome.volume_scale_evidence.is_none());
    assert_eq!(outcome.trade_side.as_deref(), Some("BUY"));
    assert_eq!(outcome.order_type.as_deref(), Some("MARKET"));
    assert_eq!(outcome.lot_size, Some(100000.0));
    assert_eq!(outcome.execution_price, Some(1.099));
    assert_eq!(outcome.gross_profit, Some(12.5));
    assert_eq!(outcome.fee, Some(-0.4));
    assert_eq!(outcome.swap, Some(-0.15));
    assert_eq!(outcome.net_profit, Some(11.85));
}

#[test]
fn execution_event_scales_close_detail_money_digits_four_fields() {
    let response = r#"{
        "payloadType": 2126,
        "payload": {
            "ctidTraderAccountId": 712345,
            "executionType": 3,
            "deal": {
                "dealId": 3001,
                "orderId": 8001,
                "positionId": 9001,
                "filledVolume": 10000000,
                "symbolId": 14,
                "executionTimestamp": 1710000201000,
                "executionPrice": 1.099,
                "tradeSide": 1,
                "closePositionDetail": {
                    "grossProfit": 1250,
                    "swap": -15,
                    "commission": -40,
                    "pnlConversionFee": -10,
                    "moneyDigits": 4
                }
            }
        }
    }"#;

    let outcome = parse_execution_outcome(response).expect("filled execution should parse");

    assert_eq!(outcome.gross_profit, Some(0.125));
    assert_eq!(outcome.fee, Some(-0.004));
    assert_eq!(outcome.swap, Some(-0.0015));
    assert_eq!(outcome.net_profit, Some(0.1185));
}

#[test]
fn execution_event_rejects_close_financials_without_broker_scale() {
    let response = r#"{
        "payloadType": 2126,
        "payload": {
            "ctidTraderAccountId": 712345,
            "executionType": 3,
            "deal": {
                "dealId": 3001,
                "orderId": 8001,
                "positionId": 9001,
                "filledVolume": 10000000,
                "symbolId": 14,
                "executionTimestamp": 1710000201000,
                "tradeSide": 1,
                "closePositionDetail": {
                    "grossProfit": 1250,
                    "swap": -15,
                    "commission": -40
                }
            }
        }
    }"#;

    let error = parse_execution_outcome(response)
        .expect_err("execution financials without moneyDigits must fail closed");
    assert!(
        error
            .to_string()
            .contains("execution.close_position_detail.money_digits")
    );
}

#[test]
fn order_error_event_maps_failed_outcome() {
    let response = r#"{
        "payloadType": 2132,
        "payload": {
            "errorCode": "ORDER_NOT_FOUND",
            "orderId": 8001,
            "positionId": 9001,
            "ctidTraderAccountId": 712345,
            "description": "Order does not exist"
        }
    }"#;

    let outcome = parse_execution_outcome(response).expect("order error should parse");

    assert_eq!(outcome.status, CTraderExecutionStatus::Failed);
    assert_eq!(outcome.order_id, Some(8001));
    assert_eq!(outcome.position_id, Some(9001));
    assert_eq!(outcome.error_code.as_deref(), Some("ORDER_NOT_FOUND"));
    assert_eq!(outcome.description.as_deref(), Some("Order does not exist"));
}

#[test]
fn production_backend_authenticates_then_executes_market_order() {
    let transport = StubTransport::with_responses(vec![
        Ok(r#"{"payloadType":2101,"payload":{}}"#.to_string()),
        Ok(r#"{"payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
        Ok(r#"{"payloadType":2126,"payload":{"ctidTraderAccountId":712345,"executionType":2,"order":{"orderId":8001,"tradeData":{"symbolId":14,"volume":10000000,"tradeSide":1,"openTimestamp":1710000000000},"orderType":1}}}"#.to_string()),
    ]);
    let request = sample_runtime_request(CTraderExecutionRequest::NewOrder(Box::new(
        CTraderNewOrderRequest {
            account_id: 712345,
            symbol_id: 14,
            order_type: CTraderOrderType::Market,
            trade_side: crate::app_services::ctrader_messages::CTraderTradeSide::Buy,
            volume: 10000000,
            limit_price: None,
            stop_price: None,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: None,
            take_profit: None,
            comment: Some("manual market".to_string()),
            base_slippage_price: None,
            slippage_in_points: Some(10),
            label: Some("operator".to_string()),
            position_id: None,
            client_order_id: Some("ticket-1".to_string()),
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_trigger_method: None,
        },
    )));

    let outcome = ProductionCTraderExecutionBackend::execute_with_transport(&transport, &request)
        .expect("execution should succeed");

    let sent_batches = transport.sent_batches();
    assert_eq!(sent_batches.len(), 1);
    assert_eq!(sent_batches[0].len(), 3);
    assert_eq!(
        sent_batches[0][2].payload_type,
        crate::app_services::ctrader_messages::CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(outcome.status, CTraderExecutionStatus::Accepted);
    assert_eq!(outcome.order_id, Some(8001));
}

#[test]
fn production_backend_rejects_cancelled_close_position_outcome() {
    let transport = StubTransport::with_responses(vec![
        Ok(r#"{"payloadType":2101,"payload":{}}"#.to_string()),
        Ok(r#"{"payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
        Ok(r#"{"payloadType":2126,"payload":{"ctidTraderAccountId":712345,"executionType":5,"position":{"positionId":9001,"tradeData":{"symbolId":14,"volume":5000000,"tradeSide":1,"openTimestamp":1710000000000},"price":1.1025}}}"#.to_string()),
    ]);
    let request = sample_runtime_request(CTraderExecutionRequest::ClosePosition(
        CTraderClosePositionRequest {
            account_id: 712345,
            position_id: 9001,
            volume: 5000000,
        },
    ));

    let error = ProductionCTraderExecutionBackend::execute_with_transport(&transport, &request)
        .expect_err("a cancelled close order must not count as an executed close");
    assert!(error.to_string().contains("Cancelled"));
    assert!(
        error
            .to_string()
            .contains("does not confirm requested operation close_position")
    );
}

#[test]
fn close_position_validator_distinguishes_execution_from_acceptance_and_cancellation() {
    let request = sample_runtime_request(CTraderExecutionRequest::ClosePosition(
        CTraderClosePositionRequest {
            account_id: 712345,
            position_id: 9001,
            volume: 5_000_000,
        },
    ));
    let mut outcome = parse_execution_outcome(
        r#"{"payloadType":2126,"payload":{"ctidTraderAccountId":712345,"executionType":3,"position":{"positionId":9001,"tradeData":{"symbolId":14,"volume":5000000,"tradeSide":1,"openTimestamp":1710000000000},"price":1.1025},"deal":{"dealId":3001,"orderId":8001,"positionId":9001,"filledVolume":5000000,"symbolId":14,"executionTimestamp":1710000201000,"tradeSide":1}}}"#,
    )
    .expect("filled close execution should parse");
    validate_execution_outcome(&request, &outcome)
        .expect("a matching filled execution is a valid close outcome, not proof of flatness");

    for status in [
        CTraderExecutionStatus::Accepted,
        CTraderExecutionStatus::Replaced,
        CTraderExecutionStatus::Cancelled,
    ] {
        outcome.status = status;
        let error = validate_execution_outcome(&request, &outcome).expect_err(
            "matching account, position and deal IDs cannot authorize the wrong status",
        );
        assert!(
            error
                .to_string()
                .contains("does not confirm requested operation close_position")
        );
    }
}

#[path = "ctrader_execution_terminal_tests.rs"]
mod terminal_collection;

#[test]
fn identical_execution_requests_have_identical_fingerprints_and_variants_do_not() {
    let mut base = CTraderNewOrderRequest {
        account_id: 712345,
        symbol_id: 14,
        order_type: CTraderOrderType::Market,
        trade_side: crate::app_services::ctrader_messages::CTraderTradeSide::Buy,
        volume: 100000,
        limit_price: None,
        stop_price: None,
        time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
        expiration_timestamp_ms: None,
        stop_loss: None,
        take_profit: None,
        comment: Some("alpha".to_string()),
        base_slippage_price: None,
        slippage_in_points: Some(10),
        label: Some("entry".to_string()),
        position_id: None,
        client_order_id: Some("id-1".to_string()),
        relative_stop_loss: None,
        relative_take_profit: None,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    };
    let a = CTraderExecutionRequest::NewOrder(Box::new(base.clone()));
    let b = CTraderExecutionRequest::NewOrder(Box::new(base.clone()));
    base.client_order_id = Some("id-2".to_string());
    let c = CTraderExecutionRequest::NewOrder(Box::new(base));

    assert_eq!(a.idempotency_fingerprint(), b.idempotency_fingerprint());
    assert_ne!(a.idempotency_fingerprint(), c.idempotency_fingerprint());
    let runtime_a = sample_runtime_request(a);
    let runtime_b = sample_runtime_request(b);
    let runtime_c = sample_runtime_request(c);
    let key_a = ProductionCTraderExecutionBackend::execution_cache_fingerprint(&runtime_a);
    let key_b = ProductionCTraderExecutionBackend::execution_cache_fingerprint(&runtime_b);
    let key_c = ProductionCTraderExecutionBackend::execution_cache_fingerprint(&runtime_c);
    assert_eq!(
        key_a, key_b,
        "the same logical intent keeps its cache identity"
    );
    assert_ne!(
        key_a, key_c,
        "independent engine intents must not share a fill"
    );
    let outcome = parse_execution_outcome(&single_opening_execution_wire().to_string())
        .expect("broker fill fixture");
    let mut session = CTraderExecutionSession::default();
    ProductionCTraderExecutionBackend::store_cached_outcome(&mut session, key_a, outcome.clone());
    assert_eq!(
        ProductionCTraderExecutionBackend::maybe_cached_outcome(&session, &key_b),
        Some(outcome)
    );
    assert!(ProductionCTraderExecutionBackend::maybe_cached_outcome(&session, &key_c).is_none());
}

#[test]
fn validate_execution_outcome_rejects_symbol_mismatch_for_new_order() {
    let request = sample_runtime_request(CTraderExecutionRequest::NewOrder(Box::new(
        CTraderNewOrderRequest {
            account_id: 712345,
            symbol_id: 14,
            order_type: CTraderOrderType::Market,
            trade_side: crate::app_services::ctrader_messages::CTraderTradeSide::Buy,
            volume: 100000,
            limit_price: None,
            stop_price: None,
            time_in_force: None,
            expiration_timestamp_ms: None,
            stop_loss: None,
            take_profit: None,
            comment: None,
            base_slippage_price: None,
            slippage_in_points: None,
            label: None,
            position_id: None,
            client_order_id: None,
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_trigger_method: None,
        },
    )));
    let outcome = CTraderExecutionOutcome {
        status: CTraderExecutionStatus::Accepted,
        account_id: 712345,
        symbol_id: Some(99),
        order_id: Some(1),
        position_id: None,
        deal_id: None,
        trade_side: Some("BUY".to_string()),
        order_type: Some("MARKET".to_string()),
        lot_size: Some(1000.0),
        requested_lot_size: Some(1000.0),
        filled_lot_size: None,
        filled_volume_raw_centi_units: None,
        volume_scale_evidence: None,
        deal_closes_position: None,
        opening_fill_evidence: None,
        execution_price: None,
        gross_profit: None,
        fee: None,
        swap: None,
        net_profit: None,
        timestamp_ms: None,
        error_code: None,
        description: None,
    };

    assert!(validate_execution_outcome(&request, &outcome).is_err());
}

#[test]
fn position_protection_amend_requires_order_replaced_confirmation() {
    let request = sample_runtime_request(CTraderExecutionRequest::AmendPositionSltp(
        CTraderAmendPositionSltpRequest {
            account_id: 712345,
            position_id: 9001,
            stop_loss: Some(1.10123),
            take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_loss_trigger_method: None,
        },
    ));
    let mut outcome = CTraderExecutionOutcome {
        status: CTraderExecutionStatus::Accepted,
        account_id: 712345,
        symbol_id: Some(14),
        order_id: None,
        position_id: Some(9001),
        deal_id: None,
        trade_side: Some("BUY".to_string()),
        order_type: None,
        lot_size: Some(1.0),
        requested_lot_size: Some(1.0),
        filled_lot_size: None,
        filled_volume_raw_centi_units: None,
        volume_scale_evidence: None,
        deal_closes_position: None,
        opening_fill_evidence: None,
        execution_price: Some(1.10234),
        gross_profit: None,
        fee: None,
        swap: None,
        net_profit: None,
        timestamp_ms: Some(1_710_000_000_000),
        error_code: None,
        description: None,
    };

    let error = validate_execution_outcome(&request, &outcome)
        .expect_err("ORDER_ACCEPTED must not confirm an SL/TP replacement");
    assert!(
        error
            .to_string()
            .contains("does not confirm requested operation")
    );

    outcome.status = CTraderExecutionStatus::Replaced;
    validate_execution_outcome(&request, &outcome)
        .expect("ORDER_REPLACED confirms the broker-side protection amend");
}

fn single_opening_execution_wire() -> serde_json::Value {
    serde_json::json!({
        "payloadType": 2126,
        "payload": {
            "ctidTraderAccountId": 712345,
            "executionType": 3,
            "order": {
                "orderId": 8001, "positionId": 9001, "orderType": 1,
                "orderStatus": 2, "executedVolume": 10000000, "closingOrder": false,
                "tradeData": {
                    "symbolId": 14, "volume": 10000000, "tradeSide": 1,
                    "openTimestamp": 1710000000000_i64
                }
            },
            "position": {
                "positionId": 9001, "positionStatus": 1, "price": 1.09876,
                "tradeData": {
                    "symbolId": 14, "volume": 10000000, "tradeSide": 1,
                    "openTimestamp": 1710000201000_i64
                }
            },
            "deal": {
                "dealId": 3001, "orderId": 8001, "positionId": 9001,
                "volume": 10000000, "filledVolume": 10000000, "symbolId": 14,
                "createTimestamp": 1710000200990_i64,
                "executionTimestamp": 1710000201001_i64, "executionPrice": 1.09876,
                "tradeSide": 1, "dealStatus": 2
            }
        }
    })
}

#[test]
fn opening_execution_preserves_exact_mutually_bound_broker_facts() {
    let outcome = parse_execution_outcome(&single_opening_execution_wire().to_string())
        .expect("complete consistent broker event");
    assert_eq!(outcome.deal_closes_position, Some(false));
    let opening = outcome
        .opening_fill_evidence
        .as_ref()
        .expect("single opening fill");
    assert_eq!(opening.account_id(), 712345);
    assert_eq!(opening.order_id(), 8001);
    assert_eq!(opening.position_id(), 9001);
    assert_eq!(opening.deal_id(), 3001);
    assert_eq!(opening.symbol_id(), 14);
    assert_eq!(opening.trade_side(), "BUY");
    assert_eq!(opening.filled_volume_raw_centi_units(), 10000000);
    assert_eq!(opening.position_open_timestamp_ms(), 1710000201000);
    assert_eq!(opening.execution_timestamp_ms(), 1710000201001);
    assert_eq!(opening.entry_price().to_bits(), 1.09876_f64.to_bits());
    assert!(
        outcome.volume_scale_evidence.is_none(),
        "wire parser invents no lot size"
    );
}

#[test]
fn execution_nested_identity_mismatches_never_project_a_mixed_outcome() {
    for (pointer, value) in [
        ("/payload/ctidTraderAccountId", 0),
        ("/payload/order/orderId", 8002),
        ("/payload/order/positionId", 9002),
        ("/payload/position/positionId", 9002),
        ("/payload/deal/orderId", 8002),
        ("/payload/deal/positionId", 9002),
        ("/payload/deal/dealId", 0),
        ("/payload/order/tradeData/symbolId", 15),
        ("/payload/position/tradeData/symbolId", 15),
        ("/payload/deal/symbolId", 15),
        ("/payload/order/tradeData/tradeSide", 2),
        ("/payload/position/tradeData/tradeSide", 2),
        ("/payload/deal/tradeSide", 2),
    ] {
        let mut wire = single_opening_execution_wire();
        *wire.pointer_mut(pointer).expect("fixture pointer") = serde_json::json!(value);
        assert!(
            parse_execution_outcome(&wire.to_string()).is_err(),
            "{pointer} must not be hidden by the order/position/deal fallback order"
        );
    }
}

#[test]
fn omitted_closing_order_cannot_mint_opposite_side_opening_evidence() {
    let mut wire = single_opening_execution_wire();
    wire["payload"]["order"]
        .as_object_mut()
        .expect("order")
        .remove("closingOrder");
    let same_side = parse_execution_outcome(&wire.to_string()).expect("optional closingOrder");
    assert!(same_side.opening_fill_evidence.is_some());
    wire["payload"]["position"]["tradeData"]["tradeSide"] = serde_json::json!(2);
    let opposite_side = parse_execution_outcome(&wire.to_string())
        .expect("observable event without an invented opening lifecycle");
    assert!(opposite_side.opening_fill_evidence.is_none());
}

#[test]
fn accepted_partial_or_incomplete_execution_never_invents_an_initial_lifecycle() {
    for (pointer, value) in [
        ("/payload/executionType", serde_json::json!(2)),
        ("/payload/executionType", serde_json::json!(11)),
        ("/payload/order/orderStatus", serde_json::Value::Null),
        ("/payload/order/executedVolume", serde_json::Value::Null),
        ("/payload/order/executedVolume", serde_json::json!(20000000)),
        ("/payload/order/closingOrder", serde_json::json!(true)),
        ("/payload/position/positionStatus", serde_json::Value::Null),
        ("/payload/position/positionStatus", serde_json::json!(3)),
        (
            "/payload/position/tradeData/volume",
            serde_json::json!(20000000),
        ),
        (
            "/payload/position/tradeData/openTimestamp",
            serde_json::Value::Null,
        ),
        (
            "/payload/position/tradeData/openTimestamp",
            serde_json::json!(1710000201002_i64),
        ),
        ("/payload/position/price", serde_json::Value::Null),
        ("/payload/position/price", serde_json::json!(1.09877)),
        ("/payload/deal/executionPrice", serde_json::Value::Null),
        ("/payload/deal/dealStatus", serde_json::Value::Null),
        ("/payload/deal/dealStatus", serde_json::json!(3)),
        ("/payload/deal/filledVolume", serde_json::json!(4000000)),
    ] {
        let mut wire = single_opening_execution_wire();
        *wire.pointer_mut(pointer).expect("fixture pointer") = value;
        let outcome = parse_execution_outcome(&wire.to_string())
            .expect("recognized event remains observable without invented proof");
        assert!(outcome.opening_fill_evidence.is_none(), "{pointer}");
    }
}

#[test]
fn netted_closing_deal_is_explicit_and_never_becomes_an_opening_lifecycle() {
    let mut wire = single_opening_execution_wire();
    wire["payload"]["order"]["tradeData"]["tradeSide"] = serde_json::json!(2);
    wire["payload"]["deal"]["tradeSide"] = serde_json::json!(2);
    wire["payload"]["order"]["closingOrder"] = serde_json::json!(true);
    wire["payload"]["deal"]["closePositionDetail"] = serde_json::json!({
        "entryPrice": 1.09876, "grossProfit": -1000, "swap": -20,
        "commission": -40, "balance": 9999940, "moneyDigits": 2
    });
    let outcome = parse_execution_outcome(&wire.to_string())
        .expect("a closing deal may legitimately oppose the position side");
    assert_eq!(outcome.deal_closes_position, Some(true));
    assert!(outcome.opening_fill_evidence.is_none());
    assert_eq!(outcome.gross_profit, Some(-10.0));
    assert_eq!(outcome.fee, Some(-0.4));
    assert_eq!(outcome.swap, Some(-0.2));
    assert!((outcome.net_profit.expect("closing money") - (-1060.0 / 100.0)).abs() < 1e-12);
}

#[test]
fn accepted_order_without_deal_does_not_claim_no_execution_or_opening_proof() {
    let mut wire = single_opening_execution_wire();
    wire["payload"]["executionType"] = serde_json::json!(2);
    wire["payload"]["deal"] = serde_json::Value::Null;
    wire["payload"]["position"] = serde_json::Value::Null;
    let outcome = parse_execution_outcome(&wire.to_string()).expect("accepted order");
    assert_eq!(outcome.deal_closes_position, None);
    assert!(outcome.opening_fill_evidence.is_none());
    assert_eq!(outcome.status, CTraderExecutionStatus::Accepted);
}

#[test]
fn execution_cache_cannot_relabel_a_demo_fill_as_live_or_another_account() {
    let demo = sample_runtime_request(CTraderExecutionRequest::ClosePosition(
        CTraderClosePositionRequest {
            account_id: 712345,
            position_id: 9001,
            volume: 10000000,
        },
    ));
    let mut live = demo.clone();
    live.environment = CTraderEnvironment::Live;
    let mut other_account = demo.clone();
    other_account.account_id = "712346".to_owned();
    let demo_key = ProductionCTraderExecutionBackend::execution_cache_fingerprint(&demo);
    let live_key = ProductionCTraderExecutionBackend::execution_cache_fingerprint(&live);
    let account_key =
        ProductionCTraderExecutionBackend::execution_cache_fingerprint(&other_account);
    assert_ne!(demo_key, live_key);
    assert_ne!(demo_key, account_key);
    let outcome = parse_execution_outcome(&single_opening_execution_wire().to_string())
        .expect("fixture broker outcome");
    let mut session = CTraderExecutionSession::default();
    ProductionCTraderExecutionBackend::store_cached_outcome(
        &mut session,
        demo_key.clone(),
        outcome.clone(),
    );
    assert_eq!(
        ProductionCTraderExecutionBackend::maybe_cached_outcome(&session, &demo_key),
        Some(outcome)
    );
    assert!(ProductionCTraderExecutionBackend::maybe_cached_outcome(&session, &live_key).is_none());
    assert!(
        ProductionCTraderExecutionBackend::maybe_cached_outcome(&session, &account_key).is_none()
    );
}

#[test]
fn authenticated_execution_submission_never_retries_send_or_response_failures() {
    let request = sample_runtime_request(CTraderExecutionRequest::NewOrder(Box::new(
        CTraderNewOrderRequest {
            account_id: 712345,
            symbol_id: 14,
            order_type: CTraderOrderType::Market,
            trade_side: crate::app_services::ctrader_messages::CTraderTradeSide::Buy,
            volume: 10000000,
            limit_price: None,
            stop_price: None,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: None,
            take_profit: None,
            comment: Some("opening-evidence-fixture".to_string()),
            base_slippage_price: None,
            slippage_in_points: Some(10),
            label: Some("engine-fixture".to_string()),
            position_id: None,
            client_order_id: Some("logical-entry-1".to_string()),
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_trigger_method: None,
        },
    )));
    let mut foreign = single_opening_execution_wire();
    foreign["payload"]["ctidTraderAccountId"] = serde_json::json!(712346);
    for (response, expected) in [
        (
            Err(anyhow!("socket timed out after write")),
            "socket timed out after write",
        ),
        (
            Ok("not-json".to_owned()),
            "failed to inspect cTrader execution response",
        ),
        (Ok(foreign.to_string()), "account"),
        (
            Ok(serde_json::json!({
                "payloadType": 2142,
                "payload": {
                    "ctidTraderAccountId": 712345, "errorCode": "NOT_ENOUGH_MONEY",
                    "description": "fixture rejection after submission"
                }
            })
            .to_string()),
            "NOT_ENOUGH_MONEY",
        ),
    ] {
        let submissions = std::cell::Cell::new(0);
        let error = ProductionCTraderExecutionBackend::execute_authenticated_once(&request, || {
            submissions.set(submissions.get() + 1);
            response
        })
        .expect_err("unusable post-submit outcome cannot trigger a resend");
        assert_eq!(submissions.get(), 1);
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    }

    let submissions = std::cell::Cell::new(0);
    let outcome = ProductionCTraderExecutionBackend::execute_authenticated_once(&request, || {
        submissions.set(submissions.get() + 1);
        Ok(single_opening_execution_wire().to_string())
    })
    .expect("one complete, mutually bound broker response");
    assert_eq!(submissions.get(), 1);
    assert!(outcome.opening_fill_evidence.is_some());
}

#[test]
fn engine_opening_converter_uses_real_fill_price_time_and_exact_broker_volume() {
    let mut outcome =
        parse_execution_outcome(&single_opening_execution_wire().to_string()).unwrap();
    outcome.volume_scale_evidence = Some(
        BrokerSymbolVolumeScaleEvidenceV1::new("demo", 712345, 14, "EURUSD", 10000000).unwrap(),
    );
    let (opening, scale, lots) = crate::app_services::live_trading::verified_opening_for_engine(
        &outcome,
        "demo",
        712345,
        14,
        "EURUSD",
        crate::app_services::broker_api::OrderSide::Buy,
    )
    .unwrap();
    assert_eq!(opening.position_id(), 9001);
    assert_eq!(opening.execution_timestamp_ms(), 1710000201001);
    assert_ne!(opening.execution_timestamp_ms(), 1710000000000);
    assert_eq!(opening.entry_price().to_bits(), 1.09876_f64.to_bits());
    assert_eq!(scale.lot_size_raw_centi_units(), 10000000);
    assert_eq!(lots, 1.0);
}

#[test]
fn engine_opening_converter_refuses_unproven_or_relabelled_outcomes() {
    let mut base = parse_execution_outcome(&single_opening_execution_wire().to_string()).unwrap();
    base.volume_scale_evidence = Some(
        BrokerSymbolVolumeScaleEvidenceV1::new("demo", 712345, 14, "EURUSD", 10000000).unwrap(),
    );
    for case in [
        "accepted",
        "partial",
        "reduction",
        "proof",
        "scale",
        "account",
        "symbol",
        "position",
        "order",
        "deal",
        "side",
        "volume",
        "price",
        "timestamp",
    ] {
        let mut outcome = base.clone();
        match case {
            "accepted" => outcome.status = CTraderExecutionStatus::Accepted,
            "partial" => outcome.status = CTraderExecutionStatus::PartialFill,
            "reduction" => outcome.deal_closes_position = Some(true),
            "proof" => outcome.opening_fill_evidence = None,
            "scale" => outcome.volume_scale_evidence = None,
            "account" => outcome.account_id = 712346,
            "symbol" => outcome.symbol_id = Some(15),
            "position" => outcome.position_id = Some(9002),
            "order" => outcome.order_id = Some(8002),
            "deal" => outcome.deal_id = Some(3002),
            "side" => outcome.trade_side = Some("SELL".to_owned()),
            "volume" => outcome.filled_volume_raw_centi_units = Some(4000000),
            "price" => outcome.execution_price = None,
            _ => outcome.timestamp_ms = Some(1710000000000),
        }
        assert!(
            crate::app_services::live_trading::verified_opening_for_engine(
                &outcome,
                "demo",
                712345,
                14,
                "EURUSD",
                crate::app_services::broker_api::OrderSide::Buy,
            )
            .is_err(),
            "{case}"
        );
    }
    for (environment, account, symbol_id, symbol, side) in [
        (
            "live",
            712345,
            14,
            "EURUSD",
            crate::app_services::broker_api::OrderSide::Buy,
        ),
        (
            "demo",
            712346,
            14,
            "EURUSD",
            crate::app_services::broker_api::OrderSide::Buy,
        ),
        (
            "demo",
            712345,
            15,
            "EURUSD",
            crate::app_services::broker_api::OrderSide::Buy,
        ),
        (
            "demo",
            712345,
            14,
            "OTHER",
            crate::app_services::broker_api::OrderSide::Buy,
        ),
        (
            "demo",
            712345,
            14,
            "EURUSD",
            crate::app_services::broker_api::OrderSide::Sell,
        ),
    ] {
        assert!(
            crate::app_services::live_trading::verified_opening_for_engine(
                &base,
                environment,
                account,
                symbol_id,
                symbol,
                side,
            )
            .is_err()
        );
    }
}
