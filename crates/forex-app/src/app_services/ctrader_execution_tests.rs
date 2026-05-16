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

    let outcome =
        ProductionCTraderExecutionBackend::execute_with_transport(&transport, &request)
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
fn production_backend_maps_cancelled_close_position_outcome() {
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

    let outcome =
        ProductionCTraderExecutionBackend::execute_with_transport(&transport, &request)
            .expect("close position should succeed");

    assert_eq!(outcome.status, CTraderExecutionStatus::Cancelled);
    assert_eq!(outcome.position_id, Some(9001));
    assert_eq!(outcome.lot_size, Some(50000.0));
}

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
