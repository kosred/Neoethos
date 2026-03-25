use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    build_account_auth_request, build_application_auth_request, build_cancel_order_request,
    build_close_position_request, build_new_order_request,
    CTraderCancelOrderRequest, CTraderNewOrderRequest, CTraderOpenApiJsonMessage,
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE, CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE,
};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq)]
pub enum CTraderExecutionRequest {
    NewOrder(Box<CTraderNewOrderRequest>),
    CancelOrder(CTraderCancelOrderRequest),
    ClosePosition(crate::app_services::ctrader_messages::CTraderClosePositionRequest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderExecutionStatus {
    Accepted,
    Filled,
    Replaced,
    Cancelled,
    PartialFill,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderExecutionRuntimeRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub request: CTraderExecutionRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderExecutionOutcome {
    pub status: CTraderExecutionStatus,
    pub account_id: i64,
    pub symbol_id: Option<i64>,
    pub order_id: Option<i64>,
    pub position_id: Option<i64>,
    pub deal_id: Option<i64>,
    pub trade_side: Option<String>,
    pub order_type: Option<String>,
    pub lot_size: Option<f64>,
    pub execution_price: Option<f64>,
    pub gross_profit: Option<f64>,
    pub fee: Option<f64>,
    pub swap: Option<f64>,
    pub net_profit: Option<f64>,
    pub timestamp_ms: Option<i64>,
    pub error_code: Option<String>,
    pub description: Option<String>,
}

pub trait CTraderExecutionBackend: Send + Sync {
    fn execute(
        &self,
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderExecutionBackend;

#[derive(Debug, Deserialize)]
struct ExecutionEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: ExecutionPayload,
}

#[derive(Debug, Deserialize)]
struct ExecutionPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(rename = "executionType")]
    execution_type: i32,
    order: Option<ExecutionOrderPayload>,
    position: Option<ExecutionPositionPayload>,
    deal: Option<ExecutionDealPayload>,
    #[serde(rename = "errorCode")]
    error_code: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExecutionOrderPayload {
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: ExecutionTradeDataPayload,
    #[serde(rename = "orderType")]
    order_type: i32,
    #[serde(rename = "executionPrice")]
    execution_price: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExecutionPositionPayload {
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: ExecutionTradeDataPayload,
    price: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExecutionTradeDataPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    volume: i64,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    #[serde(rename = "openTimestamp")]
    open_timestamp: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ExecutionDealPayload {
    #[serde(rename = "dealId")]
    deal_id: i64,
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "filledVolume")]
    filled_volume: i64,
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    #[serde(rename = "executionTimestamp")]
    execution_timestamp: i64,
    #[serde(rename = "executionPrice")]
    execution_price: Option<f64>,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    commission: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(rename = "closePositionDetail")]
    close_position_detail: Option<ExecutionClosePositionDetailPayload>,
}

#[derive(Debug, Deserialize)]
struct ExecutionClosePositionDetailPayload {
    #[serde(rename = "grossProfit")]
    gross_profit: i64,
    swap: i64,
    commission: i64,
    #[serde(rename = "pnlConversionFee")]
    pnl_conversion_fee: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OrderErrorEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: OrderErrorPayload,
}

#[derive(Debug, Deserialize)]
struct OrderErrorPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(rename = "errorCode")]
    error_code: String,
    description: Option<String>,
    #[serde(rename = "orderId")]
    order_id: Option<i64>,
    #[serde(rename = "positionId")]
    position_id: Option<i64>,
}

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderExecutionBackend {
    outcome: Arc<Mutex<Option<Result<CTraderExecutionOutcome, String>>>>,
}

impl CTraderExecutionRequest {
    #[cfg(test)]
    fn account_id(&self) -> i64 {
        match self {
            Self::NewOrder(request) => request.account_id,
            Self::CancelOrder(request) => request.account_id,
            Self::ClosePosition(request) => request.account_id,
        }
    }

    fn to_message(&self, client_msg_id: &str) -> CTraderOpenApiJsonMessage {
        match self {
            Self::NewOrder(request) => build_new_order_request(request, client_msg_id),
            Self::CancelOrder(request) => build_cancel_order_request(request, client_msg_id),
            Self::ClosePosition(request) => build_close_position_request(request, client_msg_id),
        }
    }
}

impl CTraderExecutionStatus {
    fn from_proto(value: i32) -> Result<Self> {
        match value {
            2 => Ok(Self::Accepted),
            3 => Ok(Self::Filled),
            4 => Ok(Self::Replaced),
            5 => Ok(Self::Cancelled),
            11 => Ok(Self::PartialFill),
            7 | 8 => Ok(Self::Failed),
            other => Err(anyhow!("unsupported cTrader execution type: {other}")),
        }
    }
}

impl ProductionCTraderExecutionBackend {
    fn execute_with_transport<T: CTraderOpenApiTransport>(
        transport: &T,
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome> {
        let account_id = request
            .account_id
            .parse::<i64>()
            .context("cTrader execution account id must be numeric")?;
        let order_message = request.request.to_message("execute-1");
        let responses = transport.send_sequence(&[
            build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
            build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
            order_message,
        ])?;
        if responses.len() != 3 {
            return Err(anyhow!(
                "expected 3 cTrader execution responses, received {}",
                responses.len()
            ));
        }
        ensure_payload_type(
            &responses[0],
            CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE - 2,
        )?;
        ensure_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
        parse_execution_outcome(&responses[2])
    }
}

impl CTraderExecutionBackend for ProductionCTraderExecutionBackend {
    fn execute(
        &self,
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome> {
        let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
        Self::execute_with_transport(&transport, request)
    }
}

#[cfg(test)]
impl StubCTraderExecutionBackend {
    pub fn succeed(outcome: CTraderExecutionOutcome) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(outcome)))),
        }
    }

    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
        }
    }
}

#[cfg(test)]
impl CTraderExecutionBackend for StubCTraderExecutionBackend {
    fn execute(
        &self,
        _request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome> {
        self.outcome
            .lock()
            .expect("stub execution backend lock poisoned")
            .take()
            .unwrap_or_else(|| Err("missing stub execution outcome".to_string()))
            .map_err(|err| anyhow!(err))
    }
}

fn ensure_payload_type(response_json: &str, expected_payload_type: u32) -> Result<()> {
    let envelope: Value =
        serde_json::from_str(response_json).context("failed to parse cTrader JSON envelope")?;
    let payload_type = envelope
        .get("payloadType")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing payloadType in cTrader envelope"))? as u32;
    if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!("cTrader execution transport returned error payload"));
    }
    if payload_type != expected_payload_type {
        return Err(anyhow!(
            "unexpected cTrader payload type: expected {}, got {}",
            expected_payload_type,
            payload_type
        ));
    }
    Ok(())
}

fn parse_execution_outcome(response_json: &str) -> Result<CTraderExecutionOutcome> {
    let envelope: Value =
        serde_json::from_str(response_json).context("failed to parse cTrader JSON envelope")?;
    let payload_type = envelope
        .get("payloadType")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing payloadType in cTrader envelope"))? as u32;
    match payload_type {
        CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE => parse_execution_event(response_json),
        CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE => parse_order_error_event(response_json),
        other => Err(anyhow!(
            "unexpected cTrader execution response payload type: {other}"
        )),
    }
}

fn parse_execution_event(response_json: &str) -> Result<CTraderExecutionOutcome> {
    let envelope: ExecutionEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader execution event")?;
    if envelope.payload_type != CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader execution event payload type: {}",
            envelope.payload_type
        ));
    }

    let status = CTraderExecutionStatus::from_proto(envelope.payload.execution_type)?;
    let order = envelope.payload.order;
    let position = envelope.payload.position;
    let deal = envelope.payload.deal;
    let money_digits = deal
        .as_ref()
        .and_then(|item| item.close_position_detail.as_ref())
        .and_then(|detail| detail.money_digits)
        .or_else(|| deal.as_ref().and_then(|item| item.money_digits))
        .unwrap_or(0);

    let gross_profit = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .map(|detail| scaled_money(detail.gross_profit, money_digits))
    });
    let fee = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .map(|detail| scaled_money(detail.commission, money_digits))
            .or_else(|| item.commission.map(|commission| scaled_money(commission, money_digits)))
    });
    let swap = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .map(|detail| scaled_money(detail.swap, money_digits))
    });
    let pnl_conversion_fee = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .and_then(|detail| detail.pnl_conversion_fee)
            .map(|fee| scaled_money(fee, money_digits))
    });
    let net_profit = match (gross_profit, fee, swap, pnl_conversion_fee) {
        (Some(gross), fee, swap, pnl_fee) => Some(gross + fee.unwrap_or(0.0) + swap.unwrap_or(0.0) + pnl_fee.unwrap_or(0.0)),
        _ => None,
    };

    Ok(CTraderExecutionOutcome {
        status,
        account_id: envelope.payload.ctid_trader_account_id,
        symbol_id: order
            .as_ref()
            .map(|item| item.trade_data.symbol_id)
            .or_else(|| position.as_ref().map(|item| item.trade_data.symbol_id))
            .or_else(|| deal.as_ref().map(|item| item.symbol_id)),
        order_id: order.as_ref().map(|item| item.order_id).or_else(|| deal.as_ref().map(|item| item.order_id)),
        position_id: position
            .as_ref()
            .map(|item| item.position_id)
            .or_else(|| deal.as_ref().map(|item| item.position_id)),
        deal_id: deal.as_ref().map(|item| item.deal_id),
        trade_side: order
            .as_ref()
            .map(|item| trade_side_label(item.trade_data.trade_side))
            .or_else(|| position.as_ref().map(|item| trade_side_label(item.trade_data.trade_side)))
            .or_else(|| deal.as_ref().map(|item| trade_side_label(item.trade_side))),
        order_type: order.as_ref().map(|item| order_type_label(item.order_type)),
        lot_size: order
            .as_ref()
            .map(|item| volume_to_units(item.trade_data.volume))
            .or_else(|| position.as_ref().map(|item| volume_to_units(item.trade_data.volume)))
            .or_else(|| deal.as_ref().map(|item| volume_to_units(item.filled_volume))),
        execution_price: deal
            .as_ref()
            .and_then(|item| item.execution_price)
            .or_else(|| order.as_ref().and_then(|item| item.execution_price))
            .or_else(|| position.as_ref().and_then(|item| item.price)),
        gross_profit,
        fee,
        swap,
        net_profit,
        timestamp_ms: deal
            .as_ref()
            .map(|item| item.execution_timestamp)
            .or_else(|| order.as_ref().and_then(|item| item.trade_data.open_timestamp))
            .or_else(|| position.as_ref().and_then(|item| item.trade_data.open_timestamp)),
        error_code: envelope.payload.error_code,
        description: None,
    })
}

fn parse_order_error_event(response_json: &str) -> Result<CTraderExecutionOutcome> {
    let envelope: OrderErrorEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader order error event")?;
    if envelope.payload_type != CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader order error payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(CTraderExecutionOutcome {
        status: CTraderExecutionStatus::Failed,
        account_id: envelope.payload.ctid_trader_account_id,
        symbol_id: None,
        order_id: envelope.payload.order_id,
        position_id: envelope.payload.position_id,
        deal_id: None,
        trade_side: None,
        order_type: None,
        lot_size: None,
        execution_price: None,
        gross_profit: None,
        fee: None,
        swap: None,
        net_profit: None,
        timestamp_ms: None,
        error_code: Some(envelope.payload.error_code),
        description: envelope.payload.description,
    })
}

fn scaled_money(raw: i64, money_digits: u32) -> f64 {
    let divisor = 10f64.powi(money_digits as i32);
    raw as f64 / divisor
}

fn volume_to_units(raw: i64) -> f64 {
    raw as f64 / 100.0
}

fn trade_side_label(value: i32) -> String {
    match value {
        1 => "BUY".to_string(),
        2 => "SELL".to_string(),
        other => format!("SIDE_{other}"),
    }
}

fn order_type_label(value: i32) -> String {
    match value {
        1 => "MARKET".to_string(),
        2 => "LIMIT".to_string(),
        3 => "STOP".to_string(),
        4 => "STOP_LOSS_TAKE_PROFIT".to_string(),
        5 => "MARKET_RANGE".to_string(),
        6 => "STOP_LIMIT".to_string(),
        other => format!("ORDER_{other}"),
    }
}

#[cfg(test)]
mod tests {
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
}
