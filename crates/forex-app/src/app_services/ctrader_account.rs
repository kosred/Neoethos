use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    build_account_auth_request, build_application_auth_request, build_reconcile_request,
    build_trader_request, parse_ctrader_error_payload, parse_open_api_envelope,
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE,
};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAccountRuntimeRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub return_protection_orders: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderTraderSnapshot {
    pub account_id: i64,
    pub balance: f64,
    pub leverage: Option<f64>,
    pub trader_login: Option<i64>,
    pub account_type: Option<String>,
    pub broker_name: Option<String>,
    pub money_digits: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderPositionSnapshot {
    pub position_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub volume: f64,
    pub open_timestamp_ms: Option<i64>,
    pub price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub label: Option<String>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderPendingOrderSnapshot {
    pub order_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub order_type: String,
    pub volume: f64,
    pub open_timestamp_ms: Option<i64>,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub label: Option<String>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderReconcileSnapshot {
    pub account_id: i64,
    pub positions: Vec<CTraderPositionSnapshot>,
    pub pending_orders: Vec<CTraderPendingOrderSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderAccountRuntimeSnapshot {
    pub trader: CTraderTraderSnapshot,
    pub reconcile: CTraderReconcileSnapshot,
}

pub trait CTraderAccountRuntimeBackend: Send + Sync {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderAccountRuntimeBackend;

#[cfg(test)]
#[derive(Clone)]
pub struct StubCTraderAccountRuntimeBackend {
    outcome: Arc<Mutex<Option<Result<CTraderAccountRuntimeSnapshot, String>>>>,
    last_request: Arc<Mutex<Option<CTraderAccountRuntimeRequest>>>,
}

#[derive(Debug, Deserialize)]
struct TraderEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: TraderPayload,
}

#[derive(Debug, Deserialize)]
struct TraderPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    balance: i64,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(rename = "leverageInCents")]
    leverage_in_cents: Option<u32>,
    #[serde(rename = "traderLogin")]
    trader_login: Option<i64>,
    #[serde(rename = "accountType")]
    account_type: Option<i32>,
    #[serde(rename = "brokerName")]
    broker_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReconcileEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: ReconcilePayload,
}

#[derive(Debug, Deserialize)]
struct ReconcilePayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(default, rename = "position")]
    positions: Vec<PositionPayload>,
    #[serde(default, rename = "order")]
    orders: Vec<OrderPayload>,
}

#[derive(Debug, Deserialize)]
struct PositionPayload {
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: TradeDataPayload,
    price: Option<f64>,
    #[serde(rename = "stopLoss")]
    stop_loss: Option<f64>,
    #[serde(rename = "takeProfit")]
    take_profit: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OrderPayload {
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: TradeDataPayload,
    #[serde(rename = "orderType")]
    order_type: i32,
    #[serde(rename = "limitPrice")]
    limit_price: Option<f64>,
    #[serde(rename = "stopPrice")]
    stop_price: Option<f64>,
    #[serde(rename = "stopLoss")]
    stop_loss: Option<f64>,
    #[serde(rename = "takeProfit")]
    take_profit: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TradeDataPayload {
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    volume: i64,
    #[serde(rename = "tradeSide")]
    trade_side: i32,
    #[serde(rename = "openTimestamp")]
    open_timestamp: Option<i64>,
    label: Option<String>,
    comment: Option<String>,
}

pub fn parse_trader_response(response_json: &str) -> Result<CTraderTraderSnapshot> {
    let envelope: TraderEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader trader response")?;
    if envelope.payload_type != CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader trader payload type: {}",
            envelope.payload_type
        ));
    }

    let money_digits = envelope.payload.money_digits.unwrap_or(0);
    Ok(CTraderTraderSnapshot {
        account_id: envelope.payload.ctid_trader_account_id,
        balance: scaled_money(envelope.payload.balance, money_digits),
        leverage: envelope
            .payload
            .leverage_in_cents
            .map(|value| value as f64 / 100.0),
        trader_login: envelope.payload.trader_login,
        account_type: envelope.payload.account_type.map(account_type_label),
        broker_name: envelope.payload.broker_name,
        money_digits,
    })
}

pub fn parse_reconcile_response(response_json: &str) -> Result<CTraderReconcileSnapshot> {
    let envelope: ReconcileEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader reconcile response")?;
    if envelope.payload_type != CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader reconcile payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(CTraderReconcileSnapshot {
        account_id: envelope.payload.ctid_trader_account_id,
        positions: envelope
            .payload
            .positions
            .into_iter()
            .map(|position| CTraderPositionSnapshot {
                position_id: position.position_id,
                symbol_id: position.trade_data.symbol_id,
                trade_side: trade_side_label(position.trade_data.trade_side),
                volume: volume_to_units(position.trade_data.volume),
                open_timestamp_ms: position.trade_data.open_timestamp,
                price: position.price,
                stop_loss: position.stop_loss,
                take_profit: position.take_profit,
                label: position.trade_data.label,
                comment: position.trade_data.comment,
            })
            .collect(),
        pending_orders: envelope
            .payload
            .orders
            .into_iter()
            .map(|order| CTraderPendingOrderSnapshot {
                order_id: order.order_id,
                symbol_id: order.trade_data.symbol_id,
                trade_side: trade_side_label(order.trade_data.trade_side),
                order_type: order_type_label(order.order_type),
                volume: volume_to_units(order.trade_data.volume),
                open_timestamp_ms: order.trade_data.open_timestamp,
                limit_price: order.limit_price,
                stop_price: order.stop_price,
                stop_loss: order.stop_loss,
                take_profit: order.take_profit,
                label: order.trade_data.label,
                comment: order.trade_data.comment,
            })
            .collect(),
    })
}

pub fn load_account_runtime_with_transport<T: CTraderOpenApiTransport>(
    transport: &T,
    request: &CTraderAccountRuntimeRequest,
) -> Result<CTraderAccountRuntimeSnapshot> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;
    let responses = transport.send_sequence(&[
        build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
        build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
        build_trader_request(account_id, "trader-1"),
        build_reconcile_request(account_id, request.return_protection_orders, "reconcile-1"),
    ])?;
    if responses.len() != 4 {
        return Err(anyhow!(
            "expected 4 cTrader account runtime responses, received {}",
            responses.len()
        ));
    }

    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(
        &responses[1],
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[2], CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[3], CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE)?;

    Ok(CTraderAccountRuntimeSnapshot {
        trader: parse_trader_response(&responses[2])?,
        reconcile: parse_reconcile_response(&responses[3])?,
    })
}

pub fn load_account_runtime(
    request: &CTraderAccountRuntimeRequest,
) -> Result<CTraderAccountRuntimeSnapshot> {
    let transport = ProductionCTraderOpenApiTransport::new(request.environment.endpoint_host());
    load_account_runtime_with_transport(&transport, request)
}

impl CTraderAccountRuntimeBackend for ProductionCTraderAccountRuntimeBackend {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot> {
        load_account_runtime(request)
    }
}

#[cfg(test)]
impl StubCTraderAccountRuntimeBackend {
    pub fn success(snapshot: CTraderAccountRuntimeSnapshot) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Ok(snapshot)))),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            outcome: Arc::new(Mutex::new(Some(Err(message.into())))),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn last_request(&self) -> Option<CTraderAccountRuntimeRequest> {
        self.last_request.lock().expect("last request lock").clone()
    }
}

#[cfg(test)]
impl CTraderAccountRuntimeBackend for StubCTraderAccountRuntimeBackend {
    fn load_account_runtime(
        &self,
        request: &CTraderAccountRuntimeRequest,
    ) -> Result<CTraderAccountRuntimeSnapshot> {
        *self.last_request.lock().expect("last request lock") = Some(request.clone());
        let outcome = self
            .outcome
            .lock()
            .expect("runtime outcome lock")
            .take()
            .unwrap_or_else(|| Err("stub cTrader account runtime backend exhausted".to_string()));
        outcome.map_err(|message| anyhow!(message))
    }
}

fn ensure_success_payload_type(response_json: &str, expected_payload_type: u32) -> Result<()> {
    let envelope = parse_open_api_envelope(response_json)?;
    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader response failed: {}",
            parse_ctrader_error_payload(&envelope.payload)?
        ));
    }
    if envelope.payload_type != expected_payload_type {
        return Err(anyhow!(
            "unexpected cTrader payload type: expected {}, got {}",
            expected_payload_type,
            envelope.payload_type
        ));
    }
    Ok(())
}

fn scaled_money(value: i64, digits: u32) -> f64 {
    let factor = 10_f64.powi(digits as i32);
    value as f64 / factor
}

fn volume_to_units(value: i64) -> f64 {
    value as f64 / 100.0
}

fn account_type_label(value: i32) -> String {
    match value {
        0 => "HEDGED",
        1 => "NETTED",
        2 => "SPREAD_BETTING",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn trade_side_label(value: i32) -> String {
    match value {
        1 => "BUY",
        2 => "SELL",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn order_type_label(value: i32) -> String {
    match value {
        1 => "MARKET",
        2 => "LIMIT",
        3 => "STOP",
        4 => "STOP_LOSS_TAKE_PROFIT",
        5 => "MARKET_RANGE",
        6 => "STOP_LIMIT",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_live_auth::CTraderEnvironment;
    use crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage;

    #[test]
    fn trader_response_parses_balance_and_account_metadata() {
        let response = serde_json::json!({
            "clientMsgId": "trader-1",
            "payloadType": 2122,
            "payload": {
                "ctidTraderAccountId": 712345,
                "balance": 123456789,
                "moneyDigits": 2,
                "leverageInCents": 5000,
                "traderLogin": 998877,
                "accountType": 1,
                "brokerName": "Spotware Demo Broker"
            }
        });

        let trader = parse_trader_response(&response.to_string()).expect("trader response");

        assert_eq!(trader.account_id, 712345);
        assert!((trader.balance - 1_234_567.89).abs() < 1e-9);
        assert_eq!(trader.leverage, Some(50.0));
        assert_eq!(trader.trader_login, Some(998877));
        assert_eq!(trader.account_type.as_deref(), Some("NETTED"));
        assert_eq!(trader.broker_name.as_deref(), Some("Spotware Demo Broker"));
    }

    #[test]
    fn reconcile_response_parses_positions_and_pending_orders() {
        let response = serde_json::json!({
            "clientMsgId": "reconcile-1",
            "payloadType": 2125,
            "payload": {
                "ctidTraderAccountId": 712345,
                "position": [
                    {
                        "positionId": 9001,
                        "tradeData": {
                            "symbolId": 14,
                            "volume": 2500,
                            "tradeSide": 1,
                            "openTimestamp": 1710000000000i64,
                            "label": "trend",
                            "comment": "bot"
                        },
                        "positionStatus": 1,
                        "price": 1.10123,
                        "stopLoss": 1.095,
                        "takeProfit": 1.11
                    }
                ],
                "order": [
                    {
                        "orderId": 8001,
                        "tradeData": {
                            "symbolId": 14,
                            "volume": 1500,
                            "tradeSide": 2,
                            "openTimestamp": 1710000100000i64,
                            "label": "breakout",
                            "comment": "pending"
                        },
                        "orderType": 2,
                        "orderStatus": 1,
                        "limitPrice": 1.099,
                        "stopLoss": 1.105,
                        "takeProfit": 1.09
                    }
                ]
            }
        });

        let reconcile = parse_reconcile_response(&response.to_string()).expect("reconcile");

        assert_eq!(reconcile.account_id, 712345);
        assert_eq!(reconcile.positions.len(), 1);
        assert_eq!(reconcile.pending_orders.len(), 1);
        assert_eq!(reconcile.positions[0].position_id, 9001);
        assert_eq!(reconcile.positions[0].trade_side, "BUY");
        assert_eq!(reconcile.positions[0].symbol_id, 14);
        assert!((reconcile.positions[0].volume - 25.0).abs() < 1e-9);
        assert_eq!(reconcile.pending_orders[0].order_id, 8001);
        assert_eq!(reconcile.pending_orders[0].trade_side, "SELL");
        assert_eq!(reconcile.pending_orders[0].order_type, "LIMIT");
        assert!((reconcile.pending_orders[0].limit_price.unwrap_or_default() - 1.099).abs() < 1e-9);
    }

    #[test]
    fn account_runtime_loader_authenticates_then_loads_trader_and_reconcile() {
        let transport = StubTransport::with_responses(vec![
            Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
            Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
            Ok(r#"{"clientMsgId":"trader-1","payloadType":2122,"payload":{"ctidTraderAccountId":712345,"balance":100000,"moneyDigits":2,"leverageInCents":5000,"brokerName":"Demo Broker"}}"#.to_string()),
            Ok(r#"{"clientMsgId":"reconcile-1","payloadType":2125,"payload":{"ctidTraderAccountId":712345,"position":[{"positionId":9001,"tradeData":{"symbolId":14,"volume":2500,"tradeSide":1,"openTimestamp":1710000000000},"positionStatus":1,"price":1.10123}],"order":[]}}"#.to_string()),
        ]);

        let runtime = load_account_runtime_with_transport(
            &transport,
            &CTraderAccountRuntimeRequest {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                access_token: "access".to_string(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".to_string(),
                return_protection_orders: true,
            },
        )
        .expect("account runtime");

        assert_eq!(runtime.trader.account_id, 712345);
        assert_eq!(runtime.reconcile.positions.len(), 1);
        assert_eq!(transport.sent_len(), 4);
    }

    #[test]
    fn stub_runtime_backend_records_request_and_surfaces_failure() {
        let backend = StubCTraderAccountRuntimeBackend::failure("runtime probe failed");
        let request = CTraderAccountRuntimeRequest {
            client_id: "client".to_string(),
            client_secret: "secret".to_string(),
            access_token: "access".to_string(),
            environment: CTraderEnvironment::Demo,
            account_id: "712345".to_string(),
            return_protection_orders: true,
        };

        let error = backend
            .load_account_runtime(&request)
            .expect_err("stub backend should fail");

        assert!(error.to_string().contains("runtime probe failed"));
        assert_eq!(backend.last_request(), Some(request));
    }

    struct StubTransport {
        sent: std::sync::Mutex<Vec<CTraderOpenApiJsonMessage>>,
        responses: std::sync::Mutex<Vec<anyhow::Result<String>>>,
    }

    impl StubTransport {
        fn with_responses(responses: Vec<anyhow::Result<String>>) -> Self {
            Self {
                sent: std::sync::Mutex::new(Vec::new()),
                responses: std::sync::Mutex::new(responses),
            }
        }

        fn sent_len(&self) -> usize {
            self.sent.lock().expect("sent lock").len()
        }
    }

    impl crate::app_services::ctrader_messages::CTraderOpenApiTransport for StubTransport {
        fn send_sequence(
            &self,
            messages: &[CTraderOpenApiJsonMessage],
        ) -> anyhow::Result<Vec<String>> {
            self.sent
                .lock()
                .expect("sent lock")
                .extend(messages.iter().cloned());
            let mut responses = self.responses.lock().expect("responses lock");
            let mut output = Vec::with_capacity(messages.len());
            for _ in messages {
                output.push(responses.remove(0)?);
            }
            Ok(output)
        }
    }
}
