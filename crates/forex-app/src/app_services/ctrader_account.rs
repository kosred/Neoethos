use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE, CTraderDealListRequest, CTraderOpenApiTransport,
    ProductionCTraderOpenApiTransport, build_account_auth_request, build_application_auth_request,
    build_deal_list_request, build_reconcile_request, build_trader_request,
    parse_ctrader_error_payload, parse_open_api_envelope,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
#[cfg(test)]
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_CTRADER_DEAL_LOOKBACK_HOURS: i64 = 24;
const DEFAULT_CTRADER_DEAL_MAX_ROWS: i32 = 100;

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
    /// Sum of mark-to-market PnL for currently open positions (account currency).
    /// Updated by the streaming/spot subsystem; defaults to 0.0 when no live
    /// spot data is available. Read alongside `balance` to compute live equity:
    /// `equity = balance + unrealized_pnl`. Critical for prop-firm rules that
    /// limit drawdown by EQUITY, not balance.
    pub unrealized_pnl: f64,
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
    pub swap: Option<f64>,
    pub commission: Option<f64>,
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
pub struct CTraderDealSnapshot {
    pub deal_id: i64,
    pub order_id: i64,
    pub position_id: i64,
    pub symbol_id: i64,
    pub trade_side: String,
    pub deal_status: String,
    pub volume: f64,
    pub filled_volume: f64,
    pub execution_timestamp_ms: i64,
    pub execution_price: Option<f64>,
    pub entry_price: Option<f64>,
    pub gross_profit: Option<f64>,
    pub fee: Option<f64>,
    pub swap: Option<f64>,
    pub pnl_conversion_fee: Option<f64>,
    pub net_profit: Option<f64>,
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
    pub recent_deals: Vec<CTraderDealSnapshot>,
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
struct DealListEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: DealListPayload,
}

#[derive(Debug, Deserialize)]
struct DealListPayload {
    #[serde(default, rename = "deal")]
    deals: Vec<DealPayload>,
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
    swap: Option<i64>,
    commission: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
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

#[derive(Debug, Deserialize)]
struct DealPayload {
    #[serde(rename = "dealId")]
    deal_id: i64,
    #[serde(rename = "orderId")]
    order_id: i64,
    #[serde(rename = "positionId")]
    position_id: i64,
    volume: i64,
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
    #[serde(rename = "dealStatus")]
    deal_status: i32,
    commission: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
    #[serde(rename = "closePositionDetail")]
    close_position_detail: Option<ClosePositionDetailPayload>,
}

#[derive(Debug, Deserialize)]
struct ClosePositionDetailPayload {
    #[serde(rename = "entryPrice")]
    entry_price: Option<f64>,
    #[serde(rename = "grossProfit")]
    gross_profit: i64,
    swap: i64,
    commission: i64,
    #[serde(rename = "pnlConversionFee")]
    pnl_conversion_fee: Option<i64>,
    #[serde(rename = "moneyDigits")]
    money_digits: Option<u32>,
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
        unrealized_pnl: 0.0,
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
                swap: position
                    .swap
                    .map(|raw| scaled_money(raw, position.money_digits.unwrap_or(0))),
                commission: position
                    .commission
                    .map(|raw| scaled_money(raw, position.money_digits.unwrap_or(0))),
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

pub fn parse_deal_list_response(response_json: &str) -> Result<Vec<CTraderDealSnapshot>> {
    let envelope: DealListEnvelope = serde_json::from_str(response_json)
        .context("failed to parse cTrader deal list response")?;
    if envelope.payload_type != CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader deal list payload type: {}",
            envelope.payload_type
        ));
    }

    Ok(envelope
        .payload
        .deals
        .into_iter()
        .map(|deal| {
            let gross_profit = deal
                .close_position_detail
                .as_ref()
                .map(|detail| scaled_money(detail.gross_profit, detail.money_digits.unwrap_or(0)));
            let fee = deal
                .close_position_detail
                .as_ref()
                .map(|detail| scaled_money(detail.commission, detail.money_digits.unwrap_or(0)))
                .or_else(|| {
                    deal.commission
                        .map(|commission| scaled_money(commission, deal.money_digits.unwrap_or(0)))
                });
            let swap = deal
                .close_position_detail
                .as_ref()
                .map(|detail| scaled_money(detail.swap, detail.money_digits.unwrap_or(0)));
            let pnl_conversion_fee = deal.close_position_detail.as_ref().and_then(|detail| {
                detail
                    .pnl_conversion_fee
                    .map(|fee| scaled_money(fee, detail.money_digits.unwrap_or(0)))
            });
            let net_profit = gross_profit.map(|gross| {
                gross + fee.unwrap_or(0.0) + swap.unwrap_or(0.0) + pnl_conversion_fee.unwrap_or(0.0)
            });

            CTraderDealSnapshot {
                deal_id: deal.deal_id,
                order_id: deal.order_id,
                position_id: deal.position_id,
                symbol_id: deal.symbol_id,
                trade_side: trade_side_label(deal.trade_side),
                deal_status: deal_status_label(deal.deal_status),
                volume: volume_to_units(deal.volume),
                filled_volume: volume_to_units(deal.filled_volume),
                execution_timestamp_ms: deal.execution_timestamp,
                execution_price: deal.execution_price,
                entry_price: deal
                    .close_position_detail
                    .as_ref()
                    .and_then(|detail| detail.entry_price),
                gross_profit,
                fee,
                swap,
                pnl_conversion_fee,
                net_profit,
            }
        })
        .collect())
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
        build_deal_list_request(
            &CTraderDealListRequest {
                account_id,
                from_timestamp_ms: Some(
                    current_unix_millis()? - DEFAULT_CTRADER_DEAL_LOOKBACK_HOURS * 60 * 60 * 1000,
                ),
                to_timestamp_ms: Some(current_unix_millis()?),
                max_rows: Some(DEFAULT_CTRADER_DEAL_MAX_ROWS),
            },
            "deals-1",
        ),
    ])?;
    if responses.len() != 5 {
        return Err(anyhow!(
            "expected 5 cTrader account runtime responses, received {}",
            responses.len()
        ));
    }

    ensure_success_payload_type(
        &responses[0],
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    )?;
    ensure_success_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[2], CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[3], CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE)?;
    ensure_success_payload_type(&responses[4], CTRADER_OA_DEAL_LIST_RESPONSE_PAYLOAD_TYPE)?;

    Ok(CTraderAccountRuntimeSnapshot {
        trader: parse_trader_response(&responses[2])?,
        reconcile: parse_reconcile_response(&responses[3])?,
        recent_deals: parse_deal_list_response(&responses[4])?,
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

fn deal_status_label(value: i32) -> String {
    match value {
        2 => "FILLED",
        3 => "PARTIALLY_FILLED",
        4 => "REJECTED",
        5 => "INTERNALLY_REJECTED",
        6 => "ERROR",
        7 => "MISSED",
        other => return format!("UNKNOWN({other})"),
    }
    .to_string()
}

fn current_unix_millis() -> Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow!("system clock is before unix epoch"))?
        .as_millis() as i64)
}

#[cfg(test)]
#[path = "ctrader_account_tests.rs"]
mod tests;
