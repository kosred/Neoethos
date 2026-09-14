use crate::app_services::broker_deal_economics::BrokerSymbolVolumeScaleEvidenceV1;
use crate::app_services::ctrader_historical_admission::{
    CTRADER_RESPONSE_TIMEOUT, CTraderIoPhase, CTraderMonotonicClock, CTraderOperationBudget,
    SystemCTraderMonotonicClock,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
#[cfg(test)]
use crate::app_services::ctrader_messages::CTraderOpenApiTransport;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE, CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE,
    CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderAmendOrderRequest, CTraderAmendPositionSltpRequest,
    CTraderCancelOrderRequest, CTraderNewOrderRequest, CTraderOpenApiJsonMessage, CTraderOrderType,
    ProductionCTraderBudget, ProductionCTraderSocket, arm_ctrader_socket_budget,
    arm_ctrader_socket_with_budget, build_account_auth_request, build_amend_order_request,
    build_amend_position_sltp_request, build_application_auth_request, build_cancel_order_request,
    build_close_position_request, build_new_order_request, connect_ctrader_socket,
    expected_response_payload_type, is_ctrader_auth_token_error, is_ctrader_socket_poll_timeout,
    is_matching_open_api_response, parse_ctrader_error_payload_parts, parse_open_api_envelope,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tungstenite::Message;

#[derive(Debug, Clone, PartialEq)]
pub enum CTraderExecutionRequest {
    NewOrder(Box<CTraderNewOrderRequest>),
    CancelOrder(CTraderCancelOrderRequest),
    ClosePosition(crate::app_services::ctrader_messages::CTraderClosePositionRequest),
    /// Modify an open position's SL/TP (2026-06-10). Like the others the broker
    /// answers with a `ProtoOAExecutionEvent`, so it rides the same retry +
    /// idempotency path.
    AmendPositionSltp(CTraderAmendPositionSltpRequest),
    /// Modify a RESTING (pending) order — `ProtoOAAmendOrderReq` (2109). Wired
    /// 2026-08-10 (audit #236).
    ///
    /// Distinct from [`Self::AmendPositionSltp`], which modifies a FILLED
    /// position: this one changes the order that has not filled yet — its
    /// trigger price, its volume, its expiry, its bracket. Until today
    /// `build_amend_order_request` existed with no variant to carry it and no
    /// caller, so the UI could place a pending order and cancel it but not
    /// change it: correcting a trigger price meant cancel + re-place, which is
    /// two broker round trips and a window in which the operator has no resting
    /// order at all.
    AmendOrder(Box<CTraderAmendOrderRequest>),
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

/// Typed local recovery context, not proof of no fill or permission to resend.
/// accepted_order_id is a request-bound broker order reference: a validated
/// acceptance OR direct filled/partial terminal observation, not opening proof.
#[derive(Debug, Clone)]
pub(crate) struct CTraderUnresolvedExecution {
    pub(crate) environment: CTraderEnvironment,
    pub(crate) account_id: i64,
    pub(crate) client_order_id: Option<String>,
    pub(crate) accepted_order_id: Option<i64>,
}

impl std::fmt::Display for CTraderUnresolvedExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "immediate execution unresolved; accepted_order_id={:?}; no resend",
            self.accepted_order_id
        )
    }
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
    /// Headline lot size kept for backwards compatibility. Prefer
    /// [`Self::requested_lot_size`] / [`Self::filled_lot_size`] when
    /// reasoning about partial fills.
    pub lot_size: Option<f64>,
    /// Volume the strategy asked the broker to execute, in lots
    /// (sourced from the order payload). `Some(2.0)` on a 2-lot
    /// market order regardless of fill outcome.
    pub requested_lot_size: Option<f64>,
    /// Volume the broker actually filled, in lots (sourced from the
    /// deal payload). On a clean fill matches `requested_lot_size`;
    /// on a partial fill it is strictly smaller; on a rejection it
    /// is `None` or 0.0. Lets the trading loop decide whether to
    /// scale-in the residual or cancel-and-log.
    pub filled_lot_size: Option<f64>,
    /// Exact `ProtoOADeal.filledVolume` wire integer (centi-units).
    pub filled_volume_raw_centi_units: Option<i64>,
    /// Exact broker symbol/account/environment lot-size identity used by the
    /// order-preparation path. Raw event parsers leave it absent; the submitting
    /// broker API binds it before returning the outcome to its caller.
    pub volume_scale_evidence: Option<BrokerSymbolVolumeScaleEvidenceV1>,
    /// A broker deal's closePositionDetail distinguishes a closing/reducing
    /// execution (including NETTED reductions) from an opening execution.
    /// None means that the event carried no deal, not that no fill happened.
    pub deal_closes_position: Option<bool>,
    /// Exact single-fill opening-position evidence. This is absent for accepted
    /// orders, partial/multi-fill entries, reductions and incomplete wire data.
    /// Absence never means an order was definitely not sent or filled.
    pub opening_fill_evidence: Option<CTraderOpeningFillEvidenceV1>,
    pub execution_price: Option<f64>,
    pub gross_profit: Option<f64>,
    pub fee: Option<f64>,
    pub swap: Option<f64>,
    pub net_profit: Option<f64>,
    pub timestamp_ms: Option<i64>,
    pub error_code: Option<String>,
    pub description: Option<String>,
}

/// Mutually bound facts from one genuine opening order/position/deal event.
/// No environment, currency, lot size, admission or historical completeness is
/// inferred here. The submitting API supplies its captured volume-scale scope.
#[derive(Debug, Clone, PartialEq)]
pub struct CTraderOpeningFillEvidenceV1 {
    account_id: i64,
    order_id: i64,
    position_id: i64,
    deal_id: i64,
    symbol_id: i64,
    trade_side: String,
    filled_volume_raw_centi_units: i64,
    position_open_timestamp_ms: i64,
    execution_timestamp_ms: i64,
    entry_price: f64,
}

impl CTraderOpeningFillEvidenceV1 {
    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn order_id(&self) -> i64 {
        self.order_id
    }

    pub const fn position_id(&self) -> i64 {
        self.position_id
    }

    pub const fn deal_id(&self) -> i64 {
        self.deal_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn trade_side(&self) -> &str {
        &self.trade_side
    }

    pub const fn filled_volume_raw_centi_units(&self) -> i64 {
        self.filled_volume_raw_centi_units
    }
    pub const fn position_open_timestamp_ms(&self) -> i64 {
        self.position_open_timestamp_ms
    }
    pub const fn execution_timestamp_ms(&self) -> i64 {
        self.execution_timestamp_ms
    }

    pub const fn entry_price(&self) -> f64 {
        self.entry_price
    }
}

pub trait CTraderExecutionBackend: Send + Sync {
    fn execute(&self, request: &CTraderExecutionRuntimeRequest) -> Result<CTraderExecutionOutcome>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderExecutionBackend;

#[derive(Default)]
struct CTraderExecutionSession {
    socket: Option<ProductionCTraderSocket>,
    auth_key: Option<String>,
    recent_submissions: HashMap<String, CachedExecutionOutcome>,
}

#[derive(Debug, Clone)]
struct CachedExecutionOutcome {
    created_at: Instant,
    outcome: CTraderExecutionOutcome,
}

static EXECUTION_SESSION: OnceLock<Mutex<CTraderExecutionSession>> = OnceLock::new();

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
    #[serde(rename = "orderStatus")]
    order_status: Option<i32>,
    #[serde(rename = "clientOrderId")]
    client_order_id: Option<String>,
    #[serde(rename = "executedVolume")]
    executed_volume: Option<i64>,
    #[serde(rename = "closingOrder")]
    closing_order: Option<bool>,
    #[serde(rename = "positionId")]
    position_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ExecutionPositionPayload {
    #[serde(rename = "positionId")]
    position_id: i64,
    #[serde(rename = "tradeData")]
    trade_data: ExecutionTradeDataPayload,
    price: Option<f64>,
    #[serde(rename = "positionStatus")]
    position_status: Option<i32>,
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
    #[serde(rename = "dealStatus")]
    deal_status: Option<i32>,
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
            Self::AmendPositionSltp(request) => request.account_id,
            Self::AmendOrder(request) => request.account_id,
        }
    }

    fn to_message(&self, client_msg_id: &str) -> CTraderOpenApiJsonMessage {
        match self {
            Self::NewOrder(request) => build_new_order_request(request, client_msg_id),
            Self::CancelOrder(request) => build_cancel_order_request(request, client_msg_id),
            Self::ClosePosition(request) => build_close_position_request(request, client_msg_id),
            Self::AmendPositionSltp(request) => {
                build_amend_position_sltp_request(request, client_msg_id)
            }
            Self::AmendOrder(request) => build_amend_order_request(request, client_msg_id),
        }
    }

    fn idempotency_fingerprint(&self) -> String {
        match self {
            Self::NewOrder(request) => format!(
                "new|acct={}|sym={}|side={}|otype={}|vol={}|limit={:?}|stop={:?}|tif={:?}|exp={:?}|sl={:?}|tp={:?}|comment={:?}|base_slippage={:?}|slip_pts={:?}|label={:?}|position_id={:?}|client_order_id={:?}|rsl={:?}|rtp={:?}|gsl={:?}|tsl={:?}|trigger={:?}",
                request.account_id,
                request.symbol_id,
                request.trade_side.label(),
                request.order_type.label(),
                request.volume,
                request.limit_price,
                request.stop_price,
                request.time_in_force.map(|v| v.label()),
                request.expiration_timestamp_ms,
                request.stop_loss,
                request.take_profit,
                request.comment,
                request.base_slippage_price,
                request.slippage_in_points,
                request.label,
                request.position_id,
                request.client_order_id,
                request.relative_stop_loss,
                request.relative_take_profit,
                request.guaranteed_stop_loss,
                request.trailing_stop_loss,
                request.stop_trigger_method.map(|v| v.label())
            ),
            Self::CancelOrder(request) => format!(
                "cancel|acct={}|order_id={}",
                request.account_id, request.order_id
            ),
            Self::ClosePosition(request) => format!(
                "close|acct={}|position_id={}|volume={}",
                request.account_id, request.position_id, request.volume
            ),
            Self::AmendPositionSltp(request) => format!(
                "amend_pos|acct={}|position_id={}|sl={:?}|tp={:?}|gsl={:?}|tsl={:?}|trigger={:?}",
                request.account_id,
                request.position_id,
                request.stop_loss,
                request.take_profit,
                request.guaranteed_stop_loss,
                request.trailing_stop_loss,
                request.stop_loss_trigger_method.map(|v| v.label())
            ),
            // Every field the request can carry is in the fingerprint, for the
            // same reason as `new`: two amends that differ in ANY value are
            // different intents, and collapsing them would let the idempotency
            // cache answer one with the other's result.
            Self::AmendOrder(request) => format!(
                "amend_order|acct={}|order_id={}|vol={:?}|limit={:?}|stop={:?}|exp={:?}|sl={:?}|tp={:?}|slip_pts={:?}|rsl={:?}|rtp={:?}|gsl={:?}|tsl={:?}|trigger={:?}",
                request.account_id,
                request.order_id,
                request.volume,
                request.limit_price,
                request.stop_price,
                request.expiration_timestamp_ms,
                request.stop_loss,
                request.take_profit,
                request.slippage_in_points,
                request.relative_stop_loss,
                request.relative_take_profit,
                request.guaranteed_stop_loss,
                request.trailing_stop_loss,
                request.stop_trigger_method.map(|v| v.label())
            ),
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

/// Immediate orders alone wait past acceptance. Pending orders retain their
/// existing acceptance response; their eventual execution may be hours later.
fn immediate_order(request: &CTraderExecutionRuntimeRequest) -> Option<&CTraderNewOrderRequest> {
    match &request.request {
        CTraderExecutionRequest::NewOrder(order)
            if matches!(
                order.order_type,
                CTraderOrderType::Market | CTraderOrderType::MarketRange
            ) =>
        {
            Some(order)
        }
        _ => None,
    }
}

fn immediate_execution_timeout(configured_seconds: u64) -> Duration {
    // A disabled legacy per-read timeout must not disable this absolute wait.
    Duration::from_secs(if configured_seconds == 0 {
        30
    } else {
        configured_seconds.min(30)
    })
}

fn cacheable_execution_outcome(
    request: &CTraderExecutionRuntimeRequest,
    outcome: &CTraderExecutionOutcome,
) -> bool {
    immediate_order(request).is_none() || outcome.status == CTraderExecutionStatus::Filled
}

/// Narrow injectable I/O seam: the production implementation uses the existing
/// DeadlineIo socket, which enforces the budget beneath TLS/fragmented frames.
trait ImmediateExecutionIo {
    fn send_frame(&mut self, frame: Message) -> std::result::Result<(), tungstenite::Error>;
    fn read_frame(&mut self) -> std::result::Result<Message, tungstenite::Error>;
}

impl ImmediateExecutionIo for ProductionCTraderSocket {
    fn send_frame(&mut self, frame: Message) -> std::result::Result<(), tungstenite::Error> {
        self.send(frame)
    }

    fn read_frame(&mut self) -> std::result::Result<Message, tungstenite::Error> {
        self.read()
    }
}

/// Collect one immediate order under one immutable budget and exactly one send.
/// Only the accepted broker order id is retained, not an unbounded event tape.
/// Partial/multi-fill, timeout and non-filled terminal states remain unresolved:
/// this collector does not implement durable intent/history recovery.
fn collect_immediate_execution<C: CTraderMonotonicClock>(
    request: &CTraderExecutionRuntimeRequest,
    message: &CTraderOpenApiJsonMessage,
    budget: &CTraderOperationBudget<C>,
    io: &mut impl ImmediateExecutionIo,
) -> Result<String> {
    let order =
        immediate_order(request).context("terminal collection requires an immediate order")?;
    anyhow::ensure!(
        order.account_id > 0
            && request.account_id.parse::<i64>()? == order.account_id
            && order.symbol_id > 0
            && order.volume > 0
            && !message.client_msg_id.is_empty()
            && *message == request.request.to_message(&message.client_msg_id),
        "immediate execution request identity is invalid"
    );
    let serialized = serde_json::to_string(message)?;
    let mut accepted_order_id = None;
    let mut accepted_position_id = None;
    let mut unrelated_execution_seen = false;
    let collected = (|| -> Result<String> {
        budget.check_io(CTraderIoPhase::RequestWrite)?;
        // A send error may already have reached the broker. Never resend this frame.
        io.send_frame(Message::Text(serialized.into()))?;
        budget.check_io(CTraderIoPhase::RequestWrite)?;
        loop {
            budget.check_io(CTraderIoPhase::ResponseRead)?;
            let frame = match io.read_frame() {
                Ok(frame) => frame,
                Err(error) if is_ctrader_socket_poll_timeout(&error) => {
                    budget.check_io(CTraderIoPhase::ResponseRead)?;
                    continue;
                }
                Err(error) => return Err(anyhow!(error).context("immediate response read failed")),
            };
            // Buffered or just-completed frames cannot win after the deadline.
            budget.check_io(CTraderIoPhase::ResponseRead)?;
            let text = match frame {
                Message::Text(text) => text.to_string(),
                Message::Binary(bytes) => String::from_utf8(bytes.to_vec())
                    .context("invalid UTF-8 execution frame; outcome unresolved")?,
                Message::Ping(payload) => {
                    budget.check_io(CTraderIoPhase::RequestWrite)?;
                    io.send_frame(Message::Pong(payload))?;
                    budget.check_io(CTraderIoPhase::RequestWrite)?;
                    continue;
                }
                Message::Pong(_) | Message::Frame(_) => continue,
                Message::Close(_) => anyhow::bail!(
                    "execution socket closed; accepted_order_id={accepted_order_id:?}; outcome unresolved; no resend"
                ),
            };
            if text.trim().is_empty() {
                continue;
            }
            let envelope = parse_open_api_envelope(&text)?;
            let correlated = envelope.client_msg_id == message.client_msg_id;
            let same_account = envelope
                .payload
                .get("ctidTraderAccountId")
                .and_then(Value::as_i64)
                == Some(order.account_id);
            let response_order_id = envelope
                .payload
                .get("order")
                .and_then(|item| item.get("orderId"))
                .or_else(|| envelope.payload.get("orderId"))
                .and_then(Value::as_i64);
            let known_order = same_account
                && accepted_order_id.is_some()
                && response_order_id == accepted_order_id;
            // A missing clientMsgId can describe an unsolicited follow-up only
            // after this request's actual account/order pair has been established.
            let owned_followup = known_order && envelope.client_msg_id.is_empty();
            if !correlated && !owned_followup {
                if known_order && !envelope.client_msg_id.is_empty() {
                    anyhow::bail!(
                        "known execution order has conflicting clientMsgId; outcome unresolved"
                    );
                }
                if matches!(
                    envelope.payload_type,
                    CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE
                        | CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE
                ) && !unrelated_execution_seen
                {
                    // Manual/other-order pushes are legitimate, but are not this
                    // collector's evidence. No durability/recovery claim is made.
                    tracing::debug!(target: "neoethos_app::ctrader",
                    "unrelated execution observed during terminal wait; not owned or persisted by this collector");
                    unrelated_execution_seen = true;
                }
                continue;
            }
            if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                // Preserve original broker error code/description in the existing
                // execute_authenticated_once parser, not a missing-order error.
                return Ok(text);
            }
            if envelope.payload_type == CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE {
                let error: OrderErrorEnvelope = serde_json::from_str(&text)?;
                anyhow::ensure!(
                    error.payload.ctid_trader_account_id == order.account_id
                        && accepted_order_id.is_none_or(|id| error.payload.order_id == Some(id)),
                    "order error account/order mismatch"
                );
                return Ok(text);
            }
            anyhow::ensure!(
                envelope.payload_type == CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE,
                "correlated immediate response has unexpected payload type"
            );
            let execution: ExecutionEnvelope = serde_json::from_str(&text)?;
            let payload = &execution.payload;
            anyhow::ensure!(
                payload.ctid_trader_account_id == order.account_id,
                "immediate execution account mismatch"
            );
            validate_execution_payload_links(payload)?;
            // Rejection may omit order details. It is still an error, not evidence
            // allowing automatic retry or release of an ambiguous reservation.
            if payload.execution_type == 7 && payload.order.is_none() && accepted_order_id.is_none()
            {
                return Ok(text);
            }
            let observed = payload
                .order
                .as_ref()
                .context("immediate execution missing order evidence")?;
            anyhow::ensure!(
                observed.trade_data.symbol_id == order.symbol_id
                    && trade_side_label(observed.trade_data.trade_side) == order.trade_side.label()
                    && order_type_label(observed.order_type) == order.order_type.label()
                    && observed.trade_data.volume == order.volume
                    && accepted_order_id.is_none_or(|id| observed.order_id == id)
                    && accepted_position_id.is_none_or(|id| observed.position_id.or_else(|| {
                        payload
                            .position
                            .as_ref()
                            .map(|position| position.position_id)
                    }) == Some(id))
                    && order
                        .position_id
                        .is_none_or(|id| observed.position_id == Some(id)),
                "immediate execution order identity differs from the request/acceptance"
            );
            if let Some(client_order_id) = observed.client_order_id.as_deref() {
                anyhow::ensure!(
                    order.client_order_id.as_deref() == Some(client_order_id),
                    "immediate execution root clientOrderId mismatch"
                );
            }
            // Identity is now bound to this request, including direct fills that
            // had no earlier Accepted event. Keep the reference even if parsing
            // financial/opening evidence below fails; it is not ownership proof.
            accepted_order_id = Some(observed.order_id);
            match payload.execution_type {
                2 => {
                    anyhow::ensure!(
                        observed.order_status == Some(1)
                            && observed.executed_volume.is_none_or(|volume| volume == 0)
                            && payload.deal.is_none(),
                        "acceptance contains unexpected execution progress; outcome unresolved"
                    );
                    accepted_position_id = observed.position_id.or_else(|| {
                        payload
                            .position
                            .as_ref()
                            .map(|position| position.position_id)
                    });
                }
                3 => {
                    anyhow::ensure!(
                        observed.order_status == Some(2),
                        "filled execution does not contain a filled order"
                    );
                    // Preserve the known order on failures in the same required
                    // parser/validator used by the outer backend. The successful
                    // response still follows that unchanged outer path.
                    let outcome = parse_execution_outcome(&text)?;
                    validate_execution_outcome(request, &outcome)?;
                    // NETTED/multi-fill facts cannot mint single-opening proof.
                    return Ok(text);
                }
                7 => return Ok(text),
                other => anyhow::bail!(
                    "immediate execution requires reconciliation: execution_type={other}, order_id={}, executed_volume={:?}, deal_id={:?}; no resend",
                    observed.order_id,
                    observed.executed_volume,
                    payload.deal.as_ref().map(|deal| deal.deal_id)
                ),
            }
        }
    })();
    collected
        .and_then(|text| {
            budget.check_io(CTraderIoPhase::ResponseRead)?;
            Ok(text)
        })
        .with_context(|| CTraderUnresolvedExecution {
            environment: request.environment,
            account_id: order.account_id,
            client_order_id: order.client_order_id.clone(),
            accepted_order_id,
        })
}

impl ProductionCTraderExecutionBackend {
    fn session() -> &'static Mutex<CTraderExecutionSession> {
        EXECUTION_SESSION.get_or_init(|| Mutex::new(CTraderExecutionSession::default()))
    }

    fn auth_key(request: &CTraderExecutionRuntimeRequest) -> String {
        format!(
            "{}|{}|{}|{}",
            request.environment.endpoint_host(),
            request.client_id,
            request.account_id,
            request.access_token
        )
    }
    /// Derive a stable wire correlation id from the logical request identity.
    ///
    /// ProtoMessage documents clientMsgId as an echoed request identifier, not
    /// a broker idempotency guarantee. Keep it stable for traceability, but do
    /// not resend an execution after a send/read failure. Authentication-only
    /// attempts may still retry before any execution submission begins.
    fn client_msg_id_for(phase: &str, fingerprint: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        phase.hash(&mut hasher);
        fingerprint.hash(&mut hasher);
        format!("{phase}-{:016x}", hasher.finish())
    }

    fn maybe_cached_outcome(
        session: &CTraderExecutionSession,
        fingerprint: &str,
    ) -> Option<CTraderExecutionOutcome> {
        let ttl = Duration::from_secs(30);
        session
            .recent_submissions
            .get(fingerprint)
            .and_then(|cached| {
                if cached.created_at.elapsed() <= ttl {
                    Some(cached.outcome.clone())
                } else {
                    None
                }
            })
    }

    fn store_cached_outcome(
        session: &mut CTraderExecutionSession,
        fingerprint: String,
        outcome: CTraderExecutionOutcome,
    ) {
        // Store validated outcomes for the same bounded logical request.
        // Errors (including parsed rejections) never reach this cache. A failed
        // submission is not automatically resent; callers retain unresolved
        // operation state instead of assuming that an error means no fill.
        session.recent_submissions.insert(
            fingerprint,
            CachedExecutionOutcome {
                created_at: Instant::now(),
                outcome,
            },
        );
        if session.recent_submissions.len() > 256 {
            let mut entries = session
                .recent_submissions
                .iter()
                .map(|(key, value)| (key.clone(), value.created_at))
                .collect::<Vec<_>>();
            entries.sort_by_key(|(_, created_at)| *created_at);
            for (key, _) in entries
                .into_iter()
                .take(session.recent_submissions.len() - 256)
            {
                session.recent_submissions.remove(&key);
            }
        }
    }

    fn read_matching_response(
        socket: &mut ProductionCTraderSocket,
        request: &CTraderOpenApiJsonMessage,
        expected_payload_type: u32,
        budget: &ProductionCTraderBudget,
    ) -> Result<String> {
        loop {
            budget.check_io(CTraderIoPhase::ResponseRead)?;
            let frame = match socket.read() {
                Ok(frame) => frame,
                Err(error) if is_ctrader_socket_poll_timeout(&error) => continue,
                Err(error) => {
                    return Err(anyhow!(error).context("failed to read cTrader open api response"));
                }
            };
            budget.check_io(CTraderIoPhase::ResponseRead)?;
            match frame {
                Message::Text(text) => {
                    // Unrelated protocol traffic cannot renew the absolute
                    // deadline enforced both here and below TLS by DeadlineIo.
                    if text.trim().is_empty() {
                        continue;
                    }
                    let Some(envelope) = parse_envelope_or_skip(text.as_ref()) else {
                        continue;
                    };
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        return Ok(text.to_string());
                    }
                    if is_matching_open_api_response(&envelope, request, expected_payload_type) {
                        return Ok(text.to_string());
                    }
                }
                Message::Binary(bytes) => {
                    let Ok(text) = String::from_utf8(bytes.to_vec()) else {
                        tracing::warn!(
                            target: "neoethos_app::ctrader",
                            "skipping non-UTF8 cTrader binary frame while awaiting response"
                        );
                        continue;
                    };
                    if text.trim().is_empty() {
                        continue;
                    }
                    let Some(envelope) = parse_envelope_or_skip(&text) else {
                        continue;
                    };
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        return Ok(text);
                    }
                    if is_matching_open_api_response(&envelope, request, expected_payload_type) {
                        return Ok(text);
                    }
                }
                Message::Ping(payload) => {
                    socket
                        .send(Message::Pong(payload))
                        .context("failed to reply to cTrader ping")?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => {
                    return Err(anyhow!("cTrader open api socket closed unexpectedly"));
                }
                Message::Frame(_) => {}
            }
        }
    }

    fn send_message_and_wait(
        socket: &mut ProductionCTraderSocket,
        message: &CTraderOpenApiJsonMessage,
    ) -> Result<String> {
        let expected_payload_type = expected_response_payload_type(message.payload_type)?;
        let serialized = serde_json::to_string(message)
            .context("failed to serialize cTrader open api message")?;
        let budget = arm_ctrader_socket_budget(
            socket,
            CTRADER_RESPONSE_TIMEOUT,
            None,
            CTraderIoPhase::ResponseRead,
        )?;
        socket
            .send(Message::Text(serialized.into()))
            .context("failed to send cTrader open api message")?;
        Self::read_matching_response(socket, message, expected_payload_type, &budget)
    }

    fn ensure_authenticated(
        session: &mut CTraderExecutionSession,
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<()> {
        let auth_key = Self::auth_key(request);
        if session.socket.is_some() && session.auth_key.as_deref() == Some(auth_key.as_str()) {
            return Ok(());
        }

        session.socket = None;
        let socket = connect_ctrader_socket(request.environment.endpoint_host(), None)?;
        session.socket = Some(socket);
        session.auth_key = Some(auth_key);

        let fingerprint = request.request.idempotency_fingerprint();
        let app_auth = build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            Self::client_msg_id_for("app-auth", &fingerprint),
        );
        let account_auth = build_account_auth_request(
            request
                .account_id
                .parse::<i64>()
                .context("cTrader execution account id must be numeric")?,
            &request.access_token,
            Self::client_msg_id_for("account-auth", &fingerprint),
        );

        let socket = session
            .socket
            .as_mut()
            .context("cTrader execution socket missing after connect")?;
        let response = Self::send_message_and_wait(socket, &app_auth)?;
        // D11: surface auth-token failures with a sentinel-prefixed error so
        // the trading-session caller can force-refresh the OAuth bundle and
        // retry. Previously a stale access_token would loop here forever
        // because `ensure_authenticated` reused the same token on every
        // retry. The application-auth response can also fail for other
        // reasons; only the token-expired codes trigger refresh.
        Self::ensure_auth_payload(&response, CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE)?;
        let response = Self::send_message_and_wait(socket, &account_auth)?;
        Self::ensure_auth_payload(&response, CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
        Ok(())
    }

    fn ensure_auth_payload(response: &str, expected_payload_type: u32) -> Result<()> {
        let envelope =
            parse_open_api_envelope(response).context("failed to inspect cTrader auth response")?;
        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
            let (code, message) = parse_ctrader_error_payload_parts(&envelope.payload)?;
            if is_ctrader_auth_token_error(&code) {
                return Err(anyhow!("{CTRADER_TOKEN_EXPIRED_SENTINEL}: {message}"));
            }
            return Err(anyhow!(message));
        }
        if envelope.payload_type != expected_payload_type {
            return Err(anyhow!(
                "expected cTrader payload type {expected_payload_type}, received {}",
                envelope.payload_type
            ));
        }
        Ok(())
    }

    fn execution_cache_fingerprint(request: &CTraderExecutionRuntimeRequest) -> String {
        // A broker account/position id is scoped to its server environment.
        // Logical NewOrder intent identity is the stable clientOrderId included
        // by idempotency_fingerprint; equal parameters alone are not a new id.
        format!(
            "{}|runtime_account={}|{}",
            request.environment.endpoint_host(),
            request.account_id,
            request.request.idempotency_fingerprint()
        )
    }

    /// Submit exactly once after authentication. A callback may fail after the
    /// broker received the request; neither an error nor an echoed clientMsgId
    /// proves the absence of execution. The caller must retain unresolved state.
    fn execute_authenticated_once(
        request: &CTraderExecutionRuntimeRequest,
        send: impl FnOnce() -> Result<String>,
    ) -> Result<CTraderExecutionOutcome> {
        let response = send().context(
            "cTrader execution send/response failed after submission began; outcome unresolved; automatic resend refused",
        )?;
        let response_envelope = parse_open_api_envelope(&response)
            .context("failed to inspect cTrader execution response")?;
        if response_envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
            let (error_code, error_message) =
                parse_ctrader_error_payload_parts(&response_envelope.payload)?;
            // Preserve the existing token diagnostic. It does not authorize
            // resubmitting an operation whose execution outcome is unresolved.
            if is_ctrader_auth_token_error(&error_code) {
                return Err(anyhow!("{CTRADER_TOKEN_EXPIRED_SENTINEL}: {error_message}"));
            }
            return Err(anyhow!(error_message));
        }
        let outcome = parse_execution_outcome(&response)?;
        validate_execution_outcome(request, &outcome)?;
        Ok(outcome)
    }

    fn execute_via_session(
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome> {
        let mut session = Self::session()
            .lock()
            .map_err(|_| anyhow!("cTrader execution session lock poisoned"))?;
        let fingerprint = Self::execution_cache_fingerprint(request);
        if let Some(cached) = Self::maybe_cached_outcome(&session, &fingerprint) {
            return Ok(cached);
        }

        let max_attempts = ctrader_max_attempts();
        let mut last_error = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                ctrader_backoff_sleep(attempt);
            }
            if let Err(err) = Self::ensure_authenticated(&mut session, request) {
                session.socket = None;
                session.auth_key = None;
                last_error = Some(err);
                continue;
            }

            // Only authentication retries above. Once submission begins, both
            // a transport error and an unusable response are potentially filled.
            let order_message = request
                .request
                .to_message(&Self::client_msg_id_for("execute", &fingerprint));
            let socket = session
                .socket
                .as_mut()
                .context("cTrader execution socket missing after auth")?;
            match Self::execute_authenticated_once(request, || {
                if immediate_order(request).is_some() {
                    let budget = CTraderOperationBudget::new(
                        SystemCTraderMonotonicClock,
                        immediate_execution_timeout(
                            crate::app_services::env_overrides::ctrader_read_timeout_secs(),
                        ),
                        None,
                    )?;
                    arm_ctrader_socket_with_budget(
                        socket,
                        budget.clone(),
                        CTraderIoPhase::ResponseRead,
                    )?;
                    collect_immediate_execution(request, &order_message, &budget, socket)
                } else {
                    Self::send_message_and_wait(socket, &order_message)
                }
            }) {
                Ok(outcome) => {
                    if !cacheable_execution_outcome(request, &outcome) {
                        session.socket = None;
                        session.auth_key = None;
                        anyhow::bail!(
                            "immediate execution is not terminal; outcome unresolved; no resend"
                        );
                    }
                    Self::store_cached_outcome(&mut session, fingerprint.clone(), outcome.clone());
                    return Ok(outcome);
                }
                Err(err) => {
                    session.socket = None;
                    session.auth_key = None;
                    return Err(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("cTrader execution failed")))
    }

    #[cfg(test)]
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
            build_application_auth_request(
                &request.client_id,
                &request.client_secret,
                "app-auth-1",
            ),
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
            CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
        )?;
        ensure_payload_type(&responses[1], CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE)?;
        let outcome = parse_execution_outcome(&responses[2])?;
        validate_execution_outcome(request, &outcome)?;
        Ok(outcome)
    }
}

/// Parse a cTrader JSON envelope, returning `None` (with a warning) instead of
/// an error when the frame is malformed. Used by the response read loop so a
/// single bad/out-of-band frame is skipped rather than aborting the request and
/// forcing a session reset + order retry. **2026-06-10 defensive-parse fix.**
fn parse_envelope_or_skip(frame: &str) -> Option<CTraderOpenApiJsonMessage> {
    match parse_open_api_envelope(frame) {
        Ok(envelope) => Some(envelope),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::ctrader",
                error = %err,
                "skipping unparseable cTrader frame while awaiting a response"
            );
            None
        }
    }
}

impl CTraderExecutionBackend for ProductionCTraderExecutionBackend {
    fn execute(&self, request: &CTraderExecutionRuntimeRequest) -> Result<CTraderExecutionOutcome> {
        let outcome = Self::execute_via_session(request)?;
        let entry = crate::app_services::live_journal::LiveTradeJournalEntry::from_outcome(
            request_action_label(&request.request),
            request,
            &outcome,
        );
        crate::app_services::live_journal::record_live_outcome_best_effort(&entry);
        // **2026-05-25 — operator directive "uniform push everywhere"
        // + F-231 closure**: a successful order placement / cancel /
        // close has just changed the account state (margin, free
        // margin, position list, possibly equity if it was a close).
        // Fire the global refresh trigger so the bridge runs an
        // immediate `refresh_once` and pushes the new snapshot to
        // every SSE subscriber within ~750 ms — instead of the
        // operator waiting up to 5 s for the bridge safety timer.
        //
        // This is the synchronous-request path. The future
        // spontaneous-event listener (margin call from the broker,
        // SL/TP hit without our request) will call the same trigger.
        crate::server::state::trigger_global_account_refresh();
        Ok(outcome)
    }
}

fn request_action_label(request: &CTraderExecutionRequest) -> &'static str {
    match request {
        CTraderExecutionRequest::NewOrder(_) => "new_order",
        CTraderExecutionRequest::CancelOrder(_) => "cancel_order",
        CTraderExecutionRequest::ClosePosition(_) => "close_position",
        CTraderExecutionRequest::AmendPositionSltp(_) => "amend_position_sltp",
        CTraderExecutionRequest::AmendOrder(_) => "amend_order",
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

/// Maximum authentication attempts before an execution is submitted.
/// Tunable via NEOETHOS_BOT_CTRADER_MAX_ATTEMPTS (clamped to [1, 5];
/// default 3). No execution send/read/parse failure is automatically retried.
///
/// Thin shim over the canonical env_overrides getter; unchanged configuration
/// controls only the definitely-pre-submission authentication retry loop.
fn ctrader_max_attempts() -> u32 {
    crate::app_services::env_overrides::ctrader_max_attempts()
}

/// Base backoff in ms for retries; tunable via
/// `NEOETHOS_BOT_CTRADER_BACKOFF_BASE_MS` (clamped to `[10, 2000]`; default 200).
///
/// **F-CORE3 closure (2026-05-25)**: thin shim over the canonical
/// `env_overrides::ctrader_backoff_base_ms` typed getter.
fn ctrader_backoff_base_ms() -> u64 {
    crate::app_services::env_overrides::ctrader_backoff_base_ms()
}

/// Sleep before the n-th retry attempt (n >= 1).
/// Delay = `base * 2^(n-1)` plus 0-99ms jitter derived from the wall clock,
/// capped at 5 seconds total. The jitter spreads simultaneous retries from
/// concurrent workers so they do not collide on the broker.
fn ctrader_backoff_sleep(attempt: u32) {
    crate::app_services::backoff::backoff_sleep(attempt, ctrader_backoff_base_ms());
}

#[cfg(test)]
fn ensure_payload_type(response_json: &str, expected_payload_type: u32) -> Result<()> {
    let envelope: Value =
        serde_json::from_str(response_json).context("failed to parse cTrader JSON envelope")?;
    let payload_type = envelope
        .get("payloadType")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing payloadType in cTrader envelope"))?
        as u32;
    if payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader execution transport returned error payload"
        ));
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
        .ok_or_else(|| anyhow!("missing payloadType in cTrader envelope"))?
        as u32;
    match payload_type {
        CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE => parse_execution_event(response_json),
        CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE => parse_order_error_event(response_json),
        other => Err(anyhow!(
            "unexpected cTrader execution response payload type: {other}"
        )),
    }
}

fn validate_execution_payload_links(payload: &ExecutionPayload) -> Result<()> {
    anyhow::ensure!(
        payload.ctid_trader_account_id > 0,
        "execution account id must be positive"
    );
    if let Some(order) = &payload.order {
        anyhow::ensure!(
            order.order_id > 0
                && order.trade_data.symbol_id > 0
                && matches!(order.trade_data.trade_side, 1 | 2)
                && order.position_id.is_none_or(|id| id > 0),
            "execution order identity is invalid"
        );
    }
    if let Some(position) = &payload.position {
        anyhow::ensure!(
            position.position_id > 0
                && position.trade_data.symbol_id > 0
                && matches!(position.trade_data.trade_side, 1 | 2),
            "execution position identity is invalid"
        );
    }
    if let Some(deal) = &payload.deal {
        anyhow::ensure!(
            deal.deal_id > 0
                && deal.order_id > 0
                && deal.position_id > 0
                && deal.symbol_id > 0
                && matches!(deal.trade_side, 1 | 2),
            "execution deal identity is invalid"
        );
        if let Some(order) = &payload.order {
            anyhow::ensure!(
                order.order_id == deal.order_id
                    && order.trade_data.symbol_id == deal.symbol_id
                    && order.trade_data.trade_side == deal.trade_side
                    && order.position_id.is_none_or(|id| id == deal.position_id),
                "execution order/deal identity mismatch"
            );
        }
        if let Some(position) = &payload.position {
            anyhow::ensure!(
                position.position_id == deal.position_id
                    && position.trade_data.symbol_id == deal.symbol_id,
                "execution position/deal identity mismatch"
            );
            // A close/reduction has the opposite side from the original
            // position. Do not impose an opening-side rule on closing deals.
            if deal.close_position_detail.is_none()
                && payload
                    .order
                    .as_ref()
                    .is_some_and(|order| order.closing_order == Some(false))
            {
                anyhow::ensure!(
                    position.trade_data.trade_side == deal.trade_side,
                    "opening execution position/deal side mismatch"
                );
            }
        }
    }
    if let (Some(order), Some(position)) = (&payload.order, &payload.position) {
        anyhow::ensure!(
            order.trade_data.symbol_id == position.trade_data.symbol_id
                && order
                    .position_id
                    .is_none_or(|id| id == position.position_id),
            "execution order/position identity mismatch"
        );
    }
    Ok(())
}

fn opening_fill_evidence(
    payload: &ExecutionPayload,
    status: CTraderExecutionStatus,
) -> Option<CTraderOpeningFillEvidenceV1> {
    let order = payload.order.as_ref()?;
    let position = payload.position.as_ref()?;
    let deal = payload.deal.as_ref()?;
    // This is intentionally a single, complete opening fill. A larger existing
    // NETTED position, a multi-fill order, or a reduction requires recovery of
    // the real lifecycle; none can be reconstructed from the requested lots.
    if status != CTraderExecutionStatus::Filled
        || order.order_status != Some(2)
        || position.position_status != Some(1)
        || deal.deal_status != Some(2)
        || order.closing_order == Some(true)
        || deal.close_position_detail.is_some()
        || position.trade_data.trade_side != deal.trade_side
        || deal.filled_volume <= 0
        || deal.filled_volume > crate::app_services::broker_deal_economics::MAX_EXACT_BROKER_VOLUME
        || order.executed_volume != Some(deal.filled_volume)
        || order.trade_data.volume != deal.filled_volume
        || position.trade_data.volume != deal.filled_volume
    {
        return None;
    }
    let position_open_timestamp_ms = position.trade_data.open_timestamp?;
    let entry_price = position.price?;
    let execution_price = deal.execution_price?;
    if position_open_timestamp_ms <= 0
        || deal.execution_timestamp < position_open_timestamp_ms
        || !entry_price.is_finite()
        || entry_price <= 0.0
        || !execution_price.is_finite()
        || execution_price <= 0.0
        || entry_price.to_bits() != execution_price.to_bits()
    {
        return None;
    }
    Some(CTraderOpeningFillEvidenceV1 {
        account_id: payload.ctid_trader_account_id,
        order_id: order.order_id,
        position_id: position.position_id,
        deal_id: deal.deal_id,
        symbol_id: deal.symbol_id,
        trade_side: trade_side_label(deal.trade_side),
        filled_volume_raw_centi_units: deal.filled_volume,
        position_open_timestamp_ms,
        execution_timestamp_ms: deal.execution_timestamp,
        entry_price,
    })
}

fn parse_execution_event(response_json: &str) -> Result<CTraderExecutionOutcome> {
    let envelope: ExecutionEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader execution event")?;
    if envelope.payload_type != CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader execution event payload type: {}",
            envelope.payload_type
        ));
    }

    let status = CTraderExecutionStatus::from_proto(envelope.payload.execution_type)?;
    validate_execution_payload_links(&envelope.payload)?;
    let opening_fill_evidence = opening_fill_evidence(&envelope.payload, status);
    let deal_closes_position = envelope
        .payload
        .deal
        .as_ref()
        .map(|deal| deal.close_position_detail.is_some());
    let order = envelope.payload.order;
    let position = envelope.payload.position;
    let deal = envelope.payload.deal;
    let (gross_profit, fee, swap, net_profit) = match deal.as_ref() {
        Some(item) => match item.close_position_detail.as_ref() {
            Some(detail) => {
                let money_digits = crate::app_services::ctrader_money::required_money_digits(
                    detail.money_digits,
                    "execution.close_position_detail.money_digits",
                )?;
                let gross = scaled_money(detail.gross_profit, money_digits)?;
                let fee = scaled_money(detail.commission, money_digits)?;
                let swap = scaled_money(detail.swap, money_digits)?;
                // ProtoOAClosePositionDetail.pnlConversionFee is optional and
                // is present only when the broker applied quote/deposit
                // conversion. Absence is therefore a broker-defined zero, not
                // a locally reconstructed fee.
                let conversion_fee = detail
                    .pnl_conversion_fee
                    .map(|raw| scaled_money(raw, money_digits))
                    .transpose()?
                    .unwrap_or_default();
                (
                    Some(gross),
                    Some(fee),
                    Some(swap),
                    Some(gross + fee + swap + conversion_fee),
                )
            }
            None => {
                let fee = match item.commission {
                    Some(raw) => {
                        let money_digits =
                            crate::app_services::ctrader_money::required_money_digits(
                                item.money_digits,
                                "execution.deal.money_digits",
                            )?;
                        Some(scaled_money(raw, money_digits)?)
                    }
                    None => None,
                };
                (None, fee, None, None)
            }
        },
        None => (None, None, None, None),
    };

    Ok(CTraderExecutionOutcome {
        status,
        account_id: envelope.payload.ctid_trader_account_id,
        symbol_id: order
            .as_ref()
            .map(|item| item.trade_data.symbol_id)
            .or_else(|| position.as_ref().map(|item| item.trade_data.symbol_id))
            .or_else(|| deal.as_ref().map(|item| item.symbol_id)),
        order_id: order
            .as_ref()
            .map(|item| item.order_id)
            .or_else(|| deal.as_ref().map(|item| item.order_id)),
        position_id: position
            .as_ref()
            .map(|item| item.position_id)
            .or_else(|| deal.as_ref().map(|item| item.position_id)),
        deal_id: deal.as_ref().map(|item| item.deal_id),
        trade_side: order
            .as_ref()
            .map(|item| trade_side_label(item.trade_data.trade_side))
            .or_else(|| {
                position
                    .as_ref()
                    .map(|item| trade_side_label(item.trade_data.trade_side))
            })
            .or_else(|| deal.as_ref().map(|item| trade_side_label(item.trade_side))),
        order_type: order.as_ref().map(|item| order_type_label(item.order_type)),
        lot_size: order
            .as_ref()
            .map(|item| volume_to_units(item.trade_data.volume))
            .or_else(|| {
                position
                    .as_ref()
                    .map(|item| volume_to_units(item.trade_data.volume))
            })
            .or_else(|| {
                deal.as_ref()
                    .map(|item| volume_to_units(item.filled_volume))
            }),
        requested_lot_size: order
            .as_ref()
            .map(|item| volume_to_units(item.trade_data.volume))
            .or_else(|| {
                position
                    .as_ref()
                    .map(|item| volume_to_units(item.trade_data.volume))
            }),
        filled_lot_size: deal
            .as_ref()
            .map(|item| volume_to_units(item.filled_volume)),
        filled_volume_raw_centi_units: deal.as_ref().map(|item| item.filled_volume),
        volume_scale_evidence: None,
        deal_closes_position,
        opening_fill_evidence,
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
            .or_else(|| {
                order
                    .as_ref()
                    .and_then(|item| item.trade_data.open_timestamp)
            })
            .or_else(|| {
                position
                    .as_ref()
                    .and_then(|item| item.trade_data.open_timestamp)
            }),
        error_code: envelope.payload.error_code,
        description: None,
    })
}

fn parse_order_error_event(response_json: &str) -> Result<CTraderExecutionOutcome> {
    let envelope: OrderErrorEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader order error event")?;
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
        requested_lot_size: None,
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
        error_code: Some(envelope.payload.error_code),
        description: envelope.payload.description,
    })
}

fn scaled_money(raw: i64, money_digits: u32) -> Result<f64> {
    crate::app_services::ctrader_money::scale_ctrader_money_int(raw, money_digits as i32)
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

fn validate_execution_outcome(
    request: &CTraderExecutionRuntimeRequest,
    outcome: &CTraderExecutionOutcome,
) -> Result<()> {
    let requested_account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader execution account id must be numeric")?;
    if outcome.account_id != requested_account_id {
        anyhow::bail!(
            "cTrader execution response account mismatch: expected {}, got {}",
            requested_account_id,
            outcome.account_id
        );
    }

    // D10: surface broker-side rejections / partial fills explicitly. The
    // previous implementation only checked IDs, so a PartialFill or Failed
    // status was silently treated as success — caller could not see that
    // `filled_volume < requested_volume`. We bail here on Failed and flag
    // PartialFill as an error so the trading loop can decide between retry
    // for the residual or cancel-and-log. Set
    // `NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL=1` to opt back into the previous
    // permissive behaviour (e.g. for replay tests).
    if matches!(outcome.status, CTraderExecutionStatus::Failed) {
        anyhow::bail!(
            "cTrader execution rejected: status=Failed code={:?} description={:?}",
            outcome.error_code,
            outcome.description
        );
    }
    if matches!(outcome.status, CTraderExecutionStatus::PartialFill) {
        // F-CORE3 closure (2026-05-25): canonical getter.
        let allow_partial = crate::app_services::env_overrides::ctrader_allow_partial_fill();
        if !allow_partial {
            anyhow::bail!(
                "cTrader execution returned PartialFill (deal_id={:?}, requested={:?}, filled={:?}); \
                 set NEOETHOS_BOT_CTRADER_ALLOW_PARTIAL_FILL=1 to accept partial fills",
                outcome.deal_id,
                outcome.requested_lot_size,
                outcome.filled_lot_size
            );
        }
        tracing::warn!(
            target: "neoethos_app::ctrader",
            deal_id = ?outcome.deal_id,
            requested_lot_size = ?outcome.requested_lot_size,
            filled_lot_size = ?outcome.filled_lot_size,
            "cTrader execution accepted PartialFill; trading loop should handle residual"
        );
    }

    // A transport-level `Ok` is not enough: the execution type must describe
    // the operation we sent. In particular, cTrader documents a successful
    // `ProtoOAAmendPositionSLTPReq` as `ORDER_REPLACED`. Treating an unrelated
    // `ORDER_ACCEPTED`, `ORDER_FILLED`, or `ORDER_CANCELLED` event as a
    // confirmed protection amend lets the live loop advance its local stop
    // while the broker still holds the previous one.
    let status_matches_request = match &request.request {
        CTraderExecutionRequest::NewOrder(_) => matches!(
            outcome.status,
            CTraderExecutionStatus::Accepted
                | CTraderExecutionStatus::Filled
                | CTraderExecutionStatus::PartialFill
        ),
        CTraderExecutionRequest::CancelOrder(_) => {
            matches!(outcome.status, CTraderExecutionStatus::Cancelled)
        }
        CTraderExecutionRequest::ClosePosition(_) => matches!(
            outcome.status,
            // Cancellation confirms neither an executed close nor a flat
            // position. Explicitly allowed partial fills remain outcomes;
            // the caller must reconcile the broker's remaining position.
            CTraderExecutionStatus::Filled | CTraderExecutionStatus::PartialFill
        ),
        CTraderExecutionRequest::AmendPositionSltp(_) | CTraderExecutionRequest::AmendOrder(_) => {
            matches!(outcome.status, CTraderExecutionStatus::Replaced)
        }
    };
    if !status_matches_request {
        anyhow::bail!(
            "cTrader execution status {:?} does not confirm requested operation {}",
            outcome.status,
            request_action_label(&request.request)
        );
    }

    match &request.request {
        CTraderExecutionRequest::NewOrder(inner) => {
            if outcome.trade_side.as_deref() != Some(inner.trade_side.label()) {
                anyhow::bail!("cTrader new-order response trade side differs from the request");
            }
            if outcome.symbol_id != Some(inner.symbol_id) {
                anyhow::bail!(
                    "cTrader new-order response symbol mismatch: expected {}, got {:?}",
                    inner.symbol_id,
                    outcome.symbol_id
                );
            }
            if outcome.order_id.is_none()
                && outcome.position_id.is_none()
                && outcome.deal_id.is_none()
            {
                anyhow::bail!(
                    "cTrader new-order response did not include an order, position, or deal id"
                );
            }
        }
        CTraderExecutionRequest::CancelOrder(inner) => {
            if outcome.order_id != Some(inner.order_id) {
                anyhow::bail!(
                    "cTrader cancel-order response order mismatch: expected {}, got {:?}",
                    inner.order_id,
                    outcome.order_id
                );
            }
        }
        CTraderExecutionRequest::ClosePosition(inner) => {
            if outcome.position_id != Some(inner.position_id) {
                anyhow::bail!(
                    "cTrader close-position response position mismatch: expected {}, got {:?}",
                    inner.position_id,
                    outcome.position_id
                );
            }
        }
        CTraderExecutionRequest::AmendPositionSltp(inner) => {
            if outcome.position_id != Some(inner.position_id) {
                anyhow::bail!(
                    "cTrader amend-position-SLTP response position mismatch: expected {}, got {:?}",
                    inner.position_id,
                    outcome.position_id
                );
            }
        }
        CTraderExecutionRequest::AmendOrder(inner) => {
            // Same check the cancel arm makes, and for the same reason: the
            // broker must answer about the order we named. A reply about a
            // different order id is not a success we can report.
            if outcome.order_id != Some(inner.order_id) {
                anyhow::bail!(
                    "cTrader amend-order response order mismatch: expected {}, got {:?}",
                    inner.order_id,
                    outcome.order_id
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "ctrader_execution_tests.rs"]
mod tests;
