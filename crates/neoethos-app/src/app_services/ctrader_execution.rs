use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE, CTRADER_OA_ORDER_ERROR_EVENT_PAYLOAD_TYPE,
    CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderCancelOrderRequest, CTraderNewOrderRequest,
    CTraderOpenApiJsonMessage, CTraderOpenApiTransport, build_account_auth_request,
    build_application_auth_request, build_cancel_order_request, build_close_position_request,
    build_new_order_request, expected_response_payload_type, is_ctrader_auth_token_error,
    is_matching_open_api_response, parse_ctrader_error_payload_parts, parse_open_api_envelope,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::net::TcpStream;
#[cfg(test)]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, connect};

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
    fn execute(&self, request: &CTraderExecutionRuntimeRequest) -> Result<CTraderExecutionOutcome>;
}

#[derive(Clone, Default)]
pub struct ProductionCTraderExecutionBackend;

#[derive(Debug, Default)]
struct CTraderExecutionSession {
    socket: Option<tungstenite::WebSocket<MaybeTlsStream<TcpStream>>>,
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

    fn client_msg_id_for(phase: &str, fingerprint: &str, attempt: u32) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        phase.hash(&mut hasher);
        fingerprint.hash(&mut hasher);
        attempt.hash(&mut hasher);
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
        if outcome.status == CTraderExecutionStatus::Failed {
            return;
        }
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
        socket: &mut tungstenite::WebSocket<MaybeTlsStream<TcpStream>>,
        request: &CTraderOpenApiJsonMessage,
        expected_payload_type: u32,
    ) -> Result<String> {
        loop {
            match socket
                .read()
                .context("failed to read cTrader open api response")?
            {
                Message::Text(text) => {
                    if text.trim().is_empty() {
                        return Err(anyhow!("empty cTrader open api response"));
                    }
                    let envelope = parse_open_api_envelope(text.as_ref())?;
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        return Ok(text.to_string());
                    }
                    if is_matching_open_api_response(&envelope, request, expected_payload_type) {
                        return Ok(text.to_string());
                    }
                }
                Message::Binary(bytes) => {
                    let text = String::from_utf8(bytes.to_vec())
                        .context("failed to decode cTrader binary response")?;
                    if text.trim().is_empty() {
                        return Err(anyhow!("empty cTrader open api response"));
                    }
                    let envelope = parse_open_api_envelope(&text)?;
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
        socket: &mut tungstenite::WebSocket<MaybeTlsStream<TcpStream>>,
        message: &CTraderOpenApiJsonMessage,
    ) -> Result<String> {
        let expected_payload_type = expected_response_payload_type(message.payload_type)?;
        let serialized = serde_json::to_string(message)
            .context("failed to serialize cTrader open api message")?;
        socket
            .send(Message::Text(serialized.into()))
            .context("failed to send cTrader open api message")?;
        Self::read_matching_response(socket, message, expected_payload_type)
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
        let url = format!("wss://{}:5036", request.environment.endpoint_host());
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let (socket, _) = connect(url.as_str())
            .with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        // M10: cap the underlying TCP read at 30s so a broker stall or a
        // mismatched payload cannot wedge the trading loop forever. The loop
        // in `read_matching_response` previously blocked indefinitely; with a
        // timeout the I/O error bubbles up, the caller drops the session,
        // and the next `execute_via_session` retry re-authenticates.
        // Override via `FOREX_BOT_CTRADER_READ_TIMEOUT_SECS` (0 disables).
        // F-CORE3 closure (2026-05-25): routed through the canonical
        // `env_overrides::ctrader_read_timeout_secs` getter so the var
        // is grep-able from one place + clamped consistently.
        let read_timeout_secs: u64 = crate::app_services::env_overrides::ctrader_read_timeout_secs();
        if read_timeout_secs > 0 {
            let timeout = std::time::Duration::from_secs(read_timeout_secs);
            let apply_result = match socket.get_ref() {
                MaybeTlsStream::Plain(stream) => stream.set_read_timeout(Some(timeout)),
                MaybeTlsStream::Rustls(stream) => stream.get_ref().set_read_timeout(Some(timeout)),
                _ => Ok(()), // unknown TLS variant — not critical
            };
            if let Err(err) = apply_result {
                tracing::warn!(
                    target: "neoethos_app::ctrader",
                    error = ?err,
                    "failed to apply cTrader socket read timeout"
                );
            }
        }
        session.socket = Some(socket);
        session.auth_key = Some(auth_key);

        let fingerprint = request.request.idempotency_fingerprint();
        let app_auth = build_application_auth_request(
            &request.client_id,
            &request.client_secret,
            Self::client_msg_id_for("app-auth", &fingerprint, 0),
        );
        let account_auth = build_account_auth_request(
            request
                .account_id
                .parse::<i64>()
                .context("cTrader execution account id must be numeric")?,
            &request.access_token,
            Self::client_msg_id_for("account-auth", &fingerprint, 0),
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

    fn execute_via_session(
        request: &CTraderExecutionRuntimeRequest,
    ) -> Result<CTraderExecutionOutcome> {
        let mut session = Self::session()
            .lock()
            .map_err(|_| anyhow!("cTrader execution session lock poisoned"))?;
        let fingerprint = request.request.idempotency_fingerprint();
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

            let order_message = request.request.to_message(&Self::client_msg_id_for(
                "execute",
                &fingerprint,
                attempt,
            ));
            let socket = session
                .socket
                .as_mut()
                .context("cTrader execution socket missing after auth")?;
            match Self::send_message_and_wait(socket, &order_message) {
                Ok(response) => {
                    let response_envelope = parse_open_api_envelope(&response)
                        .context("failed to inspect cTrader execution response")?;
                    if response_envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        let (error_code, error_message) =
                            parse_ctrader_error_payload_parts(&response_envelope.payload)?;
                        session.socket = None;
                        session.auth_key = None;
                        // D11: tag token-failure errors with the sentinel so
                        // the caller knows to force-refresh the OAuth bundle
                        // before retrying. Other errors (insufficient margin,
                        // invalid stop, etc.) bubble up unchanged.
                        if is_ctrader_auth_token_error(&error_code) {
                            return Err(anyhow!(
                                "{CTRADER_TOKEN_EXPIRED_SENTINEL}: {error_message}"
                            ));
                        }
                        return Err(anyhow!(error_message));
                    }
                    let outcome = parse_execution_outcome(&response)?;
                    validate_execution_outcome(request, &outcome)?;
                    Self::store_cached_outcome(&mut session, fingerprint.clone(), outcome.clone());
                    return Ok(outcome);
                }
                Err(err) => {
                    session.socket = None;
                    session.auth_key = None;
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("cTrader execution failed")))
    }

    #[allow(dead_code)]
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
        parse_execution_outcome(&responses[2])
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

/// Maximum number of attempts (initial + retries) for a single
/// `execute_via_session` call. Tunable via `FOREX_BOT_CTRADER_MAX_ATTEMPTS`
/// (clamped to `[1, 5]`; default 3). The default is deliberately small —
/// retry safety relies on the broker deduping by `clientOrderId`.
///
/// **F-CORE3 closure (2026-05-25)**: thin shim over the canonical
/// `env_overrides::ctrader_max_attempts` typed getter. Kept as a
/// local function so existing call-sites don't need to re-import.
fn ctrader_max_attempts() -> u32 {
    crate::app_services::env_overrides::ctrader_max_attempts()
}

/// Base backoff in ms for retries; tunable via
/// `FOREX_BOT_CTRADER_BACKOFF_BASE_MS` (clamped to `[10, 2000]`; default 200).
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
    let order = envelope.payload.order;
    let position = envelope.payload.position;
    let deal = envelope.payload.deal;
    // Same defence as ctrader_account::required_money_digits — a missing
    // money_digits field would silently scale every monetary value 100×
    // because `10^0 = 1`. Default to 2 (typical fiat precision) and
    // log loudly so the operator notices the protocol regression.
    let money_digits = deal
        .as_ref()
        .and_then(|item| item.close_position_detail.as_ref())
        .and_then(|detail| detail.money_digits)
        .or_else(|| deal.as_ref().and_then(|item| item.money_digits));
    let money_digits = crate::app_services::ctrader_money::required_money_digits(
        money_digits,
        "execution.money_digits",
    );

    let gross_profit = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .map(|detail| scaled_money(detail.gross_profit, money_digits))
    });
    let fee = deal.as_ref().and_then(|item| {
        item.close_position_detail
            .as_ref()
            .map(|detail| scaled_money(detail.commission, money_digits))
            .or_else(|| {
                item.commission
                    .map(|commission| scaled_money(commission, money_digits))
            })
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
        (Some(gross), fee, swap, pnl_fee) => {
            Some(gross + fee.unwrap_or(0.0) + swap.unwrap_or(0.0) + pnl_fee.unwrap_or(0.0))
        }
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
    // Centralised cTrader spec helper (see ctrader_money.rs and
    // docs/audits/research/ctrader_api_full_reference.md §5.14). Out-of-range
    // `money_digits` triggers an error log + legacy fiat fallback so a single
    // malformed broker payload cannot panic the execution event parser.
    match crate::app_services::ctrader_money::scale_ctrader_money_int(raw, money_digits as i32) {
        Ok(v) => v,
        Err(err) => {
            tracing::error!(
                target: "neoethos_app::ctrader",
                money_digits,
                error = %err,
                "execution event money scaling rejected by spec helper; falling back to fiat default (2)"
            );
            (raw as f64) / 100.0
        }
    }
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
    // `FOREX_BOT_CTRADER_ALLOW_PARTIAL_FILL=1` to opt back into the previous
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
                 set FOREX_BOT_CTRADER_ALLOW_PARTIAL_FILL=1 to accept partial fills",
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

    match &request.request {
        CTraderExecutionRequest::NewOrder(inner) => {
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
    }

    Ok(())
}

#[cfg(test)]
#[path = "ctrader_execution_tests.rs"]
mod tests;
