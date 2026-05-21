use crate::app_services::ctrader_data::HistoricalBar;
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE, build_account_auth_request, build_application_auth_request,
    build_subscribe_live_trendbar_request, build_subscribe_spots_request,
    expected_response_payload_type, is_matching_open_api_response, parse_account_disconnect_event,
    parse_ctrader_error_payload, parse_open_api_envelope, trendbar_period_value,
};
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};
type CTraderSocket = WebSocket<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderLiveChartUpdateRequest {
    pub client_id: String,
    pub client_secret: String,
    pub access_token: String,
    pub environment: CTraderEnvironment,
    pub account_id: String,
    pub symbol_id: i64,
    pub digits: i32,
    pub timeframe: String,
    pub subscribe_to_spot_timestamp: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CTraderLiveChartUpdate {
    pub symbol_id: i64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub timestamp_ms: Option<i64>,
    pub latest_trendbar: Option<HistoricalBar>,
}

impl CTraderLiveChartUpdate {
    /// Returns `(bid + ask) / 2` only when both sides are present.
    /// Previously this fell back to the available side, which silently
    /// biased SL/TP evaluation by half a spread when one quote was stale.
    /// Callers that need a one-sided fallback must opt in via
    /// [`Self::bid`] / [`Self::ask`].
    pub fn mid_price(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    pub fn bid(&self) -> Option<f64> {
        self.bid
    }

    pub fn ask(&self) -> Option<f64> {
        self.ask
    }
}

pub trait CTraderLiveStreamingBackend: Send + Sync {
    fn load_live_chart_update(
        &self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> Result<CTraderLiveChartUpdate>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProductionCTraderLiveStreamingBackend;

pub trait CTraderLiveStreamingTransport {
    fn authenticate_subscribe_and_wait_for_spot(
        &self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> Result<(Vec<String>, String)>;
}

#[derive(Debug, Clone)]
pub struct ProductionCTraderLiveStreamingTransport {
    endpoint_host: String,
}

impl ProductionCTraderLiveStreamingTransport {
    pub fn new(endpoint_host: impl Into<String>) -> Self {
        Self {
            endpoint_host: endpoint_host.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CTraderStreamingSessionKey {
    endpoint_host: String,
    client_id: String,
    client_secret: String,
    access_token: String,
    account_id: i64,
    symbol_id: i64,
    timeframe: String,
    subscribe_to_spot_timestamp: bool,
}

impl CTraderStreamingSessionKey {
    fn from_request(
        endpoint_host: &str,
        request: &CTraderLiveChartUpdateRequest,
        account_id: i64,
    ) -> Self {
        Self {
            endpoint_host: endpoint_host.to_string(),
            client_id: request.client_id.clone(),
            client_secret: request.client_secret.clone(),
            access_token: request.access_token.clone(),
            account_id,
            symbol_id: request.symbol_id,
            timeframe: request.timeframe.clone(),
            subscribe_to_spot_timestamp: request.subscribe_to_spot_timestamp,
        }
    }
}

struct CTraderStreamingSession {
    key: CTraderStreamingSessionKey,
    responses: Vec<String>,
    socket: CTraderSocket,
}

/// Sentinel prefix surfaced through the streaming `anyhow::Error` chain
/// when the broker emits `ProtoOAAccountDisconnectEvent`. Callers (UI /
/// reconnect loop) match on this string to distinguish a session that
/// the broker dropped server-side from generic transport failures.
/// Mirrors the existing `CTRADER_TOKEN_EXPIRED_SENTINEL` pattern.
pub const CTRADER_ACCOUNT_DISCONNECT_SENTINEL: &str = "CTRADER_ACCOUNT_DISCONNECT";

static CTRADER_STREAMING_SESSION: OnceLock<Mutex<Option<CTraderStreamingSession>>> =
    OnceLock::new();

fn streaming_session_cache() -> &'static Mutex<Option<CTraderStreamingSession>> {
    CTRADER_STREAMING_SESSION.get_or_init(|| Mutex::new(None))
}

#[derive(Debug, Deserialize)]
struct SpotEventEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: u32,
    payload: SpotEventPayload,
}

#[derive(Debug, Deserialize)]
struct SpotEventPayload {
    #[serde(rename = "ctidTraderAccountId")]
    ctid_trader_account_id: i64,
    #[serde(rename = "symbolId")]
    symbol_id: i64,
    bid: Option<u64>,
    ask: Option<u64>,
    #[serde(default)]
    trendbar: Vec<SpotTrendbarPayload>,
    timestamp: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct SpotTrendbarPayload {
    volume: Option<i64>,
    low: i64,
    #[serde(rename = "deltaOpen")]
    delta_open: Option<u64>,
    #[serde(rename = "deltaClose")]
    delta_close: Option<u64>,
    #[serde(rename = "deltaHigh")]
    delta_high: Option<u64>,
    #[serde(rename = "utcTimestampInMinutes")]
    utc_timestamp_in_minutes: Option<u32>,
}

pub fn parse_spot_event(
    response_json: &str,
    expected_account_id: i64,
    expected_symbol_id: i64,
    digits: i32,
) -> Result<CTraderLiveChartUpdate> {
    let envelope: SpotEventEnvelope =
        serde_json::from_str(response_json).context("failed to parse cTrader spot event")?;
    if envelope.payload_type != CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
        return Err(anyhow!(
            "unexpected cTrader spot event payload type: {}",
            envelope.payload_type
        ));
    }
    if envelope.payload.ctid_trader_account_id != expected_account_id {
        return Err(anyhow!(
            "unexpected cTrader spot event account id: {}",
            envelope.payload.ctid_trader_account_id
        ));
    }
    if envelope.payload.symbol_id != expected_symbol_id {
        return Err(anyhow!(
            "unexpected cTrader spot event symbol id: {}",
            envelope.payload.symbol_id
        ));
    }

    Ok(CTraderLiveChartUpdate {
        symbol_id: envelope.payload.symbol_id,
        bid: envelope
            .payload
            .bid
            .map(|value| scaled_price(value as i64, digits)),
        ask: envelope
            .payload
            .ask
            .map(|value| scaled_price(value as i64, digits)),
        timestamp_ms: envelope.payload.timestamp,
        latest_trendbar: envelope
            .payload
            .trendbar
            .into_iter()
            .last()
            .and_then(|bar| normalize_spot_trendbar(bar, digits)),
    })
}

pub fn merge_live_trendbar_into_bars(
    existing_bars: &[HistoricalBar],
    live_trendbar: Option<&HistoricalBar>,
) -> Vec<HistoricalBar> {
    let Some(live_trendbar) = live_trendbar else {
        return existing_bars.to_vec();
    };

    let mut merged = existing_bars.to_vec();
    match merged.last().map(|bar| bar.timestamp_ms) {
        Some(last_timestamp) if last_timestamp == live_trendbar.timestamp_ms => {
            if let Some(last) = merged.last_mut() {
                *last = live_trendbar.clone();
            }
        }
        Some(last_timestamp) if last_timestamp > live_trendbar.timestamp_ms => {}
        _ => merged.push(live_trendbar.clone()),
    }
    merged
}

/// Which quote to use when merging a spot update into chart bars.
/// `Mid` is the default and matches the historical OHLCV files used
/// during backtest, so chart display lines up with the data the
/// strategies were trained on. `Bid` / `Ask` are useful for visualising
/// the side a trade would actually fill at, or for chart parity tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeQuoteSide {
    Mid,
    Bid,
    Ask,
}

impl MergeQuoteSide {
    pub fn from_env() -> Self {
        match std::env::var("FOREX_BOT_CHART_MERGE_SIDE")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("bid") => Self::Bid,
            Some("ask") => Self::Ask,
            _ => Self::Mid,
        }
    }

    fn select(self, update: &CTraderLiveChartUpdate) -> Option<f64> {
        match self {
            Self::Mid => update.mid_price(),
            Self::Bid => update.bid(),
            Self::Ask => update.ask(),
        }
    }
}

pub fn merge_live_spot_update_into_bars(
    existing_bars: &[HistoricalBar],
    live_update: Option<&CTraderLiveChartUpdate>,
) -> Vec<HistoricalBar> {
    merge_live_spot_update_into_bars_with_side(
        existing_bars,
        live_update,
        MergeQuoteSide::from_env(),
    )
}

/// Side-aware variant of [`merge_live_spot_update_into_bars`]. Picks the
/// merge price from the chosen quote side (mid / bid / ask) and falls
/// back to copying `existing_bars` unchanged when the chosen side is
/// missing. Strategies and chart layers that need to reason in pure
/// bid- or ask-space (e.g. for slippage parity testing) should call
/// this directly instead of the env-driven default.
pub fn merge_live_spot_update_into_bars_with_side(
    existing_bars: &[HistoricalBar],
    live_update: Option<&CTraderLiveChartUpdate>,
    side: MergeQuoteSide,
) -> Vec<HistoricalBar> {
    let Some(live_update) = live_update else {
        return existing_bars.to_vec();
    };

    if live_update.latest_trendbar.is_some() {
        return merge_live_trendbar_into_bars(existing_bars, live_update.latest_trendbar.as_ref());
    }

    let Some(merge_price) = side.select(live_update) else {
        return existing_bars.to_vec();
    };
    let Some(last_bar) = existing_bars.last() else {
        return existing_bars.to_vec();
    };

    let mut merged = existing_bars.to_vec();
    if let Some(last) = merged.last_mut() {
        *last = HistoricalBar {
            timestamp_ms: last_bar.timestamp_ms,
            open: last_bar.open,
            high: last_bar.high.max(merge_price),
            low: last_bar.low.min(merge_price),
            close: merge_price,
            volume: last_bar.volume,
        };
    }
    merged
}

pub fn load_live_chart_update_with_transport<T: CTraderLiveStreamingTransport>(
    transport: &T,
    request: &CTraderLiveChartUpdateRequest,
) -> Result<CTraderLiveChartUpdate> {
    let account_id = request
        .account_id
        .parse::<i64>()
        .context("cTrader account id must be numeric")?;

    let (responses, spot_event_json) =
        transport.authenticate_subscribe_and_wait_for_spot(request)?;
    if responses.len() != 4 {
        return Err(anyhow!(
            "expected 4 cTrader live subscription responses, received {}",
            responses.len()
        ));
    }

    ensure_success_payload_type(
        &responses[0],
        "app-auth-1",
        expected_response_payload_type(
            build_application_auth_request(
                &request.client_id,
                &request.client_secret,
                "app-auth-1",
            )
            .payload_type,
        )?,
    )?;
    ensure_success_payload_type(
        &responses[1],
        "account-auth-1",
        expected_response_payload_type(
            build_account_auth_request(account_id, &request.access_token, "account-auth-1")
                .payload_type,
        )?,
    )?;
    ensure_success_payload_type(
        &responses[2],
        "subscribe-spots-1",
        expected_response_payload_type(
            build_subscribe_spots_request(
                account_id,
                &[request.symbol_id],
                request.subscribe_to_spot_timestamp,
                "subscribe-spots-1",
            )
            .payload_type,
        )?,
    )?;
    ensure_success_payload_type(
        &responses[3],
        "subscribe-trendbar-1",
        expected_response_payload_type(
            build_subscribe_live_trendbar_request(
                account_id,
                request.symbol_id,
                trendbar_period_value(&request.timeframe)?,
                "subscribe-trendbar-1",
            )
            .payload_type,
        )?,
    )?;

    parse_spot_event(
        &spot_event_json,
        account_id,
        request.symbol_id,
        request.digits,
    )
}

pub fn load_live_chart_update(
    request: &CTraderLiveChartUpdateRequest,
) -> Result<CTraderLiveChartUpdate> {
    let transport =
        ProductionCTraderLiveStreamingTransport::new(request.environment.endpoint_host());
    let max_attempts = streaming_max_attempts();
    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..max_attempts {
        if attempt > 0 {
            streaming_backoff_sleep(attempt);
        }
        match load_live_chart_update_with_transport(&transport, request) {
            Ok(update) => return Ok(update),
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_app::ctrader",
                    attempt = attempt + 1,
                    max_attempts,
                    error = %err,
                    "cTrader streaming attempt failed"
                );
                last_error = Some(err);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("cTrader streaming attempts exhausted")))
}

/// Maximum number of attempts (initial + retries) for `load_live_chart_update`.
/// Tunable via `FOREX_BOT_CTRADER_STREAM_MAX_ATTEMPTS` (clamped to `[1, 5]`;
/// default 3). Retry is safe here because each call is a stateless poll.
fn streaming_max_attempts() -> u32 {
    std::env::var("FOREX_BOT_CTRADER_STREAM_MAX_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(3)
        .clamp(1, 5)
}

/// Base backoff in ms; tunable via `FOREX_BOT_CTRADER_STREAM_BACKOFF_BASE_MS`
/// (clamped to `[10, 2000]`; default 200).
fn streaming_backoff_base_ms() -> u64 {
    std::env::var("FOREX_BOT_CTRADER_STREAM_BACKOFF_BASE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(200)
        .clamp(10, 2_000)
}

fn streaming_backoff_sleep(attempt: u32) {
    crate::app_services::backoff::backoff_sleep(attempt, streaming_backoff_base_ms());
}

impl CTraderLiveStreamingBackend for ProductionCTraderLiveStreamingBackend {
    fn load_live_chart_update(
        &self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> Result<CTraderLiveChartUpdate> {
        load_live_chart_update(request)
    }
}

impl CTraderLiveStreamingTransport for ProductionCTraderLiveStreamingTransport {
    fn authenticate_subscribe_and_wait_for_spot(
        &self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> Result<(Vec<String>, String)> {
        let account_id = request
            .account_id
            .parse::<i64>()
            .context("cTrader account id must be numeric")?;
        let key =
            CTraderStreamingSessionKey::from_request(&self.endpoint_host, request, account_id);
        let mut session = {
            let mut cache = streaming_session_cache()
                .lock()
                .expect("cTrader streaming session cache lock");
            match cache.take() {
                Some(session) if session.key == key => session,
                Some(session) => {
                    let mut socket = session.socket;
                    let _ = socket.close(None);
                    drop(cache);
                    self.open_streaming_session(request, account_id, key.clone())?
                }
                None => {
                    drop(cache);
                    self.open_streaming_session(request, account_id, key.clone())?
                }
            }
        };

        let spot_event = self.read_next_spot_event(
            &mut session.socket,
            account_id,
            request.symbol_id,
            request.digits,
        )?;
        let responses = session.responses.clone();
        let mut cache = streaming_session_cache()
            .lock()
            .expect("cTrader streaming session cache lock");
        *cache = Some(session);
        Ok((responses, spot_event))
    }
}

impl ProductionCTraderLiveStreamingTransport {
    fn open_streaming_session(
        &self,
        request: &CTraderLiveChartUpdateRequest,
        account_id: i64,
        key: CTraderStreamingSessionKey,
    ) -> Result<CTraderStreamingSession> {
        let period = trendbar_period_value(&request.timeframe)?;
        let messages = vec![
            build_application_auth_request(
                &request.client_id,
                &request.client_secret,
                "app-auth-1",
            ),
            build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
            build_subscribe_spots_request(
                account_id,
                &[request.symbol_id],
                request.subscribe_to_spot_timestamp,
                "subscribe-spots-1",
            ),
            build_subscribe_live_trendbar_request(
                account_id,
                request.symbol_id,
                period,
                "subscribe-trendbar-1",
            ),
        ];

        let url = format!("wss://{}:5036", self.endpoint_host);
        crate::app_services::ctrader_tls::ensure_ctrader_rustls_provider();
        let (mut socket, _) = connect(url.as_str())
            .with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        let mut responses = Vec::with_capacity(messages.len());

        for message in &messages {
            let expected_payload_type = expected_response_payload_type(message.payload_type)?;
            let serialized = serde_json::to_string(message)
                .context("failed to serialize cTrader streaming message")?;
            socket
                .send(Message::Text(serialized.into()))
                .context("failed to send cTrader streaming message")?;

            loop {
                match socket
                    .read()
                    .context("failed to read cTrader streaming response")?
                {
                    Message::Text(text) => {
                        let envelope = parse_open_api_envelope(text.as_ref())?;
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            let _ = socket.close(None);
                            return Err(anyhow!(
                                "cTrader streaming request failed: {}",
                                parse_ctrader_error_payload(&envelope.payload)?
                            ));
                        }
                        if envelope.payload_type == CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE
                        {
                            let reason =
                                handle_account_disconnect_event(text.as_ref(), &mut socket);
                            return Err(anyhow!(
                                "{}: {}",
                                CTRADER_ACCOUNT_DISCONNECT_SENTINEL,
                                reason
                            ));
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type)
                        {
                            responses.push(text.to_string());
                            break;
                        }
                    }
                    Message::Binary(bytes) => {
                        let text = String::from_utf8(bytes.to_vec())
                            .context("failed to decode cTrader streaming response")?;
                        let envelope = parse_open_api_envelope(&text)?;
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            let _ = socket.close(None);
                            return Err(anyhow!(
                                "cTrader streaming request failed: {}",
                                parse_ctrader_error_payload(&envelope.payload)?
                            ));
                        }
                        if envelope.payload_type == CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE
                        {
                            let reason = handle_account_disconnect_event(&text, &mut socket);
                            return Err(anyhow!(
                                "{}: {}",
                                CTRADER_ACCOUNT_DISCONNECT_SENTINEL,
                                reason
                            ));
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type)
                        {
                            responses.push(text);
                            break;
                        }
                    }
                    Message::Ping(payload) => {
                        socket
                            .send(Message::Pong(payload))
                            .context("failed to reply to cTrader ping")?;
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) => {
                        return Err(anyhow!("cTrader streaming socket closed unexpectedly"));
                    }
                    Message::Frame(_) => {}
                }
            }
        }

        Ok(CTraderStreamingSession {
            key,
            responses,
            socket,
        })
    }

    fn read_next_spot_event(
        &self,
        socket: &mut CTraderSocket,
        account_id: i64,
        symbol_id: i64,
        digits: i32,
    ) -> Result<String> {
        loop {
            match socket.read().context("failed to read cTrader spot event")? {
                Message::Text(text) => {
                    if text.trim().is_empty() {
                        return Err(anyhow!("empty cTrader spot event"));
                    }
                    let envelope = parse_open_api_envelope(text.as_ref())?;
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        let _ = socket.close(None);
                        return Err(anyhow!(
                            "cTrader spot event stream failed: {}",
                            parse_ctrader_error_payload(&envelope.payload)?
                        ));
                    }
                    if envelope.payload_type == CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE {
                        let reason = handle_account_disconnect_event(text.as_ref(), socket);
                        return Err(anyhow!(
                            "{}: {}",
                            CTRADER_ACCOUNT_DISCONNECT_SENTINEL,
                            reason
                        ));
                    }
                    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
                        let parsed =
                            parse_spot_event(text.as_ref(), account_id, symbol_id, digits)?;
                        let _ = parsed; // keep parsing path honest without changing return contract
                        return Ok(text.to_string());
                    }
                }
                Message::Binary(bytes) => {
                    let text = String::from_utf8(bytes.to_vec())
                        .context("failed to decode cTrader spot event")?;
                    let envelope = parse_open_api_envelope(&text)?;
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        let _ = socket.close(None);
                        return Err(anyhow!(
                            "cTrader spot event stream failed: {}",
                            parse_ctrader_error_payload(&envelope.payload)?
                        ));
                    }
                    if envelope.payload_type == CTRADER_OA_ACCOUNT_DISCONNECT_EVENT_PAYLOAD_TYPE {
                        let reason = handle_account_disconnect_event(&text, socket);
                        return Err(anyhow!(
                            "{}: {}",
                            CTRADER_ACCOUNT_DISCONNECT_SENTINEL,
                            reason
                        ));
                    }
                    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
                        let parsed = parse_spot_event(&text, account_id, symbol_id, digits)?;
                        let _ = parsed;
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
                    return Err(anyhow!(
                        "cTrader spot stream closed before first spot event"
                    ));
                }
                Message::Frame(_) => {}
            }
        }
    }
}

/// Handle a `ProtoOAAccountDisconnectEvent` mid-stream: emit a `warn!`
/// log line so operators see the disconnect in the structured log,
/// close the socket so the cached session entry is dropped on the
/// next call (which forces re-auth), and return a human-readable
/// reason that the caller threads into the `anyhow::Error` chain.
///
/// We intentionally do NOT panic on a malformed disconnect payload;
/// the broker dropped us regardless, so we surface a best-effort
/// reason and let the reconnect loop take over.
fn handle_account_disconnect_event(payload_text: &str, socket: &mut CTraderSocket) -> String {
    let reason = match parse_account_disconnect_event(payload_text) {
        Ok(event) => format!(
            "account_id={} dropped by broker (session must re-auth)",
            event.ctid_trader_account_id
        ),
        Err(err) => {
            format!("account session dropped by broker (failed to parse disconnect payload: {err})")
        }
    };
    tracing::warn!(
        target: "neoethos_app::ctrader",
        reason = %reason,
        "cTrader account disconnect event: {}",
        reason
    );
    // Drop the cached session so the next `load_live_chart_update`
    // call opens a fresh connection and re-runs the auth + subscribe
    // sequence; this is the "needs_reconnect" signal the UI wants.
    let _ = socket.close(None);
    if let Ok(mut cache) = streaming_session_cache().lock() {
        if let Some(stale) = cache.take() {
            let mut socket = stale.socket;
            let _ = socket.close(None);
        }
    }
    reason
}

fn ensure_success_payload_type(
    response_json: &str,
    expected_client_msg_id: &str,
    expected_payload_type: u32,
) -> Result<()> {
    let envelope = parse_open_api_envelope(response_json)?;
    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
        return Err(anyhow!(
            "cTrader response failed: {}",
            parse_ctrader_error_payload(&envelope.payload)?
        ));
    }
    if envelope.client_msg_id != expected_client_msg_id {
        return Err(anyhow!(
            "unexpected cTrader client message id: expected {}, got {}",
            expected_client_msg_id,
            envelope.client_msg_id
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

fn normalize_spot_trendbar(payload: SpotTrendbarPayload, digits: i32) -> Option<HistoricalBar> {
    let timestamp_ms = i64::from(payload.utc_timestamp_in_minutes?) * 60_000;
    let low = scaled_price(payload.low, digits);
    let open = scaled_price(
        checked_low_plus_delta(payload.low, payload.delta_open)?,
        digits,
    );
    let close = scaled_price(
        checked_low_plus_delta(payload.low, payload.delta_close)?,
        digits,
    );
    let high = scaled_price(
        checked_low_plus_delta(payload.low, payload.delta_high)?,
        digits,
    );
    Some(HistoricalBar {
        timestamp_ms,
        open,
        high,
        low,
        close,
        volume: payload.volume,
    })
}

/// Reconstructs `low + delta` while detecting silent wraparound.
/// cTrader sends `delta_*` as `u64`; the protocol's real range is well
/// below `i64::MAX` but a malformed broker payload could push the cast
/// negative or overflow the addition. Returns `None` on either case so
/// the caller can drop the bar instead of producing a garbage price.
fn checked_low_plus_delta(low: i64, delta: Option<u64>) -> Option<i64> {
    let delta = delta.unwrap_or(0);
    let delta_i64 = i64::try_from(delta).ok()?;
    low.checked_add(delta_i64)
}

fn scaled_price(value: i64, digits: i32) -> f64 {
    let raw = value as f64 / 100000.0;
    let factor = 10_f64.powi(digits.max(0));
    (raw * factor).round() / factor
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct StubCTraderLiveStreamingBackend {
    update: CTraderLiveChartUpdate,
    requests: std::sync::Arc<std::sync::Mutex<Vec<CTraderLiveChartUpdateRequest>>>,
}

#[cfg(test)]
impl StubCTraderLiveStreamingBackend {
    pub fn success(update: CTraderLiveChartUpdate) -> Self {
        Self {
            update,
            requests: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn call_count(&self) -> usize {
        self.requests.lock().expect("requests lock").len()
    }
}

#[cfg(test)]
impl CTraderLiveStreamingBackend for StubCTraderLiveStreamingBackend {
    fn load_live_chart_update(
        &self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> Result<CTraderLiveChartUpdate> {
        self.requests
            .lock()
            .expect("requests lock")
            .push(request.clone());
        Ok(self.update.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage;
    use std::sync::{Arc, Mutex};

    #[test]
    fn spot_event_parser_normalizes_quote_and_trendbar_fields() {
        let event = serde_json::json!({
            "payloadType": 2131,
            "payload": {
                "ctidTraderAccountId": 712345,
                "symbolId": 14,
                "bid": 109995,
                "ask": 110015,
                "timestamp": 1710000200000i64,
                "trendbar": [{
                    "volume": 9,
                    "low": 109950,
                    "deltaOpen": 50,
                    "deltaClose": 125,
                    "deltaHigh": 225,
                    "utcTimestampInMinutes": 28500000
                }]
            }
        });

        let parsed = parse_spot_event(&event.to_string(), 712345, 14, 5).expect("spot event");

        assert_eq!(parsed.symbol_id, 14);
        assert_eq!(parsed.bid, Some(1.09995));
        assert_eq!(parsed.ask, Some(1.10015));
        assert_eq!(parsed.timestamp_ms, Some(1710000200000));
        assert_eq!(
            parsed.latest_trendbar,
            Some(HistoricalBar {
                timestamp_ms: 1_710_000_000_000,
                open: 1.1,
                high: 1.10175,
                low: 1.0995,
                close: 1.10075,
                volume: Some(9),
            })
        );
    }

    #[test]
    fn spot_event_parser_honors_symbol_digits() {
        let event = serde_json::json!({
            "payloadType": 2131,
            "payload": {
                "ctidTraderAccountId": 712345,
                "symbolId": 14,
                "bid": 109900,
                "ask": 110100,
                "timestamp": 1710000200000i64
            }
        });

        let parsed = parse_spot_event(&event.to_string(), 712345, 14, 3).expect("spot event");

        assert_eq!(parsed.bid, Some(1.099));
        assert_eq!(parsed.ask, Some(1.101));
    }

    #[test]
    fn live_chart_update_loader_authenticates_and_subscribes_before_consuming_spot_event() {
        let transport = StubStreamingTransport::success(
            vec![
                r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#,
                r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#,
                r#"{"clientMsgId":"subscribe-spots-1","payloadType":2128,"payload":{"ctidTraderAccountId":712345}}"#,
                r#"{"clientMsgId":"subscribe-trendbar-1","payloadType":2165,"payload":{"ctidTraderAccountId":712345}}"#,
            ],
            r#"{"payloadType":2131,"payload":{"ctidTraderAccountId":712345,"symbolId":14,"bid":109995,"ask":110015,"timestamp":1710000200000,"trendbar":[{"volume":9,"low":109950,"deltaOpen":50,"deltaClose":125,"deltaHigh":225,"utcTimestampInMinutes":28500000}]}}"#,
        );

        let update = load_live_chart_update_with_transport(
            &transport,
            &CTraderLiveChartUpdateRequest {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                access_token: "access".to_string(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".to_string(),
                symbol_id: 14,
                digits: 5,
                timeframe: "M5".to_string(),
                subscribe_to_spot_timestamp: true,
            },
        )
        .expect("live update");

        assert_eq!(update.bid, Some(1.09995));
        assert_eq!(update.ask, Some(1.10015));
        assert_eq!(transport.sent_len(), 4);
        assert_eq!(
            transport.last_sent_payload_types(),
            vec![2100, 2102, 2127, 2135]
        );
    }

    #[test]
    fn response_validator_rejects_unexpected_client_message_id() {
        let response = serde_json::json!({
            "clientMsgId": "wrong-id",
            "payloadType": 2101,
            "payload": {}
        });

        let err = ensure_success_payload_type(&response.to_string(), "expected-id", 2101)
            .expect_err("client id mismatch should fail");
        assert!(
            err.to_string()
                .contains("unexpected cTrader client message id")
        );
    }

    #[test]
    fn merge_live_trendbar_replaces_matching_timestamp_or_appends_new_one() {
        let existing = vec![
            HistoricalBar {
                timestamp_ms: 1000,
                open: 1.0,
                high: 1.1,
                low: 0.9,
                close: 1.05,
                volume: Some(1),
            },
            HistoricalBar {
                timestamp_ms: 2000,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: Some(2),
            },
        ];
        let replacement = HistoricalBar {
            timestamp_ms: 2000,
            open: 1.2,
            high: 1.3,
            low: 1.1,
            close: 1.25,
            volume: Some(3),
        };
        let appended = HistoricalBar {
            timestamp_ms: 3000,
            open: 1.3,
            high: 1.4,
            low: 1.2,
            close: 1.35,
            volume: Some(4),
        };

        let replaced = merge_live_trendbar_into_bars(&existing, Some(&replacement));
        let appended_result = merge_live_trendbar_into_bars(&replaced, Some(&appended));

        assert_eq!(replaced.len(), 2);
        assert_eq!(replaced[1], replacement);
        assert_eq!(appended_result.len(), 3);
        assert_eq!(appended_result[2], appended);
    }

    #[test]
    fn merge_live_spot_update_uses_mid_quote_when_no_trendbar_is_present() {
        let existing = vec![HistoricalBar {
            timestamp_ms: 2_000,
            open: 1.1000,
            high: 1.1010,
            low: 1.0990,
            close: 1.1005,
            volume: Some(2),
        }];

        let merged = merge_live_spot_update_into_bars(
            &existing,
            Some(&CTraderLiveChartUpdate {
                symbol_id: 14,
                bid: Some(1.0985),
                ask: Some(1.1015),
                timestamp_ms: Some(2_100),
                latest_trendbar: None,
            }),
        );

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].timestamp_ms, 2_000);
        assert!((merged[0].close - 1.1000).abs() < 1e-9);
        assert!((merged[0].high - 1.1010).abs() < 1e-9);
        assert!((merged[0].low - 1.0990).abs() < 1e-9);
    }

    #[test]
    fn merge_live_spot_update_with_side_uses_requested_quote() {
        let existing = vec![HistoricalBar {
            timestamp_ms: 2_000,
            open: 1.1000,
            high: 1.1010,
            low: 1.0990,
            close: 1.1005,
            volume: Some(2),
        }];
        let update = CTraderLiveChartUpdate {
            symbol_id: 14,
            bid: Some(1.0985),
            ask: Some(1.1015),
            timestamp_ms: Some(2_100),
            latest_trendbar: None,
        };

        let bid_merged = merge_live_spot_update_into_bars_with_side(
            &existing,
            Some(&update),
            MergeQuoteSide::Bid,
        );
        let ask_merged = merge_live_spot_update_into_bars_with_side(
            &existing,
            Some(&update),
            MergeQuoteSide::Ask,
        );
        let mid_merged = merge_live_spot_update_into_bars_with_side(
            &existing,
            Some(&update),
            MergeQuoteSide::Mid,
        );

        assert!((bid_merged[0].close - 1.0985).abs() < 1e-9);
        assert!((ask_merged[0].close - 1.1015).abs() < 1e-9);
        assert!((mid_merged[0].close - 1.1000).abs() < 1e-9);
    }

    #[test]
    fn mid_price_requires_both_sides_to_avoid_half_spread_bias() {
        let bid_only = CTraderLiveChartUpdate {
            symbol_id: 1,
            bid: Some(1.1000),
            ask: None,
            timestamp_ms: Some(0),
            latest_trendbar: None,
        };
        let ask_only = CTraderLiveChartUpdate {
            symbol_id: 1,
            bid: None,
            ask: Some(1.1010),
            timestamp_ms: Some(0),
            latest_trendbar: None,
        };
        let both = CTraderLiveChartUpdate {
            symbol_id: 1,
            bid: Some(1.1000),
            ask: Some(1.1010),
            timestamp_ms: Some(0),
            latest_trendbar: None,
        };
        assert!(bid_only.mid_price().is_none());
        assert!(ask_only.mid_price().is_none());
        assert!((both.mid_price().unwrap() - 1.1005).abs() < 1e-9);
        assert_eq!(bid_only.bid(), Some(1.1000));
        assert_eq!(ask_only.ask(), Some(1.1010));
    }

    struct StubStreamingTransport {
        sent: Arc<Mutex<Vec<CTraderOpenApiJsonMessage>>>,
        responses: Vec<String>,
        spot_event: String,
    }

    impl StubStreamingTransport {
        fn success(responses: Vec<&str>, spot_event: &str) -> Self {
            Self {
                sent: Arc::new(Mutex::new(Vec::new())),
                responses: responses.into_iter().map(str::to_string).collect(),
                spot_event: spot_event.to_string(),
            }
        }

        fn sent_len(&self) -> usize {
            self.sent.lock().expect("sent lock").len()
        }

        fn last_sent_payload_types(&self) -> Vec<u32> {
            self.sent
                .lock()
                .expect("sent lock")
                .iter()
                .map(|message| message.payload_type)
                .collect()
        }
    }

    impl CTraderLiveStreamingTransport for StubStreamingTransport {
        fn authenticate_subscribe_and_wait_for_spot(
            &self,
            request: &CTraderLiveChartUpdateRequest,
        ) -> Result<(Vec<String>, String)> {
            let account_id = request
                .account_id
                .parse::<i64>()
                .context("cTrader account id must be numeric")?;
            let period = trendbar_period_value(&request.timeframe)?;
            self.sent.lock().expect("sent lock").extend([
                build_application_auth_request(
                    &request.client_id,
                    &request.client_secret,
                    "app-auth-1",
                ),
                build_account_auth_request(account_id, &request.access_token, "account-auth-1"),
                build_subscribe_spots_request(
                    account_id,
                    &[request.symbol_id],
                    request.subscribe_to_spot_timestamp,
                    "subscribe-spots-1",
                ),
                build_subscribe_live_trendbar_request(
                    account_id,
                    request.symbol_id,
                    period,
                    "subscribe-trendbar-1",
                ),
            ]);
            Ok((self.responses.clone(), self.spot_event.clone()))
        }
    }
}
