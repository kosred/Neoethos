use crate::app_services::ctrader_data::HistoricalBar;
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    build_account_auth_request, build_application_auth_request,
    build_subscribe_live_trendbar_request, build_subscribe_spots_request,
    expected_response_payload_type, parse_ctrader_error_payload, parse_open_api_envelope,
    trendbar_period_value, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE,
};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tungstenite::{connect, Message};

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
        bid: envelope.payload.bid.map(|value| scaled_price(value as i64, digits)),
        ask: envelope.payload.ask.map(|value| scaled_price(value as i64, digits)),
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
        expected_response_payload_type(
            build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1")
                .payload_type,
        )?,
    )?;
    ensure_success_payload_type(
        &responses[1],
        expected_response_payload_type(
            build_account_auth_request(account_id, &request.access_token, "account-auth-1")
                .payload_type,
        )?,
    )?;
    ensure_success_payload_type(
        &responses[2],
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

    parse_spot_event(&spot_event_json, account_id, request.symbol_id, request.digits)
}

pub fn load_live_chart_update(
    request: &CTraderLiveChartUpdateRequest,
) -> Result<CTraderLiveChartUpdate> {
    let transport = ProductionCTraderLiveStreamingTransport::new(request.environment.endpoint_host());
    load_live_chart_update_with_transport(&transport, request)
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
        let period = trendbar_period_value(&request.timeframe)?;
        let messages = vec![
            build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
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
        let (mut socket, _) =
            connect(url.as_str()).with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        let mut responses = Vec::with_capacity(messages.len());

        for message in &messages {
            let expected_payload_type = expected_response_payload_type(message.payload_type)?;
            let serialized = serde_json::to_string(message)
                .context("failed to serialize cTrader streaming message")?;
            socket
                .send(Message::Text(serialized.into()))
                .context("failed to send cTrader streaming message")?;

            loop {
                match socket.read().context("failed to read cTrader streaming response")? {
                    Message::Text(text) => {
                        let envelope = parse_open_api_envelope(text.as_ref())?;
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            let _ = socket.close(None);
                            return Err(anyhow!(
                                "cTrader streaming request failed: {}",
                                parse_ctrader_error_payload(&envelope.payload)?
                            ));
                        }
                        if envelope.payload_type == expected_payload_type
                            && envelope.client_msg_id == message.client_msg_id
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
                        if envelope.payload_type == expected_payload_type
                            && envelope.client_msg_id == message.client_msg_id
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

        loop {
            match socket.read().context("failed to read cTrader spot event")? {
                Message::Text(text) => {
                    let envelope = parse_open_api_envelope(text.as_ref())?;
                    if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        let _ = socket.close(None);
                        return Err(anyhow!(
                            "cTrader spot event stream failed: {}",
                            parse_ctrader_error_payload(&envelope.payload)?
                        ));
                    }
                    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
                        let _ = socket.close(None);
                        return Ok((responses, text.to_string()));
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
                    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
                        let _ = socket.close(None);
                        return Ok((responses, text));
                    }
                }
                Message::Ping(payload) => {
                    socket
                        .send(Message::Pong(payload))
                        .context("failed to reply to cTrader ping")?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => {
                    return Err(anyhow!("cTrader spot stream closed before first spot event"));
                }
                Message::Frame(_) => {}
            }
        }
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

fn normalize_spot_trendbar(payload: SpotTrendbarPayload, digits: i32) -> Option<HistoricalBar> {
    let timestamp_ms = i64::from(payload.utc_timestamp_in_minutes?) * 60_000;
    let low = scaled_price(payload.low, digits);
    let open = scaled_price(payload.low + payload.delta_open.unwrap_or_default() as i64, digits);
    let close = scaled_price(payload.low + payload.delta_close.unwrap_or_default() as i64, digits);
    let high = scaled_price(payload.low + payload.delta_high.unwrap_or_default() as i64, digits);
    Some(HistoricalBar {
        timestamp_ms,
        open,
        high,
        low,
        close,
        volume: payload.volume,
    })
}

fn scaled_price(value: i64, digits: i32) -> f64 {
    let raw = value as f64 / 100000.0;
    let factor = 10_f64.powi(digits.max(0));
    (raw * factor).round() / factor
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
                build_application_auth_request(&request.client_id, &request.client_secret, "app-auth-1"),
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
