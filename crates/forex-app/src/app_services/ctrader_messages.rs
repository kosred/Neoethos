use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tungstenite::{connect, Message};

pub const CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE: u32 = 2100;
pub const CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE: u32 = 2101;
pub const CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE: u32 = 2102;
pub const CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE: u32 = 2103;
pub const CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE: u32 = 2121;
pub const CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE: u32 = 2122;
pub const CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE: u32 = 2124;
pub const CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE: u32 = 2125;
pub const CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE: u32 = 2127;
pub const CTRADER_OA_SUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE: u32 = 2128;
pub const CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE: u32 = 2129;
pub const CTRADER_OA_UNSUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE: u32 = 2130;
pub const CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE: u32 = 2131;
pub const CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE: u32 = 2135;
pub const CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE: u32 = 2136;
pub const CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE: u32 = 2114;
pub const CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE: u32 = 2115;
pub const CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE: u32 = 2116;
pub const CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE: u32 = 2117;
pub const CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE: u32 = 2137;
pub const CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE: u32 = 2138;
pub const CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE: u32 = 2142;
pub const CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE: u32 = 2145;
pub const CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE: u32 = 2146;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE: u32 = 2149;
pub const CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE: u32 = 2150;
pub const CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE: u32 = 2165;
pub const CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE: u32 = 2166;
pub const CTRADER_QUOTE_TYPE_BID: i32 = 1;
pub const CTRADER_QUOTE_TYPE_ASK: i32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTraderOpenApiJsonMessage {
    #[serde(rename = "clientMsgId")]
    pub client_msg_id: String,
    #[serde(rename = "payloadType")]
    pub payload_type: u32,
    pub payload: Value,
}

pub trait CTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>>;
}

pub struct ProductionCTraderOpenApiTransport {
    endpoint_host: String,
}

impl ProductionCTraderOpenApiTransport {
    pub fn new(endpoint_host: impl Into<String>) -> Self {
        Self {
            endpoint_host: endpoint_host.into(),
        }
    }
}

pub fn build_application_auth_json(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "clientId": client_id,
            "clientSecret": client_secret,
        }),
    }
}

pub fn build_application_auth_request(
    client_id: &str,
    client_secret: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    build_application_auth_json(client_id, client_secret, client_msg_id)
}

pub fn build_account_auth_request(
    ctid_trader_account_id: i64,
    access_token: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "accessToken": access_token,
        }),
    }
}

pub fn build_account_list_by_access_token_request(
    access_token: &str,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "accessToken": access_token,
        }),
    }
}

pub fn build_trader_request(
    ctid_trader_account_id: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
        }),
    }
}

pub fn build_reconcile_request(
    ctid_trader_account_id: i64,
    return_protection_orders: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "returnProtectionOrders": return_protection_orders,
        }),
    }
}

pub fn build_subscribe_spots_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    subscribe_to_spot_timestamp: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
            "subscribeToSpotTimestamp": subscribe_to_spot_timestamp,
        }),
    }
}

pub fn build_unsubscribe_spots_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

pub fn build_subscribe_live_trendbar_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "period": period,
        }),
    }
}

pub fn build_unsubscribe_live_trendbar_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "period": period,
        }),
    }
}

pub fn build_symbols_list_request(
    ctid_trader_account_id: i64,
    include_archived_symbols: bool,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "includeArchivedSymbols": include_archived_symbols,
        }),
    }
}

pub fn build_symbol_by_id_request(
    ctid_trader_account_id: i64,
    symbol_ids: &[i64],
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_ids,
        }),
    }
}

pub fn build_get_trendbars_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period: i32,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    count: Option<u32>,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    let mut payload = serde_json::json!({
        "ctidTraderAccountId": ctid_trader_account_id,
        "symbolId": symbol_id,
        "period": period,
        "fromTimestamp": from_timestamp_ms,
        "toTimestamp": to_timestamp_ms,
    });
    if let Some(count) = count {
        payload["count"] = serde_json::json!(count);
    }
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE,
        payload,
    }
}

pub fn trendbar_period_value(label: &str) -> Result<i32> {
    match label.trim().to_ascii_uppercase().as_str() {
        "M1" => Ok(1),
        "M2" => Ok(2),
        "M3" => Ok(3),
        "M4" => Ok(4),
        "M5" => Ok(5),
        "M10" => Ok(6),
        "M15" => Ok(7),
        "M30" => Ok(8),
        "H1" => Ok(9),
        "H4" => Ok(10),
        "H12" => Ok(11),
        "D1" => Ok(12),
        "W1" => Ok(13),
        "MN1" => Ok(14),
        other => Err(anyhow!("unsupported cTrader trendbar period label {}", other)),
    }
}

pub fn build_get_tick_data_request(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    quote_type: i32,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    client_msg_id: impl Into<String>,
) -> CTraderOpenApiJsonMessage {
    CTraderOpenApiJsonMessage {
        client_msg_id: client_msg_id.into(),
        payload_type: CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE,
        payload: serde_json::json!({
            "ctidTraderAccountId": ctid_trader_account_id,
            "symbolId": symbol_id,
            "type": quote_type,
            "fromTimestamp": from_timestamp_ms,
            "toTimestamp": to_timestamp_ms,
        }),
    }
}

pub fn parse_open_api_envelope(response_json: &str) -> Result<CTraderOpenApiJsonMessage> {
    serde_json::from_str(response_json).context("failed to parse cTrader JSON envelope")
}

pub fn expected_response_payload_type(request_payload_type: u32) -> Result<u32> {
    match request_payload_type {
        CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_TRADER_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_UNSUBSCRIBE_SPOTS_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_SYMBOL_BY_ID_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE => Ok(CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE),
        CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE => {
            Ok(CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_RESPONSE_PAYLOAD_TYPE)
        }
        CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE => Err(anyhow!(
            "cTrader spot events are push-only payloads and are not valid request messages"
        )),
        payload_type => Err(anyhow!(
            "unsupported cTrader request payload type: {}",
            payload_type
        )),
    }
}

pub fn is_matching_open_api_response(
    envelope: &CTraderOpenApiJsonMessage,
    request: &CTraderOpenApiJsonMessage,
    expected_payload_type: u32,
) -> bool {
    if envelope.payload_type == CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE {
        return false;
    }
    envelope.payload_type == expected_payload_type && envelope.client_msg_id == request.client_msg_id
}

pub fn parse_ctrader_error_payload(payload: &Value) -> Result<String> {
    #[derive(Debug, Deserialize)]
    struct CTraderErrorPayload {
        #[serde(rename = "errorCode")]
        error_code: String,
        description: Option<String>,
    }

    let error: CTraderErrorPayload = serde_json::from_value(payload.clone())
        .context("failed to parse cTrader error payload")?;
    Ok(match error.description {
        Some(description) if !description.trim().is_empty() => {
            format!("{}: {}", error.error_code, description)
        }
        _ => error.error_code,
    })
}

impl CTraderOpenApiTransport for ProductionCTraderOpenApiTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        let url = format!("wss://{}:5036", self.endpoint_host);
        let (mut socket, _) =
            connect(url.as_str()).with_context(|| format!("failed to connect to cTrader endpoint {url}"))?;
        let mut responses = Vec::with_capacity(messages.len());

        for message in messages {
            let expected_payload_type = expected_response_payload_type(message.payload_type)?;
            let serialized = serde_json::to_string(message)
                .context("failed to serialize cTrader open api message")?;
            socket
                .send(Message::Text(serialized.into()))
                .context("failed to send cTrader open api message")?;

            loop {
                match socket.read().context("failed to read cTrader open api response")? {
                    Message::Text(text) => {
                        if text.trim().is_empty() {
                            return Err(anyhow!("empty cTrader open api response"));
                        }
                        let envelope = parse_open_api_envelope(text.as_ref())?;
                        if envelope.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                            responses.push(text.to_string());
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type) {
                            responses.push(text.to_string());
                            break;
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
                            responses.push(text);
                            let _ = socket.close(None);
                            return Ok(responses);
                        }
                        if is_matching_open_api_response(&envelope, message, expected_payload_type) {
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
                        return Err(anyhow!("cTrader open api socket closed unexpectedly"));
                    }
                    Message::Frame(_) => {}
                }
            }
        }

        let _ = socket.close(None);
        Ok(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_auth_request_uses_documented_payload_type() {
        let message = build_application_auth_request("client-id", "secret-456", "cm-id-2");

        assert_eq!(message.client_msg_id, "cm-id-2");
        assert_eq!(message.payload_type, CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("clientId").and_then(serde_json::Value::as_str),
            Some("client-id")
        );
        assert_eq!(
            message.payload.get("clientSecret").and_then(serde_json::Value::as_str),
            Some("secret-456")
        );
    }

    #[test]
    fn account_auth_request_uses_documented_payload_type_and_account_id() {
        let message = build_account_auth_request(7001, "token-123", "account-auth-1");

        assert_eq!(message.payload_type, CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message.payload.get("accessToken").and_then(serde_json::Value::as_str),
            Some("token-123")
        );
    }

    #[test]
    fn account_list_request_uses_documented_payload_type() {
        let message = build_account_list_by_access_token_request("access-token-123", "cm-id-1");

        assert_eq!(message.client_msg_id, "cm-id-1");
        assert_eq!(
            message.payload_type,
            CTRADER_OA_GET_ACCOUNTS_BY_ACCESS_TOKEN_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            message.payload.get("accessToken").and_then(serde_json::Value::as_str),
            Some("access-token-123")
        );
    }

    #[test]
    fn trader_request_uses_documented_payload_type_and_account_id() {
        let message = build_trader_request(7001, "trader-1");

        assert_eq!(message.client_msg_id, "trader-1");
        assert_eq!(message.payload_type, CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
    }

    #[test]
    fn reconcile_request_uses_documented_payload_type_and_optional_protection_flag() {
        let message = build_reconcile_request(7001, true, "reconcile-1");

        assert_eq!(message.client_msg_id, "reconcile-1");
        assert_eq!(message.payload_type, CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message
                .payload
                .get("returnProtectionOrders")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn subscribe_spots_request_uses_documented_symbol_ids_and_timestamp_flag() {
        let message = build_subscribe_spots_request(7001, &[14, 15], true, "spots-1");

        assert_eq!(message.client_msg_id, "spots-1");
        assert_eq!(message.payload_type, CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message.payload.get("symbolId").and_then(serde_json::Value::as_array).map(|items| items.len()),
            Some(2)
        );
        assert_eq!(
            message
                .payload
                .get("subscribeToSpotTimestamp")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn subscribe_live_trendbar_request_uses_documented_period_and_symbol_id() {
        let message = build_subscribe_live_trendbar_request(7001, 14, 7, "live-bars-1");

        assert_eq!(message.client_msg_id, "live-bars-1");
        assert_eq!(
            message.payload_type,
            CTRADER_OA_SUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message.payload.get("symbolId").and_then(serde_json::Value::as_i64),
            Some(14)
        );
        assert_eq!(
            message.payload.get("period").and_then(serde_json::Value::as_i64),
            Some(7)
        );
    }

    #[test]
    fn unsubscribe_requests_use_documented_payload_types() {
        let spots = build_unsubscribe_spots_request(7001, &[14], "spots-off-1");
        let bars = build_unsubscribe_live_trendbar_request(7001, 14, 7, "bars-off-1");

        assert_eq!(
            spots.payload_type,
            CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
        );
        assert_eq!(
            bars.payload_type,
            CTRADER_OA_UNSUBSCRIBE_LIVE_TRENDBAR_REQUEST_PAYLOAD_TYPE
        );
    }

    #[test]
    fn documented_spot_event_payload_type_constant_matches_official_message_id() {
        assert_eq!(CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE, 2131);
    }

    #[test]
    fn symbols_list_request_uses_documented_payload_type_and_account_id() {
        let message = build_symbols_list_request(7001, true, "symbols-list-1");

        assert_eq!(message.client_msg_id, "symbols-list-1");
        assert_eq!(message.payload_type, CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message
                .payload
                .get("includeArchivedSymbols")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn trendbars_request_uses_documented_payload_and_required_fields() {
        let message = build_get_trendbars_request(
            7001,
            9001,
            7,
            1_700_000_000_000,
            1_700_000_900_000,
            Some(400),
            "trendbars-1",
        );

        assert_eq!(message.client_msg_id, "trendbars-1");
        assert_eq!(message.payload_type, CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message.payload.get("symbolId").and_then(serde_json::Value::as_i64),
            Some(9001)
        );
        assert_eq!(
            message.payload.get("period").and_then(serde_json::Value::as_i64),
            Some(7)
        );
        assert_eq!(
            message.payload.get("fromTimestamp").and_then(serde_json::Value::as_i64),
            Some(1_700_000_000_000)
        );
        assert_eq!(
            message.payload.get("toTimestamp").and_then(serde_json::Value::as_i64),
            Some(1_700_000_900_000)
        );
        assert_eq!(
            message.payload.get("count").and_then(serde_json::Value::as_u64),
            Some(400)
        );
    }

    #[test]
    fn trendbar_period_value_matches_documented_ctrader_enum() {
        assert_eq!(trendbar_period_value("M1").expect("M1 should map"), 1);
        assert_eq!(trendbar_period_value("m15").expect("M15 should map"), 7);
        assert_eq!(trendbar_period_value("H1").expect("H1 should map"), 9);
        assert_eq!(trendbar_period_value("MN1").expect("MN1 should map"), 14);
    }

    #[test]
    fn tick_data_request_uses_documented_payload_and_quote_type() {
        let message = build_get_tick_data_request(
            7001,
            9001,
            CTRADER_QUOTE_TYPE_ASK,
            1_700_000_000_000,
            1_700_000_100_000,
            "ticks-1",
        );

        assert_eq!(message.client_msg_id, "ticks-1");
        assert_eq!(message.payload_type, CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE);
        assert_eq!(
            message.payload.get("ctidTraderAccountId").and_then(serde_json::Value::as_i64),
            Some(7001)
        );
        assert_eq!(
            message.payload.get("symbolId").and_then(serde_json::Value::as_i64),
            Some(9001)
        );
        assert_eq!(
            message.payload.get("type").and_then(serde_json::Value::as_i64),
            Some(i64::from(CTRADER_QUOTE_TYPE_ASK))
        );
    }

    #[test]
    fn ctrader_error_payloads_surface_code_and_description() {
        let error = parse_ctrader_error_payload(&serde_json::json!({
            "errorCode": "ACCOUNT_NOT_AUTHORIZED",
            "description": "The trading account is not authorized"
        }))
        .expect("error payload should parse");

        assert_eq!(
            error,
            "ACCOUNT_NOT_AUTHORIZED: The trading account is not authorized"
        );
    }
}
