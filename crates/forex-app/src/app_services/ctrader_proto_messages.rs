// Phase C3 audit + Flutter pivot context (2026-05-18 operator
// directive): this file is the cTrader Open API proto-builder
// surface. Every `build_*_request` and `parse_*_response` helper
// exists for spec parity with Spotware's published API. Consumers
// that DO call these helpers live in `ctrader_streaming.rs`,
// `ctrader_history.rs`, and the upcoming Flutter API layer's
// REST/gRPC bridge to the bot's broker channel.
//
// FILE-LOCAL allow only — NOT a workspace lint override.
#![allow(dead_code)]

use crate::app_services::ctrader_messages::{
    CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE, CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE, CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE,
    CTraderOpenApiJsonMessage,
};
use crate::app_services::ctrader_openapi::{
    ProtoHeartbeatEvent, ProtoMessage, ProtoOAAccountAuthReq, ProtoOAApplicationAuthReq,
    ProtoOAGetTrendbarsReq, ProtoOAGetTrendbarsRes, ProtoOAPayloadType, ProtoOAReconcileReq,
    ProtoOAReconcileRes, ProtoOATrendbarPeriod, ProtoPayloadType,
};
use anyhow::{Context, Result, anyhow};
use protobuf::{Message, Serialize};
use std::io::Read;

pub fn build_proto_message<M: Message + Serialize>(
    payload_type: u32,
    message: &M,
    client_msg_id: Option<String>,
) -> Result<Vec<u8>> {
    let mut envelope = ProtoMessage::new();
    let mut mut_envelope = envelope.as_mut();
    mut_envelope.set_payloadType(payload_type);
    mut_envelope.set_payload(message.serialize()?);
    if let Some(id) = client_msg_id {
        mut_envelope.set_clientMsgId(id);
    }
    envelope
        .serialize()
        .context("failed to serialize ProtoMessage envelope")
}

pub fn build_app_auth_req(
    client_id: &str,
    client_secret: &str,
    client_msg_id: Option<String>,
) -> Result<Vec<u8>> {
    let mut req = ProtoOAApplicationAuthReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_clientId(client_id.to_string());
    mut_req.set_clientSecret(client_secret.to_string());

    build_proto_message(
        i32::from(ProtoOAPayloadType::ProtoOaApplicationAuthReq) as u32,
        &req,
        client_msg_id,
    )
}

pub fn build_account_auth_req(
    account_id: i64,
    access_token: &str,
    client_msg_id: Option<String>,
) -> Result<Vec<u8>> {
    let mut req = ProtoOAAccountAuthReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_ctidTraderAccountId(account_id);
    mut_req.set_accessToken(access_token.to_string());

    build_proto_message(
        i32::from(ProtoOAPayloadType::ProtoOaAccountAuthReq) as u32,
        &req,
        client_msg_id,
    )
}

pub fn build_heartbeat() -> Result<Vec<u8>> {
    let req = ProtoHeartbeatEvent::new();
    build_proto_message(
        i32::from(ProtoPayloadType::HeartbeatEvent) as u32,
        &req,
        None,
    )
}

pub fn parse_proto_message(data: &[u8]) -> Result<ProtoMessage> {
    ProtoMessage::parse(data)
        .map_err(|e| anyhow::anyhow!("failed to parse ProtoMessage envelope: {:?}", e))
}

// ─────────────────────────────────────────────────────────────────────────────
// Length-prefix framing codec for native Protobuf-over-TCP (port 5035).
//
// Per cTrader Open API spec §1.5
// (docs/audits/research/ctrader_api_full_reference.md §1.5):
//
//     [ 4-byte big-endian length prefix ][ serialised ProtoMessage bytes ]
//
// The 4-byte length prefix is REQUIRED only on the raw-TCP Protobuf
// transport (port 5035). WebSocket transport (port 5036) frames messages
// itself, so the JSON-WSS path does NOT use this helper. The docs say
// "little-endian … reverse the length bytes" — Spotware's reference .NET
// and Python SDKs both write the length **big-endian** because the
// help-centre text is describing the native-platform translation; we
// follow the SDK behaviour.
//
// This codec is the wire-format-only layer; the higher-level transport
// that frames + sends + receives messages over TLS+TCP lives in
// `ctrader_messages.rs` (`ProductionCTraderOpenApiProtobufTransport`).
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum size (bytes) of a single framed Protobuf payload. Spotware
/// does not publish an explicit cap, but `ProtoOAReconcileRes` /
/// `DealListRes` payloads observed in practice top out around ~1 MB. We
/// bound at 16 MB to surface clearly buggy length prefixes without
/// artificially limiting legitimate large-history responses.
pub const CTRADER_PROTOBUF_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Prepend the 4-byte big-endian length prefix to a serialised
/// `ProtoMessage` envelope. Output is suitable for direct write to the
/// raw-TCP socket.
pub fn frame_with_length_prefix(serialized_proto_message: &[u8]) -> Vec<u8> {
    let len = serialized_proto_message.len() as u32;
    let mut out = Vec::with_capacity(4 + serialized_proto_message.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(serialized_proto_message);
    out
}

/// Read one length-prefixed `ProtoMessage` envelope from a synchronous
/// reader. Returns the serialised envelope bytes (without the length
/// prefix). Errors if the frame exceeds
/// [`CTRADER_PROTOBUF_MAX_FRAME_BYTES`] to bound memory on a malformed
/// peer.
pub fn read_length_prefixed_frame<R: Read>(reader: &mut R) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .context("failed to read cTrader Protobuf frame length prefix")?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 {
        return Err(anyhow!(
            "cTrader Protobuf frame declared zero-length payload"
        ));
    }
    if len > CTRADER_PROTOBUF_MAX_FRAME_BYTES {
        return Err(anyhow!(
            "cTrader Protobuf frame payload {} bytes exceeds bound {}",
            len,
            CTRADER_PROTOBUF_MAX_FRAME_BYTES
        ));
    }
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .context("failed to read cTrader Protobuf frame payload")?;
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-endpoint Protobuf builders (v0.4.5 migration batch — items 2 + 3
// from the cTrader exhaustive docs sweep migration order: reconcile +
// historical bars).
//
// Each helper returns a fully-framed `Vec<u8>` ready to write to the TCP
// socket (length prefix + serialised `ProtoMessage`). The
// `client_msg_id` is copied into the envelope so the response can be
// matched on the same field the JSON-WSS path uses.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a framed `ProtoOAReconcileReq` (payload type 2124) ready for
/// the raw-TCP Protobuf transport on port 5035.
pub fn build_reconcile_req_proto(
    ctid_trader_account_id: i64,
    return_protection_orders: bool,
    client_msg_id: Option<String>,
) -> Result<Vec<u8>> {
    let mut req = ProtoOAReconcileReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_ctidTraderAccountId(ctid_trader_account_id);
    mut_req.set_returnProtectionOrders(return_protection_orders);

    let serialized = build_proto_message(
        i32::from(ProtoOAPayloadType::ProtoOaReconcileReq) as u32,
        &req,
        client_msg_id,
    )?;
    Ok(frame_with_length_prefix(&serialized))
}

/// Build a framed `ProtoOAGetTrendbarsReq` (payload type 2137) ready for
/// the raw-TCP Protobuf transport on port 5035. `period_value` is the
/// raw `ProtoOATrendbarPeriod` enum value (M1=1, M3=3, M5=5, M15=7,
/// M30=8, H1=9, H4=10, H12=11, D1=12, W1=13, MN1=14) — use
/// [`crate::app_services::ctrader_messages::trendbar_period_value`] to
/// map a canonical timeframe label to this integer.
pub fn build_get_trendbars_req_proto(
    ctid_trader_account_id: i64,
    symbol_id: i64,
    period_value: i32,
    from_timestamp_ms: i64,
    to_timestamp_ms: i64,
    count: Option<u32>,
    client_msg_id: Option<String>,
) -> Result<Vec<u8>> {
    let mut req = ProtoOAGetTrendbarsReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_ctidTraderAccountId(ctid_trader_account_id);
    mut_req.set_symbolId(symbol_id);
    let period = ProtoOATrendbarPeriod::try_from(period_value).map_err(|_| {
        anyhow!(
            "unsupported ProtoOATrendbarPeriod value {} (not in vendored enum)",
            period_value
        )
    })?;
    mut_req.set_period(period);
    mut_req.set_fromTimestamp(from_timestamp_ms);
    mut_req.set_toTimestamp(to_timestamp_ms);
    if let Some(count) = count {
        mut_req.set_count(count);
    }

    let serialized = build_proto_message(
        i32::from(ProtoOAPayloadType::ProtoOaGetTrendbarsReq) as u32,
        &req,
        client_msg_id,
    )?;
    Ok(frame_with_length_prefix(&serialized))
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-endpoint Protobuf → JSON adapters.
//
// The existing parser layer in `ctrader_account.rs` /
// `ctrader_history.rs` operates on JSON envelopes (the
// `CTraderOpenApiJsonMessage` shape). To keep the migration to native
// Protobuf strictly a wire-layer change, the Protobuf transport
// translates each parsed `ProtoMessage` envelope into the same JSON
// shape that the JSON-WSS proxy would have returned. Every existing
// caller (reconcile / trendbars parsing) keeps working with no source
// changes.
//
// Only the fields actually consumed by downstream parsers are emitted.
// New fields can be added incrementally as the migration scope grows.
// ─────────────────────────────────────────────────────────────────────────────

/// Translate a parsed `ProtoMessage` envelope into the JSON envelope
/// shape produced by the JSON-WSS proxy. Returns the JSON-encoded
/// string. Only the payload types from the v0.4.5 migration batch
/// (reconcile + trendbars) are supported; other payload types are
/// returned as an error so the caller can fall back to JSON-WSS.
pub fn proto_envelope_to_json_string(envelope_bytes: &[u8]) -> Result<String> {
    let envelope = parse_proto_message(envelope_bytes)?;
    let view = envelope.as_view();
    let payload_type = view.payloadType();
    let client_msg_id = view.clientMsgId().to_string();
    let payload_bytes: Vec<u8> = view.payload().to_vec();

    let payload_json = match payload_type {
        x if x == CTRADER_OA_RECONCILE_RESPONSE_PAYLOAD_TYPE => {
            proto_reconcile_res_to_json_payload(&payload_bytes)?
        }
        x if x == CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE => {
            proto_get_trendbars_res_to_json_payload(&payload_bytes)?
        }
        other => {
            return Err(anyhow!(
                "cTrader Protobuf transport: payload type {} not yet supported by the v0.4.5 \
                 migration batch; fall back to JSON-WSS for this endpoint",
                other
            ));
        }
    };

    let envelope_json = serde_json::json!({
        "clientMsgId": client_msg_id,
        "payloadType": payload_type,
        "payload": payload_json,
    });
    serde_json::to_string(&envelope_json)
        .context("failed to serialise translated Protobuf envelope as JSON")
}

fn proto_reconcile_res_to_json_payload(payload: &[u8]) -> Result<serde_json::Value> {
    let res = ProtoOAReconcileRes::parse(payload)
        .map_err(|e| anyhow!("failed to parse ProtoOAReconcileRes: {:?}", e))?;
    let view = res.as_view();
    let positions: Vec<serde_json::Value> = view
        .position()
        .iter()
        .map(|p| {
            // Mirror the JSON-WSS proxy field set that
            // `ctrader_account.rs::PositionPayload` consumes
            // (positionId / tradeData / price / stopLoss / takeProfit /
            // swap / commission). `tradeData` is a required sub-message
            // on the proto, so we emit it unconditionally if present.
            let mut v = serde_json::json!({
                "positionId": p.positionId(),
                "swap": p.swap(),
            });
            let trade_data = p.tradeData();
            let mut td = serde_json::json!({
                "symbolId": trade_data.symbolId(),
                "volume": trade_data.volume(),
                "tradeSide": i32::from(trade_data.tradeSide()),
            });
            if let Some(ts) = trade_data.openTimestamp_opt().into_option() {
                td["openTimestamp"] = serde_json::json!(ts);
            }
            if let Some(label) = trade_data.label_opt().into_option() {
                td["label"] = serde_json::json!(label.to_string());
            }
            if let Some(comment) = trade_data.comment_opt().into_option() {
                td["comment"] = serde_json::json!(comment.to_string());
            }
            v["tradeData"] = td;
            if let Some(price) = p.price_opt().into_option() {
                v["price"] = serde_json::json!(price);
            }
            if let Some(sl) = p.stopLoss_opt().into_option() {
                v["stopLoss"] = serde_json::json!(sl);
            }
            if let Some(tp) = p.takeProfit_opt().into_option() {
                v["takeProfit"] = serde_json::json!(tp);
            }
            if let Some(commission) = p.commission_opt().into_option() {
                v["commission"] = serde_json::json!(commission);
            }
            if let Some(money_digits) = p.moneyDigits_opt().into_option() {
                v["moneyDigits"] = serde_json::json!(money_digits);
            }
            v
        })
        .collect();
    let orders: Vec<serde_json::Value> = view
        .order()
        .iter()
        .map(|o| {
            // Mirror `ctrader_account.rs::OrderPayload`: orderId /
            // tradeData / orderType / limitPrice / stopPrice / stopLoss /
            // takeProfit. `clientOrderId` lives on `ProtoOAOrder` in the
            // proto schema but the JSON-WSS proxy nests it inside
            // `tradeData` — we mirror the proxy convention so the
            // existing `TradeDataPayload.client_order_id` parsing works.
            let mut v = serde_json::json!({
                "orderId": o.orderId(),
                "orderType": i32::from(o.orderType()),
            });
            let trade_data = o.tradeData();
            let mut td = serde_json::json!({
                "symbolId": trade_data.symbolId(),
                "volume": trade_data.volume(),
                "tradeSide": i32::from(trade_data.tradeSide()),
            });
            if let Some(ts) = trade_data.openTimestamp_opt().into_option() {
                td["openTimestamp"] = serde_json::json!(ts);
            }
            if let Some(label) = trade_data.label_opt().into_option() {
                td["label"] = serde_json::json!(label.to_string());
            }
            if let Some(comment) = trade_data.comment_opt().into_option() {
                td["comment"] = serde_json::json!(comment.to_string());
            }
            if let Some(client_order_id) = o.clientOrderId_opt().into_option() {
                td["clientOrderId"] = serde_json::json!(client_order_id.to_string());
            }
            v["tradeData"] = td;
            if let Some(price) = o.limitPrice_opt().into_option() {
                v["limitPrice"] = serde_json::json!(price);
            }
            if let Some(price) = o.stopPrice_opt().into_option() {
                v["stopPrice"] = serde_json::json!(price);
            }
            if let Some(sl) = o.stopLoss_opt().into_option() {
                v["stopLoss"] = serde_json::json!(sl);
            }
            if let Some(tp) = o.takeProfit_opt().into_option() {
                v["takeProfit"] = serde_json::json!(tp);
            }
            v
        })
        .collect();
    Ok(serde_json::json!({
        "ctidTraderAccountId": view.ctidTraderAccountId(),
        "position": positions,
        "order": orders,
    }))
}

fn proto_get_trendbars_res_to_json_payload(payload: &[u8]) -> Result<serde_json::Value> {
    let res = ProtoOAGetTrendbarsRes::parse(payload)
        .map_err(|e| anyhow!("failed to parse ProtoOAGetTrendbarsRes: {:?}", e))?;
    let view = res.as_view();
    let trendbars: Vec<serde_json::Value> = view
        .trendbar()
        .iter()
        .map(|tb| {
            let mut v = serde_json::json!({
                "low": tb.low(),
                "deltaOpen": tb.deltaOpen(),
                "deltaHigh": tb.deltaHigh(),
                "deltaClose": tb.deltaClose(),
                "utcTimestampInMinutes": tb.utcTimestampInMinutes(),
                "volume": tb.volume(),
            });
            if let Some(period) = tb.period_opt().into_option() {
                v["period"] = serde_json::json!(i32::from(period));
            }
            v
        })
        .collect();
    Ok(serde_json::json!({
        "ctidTraderAccountId": view.ctidTraderAccountId(),
        "symbolId": view.symbolId(),
        "period": i32::from(view.period()),
        "timestamp": view.timestamp(),
        "trendbar": trendbars,
    }))
}

/// True when the v0.4.5 Protobuf migration batch supports the given
/// payload type end-to-end (Protobuf encode + decode + JSON envelope
/// translation). Other payload types must fall back to JSON-WSS.
pub fn protobuf_transport_supports_payload_type(request_payload_type: u32) -> bool {
    matches!(
        request_payload_type,
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE | CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE
    )
}

/// Translate a `CTraderOpenApiJsonMessage` into a fully-framed Protobuf
/// request envelope (length prefix + serialised `ProtoMessage`). Returns
/// an error for payload types the v0.4.5 migration batch does not yet
/// cover.
pub fn json_message_to_framed_protobuf(message: &CTraderOpenApiJsonMessage) -> Result<Vec<u8>> {
    let client_msg_id = Some(message.client_msg_id.clone());
    match message.payload_type {
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE => {
            let account_id = message
                .payload
                .get("ctidTraderAccountId")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("reconcile request missing ctidTraderAccountId"))?;
            let return_protection_orders = message
                .payload
                .get("returnProtectionOrders")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            build_reconcile_req_proto(account_id, return_protection_orders, client_msg_id)
        }
        CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE => {
            let account_id = message
                .payload
                .get("ctidTraderAccountId")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("trendbars request missing ctidTraderAccountId"))?;
            let symbol_id = message
                .payload
                .get("symbolId")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("trendbars request missing symbolId"))?;
            let period = message
                .payload
                .get("period")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("trendbars request missing period"))?
                as i32;
            let from_timestamp = message
                .payload
                .get("fromTimestamp")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("trendbars request missing fromTimestamp"))?;
            let to_timestamp = message
                .payload
                .get("toTimestamp")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("trendbars request missing toTimestamp"))?;
            let count = message
                .payload
                .get("count")
                .and_then(serde_json::Value::as_u64)
                .map(|c| c as u32);
            build_get_trendbars_req_proto(
                account_id,
                symbol_id,
                period,
                from_timestamp,
                to_timestamp,
                count,
                client_msg_id,
            )
        }
        other => Err(anyhow!(
            "cTrader Protobuf transport (v0.4.5 batch): payload type {} not yet supported; \
             fall back to JSON-WSS for this endpoint",
            other
        )),
    }
}
