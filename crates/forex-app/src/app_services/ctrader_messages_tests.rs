use super::*;

#[test]
fn application_auth_request_uses_documented_payload_type() {
    let message = build_application_auth_request("client-id", "secret-456", "cm-id-2");

    assert_eq!(message.client_msg_id, "cm-id-2");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_APPLICATION_AUTH_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("clientId")
            .and_then(serde_json::Value::as_str),
        Some("client-id")
    );
    assert_eq!(
        message
            .payload
            .get("clientSecret")
            .and_then(serde_json::Value::as_str),
        Some("secret-456")
    );
}

#[test]
fn account_auth_request_uses_documented_payload_type_and_account_id() {
    let message = build_account_auth_request(7001, "token-123", "account-auth-1");

    assert_eq!(
        message.payload_type,
        CTRADER_OA_ACCOUNT_AUTH_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("accessToken")
            .and_then(serde_json::Value::as_str),
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
        message
            .payload
            .get("accessToken")
            .and_then(serde_json::Value::as_str),
        Some("access-token-123")
    );
}

#[test]
fn trader_request_uses_documented_payload_type_and_account_id() {
    let message = build_trader_request(7001, "trader-1");

    assert_eq!(message.client_msg_id, "trader-1");
    assert_eq!(message.payload_type, CTRADER_OA_TRADER_REQUEST_PAYLOAD_TYPE);
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
}

#[test]
fn reconcile_request_uses_documented_payload_type_and_optional_protection_flag() {
    let message = build_reconcile_request(7001, true, "reconcile-1");

    assert_eq!(message.client_msg_id, "reconcile-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
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
    assert_eq!(
        message.payload_type,
        CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("symbolId")
            .and_then(serde_json::Value::as_array)
            .map(|items| items.len()),
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
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("symbolId")
            .and_then(serde_json::Value::as_i64),
        Some(14)
    );
    assert_eq!(
        message
            .payload
            .get("period")
            .and_then(serde_json::Value::as_i64),
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
    assert_eq!(
        message.payload_type,
        CTRADER_OA_SYMBOLS_LIST_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
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
    assert_eq!(
        message.payload_type,
        CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("symbolId")
            .and_then(serde_json::Value::as_i64),
        Some(9001)
    );
    assert_eq!(
        message
            .payload
            .get("period")
            .and_then(serde_json::Value::as_i64),
        Some(7)
    );
    assert_eq!(
        message
            .payload
            .get("fromTimestamp")
            .and_then(serde_json::Value::as_i64),
        Some(1_700_000_000_000)
    );
    assert_eq!(
        message
            .payload
            .get("toTimestamp")
            .and_then(serde_json::Value::as_i64),
        Some(1_700_000_900_000)
    );
    assert_eq!(
        message
            .payload
            .get("count")
            .and_then(serde_json::Value::as_u64),
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
    assert_eq!(
        message.payload_type,
        CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("symbolId")
            .and_then(serde_json::Value::as_i64),
        Some(9001)
    );
    assert_eq!(
        message
            .payload
            .get("type")
            .and_then(serde_json::Value::as_i64),
        Some(i64::from(CTRADER_QUOTE_TYPE_ASK))
    );
}

#[test]
fn deal_list_request_uses_documented_payload_and_optional_filters() {
    let message = build_deal_list_request(
        &CTraderDealListRequest {
            account_id: 7001,
            from_timestamp_ms: Some(1_700_000_000_000),
            to_timestamp_ms: Some(1_700_000_100_000),
            max_rows: Some(50),
        },
        "deals-1",
    );

    assert_eq!(message.client_msg_id, "deals-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("fromTimestamp")
            .and_then(serde_json::Value::as_i64),
        Some(1_700_000_000_000)
    );
    assert_eq!(
        message
            .payload
            .get("toTimestamp")
            .and_then(serde_json::Value::as_i64),
        Some(1_700_000_100_000)
    );
    assert_eq!(
        message
            .payload
            .get("maxRows")
            .and_then(serde_json::Value::as_i64),
        Some(50)
    );
}

#[test]
fn new_order_request_uses_documented_trade_payload() {
    let message = build_new_order_request(
        &CTraderNewOrderRequest {
            account_id: 7001,
            symbol_id: 14,
            order_type: CTraderOrderType::Market,
            trade_side: CTraderTradeSide::Buy,
            volume: 1500,
            limit_price: None,
            stop_price: None,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: Some(1.095),
            take_profit: Some(1.105),
            comment: Some("bot-entry".to_string()),
            base_slippage_price: None,
            slippage_in_points: Some(15),
            label: Some("trend".to_string()),
            position_id: None,
            client_order_id: Some("client-order-1".to_string()),
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: Some(false),
            trailing_stop_loss: Some(true),
            stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
        },
        "new-order-1",
    );

    assert_eq!(message.client_msg_id, "new-order-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("symbolId")
            .and_then(serde_json::Value::as_i64),
        Some(14)
    );
    assert_eq!(
        message
            .payload
            .get("orderType")
            .and_then(serde_json::Value::as_i64),
        Some(1)
    );
    assert_eq!(
        message
            .payload
            .get("tradeSide")
            .and_then(serde_json::Value::as_i64),
        Some(1)
    );
    assert_eq!(
        message
            .payload
            .get("volume")
            .and_then(serde_json::Value::as_i64),
        Some(1500)
    );
    assert_eq!(
        message
            .payload
            .get("timeInForce")
            .and_then(serde_json::Value::as_i64),
        Some(3)
    );
    assert_eq!(
        message
            .payload
            .get("clientOrderId")
            .and_then(serde_json::Value::as_str),
        Some("client-order-1")
    );
}

#[test]
fn amend_order_request_uses_documented_identifiers_and_optional_fields() {
    let message = build_amend_order_request(
        &CTraderAmendOrderRequest {
            account_id: 7001,
            order_id: 8001,
            volume: Some(1200),
            limit_price: Some(1.0985),
            stop_price: None,
            expiration_timestamp_ms: None,
            stop_loss: Some(1.0940),
            take_profit: Some(1.1060),
            slippage_in_points: Some(12),
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: Some(false),
            trailing_stop_loss: Some(true),
            stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
        },
        "amend-order-1",
    );

    assert_eq!(message.client_msg_id, "amend-order-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_AMEND_ORDER_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64),
        Some(7001)
    );
    assert_eq!(
        message
            .payload
            .get("orderId")
            .and_then(serde_json::Value::as_i64),
        Some(8001)
    );
    assert_eq!(
        message
            .payload
            .get("limitPrice")
            .and_then(serde_json::Value::as_f64),
        Some(1.0985)
    );
}

#[test]
fn cancel_order_request_uses_documented_order_id() {
    let message = build_cancel_order_request(
        &CTraderCancelOrderRequest {
            account_id: 7001,
            order_id: 8001,
        },
        "cancel-order-1",
    );

    assert_eq!(message.client_msg_id, "cancel-order-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_CANCEL_ORDER_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("orderId")
            .and_then(serde_json::Value::as_i64),
        Some(8001)
    );
}

#[test]
fn close_position_request_uses_documented_position_id_and_volume() {
    let message = build_close_position_request(
        &CTraderClosePositionRequest {
            account_id: 7001,
            position_id: 9001,
            volume: 500,
        },
        "close-position-1",
    );

    assert_eq!(message.client_msg_id, "close-position-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_CLOSE_POSITION_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        message
            .payload
            .get("positionId")
            .and_then(serde_json::Value::as_i64),
        Some(9001)
    );
    assert_eq!(
        message
            .payload
            .get("volume")
            .and_then(serde_json::Value::as_i64),
        Some(500)
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

#[test]
fn ctrader_error_payload_parts_separates_code_and_message() {
    let (code, message) = parse_ctrader_error_payload_parts(&serde_json::json!({
        "errorCode": "OA_AUTH_TOKEN_EXPIRED",
        "description": "OAuth access token has expired"
    }))
    .expect("error payload should parse");

    assert_eq!(code, "OA_AUTH_TOKEN_EXPIRED");
    assert_eq!(
        message,
        "OA_AUTH_TOKEN_EXPIRED: OAuth access token has expired"
    );
}

#[test]
fn auth_token_error_classifier_matches_known_codes() {
    for code in [
        "OA_AUTH_TOKEN_EXPIRED",
        "ACCESS_TOKEN_EXPIRED",
        "TOKEN_EXPIRED",
        "INVALID_TOKEN",
        "INVALID_ACCESS_TOKEN",
        "CH_ACCESS_TOKEN_INVALID",
        "CH_ACCESS_TOKEN_EXPIRED",
    ] {
        assert!(
            is_ctrader_auth_token_error(code),
            "expected {code} to be classified as a token-expired error"
        );
    }
}

#[test]
fn auth_token_error_classifier_rejects_unrelated_codes() {
    for code in [
        "ACCOUNT_NOT_AUTHORIZED",
        "INSUFFICIENT_FUNDS",
        "MARKET_CLOSED",
        "INVALID_VOLUME",
        "",
    ] {
        assert!(
            !is_ctrader_auth_token_error(code),
            "expected {code} NOT to be classified as a token-expired error"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// v0.4.5 cTrader native Protobuf-over-TCP transport tests.
//
// Covers the wire-layer codec + the transport-selector env-var contract.
// No network calls; live broker fixture is gated behind `#[ignore]`.
// See `docs/audits/research/ctrader_api_full_reference.md` §10 item #3
// for the migration scope (reconcile + historical bars).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn protobuf_transport_length_prefix_round_trips_for_reconcile_request() {
    use crate::app_services::ctrader_proto_messages::{
        build_reconcile_req_proto, parse_proto_message, read_length_prefixed_frame,
    };

    // Build a framed reconcile request through the public Protobuf
    // builder used by the transport (matches what
    // `ProductionCTraderOpenApiProtobufTransport::send_sequence_protobuf`
    // writes to the wire).
    let framed = build_reconcile_req_proto(
        9_999_001,
        true,
        Some("reconcile-codec-roundtrip-1".to_string()),
    )
    .expect("framed reconcile request must build");

    // The wire format is 4-byte big-endian length prefix + serialised
    // ProtoMessage envelope (spec §1.5). Round-trip through the reader
    // codec to recover the envelope.
    let mut cursor = std::io::Cursor::new(framed.clone());
    let envelope_bytes =
        read_length_prefixed_frame(&mut cursor).expect("frame reader must decode the prefix");

    // The length prefix is 4 bytes — payload size = total - 4.
    assert_eq!(
        envelope_bytes.len(),
        framed.len() - 4,
        "decoded envelope size must equal framed size minus the 4-byte length prefix"
    );

    // Parse the envelope and verify the payloadType is reconcile-request
    // (2124) and the client_msg_id round-trips.
    let envelope =
        parse_proto_message(&envelope_bytes).expect("envelope bytes must parse as ProtoMessage");
    let view = envelope.as_view();
    assert_eq!(
        view.payloadType(),
        CTRADER_OA_RECONCILE_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        view.clientMsgId().to_string(),
        "reconcile-codec-roundtrip-1"
    );
}

#[test]
fn transport_selector_picks_json_wss_by_default() {
    // The env var must be unset (or unrecognised) for the default
    // JSON-WSS path. Save & restore so we don't leak state between
    // tests if they share the same process.
    let prior = std::env::var(CTRADER_TRANSPORT_ENV_VAR).ok();
    // Safety: tests are single-threaded by default within the cargo
    // test harness for the same binary; we restore on every exit path
    // below to avoid cross-test bleed.
    unsafe {
        std::env::remove_var(CTRADER_TRANSPORT_ENV_VAR);
    }
    let selected = select_ctrader_transport_from_env();
    if let Some(p) = prior {
        unsafe {
            std::env::set_var(CTRADER_TRANSPORT_ENV_VAR, p);
        }
    }
    assert_eq!(selected, CTraderTransportKind::JsonWss);
    assert_eq!(selected.label(), "json_wss");
}

#[test]
fn transport_selector_picks_protobuf_when_env_set() {
    let prior = std::env::var(CTRADER_TRANSPORT_ENV_VAR).ok();
    unsafe {
        std::env::set_var(CTRADER_TRANSPORT_ENV_VAR, "protobuf");
    }
    let selected = select_ctrader_transport_from_env();
    if let Some(p) = prior {
        unsafe {
            std::env::set_var(CTRADER_TRANSPORT_ENV_VAR, p);
        }
    } else {
        unsafe {
            std::env::remove_var(CTRADER_TRANSPORT_ENV_VAR);
        }
    }
    assert_eq!(selected, CTraderTransportKind::Protobuf);
    assert_eq!(selected.label(), "protobuf");
}

#[test]
#[ignore = "needs cTrader broker fixture / live demo credentials"]
fn protobuf_transport_full_reconcile_against_live_demo() {
    // Integration check for the end-to-end Protobuf-over-TCP path
    // against the live demo proxy on port 5035. Marked `#[ignore]`
    // because it requires (a) live network access to
    // `demo.ctraderapi.com:5035`, (b) the `ctrader-protobuf-streaming`
    // Cargo feature to be enabled at build time, and (c) a valid
    // OAuth account auth pre-amble that the test does NOT perform —
    // it only exercises the framing + reconcile parsing once a real
    // session is open.
    //
    // To run manually:
    //   cargo test --no-default-features \
    //              --features ctrader-protobuf-streaming \
    //              -- --ignored protobuf_transport_full_reconcile_against_live_demo
    //
    // No synthetic broker payloads — this test deliberately does
    // nothing under default `cargo test` so we never fabricate cTrader
    // responses just to satisfy a unit-test green tick. The codec
    // round-trip above plus the transport selector tests cover the
    // pure-Rust surface; a live fixture is needed to verify the wire.
}
