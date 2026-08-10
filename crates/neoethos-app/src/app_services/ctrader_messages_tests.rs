use super::*;

#[test]
fn parse_open_api_envelope_tolerates_heartbeat_without_client_msg_id() {
    // v0.4.13 regression test — the cTrader Open API server emits
    // `ProtoHeartbeatEvent` frames (payloadType 51) every ~30 s with
    // neither `clientMsgId` nor `payload` populated. Before the
    // `#[serde(default)]` annotations on `CTraderOpenApiJsonMessage`
    // those frames blew up the WSS read loop with the generic
    // "failed to parse cTrader JSON envelope" error and the wizard's
    // account-discovery leg aborted on the first heartbeat that
    // raced the application-auth response.
    let heartbeat = r#"{"payloadType":51}"#;
    let envelope = parse_open_api_envelope(heartbeat).expect("heartbeat must parse");
    assert_eq!(envelope.payload_type, CTRADER_OA_HEARTBEAT_PAYLOAD_TYPE);
    assert_eq!(envelope.client_msg_id, "");
    assert!(envelope.payload.is_null());
}

#[test]
fn parse_open_api_envelope_error_includes_response_head_for_diagnosis() {
    // v0.4.13 — the error context now includes a 200-char head of
    // the offending body so a future schema drift is debuggable from
    // the wizard's status surface alone (no extra logs required).
    let malformed = "this is not JSON at all";
    let err = parse_open_api_envelope(malformed).unwrap_err().to_string();
    assert!(err.contains("len=23"), "len missing from error: {err}");
    assert!(err.contains("head="), "head missing from error: {err}");
}

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
fn unsubscribe_requests_use_documented_payload_types() {
    // 2026-08-09 (D2): the live-trendbar half went with
    // `build_unsubscribe_live_trendbar_request`. Spot unsubscribe stays — that
    // builder is a KEEP-and-WIRE item (the streamer subscribes and never
    // unsubscribes).
    let spots = build_unsubscribe_spots_request(7001, &[14], "spots-off-1");

    assert_eq!(
        spots.payload_type,
        CTRADER_OA_UNSUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
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
// cTrader transport-selector tests. No network calls.
//
// (2026-08-08 dead-code purge: the ctrader_proto_messages length-prefix
// framing codec and its pinning test were deleted — the module had no
// non-test consumer; the live transport is JSON-WSS.)
// ─────────────────────────────────────────────────────────────────────────────

// (2026-08-09 batch D2b: the two transport-selector tests were deleted with the
// selector itself. They pinned `NEOETHOS_BOT_CTRADER_TRANSPORT=protobuf` to a
// `CTraderTransportKind::Protobuf` whose codec batch D2 had already removed —
// a green assertion about a wire format this binary can no longer speak.)

// ─────────────────────────────────────────────────────────────────────────
// 2026-06-10 — full ProtoOAPayloadType API-completeness pass. Each builder
// is pinned to its documented payloadType and field names so a future wire-
// format drift fails loudly here instead of silently at the broker.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn amend_position_sltp_request_uses_documented_payload_and_fields() {
    let request = CTraderAmendPositionSltpRequest {
        account_id: 7001,
        position_id: 42,
        stop_loss: Some(1.0850),
        take_profit: Some(1.0950),
        guaranteed_stop_loss: Some(true),
        trailing_stop_loss: Some(false),
        stop_loss_trigger_method: Some(CTraderOrderTriggerMethod::Opposite),
    };
    let message = build_amend_position_sltp_request(&request, "amend-pos-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(message.payload_type, 2110);
    assert_eq!(
        message.payload.get("positionId").and_then(Value::as_i64),
        Some(42)
    );
    assert_eq!(
        message.payload.get("stopLoss").and_then(Value::as_f64),
        Some(1.0850)
    );
    assert_eq!(
        message.payload.get("takeProfit").and_then(Value::as_f64),
        Some(1.0950)
    );
    assert_eq!(
        message
            .payload
            .get("guaranteedStopLoss")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        message
            .payload
            .get("stopLossTriggerMethod")
            .and_then(Value::as_i64),
        Some(CTRADER_ORDER_TRIGGER_METHOD_OPPOSITE as i64)
    );
}

#[test]
fn amend_position_sltp_omits_untouched_fields() {
    // Only the SL is being moved (e.g. trail) — TP and the flags must NOT be
    // sent so the broker leaves them as-is.
    let request = CTraderAmendPositionSltpRequest {
        account_id: 1,
        position_id: 2,
        stop_loss: Some(1.2345),
        take_profit: None,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_loss_trigger_method: None,
    };
    let message = build_amend_position_sltp_request(&request, "amend-pos-2");
    assert!(message.payload.get("stopLoss").is_some());
    assert!(message.payload.get("takeProfit").is_none());
    assert!(message.payload.get("trailingStopLoss").is_none());
    assert!(message.payload.get("stopLossTriggerMethod").is_none());
}

#[test]
fn amend_position_sltp_response_is_an_execution_event() {
    // Like the other trade actions, the broker answers a 2110 with a
    // ProtoOAExecutionEvent (2126) — the matcher and the expected-response
    // map must both agree, or the execution backend hangs waiting for a 2111.
    assert_eq!(
        expected_response_payload_type(CTRADER_OA_AMEND_POSITION_SLTP_REQUEST_PAYLOAD_TYPE)
            .unwrap(),
        CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE
    );
    let request = build_amend_position_sltp_request(
        &CTraderAmendPositionSltpRequest {
            account_id: 1,
            position_id: 2,
            stop_loss: Some(1.0),
            take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_loss_trigger_method: None,
        },
        "amend-pos-3",
    );
    let exec_event = CTraderOpenApiJsonMessage {
        client_msg_id: "amend-pos-3".to_string(),
        payload_type: CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE,
        payload: Value::Null,
    };
    assert!(is_matching_open_api_response(
        &exec_event,
        &request,
        CTRADER_OA_EXECUTION_EVENT_PAYLOAD_TYPE
    ));
}

#[test]
fn version_request_carries_no_account() {
    let message = build_version_request("ver-1");
    assert_eq!(message.payload_type, CTRADER_OA_VERSION_REQUEST_PAYLOAD_TYPE);
    assert_eq!(message.payload_type, 2104);
    assert_eq!(
        expected_response_payload_type(CTRADER_OA_VERSION_REQUEST_PAYLOAD_TYPE).unwrap(),
        CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE
    );
}

#[test]
fn expected_margin_request_uses_documented_payload_and_fields() {
    let message = build_expected_margin_request(7001, 1, &[10_000_000, 20_000_000], "em-1");
    assert_eq!(
        message.payload_type,
        CTRADER_OA_EXPECTED_MARGIN_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(message.payload_type, 2139);
    assert_eq!(
        message.payload.get("symbolId").and_then(Value::as_i64),
        Some(1)
    );
    let volumes = message
        .payload
        .get("volume")
        .and_then(Value::as_array)
        .expect("volume array");
    assert_eq!(volumes.len(), 2);
    assert_eq!(volumes[0].as_i64(), Some(10_000_000));
    assert_eq!(
        expected_response_payload_type(CTRADER_OA_EXPECTED_MARGIN_REQUEST_PAYLOAD_TYPE).unwrap(),
        CTRADER_OA_EXPECTED_MARGIN_RESPONSE_PAYLOAD_TYPE
    );
}

#[test]
fn order_list_and_cash_flow_history_carry_time_window() {
    let orders = build_order_list_request(7001, 1_000, 2_000, "ol-1");
    assert_eq!(orders.payload_type, CTRADER_OA_ORDER_LIST_REQUEST_PAYLOAD_TYPE);
    assert_eq!(orders.payload_type, 2175);
    assert_eq!(
        orders.payload.get("fromTimestamp").and_then(Value::as_i64),
        Some(1_000)
    );
    assert_eq!(
        orders.payload.get("toTimestamp").and_then(Value::as_i64),
        Some(2_000)
    );
    assert_eq!(
        expected_response_payload_type(CTRADER_OA_ORDER_LIST_REQUEST_PAYLOAD_TYPE).unwrap(),
        CTRADER_OA_ORDER_LIST_RESPONSE_PAYLOAD_TYPE
    );

    let cash = build_cash_flow_history_list_request(7001, 3_000, 4_000, "cf-1");
    assert_eq!(
        cash.payload_type,
        CTRADER_OA_CASH_FLOW_HISTORY_LIST_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(cash.payload_type, 2143);
    assert_eq!(
        cash.payload.get("toTimestamp").and_then(Value::as_i64),
        Some(4_000)
    );
}

#[test]
fn ctid_profile_request_carries_access_token() {
    // 2026-08-09 (D2): the `build_refresh_token_request` half was deleted with
    // the builder — OAuth refresh goes over HTTPS (`ctrader_live_auth.rs:845`),
    // never over the Open API socket.
    let profile = build_get_ctid_profile_by_token_request("access-xyz", "pr-1");
    assert_eq!(
        profile.payload_type,
        CTRADER_OA_GET_CTID_PROFILE_BY_TOKEN_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        profile.payload.get("accessToken").and_then(Value::as_str),
        Some("access-xyz")
    );
}

#[test]
fn margin_call_request_uses_documented_payload() {
    // 2026-08-09 (D2): the depth-quote half went with
    // `build_{,un}subscribe_depth_quotes_request`.
    // 2026-08-09 (#238): `build_margin_call_list_request` is no longer a WIRE
    // item — `broker_api::fetch_margin_status_blocking` calls it and
    // `app_services::margin_call` polls that. The comment that used to say
    // "nothing reads yet" was true this morning and is false now.
    let mc = build_margin_call_list_request(7001, "mc-1");
    assert_eq!(
        mc.payload_type,
        CTRADER_OA_MARGIN_CALL_LIST_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(
        mc.payload
            .get("ctidTraderAccountId")
            .and_then(Value::as_i64),
        Some(7001)
    );
}

#[test]
fn margin_call_list_response_parses_thresholds_and_reports_the_tightest() {
    // The tightest threshold is the LARGEST percentage, because margin level
    // falls toward a call. Getting this backwards would halt on the loosest
    // threshold only — i.e. far too late.
    let json = r#"{
        "payloadType": 2168,
        "payload": {
            "ctidTraderAccountId": 7001,
            "marginCall": [
                {"marginCallType":"MARGIN_CALL_THRESHOLD_1","marginLevelThreshold":60.0,
                 "utcLastUpdateTimestamp":1700000000000},
                {"marginCallType":"MARGIN_CALL_THRESHOLD_2","marginLevelThreshold":100.0},
                {"marginCallType":"MARGIN_CALL_THRESHOLD_3","marginLevelThreshold":80.0}
            ]
        }
    }"#;
    let snap = parse_margin_call_list_response(json).expect("must parse");
    assert_eq!(snap.account_id, 7001);
    assert_eq!(snap.thresholds.len(), 3);
    assert_eq!(snap.unusable_rows, 0);
    assert!(snap.unusable_reasons.is_empty());
    assert_eq!(snap.tightest_threshold_pct(), Some(100.0));
    assert_eq!(
        snap.thresholds[0].utc_last_update_timestamp_ms,
        Some(1700000000000)
    );
}

#[test]
fn margin_call_list_response_counts_rows_it_cannot_use_instead_of_dropping_them() {
    // NO SILENT DROPS. A row the broker sent whose threshold we could not read
    // must be COUNTED, with a reason — otherwise the watchdog silently behaves
    // as though the broker configured fewer thresholds than it did, which is
    // the optimistic (dangerous) direction.
    let json = r#"{
        "payloadType": 2168,
        "payload": {
            "ctidTraderAccountId": 7001,
            "marginCall": [
                {"marginCallType":"MARGIN_CALL_THRESHOLD_1","marginLevelThreshold":50.0},
                {"marginCallType":"MARGIN_CALL_THRESHOLD_2"},
                {"marginCallType":"MARGIN_CALL_THRESHOLD_3","marginLevelThreshold":0.0}
            ]
        }
    }"#;
    let snap = parse_margin_call_list_response(json).expect("must parse");
    assert_eq!(snap.thresholds.len(), 1);
    assert_eq!(snap.unusable_rows, 2);
    assert_eq!(snap.unusable_reasons.len(), 2);
    assert_eq!(snap.tightest_threshold_pct(), Some(50.0));
}

#[test]
fn margin_call_list_response_keeps_a_threshold_whose_type_label_is_unreadable() {
    // A missing/odd `marginCallType` degrades the LABEL, not the threshold.
    // Discarding a real threshold because its name did not parse would be a
    // silent loss of protection.
    let json = r#"{
        "payloadType": 2168,
        "payload": {
            "ctidTraderAccountId": 7001,
            "marginCall": [
                {"marginLevelThreshold": 120.0},
                {"marginCallType": 2, "marginLevelThreshold": 90.0}
            ]
        }
    }"#;
    let snap = parse_margin_call_list_response(json).expect("must parse");
    assert_eq!(snap.thresholds.len(), 2);
    assert_eq!(snap.unusable_rows, 0);
    assert_eq!(snap.thresholds[0].margin_call_type, "UNKNOWN");
    assert_eq!(snap.thresholds[1].margin_call_type, "MARGIN_CALL_TYPE_2");
    // The unreadable label is still REPORTED even though nothing was dropped.
    assert_eq!(snap.unusable_reasons.len(), 1);
    assert_eq!(snap.tightest_threshold_pct(), Some(120.0));
}

#[test]
fn margin_call_list_response_with_no_rows_has_no_threshold_to_compare_against() {
    // An account with no configured margin call yields `None`, which the
    // watchdog logs loudly and does NOT treat as a breach.
    let json = r#"{"payloadType":2168,"payload":{"ctidTraderAccountId":7001}}"#;
    let snap = parse_margin_call_list_response(json).expect("must parse");
    assert!(snap.thresholds.is_empty());
    assert_eq!(snap.unusable_rows, 0);
    assert_eq!(snap.tightest_threshold_pct(), None);
}

#[test]
fn margin_call_list_response_rejects_the_wrong_payload_type() {
    let json = r#"{"payloadType":2122,"payload":{"ctidTraderAccountId":7001}}"#;
    let err = parse_margin_call_list_response(json)
        .expect_err("a non-2168 envelope must not be read as a margin-call list")
        .to_string();
    assert!(err.contains("2122"), "{err}");
}
