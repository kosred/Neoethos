use super::*;



#[test]
fn authorize_url_uses_selected_callback_port() {
    let config = CTraderLoopbackConfig::new(43001, vec![43002, 43003], "/callback");

    let authorize_url = build_authorize_url(
        "client-id",
        "http://127.0.0.1:43001/callback",
        43002,
        "trading",
    )
    .expect("authorize url should build");

    assert!(authorize_url.contains("client_id=client-id"));
    assert!(authorize_url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A43002%2Fcallback"));
    assert_eq!(config.allowed_ports(), &[43001, 43002, 43003]);
}

#[test]
fn authorize_url_rejects_malformed_redirect_uri() {
    let err = build_authorize_url("client-id", "not-a-valid-redirect", 43002, "trading")
        .expect_err("malformed redirect must fail");

    assert!(err.to_string().contains("redirect URI"));
}

#[test]
fn default_loopback_config_rejects_non_loopback_redirect_host() {
    let err = build_default_loopback_config("http://example.com:43001/callback")
        .expect_err("non-loopback redirect host must fail");

    assert!(err.to_string().contains("loopback"));
}

#[test]
fn default_loopback_config_preserves_localhost_for_listener_binding() {
    let config = build_default_loopback_config("http://localhost:43001/callback")
        .expect("localhost loopback redirect should build");

    assert_eq!(config.bind_host(), "localhost");
    assert_eq!(config.allowed_ports(), &[43001, 43002, 43003]);
    assert_eq!(config.callback_path(), "/callback");
}

#[test]
fn default_loopback_config_accepts_ipv6_loopback_redirect_host() {
    let config = build_default_loopback_config("http://[::1]:43001/callback")
        .expect("IPv6 loopback redirect should build");

    assert_eq!(config.bind_host(), "::1");
    assert_eq!(config.allowed_ports(), &[43001, 43002, 43003]);
    assert_eq!(config.callback_path(), "/callback");
}

#[test]
fn authorize_url_rewrites_ipv6_loopback_port() {
    let authorize_url =
        build_authorize_url("client-id", "http://[::1]:43001/callback", 43002, "trading")
            .expect("IPv6 authorize url should build");

    assert!(
        authorize_url.contains("redirect_uri=http%3A%2F%2F%5B%3A%3A1%5D%3A43002%2Fcallback")
    );
}

#[test]
fn callback_capture_times_out_when_browser_never_redirects() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("test listener should bind");
    let backend = ProductionCTraderLiveAuthBackend;

    let err = backend
        .capture_authorization_code_with_timeout(
            listener,
            "/callback",
            std::time::Duration::from_millis(10),
        )
        .expect_err("missing browser callback should time out");

    assert!(err.to_string().contains("timed out waiting"));
}

#[test]
fn callback_parser_accepts_expected_path_and_extracts_code() {
    let parsed = parse_callback_request("/callback?code=auth-code-123", "/callback")
        .expect("callback should parse");

    assert_eq!(parsed.authorization_code, "auth-code-123");
}

#[test]
fn callback_parser_decodes_percent_encoded_authorization_code() {
    let parsed = parse_callback_request("/callback?code=auth%2Bcode%252F123", "/callback")
        .expect("callback should decode");

    assert_eq!(parsed.authorization_code, "auth+code%2F123");
}

#[test]
fn callback_parser_surfaces_ctrader_denial_errors() {
    let err = parse_callback_request(
        "/callback?error=access_denied&error_description=operator%20cancelled",
        "/callback",
    )
    .expect_err("denied callback should fail");

    assert!(
        err.to_string()
            .contains("cTrader authorization denied: access_denied")
    );
    assert!(err.to_string().contains("operator cancelled"));
}

#[test]
fn token_exchange_request_uses_documented_query_parameters() {
    let url = build_token_exchange_url(
        "https://openapi.ctrader.com",
        "authorization_code",
        "auth-code-123",
        "http://127.0.0.1:43001/callback",
        "client-id",
        "secret-456",
    );

    assert_eq!(
        url,
        "https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=auth-code-123&redirect_uri=http%3A%2F%2F127.0.0.1%3A43001%2Fcallback&client_id=client-id&client_secret=secret-456"
    );
}

#[test]
fn refresh_token_request_uses_documented_query_parameters() {
    let url = build_refresh_token_exchange_url(
        "https://openapi.ctrader.com",
        "refresh-token-123",
        "client-id",
        "secret-456",
    );

    assert_eq!(
        url,
        "https://openapi.ctrader.com/apps/token?grant_type=refresh_token&refresh_token=refresh-token-123&client_id=client-id&client_secret=secret-456"
    );
}

#[test]
fn refreshed_token_response_parses_new_token_values() {
    let response = serde_json::json!({
        "accessToken": "new-access",
        "refreshToken": "new-refresh",
        "tokenType": "bearer",
        "expiresIn": 2628000,
        "errorCode": null,
        "description": null
    });

    let bundle = parse_token_bundle_response(&response.to_string(), "trading", 1_774_147_200)
        .expect("refresh response should parse");

    assert_eq!(bundle.access_token, "new-access");
    assert_eq!(bundle.refresh_token, "new-refresh");
    assert_eq!(bundle.token_type, "bearer");
    assert_eq!(bundle.expires_in, 2_628_000);
    assert_eq!(bundle.scope, "trading");
    assert_eq!(bundle.created_at_unix, 1_774_147_200);
}

#[test]
fn application_auth_request_uses_documented_payload_type() {
    let message = build_application_auth_json("client-id", "secret-456", "cm-id-2");

    assert_eq!(message.client_msg_id, "cm-id-2");
    assert_eq!(message.payload_type, 2100);
    assert_eq!(
        message
            .payload
            .get("clientId")
            .and_then(|value| value.as_str()),
        Some("client-id")
    );
    assert_eq!(
        message
            .payload
            .get("clientSecret")
            .and_then(|value| value.as_str()),
        Some("secret-456")
    );
}

#[test]
fn account_discovery_request_uses_documented_json_payload_type() {
    let request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "access-token-123".to_string(),
        environment: CTraderEnvironment::Demo,
    };

    let message = build_account_list_by_access_token_json(&request, "cm-id-1");

    assert_eq!(message.client_msg_id, "cm-id-1");
    assert_eq!(message.payload_type, 2149);
    assert_eq!(
        message
            .payload
            .get("accessToken")
            .and_then(|value| value.as_str()),
        Some("access-token-123")
    );
}

#[test]
fn account_list_response_parses_discovered_accounts() {
    let response = serde_json::json!({
        "clientMsgId": "server-msg-1",
        "payloadType": 2150,
        "payload": {
            "accessToken": "access-token-123",
            "permissionScope": "SCOPE_TRADE",
            "ctidTraderAccount": [
                {
                    "ctidTraderAccountId": 101,
                    "isLive": true,
                    "traderLogin": 500101,
                    "brokerTitleShort": "Broker A"
                },
                {
                    "ctidTraderAccountId": 202,
                    "isLive": false,
                    "traderLogin": 500202,
                    "brokerTitleShort": "Broker B"
                }
            ]
        }
    });

    let result = parse_account_list_by_access_token_json(&response.to_string())
        .expect("account list response should parse");

    assert_eq!(result.access_token, "access-token-123");
    assert_eq!(result.permission_scope, "SCOPE_TRADE");
    assert_eq!(result.accounts.len(), 2);
    assert_eq!(result.accounts[0].account_id, "101");
    assert_eq!(result.accounts[0].broker_title, "Broker A");
    assert_eq!(result.accounts[0].trader_login, Some(500101));
    assert_eq!(result.accounts[0].is_live, Some(true));
    assert_eq!(result.accounts[1].is_live, Some(false));
}

#[test]
fn account_discovery_request_can_be_built_for_live_and_demo_environments() {
    let live_request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "live-token".to_string(),
        environment: CTraderEnvironment::Live,
    };
    let demo_request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "demo-token".to_string(),
        environment: CTraderEnvironment::Demo,
    };

    assert_eq!(live_request.endpoint_host(), "live.ctraderapi.com");
    assert_eq!(demo_request.endpoint_host(), "demo.ctraderapi.com");
}

#[test]
fn account_discovery_exchange_sends_app_auth_then_account_list() {
    let transport = StubCTraderOpenApiTransport::with_responses(vec![
        Ok(serde_json::json!({
            "clientMsgId": "app-auth-1",
            "payloadType": 2101,
            "payload": {}
        })
        .to_string()),
        Ok(serde_json::json!({
            "clientMsgId": "account-list-1",
            "payloadType": 2150,
            "payload": {
                "accessToken": "access-token-123",
                "permissionScope": "SCOPE_TRADE",
                "ctidTraderAccount": [
                    {
                        "ctidTraderAccountId": 101,
                        "isLive": true,
                        "traderLogin": 500101,
                        "brokerTitleShort": "Broker A"
                    }
                ]
            }
        })
        .to_string()),
    ]);
    let request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "access-token-123".to_string(),
        environment: CTraderEnvironment::Live,
    };

    let result = perform_account_discovery_with_transport(&transport, &request)
        .expect("account discovery should succeed");
    let sent = transport.sent_messages();

    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].payload_type, 2100);
    assert_eq!(sent[1].payload_type, 2149);
    assert_eq!(result.accounts.len(), 1);
    assert_eq!(result.accounts[0].account_id, "101");
    assert_eq!(result.accounts[0].is_live, Some(true));
}

#[test]
fn account_discovery_exchange_surfaces_ctrader_error_payload() {
    let transport = StubCTraderOpenApiTransport::with_responses(vec![Ok(serde_json::json!({
        "clientMsgId": "app-auth-1",
        "payloadType": 2142,
        "payload": {
            "errorCode": "INVALID_ACCESS_TOKEN",
            "description": "Access token is expired"
        }
    })
    .to_string())]);
    let request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "access-token-123".to_string(),
        environment: CTraderEnvironment::Live,
    };

    let err = perform_account_discovery_with_transport(&transport, &request)
        .expect_err("error payload should fail the exchange");

    assert!(err.to_string().contains("INVALID_ACCESS_TOKEN"));
}

#[test]
fn account_discovery_exchange_surfaces_account_list_error_payload() {
    let transport = StubCTraderOpenApiTransport::with_responses(vec![
        Ok(serde_json::json!({
            "clientMsgId": "app-auth-1",
            "payloadType": 2101,
            "payload": {}
        })
        .to_string()),
        Ok(serde_json::json!({
            "clientMsgId": "account-list-1",
            "payloadType": 2142,
            "payload": {
                "errorCode": "ACCOUNTS_LIST_FAILED",
                "description": "Access token has no linked accounts"
            }
        })
        .to_string()),
    ]);
    let request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "access-token-123".to_string(),
        environment: CTraderEnvironment::Live,
    };

    let err = perform_account_discovery_with_transport(&transport, &request)
        .expect_err("account list error payload should fail the exchange");

    assert!(err.to_string().contains("cTrader account list failed"));
    assert!(err.to_string().contains("ACCOUNTS_LIST_FAILED"));
}

#[test]
fn account_discovery_exchange_ignores_unrelated_frames_until_expected_response() {
    let transport = StubCTraderOpenApiTransport::with_responses(vec![
        Ok(serde_json::json!({
            "clientMsgId": "noise-1",
            "payloadType": 9999,
            "payload": {}
        })
        .to_string()),
        Ok(serde_json::json!({
            "clientMsgId": "app-auth-1",
            "payloadType": 2101,
            "payload": {}
        })
        .to_string()),
        Ok(serde_json::json!({
            "clientMsgId": "noise-2",
            "payloadType": 9998,
            "payload": {}
        })
        .to_string()),
        Ok(serde_json::json!({
            "clientMsgId": "account-list-1",
            "payloadType": 2150,
            "payload": {
                "accessToken": "access-token-123",
                "permissionScope": "SCOPE_TRADE",
                "ctidTraderAccount": [
                    {
                        "ctidTraderAccountId": 101,
                        "isLive": true,
                        "traderLogin": 500101,
                        "brokerTitleShort": "Broker A"
                    }
                ]
            }
        })
        .to_string()),
    ]);
    let request = CTraderAccountDiscoveryRequest {
        client_id: "client-id".to_string(),
        client_secret: "secret-456".to_string(),
        access_token: "access-token-123".to_string(),
        environment: CTraderEnvironment::Live,
    };

    let result = perform_account_discovery_with_transport(&transport, &request)
        .expect("account discovery should ignore unrelated frames");

    assert_eq!(result.accounts.len(), 1);
    assert_eq!(result.accounts[0].account_id, "101");
}
