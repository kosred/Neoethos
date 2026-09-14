//! Integration tests for the end-to-end cTrader API message flows.
//! All tests use stub transports — no live credentials required.
//!
//! TODO(real-data): every JSON payload in this file is a hand-crafted
//! string (e.g. `r#"{"clientMsgId":"app-auth-1","payloadType":2101,…}"#`).
//! Replace each helper with a captured cTrader response recorded from
//! the demo/live Open API endpoint for the corresponding payload type
//! so the parser is asserted against real broker bytes — including
//! optional fields and version-shift padding — rather than a model of
//! what we think the response looks like.

#[cfg(test)]
mod ctrader_integration_tests {
    use crate::app_services::ctrader_data::{
        CTraderSymbolInfo, CTraderSymbolLookupRequest, parse_trendbars_response,
        resolve_symbol_with_transport,
    };
    use crate::app_services::ctrader_live_auth::{
        CTraderAccountDiscoveryRequest, CTraderEnvironment,
        perform_account_discovery_with_transport,
    };
    use crate::app_services::ctrader_messages::{
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE,
        CTraderOpenApiJsonMessage, CTraderOpenApiTransport, build_application_auth_request,
    };
    use anyhow::{Result, anyhow};
    use std::sync::Mutex;

    // ─── Shared stub transport ──────────────────────────────────────────────

    struct SequenceTransport {
        sent: Mutex<Vec<CTraderOpenApiJsonMessage>>,
        queue: Mutex<Vec<anyhow::Result<String>>>,
    }

    impl SequenceTransport {
        fn with(responses: Vec<anyhow::Result<String>>) -> Self {
            Self {
                sent: Mutex::new(Vec::new()),
                queue: Mutex::new(responses),
            }
        }

        fn sent_count(&self) -> usize {
            self.sent.lock().unwrap().len()
        }

        fn sent_payload_types(&self) -> Vec<u32> {
            self.sent
                .lock()
                .unwrap()
                .iter()
                .map(|m| m.payload_type)
                .collect()
        }
    }

    impl CTraderOpenApiTransport for SequenceTransport {
        fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
            use crate::app_services::ctrader_messages::{
                CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE, parse_open_api_envelope,
            };
            self.sent.lock().unwrap().extend(messages.iter().cloned());
            let mut queue = self.queue.lock().unwrap();
            let mut out = Vec::with_capacity(messages.len());
            for _ in messages {
                if queue.is_empty() {
                    return Err(anyhow!("stub transport exhausted"));
                }
                let response = queue.remove(0)?;
                // Mirror production transport: early return on error payload
                if let Ok(env) = parse_open_api_envelope(&response) {
                    if env.payload_type == CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE {
                        out.push(response);
                        return Ok(out);
                    }
                }
                out.push(response);
            }
            Ok(out)
        }
    }

    // ─── Helper JSON builders ───────────────────────────────────────────────

    fn app_auth_ok() -> String {
        r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.into()
    }

    fn account_auth_ok(account_id: i64) -> String {
        format!(
            r#"{{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{{"ctidTraderAccountId":{account_id}}}}}"#
        )
    }

    fn symbols_list_ok(account_id: i64, symbols: &[(&str, i64)]) -> String {
        let symbol_json: Vec<String> = symbols
            .iter()
            .map(|(name, id)| {
                format!(
                    r#"{{"symbolId":{id},"symbolName":"{name}","enabled":true,"description":"{name}"}}"#
                )
            })
            .collect();
        format!(
            r#"{{"clientMsgId":"symbols-1","payloadType":2115,"payload":{{"ctidTraderAccountId":{account_id},"symbol":[{}]}}}}"#,
            symbol_json.join(",")
        )
    }

    fn symbol_by_id_ok(account_id: i64, symbol_id: i64, digits: i32) -> String {
        format!(
            r#"{{"clientMsgId":"symbol-by-id-1","payloadType":2117,"payload":{{"ctidTraderAccountId":{account_id},"symbol":[{{"symbolId":{symbol_id},"digits":{digits},"pipPosition":4,"tradingMode":0}}]}}}}"#
        )
    }

    fn error_response(code: &str, description: &str) -> String {
        format!(
            r#"{{"clientMsgId":"err-1","payloadType":2142,"payload":{{"errorCode":"{code}","description":"{description}"}}}}"#
        )
    }

    // ─── Auth message tests ─────────────────────────────────────────────────

    #[test]
    fn app_auth_request_payload_type_is_2100() {
        let msg = build_application_auth_request("cid", "csec", "t1");
        assert_eq!(msg.payload_type, 2100);
        assert_eq!(
            msg.payload.get("clientId").and_then(|v| v.as_str()),
            Some("cid")
        );
    }

    #[test]
    fn app_auth_response_constant_is_2101() {
        assert_eq!(CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, 2101);
    }

    #[test]
    fn error_response_constant_is_2142() {
        assert_eq!(CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE, 2142);
    }

    // ─── Symbol resolution flow ─────────────────────────────────────────────

    #[test]
    fn symbol_resolution_sends_auth_then_symbols_list_then_detail() {
        // v0.5.1.1: resolve_symbol_with_transport opens two WSS connections
        // (one per send_sequence call) and must re-authenticate on each.
        // Batch 1: app-auth + account-auth + symbols-list (3 messages)
        // Batch 2: app-auth + account-auth + symbol-by-id (3 messages)
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbols_list_ok(712345, &[("EURUSD", 14)])),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbol_by_id_ok(712345, 14, 5)),
        ]);

        let result = resolve_symbol_with_transport(
            &transport,
            &CTraderSymbolLookupRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "EURUSD".into(),
            },
        )
        .expect("symbol resolution should succeed");

        assert_eq!(result.account_id, 712345);
        assert_eq!(result.light_symbol.symbol_id, 14);
        assert_eq!(result.symbol.digits, 5);
        assert_eq!(transport.sent_count(), 6);
        // Expected: app-auth(2100), account-auth(2102), symbols-list(2114),
        //           app-auth(2100), account-auth(2102), symbol-by-id(2116)
        assert_eq!(
            transport.sent_payload_types(),
            vec![2100, 2102, 2114, 2100, 2102, 2116]
        );
    }

    #[test]
    fn symbol_resolution_is_case_insensitive_and_strips_slash() {
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbols_list_ok(712345, &[("EUR/USD", 14)])),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbol_by_id_ok(712345, 14, 5)),
        ]);

        let result = resolve_symbol_with_transport(
            &transport,
            &CTraderSymbolLookupRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "eurusd".into(),
            },
        )
        .expect("symbol should match despite case/slash difference");

        assert_eq!(result.light_symbol.symbol_id, 14);
    }

    #[test]
    fn symbol_resolution_fails_when_symbol_not_in_list() {
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbols_list_ok(712345, &[("GBPUSD", 15)])),
        ]);

        let err = resolve_symbol_with_transport(
            &transport,
            &CTraderSymbolLookupRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "EURUSD".into(),
            },
        )
        .expect_err("unknown symbol must fail");

        assert!(err.to_string().contains("EURUSD"));
    }

    #[test]
    fn symbol_resolution_surfaces_ctrader_error_on_app_auth_failure() {
        let transport = SequenceTransport::with(vec![Ok(error_response(
            "INVALID_CLIENT",
            "Client credentials rejected",
        ))]);

        let err = resolve_symbol_with_transport(
            &transport,
            &CTraderSymbolLookupRequest {
                client_id: "bad-cid".into(),
                client_secret: "bad-secret".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "EURUSD".into(),
            },
        )
        .expect_err("bad credentials must fail");

        assert!(err.to_string().contains("INVALID_CLIENT"));
    }

    fn account_bound_symbol_transport(
        account_id: i64,
        list: String,
        detail: String,
    ) -> SequenceTransport {
        SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(account_id)),
            Ok(list),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(account_id)),
            Ok(detail),
        ])
    }

    fn account_bound_symbol_request(
        account_id: i64,
        environment: CTraderEnvironment,
    ) -> CTraderSymbolLookupRequest {
        CTraderSymbolLookupRequest {
            client_id: "fixture-client".into(),
            client_secret: "fixture-secret".into(),
            access_token: "fixture-token".into(),
            environment,
            account_id: account_id.to_string(),
            symbol_name: "EURUSD".into(),
        }
    }

    #[test]
    fn symbol_resolution_binds_both_response_accounts_in_demo_and_live() {
        for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
            for account_id in [42, 99] {
                let transport = account_bound_symbol_transport(
                    account_id,
                    symbols_list_ok(account_id, &[("EUR/USD", 14)]),
                    symbol_by_id_ok(account_id, 14, 5),
                );
                let resolved = resolve_symbol_with_transport(
                    &transport,
                    &account_bound_symbol_request(account_id, environment),
                )
                .expect("both actual response envelopes match the requested account");
                assert_eq!(resolved.account_id, account_id);
                assert_eq!(resolved.light_symbol.symbol_id, 14);
                assert_eq!(resolved.symbol.symbol_id, 14);
                assert_eq!(resolved.symbol.symbol_name, "EUR/USD");
                assert_eq!(resolved.symbol.digits, 5);
                assert_eq!(resolved.symbol.pip_position, 4);
                assert!(resolved.symbol.financials.is_some());

                let sent = transport.sent.lock().expect("sent messages");
                assert_eq!(sent.len(), 6);
                for index in [1, 2, 4, 5] {
                    assert_eq!(
                        sent[index].payload["ctidTraderAccountId"].as_i64(),
                        Some(account_id),
                        "both connections authenticate and request the same account"
                    );
                }
                assert_eq!(sent[5].payload["symbolId"], serde_json::json!([14]));
                assert!(transport.queue.lock().expect("response queue").is_empty());
            }
        }
    }

    #[test]
    fn symbol_resolution_rejects_foreign_list_before_requesting_detail() {
        let transport = account_bound_symbol_transport(
            42,
            symbols_list_ok(99, &[("EURUSD", 14)]),
            symbol_by_id_ok(42, 14, 5),
        );
        let error = resolve_symbol_with_transport(
            &transport,
            &account_bound_symbol_request(42, CTraderEnvironment::Demo),
        )
        .expect_err("a valid symbol row from account 99 cannot be relabelled as account 42");
        let chain = format!("{error:#}");
        assert!(chain.contains("symbols-list response identity"), "{chain}");
        assert!(chain.contains("expected 42, received 99"), "{chain}");
        assert_eq!(transport.sent_count(), 3);
        assert_eq!(transport.queue.lock().expect("response queue").len(), 3);
    }

    #[test]
    fn symbol_resolution_rejects_foreign_detail_before_returning_financials() {
        let transport = account_bound_symbol_transport(
            42,
            symbols_list_ok(42, &[("EURUSD", 14)]),
            symbol_by_id_ok(99, 14, 5),
        );
        let error = resolve_symbol_with_transport(
            &transport,
            &account_bound_symbol_request(42, CTraderEnvironment::Demo),
        )
        .expect_err("matching symbol ID does not authorize another account's metadata");
        let chain = format!("{error:#}");
        assert!(chain.contains("symbol-by-id response identity"), "{chain}");
        assert!(chain.contains("expected 42, received 99"), "{chain}");
        assert_eq!(transport.sent_count(), 6);
        assert!(transport.queue.lock().expect("response queue").is_empty());
    }

    #[test]
    fn symbol_resolution_requires_integer_account_on_each_data_envelope() {
        for detail_stage in [false, true] {
            for account in [
                None,
                Some(serde_json::Value::Null),
                Some(serde_json::json!("42")),
                Some(serde_json::json!(42.5)),
                Some(serde_json::json!(u64::MAX)),
                Some(serde_json::json!(0)),
                Some(serde_json::json!(-42)),
            ] {
                let mut list: serde_json::Value =
                    serde_json::from_str(&symbols_list_ok(42, &[("EURUSD", 14)])).unwrap();
                let mut detail: serde_json::Value =
                    serde_json::from_str(&symbol_by_id_ok(42, 14, 5)).unwrap();
                let response = if detail_stage { &mut detail } else { &mut list };
                let payload = response["payload"].as_object_mut().unwrap();
                match account {
                    Some(value) => {
                        payload.insert("ctidTraderAccountId".into(), value);
                    }
                    None => {
                        payload.remove("ctidTraderAccountId");
                    }
                }
                let transport =
                    account_bound_symbol_transport(42, list.to_string(), detail.to_string());
                let error = resolve_symbol_with_transport(
                    &transport,
                    &account_bound_symbol_request(42, CTraderEnvironment::Demo),
                )
                .expect_err("required response account must be an exact matching integer");
                let chain = format!("{error:#}");
                let stage = if detail_stage {
                    "symbol-by-id"
                } else {
                    "symbols-list"
                };
                assert!(
                    chain.contains(&format!("{stage} response identity")),
                    "{chain}"
                );
                assert_eq!(transport.sent_count(), if detail_stage { 6 } else { 3 });
            }
        }
    }

    #[test]
    fn symbol_resolution_checks_account_before_deserializing_symbol_rows() {
        for detail_stage in [false, true] {
            let mut list: serde_json::Value =
                serde_json::from_str(&symbols_list_ok(42, &[("EURUSD", 14)])).unwrap();
            let mut detail: serde_json::Value =
                serde_json::from_str(&symbol_by_id_ok(42, 14, 5)).unwrap();
            let response = if detail_stage { &mut detail } else { &mut list };
            response["payload"]["ctidTraderAccountId"] = serde_json::json!(99);
            response["payload"]["symbol"] = serde_json::json!([{"symbolId": "invalid-row"}]);
            let transport =
                account_bound_symbol_transport(42, list.to_string(), detail.to_string());
            let error = resolve_symbol_with_transport(
                &transport,
                &account_bound_symbol_request(42, CTraderEnvironment::Demo),
            )
            .expect_err("foreign envelope identity must be rejected before parsing its rows");
            let chain = format!("{error:#}");
            assert!(chain.contains("expected 42, received 99"), "{chain}");
            assert!(!chain.contains("invalid-row"), "{chain}");
            assert_eq!(transport.sent_count(), if detail_stage { 6 } else { 3 });
        }
    }

    #[test]
    fn symbol_resolution_rejects_wrong_data_type_and_preserves_broker_error_code() {
        for detail_stage in [false, true] {
            for broker_error in [false, true] {
                let invalid = if broker_error {
                    error_response("ACCOUNT_NOT_AUTHORIZED", "fixture account refusal")
                } else {
                    // A valid account ID on an unrelated response is not symbol authority.
                    account_auth_ok(42)
                };
                let list = if detail_stage {
                    symbols_list_ok(42, &[("EURUSD", 14)])
                } else {
                    invalid.clone()
                };
                let detail = if detail_stage {
                    invalid
                } else {
                    symbol_by_id_ok(42, 14, 5)
                };
                let transport = account_bound_symbol_transport(42, list, detail);
                let error = resolve_symbol_with_transport(
                    &transport,
                    &account_bound_symbol_request(42, CTraderEnvironment::Demo),
                )
                .expect_err("error or unrelated response must never supply symbol metadata");
                let chain = format!("{error:#}");
                if broker_error {
                    assert!(chain.contains("ACCOUNT_NOT_AUTHORIZED"), "{chain}");
                    assert!(chain.contains("fixture account refusal"), "{chain}");
                } else {
                    assert!(chain.contains("unexpected cTrader payload type"), "{chain}");
                }
                assert_eq!(transport.sent_count(), if detail_stage { 6 } else { 3 });
            }
        }
    }

    // ─── Account discovery flow ─────────────────────────────────────────────

    #[test]
    fn account_discovery_sends_app_auth_then_account_list() {
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(r#"{"clientMsgId":"account-list-1","payloadType":2150,"payload":{"accessToken":"tok","permissionScope":"SCOPE_TRADE","ctidTraderAccount":[{"ctidTraderAccountId":101,"isLive":false,"traderLogin":500101,"brokerTitleShort":"IC Markets"}]}}"#.into()),
        ]);

        let result = perform_account_discovery_with_transport(
            &transport,
            &CTraderAccountDiscoveryRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
            },
        )
        .expect("account discovery should succeed");

        assert_eq!(transport.sent_count(), 2);
        assert_eq!(transport.sent_payload_types(), vec![2100, 2149]);
        assert_eq!(result.accounts.len(), 1);
        assert_eq!(result.accounts[0].account_id, "101");
        assert_eq!(result.accounts[0].is_live, Some(false));
    }

    #[test]
    fn account_discovery_surfaces_app_auth_error() {
        let transport = SequenceTransport::with(vec![Ok(error_response(
            "INVALID_CLIENT",
            "Bad credentials",
        ))]);

        let err = perform_account_discovery_with_transport(
            &transport,
            &CTraderAccountDiscoveryRequest {
                client_id: "bad".into(),
                client_secret: "bad".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
            },
        )
        .expect_err("bad app auth must fail");

        assert!(err.to_string().contains("INVALID_CLIENT"));
    }

    #[test]
    fn demo_environment_uses_demo_endpoint() {
        assert_eq!(
            CTraderEnvironment::Demo.endpoint_host(),
            "demo.ctraderapi.com"
        );
    }

    #[test]
    fn live_environment_uses_live_endpoint() {
        assert_eq!(
            CTraderEnvironment::Live.endpoint_host(),
            "live.ctraderapi.com"
        );
    }

    // ─── Price scaling invariants ───────────────────────────────────────────

    #[test]
    fn trendbar_price_scaling_5_digits_is_correct() {
        let response = serde_json::json!({
            "clientMsgId": "tb-1",
            "payloadType": 2138,
            "payload": {
                "period": "M1",
                "symbolId": 1,
                "hasMore": false,
                "trendbar": [{
                    "volume": 1,
                    "low": 109950,
                    "deltaOpen": 50,
                    "deltaClose": 100,
                    "deltaHigh": 200,
                    "utcTimestampInMinutes": 29000000
                }]
            }
        });

        let symbol = CTraderSymbolInfo {
            symbol_id: 1,
            symbol_name: "EURUSD".into(),
            display_name: "EURUSD".into(),
            digits: 5,
            pip_position: 4,
            is_archived: false,
            is_trading_enabled: true,
            min_volume: None,
            max_volume: None,
            step_volume: None,
            lot_size: None,
            pnl_conversion_fee_rate: None,
            financials: None,
        };

        let result = parse_trendbars_response(&response.to_string(), &symbol).unwrap();

        assert_eq!(result.bars.len(), 1);
        assert!((result.bars[0].low - 1.09950).abs() < 1e-9);
        assert!((result.bars[0].open - 1.10000).abs() < 1e-9);
        assert!((result.bars[0].close - 1.10050).abs() < 1e-9);
        assert!((result.bars[0].high - 1.10150).abs() < 1e-9);
    }

    #[test]
    fn trendbar_timestamp_conversion_minutes_to_ms() {
        let response = serde_json::json!({
            "clientMsgId": "tb-1",
            "payloadType": 2138,
            "payload": {
                "period": "H1",
                "symbolId": 1,
                "hasMore": false,
                "trendbar": [{
                    "volume": 1,
                    "low": 110000,
                    "deltaOpen": 0,
                    "deltaClose": 0,
                    "deltaHigh": 0,
                    "utcTimestampInMinutes": 30000000
                }]
            }
        });

        let symbol = CTraderSymbolInfo {
            symbol_id: 1,
            symbol_name: "EURUSD".into(),
            display_name: "EURUSD".into(),
            digits: 5,
            pip_position: 4,
            is_archived: false,
            is_trading_enabled: true,
            min_volume: None,
            max_volume: None,
            step_volume: None,
            lot_size: None,
            pnl_conversion_fee_rate: None,
            financials: None,
        };

        let result = parse_trendbars_response(&response.to_string(), &symbol).unwrap();

        assert_eq!(result.bars[0].timestamp_ms, 30_000_000_i64 * 60_000);
    }

    // ─── Trendbar period mapping ────────────────────────────────────────────

    #[test]
    fn trendbar_period_mapping_covers_all_standard_timeframes() {
        use crate::app_services::ctrader_messages::trendbar_period_value;

        let cases = [
            ("M1", 1),
            ("M5", 5),
            ("M15", 7),
            ("M30", 8),
            ("H1", 9),
            ("H4", 10),
            ("D1", 12),
            ("W1", 13),
        ];

        for (label, expected) in cases {
            assert_eq!(
                trendbar_period_value(label).unwrap_or_else(|_| panic!("{label} should map")),
                expected,
                "failed for {label}"
            );
        }
    }

    #[test]
    fn trendbar_period_mapping_is_case_insensitive() {
        use crate::app_services::ctrader_messages::trendbar_period_value;

        assert_eq!(trendbar_period_value("m1").unwrap(), 1);
        assert_eq!(trendbar_period_value("h1").unwrap(), 9);
        assert_eq!(trendbar_period_value("d1").unwrap(), 12);
    }
}
