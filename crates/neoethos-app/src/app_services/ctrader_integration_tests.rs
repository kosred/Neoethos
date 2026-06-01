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
        CTraderChartHistoryRequest, CTraderSymbolInfo, CTraderSymbolLookupRequest,
        load_chart_history_with_transport, load_historical_bars_only_with_transport,
        parse_trendbars_response, resolve_symbol_with_transport,
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

    fn symbol_by_id_ok(symbol_id: i64, digits: i32) -> String {
        format!(
            r#"{{"clientMsgId":"symbol-by-id-1","payloadType":2117,"payload":{{"symbol":[{{"symbolId":{symbol_id},"digits":{digits},"pipPosition":4,"tradingMode":0}}]}}}}"#
        )
    }

    fn trendbars_ok(symbol_id: i64, period: &str) -> String {
        format!(
            r#"{{"clientMsgId":"trendbars-1","payloadType":2138,"payload":{{"period":"{period}","symbolId":{symbol_id},"trendbar":[{{"volume":10,"low":110000,"deltaOpen":30,"deltaClose":80,"deltaHigh":150,"utcTimestampInMinutes":28500000}}],"hasMore":false}}}}"#
        )
    }

    fn ticks_ok(symbol_id: i64) -> String {
        format!(
            r#"{{"clientMsgId":"ticks-1","payloadType":2146,"payload":{{"symbolId":{symbol_id},"hasMore":false,"tickData":[{{"timestamp":1710000000000,"tick":110020}},{{"timestamp":300,"tick":109980}}]}}}}"#
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
            Ok(symbol_by_id_ok(14, 5)),
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
            Ok(symbol_by_id_ok(14, 5)),
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

    // ─── Historical bars fetch ──────────────────────────────────────────────

    #[test]
    fn bars_only_flow_sends_9_messages_and_returns_bar() {
        // Each production send_sequence call opens a fresh WSS connection,
        // so symbol list, symbol detail, and trendbars each authenticate.
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbols_list_ok(712345, &[("EURUSD", 14)])),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbol_by_id_ok(14, 5)),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(trendbars_ok(14, "M15")),
        ]);

        let result = load_historical_bars_only_with_transport(
            &transport,
            &CTraderChartHistoryRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "EURUSD".into(),
                timeframe: "M15".into(),
                from_timestamp_ms: 1_709_000_000_000,
                to_timestamp_ms: 1_710_000_000_000,
                count: Some(96),
            },
        )
        .expect("bars-only fetch should succeed");

        assert_eq!(transport.sent_count(), 9);
        assert_eq!(result.bars.len(), 1);
        assert!(!result.has_more);
        assert!((result.bars[0].low - 1.10000).abs() < 1e-9);
        assert!((result.bars[0].close - 1.10080).abs() < 1e-9);
    }

    #[test]
    fn full_chart_history_flow_sends_7_messages_and_returns_bars_and_ticks() {
        // v0.5.1.1: resolve_symbol uses two send_sequence calls (re-auth on
        // each, 6 messages), plus 3 data messages = 9 total.
        let transport = SequenceTransport::with(vec![
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbols_list_ok(712345, &[("EURUSD", 14)])),
            Ok(app_auth_ok()),
            Ok(account_auth_ok(712345)),
            Ok(symbol_by_id_ok(14, 5)),
            Ok(trendbars_ok(14, "M5")),
            Ok(ticks_ok(14)),
            Ok(ticks_ok(14)),
        ]);

        let result = load_chart_history_with_transport(
            &transport,
            &CTraderChartHistoryRequest {
                client_id: "cid".into(),
                client_secret: "csec".into(),
                access_token: "tok".into(),
                environment: CTraderEnvironment::Demo,
                account_id: "712345".into(),
                symbol_name: "EURUSD".into(),
                timeframe: "M5".into(),
                from_timestamp_ms: 1_709_000_000_000,
                to_timestamp_ms: 1_710_000_000_000,
                count: Some(96),
            },
        )
        .expect("full chart history should succeed");

        assert_eq!(transport.sent_count(), 9);
        assert_eq!(result.bars.len(), 1);
        assert_eq!(result.bid_ticks.len(), 2);
        assert_eq!(result.ask_ticks.len(), 2);
        assert_eq!(
            result.live_subscription_plan.subscribe_spots.payload_type,
            2127
        );
        assert_eq!(
            result
                .live_subscription_plan
                .subscribe_trendbars
                .payload_type,
            2135
        );
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
