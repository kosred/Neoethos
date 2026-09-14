use super::{
    CTraderEnvironment, CTraderLightSymbolInfo, CTraderResolvedSymbol, ResolvedCreds,
    parse_symbol_by_id_response, prepare_new_order_from_resolved_creds,
};
use std::cell::Cell;

fn fixture_creds(account_id: &str, environment: CTraderEnvironment) -> ResolvedCreds {
    ResolvedCreds {
        client_id: "fixture-client".to_string(),
        client_secret: "fixture-secret".to_string(),
        access_token: "fixture-token".to_string(),
        account_id_str: account_id.to_string(),
        environment,
        env_label: match environment {
            CTraderEnvironment::Demo => "Demo",
            CTraderEnvironment::Live => "Live",
        },
    }
}

fn fixture_symbol(account_id: i64) -> CTraderResolvedSymbol {
    let payload = serde_json::json!({
        "payloadType": 2117,
        "payload": {
            "ctidTraderAccountId": account_id,
            "symbol": [{
                "symbolId": 7,
                "digits": 5,
                "pipPosition": 4,
                "tradingMode": 0,
                "enableShortSelling": true,
                "lotSize": 10_000_000,
                "minVolume": 100_000,
                "maxVolume": 1_000_000_000,
                "stepVolume": 100_000
            }]
        }
    });
    let mut symbol = parse_symbol_by_id_response(&payload.to_string())
        .expect("synthetic full-symbol response")
        .remove(0);
    symbol.symbol_name = "EURUSD".to_string();
    symbol.display_name = "EURUSD".to_string();
    CTraderResolvedSymbol {
        account_id,
        light_symbol: CTraderLightSymbolInfo {
            symbol_id: 7,
            symbol_name: "EURUSD".to_string(),
            enabled: true,
            description: None,
            symbol_category_id: None,
            base_asset_id: None,
            quote_asset_id: None,
        },
        symbol,
    }
}

#[test]
fn same_environment_account_switch_refuses_before_symbol_lookup_or_order_preparation() {
    let lookup_calls = Cell::new(0);
    let result = prepare_new_order_from_resolved_creds(
        fixture_creds("99", CTraderEnvironment::Demo),
        "EURUSD",
        0.01,
        Some(12.0),
        Some(24.0),
        Some(42),
        |_| {
            lookup_calls.set(lookup_calls.get() + 1);
            Ok(fixture_symbol(99))
        },
    );
    let error = result
        .err()
        .expect("changed account must not produce an order for submission");
    assert!(
        error
            .to_string()
            .contains("does not match the admitted account")
    );
    assert_eq!(lookup_calls.get(), 0, "must refuse before symbol transport");
}

#[test]
fn matched_account_keeps_one_credential_snapshot_and_exact_order_units() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        let lookup_calls = Cell::new(0);
        let prepared = prepare_new_order_from_resolved_creds(
            fixture_creds("42", environment),
            "EURUSD",
            0.01,
            Some(12.0),
            Some(24.0),
            Some(42),
            |request| {
                lookup_calls.set(lookup_calls.get() + 1);
                assert_eq!(request.account_id, "42");
                assert_eq!(request.environment, environment);
                assert_eq!(request.symbol_name, "EURUSD");
                assert_eq!(request.client_id, "fixture-client");
                assert_eq!(request.client_secret, "fixture-secret");
                assert_eq!(request.access_token, "fixture-token");
                Ok(fixture_symbol(42))
            },
        )
        .expect("matching admitted account");
        assert_eq!(lookup_calls.get(), 1);
        assert_eq!(prepared.account_id, 42);
        assert_eq!(prepared.creds.account_id_str, "42");
        assert_eq!(prepared.creds.environment, environment);
        assert_eq!(prepared.creds.client_id, "fixture-client");
        assert_eq!(prepared.creds.client_secret, "fixture-secret");
        assert_eq!(prepared.creds.access_token, "fixture-token");
        assert_eq!(prepared.symbol_id, 7);
        // Independently expected: 0.01 * 10,000,000 centi-units, and
        // 12/24 pips * 10^-4 price units * 100,000 relative-distance units.
        assert_eq!(prepared.volume_units, 100_000);
        assert_eq!(prepared.relative_stop_loss, Some(120));
        assert_eq!(prepared.relative_take_profit, Some(240));
        assert!(prepared.broker_allows_new_positions);
        assert!(prepared.broker_allows_short_selling);
    }
}

#[test]
fn manual_order_without_admission_pin_preserves_the_selected_account() {
    for account_id in [42, 99] {
        let prepared = prepare_new_order_from_resolved_creds(
            fixture_creds(&account_id.to_string(), CTraderEnvironment::Demo),
            "EURUSD",
            0.01,
            None,
            None,
            None,
            |request| {
                assert_eq!(request.account_id, account_id.to_string());
                Ok(fixture_symbol(account_id))
            },
        )
        .expect("manual order remains bound to its resolved selected account");
        assert_eq!(prepared.account_id, account_id);
        assert_eq!(prepared.creds.account_id_str, account_id.to_string());
        assert_eq!(prepared.volume_units, 100_000);
        assert_eq!(prepared.relative_stop_loss, None);
        assert_eq!(prepared.relative_take_profit, None);
    }
}

#[test]
fn malformed_or_nonpositive_account_identity_never_reaches_symbol_lookup() {
    for (resolved_account, expected_account) in [
        ("invalid", Some(42)),
        ("", None),
        ("9223372036854775808", Some(42)),
        ("0", None),
        ("-1", None),
        ("42", Some(0)),
        ("42", Some(-1)),
    ] {
        let lookup_calls = Cell::new(0);
        let result = prepare_new_order_from_resolved_creds(
            fixture_creds(resolved_account, CTraderEnvironment::Demo),
            "EURUSD",
            0.01,
            Some(12.0),
            Some(24.0),
            expected_account,
            |_| {
                lookup_calls.set(lookup_calls.get() + 1);
                Ok(fixture_symbol(42))
            },
        );
        assert!(
            result.is_err(),
            "invalid identity must not prepare an order"
        );
        assert_eq!(lookup_calls.get(), 0);
    }
}

#[test]
fn foreign_symbol_account_is_rejected_before_preparing_any_order() {
    for expected_account in [Some(42), None] {
        let lookup_calls = Cell::new(0);
        let result = prepare_new_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            "EURUSD",
            0.01,
            Some(12.0),
            Some(24.0),
            expected_account,
            |_| {
                lookup_calls.set(lookup_calls.get() + 1);
                let mut foreign = fixture_symbol(99);
                // Account refusal must precede even the financial metadata check.
                foreign.symbol.financials = None;
                Ok(foreign)
            },
        );
        let error = result
            .err()
            .expect("foreign account cannot prepare an order");
        assert_eq!(lookup_calls.get(), 1);
        assert!(
            error
                .to_string()
                .contains("symbol metadata belongs to a different")
        );
    }
}

#[test]
fn symbol_lookup_failure_is_propagated_without_an_order_fallback() {
    let result = prepare_new_order_from_resolved_creds(
        fixture_creds("42", CTraderEnvironment::Demo),
        "EURUSD",
        0.01,
        Some(12.0),
        Some(24.0),
        Some(42),
        |_| Err(anyhow::anyhow!("fixture symbol lookup refused")),
    );
    let error = result
        .err()
        .expect("failed lookup must not prepare an order");
    assert_eq!(error.to_string(), "fixture symbol lookup refused");
}

struct MarginWireTransport {
    responses: Vec<serde_json::Value>,
    calls: Cell<usize>,
}

impl super::CTraderOpenApiTransport for MarginWireTransport {
    fn send_sequence(
        &self,
        messages: &[crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage],
    ) -> anyhow::Result<Vec<String>> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(
            messages.len(),
            6,
            "the existing poll remains one six-request sequence"
        );
        for message in &messages[1..] {
            assert_eq!(message.payload["ctidTraderAccountId"], 42);
        }
        Ok(self
            .responses
            .iter()
            .map(serde_json::Value::to_string)
            .collect())
    }
}

fn margin_wire_responses() -> Vec<serde_json::Value> {
    use serde_json::json;
    vec![
        json!({"payloadType": 2101, "payload": {}}),
        json!({"payloadType": 2103, "payload": {"ctidTraderAccountId": 42}}),
        json!({"payloadType": 2168, "payload": {
            "ctidTraderAccountId": 42,
            "marginCall": [{"marginCallType": 1, "marginLevelThreshold": 500.0}]
        }}),
        json!({"payloadType": 2122, "payload": {
            "ctidTraderAccountId": 42,
            "trader": {"balance": 100_000, "moneyDigits": 2}
        }}),
        json!({"payloadType": 2125, "payload": {
            "ctidTraderAccountId": 42,
            "position": [{
                "positionId": 7, "positionStatus": 1,
                "tradeData": {"symbolId": 7, "volume": 100_000, "tradeSide": 1},
                "usedMargin": 25_000, "moneyDigits": 2
            }]
        }}),
        json!({"payloadType": 2188, "payload": {
            "ctidTraderAccountId": 42, "moneyDigits": 8,
            "positionUnrealizedPnL": [{
                "positionId": 7,
                "grossUnrealizedPnL": -1_000_000_000_i64,
                "netUnrealizedPnL": -1_234_567_890_i64
            }]
        }}),
    ]
}

fn poll_margin_fixture(
    responses: Vec<serde_json::Value>,
    environment: CTraderEnvironment,
) -> (anyhow::Result<super::MarginStatus>, usize) {
    let transport = MarginWireTransport {
        responses,
        calls: Cell::new(0),
    };
    let result =
        super::fetch_margin_status_with_transport(&transport, &fixture_creds("42", environment));
    (result, transport.calls.get())
}

#[test]
fn margin_wire_poll_preserves_independent_money_scales_and_existing_formula() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        let (result, calls) = poll_margin_fixture(margin_wire_responses(), environment);
        let status = result.expect("matching complete broker responses");
        assert_eq!(calls, 1);
        assert_eq!(status.account_id, 42);
        assert_eq!(
            status.environment_label,
            fixture_creds("42", environment).env_label
        );
        // Independent wire interpretation: balance 100000/100, PnL
        // -1234567890/100000000, used margin 25000/100.
        assert_eq!(status.balance, 1000.0);
        assert_eq!(status.unrealized_pnl, -12.3456789);
        assert!((status.equity - 987.6543211).abs() < 1e-10);
        assert_eq!(status.used_margin, 250.0);
        assert!((status.margin_level_pct.expect("used margin") - 395.06172844).abs() < 1e-10);
        assert_eq!(status.breached_threshold_pct, Some(500.0));
        assert_eq!(status.open_position_count, 1);
        assert_eq!(status.positions_missing_used_margin, 0);
    }
}

#[test]
fn margin_wire_poll_rejects_every_foreign_or_missing_account_even_for_empty_sets() {
    for index in 1..6 {
        for missing in [false, true] {
            for empty in [false, true] {
                let mut responses = margin_wire_responses();
                if empty {
                    responses[4]["payload"]["position"] = serde_json::json!([]);
                    responses[5]["payload"]["positionUnrealizedPnL"] = serde_json::json!([]);
                }
                if missing {
                    responses[index]["payload"]
                        .as_object_mut()
                        .expect("payload object")
                        .remove("ctidTraderAccountId");
                } else {
                    responses[index]["payload"]["ctidTraderAccountId"] = serde_json::json!(99);
                }
                let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
                let error = result.expect_err("foreign/absent envelope account must fail closed");
                assert_eq!(calls, 1, "identity rejection must not retry the transport");
                assert!(
                    error
                        .to_string()
                        .contains(super::MARGIN_STATUS_UNREADABLE_SENTINEL)
                );
                assert!(error.to_string().contains("response account"));
            }
        }
    }
}

#[test]
fn margin_wire_poll_rejects_missing_unknown_and_duplicate_position_pnl() {
    for case in 0..4 {
        let mut responses = margin_wire_responses();
        let expected = match case {
            0 => {
                responses[5]["payload"]["positionUnrealizedPnL"] = serde_json::json!([]);
                "omitted open positions"
            }
            1 => {
                responses[5]["payload"]["positionUnrealizedPnL"][0]["positionId"] =
                    serde_json::json!(99);
                "unknown position"
            }
            2 => {
                let row = responses[5]["payload"]["positionUnrealizedPnL"][0].clone();
                responses[5]["payload"]["positionUnrealizedPnL"] =
                    serde_json::json!([row.clone(), row]);
                "duplicate position"
            }
            _ => {
                let row = responses[4]["payload"]["position"][0].clone();
                responses[4]["payload"]["position"] = serde_json::json!([row.clone(), row]);
                "duplicate position"
            }
        };
        let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
        let error = result.expect_err("position/PnL coverage must be exact");
        assert_eq!(calls, 1);
        assert!(
            error
                .to_string()
                .contains(super::MARGIN_STATUS_UNREADABLE_SENTINEL)
        );
        assert!(error.to_string().contains(expected), "{error:#}");
    }
}

#[test]
fn margin_empty_account_requires_valid_pnl_scale_without_requiring_closed_trades() {
    for digits in [0_u32, 2, 8, 10] {
        let mut responses = margin_wire_responses();
        responses[4]["payload"]["position"] = serde_json::json!([]);
        responses[5]["payload"]["positionUnrealizedPnL"] = serde_json::json!([]);
        responses[5]["payload"]["moneyDigits"] = serde_json::json!(digits);
        let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
        let status = result.expect("a new account with zero positions is valid");
        assert_eq!(calls, 1);
        assert_eq!(status.unrealized_pnl, 0.0);
        assert_eq!(status.equity, 1000.0);
        assert_eq!(status.used_margin, 0.0);
        assert_eq!(status.margin_level_pct, None);
        assert_eq!(status.open_position_count, 0);
    }
    for digits in [None, Some(11_u32), Some(u32::MAX)] {
        let mut responses = margin_wire_responses();
        responses[4]["payload"]["position"] = serde_json::json!([]);
        responses[5]["payload"]["positionUnrealizedPnL"] = serde_json::json!([]);
        match digits {
            Some(digits) => responses[5]["payload"]["moneyDigits"] = serde_json::json!(digits),
            None => {
                responses[5]["payload"]
                    .as_object_mut()
                    .expect("payload object")
                    .remove("moneyDigits");
            }
        }
        let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
        let error = result.expect_err("empty arrays cannot bypass scale validation");
        assert_eq!(calls, 1);
        assert!(
            error
                .to_string()
                .contains(super::MARGIN_STATUS_UNREADABLE_SENTINEL)
        );
        assert!(format!("{error:#}").contains("money"), "{error:#}");
    }
}

#[test]
fn margin_wire_poll_keeps_negative_equity_and_missing_margin_diagnostics() {
    let mut responses = margin_wire_responses();
    responses[5]["payload"]["positionUnrealizedPnL"][0]["netUnrealizedPnL"] =
        serde_json::json!(-200_000_000_000_i64);
    let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
    let status = result.expect("broker-reported loss is not a malformed reply");
    assert_eq!(calls, 1);
    assert_eq!(status.equity, -1000.0);
    assert_eq!(status.margin_level_pct, Some(-400.0));
    assert_eq!(status.breached_threshold_pct, Some(500.0));

    let mut responses = margin_wire_responses();
    responses[4]["payload"]["position"][0]
        .as_object_mut()
        .expect("position object")
        .remove("usedMargin");
    let (result, _) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
    let status = result.expect("existing explicit missing-margin diagnostic remains");
    assert_eq!(status.positions_missing_used_margin, 1);
    assert_eq!(status.used_margin, 0.0);
}

#[test]
fn margin_wire_poll_preserves_broker_error_and_wrong_type_before_identity_checks() {
    let mut responses = margin_wire_responses();
    responses[5] = serde_json::json!({
        "payloadType": 2142,
        "payload": {
            "errorCode": "BLOCKED_PAYLOAD_TYPE",
            "description": "fixture margin request blocked",
            "retryAfter": 120
        }
    });
    let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
    let error = result.expect_err("broker refusal must retain its original detail");
    assert_eq!(calls, 1);
    assert!(error.to_string().contains("BLOCKED_PAYLOAD_TYPE"));
    assert!(error.to_string().contains("fixture margin request blocked"));
    assert!(!error.to_string().contains("response account"));

    let mut responses = margin_wire_responses();
    responses[5] = serde_json::json!({"payloadType": 2122, "payload": {}});
    let (result, calls) = poll_margin_fixture(responses, CTraderEnvironment::Demo);
    let error = result.expect_err("wrong payload type precedes absent account field");
    assert_eq!(calls, 1);
    assert!(error.to_string().contains("payload type"));
    assert!(!error.to_string().contains("response account"));
}

#[test]
fn margin_nonpositive_or_invalid_account_never_reaches_transport() {
    for account in ["0", "-1", "invalid", "9223372036854775808"] {
        let transport = MarginWireTransport {
            responses: margin_wire_responses(),
            calls: Cell::new(0),
        };
        assert!(
            super::fetch_margin_status_with_transport(
                &transport,
                &fixture_creds(account, CTraderEnvironment::Demo),
            )
            .is_err()
        );
        assert_eq!(transport.calls.get(), 0);
    }
}

#[test]
fn new_order_requires_every_positive_grid_field_even_without_admission_pin() {
    for expected_account in [Some(42), None] {
        for field in 0..3 {
            for invalid in [None, Some(0), Some(-1)] {
                let result = prepare_new_order_from_resolved_creds(
                    fixture_creds("42", CTraderEnvironment::Demo),
                    "EURUSD",
                    0.01,
                    Some(12.0),
                    Some(24.0),
                    expected_account,
                    |_| {
                        let mut symbol = fixture_symbol(42);
                        match field {
                            0 => symbol.symbol.min_volume = invalid,
                            1 => symbol.symbol.max_volume = invalid,
                            _ => symbol.symbol.step_volume = invalid,
                        }
                        Ok(symbol)
                    },
                );
                let error = result
                    .err()
                    .expect("incomplete/invalid grid cannot prepare an order");
                let field_name = ["minVolume", "maxVolume", "stepVolume"][field];
                assert!(error.to_string().contains(field_name), "{error:#}");
            }
        }
    }
}

#[test]
fn new_order_rejects_inverted_grid_and_preserves_exact_bounds_and_step() {
    let result = prepare_new_order_from_resolved_creds(
        fixture_creds("42", CTraderEnvironment::Demo),
        "EURUSD",
        0.01,
        None,
        None,
        Some(42),
        |_| {
            let mut symbol = fixture_symbol(42);
            symbol.symbol.min_volume = Some(2_000_000_000);
            Ok(symbol)
        },
    );
    assert!(
        result
            .err()
            .expect("inverted grid")
            .to_string()
            .contains("inverted volume bounds")
    );

    for (lots, expected) in [
        (0.001, "below broker min_volume"),
        (101.0, "exceeds broker max_volume"),
        (0.015, "not aligned to broker stepVolume"),
    ] {
        let result = prepare_new_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            "EURUSD",
            lots,
            None,
            None,
            Some(42),
            |_| Ok(fixture_symbol(42)),
        );
        assert!(
            result
                .err()
                .expect("invalid requested volume")
                .to_string()
                .contains(expected)
        );
    }
    let prepared = prepare_new_order_from_resolved_creds(
        fixture_creds("42", CTraderEnvironment::Demo),
        "EURUSD",
        0.01,
        None,
        None,
        Some(42),
        |_| {
            let mut symbol = fixture_symbol(42);
            symbol.symbol.max_volume = symbol.symbol.min_volume;
            Ok(symbol)
        },
    )
    .expect("one valid exact grid point must remain allowed");
    assert_eq!(prepared.volume_units, 100_000);
}

#[test]
fn no_volume_amendment_requires_valid_symbol_identity_even_without_entry_metadata() {
    for expected_account in [Some(42), None] {
        for (symbol_id, symbol_name, valid) in [
            (0, "EURUSD", false),
            (-1, "EURUSD", false),
            (7, "", false),
            (7, " \t", false),
            (7, " EURUSD", false),
            (7, "EURUSD ", false),
            (7, "EURUSD", true),
        ] {
            let calls = Cell::new(0);
            let result = super::prepare_amend_order_from_resolved_creds(
                fixture_creds("42", CTraderEnvironment::Live),
                900,
                "EURUSD",
                super::CTraderOrderType::Limit,
                None,
                Some(1.25),
                None,
                None,
                None,
                expected_account,
                |request| {
                    calls.set(calls.get() + 1);
                    assert_eq!(request.account_id, "42");
                    assert_eq!(request.environment, CTraderEnvironment::Live);
                    let mut resolved = fixture_symbol(42);
                    // Keep both representations equal: the rejected condition
                    // is identity sanity, not a new symbol matching policy.
                    resolved.light_symbol.symbol_id = symbol_id;
                    resolved.symbol.symbol_id = symbol_id;
                    resolved.light_symbol.symbol_name = symbol_name.to_owned();
                    resolved.symbol.symbol_name = symbol_name.to_owned();
                    resolved.symbol.financials = None;
                    resolved.symbol.lot_size = None;
                    resolved.symbol.min_volume = None;
                    resolved.symbol.max_volume = None;
                    resolved.symbol.step_volume = None;
                    Ok(resolved)
                },
            );
            assert_eq!(calls.get(), 1);
            if valid {
                let runtime =
                    result.expect("valid identity needs no entry metadata for no-volume amendment");
                assert_eq!(runtime.account_id, "42");
                assert_eq!(runtime.environment, CTraderEnvironment::Live);
                let super::CTraderExecutionRequest::AmendOrder(amend) = runtime.request else {
                    panic!("must prepare an amendment, never an entry");
                };
                assert_eq!(amend.volume, None);
                let wire = crate::app_services::ctrader_messages::build_amend_order_request(
                    &amend,
                    "fixture-amend-identity",
                );
                assert!(wire.payload.get("volume").is_none());
                assert_eq!(wire.payload["limitPrice"], 1.25);
            } else {
                let error = result.expect_err("invalid identity cannot prepare an amendment");
                assert!(
                    error.to_string().contains("invalid symbol identity"),
                    "{error:#}"
                );
            }
        }
    }
}

#[test]
fn no_volume_amendment_omits_wire_volume_without_entry_grid_or_lot_size() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        for order_type in [
            super::CTraderOrderType::Limit,
            super::CTraderOrderType::Stop,
        ] {
            // Price-only, expiry-only, SL/TP-only and the combined amendment.
            for (price, expiry, stop_loss, take_profit) in [
                (Some(1.25), None, None, None),
                (None, Some(1_800_000_000_000), None, None),
                (None, None, Some(12.0), Some(24.0)),
                (Some(1.25), Some(1_800_000_000_000), Some(12.0), Some(24.0)),
            ] {
                for malformed_grid in [false, true] {
                    let calls = Cell::new(0);
                    let runtime = super::prepare_amend_order_from_resolved_creds(
                        fixture_creds("42", environment),
                        900,
                        "EURUSD",
                        order_type,
                        None,
                        price,
                        stop_loss,
                        take_profit,
                        expiry,
                        Some(42),
                        |request| {
                            calls.set(calls.get() + 1);
                            assert_eq!(request.account_id, "42");
                            assert_eq!(request.environment, environment);
                            assert_eq!(request.symbol_name, "EURUSD");
                            assert_eq!(request.client_id, "fixture-client");
                            assert_eq!(request.client_secret, "fixture-secret");
                            assert_eq!(request.access_token, "fixture-token");
                            let mut resolved = fixture_symbol(42);
                            // No volume is requested: neither absent nor malformed
                            // entry-only metadata may veto this amendment.
                            resolved.symbol.lot_size = None;
                            resolved.symbol.financials = None;
                            resolved.symbol.min_volume = malformed_grid.then_some(0);
                            resolved.symbol.max_volume = malformed_grid.then_some(-1);
                            resolved.symbol.step_volume = malformed_grid.then_some(0);
                            Ok(resolved)
                        },
                    )
                    .expect("amendment does not invent an entry volume");
                    assert_eq!(calls.get(), 1);
                    assert_eq!(runtime.environment, environment);
                    assert_eq!(runtime.account_id, "42");
                    assert_eq!(runtime.client_id, "fixture-client");
                    assert_eq!(runtime.client_secret, "fixture-secret");
                    assert_eq!(runtime.access_token, "fixture-token");
                    let super::CTraderExecutionRequest::AmendOrder(amend) = runtime.request else {
                        panic!("must prepare an amendment, never a new order");
                    };
                    assert_eq!(amend.account_id, 42);
                    assert_eq!(amend.order_id, 900);
                    assert_eq!(amend.volume, None);
                    assert_eq!(amend.expiration_timestamp_ms, expiry);
                    assert_eq!(amend.relative_stop_loss, stop_loss.map(|_| 120));
                    assert_eq!(amend.relative_take_profit, take_profit.map(|_| 240));
                    let wire = crate::app_services::ctrader_messages::build_amend_order_request(
                        &amend,
                        "fixture-amend",
                    );
                    assert_eq!(wire.payload_type, 2109);
                    assert_eq!(wire.payload["ctidTraderAccountId"], 42);
                    assert_eq!(wire.payload["orderId"], 900);
                    assert!(
                        wire.payload.get("volume").is_none(),
                        "None must omit the wire field"
                    );
                    let (price_field, absent_price_field) = match order_type {
                        super::CTraderOrderType::Limit => ("limitPrice", "stopPrice"),
                        _ => ("stopPrice", "limitPrice"),
                    };
                    assert_eq!(
                        wire.payload
                            .get(price_field)
                            .and_then(serde_json::Value::as_f64),
                        price
                    );
                    assert!(wire.payload.get(absent_price_field).is_none());
                    assert_eq!(
                        wire.payload
                            .get("expirationTimestamp")
                            .and_then(serde_json::Value::as_i64),
                        expiry
                    );
                    assert_eq!(
                        wire.payload
                            .get("relativeStopLoss")
                            .and_then(serde_json::Value::as_i64),
                        stop_loss.map(|_| 120)
                    );
                    assert_eq!(
                        wire.payload
                            .get("relativeTakeProfit")
                            .and_then(serde_json::Value::as_i64),
                        take_profit.map(|_| 240)
                    );
                }
            }
        }
    }
}

#[test]
fn volume_amendment_requires_the_same_complete_grid_and_exact_units_as_entry() {
    for omitted_field in ["lotSize", "minVolume", "maxVolume", "stepVolume"] {
        let result = super::prepare_amend_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            900,
            "EURUSD",
            super::CTraderOrderType::Limit,
            Some(0.01),
            Some(1.25),
            Some(12.0),
            Some(24.0),
            None,
            Some(42),
            |_| {
                let mut resolved = fixture_symbol(42);
                match omitted_field {
                    "lotSize" => resolved.symbol.lot_size = None,
                    "minVolume" => resolved.symbol.min_volume = None,
                    "maxVolume" => resolved.symbol.max_volume = None,
                    "stepVolume" => resolved.symbol.step_volume = None,
                    _ => unreachable!(),
                }
                Ok(resolved)
            },
        );
        let error = result.expect_err("an explicit volume requires its exact entry contract");
        assert!(error.to_string().contains(omitted_field), "{error:#}");
    }
    let runtime = super::prepare_amend_order_from_resolved_creds(
        fixture_creds("42", CTraderEnvironment::Live),
        900,
        "EURUSD",
        super::CTraderOrderType::Stop,
        Some(0.01),
        Some(1.25),
        Some(12.0),
        Some(24.0),
        None,
        Some(42),
        |_| Ok(fixture_symbol(42)),
    )
    .expect("valid explicit volume");
    assert_eq!(runtime.environment, CTraderEnvironment::Live);
    let super::CTraderExecutionRequest::AmendOrder(amend) = runtime.request else {
        panic!("must prepare an amendment");
    };
    let wire = crate::app_services::ctrader_messages::build_amend_order_request(
        &amend,
        "fixture-amend-volume",
    );
    // Independent broker-unit expectation, not a value copied from the entry result.
    assert_eq!(wire.payload["volume"], 100_000);
    assert_eq!(wire.payload["stopPrice"], 1.25);
    assert_eq!(wire.payload["relativeStopLoss"], 120);
    assert_eq!(wire.payload["relativeTakeProfit"], 240);
}

#[test]
fn amendments_keep_account_pin_foreign_metadata_and_lookup_error_refusals() {
    for volume in [None, Some(0.01)] {
        for account in ["99", "0", "-1", "invalid"] {
            let calls = Cell::new(0);
            let result = super::prepare_amend_order_from_resolved_creds(
                fixture_creds(account, CTraderEnvironment::Demo),
                900,
                "EURUSD",
                super::CTraderOrderType::Limit,
                volume,
                Some(1.25),
                None,
                None,
                None,
                Some(42),
                |_| {
                    calls.set(calls.get() + 1);
                    Ok(fixture_symbol(42))
                },
            );
            assert!(result.is_err());
            assert_eq!(
                calls.get(),
                0,
                "identity refusal must precede symbol lookup"
            );
        }
        let result = super::prepare_amend_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            900,
            "EURUSD",
            super::CTraderOrderType::Limit,
            volume,
            Some(1.25),
            None,
            None,
            None,
            Some(42),
            |_| {
                let mut foreign = fixture_symbol(99);
                foreign.symbol.financials = None;
                foreign.symbol.lot_size = None;
                Ok(foreign)
            },
        );
        assert!(
            result
                .expect_err("foreign account")
                .to_string()
                .contains("symbol metadata belongs to a different")
        );
        let result = super::prepare_amend_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            900,
            "EURUSD",
            super::CTraderOrderType::Limit,
            volume,
            Some(1.25),
            None,
            None,
            None,
            Some(42),
            |_| {
                Err(anyhow::anyhow!(
                    "fixture broker symbol lookup rejected: SYMBOL_NOT_FOUND"
                ))
            },
        );
        assert_eq!(
            result.expect_err("broker refusal").to_string(),
            "fixture broker symbol lookup rejected: SYMBOL_NOT_FOUND"
        );
    }
}

#[test]
fn no_volume_amendment_still_requires_exact_valid_relative_brackets() {
    for distance in [f64::NAN, f64::INFINITY, 0.0, -1.0, 0.01] {
        let result = super::prepare_amend_order_from_resolved_creds(
            fixture_creds("42", CTraderEnvironment::Demo),
            900,
            "EURUSD",
            super::CTraderOrderType::Limit,
            None,
            None,
            Some(distance),
            None,
            None,
            Some(42),
            |_| {
                let mut resolved = fixture_symbol(42);
                resolved.symbol.lot_size = None;
                resolved.symbol.min_volume = None;
                resolved.symbol.max_volume = None;
                resolved.symbol.step_volume = None;
                Ok(resolved)
            },
        );
        assert!(
            result.is_err(),
            "invalid or unrepresentable bracket {distance} must not prepare"
        );
    }
}

struct AmendDetailsWireTransport {
    responses: Vec<serde_json::Value>,
    calls: Cell<usize>,
    fail: bool,
}

impl AmendDetailsWireTransport {
    fn new(responses: Vec<serde_json::Value>) -> Self {
        Self {
            responses,
            calls: Cell::new(0),
            fail: false,
        }
    }
}

impl super::CTraderOpenApiTransport for AmendDetailsWireTransport {
    fn send_sequence(
        &self,
        messages: &[crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage],
    ) -> anyhow::Result<Vec<String>> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].payload_type, 2100);
        assert_eq!(messages[0].payload["clientId"], "fixture-client");
        assert_eq!(messages[0].payload["clientSecret"], "fixture-secret");
        assert_eq!(messages[1].payload_type, 2102);
        assert_eq!(messages[1].payload["ctidTraderAccountId"], 42);
        assert_eq!(messages[1].payload["accessToken"], "fixture-token");
        assert_eq!(messages[2].payload_type, 2181);
        assert_eq!(messages[2].payload["ctidTraderAccountId"], 42);
        assert_eq!(messages[2].payload["orderId"], 900);
        assert_eq!(messages[2].client_msg_id, "amend-order-details");
        if self.fail {
            anyhow::bail!("fixture order-details transport failure");
        }
        Ok(self
            .responses
            .iter()
            .map(serde_json::Value::to_string)
            .collect())
    }
}

fn amendment_details_responses(order_type: super::CTraderOrderType) -> Vec<serde_json::Value> {
    use serde_json::json;
    vec![
        json!({"clientMsgId":"amend-details-app-auth","payloadType":2101,"payload":{}}),
        json!({"clientMsgId":"amend-details-account-auth","payloadType":2103,
            "payload":{"ctidTraderAccountId":42}}),
        json!({"clientMsgId":"amend-order-details","payloadType":2182,"payload":{
            "ctidTraderAccountId":42,
            "order":{"orderId":900,"orderType":match order_type {
                super::CTraderOrderType::Limit => 2,
                super::CTraderOrderType::Stop => 3,
                _ => panic!("fixture covers the two supported pending types"),
            },"orderStatus":1,
                "tradeData":{"symbolId":7,"volume":100_000,"tradeSide":1}},
            "deal":[]
        }}),
    ]
}

// Exercise the exact production closure between the prepared-amend helper and
// the existing symbol resolver, using actual OrderDetails wire envelopes.
fn prepare_bound_amend_fixture(
    transport: &AmendDetailsWireTransport,
    environment: CTraderEnvironment,
    volume: Option<f64>,
    order_type: super::CTraderOrderType,
    resolve: impl FnOnce(&super::CTraderSymbolLookupRequest) -> anyhow::Result<CTraderResolvedSymbol>,
) -> anyhow::Result<super::CTraderExecutionRuntimeRequest> {
    super::prepare_amend_order_from_resolved_creds(
        fixture_creds("42", environment),
        900,
        "EURUSD",
        order_type,
        volume,
        Some(1.25),
        Some(12.0),
        Some(24.0),
        Some(1_800_000_000_000),
        Some(42),
        |request| {
            super::resolve_amend_order_symbol_with_transport(
                transport, request, 900, order_type, resolve,
            )
        },
    )
}

#[test]
fn amendment_details_bind_real_order_before_preparing_exact_limit_stop_wire_fields() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        for order_type in [
            super::CTraderOrderType::Limit,
            super::CTraderOrderType::Stop,
        ] {
            for volume in [None, Some(0.01)] {
                let transport =
                    AmendDetailsWireTransport::new(amendment_details_responses(order_type));
                let lookups = Cell::new(0);
                let runtime = prepare_bound_amend_fixture(
                    &transport,
                    environment,
                    volume,
                    order_type,
                    |request| {
                        lookups.set(lookups.get() + 1);
                        assert_eq!(transport.calls.get(), 1, "order details precede metadata");
                        assert_eq!(request.account_id, "42");
                        assert_eq!(request.environment, environment);
                        assert_eq!(request.client_id, "fixture-client");
                        assert_eq!(request.client_secret, "fixture-secret");
                        assert_eq!(request.access_token, "fixture-token");
                        assert_eq!(request.symbol_name, "EURUSD");
                        let mut symbol = fixture_symbol(42);
                        if volume.is_none() {
                            symbol.symbol.financials = None;
                            symbol.symbol.lot_size = None;
                            symbol.symbol.min_volume = None;
                            symbol.symbol.max_volume = None;
                            symbol.symbol.step_volume = None;
                        }
                        Ok(symbol)
                    },
                )
                .expect("matching pending order and exact metadata");
                assert_eq!(transport.calls.get(), 1);
                assert_eq!(lookups.get(), 1);
                assert_eq!(runtime.environment, environment);
                assert_eq!(runtime.account_id, "42");
                assert_eq!(runtime.client_id, "fixture-client");
                assert_eq!(runtime.client_secret, "fixture-secret");
                assert_eq!(runtime.access_token, "fixture-token");
                let super::CTraderExecutionRequest::AmendOrder(amend) = runtime.request else {
                    panic!("must prepare only the existing amendment request");
                };
                let wire = crate::app_services::ctrader_messages::build_amend_order_request(
                    &amend,
                    "fixture-bound-amend",
                );
                assert_eq!(wire.payload_type, 2109);
                assert_eq!(wire.payload["ctidTraderAccountId"], 42);
                assert_eq!(wire.payload["orderId"], 900);
                assert_eq!(
                    wire.payload
                        .get("volume")
                        .and_then(serde_json::Value::as_i64),
                    volume.map(|_| 100_000)
                );
                assert_eq!(wire.payload["relativeStopLoss"], 120);
                assert_eq!(wire.payload["relativeTakeProfit"], 240);
                assert_eq!(wire.payload["expirationTimestamp"], 1_800_000_000_000_i64);
                let (present, absent) = if order_type == super::CTraderOrderType::Limit {
                    ("limitPrice", "stopPrice")
                } else {
                    ("stopPrice", "limitPrice")
                };
                assert_eq!(wire.payload[present], 1.25);
                assert!(wire.payload.get(absent).is_none());
            }
        }
    }
}

#[test]
fn amendment_details_reject_foreign_or_missing_account_before_parsing_empty_order() {
    for index in [1, 2] {
        for account in [None, Some(99)] {
            let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
            responses[2]["payload"]["order"] = serde_json::Value::Null;
            match account {
                Some(account) => {
                    responses[index]["payload"]["ctidTraderAccountId"] = account.into()
                }
                None => {
                    responses[index]["payload"]
                        .as_object_mut()
                        .unwrap()
                        .remove("ctidTraderAccountId");
                }
            }
            let transport = AmendDetailsWireTransport::new(responses);
            let result = prepare_bound_amend_fixture(
                &transport,
                CTraderEnvironment::Demo,
                None,
                super::CTraderOrderType::Limit,
                |_| panic!("invalid account must never reach symbol metadata"),
            );
            assert!(
                result
                    .expect_err("foreign/missing account")
                    .to_string()
                    .contains("response account")
            );
            assert_eq!(transport.calls.get(), 1);
        }
    }
}

#[test]
fn amendment_details_reject_wrong_order_type_status_and_missing_order_before_lookup() {
    use serde_json::json;
    let base = amendment_details_responses(super::CTraderOrderType::Limit);
    let mut cases = Vec::new();
    for value in [json!(0), json!(-1), json!(901), serde_json::Value::Null] {
        let mut rows = base.clone();
        rows[2]["payload"]["order"]["orderId"] = value;
        cases.push(rows);
    }
    for value in [
        json!(1),
        json!(3),
        json!(4),
        json!(5),
        json!(6),
        json!(99),
        json!(null),
    ] {
        let mut rows = base.clone();
        rows[2]["payload"]["order"]["orderType"] = value;
        cases.push(rows);
    }
    for value in [
        json!(0),
        json!(2),
        json!(3),
        json!(4),
        json!(5),
        json!(99),
        json!(null),
    ] {
        let mut rows = base.clone();
        rows[2]["payload"]["order"]["orderStatus"] = value;
        cases.push(rows);
    }
    for value in [json!(null), json!([]), json!({})] {
        let mut rows = base.clone();
        rows[2]["payload"]["order"] = value;
        cases.push(rows);
    }
    for field in ["orderId", "orderType", "orderStatus", "tradeData"] {
        let mut rows = base.clone();
        rows[2]["payload"]["order"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        cases.push(rows);
    }
    let mut rows = base;
    rows[2]["payload"].as_object_mut().unwrap().remove("order");
    cases.push(rows);
    for responses in cases {
        for volume in [None, Some(0.01)] {
            let transport = AmendDetailsWireTransport::new(responses.clone());
            assert!(
                prepare_bound_amend_fixture(
                    &transport,
                    CTraderEnvironment::Demo,
                    volume,
                    super::CTraderOrderType::Limit,
                    |_| panic!("invalid order must never reach metadata or bracket conversion"),
                )
                .is_err()
            );
            assert_eq!(transport.calls.get(), 1);
        }
    }
}

#[test]
fn amendment_order_symbol_must_match_both_metadata_ids_before_bracket_conversion() {
    for volume in [None, Some(0.01)] {
        for wrong_field in ["order", "light", "full", "account"] {
            let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
            if wrong_field == "order" {
                responses[2]["payload"]["order"]["tradeData"]["symbolId"] = 8.into();
            }
            let transport = AmendDetailsWireTransport::new(responses);
            let lookups = Cell::new(0);
            let result = prepare_bound_amend_fixture(
                &transport,
                CTraderEnvironment::Demo,
                volume,
                super::CTraderOrderType::Limit,
                |_| {
                    lookups.set(lookups.get() + 1);
                    let mut symbol = fixture_symbol(42);
                    match wrong_field {
                        "light" => symbol.light_symbol.symbol_id = 8,
                        "full" => symbol.symbol.symbol_id = 8,
                        "account" => symbol.account_id = 99,
                        _ => {}
                    }
                    // Identity rejection precedes both volume and bracket math.
                    symbol.symbol.financials = None;
                    symbol.symbol.digits = 6;
                    Ok(symbol)
                },
            );
            assert!(
                result
                    .expect_err("unrelated metadata")
                    .to_string()
                    .contains("order symbol/account differs")
            );
            assert_eq!(lookups.get(), 1);
            assert_eq!(transport.calls.get(), 1);
        }
    }
    for value in [
        serde_json::json!(0),
        serde_json::json!(-1),
        serde_json::Value::Null,
    ] {
        let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
        responses[2]["payload"]["order"]["tradeData"]["symbolId"] = value;
        let transport = AmendDetailsWireTransport::new(responses);
        assert!(
            prepare_bound_amend_fixture(
                &transport,
                CTraderEnvironment::Demo,
                None,
                super::CTraderOrderType::Limit,
                |_| panic!("invalid source symbol must never reach lookup"),
            )
            .is_err()
        );
    }
}

#[test]
fn amendment_details_keep_broker_error_type_and_transport_failures_without_fallback() {
    use serde_json::json;
    for index in 0..3 {
        for full_length in [false, true] {
            let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
            responses[index] = json!({"payloadType":2142,"payload":{
                "errorCode":"ACCESS_DENIED","description":"fixture details denied","retryAfter":120
            }});
            if !full_length {
                responses.truncate(index + 1);
            }
            let transport = AmendDetailsWireTransport::new(responses);
            let error = prepare_bound_amend_fixture(
                &transport,
                CTraderEnvironment::Demo,
                None,
                super::CTraderOrderType::Limit,
                |_| panic!("broker error must not reach metadata"),
            )
            .expect_err("original broker error must be preserved");
            let message = error.to_string();
            assert!(
                message.contains("ACCESS_DENIED")
                    && message.contains("fixture details denied")
                    && message.contains("retryAfter=120s"),
                "{message}"
            );
            assert!(!message.contains("response account"), "{message}");
            assert_eq!(transport.calls.get(), 1, "lookup is not an execution retry");
        }
    }
    let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
    responses[2] = json!({"payloadType":2122,"payload":{}});
    let transport = AmendDetailsWireTransport::new(responses);
    let error = prepare_bound_amend_fixture(
        &transport,
        CTraderEnvironment::Demo,
        None,
        super::CTraderOrderType::Limit,
        |_| panic!("wrong payload cannot reach metadata"),
    )
    .expect_err("wrong payload");
    assert!(error.to_string().contains("payload type"));
    assert!(!error.to_string().contains("response account"));

    let mut transport = AmendDetailsWireTransport::new(Vec::new());
    transport.fail = true;
    let error = prepare_bound_amend_fixture(
        &transport,
        CTraderEnvironment::Demo,
        None,
        super::CTraderOrderType::Limit,
        |_| panic!("transport error cannot reach metadata"),
    )
    .expect_err("transport error");
    assert_eq!(error.to_string(), "fixture order-details transport failure");
}

#[test]
fn amendment_details_reject_empty_partial_extra_responses_and_invalid_request_identity() {
    for length in [0, 1, 2, 4] {
        let mut responses = amendment_details_responses(super::CTraderOrderType::Limit);
        if length == 4 {
            responses.push(responses[2].clone());
        } else {
            responses.truncate(length);
        }
        let transport = AmendDetailsWireTransport::new(responses);
        let error = prepare_bound_amend_fixture(
            &transport,
            CTraderEnvironment::Demo,
            None,
            super::CTraderOrderType::Limit,
            |_| panic!("incomplete sequence cannot reach metadata"),
        )
        .expect_err("exact response count required");
        assert!(error.to_string().contains("expected 3"), "{error:#}");
        assert_eq!(transport.calls.get(), 1);
    }
    for (account, pin, order_id) in [
        ("99", Some(42), 900),
        ("0", None, 900),
        ("invalid", None, 900),
        ("42", Some(42), 0),
        ("42", None, -1),
    ] {
        let transport = AmendDetailsWireTransport::new(Vec::new());
        let result = super::prepare_amend_order_from_resolved_creds(
            fixture_creds(account, CTraderEnvironment::Demo),
            order_id,
            "EURUSD",
            super::CTraderOrderType::Limit,
            None,
            Some(1.25),
            None,
            None,
            None,
            pin,
            |request| {
                super::resolve_amend_order_symbol_with_transport(
                    &transport,
                    request,
                    order_id,
                    super::CTraderOrderType::Limit,
                    |_| panic!("invalid request cannot reach metadata"),
                )
            },
        );
        assert!(result.is_err());
        assert_eq!(transport.calls.get(), 0);
    }
}

#[test]
fn bound_volume_amendment_still_refuses_missing_grid_and_propagates_symbol_errors() {
    let transport =
        AmendDetailsWireTransport::new(amendment_details_responses(super::CTraderOrderType::Limit));
    let error = prepare_bound_amend_fixture(
        &transport,
        CTraderEnvironment::Demo,
        Some(0.01),
        super::CTraderOrderType::Limit,
        |_| {
            let mut symbol = fixture_symbol(42);
            symbol.symbol.step_volume = None;
            Ok(symbol)
        },
    )
    .expect_err("binding the order does not supply missing entry volume evidence");
    assert!(error.to_string().contains("stepVolume"));
    let transport =
        AmendDetailsWireTransport::new(amendment_details_responses(super::CTraderOrderType::Limit));
    let error = prepare_bound_amend_fixture(
        &transport,
        CTraderEnvironment::Demo,
        None,
        super::CTraderOrderType::Limit,
        |_| Err(anyhow::anyhow!("fixture symbol lookup failed")),
    )
    .expect_err("existing resolver failure");
    assert_eq!(error.to_string(), "fixture symbol lookup failed");
}
