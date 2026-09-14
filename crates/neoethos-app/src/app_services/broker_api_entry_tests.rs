use super::*;
use serde_json::{Value, json};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

struct Wire {
    responses: Vec<Value>,
    calls: Cell<usize>,
}
impl CTraderOpenApiTransport for Wire {
    fn send_sequence(
        &self,
        messages: &[crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage],
    ) -> Result<Vec<String>> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(messages.len(), 8);
        assert_eq!(
            messages.iter().map(|m| m.payload_type).collect::<Vec<_>>(),
            vec![2100, 2102, 2167, 2121, 2124, 2187, 2112, 2114]
        );
        for message in &messages[1..] {
            assert_eq!(message.payload["ctidTraderAccountId"], 42);
        }
        Ok(self.responses.iter().map(Value::to_string).collect())
    }
}

fn creds(account_id: &str) -> ResolvedCreds {
    ResolvedCreds {
        client_id: "synthetic-client".into(),
        client_secret: "synthetic-secret".into(),
        access_token: "synthetic-token".into(),
        account_id_str: account_id.into(),
        environment: CTraderEnvironment::Demo,
        env_label: "Demo",
    }
}
fn wire() -> Wire {
    Wire {
        calls: Cell::new(0),
        responses: vec![
            json!({"payloadType":2101,"payload":{}}),
            json!({"payloadType":2103,"payload":{"ctidTraderAccountId":42}}),
            json!({"payloadType":2168,"payload":{"ctidTraderAccountId":42,
            "marginCall":[{"marginCallType":1,"marginLevelThreshold":500.0}]}}),
            json!({"payloadType":2122,"payload":{"ctidTraderAccountId":42,
            "trader":{"balance":1_000_000,"moneyDigits":2,"depositAssetId":8}}}),
            json!({"payloadType":2125,"payload":{"ctidTraderAccountId":42,"position":[],"order":[]}}),
            json!({"payloadType":2188,"payload":{"ctidTraderAccountId":42,"moneyDigits":8,
            "positionUnrealizedPnL":[]}}),
            json!({"payloadType":2113,"payload":{"ctidTraderAccountId":42,"asset":[
            {"assetId":4,"name":"EUR"},{"assetId":8,"name":"USD"},{"assetId":9,"name":"GBP"}]}}),
            json!({"payloadType":2115,"payload":{"ctidTraderAccountId":42,"symbol":[
            {"symbolId":7,"symbolName":"EURUSD","enabled":true,"baseAssetId":4,"quoteAssetId":8}]}}),
        ],
    }
}
fn full_symbol(wire: &Wire) -> CTraderResolvedSymbol {
    let payload = json!({"payloadType":2117,"payload":{"ctidTraderAccountId":42,"symbol":[{
        "symbolId":7,"digits":5,"pipPosition":4,"tradingMode":0,"enableShortSelling":true,
        "lotSize":10_000_000,"minVolume":100_000,"maxVolume":1_000_000_000,
        "stepVolume":100_000
    }]}});
    let light = parse_symbols_list_response(&wire.responses[7].to_string())
        .unwrap()
        .symbols
        .remove(0);
    let mut symbol = parse_symbol_by_id_response(&payload.to_string())
        .unwrap()
        .remove(0);
    symbol.symbol_name = light.symbol_name.clone();
    symbol.display_name = light.symbol_name.clone();
    CTraderResolvedSymbol {
        account_id: 42,
        light_symbol: light,
        symbol,
    }
}
fn context(wire: &Wire) -> Result<LiveEntryContext> {
    fetch_with_transport(wire, &creds("42"), "EURUSD", 42, 7, |_| {
        Ok(full_symbol(wire))
    })
}
fn valuation(_: &LiveEntryContext) -> Result<LiveEntryValuation> {
    Ok(LiveEntryValuation {
        bid: 1.1,
        ask: 1.1001,
        price_for_risk: 1.1001,
        quote_to_account_rate: 1.0,
        quote_valid_until: Instant::now() + Duration::from_secs(120),
    })
}

#[test]
fn flat_first_entry_uses_real_parsers_and_shared_wire_without_closed_history() {
    let wire = wire();
    let ctx = context(&wire).unwrap();
    assert_eq!(wire.calls.get(), 1);
    assert_eq!(
        (
            ctx.margin().balance,
            ctx.margin().equity,
            ctx.margin().open_position_count
        ),
        (10_000.0, 10_000.0, 0)
    );
    assert_eq!(ctx.account_currency(), "USD");
    assert_eq!(ctx.trader_money_digits(), 2);
    assert_eq!(
        (
            ctx.metadata().pip_size,
            ctx.metadata().contract_size,
            ctx.metadata().pip_value_quote
        ),
        (0.0001, 100_000.0, 10.0)
    );
    assert_eq!(
        ctx.metadata()
            .risk_money_to_lots(100.0, 20.0, "USD", Some(1.0), Some(1.1001)),
        Some(0.5)
    );
    let client_order_id = ctx.client_order_id().to_owned();
    assert_eq!(client_order_id.len(), 36);
    let marker = AtomicBool::new(false);
    let calls = Cell::new(0);
    let error = ctx
        .submit_with(
            creds("42"),
            OrderSide::Buy,
            0.5,
            Some(20.0),
            Some(40.0),
            None,
            120_000,
            |_| Ok(full_symbol(&wire)),
            valuation,
            |metadata, value| {
                assert_eq!(metadata.pip_value_quote, 10.0);
                assert_eq!(value.quote_to_account_rate, 1.0);
                assert!(!marker.load(Ordering::Acquire));
                Ok(())
            },
            &marker,
            |_| Ok(()),
            Instant::now,
            |request| {
                calls.set(calls.get() + 1);
                assert!(marker.load(Ordering::Acquire));
                assert_eq!(request.environment, CTraderEnvironment::Demo);
                assert_eq!(request.account_id, "42");
                let CTraderExecutionRequest::NewOrder(order) = &request.request else {
                    panic!("market request required")
                };
                assert_eq!(
                    order.client_order_id.as_deref(),
                    Some(client_order_id.as_str())
                );
                assert_eq!(
                    (order.account_id, order.symbol_id, order.volume),
                    (42, 7, 5_000_000)
                );
                assert_eq!(
                    (order.relative_stop_loss, order.relative_take_profit),
                    (Some(200), Some(400))
                );
                Err(anyhow!(
                    "synthetic executor reached; no network or filled-position claim"
                ))
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("synthetic executor reached"));
    assert_eq!(calls.get(), 1);
    assert!(
        marker.load(Ordering::Acquire),
        "post-send errors stay indeterminate, not definite refusal"
    );
}

#[test]
fn actual_entry_request_is_durable_before_the_only_executor_call_and_blocks_restart() {
    use crate::app_services::account_risk::{AccountRiskIdentity, AccountRiskRegistry};
    let root = std::env::temp_dir().join(format!(
        "neoethos-entry-intent-wire-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let identity = AccountRiskIdentity::new("demo", 42, "USD").unwrap();
    let registry = AccountRiskRegistry::new();
    let authority = registry.acquire_entry(identity.clone(), &root).unwrap();
    authority.lock().unwrap().prepare_day(20_260_908).unwrap();
    authority
        .lock()
        .unwrap()
        .try_reserve_entry(20_260_908, None)
        .unwrap();
    let wire = wire();
    let ctx = context(&wire).unwrap();
    let client_id = ctx.client_order_id().to_owned();
    let marker = std::sync::Arc::new(AtomicBool::new(false));
    let calls = Cell::new(0);
    let error = ctx
        .submit_with(
            creds("42"),
            OrderSide::Buy,
            0.5,
            Some(20.0),
            Some(40.0),
            None,
            120_000,
            |_| Ok(full_symbol(&wire)),
            valuation,
            |_, _| Ok(()),
            &marker,
            |request| {
                assert!(!marker.load(Ordering::Acquire));
                authority
                    .lock()
                    .unwrap()
                    .begin_submission(20_260_908, request, &marker)
            },
            Instant::now,
            |request| {
                calls.set(calls.get() + 1);
                assert!(marker.load(Ordering::Acquire));
                let restored = AccountRiskRegistry::new()
                    .acquire_entry(identity.clone(), &root)
                    .unwrap();
                assert_eq!(
                    restored.lock().unwrap().unresolved_client_order_id(),
                    Some(client_id.as_str())
                );
                let CTraderExecutionRequest::NewOrder(order) = &request.request else {
                    panic!("entry")
                };
                let path = root
                    .join("runtime")
                    .join("account-entry")
                    .join("demo-42.json");
                let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                assert_eq!(saved["unresolvedIntent"]["symbolId"], order.symbol_id);
                assert_eq!(
                    saved["unresolvedIntent"]["volumeRawCentiUnits"],
                    order.volume
                );
                Err(anyhow!("synthetic unresolved execution; no network"))
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("synthetic unresolved"));
    assert_eq!(calls.get(), 1);
    let restored = AccountRiskRegistry::new()
        .acquire_entry(identity, &root)
        .unwrap();
    let mut state = restored.lock().unwrap();
    state.prepare_day(20_260_909).unwrap();
    assert_eq!(
        state.try_reserve_entry(20_260_909, None).unwrap_err().rule,
        "risk.unresolved_entry"
    );
    drop(state);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn failed_durable_intent_callback_never_marks_or_invokes_execution() {
    let wire = wire();
    let marker = AtomicBool::new(false);
    let calls = Cell::new(0);
    let persists = Cell::new(0);
    let result = context(&wire).unwrap().submit_with(
        creds("42"),
        OrderSide::Buy,
        0.5,
        Some(20.0),
        Some(40.0),
        None,
        120_000,
        |_| Ok(full_symbol(&wire)),
        valuation,
        |_, _| Ok(()),
        &marker,
        |_| {
            persists.set(persists.get() + 1);
            Err(anyhow!("synthetic durable write failure"))
        },
        Instant::now,
        |_| {
            calls.set(calls.get() + 1);
            Err(anyhow!("must not execute"))
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("durable write failure")
    );
    assert_eq!(persists.get(), 1);
    assert_eq!(calls.get(), 0);
    assert!(!marker.load(Ordering::Acquire));
}

#[test]
fn delayed_successful_intent_persistence_expires_original_observations_without_send() {
    use crate::app_services::account_risk::{AccountRiskIdentity, AccountRiskRegistry};
    for expired in ["primary", "conversion", "account"] {
        let root = std::env::temp_dir().join(format!(
            "neoethos-entry-expiry-{expired}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let identity = AccountRiskIdentity::new("demo", 42, "GBP").unwrap();
        let authority = AccountRiskRegistry::new()
            .acquire_entry(identity.clone(), &root)
            .unwrap();
        authority.lock().unwrap().prepare_day(20_260_908).unwrap();
        authority
            .lock()
            .unwrap()
            .try_reserve_entry(20_260_908, None)
            .unwrap();
        let mut wire = wire();
        wire.responses[3]["payload"]["trader"]["depositAssetId"] = json!(9);
        wire.responses[7]["payload"]["symbol"].as_array_mut().unwrap().push(
            json!({"symbolId":19,"symbolName":"USDGBP","enabled":true,"baseAssetId":8,"quoteAssetId":9}));
        let ctx = context(&wire).unwrap();
        let client_id = ctx.client_order_id().to_owned();
        let marker = std::sync::Arc::new(AtomicBool::new(false));
        let persists = Cell::new(0);
        let calls = Cell::new(0);
        let clock = Cell::new(Instant::now());
        let session = SpotSessionId::synthetic_for_test(1);
        let error = ctx
            .submit_with(
                creds("42"),
                OrderSide::Buy,
                0.5,
                Some(20.0),
                Some(40.0),
                None,
                120_000,
                |_| Ok(full_symbol(&wire)),
                |context| {
                    context.valuation_with(session, 119_001, 120_000, |id| {
                        let mut quote = synthetic_quote(session, id, 1.25, 1.26);
                        quote.bid_received_at_unix_ms = 119_001;
                        quote.ask_received_at_unix_ms = 119_001;
                        quote.bid_broker_timestamp_ms = 119_001;
                        quote.ask_broker_timestamp_ms = 119_001;
                        if (expired == "primary" && id == 7)
                            || (expired == "conversion" && id == 19)
                        {
                            quote.bid_broker_timestamp_ms = 1; // 1,000 ms remain, although receipt is recent.
                        }
                        Ok(quote)
                    })
                },
                |_, _| Ok(()),
                &marker,
                |request| {
                    authority
                        .lock()
                        .unwrap()
                        .begin_submission(20_260_908, request, &marker)?;
                    persists.set(persists.get() + 1);
                    // Synthetic successful fsync delay; no sleeping or network.
                    clock.set(
                        Instant::now()
                            + Duration::from_secs(if expired == "account" { 121 } else { 2 }),
                    );
                    Ok(())
                },
                || clock.get(),
                |_| {
                    calls.set(calls.get() + 1);
                    Err(anyhow!("must not execute"))
                },
            )
            .unwrap_err();
        assert!(
            error.to_string().contains(if expired == "account" {
                "account/symbol observation expired"
            } else {
                "primary/conversion quote expired during intent persistence"
            }),
            "{expired}: {error:#}"
        );
        assert_eq!(persists.get(), 1, "{expired}");
        assert_eq!(calls.get(), 0, "{expired}");
        assert!(!marker.load(Ordering::Acquire), "{expired}");
        let restored = AccountRiskRegistry::new()
            .acquire_entry(identity.clone(), &root)
            .unwrap();
        assert_eq!(
            restored.lock().unwrap().unresolved_client_order_id(),
            Some(client_id.as_str())
        );
        assert!(
            restored
                .lock()
                .unwrap()
                .release_unsent_entry(20_260_908, &client_id, &marker)
                .is_err(),
            "a restarted authority cannot borrow the live attempt marker"
        );
        {
            let mut state = authority.lock().unwrap();
            assert_eq!(
                state.try_reserve_entry(20_260_908, None).unwrap_err().rule,
                "risk.unresolved_entry"
            );
            // Only the completed, zero-executor local attempt may clear its exact intent.
            state
                .release_unsent_entry(20_260_908, &client_id, &marker)
                .unwrap();
            assert_eq!(state.unresolved_client_order_id(), None);
            assert_eq!(state.entries_today(), 0);
        }
        let reloaded = AccountRiskRegistry::new()
            .acquire_entry(identity, &root)
            .unwrap();
        let mut state = reloaded.lock().unwrap();
        assert_eq!(state.unresolved_client_order_id(), None);
        assert_eq!(state.try_reserve_entry(20_260_908, Some(1)).unwrap(), 0);
        assert_eq!(calls.get(), 0, "clearing is not an executor retry");
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }
}

#[test]
fn durable_entry_authority_refuses_different_request_account_or_environment() {
    use crate::app_services::account_risk::{AccountRiskIdentity, AccountRiskRegistry};
    let root = std::env::temp_dir().join(format!(
        "neoethos-entry-scope-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let wire = wire();
    for identity in [
        AccountRiskIdentity::new("demo", 99, "USD").unwrap(),
        AccountRiskIdentity::new("live", 42, "USD").unwrap(),
    ] {
        let authority = AccountRiskRegistry::new()
            .acquire_entry(identity, &root)
            .unwrap();
        authority.lock().unwrap().prepare_day(20_260_908).unwrap();
        authority
            .lock()
            .unwrap()
            .try_reserve_entry(20_260_908, None)
            .unwrap();
        let marker = std::sync::Arc::new(AtomicBool::new(false));
        let calls = Cell::new(0);
        let result = context(&wire).unwrap().submit_with(
            creds("42"),
            OrderSide::Buy,
            0.5,
            Some(20.0),
            Some(40.0),
            None,
            120_000,
            |_| Ok(full_symbol(&wire)),
            valuation,
            |_, _| Ok(()),
            &marker,
            |request| {
                authority
                    .lock()
                    .unwrap()
                    .begin_submission(20_260_908, request, &marker)
            },
            Instant::now,
            |_| {
                calls.set(calls.get() + 1);
                Err(anyhow!("must not execute"))
            },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("exact account/environment")
        );
        assert!(!marker.load(Ordering::Acquire));
        assert_eq!(calls.get(), 0);
        assert_eq!(authority.lock().unwrap().unresolved_client_order_id(), None);
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn context_rejects_foreign_assets_symbols_and_ambiguous_asset_names() {
    for index in [6, 7] {
        let mut wire = wire();
        wire.responses[index]["payload"]["ctidTraderAccountId"] = json!(99);
        assert!(context(&wire).is_err());
    }
    let mut missing = wire();
    missing.responses[3]["payload"]["trader"]
        .as_object_mut()
        .unwrap()
        .remove("depositAssetId");
    assert!(context(&missing).is_err());
    let mut ambiguous = wire();
    ambiguous.responses[6]["payload"]["asset"]
        .as_array_mut()
        .unwrap()
        .push(json!({"assetId":99,"name":"USD"}));
    assert!(context(&ambiguous).is_err());
    let original = wire();
    for missing in ["lot", "min", "max", "step"] {
        let mut full = full_symbol(&original);
        match missing {
            "lot" => full.symbol.lot_size = None,
            "min" => full.symbol.min_volume = None,
            "max" => full.symbol.max_volume = None,
            _ => full.symbol.step_volume = None,
        }
        assert!(
            fetch_with_transport(&original, &creds("42"), "EURUSD", 42, 7, |_| Ok(full)).is_err()
        );
    }
}

#[test]
fn pre_send_rebinding_expiry_and_budget_errors_never_reach_executor() {
    for case in [
        "account",
        "environment",
        "symbol",
        "grid",
        "expired",
        "quote",
        "budget",
    ] {
        let wire = wire();
        let mut ctx = context(&wire).unwrap();
        let mut final_creds = creds(if case == "account" { "99" } else { "42" });
        if case == "environment" {
            final_creds.environment = CTraderEnvironment::Live;
        }
        if case == "expired" {
            ctx.request_started = Instant::now() - Duration::from_secs(121);
        }
        let marker = AtomicBool::new(false);
        let calls = Cell::new(0);
        let result = ctx.submit_with(
            final_creds,
            OrderSide::Buy,
            0.5,
            Some(20.0),
            Some(40.0),
            None,
            120_000,
            |_| {
                let mut full = full_symbol(&wire);
                if case == "symbol" {
                    full.symbol.symbol_id = 8;
                    full.light_symbol.symbol_id = 8;
                }
                if case == "grid" {
                    full.symbol.step_volume = Some(200_000);
                }
                Ok(full)
            },
            |context| {
                if case == "quote" {
                    Err(anyhow!("same-session quote expired"))
                } else {
                    valuation(context)
                }
            },
            |_, _| {
                if case == "budget" {
                    Err(anyhow!("fresh conversion no longer fits budget"))
                } else {
                    Ok(())
                }
            },
            &marker,
            |_| Ok(()),
            Instant::now,
            |_| {
                calls.set(calls.get() + 1);
                Err(anyhow!("must never execute"))
            },
        );
        assert!(result.is_err(), "{case}");
        assert_eq!(calls.get(), 0, "{case}");
        assert!(
            !marker.load(Ordering::Acquire),
            "{case} is definitely unsent"
        );
    }
}

#[test]
fn direct_and_inverse_loss_conversion_use_asset_identity_and_conservative_sides() {
    let mut direct = wire();
    direct.responses[3]["payload"]["trader"]["depositAssetId"] = json!(9);
    direct.responses[7]["payload"]["symbol"].as_array_mut().unwrap().push(
        json!({"symbolId":19,"symbolName":"BROKER_SUFFIX_X","enabled":true,"baseAssetId":8,"quoteAssetId":9}));
    assert!(matches!(
        context(&direct).unwrap().loss_conversion,
        LossConversion::Direct(19)
    ));
    assert_eq!(loss_rate_from_prices(0.79, 0.80, false).unwrap(), 0.80);
    let mut inverse = wire();
    inverse.responses[3]["payload"]["trader"]["depositAssetId"] = json!(9);
    inverse.responses[7]["payload"]["symbol"].as_array_mut().unwrap().push(
        json!({"symbolId":20,"symbolName":"NOT_A_SIX_LETTER_NAME","enabled":true,"baseAssetId":9,"quoteAssetId":8}));
    assert!(matches!(
        context(&inverse).unwrap().loss_conversion,
        LossConversion::Inverse(20)
    ));
    assert_eq!(loss_rate_from_prices(1.25, 1.26, true).unwrap(), 0.8);
    for prices in [
        (0.0, 1.0),
        (1.1, 1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        assert!(loss_rate_from_prices(prices.0, prices.1, false).is_err());
    }
    let mut missing = wire();
    missing.responses[3]["payload"]["trader"]["depositAssetId"] = json!(9);
    assert!(
        context(&missing)
            .err()
            .unwrap()
            .to_string()
            .contains("no direct broker asset-linked")
    );
    let mut ambiguous = direct.responses[7]["payload"]["symbol"][1].clone();
    ambiguous["symbolId"] = json!(21);
    direct.responses[7]["payload"]["symbol"]
        .as_array_mut()
        .unwrap()
        .push(ambiguous);
    assert!(
        context(&direct)
            .err()
            .unwrap()
            .to_string()
            .contains("ambiguous")
    );
}

#[test]
fn bound_pip_lookup_needs_no_margin_or_previous_trades() {
    let wire = wire();
    let full =
        resolve_bound_symbol_from_creds(&creds("42"), "EURUSD", 42, 7, |_| Ok(full_symbol(&wire)))
            .unwrap();
    assert_eq!(10.0_f64.powi(-full.symbol.pip_position), 0.0001);
    assert_eq!(
        wire.calls.get(),
        0,
        "read-only full-symbol lookup performs no margin/history request"
    );
    let lookups = Cell::new(0);
    assert!(
        resolve_bound_symbol_from_creds(&creds("99"), "EURUSD", 42, 7, |_| {
            lookups.set(lookups.get() + 1);
            Ok(full_symbol(&wire))
        })
        .is_err()
    );
    assert_eq!(
        lookups.get(),
        0,
        "foreign account refuses before full-symbol lookup"
    );
    assert!(
        resolve_bound_symbol_from_creds(&creds("42"), "EURUSD", 42, 8, |_| Ok(full_symbol(&wire)))
            .is_err()
    );
}

#[test]
fn distinct_logical_intents_keep_their_own_wire_correlation_after_error() {
    let wire = wire();
    let mut ids = Vec::new();
    for nonce in [1, 2] {
        let mut ctx = context(&wire).unwrap();
        ctx.client_order_id = format_entry_client_order_id(nonce);
        let original = ctx.client_order_id().to_owned();
        let marker = AtomicBool::new(false);
        let error = ctx
            .submit_with(
                creds("42"),
                OrderSide::Buy,
                0.5,
                Some(20.0),
                Some(40.0),
                None,
                120_000,
                |_| Ok(full_symbol(&wire)),
                valuation,
                |_, _| Ok(()),
                &marker,
                |_| Ok(()),
                Instant::now,
                |request| {
                    let CTraderExecutionRequest::NewOrder(order) = &request.request else {
                        panic!("new order required");
                    };
                    assert_eq!(order.client_order_id.as_deref(), Some(original.as_str()));
                    ids.push(order.client_order_id.clone().unwrap());
                    Err(anyhow!(
                        "synthetic indeterminate result; no retry in this producer"
                    ))
                },
            )
            .unwrap_err();
        assert!(marker.load(Ordering::Acquire));
        assert!(error.to_string().contains("indeterminate"));
        assert_eq!(ids.last(), Some(&original));
    }
    assert_ne!(
        ids[0], ids[1],
        "identical order parameters are separate logical intents"
    );
    assert!(ids.iter().all(|id| id.len() <= 50));
}

fn synthetic_quote(
    session_id: SpotSessionId,
    symbol_id: i64,
    bid: f64,
    ask: f64,
) -> FreshSpotQuote {
    FreshSpotQuote {
        session_id,
        account_id: 42,
        environment: CTraderEnvironment::Demo,
        symbol_id,
        bid,
        ask,
        bid_received_at_unix_ms: 1_000,
        ask_received_at_unix_ms: 1_000,
        bid_broker_timestamp_ms: 1_000,
        ask_broker_timestamp_ms: 1_000,
    }
}

#[test]
fn valuation_wiring_uses_primary_bid_for_base_currency_and_exact_conversion_leg() {
    let session = SpotSessionId::synthetic_for_test(1);
    for (deposit, orientation, expected_ids, expected_price, expected_rate) in [
        (8, None, vec![7], 1.26, 1.0),
        (4, None, vec![7], 1.25, 0.8),
        (9, Some((19, 8, 9)), vec![7, 19], 1.26, 0.8),
        (9, Some((20, 9, 8)), vec![7, 20], 1.26, 0.8),
    ] {
        let mut wire = wire();
        wire.responses[3]["payload"]["trader"]["depositAssetId"] = json!(deposit);
        if let Some((id, base, quote)) = orientation {
            wire.responses[7]["payload"]["symbol"]
                .as_array_mut()
                .unwrap()
                .push(
                    json!({"symbolId":id,"symbolName":"BROKER_CONVERSION","enabled":true,
                       "baseAssetId":base,"quoteAssetId":quote}),
                );
        }
        let ctx = context(&wire).unwrap();
        let mut requested = Vec::new();
        let actual = ctx
            .valuation_with(session, 1_000, 120_000, |id| {
                requested.push(id);
                Ok(if id == 19 {
                    synthetic_quote(session, id, 0.79, 0.80)
                } else {
                    synthetic_quote(session, id, 1.25, 1.26)
                })
            })
            .unwrap();
        assert_eq!((actual.bid, actual.ask), (1.25, 1.26));
        assert_eq!(actual.price_for_risk, expected_price);
        assert_eq!(actual.quote_to_account_rate, expected_rate);
        assert_eq!(
            requested, expected_ids,
            "base-account inverse reuses the primary quote"
        );
    }
}

#[test]
fn valuation_wiring_refuses_rebound_primary_or_conversion_quote_and_expired_context() {
    let session = SpotSessionId::synthetic_for_test(1);
    let foreign_session = SpotSessionId::synthetic_for_test(2);
    let mut wire = wire();
    wire.responses[3]["payload"]["trader"]["depositAssetId"] = json!(9);
    wire.responses[7]["payload"]["symbol"]
        .as_array_mut()
        .unwrap()
        .push(
            json!({"symbolId":19,"symbolName":"BROKER_CONVERSION","enabled":true,
               "baseAssetId":8,"quoteAssetId":9}),
        );
    let mut ctx = context(&wire).unwrap();
    for target in [7, 19] {
        for field in [
            "account",
            "environment",
            "symbol",
            "session",
            "prices",
            "unavailable",
        ] {
            let result = ctx.valuation_with(session, 1_000, 120_000, |id| {
                let mut quote = synthetic_quote(session, id, 1.25, 1.26);
                if id == target {
                    match field {
                        "account" => quote.account_id = 99,
                        "environment" => quote.environment = CTraderEnvironment::Live,
                        "symbol" => quote.symbol_id = 999,
                        "session" => quote.session_id = foreign_session,
                        "prices" => quote.bid = 2.0,
                        _ => return Err(anyhow!("injected quote unavailable/stale")),
                    }
                }
                Ok(quote)
            });
            assert!(result.is_err(), "target={target} field={field}");
        }
    }
    ctx.request_started = Instant::now() - Duration::from_secs(121);
    let calls = Cell::new(0);
    assert!(
        ctx.valuation_with(session, 1_000, 120_000, |id| {
            calls.set(calls.get() + 1);
            Ok(synthetic_quote(session, id, 1.25, 1.26))
        })
        .is_err()
    );
    assert_eq!(
        calls.get(),
        0,
        "expired account observation refuses before requesting quotes"
    );
}
