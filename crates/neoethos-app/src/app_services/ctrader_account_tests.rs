// TODO(real-data): every JSON value in this file is a hand-built
// model (e.g. balance=123456789, brokerName="Demo Broker", price=1.10123).
// Replace each fixture with a captured demo-account ProtoOATrader /
// ProtoOAReconcileRes / ProtoOADealList response so the parser is
// validated against bytes the broker actually emits — including
// fields cTrader marks optional but our parser silently drops.
use super::*;

use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::CTraderOpenApiJsonMessage;

#[test]
fn trader_response_parses_balance_and_account_metadata() {
    let response = serde_json::json!({
        "clientMsgId": "trader-1",
        "payloadType": 2122,
        "payload": {
            "ctidTraderAccountId": 712345,
            "trader": {
                "balance": 123456789,
                "moneyDigits": 2,
                "leverageInCents": 5000,
                "traderLogin": 998877,
                "accountType": 1,
                "brokerName": "Spotware Demo Broker"
            }
        }
    });

    let trader = parse_trader_response(&response.to_string()).expect("trader response");

    assert_eq!(trader.account_id, 712345);
    assert!((trader.balance - 1_234_567.89).abs() < 1e-9);
    assert_eq!(trader.leverage, Some(50.0));
    assert_eq!(trader.trader_login, Some(998877));
    assert_eq!(trader.account_type.as_deref(), Some("NETTED"));
    assert_eq!(trader.broker_name.as_deref(), Some("Spotware Demo Broker"));
}

#[test]
fn reconcile_response_parses_positions_and_pending_orders() {
    let response = serde_json::json!({
        "clientMsgId": "reconcile-1",
        "payloadType": 2125,
        "payload": {
            "ctidTraderAccountId": 712345,
            "position": [
                {
                    "positionId": 9001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": 2500,
                        "tradeSide": 1,
                        "openTimestamp": 1710000000000i64,
                        "label": "trend",
                        "comment": "bot"
                    },
                    "positionStatus": 1,
                    "price": 1.10123,
                    "stopLoss": 1.095,
                    "takeProfit": 1.11
                }
            ],
            "order": [
                {
                    "orderId": 8001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": 1500,
                        "tradeSide": 2,
                        "openTimestamp": 1710000100000i64,
                        "label": "breakout",
                        "comment": "pending"
                    },
                    "orderType": 2,
                    "orderStatus": 1,
                    "limitPrice": 1.099,
                    "stopLoss": 1.105,
                    "takeProfit": 1.09
                }
            ]
        }
    });

    let reconcile = parse_reconcile_response(&response.to_string()).expect("reconcile");

    assert_eq!(reconcile.account_id, 712345);
    assert_eq!(reconcile.positions.len(), 1);
    assert_eq!(reconcile.pending_orders.len(), 1);
    assert_eq!(reconcile.positions[0].position_id, 9001);
    assert_eq!(reconcile.positions[0].trade_side, "BUY");
    assert_eq!(reconcile.positions[0].symbol_id, 14);
    assert_eq!(reconcile.positions[0].volume_raw_centi_units, 2500);
    assert!((reconcile.positions[0].volume - 25.0).abs() < 1e-9);
    assert_eq!(reconcile.pending_orders[0].order_id, 8001);
    assert_eq!(reconcile.pending_orders[0].trade_side, "SELL");
    assert_eq!(reconcile.pending_orders[0].order_type, "LIMIT");
    assert!((reconcile.pending_orders[0].limit_price.unwrap_or_default() - 1.099).abs() < 1e-9);
}

#[test]
fn reconcile_response_preserves_exact_positive_position_wire_volume() {
    // The odd value above 2^53 cannot survive a round-trip through display f64.
    for wire_volume in [1_i64, 2501, 9_007_199_254_740_993, i64::MAX] {
        let response = serde_json::json!({
            "payloadType": 2125,
            "payload": {
                "ctidTraderAccountId": 712345,
                "position": [{
                    "positionId": 9001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": wire_volume,
                        "tradeSide": 1
                    }
                }]
            }
        });
        let reconcile = parse_reconcile_response(&response.to_string()).expect("positive volume");
        let position = &reconcile.positions[0];
        assert_eq!(position.volume_raw_centi_units, wire_volume);
        assert_eq!(position.volume, wire_volume as f64 / 100.0);
    }
}

#[test]
fn reconcile_response_rejects_non_positive_position_wire_volume() {
    for wire_volume in [0_i64, -1, i64::MIN] {
        let response = serde_json::json!({
            "payloadType": 2125,
            "payload": {
                "ctidTraderAccountId": 712345,
                "position": [{
                    "positionId": 9001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": wire_volume,
                        "tradeSide": 1
                    }
                }]
            }
        });
        let error = parse_reconcile_response(&response.to_string())
            .expect_err("an open position must have positive remaining wire volume");
        assert!(error.to_string().contains("position 9001"));
        assert!(error.to_string().contains("non-positive centi-unit volume"));
    }
}

#[test]
fn reconcile_response_scales_position_money_digits_four_fields() {
    let response = serde_json::json!({
        "clientMsgId": "reconcile-money-4",
        "payloadType": 2125,
        "payload": {
            "ctidTraderAccountId": 712345,
            "position": [
                {
                    "positionId": 9001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": 2500,
                        "tradeSide": 1,
                        "openTimestamp": 1710000000000i64
                    },
                    "price": 1.10123,
                    "swap": -1234,
                    "commission": -5678,
                    "mirroringCommission": -90,
                    "usedMargin": 123456,
                    "moneyDigits": 4
                }
            ],
            "order": []
        }
    });

    let reconcile = parse_reconcile_response(&response.to_string()).expect("reconcile");
    let position = &reconcile.positions[0];

    assert_eq!(position.swap, Some(-0.1234));
    assert_eq!(position.commission, Some(-0.5678));
    assert_eq!(position.mirroring_commission, Some(-0.009));
    assert_eq!(position.used_margin, Some(12.3456));
}

#[test]
fn reconcile_response_rejects_monetary_fields_without_broker_scale() {
    let response = serde_json::json!({
        "payloadType": 2125,
        "payload": {
            "ctidTraderAccountId": 712345,
            "position": [{
                "positionId": 9001,
                "tradeData": {"symbolId": 14, "volume": 2500, "tradeSide": 1},
                "swap": -1234
            }],
            "order": []
        }
    });

    let error = parse_reconcile_response(&response.to_string())
        .expect_err("a broker monetary integer without moneyDigits must not be guessed");
    assert!(error.to_string().contains("position.money_digits"));
}

#[test]
fn deal_list_response_parses_recent_deals() {
    let response = serde_json::json!({
        "clientMsgId": "deals-1",
        "payloadType": 2134,
        "payload": {
            "ctidTraderAccountId": 712345,
            "deal": [
                {
                    "dealId": 3001,
                    "orderId": 8001,
                    "positionId": 9001,
                    "volume": 1500,
                    "filledVolume": 1500,
                    "symbolId": 14,
                    "createTimestamp": 1710000200000i64,
                    "executionTimestamp": 1710000201000i64,
                    "executionPrice": 1.0990,
                    "tradeSide": 1,
                    "dealStatus": 2,
                    "commission": -40,
                    "moneyDigits": 2,
                    "closePositionDetail": {
                        "entryPrice": 1.0980,
                        "grossProfit": 1250,
                        "swap": 0,
                        "commission": -40,
                        "balance": 1001250,
                        "moneyDigits": 2
                    }
                }
            ],
            "hasMore": false
        }
    });

    let deals = parse_deal_list_response(&response.to_string()).expect("deal list");

    assert_eq!(deals.len(), 1);
    assert_eq!(deals[0].deal_id, 3001);
    assert_eq!(deals[0].trade_side, "BUY");
    assert_eq!(deals[0].deal_status, "FILLED");
    assert!((deals[0].volume - 15.0).abs() < 1e-9);
    assert_eq!(deals[0].execution_price, Some(1.0990));
    assert_eq!(deals[0].gross_profit, Some(12.5));
    assert_eq!(deals[0].fee, Some(-0.4));
    assert_eq!(
        deals[0].pnl_conversion_fee_state,
        Some(crate::app_services::broker_deal_economics::BrokerPnlConversionFeeV1::NotApplied)
    );
}

#[test]
fn deal_list_bundle_preserves_broker_pagination_evidence() {
    let response = serde_json::json!({
        "payloadType": 2134,
        "payload": {
            "ctidTraderAccountId": 712345,
            "deal": [],
            "hasMore": true
        }
    });

    let bundle = parse_deal_list_bundle_response(&response.to_string()).expect("deal-list bundle");

    assert_eq!(bundle.account_id, 712345);
    assert!(bundle.deals.is_empty());
    assert!(bundle.has_more, "hasMore must never be discarded");
}

fn closing_conversion_fixture() -> serde_json::Value {
    serde_json::json!({
        "payloadType": 2134,
        "payload": {"ctidTraderAccountId": 712345, "hasMore": false, "deal": [{
            "dealId": 3001, "orderId": 8001, "positionId": 9001,
            "volume": 10000000, "filledVolume": 10000000, "symbolId": 14,
            "executionTimestamp": 1710000201000i64, "executionPrice": 1.21,
            "tradeSide": 2, "dealStatus": 2, "commission": -450, "moneyDigits": 2,
            "label": "example", "comment": "fixture",
            "closePositionDetail": {
                "entryPrice": 1.20, "grossProfit": 82644, "swap": 0,
                "commission": -900, "moneyDigits": 2, "balance": 1081744,
                "balanceVersion": 2, "quoteToDepositConversionRate": 0.8264462809917356,
                "closedVolume": 10000000, "pnlConversionFee": 0
            }
        }]}
    })
}

#[test]
fn closing_deal_retains_broker_balance_conversion_and_commission_scopes() {
    let deals = parse_deal_list_response(&closing_conversion_fixture().to_string()).unwrap();
    let deal = &deals[0];
    assert_eq!(deal.deal_commission_raw_scaled_signed, Some(-450));
    assert_eq!(deal.deal_money_digits, Some(2));
    assert_eq!(deal.commission_raw_scaled_signed, Some(-900));
    assert_eq!(deal.component_sum_account_currency, Some(817.44));
    assert_eq!(deal.balance_after_raw_scaled, Some(1081744));
    assert_eq!(deal.balance_after, Some(10817.44));
    assert_eq!(deal.balance_version, Some(2));
    assert_eq!(
        deal.quote_to_deposit_conversion_rate,
        Some(0.8264462809917356)
    );
    assert_eq!(deal.closed_volume_raw_centi_units, Some(10000000));
    assert_eq!(deal.label.as_deref(), Some("example"));
    assert_eq!(deal.comment.as_deref(), Some("fixture"));
}

#[test]
fn absent_close_observations_remain_unknown() {
    let mut response = closing_conversion_fixture();
    let detail = response["payload"]["deal"][0]["closePositionDetail"]
        .as_object_mut()
        .unwrap();
    for key in [
        "balance",
        "balanceVersion",
        "quoteToDepositConversionRate",
        "closedVolume",
    ] {
        detail.remove(key);
    }
    let deals = parse_deal_list_response(&response.to_string()).unwrap();
    let deal = &deals[0];
    assert_eq!(deal.balance_after, None);
    assert_eq!(deal.balance_after_raw_scaled, None);
    assert_eq!(deal.balance_version, None);
    assert_eq!(deal.quote_to_deposit_conversion_rate, None);
    assert_eq!(deal.closed_volume_raw_centi_units, None);
}

#[test]
fn closing_deal_rejects_invalid_conversion_and_closed_volume() {
    for rate in [0.0, -1.0] {
        let mut response = closing_conversion_fixture();
        response["payload"]["deal"][0]["closePositionDetail"]["quoteToDepositConversionRate"] =
            serde_json::json!(rate);
        assert!(parse_deal_list_response(&response.to_string()).is_err());
    }
    for volume in [0, -1, 10000001] {
        let mut response = closing_conversion_fixture();
        response["payload"]["deal"][0]["closePositionDetail"]["closedVolume"] =
            serde_json::json!(volume);
        assert!(parse_deal_list_response(&response.to_string()).is_err());
    }
    let mut reversal = closing_conversion_fixture();
    reversal["payload"]["deal"][0]["closePositionDetail"]["closedVolume"] =
        serde_json::json!(5000000);
    let deals = parse_deal_list_response(&reversal.to_string()).unwrap();
    assert_eq!(deals[0].closed_volume_raw_centi_units, Some(5000000));
    assert_eq!(deals[0].filled_volume_raw_centi_units, 10000000);
}

#[test]
fn deal_list_response_scales_close_detail_money_digits_four_fields() {
    let response = serde_json::json!({
        "clientMsgId": "deals-money-4",
        "payloadType": 2134,
        "payload": {
            "ctidTraderAccountId": 712345,
            "deal": [
                {
                    "dealId": 3001,
                    "orderId": 8001,
                    "positionId": 9001,
                    "volume": 1500,
                    "filledVolume": 1500,
                    "symbolId": 14,
                    "executionTimestamp": 1710000201000i64,
                    "executionPrice": 1.0990,
                    "tradeSide": 1,
                    "dealStatus": 2,
                    "closePositionDetail": {
                        "entryPrice": 1.0980,
                        "grossProfit": 1250,
                        "swap": -15,
                        "commission": -40,
                        "pnlConversionFee": -10,
                        "moneyDigits": 4
                    }
                }
            ],
            "hasMore": false
        }
    });

    let deals = parse_deal_list_response(&response.to_string()).expect("deal list");

    assert_eq!(deals[0].gross_profit, Some(0.125));
    assert_eq!(deals[0].fee, Some(-0.004));
    assert_eq!(deals[0].swap, Some(-0.0015));
    assert_eq!(deals[0].pnl_conversion_fee, Some(-0.001));
    assert_eq!(deals[0].account_id, 712345);
    assert_eq!(deals[0].filled_volume_raw_centi_units, 1500);
    assert_eq!(deals[0].money_digits, Some(4));
    assert_eq!(deals[0].gross_profit_raw_scaled, Some(1250));
    assert_eq!(deals[0].commission_raw_scaled_signed, Some(-40));
    assert_eq!(deals[0].swap_raw_scaled_signed, Some(-15));
    assert_eq!(
        deals[0].pnl_conversion_fee_state,
        Some(
            crate::app_services::broker_deal_economics::BrokerPnlConversionFeeV1::Charged {
                raw_scaled_signed: -10,
            }
        )
    );
    assert_eq!(deals[0].component_sum_account_currency, Some(0.1185));
}

#[test]
fn deal_list_rejects_close_financials_without_broker_scale() {
    let response = serde_json::json!({
        "payloadType": 2134,
        "payload": {
            "ctidTraderAccountId": 712345,
            "deal": [{
                "dealId": 3001,
                "orderId": 8001,
                "positionId": 9001,
                "volume": 1500,
                "filledVolume": 1500,
                "symbolId": 14,
                "executionTimestamp": 1710000201000i64,
                "tradeSide": 1,
                "dealStatus": 2,
                "closePositionDetail": {
                    "grossProfit": 1250,
                    "swap": -15,
                    "commission": -40
                }
            }]
        }
    });

    let error = parse_deal_list_response(&response.to_string())
        .expect_err("close financials without moneyDigits must not be scaled locally");
    assert!(error.to_string().contains("deal.close.money_digits"));
}

/// §5.1.3 ship gate — balance scaling with moneyDigits=4 (high-precision
/// account, e.g. precious-metal denomination). The earlier
/// `trader_response_parses_balance_and_account_metadata` covers
/// moneyDigits=2; this test pins the high-precision path so a future
/// regression in the trader-response scaler cannot pass CI.
#[test]
fn trader_response_parses_balance_money_digits_four() {
    let response = serde_json::json!({
        "clientMsgId": "trader-md4",
        "payloadType": 2122,
        "payload": {
            "ctidTraderAccountId": 712345,
            "trader": {
                "balance": 123_456_789i64,
                "moneyDigits": 4,
                "leverageInCents": 5000,
                "traderLogin": 998877,
                "accountType": 1,
                "brokerName": "High-Precision Demo"
            }
        }
    });

    let trader = parse_trader_response(&response.to_string()).expect("trader response");

    assert_eq!(trader.money_digits, 4);
    // 123_456_789 / 10^4 = 12_345.6789
    assert!(
        (trader.balance - 12_345.6789).abs() < 1e-9,
        "balance scaling broken for moneyDigits=4: got {}",
        trader.balance
    );
}

/// §5.1.3 ship gate — swap / commission / mirroring commission / used
/// margin all carry per-position `moneyDigits`. The earlier
/// `reconcile_response_scales_position_money_digits_four_fields` covers
/// moneyDigits=4; this pins the moneyDigits=2 (fiat default) path so the
/// pre-fix `value / 100.0` behaviour can never silently regress.
#[test]
fn reconcile_response_scales_position_money_digits_two_fields() {
    let response = serde_json::json!({
        "clientMsgId": "reconcile-money-2",
        "payloadType": 2125,
        "payload": {
            "ctidTraderAccountId": 712345,
            "position": [
                {
                    "positionId": 9001,
                    "tradeData": {
                        "symbolId": 14,
                        "volume": 2500,
                        "tradeSide": 1,
                        "openTimestamp": 1710000000000i64
                    },
                    "price": 1.10123,
                    "swap": -250,                  // -2.50 USD
                    "commission": -700,            // -7.00 USD
                    "mirroringCommission": -50,    // -0.50 USD
                    "usedMargin": 36_180,          // 361.80 USD
                    "moneyDigits": 2
                }
            ],
            "order": []
        }
    });

    let reconcile = parse_reconcile_response(&response.to_string()).expect("reconcile");
    let position = &reconcile.positions[0];

    assert_eq!(position.swap, Some(-2.50));
    assert_eq!(position.commission, Some(-7.00));
    assert_eq!(position.mirroring_commission, Some(-0.50));
    assert_eq!(position.used_margin, Some(361.80));
}

/// §5.1.3 ship gate — close-position-detail scaling at moneyDigits=2.
/// `deal_list_response_scales_close_detail_money_digits_four_fields`
/// covers moneyDigits=4; this pins the fiat path so the off-by-100
/// pre-fix bug cannot regress on standard USD/EUR accounts.
#[test]
fn deal_list_close_detail_money_digits_two_fields() {
    let response = serde_json::json!({
        "clientMsgId": "deals-md2",
        "payloadType": 2134,
        "payload": {
            "ctidTraderAccountId": 712345,
            "deal": [
                {
                    "dealId": 3002,
                    "orderId": 8002,
                    "positionId": 9001,
                    "volume": 1500,
                    "filledVolume": 1500,
                    "symbolId": 14,
                    "executionTimestamp": 1710000201000i64,
                    "executionPrice": 1.0990,
                    "tradeSide": 1,
                    "dealStatus": 2,
                    "closePositionDetail": {
                        "entryPrice": 1.0980,
                        "grossProfit": 1250,        // +12.50 USD
                        "swap": -15,                // -0.15
                        "commission": -40,          // -0.40
                        "pnlConversionFee": -10,    // -0.10
                        "moneyDigits": 2
                    }
                }
            ],
            "hasMore": false
        }
    });

    let deals = parse_deal_list_response(&response.to_string()).expect("deal list");
    let d = &deals[0];
    assert_eq!(d.gross_profit, Some(12.50));
    assert_eq!(d.swap, Some(-0.15));
    assert_eq!(d.fee, Some(-0.40));
    assert_eq!(d.pnl_conversion_fee, Some(-0.10));
    // net = gross + swap + fee + pnl_conversion_fee = 12.50 - 0.15 - 0.40 - 0.10 = 11.85
    let component_sum = d
        .component_sum_account_currency
        .expect("signed broker components summed");
    assert!(
        (component_sum - 11.85).abs() < 1e-9,
        "component sum broken: {component_sum}"
    );
}

/// §5.1.3 catch-all — the remaining cTrader monetary entities listed in
/// `ctrader_money.rs` are `ProtoOABonusDepositWithdraw.*` and
/// `ProtoOADepositWithdraw.*` (top-up / withdrawal / bonus events).
/// Their proto envelopes are not yet parsed by `parse_*_response`
/// helpers in this module — when the parsers land in v0.5 they should
/// use `scale_ctrader_money_int` exactly as positions / deals do. To
/// prove the scaling primitive itself is unbiased for these entity
/// classes, drive a representative `amount` field at both moneyDigits=2
/// and moneyDigits=4 directly through the helper. This pins the
/// arithmetic contract until the proto parsers wire it in.
#[test]
fn money_scaling_table_covers_deposit_and_bonus_entities() {
    use crate::app_services::ctrader_money::scale_ctrader_money_int;

    // Each row: (entity label, raw integer, moneyDigits, expected real value)
    let cases: &[(&str, i64, i32, f64)] = &[
        // Top-up of $1,234.56 USD on a fiat account.
        ("DepositWithdraw.amount @ mD=2", 123_456, 2, 1_234.56),
        // The same deposit on a moneyDigits=4 account: $12.3456.
        ("DepositWithdraw.amount @ mD=4", 123_456, 4, 12.3456),
        // Bonus credit of $50.00 on a fiat account.
        ("BonusDepositWithdraw.amount @ mD=2", 5_000, 2, 50.00),
        // Same bonus on a moneyDigits=4 account: $0.50.
        ("BonusDepositWithdraw.amount @ mD=4", 5_000, 4, 0.50),
    ];

    for (label, raw, md, expected) in cases {
        let got = scale_ctrader_money_int(*raw, *md)
            .unwrap_or_else(|err| panic!("{label}: scaling errored: {err}"));
        assert!(
            (got - *expected).abs() < 1e-9,
            "{label}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn account_runtime_loader_authenticates_then_loads_trader_reconcile_and_deals() {
    let transport = StubTransport::with_responses(vec![
        Ok(r#"{"clientMsgId":"app-auth-1","payloadType":2101,"payload":{}}"#.to_string()),
        Ok(r#"{"clientMsgId":"account-auth-1","payloadType":2103,"payload":{"ctidTraderAccountId":712345}}"#.to_string()),
        Ok(r#"{"clientMsgId":"trader-1","payloadType":2122,"payload":{"ctidTraderAccountId":712345,"trader":{"balance":100000,"moneyDigits":2,"leverageInCents":5000,"brokerName":"Demo Broker","depositAssetId":8}}}"#.to_string()),
        Ok(r#"{"clientMsgId":"reconcile-1","payloadType":2125,"payload":{"ctidTraderAccountId":712345,"position":[{"positionId":9001,"tradeData":{"symbolId":14,"volume":2500,"tradeSide":1,"openTimestamp":1710000000000},"positionStatus":1,"price":1.10123}],"order":[]}}"#.to_string()),
        Ok(r#"{"clientMsgId":"deals-1","payloadType":2134,"payload":{"ctidTraderAccountId":712345,"deal":[{"dealId":3001,"orderId":8001,"positionId":9001,"volume":1500,"filledVolume":1500,"symbolId":14,"createTimestamp":1710000200000,"executionTimestamp":1710000201000,"executionPrice":1.099,"tradeSide":1,"dealStatus":2,"commission":-40,"moneyDigits":2,"closePositionDetail":{"entryPrice":1.098,"grossProfit":1250,"swap":0,"commission":-40,"balance":1001250,"moneyDigits":2}}],"hasMore":false}}"#.to_string()),
        Ok(r#"{"clientMsgId":"unrealized-pnl-1","payloadType":2188,"payload":{"ctidTraderAccountId":712345,"moneyDigits":2,"positionUnrealizedPnL":[{"positionId":9001,"grossUnrealizedPnL":1234,"netUnrealizedPnL":1134}]}}"#.to_string()),
        Ok(r#"{"clientMsgId":"asset-list-1","payloadType":2113,"payload":{"ctidTraderAccountId":712345,"asset":[{"assetId":8,"name":"USD","displayName":"US Dollar","digits":2}]}}"#.to_string()),
    ]);

    let runtime = load_account_runtime_with_transport(
        &transport,
        &CTraderAccountRuntimeRequest {
            client_id: "client".to_string(),
            client_secret: "secret".to_string(),
            access_token: "access".to_string(),
            environment: CTraderEnvironment::Demo,
            account_id: "712345".to_string(),
            return_protection_orders: true,
        },
    )
    .expect("account runtime");

    assert_eq!(runtime.trader.account_id, 712345);
    assert_eq!(runtime.reconcile.positions.len(), 1);
    assert_eq!(runtime.recent_deals.len(), 1);
    assert_eq!(runtime.environment, CTraderEnvironment::Demo);
    assert_eq!(runtime.unrealized_pnl, 11.34);
    assert_eq!(runtime.deposit_asset_name, "USD");
    assert_eq!(
        runtime
            .unrealized_pnl_by_position
            .get(&9001)
            .map(|row| row.net_unrealized_pnl),
        Some(11.34)
    );
    assert_eq!(transport.sent_len(), 7);
}

#[test]
fn deposit_asset_name_must_be_present_nonempty_and_unique_in_the_broker_registry() {
    let trader = parse_trader_response(
        r#"{"payloadType":2122,"payload":{"ctidTraderAccountId":712345,"trader":{"balance":100000,"moneyDigits":2,"depositAssetId":8}}}"#,
    )
    .expect("valid trader response");
    let usd = crate::app_services::ctrader_data::CTraderAssetInfo {
        asset_id: 8,
        name: "USD".to_string(),
        display_name: Some("US Dollar".to_string()),
        digits: Some(2),
    };

    assert_eq!(
        resolve_deposit_asset_name(&trader, std::slice::from_ref(&usd))
            .expect("one exact broker asset"),
        "USD"
    );
    assert!(resolve_deposit_asset_name(&trader, &[]).is_err());

    let mut blank = usd.clone();
    blank.name = "  ".to_string();
    assert!(resolve_deposit_asset_name(&trader, &[blank]).is_err());
    assert!(resolve_deposit_asset_name(&trader, &[usd.clone(), usd]).is_err());
}

#[test]
fn unrealized_pnl_rows_must_match_open_broker_positions_exactly_once() {
    let reconcile = CTraderReconcileSnapshot {
        account_id: 712345,
        positions: vec![CTraderPositionSnapshot {
            position_id: 9001,
            symbol_id: 14,
            trade_side: "BUY".to_string(),
            volume_raw_centi_units: 100_000,
            volume: 1_000.0,
            open_timestamp_ms: None,
            price: Some(1.1),
            stop_loss: None,
            take_profit: None,
            swap: None,
            commission: None,
            mirroring_commission: None,
            used_margin: None,
            label: None,
            comment: None,
            client_order_id: None,
        }],
        pending_orders: Vec::new(),
    };
    let missing = crate::app_services::ctrader_messages::CTraderUnrealizedPnLSnapshot {
        account_id: 712345,
        money_digits: 2,
        positions: Vec::new(),
    };
    assert!(reconcile_broker_unrealized_pnl(&reconcile, &missing).is_err());

    let duplicated = crate::app_services::ctrader_messages::CTraderUnrealizedPnLSnapshot {
        account_id: 712345,
        money_digits: 2,
        positions: vec![
            crate::app_services::ctrader_messages::CTraderPositionUnrealizedPnL {
                position_id: 9001,
                gross_unrealized_pnl: 12.34,
                net_unrealized_pnl: 11.34,
            },
            crate::app_services::ctrader_messages::CTraderPositionUnrealizedPnL {
                position_id: 9001,
                gross_unrealized_pnl: 12.34,
                net_unrealized_pnl: 11.34,
            },
        ],
    };
    assert!(reconcile_broker_unrealized_pnl(&reconcile, &duplicated).is_err());
}

struct StubTransport {
    sent: std::sync::Mutex<Vec<CTraderOpenApiJsonMessage>>,
    responses: std::sync::Mutex<Vec<anyhow::Result<String>>>,
}

impl StubTransport {
    fn with_responses(responses: Vec<anyhow::Result<String>>) -> Self {
        Self {
            sent: std::sync::Mutex::new(Vec::new()),
            responses: std::sync::Mutex::new(responses),
        }
    }

    fn sent_len(&self) -> usize {
        self.sent.lock().expect("sent lock").len()
    }
}

impl crate::app_services::ctrader_messages::CTraderOpenApiTransport for StubTransport {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> anyhow::Result<Vec<String>> {
        self.sent
            .lock()
            .expect("sent lock")
            .extend(messages.iter().cloned());
        let mut responses = self.responses.lock().expect("responses lock");
        let mut output = Vec::with_capacity(messages.len());
        for _ in messages {
            output.push(responses.remove(0)?);
        }
        Ok(output)
    }
}

// ─── 2026-06-10 API-completeness response parsers ──────────────────────────
// TODO(real-data): replace these hand-built fixtures with captured broker
// responses once a demo account is available (see file header).

#[test]
fn parses_order_list_response() {
    let json = r#"{"payloadType":2176,"payload":{"ctidTraderAccountId":1,"hasMore":true,
      "order":[{"orderId":5,"orderType":2,"orderStatus":2,"executedVolume":1000,
        "executionPrice":1.2345,"utcLastUpdateTimestamp":1700000000000,"timeInForce":2,
        "tradeData":{"symbolId":1,"volume":1000,"tradeSide":1,"openTimestamp":1699999999000,
          "closeTimestamp":1700000001000}}]}}"#;
    let b = parse_order_list_response(json).unwrap();
    assert_eq!(b.account_id, 1);
    assert!(b.has_more);
    let o = &b.orders[0];
    assert_eq!(o.side, "BUY");
    assert_eq!(o.order_status, "FILLED");
    assert_eq!(o.order_type, "LIMIT");
    assert_eq!(o.time_in_force.as_deref(), Some("GOOD_TILL_CANCEL"));
    let wire = serde_json::to_value(o).expect("history API row");
    assert_eq!(
        wire["volumeLots"],
        serde_json::Value::Null,
        "a raw order has no broker lotSize; base units must not be labelled as lots"
    );
    assert_eq!(wire["executedVolumeLots"], serde_json::Value::Null);
    assert_eq!(wire["volumeUnits"], 10.0); // 1000 centi-units / 100
    assert_eq!(wire["executedVolumeUnits"], 10.0);
    assert_eq!(o.close_timestamp_ms, Some(1700000001000));
    assert!(!o.is_stop_out);
}

#[test]
fn order_list_rejects_wrong_payload_type() {
    // 2138 is GET_TRENDBARS_RES — the discriminator-bug guard.
    let json = r#"{"payloadType":2138,"payload":{"ctidTraderAccountId":1,"order":[]}}"#;
    assert!(parse_order_list_response(json).is_err());
}

#[test]
fn parses_cash_flow_history_with_money_digits() {
    let json = r#"{"payloadType":2144,"payload":{"ctidTraderAccountId":1,"depositWithdraw":[
      {"operationType":0,"balanceHistoryId":7,"balance":10053099944,"delta":10000000000,
       "changeBalanceTimestamp":1700000000000,"equity":10053099944,"moneyDigits":8}]}}"#;
    let b = parse_cash_flow_history_response(json).unwrap();
    let e = &b.entries[0];
    assert_eq!(e.operation_type, "BALANCE_DEPOSIT");
    assert_eq!(e.operation_type_code, 0);
    assert!((e.balance - 100.53099944).abs() < 1e-6); // / 10^8
    assert!((e.delta - 100.0).abs() < 1e-6);
    assert_eq!(e.change_balance_timestamp_ms, 1700000000000);
}

#[test]
fn cash_flow_unknown_operation_type_is_labelled_not_guessed() {
    // We only hardcode 0/1/39; anything else stays unmapped (no invented names).
    let json = r#"{"payloadType":2144,"payload":{"ctidTraderAccountId":1,"depositWithdraw":[
      {"operationType":17,"balanceHistoryId":1,"balance":0,"delta":-500,
       "changeBalanceTimestamp":1,"moneyDigits":2}]}}"#;
    let e = &parse_cash_flow_history_response(json).unwrap().entries[0];
    assert_eq!(e.operation_type, "CHANGE_BALANCE_TYPE(17)");
    assert_eq!(e.operation_type_code, 17);
    assert!((e.delta - (-5.0)).abs() < 1e-6); // signed delta is authoritative
}

#[test]
fn parses_expected_margin() {
    let json = r#"{"payloadType":2140,"payload":{"ctidTraderAccountId":1,"moneyDigits":2,
      "margin":[{"volume":10000000,"buyMargin":33333,"sellMargin":33333}]}}"#;
    let b = parse_expected_margin_response(json).unwrap();
    let e = &b.entries[0];
    let wire = serde_json::to_value(e).expect("margin API row");
    assert_eq!(
        wire["volumeLots"],
        serde_json::Value::Null,
        "moneyDigits cannot provide the missing broker lotSize"
    );
    assert_eq!(wire["volumeUnits"], 100000.0); // 10_000_000 centi-units / 100
    assert!((e.buy_margin - 333.33).abs() < 1e-6); // / 10^2
    assert!((e.sell_margin - 333.33).abs() < 1e-6);
}

#[test]
fn parses_ctid_profile_only_user_id() {
    let json = r#"{"payloadType":2152,"payload":{"profile":{"userId":123456}}}"#;
    assert_eq!(parse_ctid_profile_response(json).unwrap().user_id, 123456);
}

#[test]
fn parses_version() {
    let json = r#"{"payloadType":2105,"payload":{"version":"4.2.1"}}"#;
    assert_eq!(parse_version_response(json).unwrap().version, "4.2.1");
}

fn empty_runtime_responses(account: i64) -> Vec<serde_json::Value> {
    let rows = [
        ("app-auth-1", 2101, serde_json::json!({})),
        (
            "account-auth-1",
            2103,
            serde_json::json!({"ctidTraderAccountId": account}),
        ),
        (
            "trader-1",
            2122,
            serde_json::json!({"ctidTraderAccountId": account,
            "trader": {"balance": 100000, "moneyDigits": 2, "depositAssetId": 8}}),
        ),
        (
            "reconcile-1",
            2125,
            serde_json::json!({"ctidTraderAccountId": account,
            "position": [], "order": []}),
        ),
        (
            "deals-1",
            2134,
            serde_json::json!({"ctidTraderAccountId": account,
            "deal": [], "hasMore": false}),
        ),
        (
            "unrealized-pnl-1",
            2188,
            serde_json::json!({"ctidTraderAccountId": account,
            "moneyDigits": 2, "positionUnrealizedPnL": []}),
        ),
        (
            "asset-list-1",
            2113,
            serde_json::json!({"ctidTraderAccountId": account,
            "asset": [{"assetId": 8, "name": "USD", "digits": 2}]}),
        ),
    ];
    rows.into_iter()
        .map(|(id, kind, payload)| {
            serde_json::json!({
                "clientMsgId": id, "payloadType": kind, "payload": payload
            })
        })
        .collect()
}

fn empty_runtime_request(environment: CTraderEnvironment) -> CTraderAccountRuntimeRequest {
    CTraderAccountRuntimeRequest {
        client_id: "fixture-client".to_owned(),
        client_secret: "fixture-secret".to_owned(),
        access_token: "fixture-access".to_owned(),
        environment,
        account_id: "712345".to_owned(),
        return_protection_orders: true,
    }
}

fn runtime_transport(rows: Vec<serde_json::Value>) -> StubTransport {
    StubTransport::with_responses(rows.into_iter().map(|row| Ok(row.to_string())).collect())
}

#[test]
fn runtime_accepts_matching_empty_account_snapshots_in_both_environments() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        let transport = runtime_transport(empty_runtime_responses(712345));
        let snapshot =
            load_account_runtime_with_transport(&transport, &empty_runtime_request(environment))
                .unwrap();
        assert_eq!(snapshot.environment, environment);
        assert_eq!(snapshot.trader.account_id, 712345);
        assert_eq!(snapshot.trader.balance, 1000.0);
        assert_eq!(snapshot.deposit_asset_name, "USD");
        assert_eq!(snapshot.unrealized_pnl, 0.0);
        assert!(snapshot.reconcile.positions.is_empty());
        assert!(snapshot.recent_deals.is_empty());
        assert_eq!(transport.sent_len(), 7);
    }
}

#[test]
fn runtime_rejects_each_foreign_account_envelope_even_when_rows_are_empty() {
    for environment in [CTraderEnvironment::Demo, CTraderEnvironment::Live] {
        for index in 1..7 {
            let mut rows = empty_runtime_responses(712345);
            rows[index]["payload"]["ctidTraderAccountId"] = serde_json::json!(99);
            let transport = runtime_transport(rows);
            let error = load_account_runtime_with_transport(
                &transport,
                &empty_runtime_request(environment),
            )
            .unwrap_err();
            assert!(
                format!("{error:#}").contains("response identity differs"),
                "response {index}: {error:#}"
            );
            assert_eq!(transport.sent_len(), 7);
        }
    }
}

#[test]
fn runtime_rejects_missing_or_malformed_account_identity_before_row_parsing() {
    for index in 1..7 {
        for identity in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!("712345")),
            Some(serde_json::json!(712345.5)),
            Some(serde_json::json!(true)),
            Some(serde_json::json!(u64::MAX)),
            Some(serde_json::json!(0)),
            Some(serde_json::json!(-1)),
        ] {
            let mut rows = empty_runtime_responses(712345);
            let payload = rows[index]["payload"].as_object_mut().unwrap();
            match identity {
                Some(value) => {
                    payload.insert("ctidTraderAccountId".to_owned(), value);
                }
                None => {
                    payload.remove("ctidTraderAccountId");
                }
            }
            let error = load_account_runtime_with_transport(
                &runtime_transport(rows),
                &empty_runtime_request(CTraderEnvironment::Demo),
            )
            .unwrap_err();
            assert!(
                format!("{error:#}").contains("response identity differs"),
                "response {index}: {error:#}"
            );
        }
    }
    for (index, field) in [(4, "deal"), (6, "asset")] {
        let mut rows = empty_runtime_responses(712345);
        rows[index]["payload"]["ctidTraderAccountId"] = serde_json::json!(99);
        rows[index]["payload"][field] = serde_json::json!("malformed rows");
        let error = load_account_runtime_with_transport(
            &runtime_transport(rows),
            &empty_runtime_request(CTraderEnvironment::Demo),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("response identity differs"));
    }
}

#[test]
fn runtime_rejects_invalid_requested_account_before_transport() {
    for account in ["", "not-an-id", "9223372036854775808", "0", "-1"] {
        let mut request = empty_runtime_request(CTraderEnvironment::Demo);
        request.account_id = account.to_owned();
        let transport = StubTransport::with_responses(Vec::new());
        let error = load_account_runtime_with_transport(&transport, &request).unwrap_err();
        assert!(format!("{error:#}").contains("cTrader account id"));
        assert_eq!(transport.sent_len(), 0);
    }
}

#[test]
fn runtime_preserves_broker_error_and_payload_type_diagnostics_before_identity() {
    for index in [4, 6] {
        for (code, attempts) in [("ACCESS_DENIED", 3_usize), ("BLOCKED_PAYLOAD_TYPE", 1)] {
            let mut rows = empty_runtime_responses(712345);
            rows[index]["payloadType"] = serde_json::json!(CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE);
            rows[index]["payload"] = serde_json::json!({"errorCode": code,
                "description": "fixture broker refusal"});
            if code == "BLOCKED_PAYLOAD_TYPE" {
                rows[index]["payload"]["retryAfter"] = serde_json::json!(120);
            }
            // The existing resilient transport repeats the complete sequence
            // three times except for the broker's explicit rate-limit refusal.
            // Supply each attempted response; never mask an exhausted fixture
            // by changing the production retry or account-identity policy.
            let responses = (0..attempts).flat_map(|_| rows.iter().cloned()).collect();
            let transport = runtime_transport(responses);
            let error = load_account_runtime_with_transport(
                &transport,
                &empty_runtime_request(CTraderEnvironment::Demo),
            )
            .unwrap_err();
            let detail = format!("{error:#}");
            assert!(detail.contains(code), "{detail}");
            assert!(detail.contains("fixture broker refusal"), "{detail}");
            assert_eq!(transport.sent_len(), 7 * attempts);
            if code == "BLOCKED_PAYLOAD_TYPE" {
                assert!(detail.contains("retryAfter=120s"), "{detail}");
            }
        }
        let mut rows = empty_runtime_responses(712345);
        rows[index]["payloadType"] = serde_json::json!(CTRADER_OA_VERSION_RESPONSE_PAYLOAD_TYPE);
        let transport = runtime_transport(rows);
        let error = load_account_runtime_with_transport(
            &transport,
            &empty_runtime_request(CTraderEnvironment::Demo),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("unexpected cTrader payload type"));
        assert_eq!(transport.sent_len(), 7);
    }
}
