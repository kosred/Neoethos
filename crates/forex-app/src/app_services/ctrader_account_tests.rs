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
            "balance": 123456789,
            "moneyDigits": 2,
            "leverageInCents": 5000,
            "traderLogin": 998877,
            "accountType": 1,
            "brokerName": "Spotware Demo Broker"
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
    assert!((reconcile.positions[0].volume - 25.0).abs() < 1e-9);
    assert_eq!(reconcile.pending_orders[0].order_id, 8001);
    assert_eq!(reconcile.pending_orders[0].trade_side, "SELL");
    assert_eq!(reconcile.pending_orders[0].order_type, "LIMIT");
    assert!((reconcile.pending_orders[0].limit_price.unwrap_or_default() - 1.099).abs() < 1e-9);
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
    assert_eq!(deals[0].net_profit, Some(0.1185));
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
            "balance": 123_456_789i64,   // raw integer wire value
            "moneyDigits": 4,
            "leverageInCents": 5000,
            "traderLogin": 998877,
            "accountType": 1,
            "brokerName": "High-Precision Demo"
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
    let net = d.net_profit.expect("net profit computed");
    assert!((net - 11.85).abs() < 1e-9, "net_profit broken: {net}");
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
        Ok(r#"{"clientMsgId":"trader-1","payloadType":2122,"payload":{"ctidTraderAccountId":712345,"balance":100000,"moneyDigits":2,"leverageInCents":5000,"brokerName":"Demo Broker"}}"#.to_string()),
        Ok(r#"{"clientMsgId":"reconcile-1","payloadType":2125,"payload":{"ctidTraderAccountId":712345,"position":[{"positionId":9001,"tradeData":{"symbolId":14,"volume":2500,"tradeSide":1,"openTimestamp":1710000000000},"positionStatus":1,"price":1.10123}],"order":[]}}"#.to_string()),
        Ok(r#"{"clientMsgId":"deals-1","payloadType":2134,"payload":{"ctidTraderAccountId":712345,"deal":[{"dealId":3001,"orderId":8001,"positionId":9001,"volume":1500,"filledVolume":1500,"symbolId":14,"createTimestamp":1710000200000,"executionTimestamp":1710000201000,"executionPrice":1.099,"tradeSide":1,"dealStatus":2,"commission":-40,"moneyDigits":2,"closePositionDetail":{"entryPrice":1.098,"grossProfit":1250,"swap":0,"commission":-40,"balance":1001250,"moneyDigits":2}}],"hasMore":false}}"#.to_string()),
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
    assert_eq!(transport.sent_len(), 5);
}

#[test]
fn stub_runtime_backend_records_request_and_surfaces_failure() {
    let backend = StubCTraderAccountRuntimeBackend::failure("runtime probe failed");
    let request = CTraderAccountRuntimeRequest {
        client_id: "client".to_string(),
        client_secret: "secret".to_string(),
        access_token: "access".to_string(),
        environment: CTraderEnvironment::Demo,
        account_id: "712345".to_string(),
        return_protection_orders: true,
    };

    let error = backend
        .load_account_runtime(&request)
        .expect_err("stub backend should fail");

    assert!(error.to_string().contains("runtime probe failed"));
    assert_eq!(backend.last_request(), Some(request));
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
