use serde::de::DeserializeOwned;

use neoethos_app::server::data_control::{FetchBody, ImportBody};
use neoethos_app::server::orders::{
    AmendPositionProtectionBody, CancelOrderBody, ClosePositionBody, NewOrderBody,
    NewPendingOrderBody,
};
use neoethos_app::server::strategy_lab::PromoteBody;

fn assert_unknown_field<T: DeserializeOwned>(fixture: &str, field: &str) {
    let error = match serde_json::from_str::<T>(fixture) {
        Ok(_) => panic!("fixture unexpectedly accepted unknown field `{field}`: {fixture}"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains(&format!("unknown field `{field}`")),
        "expected unknown-field error for `{field}`, got: {message}"
    );
}

#[test]
fn server_contract_tests_frontend_fixtures_use_camel_case() {
    let protection: AmendPositionProtectionBody = serde_json::from_str(
        r#"{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true}"#,
    )
    .expect("deserialize amendProtectionBody fixture");
    assert_eq!(protection.position_id, 42);
    assert_eq!(protection.stop_loss_price, Some(1.07125));
    assert_eq!(protection.take_profit_price, Some(1.0845));
    assert_eq!(protection.trailing_stop_loss, Some(true));

    let import: ImportBody = serde_json::from_str(
        r#"{"sourcePath":"C:/market-data/EURUSD.csv","symbol":"EURUSD","timeframe":"M5"}"#,
    )
    .expect("deserialize dataImportBody fixture");
    assert_eq!(import.source_path, "C:/market-data/EURUSD.csv");
    assert_eq!(import.symbol, "EURUSD");
    assert_eq!(import.timeframe, "M5");

    let fetch: FetchBody = serde_json::from_str(
        r#"{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000}"#,
    )
    .expect("deserialize dataFetchBody fixture");
    assert_eq!(fetch.symbol, "EURUSD");
    assert_eq!(fetch.timeframe, "M5");
    assert_eq!(fetch.from_ms, 1_700_000_000_000);
    assert_eq!(fetch.to_ms, None);

    let promotion: PromoteBody =
        serde_json::from_str(r#"{"symbol":"EURUSD","baseTf":"M5"}"#)
            .expect("deserialize promoteStrategyBody fixture");
    assert_eq!(promotion.symbol.as_deref(), Some("EURUSD"));
    assert_eq!(promotion.base_tf.as_deref(), Some("M5"));
}

#[test]
fn server_contract_tests_snake_case_aliases_are_rejected() {
    assert_unknown_field::<NewOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","volumeLots":0.1,"volume_lots":0.2,"stopLossPips":20.0,"takeProfitPips":40.0,"comment":null,"risky":false}"#,
        "volume_lots",
    );
    assert_unknown_field::<NewOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","volumeLots":0.1,"stopLossPips":20.0,"stop_loss_pips":25.0,"takeProfitPips":40.0,"comment":null,"risky":false}"#,
        "stop_loss_pips",
    );
    assert_unknown_field::<NewOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","volumeLots":0.1,"stopLossPips":20.0,"takeProfitPips":40.0,"take_profit_pips":45.0,"comment":null,"risky":false}"#,
        "take_profit_pips",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","order_type":"stop","volumeLots":0.1,"triggerPrice":1.07125,"stopLossPips":20.0,"takeProfitPips":40.0,"expiryUnixMs":null,"comment":null}"#,
        "order_type",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"volume_lots":0.2,"triggerPrice":1.07125,"stopLossPips":20.0,"takeProfitPips":40.0,"expiryUnixMs":null,"comment":null}"#,
        "volume_lots",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"triggerPrice":1.07125,"trigger_price":1.072,"stopLossPips":20.0,"takeProfitPips":40.0,"expiryUnixMs":null,"comment":null}"#,
        "trigger_price",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"triggerPrice":1.07125,"stopLossPips":20.0,"stop_loss_pips":25.0,"takeProfitPips":40.0,"expiryUnixMs":null,"comment":null}"#,
        "stop_loss_pips",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"triggerPrice":1.07125,"stopLossPips":20.0,"takeProfitPips":40.0,"take_profit_pips":45.0,"expiryUnixMs":null,"comment":null}"#,
        "take_profit_pips",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"triggerPrice":1.07125,"stopLossPips":20.0,"takeProfitPips":40.0,"expiryUnixMs":null,"expiry_unix_ms":1700003600000,"comment":null}"#,
        "expiry_unix_ms",
    );
    assert_unknown_field::<ClosePositionBody>(
        r#"{"positionId":42,"position_id":43,"volume":1000}"#,
        "position_id",
    );
    assert_unknown_field::<CancelOrderBody>(
        r#"{"orderId":84,"order_id":85}"#,
        "order_id",
    );
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"positionId":42,"position_id":43,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true}"#,
        "position_id",
    );
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"positionId":42,"stopLossPrice":1.07125,"stop_loss_price":1.07,"takeProfitPrice":1.0845,"trailingStopLoss":true}"#,
        "stop_loss_price",
    );
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"take_profit_price":1.09,"trailingStopLoss":true}"#,
        "take_profit_price",
    );
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true,"trailing_stop_loss":false}"#,
        "trailing_stop_loss",
    );
    assert_unknown_field::<FetchBody>(
        r#"{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000,"from_ms":1690000000000,"toMs":1700003600000}"#,
        "from_ms",
    );
    assert_unknown_field::<FetchBody>(
        r#"{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000,"toMs":1700003600000,"to_ms":1700007200000}"#,
        "to_ms",
    );
    assert_unknown_field::<ImportBody>(
        r#"{"sourcePath":"C:/market-data/EURUSD.csv","source_path":"C:/other.csv","symbol":"EURUSD","timeframe":"M5"}"#,
        "source_path",
    );
    assert_unknown_field::<PromoteBody>(
        r#"{"symbol":"EURUSD","baseTf":"M5","base_tf":"H1"}"#,
        "base_tf",
    );
}

#[test]
fn server_contract_tests_unknown_fields_are_rejected_by_every_body() {
    assert_unknown_field::<NewOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","volumeLots":0.1,"stopLossPips":20.0,"takeProfitPips":40.0,"comment":null,"risky":false,"riskUsd":100.0}"#,
        "riskUsd",
    );
    assert_unknown_field::<NewPendingOrderBody>(
        r#"{"symbol":"EURUSD","side":"buy","orderType":"limit","volumeLots":0.1,"triggerPrice":1.07125,"stopLossPips":20.0,"takeProfitPips":40.0,"expiryUnixMs":null,"comment":null,"maxSlippagePips":2.0}"#,
        "maxSlippagePips",
    );
    assert_unknown_field::<ClosePositionBody>(
        r#"{"positionId":42,"volume":1000,"closePrice":1.07125}"#,
        "closePrice",
    );
    assert_unknown_field::<CancelOrderBody>(
        r#"{"orderId":84,"refundUsd":100.0}"#,
        "refundUsd",
    );
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true,"moneyAtRisk":100.0}"#,
        "moneyAtRisk",
    );
    assert_unknown_field::<FetchBody>(
        r#"{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000,"toMs":1700003600000,"budgetUsd":100.0}"#,
        "budgetUsd",
    );
    assert_unknown_field::<ImportBody>(
        r#"{"sourcePath":"C:/market-data/EURUSD.csv","symbol":"EURUSD","timeframe":"M5","priceScale":100000}"#,
        "priceScale",
    );
    assert_unknown_field::<PromoteBody>(
        r#"{"symbol":"EURUSD","baseTf":"M5","capitalUsd":10000.0}"#,
        "capitalUsd",
    );
}
