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

    let promotion: PromoteBody =
        serde_json::from_str(r#"{"symbol":"EURUSD","baseTf":"M5"}"#)
            .expect("deserialize promoteStrategyBody fixture");
    assert_eq!(promotion.symbol.as_deref(), Some("EURUSD"));
    assert_eq!(promotion.base_tf.as_deref(), Some("M5"));
}

#[test]
fn server_contract_tests_snake_case_aliases_are_rejected() {
    assert_unknown_field::<AmendPositionProtectionBody>(
        r#"{"position_id":42,"stop_loss_price":1.07125,"take_profit_price":1.0845,"trailing_stop_loss":true}"#,
        "position_id",
    );
    assert_unknown_field::<ImportBody>(
        r#"{"source_path":"C:/market-data/EURUSD.csv","symbol":"EURUSD","timeframe":"M5"}"#,
        "source_path",
    );
    assert_unknown_field::<PromoteBody>(
        r#"{"symbol":"EURUSD","base_tf":"M5"}"#,
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
