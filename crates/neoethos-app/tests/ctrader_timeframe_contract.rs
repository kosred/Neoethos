use neoethos_app::app_services::ctrader_data::{
    CTraderHistoricalBarsFetchResult, CTraderSymbolInfo, parse_trendbars_response,
};
use neoethos_app::app_services::ctrader_messages::{
    CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE, trendbar_period_value,
};
use neoethos_core::CanonicalTimeframe;

fn symbol() -> CTraderSymbolInfo {
    CTraderSymbolInfo {
        symbol_id: 14,
        symbol_name: "EURUSD".to_owned(),
        display_name: "EURUSD".to_owned(),
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
    }
}

#[test]
fn every_official_period_round_trips_through_request_and_response_adapters() {
    for timeframe in CanonicalTimeframe::ALL {
        let code = timeframe.ctrader_protocol_code();
        assert_eq!(trendbar_period_value(timeframe.as_str()).unwrap(), code);

        let response = serde_json::json!({
            "clientMsgId": "trendbars-contract",
            "payloadType": CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
            "payload": {
                "period": code,
                "symbolId": 14,
                "hasMore": false,
                "trendbar": []
            }
        });
        let parsed = parse_trendbars_response(&response.to_string(), &symbol())
            .expect("official trendbar response period");
        assert_eq!(parsed.timeframe, timeframe);

        let page = CTraderHistoricalBarsFetchResult {
            symbol: symbol(),
            symbol_id: parsed.symbol_id,
            timeframe: parsed.timeframe,
            bars: parsed.bars,
            has_more: parsed.has_more,
        };
        page.validate_identity(14, timeframe)
            .expect("exact direct response identity");
    }
}

#[test]
fn non_ctrader_periods_fail_both_adapters() {
    assert!(trendbar_period_value("H2").is_err());
    let response = serde_json::json!({
        "clientMsgId": "trendbars-contract",
        "payloadType": CTRADER_OA_GET_TRENDBARS_RESPONSE_PAYLOAD_TYPE,
        "payload": {
            "period": 15,
            "symbolId": 14,
            "hasMore": false,
            "trendbar": []
        }
    });
    assert!(parse_trendbars_response(&response.to_string(), &symbol()).is_err());
}
