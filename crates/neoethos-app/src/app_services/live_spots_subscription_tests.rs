use super::*;
use crate::app_services::ctrader_messages::CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE;
use serde_json::{Value, json};

struct Wire {
    trader: Value,
    catalog: Value,
    requests: Vec<CTraderOpenApiJsonMessage>,
}
impl Wire {
    fn new(deposit: i64) -> Self {
        Self {
            trader: json!({"ctidTraderAccountId":42,
                "trader":{"balance":1000000,"moneyDigits":2,"depositAssetId":deposit}}),
            catalog: json!({"ctidTraderAccountId":42,"symbol":[
                {"symbolId":7,"symbolName":"PRIMARY_A","enabled":true,"baseAssetId":4,"quoteAssetId":8},
                {"symbolId":8,"symbolName":"PRIMARY_B","enabled":true,"baseAssetId":12,"quoteAssetId":8},
                {"symbolId":19,"symbolName":"BROKER_CONVERSION","enabled":true,"baseAssetId":8,"quoteAssetId":9},
                {"symbolId":999,"symbolName":"UNRELATED","enabled":true,"baseAssetId":20,"quoteAssetId":21}
            ]}),
            requests: Vec::new(),
        }
    }
    fn exchange(
        &mut self,
        request: &CTraderOpenApiJsonMessage,
    ) -> Result<CTraderOpenApiJsonMessage> {
        self.requests.push(request.clone());
        assert_eq!(request.payload["ctidTraderAccountId"], 42);
        let (payload_type, payload) = match request.payload_type {
            2121 => (2122, self.trader.clone()),
            2114 => (2115, self.catalog.clone()),
            _ => panic!("only the two metadata requests belong in this pre-subscription exchange"),
        };
        Ok(CTraderOpenApiJsonMessage {
            client_msg_id: request.client_msg_id.clone(),
            payload_type,
            payload,
        })
    }
    fn prepare(
        &mut self,
        configured: &[StreamedSymbol],
    ) -> Result<(Vec<StreamedSymbol>, CTraderOpenApiJsonMessage)> {
        prepare_spot_subscription(42, configured, |request| self.exchange(request))
    }
}
fn primary(id: i64, name: &str) -> StreamedSymbol {
    StreamedSymbol {
        symbol_id: id,
        symbol_name: name.into(),
        digits: 3,
    }
}
fn ids(symbols: &[StreamedSymbol]) -> Vec<i64> {
    symbols.iter().map(|symbol| symbol.symbol_id).collect()
}

#[test]
fn same_socket_metadata_builds_one_stable_timestamped_union_and_parser_set() {
    let mut wire = Wire::new(9);
    let configured = vec![
        primary(8, "PRIMARY_B"),
        primary(7, "PRIMARY_A"),
        primary(8, "PRIMARY_B"),
    ];
    let (symbols, request) = wire.prepare(&configured).unwrap();
    assert_eq!(
        wire.requests
            .iter()
            .map(|r| r.payload_type)
            .collect::<Vec<_>>(),
        vec![2121, 2114]
    );
    assert_eq!(
        ids(&symbols),
        vec![8, 7, 19],
        "one shared conversion leg; no whole-catalog subscription"
    );
    assert_eq!(configured.len(), 3, "operator inputs are not mutated");
    assert_eq!(
        (symbols[0].digits, symbols[1].digits, symbols[2].digits),
        (3, 3, 5)
    );
    assert_eq!(
        request.payload_type,
        CTRADER_OA_SUBSCRIBE_SPOTS_REQUEST_PAYLOAD_TYPE
    );
    assert_eq!(request.payload["symbolId"], json!([8, 7, 19]));
    assert_eq!(request.payload["ctidTraderAccountId"], 42);
    assert_eq!(request.payload["subscribeToSpotTimestamp"], true);
    let event = json!({"payloadType":CTRADER_OA_SPOT_EVENT_PAYLOAD_TYPE,
        "payload":{"ctidTraderAccountId":42,"symbolId":19,"bid":79000,"ask":80000,"timestamp":1000}});
    assert!(parse_spot_event_loose(&event.to_string(), 42, &symbols).is_some());
    assert!(parse_spot_event_loose(&event.to_string(), 42, &configured).is_none());
}

#[test]
fn direct_inverse_primary_only_and_reconnect_catalog_changes_use_shared_selector() {
    let configured = vec![primary(7, "PRIMARY_A")];
    for deposit in [4, 8] {
        let (symbols, _) = Wire::new(deposit).prepare(&configured).unwrap();
        assert_eq!(
            ids(&symbols),
            vec![7],
            "base/quote account uses the primary quote"
        );
    }
    let mut wire = Wire::new(9);
    assert_eq!(ids(&wire.prepare(&configured).unwrap().0), vec![7, 19]);
    // A reconnect resolves the new broker asset-linked inverse without carrying
    // a dependency accumulated during the previous connection.
    wire.catalog["symbol"][2] = json!({"symbolId":20,"symbolName":"NEW_INVERSE",
        "enabled":true,"baseAssetId":9,"quoteAssetId":8});
    assert_eq!(ids(&wire.prepare(&configured).unwrap().0), vec![7, 20]);
    assert_eq!(
        wire.requests.len(),
        4,
        "both metadata replies are re-read for the reconnect"
    );
}

#[test]
fn unsupported_or_ambiguous_conversion_keeps_primary_display_without_guessing_a_leg() {
    let configured = vec![primary(7, "PRIMARY_A")];
    for mode in ["missing", "ambiguous", "disabled", "missing_asset"] {
        let mut wire = Wire::new(9);
        match mode {
            "missing" => {
                wire.catalog["symbol"].as_array_mut().unwrap().remove(2);
            }
            "ambiguous" => {
                let mut duplicate = wire.catalog["symbol"][2].clone();
                duplicate["symbolId"] = json!(20);
                wire.catalog["symbol"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            "disabled" => wire.catalog["symbol"][2]["enabled"] = json!(false),
            _ => {
                wire.catalog["symbol"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("quoteAssetId");
            }
        }
        let (symbols, request) = wire.prepare(&configured).unwrap();
        assert_eq!(ids(&symbols), vec![7], "{mode}");
        assert_eq!(request.payload["symbolId"], json!([7]), "{mode}");
    }
}

#[test]
fn foreign_or_malformed_metadata_cannot_create_any_subscription_request() {
    let configured = vec![primary(7, "PRIMARY_A")];
    for target in [2121, 2114] {
        for fault in ["account", "type", "message", "payload", "broker_error"] {
            let mut wire = Wire::new(9);
            let result = prepare_spot_subscription(42, &configured, |request| {
                let mut response = wire.exchange(request)?;
                if request.payload_type == target {
                    match fault {
                        "account" => response.payload["ctidTraderAccountId"] = json!(99),
                        "type" => response.payload_type = 2103,
                        "message" => response.client_msg_id = "unrelated".into(),
                        "payload" => response.payload = Value::Null,
                        _ => {
                            response.payload_type = CTRADER_OA_ERROR_RESPONSE_PAYLOAD_TYPE;
                            response.payload =
                                json!({"errorCode":"SYNTHETIC_REFUSAL","description":"fixture"});
                        }
                    }
                }
                Ok(response)
            });
            assert!(result.is_err(), "target={target} fault={fault}");
            assert_eq!(wire.requests.len(), if target == 2121 { 1 } else { 2 });
        }
    }
    for fault in ["missing_deposit", "renamed", "duplicate_primary"] {
        let mut wire = Wire::new(9);
        match fault {
            "missing_deposit" => {
                wire.trader["trader"]
                    .as_object_mut()
                    .unwrap()
                    .remove("depositAssetId");
            }
            "renamed" => wire.catalog["symbol"][0]["symbolName"] = json!("OTHER"),
            _ => {
                let duplicate = wire.catalog["symbol"][0].clone();
                wire.catalog["symbol"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
        }
        assert!(wire.prepare(&configured).is_err(), "{fault}");
    }
}
