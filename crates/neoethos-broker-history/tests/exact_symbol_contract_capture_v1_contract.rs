use neoethos_broker_history::symbol_contract_cli::{
    ExactBrokerSymbolContractCaptureCliV1, publish_validated_broker_symbol_contract_response_v1,
};
use neoethos_broker_history::{BrokerEnvironment, ExactBrokerSymbolContractBindingV1};

const SOURCE: &str = include_str!("../src/symbol_contract_cli.rs");
const BINARY: &str = include_str!("../src/bin/neoethos-broker-symbol-contract.rs");

fn args(output_root: &str) -> Vec<String> {
    vec![
        "neoethos-broker-symbol-contract".to_owned(),
        "--environment".to_owned(),
        "demo".to_owned(),
        "--account-id".to_owned(),
        "46774385".to_owned(),
        "--symbol-id".to_owned(),
        "1".to_owned(),
        "--symbol-name".to_owned(),
        "EURUSD".to_owned(),
        "--output-root".to_owned(),
        output_root.to_owned(),
    ]
}

fn light_response(account_id: i64, symbol_id: i64, symbol_name: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "clientMsgId": "symbol-contract-light-symbols",
        "payloadType": 2115,
        "payload": {
            "ctidTraderAccountId": account_id,
            "symbol": [{
                "symbolId": symbol_id,
                "symbolName": symbol_name,
                "baseAssetId": 1,
                "quoteAssetId": 2
            }]
        }
    }))
    .expect("light response")
}

fn full_response(account_id: i64, symbol_id: i64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "clientMsgId": "symbol-contract-full-symbol",
        "payloadType": 2117,
        "payload": {
            "ctidTraderAccountId": account_id,
            "symbol": [{
                "symbolId": symbol_id,
                "pipPosition": 4,
                "lotSize": 10000000,
                "commissionType": 1,
                "preciseTradingCommissionRate": 3500,
                "swapCalculationType": 0,
                "swapLong": -7.2,
                "swapShort": 2.1,
                "pnlConversionFeeRate": 0
            }]
        }
    }))
    .expect("full response")
}

#[test]
fn cli_requires_exact_environment_account_symbol_identity_and_output_root() {
    let output = tempfile::tempdir().expect("output root");
    let base = args(output.path().to_str().expect("output path"));
    let prepared = ExactBrokerSymbolContractCaptureCliV1::try_parse_from(base.clone())
        .expect("strict CLI")
        .prepare()
        .expect("prepared exact capture");
    assert_eq!(prepared.binding().environment(), BrokerEnvironment::Demo);
    assert_eq!(prepared.binding().account_id(), 46_774_385);
    assert_eq!(prepared.binding().symbol_id(), 1);
    assert_eq!(prepared.binding().symbol_name(), "EURUSD");
    assert_eq!(prepared.output_root(), output.path());

    for flag in [
        "--environment",
        "--account-id",
        "--symbol-id",
        "--symbol-name",
        "--output-root",
    ] {
        let mut missing = base.clone();
        let index = missing
            .iter()
            .position(|value| value == flag)
            .expect("required flag");
        missing.drain(index..=index + 1);
        assert!(ExactBrokerSymbolContractCaptureCliV1::try_parse_from(missing).is_err());
    }
}

#[test]
fn validated_light_and_full_responses_publish_two_exact_content_addressed_artifacts() {
    let output = tempfile::tempdir().expect("output root");
    let binding =
        ExactBrokerSymbolContractBindingV1::new(BrokerEnvironment::Demo, 46_774_385, 1, "EURUSD")
            .expect("binding");
    let light = light_response(
        binding.account_id(),
        binding.symbol_id(),
        binding.symbol_name(),
    );
    let full = full_response(binding.account_id(), binding.symbol_id());

    let receipt = publish_validated_broker_symbol_contract_response_v1(
        output.path(),
        &binding,
        &light,
        &full,
    )
    .expect("publish exact metadata evidence");

    assert_eq!(receipt.binding(), &binding);
    assert_eq!(std::fs::read(receipt.light_symbols_path()).unwrap(), light);
    assert_eq!(std::fs::read(receipt.full_symbol_path()).unwrap(), full);
    assert!(
        receipt
            .light_symbols_path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("bsl1-")
    );
    assert!(
        receipt
            .full_symbol_path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("bsc1-")
    );
}

#[test]
fn account_symbol_or_name_drift_refuses_publication() {
    let binding =
        ExactBrokerSymbolContractBindingV1::new(BrokerEnvironment::Demo, 46_774_385, 1, "EURUSD")
            .expect("binding");
    for (light, full) in [
        (
            light_response(46_774_386, 1, "EURUSD"),
            full_response(46_774_385, 1),
        ),
        (
            light_response(46_774_385, 2, "EURUSD"),
            full_response(46_774_385, 1),
        ),
        (
            light_response(46_774_385, 1, "GBPUSD"),
            full_response(46_774_385, 1),
        ),
        (
            light_response(46_774_385, 1, "EURUSD"),
            full_response(46_774_385, 2),
        ),
    ] {
        let output = tempfile::tempdir().expect("output root");
        assert!(
            publish_validated_broker_symbol_contract_response_v1(
                output.path(),
                &binding,
                &light,
                &full,
            )
            .is_err()
        );
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
    }
}

#[test]
fn production_source_is_exact_account_metadata_only() {
    for required in [
        "load_exact_production_historical_credentials",
        "build_application_auth_request",
        "build_account_auth_request",
        "build_symbols_list_request",
        "build_symbol_by_id_request",
        "ProductionCTraderOpenApiSession",
        "publish_validated_broker_symbol_contract_response_v1",
    ] {
        assert!(SOURCE.contains(required), "missing exact route {required}");
    }
    for forbidden in [
        "accounts.first",
        "enabled_for_execution",
        "legacy_fallback",
        "build_get_tick_data_request",
        "capture_tick",
        "resample",
        "use_current",
    ] {
        assert!(
            !SOURCE.contains(forbidden),
            "metadata capture contains forbidden route {forbidden}"
        );
    }
    for required in [
        "initialize_source_seal_before_runtime",
        "ExactBrokerSymbolContractCaptureCliV1::try_parse_from",
        "capture_exact_production_broker_symbol_contract_v1",
        "render_exact_broker_symbol_contract_receipt_v1",
    ] {
        assert!(
            BINARY.contains(required),
            "missing strict binary route {required}"
        );
    }
}
