use neoethos_broker_history::cli::{
    HISTORICAL_START_2016_UNIX_MS, HistoricalFetchCli, render_receipt_stdout,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
    SelectedDatasetGenerationV1,
};

fn receipt() -> SelectedDatasetGenerationV1 {
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        42,
        1,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    SelectedDatasetGenerationV1::new(
        identity,
        format!("g1-{}.vortex", "1".repeat(64)),
        "2".repeat(64),
    )
    .expect("receipt")
}

#[test]
fn one_request_has_fixed_2016_start_and_a_direct_broker_timeframe() {
    let cli = HistoricalFetchCli::try_parse_from([
        "neoethos-historical-fetch",
        "--symbol",
        "EURUSD",
        "--timeframe",
        "M5",
        "--data-root",
        "./data",
    ])
    .expect("strict one-request CLI");
    let request = cli.into_requests().expect("capture request");
    assert_eq!(request.len(), 1);
    assert_eq!(request[0].from_ms, HISTORICAL_START_2016_UNIX_MS);
    assert_eq!(request[0].timeframe, CanonicalTimeframe::M5);
}

#[test]
fn success_stdout_is_only_the_exact_strict_receipt_json() {
    let receipt = receipt();
    let mut expected = receipt.to_json_bytes().expect("strict receipt JSON");
    expected.push(b'\n');
    assert_eq!(render_receipt_stdout(&receipt).expect("stdout"), expected);
    let value: serde_json::Value = serde_json::from_slice(&expected).expect("JSON object");
    let object = value.as_object().expect("receipt object");
    assert_eq!(object.len(), 5);
    assert!(object.contains_key("dataset_identity"));
    assert!(object.contains_key("generation_id"));
    assert!(object.contains_key("manifest_binding_sha256"));
    assert!(!object.contains_key("result"));
    assert!(!object.contains_key("durable_commit_id"));
}

#[test]
fn bare_generation_current_fallback_and_resampling_flags_are_rejected() {
    for forbidden in [
        ["--generation", "g1-deadbeef.vortex"],
        ["--use-current", "true"],
        ["--resample-from", "M1"],
        ["--from", "2020-01-01"],
    ] {
        let args = [
            "neoethos-historical-fetch",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "M5",
            "--data-root",
            "./data",
            forbidden[0],
            forbidden[1],
        ];
        assert!(HistoricalFetchCli::try_parse_from(args).is_err());
    }
}

#[test]
fn refresh_file_requires_the_complete_selected_generation_wire_shape() {
    let malformed = br#"{"generation_id":"g1-deadbeef.vortex"}"#;
    assert!(SelectedDatasetGenerationV1::from_json_bytes(malformed).is_err());
}
