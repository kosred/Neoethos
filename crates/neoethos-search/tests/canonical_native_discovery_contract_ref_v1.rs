#![cfg(target_os = "linux")]
use neoethos_core::Settings;
use neoethos_search::{
    CanonicalNativeDiscoveryRequestErrorV1 as Error, CanonicalResearchContractArtifactRefV1,
    CanonicalSearchInputReceiptV2, CanonicalTrendbarResearchCostAssumptionsV2,
    CanonicalTrendbarResearchExecutionContractV3, MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
    MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1, SealedCanonicalRootV1,
    load_canonical_research_contract_artifact_v1,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tempfile::TempDir;
const SCHEMA: &str = "neoethos.canonical-research-contract-artifact-ref.v1";
fn settings(root: &Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_path_buf();
    settings
}
fn contract(symbol: &str) -> CanonicalTrendbarResearchExecutionContractV3 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
        .expect("fixture receipt");
    let source_sha = format!("{:x}", Sha256::digest(b"contract-ref-v1-fixture"));
    CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol,
            account_currency: "USD",
            assumption_source_id: "neoethos.test.contract-ref.v1",
            assumption_source_sha256: &source_sha,
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.2,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .expect("fixture contract")
}
fn write_bytes(root: &Path, relative: &str, bytes: &[u8]) -> String {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("artifact parent")).expect("create parent");
    fs::write(path, bytes).expect("write artifact");
    format!("{:x}", Sha256::digest(bytes))
}
type LoadResult = Result<neoethos_search::LoadedCanonicalResearchContractV1, Error>;
fn load_ref(root: &TempDir, relative: &str, sha: String) -> LoadResult {
    let sealed = SealedCanonicalRootV1::from_startup_settings(&settings(root.path()))?;
    let reference = CanonicalResearchContractArtifactRefV1::checked_new(relative, sha)?;
    load_canonical_research_contract_artifact_v1(&sealed, reference)
}
fn load_bytes(root: &TempDir, relative: &str, bytes: &[u8]) -> LoadResult {
    load_ref(root, relative, write_bytes(root.path(), relative, bytes))
}
fn assert_contract_limit(root: &TempDir, name: &str, value: &Value, expected: &'static str) {
    let bytes = serde_json::to_vec(value).expect("encode capped contract");
    assert!(matches!(
        load_bytes(root, name, &bytes),
        Err(Error::RequestLimitExceeded { limit }) if limit == expected
    ));
}
fn assert_bad_ref(path: &str, hash: String) {
    assert!(CanonicalResearchContractArtifactRefV1::checked_new(path, hash).is_err());
}
#[test]
fn valid_exact_artifact_returns_distinct_immutable_file_and_contract_evidence() {
    let root = TempDir::new().expect("root");
    let contract = contract("EURUSD");
    let bytes = serde_json::to_vec(&contract).expect("serialize");
    let relative = "research/contracts/contract.json";
    let exact_sha = format!("{:x}", Sha256::digest(&bytes));
    let loaded = load_bytes(&root, relative, &bytes).expect("load exact contract");
    fs::write(root.path().join(relative), b"changed after load").expect("mutate path");
    assert_eq!(loaded.relative_path(), relative);
    assert_eq!(loaded.exact_artifact_sha256(), exact_sha);
    assert_eq!(loaded.byte_len(), bytes.len() as u64);
    assert_eq!(loaded.contract().symbol(), "EURUSD");
    #[cfg(feature = "gpu-cuda")]
    {
        assert_eq!(loaded.source_projection().bindings().len(), 1);
        assert_eq!(
            loaded.source_projection().anchor_dataset_identity(),
            loaded.source_projection().bindings()[0].dataset_identity()
        );
    }
    assert_eq!(
        loaded.contract_identity_sha256(),
        contract.identity_sha256().expect("domain identity")
    );
    assert_ne!(loaded.contract_identity_sha256(), exact_sha);
}
#[test]
fn reference_validation_and_wire_decode_fail_closed() {
    let hash = "0".repeat(64);
    for path in
        "|/a|a/|a//b|.|..|a/./b|a/../b|a\\b|C:a|C:/a|\\\\server\\share|a/\0b|a/\u{001f}b".split('|')
    {
        assert_bad_ref(path, hash.clone());
    }
    for invalid in [
        String::new(),
        "0".repeat(63),
        "0".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert_bad_ref("a.json", invalid);
    }
    let reference = CanonicalResearchContractArtifactRefV1::checked_new("a.json", "1".repeat(64))
        .expect("reference");
    let value = serde_json::to_value(reference).expect("serialize reference");
    assert_eq!(value["schema"], SCHEMA);
    assert_eq!(value["version"], 1);
    serde_json::from_value::<CanonicalResearchContractArtifactRefV1>(value.clone())
        .expect("round trip");
    let mut extra = value.clone();
    extra["extra"] = json!(1);
    let mut invalid = vec![extra];
    for field in ["schema", "version", "relative_path", "expected_sha256"] {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        invalid.push(missing);
    }
    let mut replace = |field: &str, replacement: Value| {
        let mut wrong = value.clone();
        wrong[field] = replacement;
        invalid.push(wrong);
    };
    for hash in ["1".repeat(63), "A".repeat(64), "g".repeat(64)] {
        replace("expected_sha256", json!(hash));
    }
    for (field, replacement) in [
        ("schema", json!("wrong")),
        ("version", json!(2)),
        ("relative_path", json!("../a.json")),
    ] {
        replace(field, replacement);
    }
    for invalid in invalid {
        assert!(serde_json::from_value::<CanonicalResearchContractArtifactRefV1>(invalid).is_err());
    }
    let duplicate = format!(
        r#"{{"schema":"{SCHEMA}","version":1,"relative_path":"a.json","expected_sha256":"{0}","expected_sha256":"{0}"}}"#,
        "1".repeat(64)
    );
    assert!(serde_json::from_str::<CanonicalResearchContractArtifactRefV1>(&duplicate).is_err());
}
#[test]
fn hash_decode_schema_and_receipt_mismatches_fail_closed() {
    let root = TempDir::new().expect("root");
    let error = |result: LoadResult| result.err().unwrap();
    let bytes = serde_json::to_vec(&contract("EURUSD")).expect("serialize");
    write_bytes(root.path(), "wrong-hash.json", &bytes);
    assert!(matches!(
        error(load_ref(&root, "wrong-hash.json", "0".repeat(64))),
        Error::ArtifactHashMismatch { .. }
    ));
    let mut unknown: Value = serde_json::from_slice(&bytes).expect("value");
    unknown["unknown"] = json!(true);
    for (name, invalid) in [
        ("unknown.json", serde_json::to_vec(&unknown).unwrap()),
        ("malformed.json", b"{".to_vec()),
    ] {
        assert!(matches!(
            error(load_bytes(&root, name, &invalid)),
            Error::ContractDecode(_)
        ));
    }
    let mut schema: Value = serde_json::from_slice(&bytes).expect("value");
    schema["schema_version"] = json!(4);
    let schema = serde_json::to_vec(&schema).unwrap();
    assert!(matches!(
        error(load_bytes(&root, "schema.json", &schema)),
        Error::ContractValidation(_)
    ));
    let mismatch = contract("GBPUSD");
    mismatch
        .validate()
        .expect("plain validation intentionally misses anchor");
    let bytes = serde_json::to_vec(&mismatch).expect("serialize");
    assert!(matches!(
        error(load_bytes(&root, "mismatch.json", &bytes)),
        Error::ContractValidation(_)
    ));
}

#[test]
fn oversized_nested_strings_fail_with_the_named_cap() {
    let root = TempDir::new().expect("root");
    let template = serde_json::to_value(contract("EURUSD")).expect("value");
    for (index, pointer) in [
        "/input_receipt/anchor_dataset_identity",
        "/input_receipt/feature_plan_identity",
        "/input_receipt/feature_provenance_identity",
        "/input_receipt/feature_content_sha256",
        "/input_receipt/feature_execution/vector_ta_math_authority",
        "/input_receipt/source_bindings/0/source_node_id",
        "/input_receipt/source_bindings/0/dataset_identity",
        "/input_receipt/source_bindings/0/manifest_schema_id",
        "/input_receipt/source_bindings/0/manifest_sha256",
        "/input_receipt/source_bindings/0/generation_id",
        "/input_receipt/source_bindings/0/vortex_sha256",
        "/input_receipt/source_bindings/0/bar_timestamp_convention",
        "/input_receipt_sha256",
        "/symbol",
        "/account_currency",
        "/assumption_source_id",
        "/assumption_source_sha256",
    ]
    .into_iter()
    .enumerate()
    {
        let mut value = template.clone();
        *value.pointer_mut(pointer).expect("string field") =
            json!("x".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 + 1));
        assert_contract_limit(
            &root,
            &format!("nested-string-{index}.json"),
            &value,
            "string_bytes_cap",
        );
    }
}

#[test]
fn oversized_source_count_fails_before_projection_with_the_named_cap() {
    let root = TempDir::new().expect("root");
    let mut value = serde_json::to_value(contract("EURUSD")).expect("value");
    let binding = value
        .pointer("/input_receipt/source_bindings/0")
        .expect("binding")
        .clone();
    *value
        .pointer_mut("/input_receipt/source_bindings")
        .expect("bindings") =
        Value::Array(vec![binding; MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 + 1]);
    assert_contract_limit(&root, "source-count.json", &value, "source_count_cap");
}
