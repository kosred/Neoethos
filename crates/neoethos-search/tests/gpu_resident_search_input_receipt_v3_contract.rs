#![cfg(feature = "gpu-cuda")]

use neoethos_search::CanonicalGpuResidentSearchInputReceiptV3;

#[test]
fn gpu_resident_v3_receipt_rejects_missing_fields_instead_of_decoding_as_v2() {
    let error = CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(br#"{}"#)
        .expect_err("missing GPU-native receipt authority must fail closed");
    assert!(
        error.to_string().contains("parse GPU-resident JSON"),
        "unexpected error: {error}"
    );
}

#[test]
fn gpu_resident_v3_receipt_rejects_a_cpu_v2_shaped_payload() {
    let legacy = br#"{
        "schema_version":2,
        "anchor_dataset_identity":"legacy",
        "feature_plan_identity":"00",
        "feature_provenance_identity":"00",
        "feature_content_sha256":"00",
        "feature_execution":{},
        "source_bindings":[]
    }"#;
    assert!(CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(legacy).is_err());
}
