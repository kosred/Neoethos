use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;

#[test]
fn producer_envelope_binds_schema_payload_and_hash() {
    let payload = br#"{"source":"fixture","version":1}"#.to_vec();
    let envelope = ProducerProvenanceEnvelopeV1::new("neoethos.test-producer.v1", payload.clone())
        .expect("valid producer envelope");

    envelope.validate().expect("untampered envelope validates");
    assert_eq!(envelope.schema_id(), "neoethos.test-producer.v1");
    assert_eq!(envelope.canonical_payload(), payload);

    let encoded = envelope.to_json_bytes().expect("serialize envelope");
    let mut wire: serde_json::Value = serde_json::from_slice(&encoded).expect("JSON envelope");
    wire["canonical_payload"] = serde_json::json!([1, 2, 3]);
    let tampered = serde_json::to_vec(&wire).expect("tampered JSON");
    assert!(
        ProducerProvenanceEnvelopeV1::from_json_bytes(&tampered).is_err(),
        "payload/hash mismatch must fail closed"
    );
}

#[test]
fn producer_envelope_rejects_bad_schema_ids_and_oversized_payloads() {
    for schema in ["", "CSV", "neoethos..v1", "../escape", "neoethos/a.v1"] {
        assert!(
            ProducerProvenanceEnvelopeV1::new(schema, vec![]).is_err(),
            "schema {schema:?} must be rejected"
        );
    }
    assert!(
        ProducerProvenanceEnvelopeV1::new(
            "neoethos.test-producer.v1",
            vec![0_u8; ProducerProvenanceEnvelopeV1::MAX_PAYLOAD_BYTES + 1],
        )
        .is_err()
    );
}
