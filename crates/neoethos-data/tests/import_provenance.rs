use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::import_provenance::{
    ImportProvenanceV1, ImportSourceFormat, VolumeMappingV1,
};
use neoethos_data::{BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe};

fn identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "operator-upload",
        "EURUSD.raw",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity")
}

#[test]
fn every_import_format_has_a_canonical_round_trip() {
    let formats = [
        ImportSourceFormat::Csv,
        ImportSourceFormat::Tsv,
        ImportSourceFormat::JsonArray,
        ImportSourceFormat::JsonLines,
        ImportSourceFormat::Parquet,
        ImportSourceFormat::ArrowIpcFile,
        ImportSourceFormat::ArrowIpcStream,
        ImportSourceFormat::Vortex,
    ];

    for (index, format) in formats.into_iter().enumerate() {
        let provenance = ImportProvenanceV1::new(
            format,
            format,
            [index as u8; 32],
            1_024 + index as u64,
            format!("test-source-{index}"),
            identity(),
            1_723_000_000_000 + index as u64,
            VolumeMappingV1::SourceFloat64,
        )
        .expect("valid provenance");
        let canonical = provenance.canonical_bytes();
        let envelope = provenance.to_envelope().expect("encode envelope");
        let decoded = ImportProvenanceV1::from_envelope(&envelope).expect("decode envelope");

        assert_eq!(decoded, provenance);
        assert_eq!(decoded.canonical_bytes(), canonical);
        assert_eq!(decoded.source_sha256(), &[index as u8; 32]);
        assert_eq!(decoded.dataset_identity(), &identity());
    }
}

#[test]
fn explicit_adapter_labels_round_trip_without_aliases_or_guessing() {
    for format in ImportSourceFormat::ALL {
        assert_eq!(format.to_string(), format.as_str());
        assert_eq!(
            format.as_str().parse::<ImportSourceFormat>(),
            Ok(format),
            "canonical adapter label must parse exactly"
        );
    }

    for rejected in ["CSV", "json", "arrow", "ipc", "parquet ", ""] {
        assert!(
            rejected.parse::<ImportSourceFormat>().is_err(),
            "non-canonical adapter label {rejected:?} must not be guessed"
        );
    }
}

#[test]
fn import_reporting_rejects_wrong_schema_and_payload_tampering() {
    let provenance = ImportProvenanceV1::new(
        ImportSourceFormat::Csv,
        ImportSourceFormat::Csv,
        [0x5a; 32],
        789,
        "stable-file-id",
        identity(),
        1_723_000_000_000,
        VolumeMappingV1::Absent,
    )
    .expect("valid provenance");
    let envelope = provenance.to_envelope().expect("encode envelope");

    let wrong_schema = ProducerProvenanceEnvelopeV1::new(
        "neoethos.not-import-provenance.v1",
        envelope.canonical_payload().to_vec(),
    )
    .expect("valid generic envelope");
    assert!(
        ImportProvenanceV1::from_envelope(&wrong_schema)
            .unwrap_err()
            .to_string()
            .contains("schema")
    );

    let mut mutated = envelope.canonical_payload().to_vec();
    let selected_format_offset = b"neoethos.import-provenance.v1\0".len() + 2;
    mutated[selected_format_offset] = 0xff;
    let rebound = ProducerProvenanceEnvelopeV1::new(ImportProvenanceV1::SCHEMA_ID, mutated)
        .expect("generic envelope rebinds mutated bytes");
    assert!(ImportProvenanceV1::from_envelope(&rebound).is_err());
}
