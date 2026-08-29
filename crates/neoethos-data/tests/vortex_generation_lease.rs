use neoethos_data::core::dataset_manifest::{
    DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest,
    collect_unreferenced_generations, open_current_generation, publish_vortex_generation,
};
use neoethos_data::{Ohlcv, ohlcv_to_vortex_chunks};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use tempfile::tempdir;

fn publish(
    root: &std::path::Path,
    identity: &CanonicalDatasetIdentity,
    expected: Option<&str>,
    value: f64,
) -> String {
    const BASE_MS: i64 = 1_700_000_040_000;
    let data = Ohlcv {
        timestamp: Some(vec![BASE_MS, BASE_MS + 60_000]),
        open: vec![value, value],
        high: vec![value + 1.0, value + 1.0],
        low: vec![value - 0.5, value - 0.5],
        close: vec![value, value],
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.fixture.v1",
        value.to_bits().to_be_bytes().to_vec(),
    )
    .expect("provenance");
    publish_vortex_generation(PublishRequest {
        configured_root: root,
        identity,
        expected_generation: expected,
        timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 60_000).expect("range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&data, 1).expect("chunks"),
    })
    .expect("publication")
    .generation()
    .to_owned()
}

#[test]
fn a_live_reader_pin_survives_two_publications_and_gc() {
    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "XAUUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let generation_n = publish(root.path(), &identity, None, 1.0);
    let reader = open_current_generation(root.path(), &identity).expect("pin generation N");
    let generation_n1 = publish(root.path(), &identity, Some(&generation_n), 2.0);
    let _generation_n2 = publish(root.path(), &identity, Some(&generation_n1), 3.0);

    collect_unreferenced_generations(root.path(), &identity).expect("gc with live reader");
    assert!(
        reader.path().exists(),
        "GC must preserve a live reader generation"
    );
    reader.reopen_verified().expect("lazy reopen while pinned");

    let old_path = reader.path().to_path_buf();
    drop(reader);
    collect_unreferenced_generations(root.path(), &identity).expect("gc after release");
    assert!(
        !old_path.exists(),
        "released unreferenced generation becomes collectible"
    );
}
