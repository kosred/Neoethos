use neoethos_data::core::dataset_manifest::{
    DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest, publish_vortex_generation,
};
use neoethos_data::{
    FeatureProfile, Ohlcv, compute_hpc_feature_frame_sized, load_canonical_timeframe,
    ohlcv_to_vortex_chunks,
};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};

const START_MS: i64 = 1_704_067_200_000;
const STEP_MS: i64 = 300_000;

fn fixture(rows: usize) -> Ohlcv {
    let mut timestamp = Vec::with_capacity(rows);
    let mut open = Vec::with_capacity(rows);
    let mut high = Vec::with_capacity(rows);
    let mut low = Vec::with_capacity(rows);
    let mut close = Vec::with_capacity(rows);
    let mut volume = Vec::with_capacity(rows);
    let mut price = 1.08_f64;
    for row in 0..rows {
        let phase = row as f64 * 0.037;
        let next = price + phase.sin() * 0.000_3 + phase.cos() * 0.000_11;
        timestamp.push(START_MS + row as i64 * STEP_MS);
        open.push(price);
        high.push(price.max(next) + 0.000_2);
        low.push(price.min(next) - 0.000_2);
        close.push(next);
        volume.push(10_000_000.0 + row as f64);
        price = next;
    }
    Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    }
}

#[test]
fn canonical_generation_is_pinned_and_bound_to_the_feature_frame() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = CanonicalDatasetIdentity::external(
        "canonical-feature-input-test",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical identity");
    let bars = fixture(512);
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.test-source.v1",
        b"canonical-feature-input".to_vec(),
    )
    .expect("producer provenance");
    let published = publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(
            START_MS,
            START_MS + (bars.len() as i64 - 1) * STEP_MS,
        )
        .expect("timestamp range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&bars, 128).expect("bounded Vortex chunks"),
    })
    .expect("publish canonical generation");

    let source = load_canonical_timeframe(root.path(), &identity).expect("verified pinned source");
    assert_eq!(source.artifact().generation_id(), published.generation());
    assert_eq!(source.ohlcv().close, bars.close);

    let frame = compute_hpc_feature_frame_sized(&source, FeatureProfile::HPC, source.len())
        .expect("provenance-bound f64 feature frame");
    assert_eq!(frame.n_samples(), bars.len());
    assert_eq!(frame.plan().final_outputs(), frame.names);
    assert_eq!(frame.provenance().bindings().len(), 1);
    assert_eq!(
        frame.provenance().bindings()[0].dataset_identity(),
        &identity
    );
    assert_eq!(
        frame.provenance().bindings()[0].generation_id(),
        published.generation()
    );
}

#[test]
fn bare_ohlcv_has_no_feature_computation_entrypoint() {
    // This compile-time contract is intentional: callers must load a verified
    // immutable generation (or use a separately typed live/derived source), so
    // production code cannot fabricate dataset provenance from a symbol string.
    let function: fn(
        &neoethos_data::CanonicalOhlcvFrame,
        FeatureProfile,
        usize,
    ) -> anyhow::Result<neoethos_data::FeatureFrame> = compute_hpc_feature_frame_sized;
    let _ = function;
}
