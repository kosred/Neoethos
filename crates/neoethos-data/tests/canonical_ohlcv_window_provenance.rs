use std::sync::Arc;

use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, Ohlcv, load_canonical_timeframe,
    publish_canonical_ohlcv_generation,
};
use neoethos_feature_contracts::{SourceArtifactBindingV1, SourceSegmentV1};

const START_MS: i64 = 1_704_067_200_000;
const STEP_MS: i64 = 300_000;

fn fixture(rows: usize) -> Ohlcv {
    let timestamp = (0..rows)
        .map(|row| START_MS + row as i64 * STEP_MS)
        .collect::<Vec<_>>();
    let close = (0..rows)
        .map(|row| 1.08 + row as f64 * 0.000_01)
        .collect::<Vec<_>>();
    Ohlcv {
        timestamp: Some(timestamp),
        open: close.clone(),
        high: close.iter().map(|value| value + 0.000_2).collect(),
        low: close.iter().map(|value| value - 0.000_2).collect(),
        close,
        volume: Some((0..rows).map(|row| 100.0 + row as f64).collect()),
    }
}

fn source() -> (
    tempfile::TempDir,
    CanonicalDatasetIdentity,
    neoethos_data::CanonicalOhlcvFrame,
) {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = CanonicalDatasetIdentity::external(
        "canonical-window-test",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical identity");
    let bars = fixture(8);
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.test-source.v1", b"canonical-window".to_vec())
            .expect("producer provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &bars,
        volume: CanonicalVolumeRef::Float64(bars.volume.as_deref().expect("volume")),
        rows_per_chunk: 4,
    })
    .expect("publish generation");
    let frame = load_canonical_timeframe(root.path(), &identity).expect("load generation");
    (root, identity, frame)
}

fn expected_binding(
    frame: &neoethos_data::CanonicalOhlcvFrame,
    source_node_id: &str,
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
) -> SourceArtifactBindingV1 {
    let full = frame
        .artifact()
        .source_binding(source_node_id)
        .expect("full binding");
    SourceArtifactBindingV1::new(
        source_node_id,
        full.dataset_identity().clone(),
        full.manifest_schema_id(),
        *full.manifest_hash(),
        full.generation_id(),
        *full.vortex_hash(),
        full.bar_timestamp_convention(),
        vec![
            SourceSegmentV1::new(row_start, row_end, timestamp_start_ms, timestamp_end_ms)
                .expect("expected source segment"),
        ],
    )
    .expect("expected source binding")
}

#[test]
fn row_window_retains_the_generation_but_binds_only_exact_original_rows() {
    let (_root, identity, full) = source();
    let window = full.row_window(2, 6).expect("checked row window");

    assert_eq!(window.artifact().identity(), &identity);
    assert_eq!(
        window.artifact().generation_id(),
        full.artifact().generation_id()
    );
    assert!(Arc::ptr_eq(
        window.artifact().lease(),
        full.artifact().lease()
    ));
    assert_eq!(
        window.ohlcv().timestamp.as_deref(),
        Some(
            &[
                START_MS + 2 * STEP_MS,
                START_MS + 3 * STEP_MS,
                START_MS + 4 * STEP_MS,
                START_MS + 5 * STEP_MS,
            ][..]
        )
    );
    assert_eq!(
        window
            .source_binding("source:test")
            .expect("window binding"),
        expected_binding(
            &full,
            "source:test",
            2,
            6,
            START_MS + 2 * STEP_MS,
            START_MS + 5 * STEP_MS,
        )
    );

    let nested = window.row_window(1, 3).expect("nested checked window");
    assert_eq!(
        nested
            .source_binding("source:test")
            .expect("nested binding"),
        expected_binding(
            &full,
            "source:test",
            3,
            5,
            START_MS + 3 * STEP_MS,
            START_MS + 4 * STEP_MS,
        )
    );
}

#[test]
fn canonical_windows_refuse_empty_or_out_of_bounds_ranges() {
    let (_root, _identity, full) = source();
    assert!(full.row_window(2, 2).is_err());
    assert!(full.row_window(5, 4).is_err());
    assert!(full.row_window(0, 9).is_err());
    assert!(full.prefix_before_timestamp_ms(START_MS).is_err());
}

#[test]
fn half_open_timestamp_cutoff_records_the_exact_direct_prefix() {
    let (_root, _identity, full) = source();
    let cutoff = START_MS + 5 * STEP_MS;
    let prefix = full
        .prefix_before_timestamp_ms(cutoff)
        .expect("non-empty prefix");

    assert!(
        prefix
            .ohlcv()
            .timestamp
            .as_deref()
            .expect("timestamps")
            .iter()
            .all(|timestamp| *timestamp < cutoff)
    );
    assert_eq!(
        prefix
            .source_binding("source:test")
            .expect("prefix binding"),
        expected_binding(&full, "source:test", 0, 5, START_MS, START_MS + 4 * STEP_MS,)
    );
}
