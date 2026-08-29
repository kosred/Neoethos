use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, FeatureBuildOptions, Ohlcv,
    load_dataset_for_identity_with_timeframes, prepare_multitimeframe_features_before_with_options,
    publish_canonical_ohlcv_generation,
};
use neoethos_feature_contracts::{SourceArtifactBindingV1, SourceSegmentV1};

const START_MS: i64 = 1_704_067_200_000;
const M5_MS: i64 = 300_000;
const H1_MS: i64 = 3_600_000;
const CUTOFF_MS: i64 = START_MS + 20 * H1_MS;

fn fixture(rows: usize, step_ms: i64, oos_shift: f64) -> Ohlcv {
    let timestamp = (0..rows)
        .map(|row| START_MS + row as i64 * step_ms)
        .collect::<Vec<_>>();
    let close = timestamp
        .iter()
        .enumerate()
        .map(|(row, timestamp)| {
            let base = 1.08 + (row as f64 * 0.037).sin() * 0.001;
            if *timestamp >= CUTOFF_MS {
                base + oos_shift
            } else {
                base
            }
        })
        .collect::<Vec<_>>();
    Ohlcv {
        timestamp: Some(timestamp),
        open: close.clone(),
        high: close.iter().map(|value| value + 0.000_2).collect(),
        low: close.iter().map(|value| value - 0.000_2).collect(),
        close,
        volume: Some((0..rows).map(|row| 10_000.0 + row as f64).collect()),
    }
}

fn publish(
    root: &std::path::Path,
    identity: &CanonicalDatasetIdentity,
    bars: &Ohlcv,
    source: &str,
) {
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.test-source.v1", source.as_bytes().to_vec())
            .expect("producer provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: bars,
        volume: CanonicalVolumeRef::Float64(bars.volume.as_deref().expect("volume")),
        rows_per_chunk: 128,
    })
    .expect("publish generation");
}

fn build_series(oos_shift: f64) -> (tempfile::TempDir, neoethos_data::SymbolDataset) {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let base_identity = CanonicalDatasetIdentity::external(
        "canonical-feature-window-test",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("base identity");
    let higher_identity = CanonicalDatasetIdentity::external(
        "canonical-feature-window-test",
        "EURUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("higher identity");
    publish(
        root.path(),
        &base_identity,
        &fixture(400, M5_MS, oos_shift),
        &format!("m5-{oos_shift}"),
    );
    publish(
        root.path(),
        &higher_identity,
        &fixture(40, H1_MS, oos_shift),
        &format!("h1-{oos_shift}"),
    );
    let dataset =
        load_dataset_for_identity_with_timeframes(root.path(), &base_identity, &["M5", "H1"])
            .expect("load exact direct series");
    (root, dataset)
}

fn expected_binding(
    dataset: &neoethos_data::SymbolDataset,
    timeframe: &str,
    source_node_id: &str,
) -> SourceArtifactBindingV1 {
    let artifact = dataset
        .source_artifacts
        .get(timeframe)
        .expect("direct artifact");
    let full = artifact
        .source_binding(source_node_id)
        .expect("full binding");
    let timestamps = dataset
        .timeframe(timeframe)
        .expect("direct frame")
        .timestamp
        .as_deref()
        .expect("timestamps");
    let row_end = timestamps.partition_point(|timestamp| *timestamp < CUTOFF_MS);
    SourceArtifactBindingV1::new(
        source_node_id,
        full.dataset_identity().clone(),
        full.manifest_schema_id(),
        *full.manifest_hash(),
        full.generation_id(),
        *full.vortex_hash(),
        full.bar_timestamp_convention(),
        vec![
            SourceSegmentV1::new(0, row_end as u64, timestamps[0], timestamps[row_end - 1])
                .expect("expected source segment"),
        ],
    )
    .expect("expected source binding")
}

fn options() -> FeatureBuildOptions {
    FeatureBuildOptions {
        higher_tfs: vec!["H1".to_owned()],
        normalization_training_rows: Some(0..240),
        ..FeatureBuildOptions::default()
    }
}

#[test]
fn every_direct_timeframe_is_independently_clipped_and_bound_to_the_same_cutoff() {
    let (_root, dataset) = build_series(0.0);
    let frame =
        prepare_multitimeframe_features_before_with_options(&dataset, "M5", &options(), CUTOFF_MS)
            .expect("windowed direct feature cube");

    assert_eq!(frame.n_samples(), 240);
    assert!(
        frame
            .timestamps
            .iter()
            .all(|timestamp| *timestamp < CUTOFF_MS)
    );
    assert_eq!(frame.provenance().bindings().len(), 2);
    for timeframe in ["M5", "H1"] {
        let identity = dataset
            .source_artifacts
            .get(timeframe)
            .expect("direct artifact")
            .identity();
        let binding = frame
            .provenance()
            .bindings()
            .iter()
            .find(|binding| binding.dataset_identity() == identity)
            .expect("feature source binding");
        assert_eq!(
            binding,
            &expected_binding(&dataset, timeframe, binding.source_node_id())
        );
    }
}

#[test]
fn changing_only_oos_values_cannot_change_any_in_sample_feature_or_normalization_input() {
    let (_first_root, first_dataset) = build_series(0.0);
    let (_second_root, second_dataset) = build_series(1000.0);
    let first = prepare_multitimeframe_features_before_with_options(
        &first_dataset,
        "M5",
        &options(),
        CUTOFF_MS,
    )
    .expect("first windowed feature cube");
    let second = prepare_multitimeframe_features_before_with_options(
        &second_dataset,
        "M5",
        &options(),
        CUTOFF_MS,
    )
    .expect("second windowed feature cube");

    assert_eq!(first.timestamps, second.timestamps);
    assert_eq!(first.names, second.names);
    assert_eq!(first.n_features(), second.n_features());
    for column in 0..first.n_features() {
        let left = first.feature_column(column).expect("first feature column");
        let right = second
            .feature_column(column)
            .expect("second feature column");
        assert_eq!(
            left.validity, right.validity,
            "validity differs at column {column}"
        );
        assert_eq!(
            left.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "values differ at column {column}"
        );
    }
}
