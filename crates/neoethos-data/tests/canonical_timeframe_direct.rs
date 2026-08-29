use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, Ohlcv, load_dataset_for_identity,
    publish_canonical_ohlcv_generation, require_direct_timeframes,
};

const BASE_MS: i64 = 1_700_000_100_000;

fn direct_fixture(timeframe: CanonicalTimeframe, rows: usize) -> Ohlcv {
    let period_ms = timeframe
        .fixed_duration_ms()
        .expect("test uses fixed broker periods");
    let values = (0..rows)
        .map(|index| 100.0 + index as f64)
        .collect::<Vec<_>>();
    Ohlcv {
        timestamp: Some(
            (0..rows)
                .map(|index| BASE_MS + index as i64 * period_ms)
                .collect(),
        ),
        open: values.clone(),
        high: values.iter().map(|value| value + 1.0).collect(),
        low: values.iter().map(|value| value - 1.0).collect(),
        close: values.iter().map(|value| value + 0.5).collect(),
        volume: Some((0..rows).map(|index| index as f64).collect()),
    }
}

fn publish_direct(
    root: &std::path::Path,
    namespace: &str,
    timeframe: CanonicalTimeframe,
) -> CanonicalDatasetIdentity {
    let identity = CanonicalDatasetIdentity::external(
        namespace,
        "TESTFX",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("direct test identity");
    let frame = direct_fixture(timeframe, 20);
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.direct-timeframe-test.v1",
        format!("direct-{namespace}-{timeframe}").into_bytes(),
    )
    .expect("direct producer provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &frame,
        volume: CanonicalVolumeRef::Float64(frame.volume.as_deref().expect("volume")),
        rows_per_chunk: 8,
    })
    .expect("publish direct timeframe");
    identity
}

#[test]
fn missing_higher_timeframe_fails_instead_of_deriving_from_m1() {
    let root = tempfile::tempdir().expect("canonical root");
    let m1 = publish_direct(root.path(), "direct-series", CanonicalTimeframe::M1);
    let dataset = load_dataset_for_identity(root.path(), &m1).expect("load direct M1");

    let error = require_direct_timeframes(
        &dataset,
        &m1,
        &[CanonicalTimeframe::M1, CanonicalTimeframe::M5],
    )
    .expect_err("M5 may never be manufactured from M1");
    assert!(
        error
            .to_string()
            .contains("missing direct canonical timeframe M5")
    );
}

#[test]
fn independently_published_same_series_timeframes_are_accepted() {
    let root = tempfile::tempdir().expect("canonical root");
    let m1 = publish_direct(root.path(), "direct-series", CanonicalTimeframe::M1);
    publish_direct(root.path(), "direct-series", CanonicalTimeframe::M5);
    let dataset = load_dataset_for_identity(root.path(), &m1).expect("load direct series");

    require_direct_timeframes(
        &dataset,
        &m1,
        &[CanonicalTimeframe::M1, CanonicalTimeframe::M5],
    )
    .expect("both timeframes have independent direct generations");
    assert_eq!(
        dataset.source_artifacts["M5"].identity().timeframe(),
        CanonicalTimeframe::M5
    );
}

#[test]
fn a_foreign_namespace_timeframe_cannot_fill_the_selected_series() {
    let root = tempfile::tempdir().expect("canonical root");
    let m1 = publish_direct(root.path(), "selected-series", CanonicalTimeframe::M1);
    publish_direct(root.path(), "foreign-series", CanonicalTimeframe::M5);
    let dataset = load_dataset_for_identity(root.path(), &m1).expect("load selected series");

    assert!(
        require_direct_timeframes(
            &dataset,
            &m1,
            &[CanonicalTimeframe::M1, CanonicalTimeframe::M5]
        )
        .is_err()
    );
}
