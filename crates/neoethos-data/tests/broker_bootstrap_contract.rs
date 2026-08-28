#[path = "../../neoethos-app/src/app_services/bootstrap_writer.rs"]
mod bootstrap_writer;

use bootstrap_writer::{
    BrokerTrendbarStreamRequest, CTraderTrendbarProvenanceV1, ctrader_inclusive_wire_to_ms,
    publish_broker_trendbar_chunks, publish_broker_trendbar_chunks_exact,
};
use neoethos_data::core::dataset_manifest::{
    ExactDatasetGenerationConflict, read_current_manifest,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
    load_canonical_timeframe,
};

#[derive(Debug, Clone, PartialEq)]
struct TestBar {
    timestamp_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<i64>,
}

struct TestBrokerStreamRequest<'a> {
    configured_root: &'a std::path::Path,
    identity: &'a CanonicalDatasetIdentity,
    expected_generation: Option<&'a str>,
    requested_from_ms: i64,
    requested_to_ms: i64,
    retrieved_unix_ms: u64,
    bars: &'a [TestBar],
}

fn publish_test_bars(
    request: TestBrokerStreamRequest<'_>,
) -> anyhow::Result<neoethos_data::core::dataset_manifest::PublishResult> {
    let timestamps = request
        .bars
        .iter()
        .map(|bar| bar.timestamp_ms)
        .collect::<Vec<_>>();
    let volume = if request.bars.iter().all(|bar| bar.volume.is_some()) {
        CanonicalVolumeChunk::Int64(
            request
                .bars
                .iter()
                .map(|bar| bar.volume.expect("all test volumes are present"))
                .collect(),
        )
    } else {
        CanonicalVolumeChunk::Absent
    };
    let chunk = CanonicalOhlcvChunk {
        timestamp_ms: timestamps,
        open: request.bars.iter().map(|bar| bar.open).collect(),
        high: request.bars.iter().map(|bar| bar.high).collect(),
        low: request.bars.iter().map(|bar| bar.low).collect(),
        close: request.bars.iter().map(|bar| bar.close).collect(),
        volume,
    };
    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: request.configured_root,
        identity: request.identity,
        expected_generation: request.expected_generation,
        requested_from_ms: request.requested_from_ms,
        requested_to_ms: request.requested_to_ms,
        retrieved_unix_ms: request.retrieved_unix_ms,
        returned_from_ms: request.bars[0].timestamp_ms,
        returned_to_ms: request.bars[request.bars.len() - 1].timestamp_ms,
        row_count: request.bars.len() as u64,
        chunks: vec![Ok(chunk)],
    })
}

fn broker_identity() -> CanonicalDatasetIdentity {
    broker_identity_for(CanonicalTimeframe::M1)
}

fn broker_identity_for(timeframe: CanonicalTimeframe) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        7_001,
        14,
        "EURUSD",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid broker identity")
}

fn sample_bars() -> Vec<TestBar> {
    vec![
        TestBar {
            timestamp_ms: 1_700_000_040_000,
            open: 1.1,
            high: 1.2,
            low: 1.0,
            close: 1.15,
            volume: Some(10),
        },
        TestBar {
            timestamp_ms: 1_700_000_100_000,
            open: 1.15,
            high: 1.25,
            low: 1.05,
            close: 1.2,
            volume: Some(16_777_217),
        },
    ]
}

fn sample_calendar_bars(gap_ms: i64) -> Vec<TestBar> {
    let first_open_ms = 1_711_836_000_000_i64;
    let second_open_ms = first_open_ms + gap_ms;
    vec![
        TestBar {
            timestamp_ms: first_open_ms,
            open: 1.1,
            high: 1.2,
            low: 1.0,
            close: 1.15,
            volume: Some(10),
        },
        TestBar {
            timestamp_ms: second_open_ms,
            open: 1.15,
            high: 1.25,
            low: 1.05,
            close: 1.2,
            volume: Some(11),
        },
    ]
}

#[test]
fn broker_download_publishes_a_reopenable_canonical_generation() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = broker_identity();
    let bars = sample_bars();
    let publication = publish_test_bars(TestBrokerStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        requested_from_ms: bars[0].timestamp_ms,
        requested_to_ms: bars[1].timestamp_ms + 60_000,
        retrieved_unix_ms: 1_800_000_000_000,
        bars: &bars,
    })
    .expect("publish broker trendbars");

    let manifest = read_current_manifest(root.path(), &identity).expect("reopen manifest");
    assert_eq!(manifest.generation_id(), publication.generation());
    assert_eq!(manifest.row_count(), 2);
    assert!(manifest.generation_path().is_file());
    assert!(!root.path().join("symbol=EURUSD").exists());

    let provenance = CTraderTrendbarProvenanceV1::from_envelope(manifest.provenance())
        .expect("decode typed broker provenance");
    assert_eq!(provenance.dataset_identity(), &identity);
    assert_eq!(
        provenance.requested_range_ms(),
        (bars[0].timestamp_ms, bars[1].timestamp_ms + 60_000)
    );
    assert_eq!(
        provenance.returned_range_ms(),
        (bars[0].timestamp_ms, bars[1].timestamp_ms)
    );
    assert_eq!(provenance.row_count(), 2);
    assert_eq!(provenance.retrieved_unix_ms(), 1_800_000_000_000);

    let loaded = load_canonical_timeframe(root.path(), &identity).expect("load exact identity");
    assert_eq!(loaded.ohlcv().close, vec![1.15, 1.2]);
    assert_eq!(loaded.ohlcv().volume, Some(vec![10.0, 16_777_217.0]));
}

#[test]
fn broker_calendar_timeframes_use_direct_authoritative_bar_opens() {
    let cases = [
        (CanonicalTimeframe::D1, 23 * 60 * 60 * 1_000),
        (CanonicalTimeframe::W1, 167 * 60 * 60 * 1_000),
        (CanonicalTimeframe::MN1, 29 * 24 * 60 * 60 * 1_000),
    ];
    for (timeframe, gap_ms) in cases {
        let root = tempfile::tempdir().expect("temporary canonical root");
        let identity = broker_identity_for(timeframe);
        let bars = sample_calendar_bars(gap_ms);
        let first_open_ms = bars[0].timestamp_ms;
        let second_open_ms = bars[1].timestamp_ms;

        publish_test_bars(TestBrokerStreamRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            requested_from_ms: first_open_ms,
            requested_to_ms: second_open_ms + 35 * 24 * 60 * 60 * 1_000,
            retrieved_unix_ms: 1_800_000_000_000,
            bars: &bars,
        })
        .expect("direct broker calendar timestamps must not use a local fixed-duration grid");

        let loaded = load_canonical_timeframe(root.path(), &identity)
            .expect("reopen the exact direct broker calendar generation");
        assert_eq!(
            loaded.ohlcv().timestamp.as_deref(),
            Some([first_open_ms, second_open_ms].as_slice())
        );
        assert_eq!(loaded.artifact().identity().timeframe(), timeframe);
    }
}

#[test]
fn broker_calendar_timeframe_still_rejects_duplicate_and_descending_rows() {
    let identity = broker_identity_for(CanonicalTimeframe::D1);
    let mutations: [fn(&mut Vec<TestBar>); 2] = [
        |bars: &mut Vec<TestBar>| bars[1].timestamp_ms = bars[0].timestamp_ms,
        |bars: &mut Vec<TestBar>| bars.swap(0, 1),
    ];
    for mutate in mutations {
        let root = tempfile::tempdir().expect("temporary canonical root");
        let mut bars = sample_calendar_bars(23 * 60 * 60 * 1_000);
        let requested_from_ms = bars[0].timestamp_ms;
        let requested_to_ms = bars[1].timestamp_ms + 24 * 60 * 60 * 1_000;
        mutate(&mut bars);

        let error = publish_test_bars(TestBrokerStreamRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            requested_from_ms,
            requested_to_ms,
            retrieved_unix_ms: 1_800_000_000_000,
            bars: &bars,
        })
        .expect_err("calendar broker rows must remain strictly increasing");
        assert!(
            format!("{error:#}").contains("duplicate or descending"),
            "unexpected error: {error:#}"
        );
        assert!(root.path().read_dir().expect("read root").next().is_none());
    }
}

#[test]
fn broker_writer_rejects_external_identity_before_creating_legacy_output() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = CanonicalDatasetIdentity::external(
        "not-a-broker",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity");
    let bars = sample_bars();
    let error = publish_test_bars(TestBrokerStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        requested_from_ms: bars[0].timestamp_ms,
        requested_to_ms: bars[1].timestamp_ms + 60_000,
        retrieved_unix_ms: 1_800_000_000_000,
        bars: &bars,
    })
    .expect_err("broker writer requires broker-bound identity");
    assert!(format!("{error:#}").contains("broker-bound"));
    assert!(root.path().read_dir().expect("read root").next().is_none());
}

#[test]
fn broker_writer_rejects_rows_outside_the_half_open_request() {
    let identity = broker_identity();
    for bad_timestamp_ms in [
        sample_bars()[1].timestamp_ms,
        sample_bars()[0].timestamp_ms - 60_000,
        sample_bars()[1].timestamp_ms + 60_000,
    ] {
        let root = tempfile::tempdir().expect("temporary canonical root");
        let mut bars = sample_bars();
        bars[1].timestamp_ms = bad_timestamp_ms;
        bars.sort_by_key(|bar| bar.timestamp_ms);
        let error = publish_test_bars(TestBrokerStreamRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            requested_from_ms: sample_bars()[0].timestamp_ms,
            requested_to_ms: sample_bars()[1].timestamp_ms,
            retrieved_unix_ms: 1_800_000_000_000,
            bars: &bars,
        })
        .expect_err("every published row must be inside [from_ms, to_ms)");
        assert!(
            format!("{error:#}").contains("outside the half-open request"),
            "unexpected error: {error:#}"
        );
        assert!(root.path().read_dir().expect("read root").next().is_none());
    }
}

#[test]
fn adjacent_half_open_pages_translate_to_non_overlapping_ctrader_wire_ranges() {
    let shared_boundary_ms = sample_bars()[1].timestamp_ms;
    let newer_logical_to_ms = shared_boundary_ms + 60_000;

    let older_wire_to_ms = ctrader_inclusive_wire_to_ms(shared_boundary_ms)
        .expect("older page inclusive wire upper bound");
    let newer_wire_to_ms = ctrader_inclusive_wire_to_ms(newer_logical_to_ms)
        .expect("newer page inclusive wire upper bound");

    assert_eq!(older_wire_to_ms, shared_boundary_ms - 1);
    assert_eq!(newer_wire_to_ms, newer_logical_to_ms - 1);
    assert!(older_wire_to_ms < shared_boundary_ms);
    assert!(shared_boundary_ms <= newer_wire_to_ms);
    assert!(ctrader_inclusive_wire_to_ms(i64::MIN).is_err());
}

#[test]
fn stale_broker_cas_leaves_no_unreferenced_generation() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = broker_identity();
    let bars = sample_bars();
    let publish = |bars: &[TestBar], expected_generation: Option<&str>| {
        publish_test_bars(TestBrokerStreamRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation,
            requested_from_ms: bars[0].timestamp_ms,
            requested_to_ms: bars[bars.len() - 1].timestamp_ms + 60_000,
            retrieved_unix_ms: 1_800_000_000_000,
            bars,
        })
    };

    let first = publish(&bars, None).expect("first publication");
    let mut updated = bars.clone();
    updated[1].close = 1.21;
    let second = publish(&updated, Some(first.generation())).expect("CAS update");
    let before = generation_count(root.path());

    let mut stale = bars;
    stale[1].close = 1.22;
    let error = publish(&stale, Some(first.generation())).expect_err("stale CAS must conflict");
    assert!(format!("{error:#}").contains("generation conflict"));
    assert_eq!(generation_count(root.path()), before);
    assert_eq!(
        read_current_manifest(root.path(), &identity)
            .expect("current manifest")
            .generation_id(),
        second.generation()
    );
}

#[test]
fn exact_broker_publication_rejects_same_generation_with_rebound_manifest() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = broker_identity();
    let bars = sample_bars();
    let first = publish_test_bars(TestBrokerStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        requested_from_ms: bars[0].timestamp_ms,
        requested_to_ms: bars[1].timestamp_ms + 60_000,
        retrieved_unix_ms: 1_800_000_000_000,
        bars: &bars,
    })
    .expect("first broker publication");
    let selected =
        SelectedDatasetGenerationV1::from_manifest(first.manifest()).expect("exact receipt");

    let rebound = publish_test_bars(TestBrokerStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: Some(first.generation()),
        requested_from_ms: bars[0].timestamp_ms,
        requested_to_ms: bars[1].timestamp_ms + 60_000,
        retrieved_unix_ms: 1_800_000_000_001,
        bars: &bars,
    })
    .expect("same bytes with new broker manifest provenance");
    assert_eq!(rebound.generation(), selected.generation_id());
    assert_ne!(
        rebound.manifest().manifest_binding_sha256(),
        selected.manifest_binding_sha256()
    );

    let chunk = CanonicalOhlcvChunk {
        timestamp_ms: bars.iter().map(|bar| bar.timestamp_ms).collect(),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: CanonicalVolumeChunk::Int64(
            bars.iter()
                .map(|bar| bar.volume.expect("fixture volume"))
                .collect(),
        ),
    };
    let error = publish_broker_trendbar_chunks_exact(
        BrokerTrendbarStreamRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: Some(selected.generation_id()),
            requested_from_ms: bars[0].timestamp_ms,
            requested_to_ms: bars[1].timestamp_ms + 60_000,
            retrieved_unix_ms: 1_800_000_000_002,
            returned_from_ms: bars[0].timestamp_ms,
            returned_to_ms: bars[1].timestamp_ms,
            row_count: bars.len() as u64,
            chunks: vec![Ok(chunk)],
        },
        &selected,
    )
    .expect_err("rebound manifest must fail the final exact broker CAS");
    assert!(
        error
            .downcast_ref::<ExactDatasetGenerationConflict>()
            .is_some(),
        "exact broker CAS must preserve the typed receipt conflict: {error:#}"
    );
}

fn generation_count(root: &std::path::Path) -> usize {
    let identity_root = root.join(broker_identity().to_path_component());
    identity_root
        .read_dir()
        .expect("read dataset root")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("g1-"))
        .count()
}
