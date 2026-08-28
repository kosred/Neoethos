use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::dataset_manifest::canonical_dataset_root;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalOhlcvReverseSpool, CanonicalOhlcvStreamPublishRequest, CanonicalTimeframe,
    CanonicalVolumeChunk, publish_canonical_ohlcv_stream, read_vortex_i64_projection_range,
};

fn identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "stream-test",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity")
}

#[test]
fn reverse_vortex_spool_yields_oldest_first_and_cleans_its_files() {
    let root = tempfile::tempdir().expect("temporary spool root");
    let older_ms = 1_700_000_040_000;
    let newer_ms = older_ms + 120_000;
    let mut spool =
        CanonicalOhlcvReverseSpool::create(root.path(), 2, 4).expect("create bounded spool");
    let spool_path = spool.path().to_owned();

    spool
        .push_latest(chunk(vec![newer_ms, newer_ms + 60_000], vec![30, 40]))
        .expect("spool newer broker page");
    spool
        .push_latest(chunk(vec![older_ms, older_ms + 60_000], vec![10, 20]))
        .expect("spool older broker page");

    let chunks = spool
        .into_oldest_first()
        .collect::<anyhow::Result<Vec<_>>>()
        .expect("read spooled pages oldest first");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].timestamp_ms, [older_ms, older_ms + 60_000]);
    assert_eq!(chunks[1].timestamp_ms, [newer_ms, newer_ms + 60_000]);
    assert!(matches!(
        &chunks[0].volume,
        CanonicalVolumeChunk::Int64(values) if values == &[10, 20]
    ));
    assert!(
        !spool_path.exists(),
        "successful consumption must remove the temporary Vortex spool"
    );
}

#[test]
fn stream_error_removes_only_a_new_empty_identity_root() {
    for pre_existing in [false, true] {
        let root = tempfile::tempdir().expect("temporary canonical root");
        let identity = identity();
        let dataset_root =
            canonical_dataset_root(root.path(), &identity).expect("resolve exact dataset root");
        let marker = dataset_root.join("operator-owned.marker");
        if pre_existing {
            std::fs::create_dir(&dataset_root).expect("create pre-existing dataset root");
            std::fs::write(&marker, b"keep").expect("create pre-existing marker");
        }
        let provenance =
            ProducerProvenanceEnvelopeV1::new("neoethos.streaming-test.v1", b"cleanup".to_vec())
                .expect("valid provenance");
        let first_ms = 1_700_000_040_000;
        let error = publish_canonical_ohlcv_stream(CanonicalOhlcvStreamPublishRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            provenance: &provenance,
            requested_from_ms: first_ms,
            requested_to_ms: first_ms + 240_000,
            expected_first_timestamp_ms: first_ms,
            expected_last_timestamp_ms: first_ms + 180_000,
            expected_row_count: 4,
            max_chunk_rows: 2,
            chunks: vec![
                Ok(chunk(vec![first_ms, first_ms + 60_000], vec![1, 2])),
                Ok(chunk(
                    vec![first_ms + 60_000, first_ms + 180_000],
                    vec![3, 4],
                )),
            ],
        })
        .expect_err("cross-chunk overlap must fail closed");
        assert!(
            format!("{error:#}").contains("overlap or descend"),
            "unexpected error: {error:#}"
        );

        if pre_existing {
            assert_eq!(std::fs::read(&marker).expect("marker survives"), b"keep");
            let entries = dataset_root
                .read_dir()
                .expect("read preserved root")
                .map(|entry| entry.expect("dataset entry").file_name())
                .collect::<Vec<_>>();
            assert_eq!(entries, [marker.file_name().expect("marker name")]);
        } else {
            assert!(
                !dataset_root.exists(),
                "a root created only for a failed candidate must be removed"
            );
        }
    }
}

#[test]
fn reverse_spool_enforces_hard_page_bounds_and_drop_cleanup() {
    let root = tempfile::tempdir().expect("temporary spool root");
    let first_ms = 1_700_000_040_000;

    let mut row_bounded =
        CanonicalOhlcvReverseSpool::create(root.path(), 1, 2).expect("row-bounded spool");
    let row_spool_path = row_bounded.path().to_owned();
    let error = row_bounded
        .push_latest(chunk(vec![first_ms, first_ms + 60_000], vec![1, 2]))
        .expect_err("oversized page must fail before a spool file is written");
    assert!(format!("{error:#}").contains("hard") || format!("{error:#}").contains("required"));
    drop(row_bounded);
    assert!(!row_spool_path.exists());

    let mut page_bounded =
        CanonicalOhlcvReverseSpool::create(root.path(), 1, 1).expect("page-bounded spool");
    let page_spool_path = page_bounded.path().to_owned();
    page_bounded
        .push_latest(chunk(vec![first_ms + 60_000], vec![2]))
        .expect("first bounded page");
    let error = page_bounded
        .push_latest(chunk(vec![first_ms], vec![1]))
        .expect_err("second page must exceed the hard page-count bound");
    assert!(format!("{error:#}").contains("hard page limit"));
    drop(page_bounded);
    assert!(!page_spool_path.exists());
}

#[test]
fn stream_rejects_a_chunk_above_its_hard_resident_row_limit() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = identity();
    let dataset_root =
        canonical_dataset_root(root.path(), &identity).expect("resolve exact dataset root");
    let first_ms = 1_700_000_040_000;
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.streaming-test.v1", b"chunk-limit".to_vec())
            .expect("valid provenance");
    let error = publish_canonical_ohlcv_stream(CanonicalOhlcvStreamPublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        requested_from_ms: first_ms,
        requested_to_ms: first_ms + 120_000,
        expected_first_timestamp_ms: first_ms,
        expected_last_timestamp_ms: first_ms + 60_000,
        expected_row_count: 2,
        max_chunk_rows: 1,
        chunks: vec![Ok(chunk(vec![first_ms, first_ms + 60_000], vec![1, 2]))],
    })
    .expect_err("oversized resident chunk must fail before publication");
    assert!(format!("{error:#}").contains("above the hard limit"));
    assert!(!dataset_root.exists());
}

fn chunk(timestamps: Vec<i64>, volumes: Vec<i64>) -> CanonicalOhlcvChunk {
    let len = timestamps.len();
    CanonicalOhlcvChunk {
        timestamp_ms: timestamps,
        open: vec![1.10; len],
        high: vec![1.20; len],
        low: vec![1.00; len],
        close: vec![1.15; len],
        volume: CanonicalVolumeChunk::Int64(volumes),
    }
}

#[test]
fn owned_chunks_publish_one_generation_with_exact_int64_volume() {
    let root = tempfile::tempdir().expect("temporary canonical root");
    let identity = identity();
    let first_ms = 1_700_000_040_000;
    let last_ms = first_ms + 180_000;
    let exact_volumes = [9_007_199_254_740_993_i64, 4, 5, 9_007_199_254_741_111];
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.streaming-test.v1", b"owned-chunks".to_vec())
            .expect("valid provenance");

    let publication = publish_canonical_ohlcv_stream(CanonicalOhlcvStreamPublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        requested_from_ms: first_ms,
        requested_to_ms: last_ms + 60_000,
        expected_first_timestamp_ms: first_ms,
        expected_last_timestamp_ms: last_ms,
        expected_row_count: 4,
        max_chunk_rows: 2,
        chunks: vec![
            Ok(chunk(
                vec![first_ms, first_ms + 60_000],
                exact_volumes[..2].to_vec(),
            )),
            Ok(chunk(
                vec![first_ms + 120_000, last_ms],
                exact_volumes[2..].to_vec(),
            )),
        ],
    })
    .expect("publish canonical owned chunks");

    assert_eq!(publication.row_count(), 4);
    assert_eq!(
        read_vortex_i64_projection_range(publication.manifest().generation_path(), "volume", 0..4,)
            .expect("read exact physical i64 volume"),
        exact_volumes
    );
}
