use neoethos_data::core::dataset_manifest::{
    CandidateWriteOutcome, DatasetTimestampRange, ProducerProvenanceEnvelopeV1,
    PublicationConflict, PublishMetadataRequest, PublishRequest, collect_unreferenced_generations,
    publish_vortex_generation, publish_vortex_generation_streaming, read_current_manifest,
};
use neoethos_data::{Ohlcv, load_vortex, ohlcv_to_vortex_chunks, write_vortex_chunks};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use vortex_array::ToCanonical;

const BASE_MS: i64 = 1_700_000_040_000;

fn fixture() -> Ohlcv {
    Ohlcv {
        timestamp: Some(vec![
            BASE_MS,
            BASE_MS + 60_000,
            BASE_MS + 120_000,
            BASE_MS + 180_000,
        ]),
        open: vec![1.0, 1.1, 1.2, 1.3],
        high: vec![1.2, 1.3, 1.4, 1.5],
        low: vec![0.9, 1.0, 1.1, 1.2],
        close: vec![1.1, 1.2, 1.3, 1.4],
        volume: Some(vec![0.0, 16_777_217.0, 5.0, 9.0]),
    }
}

#[test]
fn publication_is_immutable_verified_and_compare_and_swap_protected() {
    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let data = fixture();
    let chunks = ohlcv_to_vortex_chunks(&data, 2).expect("two bounded chunks");
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.fixture.v1", b"fixture-one".to_vec())
            .expect("provenance");

    let first = publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(1_700_000_040_000, 1_700_000_220_000)
            .expect("range"),
        provenance: &provenance,
        chunks,
    })
    .expect("first publication");

    let manifest = read_current_manifest(root.path(), &identity).expect("current manifest");
    assert_eq!(manifest.generation_id(), first.generation());
    assert_eq!(manifest.row_count(), 4);
    assert_eq!(manifest.vortex_sha256().len(), 64);
    assert!(manifest.generation_path().exists());
    assert!(
        root.path()
            .join(identity.to_path_component())
            .join("data.vortex.complete")
            .exists()
    );

    let reopened = load_vortex(manifest.generation_path()).expect("verified generation reopens");
    assert_eq!(reopened.close, data.close);
    assert_eq!(reopened.volume, data.volume);

    let stale = publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(1_700_000_040_000, 1_700_000_220_000)
            .expect("range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&data, 2).expect("chunks"),
    })
    .expect_err("a stale base must not replace the pointer");
    let conflict = stale
        .downcast_ref::<PublicationConflict>()
        .expect("generation mismatch must be a typed conflict");
    assert_eq!(conflict.expected_generation(), None);
    assert_eq!(conflict.current_generation(), Some(first.generation()));
    assert_eq!(
        read_current_manifest(root.path(), &identity)
            .expect("unchanged manifest")
            .generation_id(),
        first.generation()
    );
}

#[test]
fn concurrent_publishers_from_one_base_are_linearizable_and_clean_candidates() {
    use std::sync::{Arc, Barrier};

    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "EURUSD.concurrent",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let start = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();

    for value in [10.0_f64, 20.0] {
        let root = root.path().to_path_buf();
        let identity = identity.clone();
        let start = Arc::clone(&start);
        workers.push(std::thread::spawn(move || {
            let mut data = fixture();
            data.open.fill(value);
            data.high.fill(value + 1.0);
            data.low.fill(value - 1.0);
            data.close.fill(value + 0.5);
            let provenance = ProducerProvenanceEnvelopeV1::new(
                "neoethos.concurrent-fixture.v1",
                value.to_bits().to_be_bytes().to_vec(),
            )
            .expect("provenance");
            start.wait();
            publish_vortex_generation(PublishRequest {
                configured_root: &root,
                identity: &identity,
                expected_generation: None,
                timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 180_000)
                    .expect("range"),
                provenance: &provenance,
                chunks: ohlcv_to_vortex_chunks(&data, 2).expect("chunks"),
            })
        }));
    }

    start.wait();
    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("publisher thread"))
        .collect();
    let successes: Vec<_> = results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .collect();
    let conflicts: Vec<_> = results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect();
    assert_eq!(successes.len(), 1, "exactly one CAS publisher must win");
    assert_eq!(
        conflicts.len(),
        1,
        "exactly one CAS publisher must conflict"
    );
    assert!(conflicts[0].downcast_ref::<PublicationConflict>().is_some());

    let current = read_current_manifest(root.path(), &identity).expect("current manifest");
    assert_eq!(current.generation_id(), successes[0].generation());
    assert!(!successes[0].durable_commit_id().is_empty());
    assert_eq!(successes[0].previous_generation(), None);

    let dataset_root = root.path().join(identity.to_path_component());
    for entry in std::fs::read_dir(&dataset_root).expect("dataset directory") {
        let name = entry
            .expect("directory entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        assert!(
            !name.contains("candidate-"),
            "candidate or lease leaked after publication race: {name}"
        );
    }

    collect_unreferenced_generations(root.path(), &identity).expect("collect losing generation");
}

#[test]
fn identical_content_publishers_cannot_unlink_the_winning_generation() {
    use std::sync::{Arc, Barrier};

    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "EURUSD.identical-race",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let candidates_written = Arc::new(Barrier::new(2));
    let candidate_hashes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let provenance = Arc::new(
        ProducerProvenanceEnvelopeV1::new(
            "neoethos.identical-race-fixture.v1",
            b"identical-content".to_vec(),
        )
        .expect("provenance"),
    );

    let workers: Vec<_> = (0..2)
        .map(|_| {
            let root = root.path().to_path_buf();
            let identity = identity.clone();
            let candidates_written = Arc::clone(&candidates_written);
            let candidate_hashes = Arc::clone(&candidate_hashes);
            let provenance = Arc::clone(&provenance);
            std::thread::spawn(move || {
                publish_vortex_generation_streaming(
                    PublishMetadataRequest {
                        configured_root: &root,
                        identity: &identity,
                        expected_generation: None,
                        provenance: &provenance,
                    },
                    move |candidate_path| {
                        let write_stats = write_vortex_chunks(
                            candidate_path,
                            ohlcv_to_vortex_chunks(&fixture(), 2).expect("chunks"),
                        )?;
                        let bytes = std::fs::read(candidate_path)?;
                        candidate_hashes
                            .lock()
                            .expect("candidate hash mutex")
                            .push(format!("{:x}", Sha256::digest(bytes)));
                        candidates_written.wait();
                        Ok(CandidateWriteOutcome {
                            write_stats,
                            timestamp_range: DatasetTimestampRange::new(
                                BASE_MS,
                                BASE_MS + 180_000,
                            )?,
                        })
                    },
                )
            })
        })
        .collect();

    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("publisher thread"))
        .collect();
    let hashes = candidate_hashes.lock().expect("candidate hash mutex");
    assert_eq!(hashes.len(), 2);
    assert_eq!(
        hashes[0], hashes[1],
        "the race fixture must encode byte-identical Vortex candidates"
    );
    assert_eq!(
        results.iter().filter(|result| result.is_ok()).count(),
        1,
        "one identical-content publisher must win the expected-generation CAS"
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| {
                result
                    .as_ref()
                    .err()
                    .and_then(|error| error.downcast_ref::<PublicationConflict>())
                    .is_some()
            })
            .count(),
        1,
        "the other identical-content publisher must return a typed conflict"
    );

    let current = read_current_manifest(root.path(), &identity)
        .expect("the winning manifest must still reference a verified generation");
    assert!(
        current.generation_path().is_file(),
        "a losing publisher must never unlink the generation referenced by the winner"
    );
    let reopened = load_vortex(current.generation_path())
        .expect("the winning identical-content generation must reopen after both publishers exit");
    assert_eq!(reopened.close, fixture().close);
}

#[test]
fn failed_reopen_verification_cleans_candidate_and_lease() {
    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "XAUUSD.failure",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let provenance =
        ProducerProvenanceEnvelopeV1::new("neoethos.failure-fixture.v1", b"wrong-range".to_vec())
            .expect("provenance");
    let error = publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 60_000).expect("range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&fixture(), 2).expect("chunks"),
    })
    .expect_err("wrong declared timestamp range must fail before pointer publication");
    assert!(format!("{error:#}").contains("timestamp range mismatch"));

    let dataset_root = root.path().join(identity.to_path_component());
    assert!(!dataset_root.join("data.vortex.complete").exists());
    for entry in std::fs::read_dir(&dataset_root).expect("dataset directory") {
        let name = entry
            .expect("directory entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        assert!(!name.starts_with("candidate-"), "candidate leaked: {name}");
        assert!(
            !name.contains("candidate-") || !name.ends_with(".lease"),
            "candidate lease leaked: {name}"
        );
    }
}

#[test]
fn streaming_writer_supports_projection_and_row_range() {
    let root = tempdir().expect("temporary root");
    let path = root.path().join("bounded.vortex");
    let data = fixture();
    let chunks = ohlcv_to_vortex_chunks(&data, 2).expect("chunks");
    let stats = neoethos_data::write_vortex_chunks(&path, chunks).expect("streaming write");
    assert_eq!(stats.row_count, 4);
    assert!(stats.max_buffered_bytes <= neoethos_data::MAX_STREAMING_VORTEX_BUFFER_BYTES);

    let projected =
        neoethos_data::read_vortex_projection_range(&path, &["timestamp", "close"], 1..3)
            .expect("projected scan");
    assert_eq!(projected.len(), 2);
    let fields = projected.to_struct();
    assert!(fields.unmasked_field_by_name_opt("open").is_none());
    assert!(fields.unmasked_field_by_name_opt("close").is_some());
}

#[cfg(windows)]
#[test]
fn windows_atomic_publication_supports_verbatim_paths_beyond_max_path() {
    let root = tempdir().expect("temporary root");
    let configured_root = root.path().join("r".repeat(120));
    let identity = CanonicalDatasetIdentity::external(
        "windows-long-path-publication",
        "EURUSD.broker-suffix-with-lossless-identity",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.windows-long-path.v1",
        b"verbatim-path".to_vec(),
    )
    .expect("provenance");
    let expected_generation_path_length = configured_root
        .join(identity.to_path_component())
        .join(format!("g1-{}.vortex", "0".repeat(64)))
        .as_os_str()
        .len();
    assert!(
        expected_generation_path_length > 260,
        "fixture must exercise a path beyond legacy MAX_PATH"
    );

    let published = publish_vortex_generation(PublishRequest {
        configured_root: &configured_root,
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 180_000).expect("range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&fixture(), 2).expect("chunks"),
    })
    .expect("publish through a verbatim Windows path");
    let manifest = read_current_manifest(&configured_root, &identity).expect("manifest");
    assert_eq!(manifest.generation_id(), published.generation());
    assert_eq!(
        load_vortex(manifest.generation_path())
            .expect("reopen long-path generation")
            .close,
        fixture().close
    );
}
