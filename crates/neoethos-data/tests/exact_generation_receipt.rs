use neoethos_data::core::dataset_manifest::{
    CandidateWriteOutcome, DatasetTimestampRange, ExactDatasetGenerationConflict,
    ProducerProvenanceEnvelopeV1, PublishMetadataRequest, PublishRequest,
    collect_unreferenced_generations, publish_vortex_generation,
    publish_vortex_generation_streaming_exact,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity,
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, Ohlcv, SelectedDatasetGenerationV1,
    load_exact_canonical_timeframe, ohlcv_to_vortex_chunks, write_vortex_chunks,
};

const BASE_MS: i64 = 1_700_000_040_000;

fn external_identity(namespace: &str, timeframe: CanonicalTimeframe) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        namespace,
        "EURUSD",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical external identity")
}

fn fixture(value: f64) -> Ohlcv {
    Ohlcv {
        timestamp: Some(vec![BASE_MS, BASE_MS + 60_000]),
        open: vec![value, value + 0.1],
        high: vec![value + 0.2, value + 0.3],
        low: vec![value - 0.2, value - 0.1],
        close: vec![value + 0.05, value + 0.15],
        volume: Some(vec![0.0, 10.0]),
    }
}

fn publish(
    root: &std::path::Path,
    identity: &CanonicalDatasetIdentity,
    expected_generation: Option<&str>,
    value: f64,
    provenance_label: &str,
) -> neoethos_data::core::dataset_manifest::PublishResult {
    let data = fixture(value);
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.exact-generation-test.v1",
        provenance_label.as_bytes().to_vec(),
    )
    .expect("producer provenance");
    publish_vortex_generation(PublishRequest {
        configured_root: root,
        identity,
        expected_generation,
        timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 60_000)
            .expect("timestamp range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&data, 1).expect("Vortex chunks"),
    })
    .expect("publish generation")
}

fn fake_receipt(
    identity: CanonicalDatasetIdentity,
    discriminator: u64,
) -> SelectedDatasetGenerationV1 {
    SelectedDatasetGenerationV1::new(
        identity,
        format!("g1-{discriminator:064x}.vortex"),
        format!("{:064x}", discriminator + 10_000),
    )
    .expect("valid synthetic receipt")
}

#[test]
fn selected_receipt_rejects_malformed_generation_binding_and_json() {
    let identity = external_identity("strict-receipt", CanonicalTimeframe::M1);
    assert!(
        SelectedDatasetGenerationV1::new(identity.clone(), "../escape.vortex", "0".repeat(64))
            .is_err()
    );
    assert!(
        SelectedDatasetGenerationV1::new(
            identity.clone(),
            format!("g1-{}.vortex", "0".repeat(64)),
            "A".repeat(64),
        )
        .is_err()
    );

    let receipt = fake_receipt(identity, 1);
    let encoded = serde_json::to_vec(&receipt).expect("serialize strict receipt");
    let decoded: SelectedDatasetGenerationV1 =
        serde_json::from_slice(&encoded).expect("deserialize strict receipt");
    assert_eq!(decoded, receipt);

    let mut wire: serde_json::Value = serde_json::from_slice(&encoded).expect("receipt JSON");
    wire.as_object_mut()
        .expect("receipt object")
        .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
    assert!(
        serde_json::from_value::<SelectedDatasetGenerationV1>(wire).is_err(),
        "unknown fields must not enter the exact receipt contract"
    );
}

#[test]
fn stale_generation_before_pin_fails_with_typed_conflict() {
    let root = tempfile::tempdir().expect("canonical root");
    let identity = external_identity("stale-before-pin", CanonicalTimeframe::M1);
    let generation_n = publish(root.path(), &identity, None, 1.0, "generation-n");
    let selected =
        SelectedDatasetGenerationV1::from_manifest(generation_n.manifest()).expect("receipt N");
    publish(
        root.path(),
        &identity,
        Some(generation_n.generation()),
        2.0,
        "generation-n-plus-one",
    );

    let error = load_exact_canonical_timeframe(root.path(), &selected)
        .expect_err("a stale selected generation must not fall forward to current");
    let conflict = error
        .downcast_ref::<ExactDatasetGenerationConflict>()
        .expect("generation mismatch must be a typed conflict");
    assert_eq!(conflict.expected_generation(), selected.generation_id());
    assert_ne!(
        conflict.current_generation(),
        Some(selected.generation_id())
    );
}

#[test]
fn same_generation_with_wrong_manifest_binding_fails_closed() {
    let root = tempfile::tempdir().expect("canonical root");
    let identity = external_identity("binding-mismatch", CanonicalTimeframe::M1);
    let first = publish(root.path(), &identity, None, 1.0, "first-manifest");
    let selected =
        SelectedDatasetGenerationV1::from_manifest(first.manifest()).expect("first receipt");

    let rebound = publish(
        root.path(),
        &identity,
        Some(first.generation()),
        1.0,
        "same-bytes-different-manifest",
    );
    assert_eq!(rebound.generation(), selected.generation_id());
    assert_ne!(
        rebound.manifest().manifest_binding_sha256(),
        selected.manifest_binding_sha256()
    );

    let error = load_exact_canonical_timeframe(root.path(), &selected)
        .expect_err("same generation id with a different manifest binding is stale");
    let conflict = error
        .downcast_ref::<ExactDatasetGenerationConflict>()
        .expect("binding mismatch must be a typed conflict");
    assert_eq!(
        conflict.current_generation(),
        Some(selected.generation_id())
    );
    assert_ne!(
        conflict.current_manifest_binding_sha256(),
        Some(selected.manifest_binding_sha256())
    );
}

#[test]
fn exact_publication_rechecks_manifest_binding_after_candidate_write_under_the_cas_lock() {
    use std::sync::mpsc;

    let root = tempfile::tempdir().expect("canonical root");
    let identity = external_identity("binding-race", CanonicalTimeframe::M1);
    let first = publish(root.path(), &identity, None, 1.0, "selected-manifest");
    let selected =
        SelectedDatasetGenerationV1::from_manifest(first.manifest()).expect("selected receipt");

    let (candidate_ready_tx, candidate_ready_rx) = mpsc::sync_channel(0);
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let publisher_root = root.path().to_path_buf();
    let publisher_identity = identity.clone();
    let publisher_selected = selected.clone();
    let exact_publisher = std::thread::spawn(move || {
        let provenance = ProducerProvenanceEnvelopeV1::new(
            "neoethos.exact-publication-race.v1",
            b"losing-exact-publication".to_vec(),
        )
        .expect("exact publisher provenance");
        publish_vortex_generation_streaming_exact(
            PublishMetadataRequest {
                configured_root: &publisher_root,
                identity: &publisher_identity,
                expected_generation: Some(publisher_selected.generation_id()),
                provenance: &provenance,
            },
            &publisher_selected,
            move |candidate_path| {
                let write_stats =
                    write_vortex_chunks(candidate_path, ohlcv_to_vortex_chunks(&fixture(3.0), 1)?)?;
                candidate_ready_tx
                    .send(())
                    .expect("signal completed candidate write");
                release_rx
                    .recv()
                    .expect("release exact publisher after pointer rebound");
                Ok(CandidateWriteOutcome {
                    write_stats,
                    timestamp_range: DatasetTimestampRange::new(BASE_MS, BASE_MS + 60_000)?,
                })
            },
        )
    });

    candidate_ready_rx
        .recv()
        .expect("exact candidate must finish before the competing publication");
    let rebound = publish(
        root.path(),
        &identity,
        Some(first.generation()),
        1.0,
        "same-generation-different-binding",
    );
    assert_eq!(rebound.generation(), selected.generation_id());
    assert_ne!(
        rebound.manifest().manifest_binding_sha256(),
        selected.manifest_binding_sha256()
    );
    release_tx
        .send(())
        .expect("release exact publisher after competing publication");

    let error = exact_publisher
        .join()
        .expect("exact publication worker must not panic")
        .expect_err("same generation with a rebound manifest must conflict");
    let conflict = error
        .downcast_ref::<ExactDatasetGenerationConflict>()
        .expect("binding race must remain a typed exact-generation conflict");
    assert_eq!(conflict.expected_generation(), selected.generation_id());
    assert_eq!(
        conflict.expected_manifest_binding_sha256(),
        selected.manifest_binding_sha256()
    );
    assert_eq!(
        conflict.current_generation(),
        Some(selected.generation_id())
    );
    assert_eq!(
        conflict.current_manifest_binding_sha256(),
        Some(rebound.manifest().manifest_binding_sha256())
    );
}

#[test]
fn pointer_advance_after_pin_keeps_the_original_generation_usable() {
    let root = tempfile::tempdir().expect("canonical root");
    let identity = external_identity("advance-after-pin", CanonicalTimeframe::M1);
    let generation_n = publish(root.path(), &identity, None, 1.0, "generation-n");
    let selected =
        SelectedDatasetGenerationV1::from_manifest(generation_n.manifest()).expect("receipt N");
    let pinned =
        load_exact_canonical_timeframe(root.path(), &selected).expect("pin exact generation N");
    assert_eq!(pinned.ohlcv().close[0], 1.05);

    let generation_n1 = publish(
        root.path(),
        &identity,
        Some(generation_n.generation()),
        2.0,
        "generation-n-plus-one",
    );
    publish(
        root.path(),
        &identity,
        Some(generation_n1.generation()),
        3.0,
        "generation-n-plus-two",
    );
    collect_unreferenced_generations(root.path(), &identity).expect("GC with exact reader pin");

    assert_eq!(pinned.artifact().generation_id(), selected.generation_id());
    assert_eq!(pinned.ohlcv().close[0], 1.05);
    pinned
        .artifact()
        .lease()
        .reopen_verified()
        .expect("original immutable bytes remain reopenable while pinned");
}

#[test]
fn series_receipt_sorts_all_direct_timeframes_and_rejects_duplicates() {
    let anchor_identity = external_identity("all-direct", CanonicalTimeframe::M1);
    let anchor = fake_receipt(anchor_identity, 1);
    let mut direct = CanonicalTimeframe::ALL
        .into_iter()
        .enumerate()
        .map(|(index, timeframe)| {
            fake_receipt(
                external_identity("all-direct", timeframe),
                u64::try_from(index + 1).expect("small timeframe index"),
            )
        })
        .collect::<Vec<_>>();
    direct.reverse();

    let series = CanonicalDatasetSeriesReceiptV1::new(anchor.clone(), direct)
        .expect("all 14 independently selected direct timeframe receipts");
    assert_eq!(series.anchor(), &anchor);
    assert_eq!(
        series.direct_timeframes().len(),
        CanonicalTimeframe::ALL.len()
    );
    assert_eq!(
        series
            .direct_timeframes()
            .iter()
            .map(|receipt| receipt.identity().timeframe())
            .collect::<Vec<_>>(),
        CanonicalTimeframe::ALL
    );

    let duplicate_m1 = vec![anchor.clone(), anchor.clone()];
    assert!(CanonicalDatasetSeriesReceiptV1::new(anchor, duplicate_m1).is_err());
}

#[test]
fn series_receipt_rejects_a_direct_timeframe_from_another_ctrader_account() {
    let selected_identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Live,
        "broker-live",
        1001,
        42,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("selected cTrader identity");
    let foreign_h1_identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Live,
        "broker-live",
        2002,
        42,
        "EURUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("foreign account identity");
    let anchor = fake_receipt(selected_identity, 1);
    let foreign_h1 = fake_receipt(foreign_h1_identity, 2);

    let error = CanonicalDatasetSeriesReceiptV1::new(anchor.clone(), vec![anchor, foreign_h1])
        .expect_err("a higher timeframe from another account must not fill this direct series");
    assert!(
        error
            .to_string()
            .contains("different source/account series")
    );
}
