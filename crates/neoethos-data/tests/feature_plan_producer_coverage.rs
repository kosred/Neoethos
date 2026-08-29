use std::collections::HashSet;

use neoethos_data::core::dataset_manifest::{
    DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest, publish_vortex_generation,
};
use neoethos_data::core::feature_registry::{
    FeatureSource, PRODUCTION_FEATURE_PRODUCER_ORDER, ProductionFeatureProducerId,
    feature_column_metadata, production_feature_producer_manifest_v1,
};
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe, FeatureProfile, Ohlcv,
    compute_hpc_feature_frame_sized, load_canonical_timeframe, ohlcv_to_vortex_chunks,
};

fn production_fixture(n: usize) -> Ohlcv {
    let mut open = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut close = Vec::with_capacity(n);
    let mut volume = Vec::with_capacity(n);
    let mut timestamp = Vec::with_capacity(n);
    let mut price = 1.1_f64;

    for i in 0..n {
        let phase = i as f64 * 0.071;
        let next = price + phase.sin() * 0.0002 + phase.cos() * 0.00007;
        open.push(price);
        high.push(price.max(next) + 0.0003);
        low.push(price.min(next) - 0.0003);
        close.push(next);
        volume.push(100.0 + (i % 37) as f64 * 11.0);
        timestamp.push(1_577_836_800_000 + i as i64 * 300_000);
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
fn production_call_graph_and_manifest_are_a_bijection() {
    let manifest = production_feature_producer_manifest_v1().expect("embedded producer manifest");
    let expected = [
        ProductionFeatureProducerId::SmartMoneyConcept,
        ProductionFeatureProducerId::ClassicVectorTa,
        ProductionFeatureProducerId::Quantitative,
        ProductionFeatureProducerId::Session,
        ProductionFeatureProducerId::Regime,
        ProductionFeatureProducerId::Footprint,
    ];

    assert_eq!(
        PRODUCTION_FEATURE_PRODUCER_ORDER, expected,
        "the typed order must match the production FeatureFrame order"
    );
    assert_eq!(
        manifest.len(),
        PRODUCTION_FEATURE_PRODUCER_ORDER.len(),
        "every reachable production producer needs exactly one manifest row"
    );

    let mut producers = HashSet::new();
    for row in manifest {
        assert!(
            producers.insert(row.producer()),
            "duplicate manifest row for {:?}",
            row.producer()
        );
        assert!(
            PRODUCTION_FEATURE_PRODUCER_ORDER.contains(&row.producer()),
            "manifest row {:?} is not reachable from the production plan",
            row.producer()
        );
        assert!(
            !row.semantic_sources().entries().is_empty(),
            "{:?} has no semantic source closure",
            row.producer()
        );
        assert!(
            row.semantic_sources().entries().iter().all(|entry| {
                entry.path().starts_with("crates/") || entry.path().starts_with("vendor/")
            }),
            "{:?} contains a non-canonical repository path: {:?}",
            row.producer(),
            row.semantic_sources()
                .entries()
                .iter()
                .map(|entry| entry.path())
                .collect::<Vec<_>>()
        );
        assert!(
            row.semantic_sources()
                .entries()
                .iter()
                .all(|entry| !entry.path().contains('\\') && !entry.path().contains("..")),
            "{:?} source closure must use contained slash-separated paths",
            row.producer()
        );
        assert!(
            row.semantic_version() > 0,
            "{:?} needs an explicit non-zero semantic version",
            row.producer()
        );
    }

    assert_eq!(
        producers,
        expected.into_iter().collect(),
        "reachable and declared producer sets differ"
    );
}

#[test]
fn every_production_family_is_visible_through_column_metadata() {
    let representatives = [
        ("smc_ob", FeatureSource::SmartMoneyConcept),
        ("rsi", FeatureSource::ClassicTechnicalAnalysis),
        ("quant_log_return", FeatureSource::Quantitative),
        ("session_london_open_dist", FeatureSource::Session),
        (
            "neoethos_custom_gk_vol_ratio_state_10_50_v3",
            FeatureSource::Regime,
        ),
        ("fp_effort_result_div", FeatureSource::Footprint),
    ];

    for (name, expected_source) in representatives {
        let metadata = feature_column_metadata(name)
            .unwrap_or_else(|| panic!("production column `{name}` is absent from the registry"));
        assert_eq!(
            metadata.source, expected_source,
            "production column `{name}` is assigned to the wrong producer"
        );
    }
}

#[test]
fn known_value_affecting_dependencies_are_declared_by_their_producers() {
    let manifest = production_feature_producer_manifest_v1().expect("embedded producer manifest");
    let dependencies_for = |producer| {
        manifest
            .iter()
            .find(|row| row.producer() == producer)
            .unwrap_or_else(|| panic!("missing manifest row for {producer:?}"))
            .relevant_dependencies()
            .entries()
    };

    assert!(
        dependencies_for(ProductionFeatureProducerId::ClassicVectorTa)
            .iter()
            .any(|dependency| dependency.package_name() == "vector-ta"),
        "classic/vector-ta values are not bound to the vector-ta dependency"
    );
    assert!(
        dependencies_for(ProductionFeatureProducerId::SmartMoneyConcept)
            .iter()
            .any(|dependency| dependency.package_name() == "chrono"),
        "SMC calendar semantics are not bound to chrono"
    );
    assert!(
        dependencies_for(ProductionFeatureProducerId::Session)
            .iter()
            .any(|dependency| dependency.package_name() == "chrono"),
        "session calendar semantics are not bound to chrono"
    );
}

#[test]
fn assembled_production_frame_exposes_every_reachable_family_to_the_registry() {
    let bars = production_fixture(512);
    let root = tempfile::tempdir().expect("canonical fixture root");
    let identity = CanonicalDatasetIdentity::external(
        "feature-producer-coverage",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("dataset identity");
    let timestamps = bars.timestamp.as_ref().expect("fixture timestamps");
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.feature-producer-coverage.v1",
        b"deterministic-adversarial-fixture".to_vec(),
    )
    .expect("producer provenance");
    publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(timestamps[0], timestamps[511])
            .expect("timestamp range"),
        provenance: &provenance,
        chunks: ohlcv_to_vortex_chunks(&bars, 128).expect("bounded Vortex chunks"),
    })
    .expect("publish fixture");
    let source = load_canonical_timeframe(root.path(), &identity).expect("pinned source");
    let frame = compute_hpc_feature_frame_sized(&source, FeatureProfile::HPC, bars.len())
        .expect("production feature assembly");
    let metadata = frame
        .column_metadata()
        .expect("every emitted production column must be registered");
    let observed: HashSet<_> = metadata.iter().map(|column| column.source).collect();
    let expected: HashSet<_> = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .map(|row| row.source())
        .collect();

    assert_eq!(
        observed, expected,
        "the production assembler and producer manifest expose different families"
    );
    assert!(
        frame
            .names
            .iter()
            .any(|name| name == "fp_effort_result_div"),
        "Footprint must be emitted by the real production assembler"
    );
}

#[test]
fn quantitative_validity_contract_has_its_own_semantic_version() {
    let row = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::Quantitative)
        .expect("quantitative manifest row");
    assert_eq!(
        row.semantic_version(),
        2,
        "removing numeric warmup/denominator/session sentinels changes quantitative semantics"
    );
}

#[test]
fn smc_validity_contract_has_its_own_semantic_version() {
    let row = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::SmartMoneyConcept)
        .expect("SMC manifest row");
    assert_eq!(
        row.semantic_version(),
        3,
        "semantic-v3 binds fixed-order CPU/CUDA FVG-age log1p plus calendar, warmup, denominator, and FVG-magnet validity; semantic-v2 artifacts must be regenerated"
    );
}

#[test]
fn classic_vector_ta_validity_contract_has_its_own_semantic_version() {
    let row = production_feature_producer_manifest_v1()
        .expect("embedded producer manifest")
        .iter()
        .find(|row| row.producer() == ProductionFeatureProducerId::ClassicVectorTa)
        .expect("classic/vector-ta manifest row");
    assert_eq!(
        row.semantic_version(),
        9,
        "Classic semantic-v9 preserves the EVWMA-v8 identities and adds creator-aligned CCI Cycle; semantic-v8, older and unversioned Classic artifacts must fail closed and be regenerated"
    );
}
