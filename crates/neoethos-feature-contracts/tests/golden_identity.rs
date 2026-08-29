use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
    FeatureParameterV1, FeaturePlanV1, SourceArtifactBindingV1, SourceSegmentV1,
};

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn dataset(symbol: &str) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "golden-fixture",
        symbol,
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid dataset identity")
}

fn plan(dataset: CanonicalDatasetIdentity, formula_version: u32) -> FeaturePlanV1 {
    let source = FeatureNodeV1::source(
        "source:eurusd:m5",
        dataset,
        "neoethos.ohlcv.f64-ms.v1",
        1,
        vec![FeatureOutputV1::f64("close", 1).expect("source output")],
        hash(1),
    )
    .expect("source node");
    let indicator = FeatureNodeV1::transform(
        "indicator:rsi",
        FeatureOperationTagV1::Indicator,
        formula_version,
        vec!["source:eurusd:m5".to_owned()],
        vec![FeatureOutputV1::f64("rsi_14", 2).expect("indicator output")],
        vec![FeatureParameterV1::f64("period", 14.0).expect("period")],
        hash(2),
        hash(3),
        None,
    )
    .expect("indicator node");
    FeaturePlanV1::new(vec![indicator, source], vec!["rsi_14".to_owned()]).expect("canonical plan")
}

fn binding(
    node: &str,
    dataset: CanonicalDatasetIdentity,
    generation: &str,
    byte: u8,
) -> SourceArtifactBindingV1 {
    SourceArtifactBindingV1::new(
        node,
        dataset,
        "neoethos.dataset-manifest.v1",
        hash(byte),
        generation,
        hash(byte.wrapping_add(1)),
        BarTimestampConvention::BarOpen,
        vec![SourceSegmentV1::new(0, 100, 1_704_067_200_000, 1_704_096_900_000).expect("segment")],
    )
    .expect("binding")
}

#[test]
fn semantic_change_changes_plan_but_generation_change_only_changes_provenance() {
    let identity = dataset("EURUSD");
    let plan_v1 = plan(identity.clone(), 1);
    let same_plan = plan(identity.clone(), 1);
    let formula_repair = plan(identity.clone(), 2);

    assert_eq!(
        plan_v1.identity().to_hex(),
        "71a748db07f30f43398d548ba032c36b48033f1038adaa45359de7f91a5071af"
    );
    assert_eq!(plan_v1.identity(), same_plan.identity());
    assert_ne!(plan_v1.identity(), formula_repair.identity());
    let reopened_plan = FeaturePlanV1::from_canonical_bytes(plan_v1.canonical_bytes())
        .expect("reopen canonical feature plan");
    assert_eq!(reopened_plan, plan_v1);
    let mut tampered_plan = plan_v1.canonical_bytes().to_vec();
    *tampered_plan.last_mut().expect("non-empty plan") ^= 1;
    assert!(FeaturePlanV1::from_canonical_bytes(&tampered_plan).is_err());

    let first = DatasetFeatureArtifactProvenanceV1::new(
        &plan_v1,
        vec![binding(
            "source:eurusd:m5",
            identity.clone(),
            "generation-a",
            11,
        )],
    )
    .expect("first provenance");
    let second = DatasetFeatureArtifactProvenanceV1::new(
        &plan_v1,
        vec![binding("source:eurusd:m5", identity, "generation-b", 12)],
    )
    .expect("second provenance");
    assert_eq!(
        first.identity().to_hex(),
        "efb6f022a7e908944de5abb409bc4950341e88045b0b3d1a04112525cbc5cb4e"
    );
    let reopened_provenance =
        DatasetFeatureArtifactProvenanceV1::from_canonical_bytes(&plan_v1, first.canonical_bytes())
            .expect("reopen canonical provenance");
    assert_eq!(reopened_provenance, first);
    let mut trailing_provenance = first.canonical_bytes().to_vec();
    trailing_provenance.push(0);
    assert!(
        DatasetFeatureArtifactProvenanceV1::from_canonical_bytes(&plan_v1, &trailing_provenance,)
            .is_err()
    );
    assert_ne!(first.identity(), second.identity());
}

#[test]
fn topology_is_canonical_and_invalid_graphs_or_parameters_fail_closed() {
    let identity = dataset("EURUSD");
    let canonical = plan(identity.clone(), 1);
    let source = canonical.nodes()[0].clone();
    let indicator = canonical.nodes()[1].clone();
    let reordered = FeaturePlanV1::new(
        vec![indicator.clone(), source.clone()],
        vec!["rsi_14".to_owned()],
    )
    .expect("input ordering canonicalizes");
    assert_eq!(canonical.identity(), reordered.identity());
    assert_eq!(canonical.canonical_bytes(), reordered.canonical_bytes());

    let missing = FeatureNodeV1::transform(
        "indicator:missing",
        FeatureOperationTagV1::Indicator,
        1,
        vec!["source:absent".to_owned()],
        vec![FeatureOutputV1::f64("missing", 1).expect("output")],
        Vec::new(),
        hash(1),
        hash(2),
        None,
    )
    .expect("node shape");
    assert!(FeaturePlanV1::new(vec![missing], vec!["missing".to_owned()]).is_err());

    let cycle_a = FeatureNodeV1::transform(
        "a",
        FeatureOperationTagV1::Derived,
        1,
        vec!["b".to_owned()],
        vec![FeatureOutputV1::f64("a", 1).expect("output")],
        Vec::new(),
        hash(1),
        hash(2),
        None,
    )
    .expect("cycle a");
    let cycle_b = FeatureNodeV1::transform(
        "b",
        FeatureOperationTagV1::Derived,
        1,
        vec!["a".to_owned()],
        vec![FeatureOutputV1::f64("b", 1).expect("output")],
        Vec::new(),
        hash(1),
        hash(2),
        None,
    )
    .expect("cycle b");
    assert!(
        FeaturePlanV1::new(vec![cycle_a, cycle_b], vec!["a".to_owned(), "b".to_owned()]).is_err()
    );

    assert!(FeatureParameterV1::f64("bad", f64::NAN).is_err());
    assert!(FeatureParameterV1::f64("bad", -0.0).is_err());
}

#[test]
fn provenance_reordering_is_stable_but_swaps_and_overlaps_are_rejected() {
    let eurusd = dataset("EURUSD");
    let gbpusd = dataset("GBPUSD");
    let source_a = FeatureNodeV1::source(
        "source:a",
        eurusd.clone(),
        "neoethos.ohlcv.f64-ms.v1",
        1,
        vec![FeatureOutputV1::f64("a_close", 1).expect("output")],
        hash(1),
    )
    .expect("source a");
    let source_b = FeatureNodeV1::source(
        "source:b",
        gbpusd.clone(),
        "neoethos.ohlcv.f64-ms.v1",
        1,
        vec![FeatureOutputV1::f64("b_close", 1).expect("output")],
        hash(2),
    )
    .expect("source b");
    let plan = FeaturePlanV1::new(
        vec![source_b.clone(), source_a.clone()],
        vec!["a_close".to_owned(), "b_close".to_owned()],
    )
    .expect("two-source plan");
    let reversed_outputs = FeaturePlanV1::new(
        vec![source_a, source_b],
        vec!["b_close".to_owned(), "a_close".to_owned()],
    )
    .expect("reversed output plan");
    assert_ne!(plan.identity(), reversed_outputs.identity());
    let a = binding("source:a", eurusd.clone(), "a", 10);
    let b = binding("source:b", gbpusd.clone(), "b", 20);
    let first = DatasetFeatureArtifactProvenanceV1::new(&plan, vec![a.clone(), b.clone()])
        .expect("ordered provenance");
    let reordered =
        DatasetFeatureArtifactProvenanceV1::new(&plan, vec![b, a]).expect("reordered provenance");
    assert_eq!(first.identity(), reordered.identity());

    let swapped_a = binding("source:a", gbpusd, "a", 10);
    let swapped_b = binding("source:b", eurusd, "b", 20);
    assert!(DatasetFeatureArtifactProvenanceV1::new(&plan, vec![swapped_a, swapped_b]).is_err());

    let overlapping = SourceArtifactBindingV1::new(
        "source:a",
        dataset("EURUSD"),
        "neoethos.dataset-manifest.v1",
        hash(30),
        "overlap",
        hash(31),
        BarTimestampConvention::BarOpen,
        vec![
            SourceSegmentV1::new(0, 100, 0, 99).expect("segment one"),
            SourceSegmentV1::new(99, 120, 100, 120).expect("segment two"),
        ],
    );
    assert!(overlapping.is_err());
}
