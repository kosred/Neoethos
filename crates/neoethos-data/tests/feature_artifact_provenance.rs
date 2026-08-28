use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
    FeaturePlanV1, SourceArtifactBindingV1, SourceSegmentV1,
};

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn dataset() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::external(
        "feature-frame-test",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("dataset identity")
}

fn plan(formula_version: u32) -> FeaturePlanV1 {
    let source = FeatureNodeV1::source(
        "source:eurusd:m5",
        dataset(),
        "neoethos.ohlcv.f64-ms.v1",
        1,
        vec![FeatureOutputV1::f64("close", 1).expect("source output")],
        hash(1),
    )
    .expect("source node");
    let feature = FeatureNodeV1::transform(
        "indicator:precision",
        FeatureOperationTagV1::Indicator,
        formula_version,
        vec!["source:eurusd:m5".to_owned()],
        vec![FeatureOutputV1::f64("precision", 2).expect("feature output")],
        Vec::new(),
        hash(2),
        hash(formula_version as u8),
        None,
    )
    .expect("feature node");
    FeaturePlanV1::new(vec![feature, source], vec!["precision".to_owned()]).expect("feature plan")
}

fn provenance(plan: &FeaturePlanV1, generation: &str) -> DatasetFeatureArtifactProvenanceV1 {
    DatasetFeatureArtifactProvenanceV1::new(
        plan,
        vec![
            SourceArtifactBindingV1::new(
                "source:eurusd:m5",
                dataset(),
                "neoethos.dataset-manifest.v1",
                hash(3),
                generation,
                hash(4),
                BarTimestampConvention::BarOpen,
                vec![
                    SourceSegmentV1::new(0, 2, 1_704_067_200_000, 1_704_067_500_000)
                        .expect("source segment"),
                ],
            )
            .expect("source binding"),
        ],
    )
    .expect("artifact provenance")
}

fn frame(formula_version: u32, generation: &str) -> FeatureFrame {
    let plan = plan(formula_version);
    let provenance = provenance(&plan, generation);
    FeatureFrame::from_columns(
        vec![1_704_067_200_000, 1_704_067_500_000],
        vec![
            FeatureColumnF64::new(
                "precision",
                vec![1.0, 2.0],
                vec![FeatureCellValidity::Valid; 2],
            )
            .expect("feature column"),
        ],
        plan,
        provenance,
    )
    .expect("feature frame")
}

#[test]
fn frame_carries_plan_and_concrete_provenance_through_windows() {
    let frame = frame(7, "generation-a");
    let window = frame.row_window(1, 2).expect("row window");

    assert_eq!(frame.plan_identity(), window.plan_identity());
    assert_eq!(frame.provenance_identity(), window.provenance_identity());
    assert_eq!(window.timestamps, [1_704_067_500_000]);
}

#[test]
fn same_shape_but_different_semantics_or_generation_are_not_interchangeable() {
    let first = frame(7, "generation-a");
    let semantic_change = frame(8, "generation-a");
    let generation_change = frame(7, "generation-b");

    assert_ne!(first.plan_identity(), semantic_change.plan_identity());
    assert!(
        first
            .ensure_semantically_compatible(&semantic_change)
            .is_err()
    );
    assert_eq!(first.plan_identity(), generation_change.plan_identity());
    assert_ne!(
        first.provenance_identity(),
        generation_change.provenance_identity()
    );
    assert!(first.ensure_same_artifact(&generation_change).is_err());
}

#[test]
fn names_must_equal_the_plan_final_output_order() {
    let plan = plan(7);
    let provenance = provenance(&plan, "generation-a");
    let result = FeatureFrame::from_columns(
        vec![1_704_067_200_000, 1_704_067_500_000],
        vec![
            FeatureColumnF64::new(
                "renamed_without_semantic_change",
                vec![1.0, 2.0],
                vec![FeatureCellValidity::Valid; 2],
            )
            .expect("feature column"),
        ],
        plan,
        provenance,
    );
    assert!(result.is_err());
}
