use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    SourceArtifactBindingV1, SourceSegmentV1,
};

const TIMESTAMPS: [i64; 4] = [
    1_704_067_200_000,
    1_704_067_260_000,
    1_704_067_320_000,
    1_704_067_380_000,
];

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn frame() -> FeatureFrame {
    let dataset = CanonicalDatasetIdentity::external(
        "row-selection-test",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("dataset identity");
    let source = FeatureNodeV1::source(
        "source:eurusd:m1",
        dataset.clone(),
        "neoethos.test-features.f64-ms.v1",
        1,
        vec![
            FeatureOutputV1::f64("alpha", 2).expect("alpha output"),
            FeatureOutputV1::f64("beta", 2).expect("beta output"),
        ],
        hash(1),
    )
    .expect("source node");
    let plan = FeaturePlanV1::new(vec![source], vec!["alpha".to_owned(), "beta".to_owned()])
        .expect("feature plan");
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![
            SourceArtifactBindingV1::new(
                "source:eurusd:m1",
                dataset,
                "neoethos.dataset-manifest.v1",
                hash(2),
                "generation-row-selection",
                hash(3),
                BarTimestampConvention::BarOpen,
                vec![
                    SourceSegmentV1::new(0, 4, TIMESTAMPS[0], TIMESTAMPS[3])
                        .expect("source segment"),
                ],
            )
            .expect("source binding"),
        ],
    )
    .expect("artifact provenance");
    let columns = vec![
        FeatureColumnF64::new(
            "alpha",
            vec![10.0, 20.0, 30.0, 40.0],
            vec![FeatureCellValidity::Valid; 4],
        )
        .expect("alpha column"),
        FeatureColumnF64::new(
            "beta",
            vec![1.0, f64::NAN, 3.0, 4.0],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
            ],
        )
        .expect("beta column"),
    ];

    FeatureFrame::from_columns(TIMESTAMPS.to_vec(), columns, plan, provenance)
        .expect("feature frame")
}

#[test]
fn arbitrary_row_selection_preserves_schema_values_validity_and_receipts() {
    let source = frame();
    let plan = source.plan_identity();
    let provenance = source.provenance_identity();

    let selected = source
        .select_rows(&[0, 2, 3])
        .expect("strictly increasing row selection");

    assert_eq!(selected.names, source.names);
    assert_eq!(
        selected.timestamps,
        [TIMESTAMPS[0], TIMESTAMPS[2], TIMESTAMPS[3]]
    );
    assert_eq!(selected.plan_identity(), plan);
    assert_eq!(selected.provenance_identity(), provenance);

    let batch = selected
        .project_columns(&[1, 0], 0..selected.n_samples())
        .expect("selected projection");
    assert_eq!(batch.row_ids, [0, 2, 3]);
    assert_eq!(batch.timestamps, selected.timestamps);
    assert_eq!(batch.columns[0].name, "beta");
    assert_eq!(batch.columns[0].values, [1.0, 3.0, 4.0]);
    assert_eq!(batch.columns[0].validity, [FeatureCellValidity::Valid; 3]);
    assert_eq!(batch.columns[1].values, [10.0, 30.0, 40.0]);

    let nested = selected.row_window(1, 3).expect("nested selected window");
    let nested_batch = nested
        .project_columns(&[0, 1], 0..nested.n_samples())
        .expect("nested projection");
    assert_eq!(nested_batch.row_ids, [2, 3]);

    let projected = selected
        .select_columns(&[1])
        .expect("column projection after row selection");
    let projected_batch = projected
        .project_columns(&[0], 0..projected.n_samples())
        .expect("row receipt projection");
    assert_eq!(projected_batch.row_ids, [0, 2, 3]);
}

#[test]
fn arbitrary_row_selection_rejects_noncanonical_indices() {
    let source = frame();

    assert!(
        source.select_rows(&[]).is_err(),
        "empty selection must fail"
    );
    assert!(
        source.select_rows(&[2, 1]).is_err(),
        "descending indices must fail"
    );
    assert!(
        source.select_rows(&[1, 1]).is_err(),
        "duplicate indices must fail"
    );
    assert!(
        source.select_rows(&[0, 4]).is_err(),
        "out-of-bounds indices must fail"
    );
}
