use std::sync::Arc;

use neoethos_data::core::feature_run_lease::FeatureRunLease;
use neoethos_data::core::features::{
    FeatureCellValidity, FeatureColumnF64, FeatureData, FeatureFrame,
};
use neoethos_data::core::vortex_feature_store::{VortexFeatureStore, VortexFeatureStoreOptions};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1, FeaturePlanV1,
    SourceArtifactBindingV1, SourceSegmentV1,
};

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn contract() -> (FeaturePlanV1, DatasetFeatureArtifactProvenanceV1) {
    let dataset = CanonicalDatasetIdentity::external(
        "validity-test",
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
            FeatureOutputV1::f64("exact_zero", 2).expect("output"),
            FeatureOutputV1::f64("high_precision", 2).expect("output"),
        ],
        hash(1),
    )
    .expect("source node");
    let plan = FeaturePlanV1::new(
        vec![source],
        vec!["exact_zero".to_owned(), "high_precision".to_owned()],
    )
    .expect("feature plan");
    let provenance = DatasetFeatureArtifactProvenanceV1::new(
        &plan,
        vec![
            SourceArtifactBindingV1::new(
                "source:eurusd:m1",
                dataset,
                "neoethos.dataset-manifest.v1",
                hash(2),
                "generation-validity",
                hash(3),
                BarTimestampConvention::BarOpen,
                vec![
                    SourceSegmentV1::new(0, 3, 1_704_067_200_000, 1_704_067_320_000)
                        .expect("source segment"),
                ],
            )
            .expect("source binding"),
        ],
    )
    .expect("artifact provenance");
    (plan, provenance)
}

fn timestamps() -> Vec<i64> {
    vec![1_704_067_200_000, 1_704_067_260_000, 1_704_067_320_000]
}

fn columns() -> Vec<FeatureColumnF64> {
    vec![
        FeatureColumnF64::new(
            "exact_zero",
            vec![0.0, f64::NAN, 4.0],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Valid,
            ],
        )
        .expect("zero column"),
        FeatureColumnF64::new(
            "high_precision",
            vec![1.000_000_059_604_644_8, 1.0, f64::NAN],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::ZeroDenominator,
            ],
        )
        .expect("precision column"),
    ]
}

fn assert_frame(frame: &FeatureFrame) {
    assert_eq!(
        frame.cell(0, 0).expect("valid zero").value.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        frame.cell(0, 0).expect("valid zero").validity,
        FeatureCellValidity::Valid
    );
    assert_eq!(
        frame.cell(1, 0).expect("warmup").validity,
        FeatureCellValidity::Warmup
    );
    assert!(frame.cell(1, 0).expect("warmup").value.is_nan());
    assert_eq!(
        frame.cell(0, 1).expect("precision").value.to_bits(),
        1.000_000_059_604_644_8_f64.to_bits()
    );
    assert!(!frame.row_is_eligible(1, &[0, 1]).expect("eligibility"));
    assert!(!frame.row_is_eligible(2, &[0, 1]).expect("eligibility"));
    assert!(frame.row_is_eligible(0, &[0, 1]).expect("eligibility"));
}

#[test]
fn in_memory_frame_preserves_f64_bits_and_validity() {
    let (plan, provenance) = contract();
    let frame = FeatureFrame::from_columns(timestamps(), columns(), plan, provenance)
        .expect("in-memory frame");
    assert_frame(&frame);
    let selected = frame.project_columns(&[1], 0..2).expect("projection");
    assert_eq!(
        selected.columns[0].validity,
        [FeatureCellValidity::Valid; 2]
    );
    assert_eq!(
        selected.columns[0].values[0].to_bits(),
        1.000_000_059_604_644_8_f64.to_bits()
    );
}

#[test]
fn vortex_backing_has_the_identical_cell_contract() {
    let root = tempfile::tempdir().expect("scratch root");
    let lease = Arc::new(
        FeatureRunLease::create(root.path(), "frame-validity").expect("feature run lease"),
    );
    let source_columns = columns();
    let store = VortexFeatureStore::create(
        lease,
        &timestamps(),
        &source_columns,
        VortexFeatureStoreOptions {
            chunk_rows: 2,
            decoded_cache_bytes: 64 * 1024,
        },
    )
    .expect("Vortex store");
    let (plan, provenance) = contract();
    let frame = FeatureFrame::from_vortex(timestamps(), store, plan, provenance)
        .expect("Vortex-backed frame");
    assert_frame(&frame);
    let selected = frame.project_columns(&[0, 1], 0..3).expect("projection");
    for (actual, expected) in selected.columns.iter().zip(source_columns) {
        assert_eq!(actual.validity, expected.validity);
        for (actual, expected) in actual.values.iter().zip(expected.values) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn selected_frame_is_a_lazy_f64_validity_view_with_bound_provenance() {
    let (plan, provenance) = contract();
    let frame = FeatureFrame::from_columns(timestamps(), columns(), plan, provenance)
        .expect("source frame");
    let source_provenance = frame.provenance_identity();

    let selected = frame
        .select_columns(&[1, 0])
        .expect("reordered feature view");

    assert!(matches!(selected.data, FeatureData::View(_)));
    assert_eq!(selected.names, ["high_precision", "exact_zero"]);
    assert_eq!(selected.provenance_identity(), source_provenance);
    assert_ne!(selected.plan_identity(), frame.plan_identity());
    assert_eq!(
        selected.cell(0, 0).expect("precision cell").value.to_bits(),
        1.000_000_059_604_644_8_f64.to_bits()
    );
    assert_eq!(
        selected.cell(1, 1).expect("warmup cell").validity,
        FeatureCellValidity::Warmup
    );

    let nested = selected.row_window(1, 3).expect("nested row view");
    assert!(matches!(nested.data, FeatureData::View(_)));
    assert_eq!(nested.timestamps, timestamps()[1..3]);
    assert_eq!(nested.provenance_identity(), source_provenance);
    assert_eq!(
        nested.cell(0, 1).expect("nested warmup").validity,
        FeatureCellValidity::Warmup
    );

    assert!(frame.select_columns(&[]).is_err());
    assert!(frame.select_columns(&[0, 0]).is_err());
    assert!(frame.select_columns(&[2]).is_err());
}
