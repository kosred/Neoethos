use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::normalization::{
    NORMALIZATION_TRANSFORM_SEMANTIC_VERSION, normalize_feature_column_f64,
};

fn column(values: Vec<f64>, validity: Vec<FeatureCellValidity>) -> FeatureColumnF64 {
    FeatureColumnF64::new("robust-v2-oracle", values, validity).expect("valid oracle column")
}

#[test]
fn mad_branch_and_full_range_apply_have_exact_f64_bits() {
    let mut feature = column(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0, 1_000.0, f64::NAN],
        vec![
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Gap,
        ],
    );

    let fit = normalize_feature_column_f64(&mut feature, 0..5).expect("MAD fit");
    assert_eq!(NORMALIZATION_TRANSFORM_SEMANTIC_VERSION, 2);
    assert_eq!(fit.training_rows, 0..5);
    assert_eq!(fit.median.to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(fit.scale.to_bits(), 0x3ff7_b8ba_c710_cb29);
    assert_eq!(fit.valid_training_cells, 5);
    assert!(!fit.degenerate);
    assert_eq!(
        feature.values[..6]
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [
            0xbff5_956d_a52c_ff6a,
            0xbfe5_956d_a52c_ff6a,
            0x0000_0000_0000_0000,
            0x3fe5_956d_a52c_ff6a,
            0x3ff5_956d_a52c_ff6a,
            0x4024_0000_0000_0000,
        ]
    );
    assert_eq!(feature.validity[6], FeatureCellValidity::Gap);
    assert_eq!(feature.values[6].to_bits(), f64::NAN.to_bits());
}

#[test]
fn zero_mad_uses_exact_sorted_population_standard_deviation_fallback() {
    let mut feature = column(
        vec![0.0, 0.0, 0.0, 1.0, 1.0],
        vec![FeatureCellValidity::Valid; 5],
    );

    let fit = normalize_feature_column_f64(&mut feature, 0..4).expect("fallback fit");
    assert_eq!(fit.median.to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(fit.scale.to_bits(), 0x3fdb_b67a_e858_4caa);
    assert_eq!(fit.valid_training_cells, 4);
    assert!(!fit.degenerate);
    assert_eq!(
        feature
            .values
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x4002_79a7_4590_331d,
            0x4002_79a7_4590_331d,
        ]
    );
}
