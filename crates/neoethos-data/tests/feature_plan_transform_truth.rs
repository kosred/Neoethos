use neoethos_data::Ohlcv;
use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::normalization::normalize_feature_column_f64;
use neoethos_data::core::{
    cross_pair_features::{
        CROSS_PAIR_TRANSFORM_SEMANTIC_VERSION, compute_cross_pair_feature_columns_f64,
    },
    features::{HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION, align_feature_columns_by_ms},
    normalization::NORMALIZATION_TRANSFORM_SEMANTIC_VERSION,
};

fn column(name: &str, values: Vec<f64>, validity: Vec<FeatureCellValidity>) -> FeatureColumnF64 {
    FeatureColumnF64::new(name, values, validity).expect("valid test column")
}

#[test]
fn normalization_preserves_invalidity_and_valid_mathematical_zero() {
    let mut feature = column(
        "truth",
        vec![f64::NAN, 0.0, 1.0, 2.0, f64::NAN, 4.0],
        vec![
            FeatureCellValidity::Warmup,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Gap,
            FeatureCellValidity::Valid,
        ],
    );

    let fit = normalize_feature_column_f64(&mut feature, 1..4).expect("normalization");
    assert_eq!(fit.training_rows, 1..4);
    assert_eq!(feature.validity[0], FeatureCellValidity::Warmup);
    assert_eq!(feature.validity[4], FeatureCellValidity::Gap);
    assert!(feature.values[0].is_nan());
    assert!(feature.values[4].is_nan());
    assert_eq!(feature.validity[1], FeatureCellValidity::Valid);
    assert!(feature.values[1].is_finite());
    assert_ne!(
        feature.values[1].to_bits(),
        f64::NAN.to_bits(),
        "a valid mathematical zero must remain a valid numeric observation"
    );
}

#[test]
fn normalization_fit_is_invariant_to_future_and_validation_values() {
    let prefix = vec![1.0, 2.0, 4.0, 8.0];
    let mut first = column(
        "causal",
        prefix
            .iter()
            .copied()
            .chain([16.0, 32.0, 64.0, 128.0])
            .collect(),
        vec![FeatureCellValidity::Valid; 8],
    );
    let mut perturbed_future = column(
        "causal",
        prefix
            .iter()
            .copied()
            .chain([1.0e100, -1.0e100, 7.0e80, -9.0e90])
            .collect(),
        vec![FeatureCellValidity::Valid; 8],
    );

    let first_fit = normalize_feature_column_f64(&mut first, 0..4).expect("first fit");
    let second_fit = normalize_feature_column_f64(&mut perturbed_future, 0..4).expect("second fit");
    assert_eq!(first_fit, second_fit, "future values changed fitted state");
    assert_eq!(
        &first.values[..4],
        &perturbed_future.values[..4],
        "future values changed earlier normalized features"
    );
}

#[test]
fn invalid_training_cells_do_not_enter_the_fit() {
    let mut with_invalid_outlier = column(
        "masked",
        vec![1.0, 2.0, f64::NAN, 3.0, 4.0],
        vec![
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Gap,
            FeatureCellValidity::Valid,
            FeatureCellValidity::Valid,
        ],
    );
    let mut without_row = column(
        "masked",
        vec![1.0, 2.0, 3.0, 4.0],
        vec![FeatureCellValidity::Valid; 4],
    );

    let masked_fit =
        normalize_feature_column_f64(&mut with_invalid_outlier, 0..5).expect("masked fit");
    let compact_fit = normalize_feature_column_f64(&mut without_row, 0..4).expect("compact fit");
    assert_eq!(masked_fit.median.to_bits(), compact_fit.median.to_bits());
    assert_eq!(masked_fit.scale.to_bits(), compact_fit.scale.to_bits());
    assert_eq!(with_invalid_outlier.validity[2], FeatureCellValidity::Gap);
    assert!(with_invalid_outlier.values[2].is_nan());
}

#[test]
fn constant_training_column_is_explicitly_degenerate_not_zero_signal() {
    let mut feature = column(
        "constant",
        vec![7.0; 6],
        vec![FeatureCellValidity::Valid; 6],
    );
    let fit = normalize_feature_column_f64(&mut feature, 0..4).expect("degenerate fit");

    assert!(fit.degenerate);
    assert!(
        feature
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Degenerate)
    );
    assert!(feature.values.iter().all(|value| value.is_nan()));
}

#[test]
fn non_finite_cell_cannot_be_marked_valid() {
    let error = FeatureColumnF64::new(
        "bad",
        vec![1.0, f64::INFINITY],
        vec![FeatureCellValidity::Valid, FeatureCellValidity::Valid],
    )
    .expect_err("valid non-finite cells must fail at the producer boundary");
    assert!(error.to_string().contains("non-finite"));
}

#[test]
fn repaired_transform_contracts_have_explicit_semantic_versions() {
    assert_eq!(NORMALIZATION_TRANSFORM_SEMANTIC_VERSION, 2);
    assert_eq!(HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION, 3);
    assert_eq!(CROSS_PAIR_TRANSFORM_SEMANTIC_VERSION, 2);
}

const M1_MS: i64 = 60_000;
// Exact M5 boundary for deterministic direct-source alignment fixtures.
const TEST_EPOCH_MS: i64 = 1_700_000_100_000;

fn ohlcv(closes: &[f64], timestamps: Option<Vec<i64>>) -> Ohlcv {
    Ohlcv {
        timestamp: timestamps,
        open: closes.to_vec(),
        high: closes.iter().map(|value| value + 0.01).collect(),
        low: closes.iter().map(|value| value - 0.01).collect(),
        close: closes.to_vec(),
        volume: Some(vec![1.0; closes.len()]),
    }
}

fn ms_grid(minutes: &[i64]) -> Vec<i64> {
    minutes
        .iter()
        .map(|minute| TEST_EPOCH_MS + minute * M1_MS)
        .collect()
}

fn find_column<'a>(columns: &'a [FeatureColumnF64], name: &str) -> &'a FeatureColumnF64 {
    columns
        .iter()
        .find(|column| column.name == name)
        .unwrap_or_else(|| panic!("missing feature column {name}"))
}

#[test]
fn cross_pair_requires_canonical_timestamps_instead_of_index_fallback() {
    let base = ohlcv(&[1.0, 1.1, 1.2], Some(ms_grid(&[0, 1, 2])));
    let related = ohlcv(&[2.0, 2.1, 2.2], None);

    let error = compute_cross_pair_feature_columns_f64(
        &base,
        &[("GBPUSD".to_owned(), &related)],
        &[2],
        M1_MS,
    )
    .expect_err("missing related timestamps must not fall back to row index");

    assert!(error.to_string().contains("timestamp"));
}

#[test]
fn cross_pair_alignment_is_causal_and_marks_stale_observations() {
    let base = ohlcv(&[1.0, 1.1, 1.2, 1.3], Some(ms_grid(&[0, 1, 2, 3])));
    let related = ohlcv(&[2.0, 4.0], Some(ms_grid(&[0, 3])));

    let columns = compute_cross_pair_feature_columns_f64(
        &base,
        &[("GBPUSD".to_owned(), &related)],
        &[2],
        M1_MS,
    )
    .expect("typed cross-pair features");
    let spread = find_column(&columns, "spread_GBPUSD");

    assert_eq!(spread.validity[0], FeatureCellValidity::Valid);
    assert_eq!(spread.validity[1], FeatureCellValidity::Valid);
    assert_eq!(spread.validity[2], FeatureCellValidity::Stale);
    assert_eq!(spread.validity[3], FeatureCellValidity::Valid);
    assert!(spread.values[2].is_nan());
    let expected_at_minute_one = 1.1_f64.ln() - 2.0_f64.ln();
    assert_eq!(spread.values[1].to_bits(), expected_at_minute_one.to_bits());
    assert_ne!(
        spread.values[1].to_bits(),
        (1.1_f64.ln() - 4.0_f64.ln()).to_bits(),
        "the future related bar leaked into an earlier base row"
    );
}

#[test]
fn cross_pair_warmup_and_zero_denominator_are_not_numeric_zero() {
    let base = ohlcv(&[1.0, 1.0, 1.0, 1.0], Some(ms_grid(&[0, 1, 2, 3])));
    let related = ohlcv(&[2.0, 2.0, 2.0, 2.0], Some(ms_grid(&[0, 1, 2, 3])));

    let columns = compute_cross_pair_feature_columns_f64(
        &base,
        &[("GBPUSD".to_owned(), &related)],
        &[2],
        M1_MS,
    )
    .expect("typed cross-pair features");
    let correlation = find_column(&columns, "xcorr_GBPUSD_2");
    let zscore = find_column(&columns, "spread_z_GBPUSD_2");

    assert_eq!(correlation.validity[0], FeatureCellValidity::Warmup);
    assert_eq!(correlation.validity[1], FeatureCellValidity::Warmup);
    assert_eq!(
        correlation.validity[2],
        FeatureCellValidity::ZeroDenominator
    );
    assert_eq!(zscore.validity[0], FeatureCellValidity::Warmup);
    assert_eq!(zscore.validity[1], FeatureCellValidity::ZeroDenominator);
    assert!(correlation.values[..=2].iter().all(|value| value.is_nan()));
    assert!(zscore.values[..=1].iter().all(|value| value.is_nan()));
}

#[test]
fn cross_pair_prefix_is_invariant_to_future_append_and_perturbation() {
    let prefix_base = [1.0, 1.1, 1.2, 1.4];
    let prefix_related = [2.0, 2.2, 2.1, 2.5];
    let short_base = ohlcv(&prefix_base, Some(ms_grid(&[0, 1, 2, 3])));
    let short_related = ohlcv(&prefix_related, Some(ms_grid(&[0, 1, 2, 3])));
    let long_base = ohlcv(
        &[1.0, 1.1, 1.2, 1.4, 50_000.0, 0.000_02],
        Some(ms_grid(&[0, 1, 2, 3, 4, 5])),
    );
    let long_related = ohlcv(
        &[2.0, 2.2, 2.1, 2.5, 0.000_03, 70_000.0],
        Some(ms_grid(&[0, 1, 2, 3, 4, 5])),
    );

    let short = compute_cross_pair_feature_columns_f64(
        &short_base,
        &[("GBPUSD".to_owned(), &short_related)],
        &[2],
        M1_MS,
    )
    .expect("short features");
    let long = compute_cross_pair_feature_columns_f64(
        &long_base,
        &[("GBPUSD".to_owned(), &long_related)],
        &[2],
        M1_MS,
    )
    .expect("long features");

    for (short_column, long_column) in short.iter().zip(&long) {
        assert_eq!(short_column.name, long_column.name);
        assert_eq!(short_column.validity, long_column.validity[..4]);
        for row in 0..4 {
            assert_eq!(
                short_column.values[row].to_bits(),
                long_column.values[row].to_bits(),
                "future append changed {} row {row}",
                short_column.name
            );
        }
    }
}

#[test]
fn typed_higher_timeframe_alignment_uses_close_availability_and_staleness() {
    let base_ms = ms_grid(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let higher_ms = ms_grid(&[0, 5]);
    let source = column(
        "M5_truth",
        vec![10.0, 20.0],
        vec![FeatureCellValidity::Valid; 2],
    );

    let aligned = align_feature_columns_by_ms(
        &base_ms,
        &higher_ms,
        &[source],
        true,
        Some(2 * M1_MS),
        5 * M1_MS,
    )
    .expect("typed HTF alignment");
    let result = &aligned[0];

    assert!(
        result.validity[..5]
            .iter()
            .all(|reason| *reason == FeatureCellValidity::AlignmentMissing)
    );
    assert_eq!(result.values[5], 10.0);
    assert_eq!(result.values[7], 10.0);
    assert_eq!(result.validity[8], FeatureCellValidity::Stale);
    assert_eq!(result.validity[9], FeatureCellValidity::Stale);
    assert_eq!(result.values[10], 20.0);
}

#[test]
fn typed_higher_timeframe_alignment_propagates_source_invalidity() {
    let source = column(
        "M5_truth",
        vec![f64::NAN, 20.0],
        vec![FeatureCellValidity::Warmup, FeatureCellValidity::Valid],
    );
    let aligned = align_feature_columns_by_ms(
        &ms_grid(&[0, 5, 10]),
        &ms_grid(&[0, 5]),
        &[source],
        true,
        Some(10 * M1_MS),
        5 * M1_MS,
    )
    .expect("typed HTF alignment");

    assert_eq!(
        aligned[0].validity[0],
        FeatureCellValidity::AlignmentMissing
    );
    assert_eq!(aligned[0].validity[1], FeatureCellValidity::Warmup);
    assert_eq!(aligned[0].validity[2], FeatureCellValidity::Valid);
}
