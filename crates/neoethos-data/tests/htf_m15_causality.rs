//! Adversarial regression tests for open-stamped, direct H1 -> M15 alignment.
//! A completed H1 feature may be consulted only at or after the hour close.
//! This test deliberately uses a future H1 value unlike the earlier hour.

use neoethos_data::core::features::{
    FeatureCellValidity, FeatureColumnF64, align_feature_columns_by_ms,
};

const MINUTE_MS: i64 = 60_000;
const HOUR_MS: i64 = 60 * MINUTE_MS;

fn at(hour: i64, minute: i64) -> i64 {
    (hour * 60 + minute) * MINUTE_MS
}

fn h1_column(values: Vec<f64>) -> FeatureColumnF64 {
    let length = values.len();
    FeatureColumnF64::new("H1_test_feature", values, vec![FeatureCellValidity::Valid; length])
        .expect("valid feature")
}

#[test]
fn m15_does_not_see_the_0900_h1_bar_until_1000() {
    // H1 08:00 opens and closes at 09:00; H1 09:00 closes at 10:00.
    let h1_open = [at(8, 0), at(9, 0), at(10, 0)];
    let m15_open = [
        at(9, 0), at(9, 15), at(9, 30), at(9, 45),
        at(10, 0), at(10, 15), at(10, 30), at(10, 45),
        at(11, 0),
    ];
    let feature = h1_column(vec![80.0, 9000.0, -10000.0]);
    let aligned = align_feature_columns_by_ms(
        &m15_open,
        &h1_open,
        &[feature],
        true,
        Some(2 * HOUR_MS),
        HOUR_MS,
    )
    .expect("causal H1 to M15 alignment");
    let observed = &aligned[0];
    assert_eq!(&observed.values[0..4], &[80.0; 4], "forming H1 leaked into M15");
    assert_eq!(&observed.values[4..8], &[9000.0; 4]);
    assert_eq!(observed.values[8], -10000.0);
    assert!(observed.validity.iter().all(|v| *v == FeatureCellValidity::Valid));
}

#[test]
fn future_h1_append_cannot_change_earlier_m15_features() {
    let m15_open = [at(9, 0), at(9, 15), at(9, 30), at(9, 45)];
    let history_open = [at(8, 0)];
    let with_future_open = [at(8, 0), at(9, 0), at(10, 0)];

    let history = align_feature_columns_by_ms(
        &m15_open, &history_open, &[h1_column(vec![80.0])],
        true, Some(2 * HOUR_MS), HOUR_MS,
    )
    .expect("short prefix");
    let future = align_feature_columns_by_ms(
        &m15_open, &with_future_open, &[h1_column(vec![80.0, -1e9, 1e9])],
        true, Some(2 * HOUR_MS), HOUR_MS,
    )
    .expect("future-appended history");

    assert_eq!(history[0].validity, future[0].validity);
    assert_eq!(history[0].values, future[0].values, "future H1 changed past M15");
}

#[test]
fn current_h1_is_unavailable_before_first_confirmed_hour_close() {
    let m15_open = [at(8, 0), at(8, 15), at(8, 30), at(8, 45)];
    let result = align_feature_columns_by_ms(
        &m15_open, &[at(8, 0)], &[h1_column(vec![9999.0])],
        true, Some(2 * HOUR_MS), HOUR_MS,
    )
    .expect("alignment without earlier closed H1");
    assert!(result[0].values.iter().all(|x| x.is_nan()));
    assert!(result[0].validity.iter().all(|v| *v == FeatureCellValidity::AlignmentMissing));
}
