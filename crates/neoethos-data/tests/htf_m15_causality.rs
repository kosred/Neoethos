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


#[test]
fn committed_real_ctrader_m1_fixture_never_exposes_h1_close_early() {
    // Real EURUSD M1 rows, saved in the repository by its existing cTrader
    // test-fixture producer. This checks the ACTUAL alignment primitive, not
    // simulated trading profitability. 100 consecutive M1 bars span
    // 2025-01-13 09:00..10:39 UTC: only 09:00..10:00 is a finished H1.
    let raw: serde_json::Value =
        serde_json::from_str(include_str!("../test_fixtures/eurusd_m1_100bars.json"))
            .expect("committed cTrader fixture JSON");
    let bars = raw["bars"].as_array().expect("bars array");
    assert_eq!(bars.len(), 100);
    let first = bars[0]["t"].as_i64().expect("M1 epoch milliseconds");
    let bar_time = |i: usize| bars[i]["t"].as_i64().expect("M1 timestamp");
    for i in 0..bars.len() {
        assert_eq!(bar_time(i), first + i as i64 * MINUTE_MS);
    }
    assert_eq!(first, 1_736_758_800_000);
    let h1_closed_value = bars[59]["c"].as_f64().expect("H1 closing value");
    assert!((h1_closed_value - 1.08388).abs() < 1e-10);

    // Only fully observed 15-minute bars; the last 10 M1 rows are ignored.
    let m15_open = (0..6_usize).map(|i| bar_time(i * 15)).collect::<Vec<_>>();
    let full = align_feature_columns_by_ms(
        &m15_open, &[first], &[h1_column(vec![h1_closed_value])],
        true, Some(2 * HOUR_MS), HOUR_MS,
    ).expect("actual H1-to-M15 alignment over the real fixture");
    assert_eq!(full[0].values.len(), 6);
    for row in 0..4 {
        assert_eq!(full[0].validity[row], FeatureCellValidity::AlignmentMissing);
        assert!(full[0].values[row].is_nan());
    }
    for row in 4..6 {
        assert_eq!(full[0].validity[row], FeatureCellValidity::Valid);
        assert_eq!(full[0].values[row], h1_closed_value);
    }

    // The first 75 M1 rows can only complete the first five M15 bars.
    // Appending the future 25 M1 rows must not rewrite the earlier H1 view.
    let prefix_open = &m15_open[..5];
    let prefix = align_feature_columns_by_ms(
        prefix_open, &[first], &[h1_column(vec![h1_closed_value])],
        true, Some(2 * HOUR_MS), HOUR_MS,
    ).expect("prefix-only alignment");
    assert_eq!(prefix[0].values, full[0].values[..5]);
    assert_eq!(prefix[0].validity, full[0].validity[..5]);
}
