use neoethos_data::Ohlcv;
use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::session_features::{
    compute_session_feature_columns, compute_session_feature_columns_f64,
};

const M30_MILLIS: i64 = 30 * 60 * 1_000;
const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

fn fixture_with_rows(rows: usize) -> Ohlcv {
    let first_timestamp = 1_704_067_200_000_i64; // 2024-01-01 00:00:00 UTC.
    let mut timestamp = Vec::with_capacity(rows);
    let mut open = Vec::with_capacity(rows);
    let mut high = Vec::with_capacity(rows);
    let mut low = Vec::with_capacity(rows);
    let mut close = Vec::with_capacity(rows);
    let mut volume = Vec::with_capacity(rows);
    for row in 0..rows {
        timestamp.push(first_timestamp + row as i64 * M30_MILLIS);
        let row_open = 100.0 + row as f64 * 0.01;
        let signed_body = (row as i32 % 5 - 2) as f64 * 0.001;
        let row_close = row_open + signed_body;
        let wick = 0.004 + (row % 3) as f64 * 0.0005;
        open.push(row_open);
        close.push(row_close);
        high.push(row_open.max(row_close) + wick);
        low.push(row_open.min(row_close) - wick);
        volume.push(100.0 + (row % 7) as f64 * 5.0);
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

fn sparse_boundary_fixture() -> Ohlcv {
    let source = fixture_with_rows(98);
    let omitted = [14_usize, 24, 48, 62, 72];
    let keep = |row: usize| !omitted.contains(&row);
    Ohlcv {
        timestamp: source.timestamp.map(|values| {
            values
                .into_iter()
                .enumerate()
                .filter_map(|(row, value)| keep(row).then_some(value))
                .collect()
        }),
        open: source
            .open
            .into_iter()
            .enumerate()
            .filter_map(|(row, value)| keep(row).then_some(value))
            .collect(),
        high: source
            .high
            .into_iter()
            .enumerate()
            .filter_map(|(row, value)| keep(row).then_some(value))
            .collect(),
        low: source
            .low
            .into_iter()
            .enumerate()
            .filter_map(|(row, value)| keep(row).then_some(value))
            .collect(),
        close: source
            .close
            .into_iter()
            .enumerate()
            .filter_map(|(row, value)| keep(row).then_some(value))
            .collect(),
        volume: source.volume.map(|values| {
            values
                .into_iter()
                .enumerate()
                .filter_map(|(row, value)| keep(row).then_some(value))
                .collect()
        }),
    }
}

fn subminute_boundary_fixture() -> Ohlcv {
    let day = 1_704_067_200_000_i64;
    let timestamps = vec![
        day,
        day + 6 * 3_600_000 + 59 * 60_000 + 59_000,
        day + 7 * 3_600_000,
        day + 7 * 3_600_000 + 30_000,
        day + 7 * 3_600_000 + 60_000,
        day + 12 * 3_600_000,
        day + 12 * 3_600_000 + 30_000,
    ];
    let rows = timestamps.len();
    let open = (0..rows)
        .map(|row| 100.0 + row as f64 * 0.1)
        .collect::<Vec<_>>();
    let close = open
        .iter()
        .enumerate()
        .map(|(row, value)| value + (row as f64 + 1.0) * 0.001)
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .map(|(open, close)| (*open).max(*close) + 0.01)
        .collect();
    let low = open
        .iter()
        .zip(&close)
        .map(|(open, close)| (*open).min(*close) - 0.01)
        .collect();
    Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: Some(vec![10.0; rows]),
    }
}

fn column<'a>(columns: &'a [FeatureColumnF64], name: &str) -> &'a FeatureColumnF64 {
    columns
        .iter()
        .find(|column| column.name == name)
        .unwrap_or_else(|| panic!("missing Session-v2 oracle column {name}"))
}

fn assert_invalid(column: &FeatureColumnF64, row: usize, reason: FeatureCellValidity) {
    assert_eq!(column.validity[row], reason, "{} row {row}", column.name);
    assert_eq!(
        column.values[row].to_bits(),
        CANONICAL_NAN_BITS,
        "{} row {row} canonical invalid payload",
        column.name
    );
}

fn cumulative_atr(bars: &Ohlcv, row: usize) -> f64 {
    if row == 0 {
        return bars.high[0] - bars.low[0];
    }
    let mut sum = 0.0;
    for index in 1..=row {
        let true_range = (bars.high[index] - bars.low[index])
            .max((bars.high[index] - bars.close[index - 1]).abs())
            .max((bars.low[index] - bars.close[index - 1]).abs());
        sum += true_range;
    }
    sum / row as f64
}

#[test]
fn asian_london_new_york_overlap_and_utc_day_boundaries() {
    let bars = fixture_with_rows(98);
    let columns = compute_session_feature_columns_f64(&bars).expect("canonical Session fixture");
    assert_eq!(columns.len(), 23);

    let asian = column(&columns, "session_asian_open_dist");
    let london = column(&columns, "session_london_open_dist");
    let new_york = column(&columns, "session_ny_open_dist");
    let overlap = column(&columns, "session_london_ny_overlap");
    let open_gap = column(&columns, "session_open_gap");
    let previous = column(&columns, "session_prev_close_dist");
    let daily = column(&columns, "daily_range_pct");

    assert_eq!(asian.validity[0], FeatureCellValidity::Valid);
    assert_invalid(london, 13, FeatureCellValidity::Warmup);
    assert_eq!(london.validity[14], FeatureCellValidity::Valid); // 07:00.
    assert_invalid(new_york, 23, FeatureCellValidity::Warmup);
    assert_eq!(new_york.validity[24], FeatureCellValidity::Valid); // 12:00.
    assert_eq!(overlap.values[23].to_bits(), 0.0_f64.to_bits());
    assert_eq!(overlap.values[24].to_bits(), 1.0_f64.to_bits());
    assert_eq!(overlap.values[31].to_bits(), 1.0_f64.to_bits());
    assert_eq!(overlap.values[32].to_bits(), 0.0_f64.to_bits());
    assert_invalid(open_gap, 14, FeatureCellValidity::Warmup);
    assert_invalid(open_gap, 24, FeatureCellValidity::Warmup);
    assert_eq!(previous.validity[47], FeatureCellValidity::Warmup);
    assert_eq!(previous.validity[48], FeatureCellValidity::Valid); // second UTC day.
    assert_eq!(daily.validity[47], FeatureCellValidity::Valid);
    assert_eq!(daily.validity[48], FeatureCellValidity::Valid);
}

#[test]
fn sparse_observed_rows_preserve_exact_open_only_resets() {
    let sparse = sparse_boundary_fixture();
    let columns = compute_session_feature_columns_f64(&sparse).expect("sparse canonical fixture");
    let london = column(&columns, "session_london_open_dist");
    let new_york = column(&columns, "session_ny_open_dist");
    let daily = column(&columns, "daily_range_pct");
    assert!(
        london
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup),
        "missing every exact 07:00 row must never invent a London reset"
    );
    assert!(
        new_york
            .validity
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup),
        "missing every exact 12:00 row must never invent a New York reset"
    );
    assert_eq!(daily.validity[0], FeatureCellValidity::Valid);
}

#[test]
fn subminute_rows_repeat_the_cpu_minute_boundary_reset() {
    let bars = subminute_boundary_fixture();
    let columns = compute_session_feature_columns_f64(&bars).expect("subminute boundary fixture");
    let london_open = column(&columns, "session_london_open_dist");
    assert_eq!(
        london_open.values[3].to_bits(),
        ((bars.close[3] - bars.open[3]) / cumulative_atr(&bars, 3)).to_bits(),
        "07:00:30 is a second exact minute-boundary reset"
    );
    assert_eq!(
        london_open.values[4].to_bits(),
        ((bars.close[4] - bars.open[3]) / cumulative_atr(&bars, 4)).to_bits(),
        "07:01 continues from the last observed 07:00 reset"
    );
    let open_gap = column(&columns, "session_open_gap");
    assert_eq!(open_gap.validity[2], FeatureCellValidity::Warmup);
    assert_eq!(open_gap.validity[3], FeatureCellValidity::Valid);
    assert_eq!(open_gap.validity[5], FeatureCellValidity::Valid);
    assert_eq!(open_gap.validity[6], FeatureCellValidity::Valid);
}

#[test]
fn zero_atr_and_zero_volume_emit_canonical_nan_with_typed_reason() {
    let timestamp = vec![1_704_067_200_000, 1_704_067_200_000 + M30_MILLIS];
    let bars = Ohlcv {
        timestamp: Some(timestamp),
        open: vec![100.0, 100.0],
        high: vec![100.0, 100.1],
        low: vec![100.0, 99.9],
        close: vec![100.0, 100.0],
        volume: Some(vec![0.0, 0.0]),
    };
    let columns = compute_session_feature_columns_f64(&bars).expect("zero-denominator fixture");
    assert_invalid(
        column(&columns, "session_asian_open_dist"),
        0,
        FeatureCellValidity::ZeroDenominator,
    );
    assert_invalid(
        column(&columns, "daily_vwap_dist"),
        0,
        FeatureCellValidity::ZeroDenominator,
    );
    assert_eq!(
        column(&columns, "session_asian_open_dist").validity[1],
        FeatureCellValidity::Valid
    );
    assert_invalid(
        column(&columns, "daily_vwap_dist"),
        1,
        FeatureCellValidity::ZeroDenominator,
    );
}

#[test]
fn legacy_value_clock_infers_seconds_while_typed_resident_authority_rejects_them() {
    let millis = fixture_with_rows(40);
    let mut seconds = millis.clone();
    seconds.timestamp = millis
        .timestamp
        .as_ref()
        .map(|timestamps| timestamps.iter().map(|value| value / 1_000).collect());
    let millisecond_values = compute_session_feature_columns(&millis);
    let second_values = compute_session_feature_columns(&seconds);
    assert_eq!(millisecond_values.len(), second_values.len());
    for ((left_name, left), (right_name, right)) in millisecond_values.iter().zip(&second_values) {
        assert_eq!(left_name, right_name);
        assert_eq!(
            left.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            right
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    let error = compute_session_feature_columns_f64(&seconds)
        .expect_err("typed resident authority accepts canonical milliseconds only");
    assert!(error.to_string().contains("canonical timestamp"));
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
#[test]
fn rtx_device_fixture_matches_all_session_v2_value_bits_and_validity_codes() {
    use neoethos_gpu_cuda::resident_session_v2_device_fixture::run_resident_session_v2_device_fixture;

    for bars in [
        fixture_with_rows(98),
        fixture_with_rows(1_024),
        sparse_boundary_fixture(),
        subminute_boundary_fixture(),
    ] {
        let expected =
            compute_session_feature_columns_f64(&bars).expect("typed CPU Session-v2 oracle");
        let actual = run_resident_session_v2_device_fixture(
            &bars.open,
            &bars.high,
            &bars.low,
            &bars.close,
            bars.volume.as_deref().expect("fixture volume"),
            bars.timestamp.as_deref().expect("fixture timestamps"),
        )
        .expect("resident Session-v2 test-only parity D2H fixture");
        for (index, expected_column) in expected.iter().enumerate() {
            let offset = index * bars.len();
            assert_eq!(
                expected_column
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual.values[offset..offset + bars.len()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{} value bits",
                expected_column.name
            );
            assert_eq!(
                expected_column
                    .validity
                    .iter()
                    .map(|validity| validity.code())
                    .collect::<Vec<_>>(),
                actual.validity_u8[offset..offset + bars.len()],
                "{} validity",
                expected_column.name
            );
        }
    }
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
#[test]
#[ignore = "RTX nsys kernel-distribution gate"]
fn rtx_session_v2_4096_resident_kernel_perf_fixture() {
    use neoethos_gpu_cuda::resident_session_v2_device_fixture::run_resident_session_v2_device_perf_fixture;

    let bars = fixture_with_rows(4_096);
    run_resident_session_v2_device_perf_fixture(
        &bars.open,
        &bars.high,
        &bars.low,
        &bars.close,
        bars.volume.as_deref().expect("fixture volume"),
        bars.timestamp.as_deref().expect("fixture timestamps"),
        257,
    )
    .expect("resident Session-v2 zero-feature-D2H perf fixture");
}
