use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::quant_features::{
    compute_quant_feature_columns_f64, compute_quant_feature_columns_v3_f64,
};
use neoethos_data::{CanonicalTimeframe, Ohlcv};

#[path = "../src/core/quant_exact_math_v3.rs"]
mod quant_exact_math_v3;
#[path = "../../neoethos-gpu-cuda/src/resident_quant_v3_census.rs"]
mod resident_quant_v3_census;

use quant_exact_math_v3::quant_log_positive_f64_v3;
use resident_quant_v3_census::{
    RESIDENT_QUANT_COLUMN_NAMES_V3, RESIDENT_QUANT_ROUTE_CENSUS_V3,
    RESIDENT_QUANT_TRADING_SESSIONS_PER_YEAR_V3, ResidentQuantRouteLineageV3,
};

const UTC_DAY_MILLIS: i64 = 86_400_000;
const ASIAN_SESSION_MILLIS: i64 = 8 * 60 * 60 * 1_000;
const M30_MILLIS: i64 = 30 * 60 * 1_000;
#[cfg(feature = "gpu-cuda-device-fixtures")]
const M15_MILLIS: i64 = 15 * 60 * 1_000;
const EPS: f64 = 1e-12;

type OracleResult<T> = Result<T, String>;

#[derive(Debug, Clone, Copy)]
struct CompletedUtcDay {
    high: f64,
    low: f64,
    close: f64,
}

fn ordinary_m30_fixture() -> Ohlcv {
    fixture_with_rows(600)
}

fn fixture_with_rows(rows: usize) -> Ohlcv {
    fixture_with_rows_at_timeframe(rows, M30_MILLIS)
}

fn fixture_with_rows_at_timeframe(rows: usize, timeframe_millis: i64) -> Ohlcv {
    let first_timestamp = 1_704_067_200_000_i64; // 2024-01-01 00:00:00 UTC.
    let mut timestamp = Vec::with_capacity(rows);
    let mut open = Vec::with_capacity(rows);
    let mut high = Vec::with_capacity(rows);
    let mut low = Vec::with_capacity(rows);
    let mut close = Vec::with_capacity(rows);
    let mut volume = Vec::with_capacity(rows);
    for row in 0..rows {
        timestamp.push(first_timestamp + row as i64 * timeframe_millis);
        let row_open = 100.0 + row as f64 * 0.000_125 + (row % 17) as f64 * 0.0025;
        let signed_body = (row as i32 % 7 - 3) as f64 * 0.0007;
        let row_close = row_open + signed_body;
        let wick = 0.003 + (row % 5) as f64 * 0.0002;
        open.push(row_open);
        close.push(row_close);
        high.push(row_open.max(row_close) + wick);
        low.push(row_open.min(row_close) - wick);
        volume.push(100.0 + (row % 23) as f64 * 3.0);
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

fn exact_grid_gap_fixture() -> Ohlcv {
    let source = fixture_with_rows(620);
    let omitted = [31_usize, 32, 97, 241, 242, 243];
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

fn positive_subfloor_fixture() -> Ohlcv {
    let rows = 140;
    let first_timestamp = 1_704_067_200_000_i64;
    let mut bars = Ohlcv {
        timestamp: Some(
            (0..rows)
                .map(|row| first_timestamp + row as i64 * M30_MILLIS)
                .collect(),
        ),
        open: vec![5.0e-11; rows],
        high: vec![6.0e-11; rows],
        low: vec![4.0e-11; rows],
        close: vec![5.0e-11; rows],
        volume: Some(vec![100.0; rows]),
    };
    for (row, close) in bars.close.iter_mut().enumerate() {
        *close += (row % 3) as f64 * 1.0e-13;
    }
    bars
}

fn flat_close_transition_fixture() -> Ohlcv {
    let mut bars = fixture_with_rows(600);
    for row in (25..bars.len()).step_by(7) {
        bars.close[row] = bars.close[row - 1];
        let wick = 0.003 + (row % 5) as f64 * 0.0002;
        bars.high[row] = bars.open[row].max(bars.close[row]) + wick;
        bars.low[row] = bars.open[row].min(bars.close[row]) - wick;
    }
    bars
}

fn validate_grid(bars: &Ohlcv, timeframe_millis: i64) -> OracleResult<(usize, usize, usize)> {
    if timeframe_millis <= 0
        || UTC_DAY_MILLIS % timeframe_millis != 0
        || ASIAN_SESSION_MILLIS % timeframe_millis != 0
    {
        return Err("Quant-v3 requires a fixed grid dividing UTC day and Asian session".into());
    }
    let bars_per_day =
        usize::try_from(UTC_DAY_MILLIS / timeframe_millis).map_err(|_| "bars-per-day width")?;
    let bars_per_asian = usize::try_from(ASIAN_SESSION_MILLIS / timeframe_millis)
        .map_err(|_| "bars-per-Asian-session width")?;
    if bars_per_asian < 12 {
        return Err("Quant-v3 requires at least twelve Asian-session bars".into());
    }
    let timestamps = bars
        .timestamp
        .as_deref()
        .ok_or_else(|| "Quant-v3 requires canonical millisecond timestamps".to_owned())?;
    if timestamps.len() != bars.len() || timestamps.is_empty() {
        return Err("Quant-v3 timestamp extent mismatch".into());
    }
    for (row, timestamp) in timestamps.iter().copied().enumerate() {
        if timestamp.rem_euclid(timeframe_millis) != 0 {
            return Err(format!("Quant-v3 timestamp row {row} is off-grid"));
        }
    }
    for (row, pair) in timestamps.windows(2).enumerate() {
        let gap = pair[1] - pair[0];
        if gap <= 0 || gap % timeframe_millis != 0 {
            return Err(format!("Quant-v3 timestamp gap after row {row} is invalid"));
        }
    }
    Ok((bars_per_asian, bars_per_day, bars_per_day * 5))
}

fn replace_column(
    columns: &mut [FeatureColumnF64],
    name: &str,
    values: Vec<f64>,
    validity: Vec<FeatureCellValidity>,
) -> OracleResult<()> {
    let replacement =
        FeatureColumnF64::new(name, values, validity).map_err(|error| error.to_string())?;
    let slot = columns
        .iter_mut()
        .find(|column| column.name == name)
        .ok_or_else(|| format!("Quant-v3 oracle omitted {name}"))?;
    *slot = replacement;
    Ok(())
}

fn fixed_validity(rows: usize, warmup: usize) -> Vec<FeatureCellValidity> {
    (0..rows)
        .map(|row| {
            if row < warmup {
                FeatureCellValidity::Warmup
            } else {
                FeatureCellValidity::Valid
            }
        })
        .collect()
}

fn cloned_validity(
    columns: &[FeatureColumnF64],
    name: &str,
) -> OracleResult<Vec<FeatureCellValidity>> {
    columns
        .iter()
        .find(|column| column.name == name)
        .map(|column| column.validity.clone())
        .ok_or_else(|| format!("Quant-v3 oracle omitted {name}"))
}

fn exact_log(value: f64) -> OracleResult<f64> {
    quant_log_positive_f64_v3(value).ok_or_else(|| format!("Quant-v3 exact log rejected {value}"))
}

fn install_exact_log_migrations(
    bars: &Ohlcv,
    bars_per_day: usize,
    columns: &mut [FeatureColumnF64],
) -> OracleResult<()> {
    let rows = bars.len();
    let mut log_returns = vec![0.0; rows];
    for row in 1..rows {
        if bars.close[row - 1].abs() > 1e-10 && bars.close[row].abs() > 1e-10 {
            log_returns[row] = exact_log(bars.close[row] / bars.close[row - 1])?;
        }
    }
    replace_column(
        columns,
        "quant_log_return",
        log_returns.clone(),
        cloned_validity(columns, "quant_log_return")?,
    )?;

    let mut log_volatility = vec![0.0; rows];
    for row in 0..rows {
        let range = bars.high[row] - bars.low[row];
        if range > 1e-10 {
            log_volatility[row] = exact_log(range)?;
        }
    }
    replace_column(
        columns,
        "quant_log_volatility",
        log_volatility,
        cloned_validity(columns, "quant_log_volatility")?,
    )?;

    let annualization_periods = RESIDENT_QUANT_TRADING_SESSIONS_PER_YEAR_V3
        .checked_mul(bars_per_day as u64)
        .ok_or_else(|| "Quant-v3 annualization periods overflow".to_owned())?;
    let annualization_scale = (annualization_periods as f64).sqrt();

    for window in [5_usize, 10, 20, 50] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum_squared = 0.0;
            for value in &log_returns[(row - window + 1)..=row] {
                sum_squared += value * value;
            }
            values[row] = (sum_squared / window as f64).sqrt() * annualization_scale;
        }
        replace_column(
            columns,
            &format!("quant_realized_vol_{window}"),
            values,
            fixed_validity(rows, window),
        )?;
    }

    for window in [10_usize, 20] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum = 0.0;
            for index in (row - window + 1)..=row {
                if bars.open[index].abs() > 1e-10 {
                    let up = exact_log(bars.high[index] / bars.open[index])?;
                    let down = exact_log(bars.low[index] / bars.open[index])?;
                    let close = exact_log(bars.close[index] / bars.open[index])?;
                    sum += 0.5 * (up - down).powi(2) - (exact_log(2.0)? - 1.0) * close.powi(2);
                }
            }
            values[row] = (sum / window as f64).abs().sqrt() * annualization_scale;
        }
        replace_column(
            columns,
            &format!("quant_gk_vol_{window}"),
            values,
            fixed_validity(rows, window),
        )?;
    }

    for window in [10_usize, 20] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum = 0.0;
            for index in (row - window + 1)..=row {
                if bars.low[index] > 1e-10 {
                    let high_low = exact_log(bars.high[index] / bars.low[index])?;
                    sum += high_low * high_low;
                }
            }
            let factor = 1.0 / (4.0 * window as f64 * exact_log(2.0)?);
            values[row] = (factor * sum).sqrt() * annualization_scale;
        }
        replace_column(
            columns,
            &format!("quant_parkinson_vol_{window}"),
            values,
            fixed_validity(rows, window),
        )?;
    }

    let mut vol_ratio = vec![0.0; rows];
    let mut vol_ratio_validity = fixed_validity(rows, 20);
    for row in 20..rows {
        let mut short_squared = 0.0;
        let mut long_squared = 0.0;
        for value in &log_returns[(row - 4)..=row] {
            short_squared += value * value;
        }
        for value in &log_returns[(row - 19)..=row] {
            long_squared += value * value;
        }
        let short = (short_squared / 5.0).sqrt();
        let long = (long_squared / 20.0).sqrt();
        if long_squared <= EPS {
            vol_ratio_validity[row] = FeatureCellValidity::ZeroDenominator;
        }
        vol_ratio[row] = if long > 1e-10 { short / long } else { 1.0 };
    }
    replace_column(columns, "quant_vol_ratio", vol_ratio, vol_ratio_validity)?;

    let mut hurst = vec![0.5; rows];
    let mut hurst_validity = fixed_validity(rows, 100);
    for row in 100..rows {
        let slice = &log_returns[(row - 99)..=row];
        let mean = slice.iter().sum::<f64>() / 100.0;
        let mut cumulative = Vec::with_capacity(100);
        let mut running = 0.0;
        for value in slice {
            running += value - mean;
            cumulative.push(running);
        }
        let spread = cumulative.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - cumulative.iter().copied().fold(f64::INFINITY, f64::min);
        let variance = slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 99.0;
        let deviation = variance.sqrt();
        if deviation <= EPS || spread <= EPS {
            hurst_validity[row] = FeatureCellValidity::ZeroDenominator;
            continue;
        }
        hurst[row] = (exact_log(spread / deviation)? / exact_log(100.0)?).clamp(0.0, 1.0);
    }
    replace_column(columns, "quant_hurst_100", hurst, hurst_validity)?;

    for lag in [1_usize, 5, 10] {
        let mut autocorrelation = vec![0.0; rows];
        let mut autocorrelation_validity = fixed_validity(rows, 50 + lag);
        for row in (50 + lag)..rows {
            let slice = &log_returns[(row - 49)..=row];
            let mean = slice.iter().sum::<f64>() / 50.0;
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for offset in lag..50 {
                let current = slice[offset] - mean;
                let lagged = slice[offset - lag] - mean;
                numerator += current * lagged;
                denominator += current * current;
            }
            if denominator <= EPS {
                autocorrelation_validity[row] = FeatureCellValidity::ZeroDenominator;
            }
            autocorrelation[row] = if denominator.abs() > EPS {
                numerator / denominator
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
        }
        let name = format!("quant_autocorr_{lag}");
        replace_column(columns, &name, autocorrelation, autocorrelation_validity)?;
    }

    let mut skewness = vec![0.0; rows];
    let mut kurtosis = vec![0.0; rows];
    let mut skewness_validity = fixed_validity(rows, 30);
    let mut kurtosis_validity = fixed_validity(rows, 30);
    for row in 30..rows {
        let slice = &log_returns[(row - 29)..=row];
        let mean = slice.iter().sum::<f64>() / 30.0;
        let mut second = 0.0;
        let mut third = 0.0;
        let mut fourth = 0.0;
        for value in slice {
            let deviation = value - mean;
            second += deviation * deviation;
            third += deviation * deviation * deviation;
            fourth += deviation * deviation * deviation * deviation;
        }
        second /= 30.0;
        third /= 30.0;
        fourth /= 30.0;
        let standard_deviation = second.sqrt();
        if standard_deviation <= EPS {
            skewness_validity[row] = FeatureCellValidity::ZeroDenominator;
            kurtosis_validity[row] = FeatureCellValidity::ZeroDenominator;
            continue;
        }
        skewness[row] = (third / standard_deviation.powi(3)).clamp(-10.0, 10.0);
        kurtosis[row] = (fourth / standard_deviation.powi(4) - 3.0).clamp(-10.0, 50.0);
    }
    replace_column(columns, "quant_skewness_30", skewness, skewness_validity)?;
    replace_column(columns, "quant_kurtosis_30", kurtosis, kurtosis_validity)?;

    let mut fractal_dimension = vec![1.5; rows];
    for row in 30..rows {
        let slice = &bars.close[(row - 30)..=row];
        let maximum = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let minimum = slice.iter().copied().fold(f64::INFINITY, f64::min);
        if maximum - minimum > 1e-10 {
            let mut sign_changes = 0_usize;
            for offset in 2..slice.len() {
                let current = slice[offset] - slice[offset - 1];
                let previous = slice[offset - 1] - slice[offset - 2];
                if current * previous < 0.0 {
                    sign_changes += 1;
                }
            }
            let points = slice.len() as f64;
            let sign_changes = sign_changes as f64;
            let numerator = exact_log(points)?;
            let denominator = numerator + exact_log(points / (points + 0.4 * sign_changes))?;
            fractal_dimension[row] = (numerator / denominator).clamp(1.0, 2.0);
        }
    }
    replace_column(
        columns,
        "quant_fractal_dim",
        fractal_dimension,
        cloned_validity(columns, "quant_fractal_dim")?,
    )?;
    Ok(())
}

fn install_temporal_migrations(bars: &Ohlcv, columns: &mut [FeatureColumnF64]) -> OracleResult<()> {
    const NAMES: [&str; 14] = [
        "quant_prev_day_h_dist",
        "quant_prev_day_l_dist",
        "quant_prev_week_h_dist",
        "quant_prev_week_l_dist",
        "quant_orb_4",
        "quant_orb_8",
        "quant_orb_12",
        "quant_pivot_dist",
        "quant_r1_dist",
        "quant_r2_dist",
        "quant_s1_dist",
        "quant_s2_dist",
        "quant_cam_r3_dist",
        "quant_cam_s3_dist",
    ];
    let rows = bars.len();
    let timestamps = bars.timestamp.as_deref().expect("validated timestamps");
    let mut values: [Vec<f64>; 14] = std::array::from_fn(|_| vec![0.0; rows]);
    let mut validity: [Vec<FeatureCellValidity>; 14] =
        std::array::from_fn(|_| vec![FeatureCellValidity::Warmup; rows]);

    let mut completed_days: [Option<CompletedUtcDay>; 5] = [None; 5];
    let mut completed_day_count = 0_usize;
    let mut current_day_key = timestamps[0].div_euclid(UTC_DAY_MILLIS);
    let mut current_day = CompletedUtcDay {
        high: f64::NEG_INFINITY,
        low: f64::INFINITY,
        close: bars.close[0],
    };
    let mut orb_count = 0_usize;
    let mut orb_high = [f64::NEG_INFINITY; 3];
    let mut orb_low = [f64::INFINITY; 3];
    let orb_thresholds = [4_usize, 8, 12];

    for row in 0..rows {
        let day_key = timestamps[row].div_euclid(UTC_DAY_MILLIS);
        if day_key != current_day_key {
            if completed_day_count < completed_days.len() {
                completed_days[completed_day_count] = Some(current_day);
                completed_day_count += 1;
            } else {
                completed_days.rotate_left(1);
                completed_days[completed_days.len() - 1] = Some(current_day);
            }
            current_day_key = day_key;
            current_day = CompletedUtcDay {
                high: f64::NEG_INFINITY,
                low: f64::INFINITY,
                close: bars.close[row],
            };
            orb_count = 0;
            orb_high.fill(f64::NEG_INFINITY);
            orb_low.fill(f64::INFINITY);
        }

        if completed_day_count > 0 {
            let previous = completed_days[(completed_day_count - 1).min(4)]
                .expect("completed-day ring is dense");
            let previous_range = previous.high - previous.low;
            for index in [0_usize, 1] {
                validity[index][row] = if previous_range <= EPS {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    FeatureCellValidity::Valid
                };
            }
            let denominator = previous_range.max(1e-10);
            values[0][row] = (bars.close[row] - previous.high) / denominator;
            values[1][row] = (bars.close[row] - previous.low) / denominator;

            let pivot = (previous.high + previous.low + previous.close) / 3.0;
            let r1 = 2.0 * pivot - previous.low;
            let r2 = pivot + previous_range;
            let s1 = 2.0 * pivot - previous.high;
            let s2 = pivot - previous_range;
            let cam_r3 = previous.close + previous_range * 1.1 / 4.0;
            let cam_s3 = previous.close - previous_range * 1.1 / 4.0;
            let current_range = bars.high[row] - bars.low[row];
            let current_denominator = current_range.max(1e-10);
            for index in 7..14 {
                validity[index][row] = if current_range <= EPS {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    FeatureCellValidity::Valid
                };
            }
            for (slot, level) in [pivot, r1, r2, s1, s2, cam_r3, cam_s3]
                .into_iter()
                .enumerate()
            {
                values[7 + slot][row] = (bars.close[row] - level) / current_denominator;
            }
        }

        if completed_day_count == completed_days.len() {
            let mut week_high = f64::NEG_INFINITY;
            let mut week_low = f64::INFINITY;
            for day in completed_days.into_iter().flatten() {
                week_high = week_high.max(day.high);
                week_low = week_low.min(day.low);
            }
            let week_range = week_high - week_low;
            for index in [2_usize, 3] {
                validity[index][row] = if week_range <= EPS {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    FeatureCellValidity::Valid
                };
            }
            let denominator = week_range.max(1e-10);
            values[2][row] = (bars.close[row] - week_high) / denominator;
            values[3][row] = (bars.close[row] - week_low) / denominator;
        }

        for (slot, threshold) in orb_thresholds.into_iter().enumerate() {
            if orb_count >= threshold {
                validity[4 + slot][row] = FeatureCellValidity::Valid;
                values[4 + slot][row] = if bars.close[row] > orb_high[slot] {
                    1.0
                } else if bars.close[row] < orb_low[slot] {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        let millis_in_day = timestamps[row].rem_euclid(UTC_DAY_MILLIS);
        if millis_in_day < ASIAN_SESSION_MILLIS {
            for (slot, threshold) in orb_thresholds.into_iter().enumerate() {
                if orb_count < threshold {
                    orb_high[slot] = orb_high[slot].max(bars.high[row]);
                    orb_low[slot] = orb_low[slot].min(bars.low[row]);
                }
            }
            orb_count += 1;
        }

        current_day.high = current_day.high.max(bars.high[row]);
        current_day.low = current_day.low.min(bars.low[row]);
        current_day.close = bars.close[row];
    }

    for index in 0..NAMES.len() {
        replace_column(
            columns,
            NAMES[index],
            std::mem::take(&mut values[index]),
            std::mem::take(&mut validity[index]),
        )?;
    }
    Ok(())
}

fn typed_quant_v3_oracle(
    bars: &Ohlcv,
    timeframe_millis: i64,
) -> OracleResult<Vec<FeatureColumnF64>> {
    let (_, bars_per_day, _) = validate_grid(bars, timeframe_millis)?;
    let mut columns = compute_quant_feature_columns_f64(bars).map_err(|error| error.to_string())?;
    install_exact_log_migrations(bars, bars_per_day, &mut columns)?;
    install_temporal_migrations(bars, &mut columns)?;
    let names: Vec<&str> = columns.iter().map(|column| column.name.as_str()).collect();
    if names != RESIDENT_QUANT_COLUMN_NAMES_V3 {
        return Err("Quant-v3 oracle schema order drifted".into());
    }
    Ok(columns)
}

fn assert_bitwise_preserved_v2_routes(v2: &[FeatureColumnF64], v3: &[FeatureColumnF64]) {
    let mut compared = 0;
    for (index, route) in RESIDENT_QUANT_ROUTE_CENSUS_V3.iter().enumerate() {
        if route.lineage != ResidentQuantRouteLineageV3::V2BitwisePreserved {
            continue;
        }
        compared += 1;
        assert_eq!(v2[index].name, v3[index].name);
        assert_eq!(
            v2[index].validity, v3[index].validity,
            "{} validity",
            route.name
        );
        assert_eq!(
            v2[index]
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            v3[index]
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} value bits",
            route.name
        );
    }
    assert_eq!(compared, 31);
}

fn assert_migrated_v3_routes(columns: &[FeatureColumnF64]) {
    let mut compared = 0;
    for (index, route) in RESIDENT_QUANT_ROUTE_CENSUS_V3.iter().enumerate() {
        if route.lineage == ResidentQuantRouteLineageV3::V2BitwisePreserved {
            continue;
        }
        compared += 1;
        assert!(
            columns[index]
                .validity
                .iter()
                .all(|validity| *validity != FeatureCellValidity::MissingInput),
            "{} retained v2 MissingInput under v3",
            route.name
        );
        for (value, validity) in columns[index].values.iter().zip(&columns[index].validity) {
            if validity.is_valid() {
                assert!(value.is_finite(), "{} valid value", route.name);
            } else {
                assert_eq!(
                    value.to_bits(),
                    0x7ff8_0000_0000_0000,
                    "{} invalid value is not the canonical quiet NaN",
                    route.name
                );
            }
        }
    }
    assert_eq!(compared, 32);
}

fn column<'a>(columns: &'a [FeatureColumnF64], name: &str) -> &'a FeatureColumnF64 {
    columns
        .iter()
        .find(|column| column.name == name)
        .unwrap_or_else(|| panic!("missing Quant-v3 oracle column {name}"))
}

#[test]
fn production_quant_v3_cpu_authority_matches_the_frozen_oracle() {
    for (fixture_name, timeframe, bars) in [
        (
            "ordinary_m1",
            CanonicalTimeframe::M1,
            fixture_with_rows_at_timeframe(1_600, 60_000),
        ),
        (
            "ordinary_m30",
            CanonicalTimeframe::M30,
            ordinary_m30_fixture(),
        ),
        (
            "exact_grid_gap",
            CanonicalTimeframe::M30,
            exact_grid_gap_fixture(),
        ),
        (
            "positive_subfloor",
            CanonicalTimeframe::M30,
            positive_subfloor_fixture(),
        ),
        (
            "flat_close_transition",
            CanonicalTimeframe::M30,
            flat_close_transition_fixture(),
        ),
    ] {
        let timeframe_millis = timeframe
            .fixed_duration_ms()
            .expect("admitted parity fixtures use fixed intraday timeframes");
        let expected =
            typed_quant_v3_oracle(&bars, timeframe_millis).expect("frozen typed Quant-v3 oracle");
        let actual = compute_quant_feature_columns_v3_f64(&bars, timeframe)
            .expect("production typed Quant-v3 CPU authority");
        assert_eq!(actual.len(), expected.len(), "{fixture_name} width");
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.name, expected.name, "{fixture_name} name");
            assert_eq!(
                actual.validity, expected.validity,
                "{fixture_name}:{} validity",
                actual.name
            );
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{fixture_name}:{} value bits",
                actual.name
            );
        }
    }
}

#[test]
fn production_quant_v3_rejects_nonresident_timeframes() {
    let bars = ordinary_m30_fixture();
    for timeframe in [
        CanonicalTimeframe::H1,
        CanonicalTimeframe::H4,
        CanonicalTimeframe::H12,
        CanonicalTimeframe::D1,
        CanonicalTimeframe::W1,
        CanonicalTimeframe::MN1,
    ] {
        let error = compute_quant_feature_columns_v3_f64(&bars, timeframe)
            .expect_err("nonresident Quant-v3 timeframe must fail closed");
        assert!(
            error.to_string().contains("Quant-v3")
                || error.to_string().contains("resident Quant-v3"),
            "{timeframe}: {error}"
        );
    }
}

#[test]
fn ordinary_m30_fixture_preserves_31_routes_and_defines_all_32_migrations() {
    let bars = ordinary_m30_fixture();
    let v2 = compute_quant_feature_columns_f64(&bars).expect("valid current Quant-v2 oracle");
    let v3 = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("valid typed Quant-v3 oracle");
    assert_bitwise_preserved_v2_routes(&v2, &v3);
    assert_migrated_v3_routes(&v3);
}

#[test]
fn exact_grid_gap_fixture_remains_typed_and_deterministic() {
    let bars = exact_grid_gap_fixture();
    let first = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("exact-grid gaps are admitted");
    let second = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("repeat oracle");
    for (left, right) in first.iter().zip(&second) {
        assert_eq!(left.name, right.name);
        assert_eq!(left.validity, right.validity);
        assert_eq!(
            left.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn nonfinite_input_fails_closed() {
    let mut bars = ordinary_m30_fixture();
    bars.high[17] = f64::NAN;
    let error = typed_quant_v3_oracle(&bars, M30_MILLIS)
        .expect_err("nonfinite Quant input must fail before a native launch");
    assert!(error.contains("non-finite"));
}

#[test]
fn positive_prices_below_the_legacy_floor_keep_zero_values_without_widening_validity() {
    let bars = positive_subfloor_fixture();
    let columns = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("positive sub-floor fixture");
    let log_return = column(&columns, "quant_log_return");
    assert!(
        log_return.values[1..]
            .iter()
            .all(|value| value.to_bits() == 0.0_f64.to_bits())
    );
    assert!(
        log_return.validity[1..]
            .iter()
            .all(|validity| validity.is_valid())
    );
    let log_range = column(&columns, "quant_log_volatility");
    assert!(
        log_range
            .values
            .iter()
            .all(|value| value.to_bits() == 0.0_f64.to_bits())
    );
    assert!(
        log_range
            .validity
            .iter()
            .all(|validity| validity.is_valid())
    );
}

#[test]
fn cum_delta_zscore_preserves_floor_aware_values_and_unfloored_validity() {
    let bars = positive_subfloor_fixture();
    let legacy_columns =
        compute_quant_feature_columns_f64(&bars).expect("current Quant-v2 authority");
    let migrated_columns = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("typed Quant-v3 oracle");
    let legacy = column(&legacy_columns, "quant_cum_delta_zscore");
    let migrated = column(&migrated_columns, "quant_cum_delta_zscore");

    for row in 50..bars.len() {
        assert!(legacy.values[row].to_bits() == 0.0_f64.to_bits());
        assert!(legacy.validity[row] == FeatureCellValidity::Valid);
        assert_eq!(migrated.values[row].to_bits(), legacy.values[row].to_bits());
        assert_eq!(migrated.validity[row], legacy.validity[row]);
    }

    let ordinary = typed_quant_v3_oracle(&ordinary_m30_fixture(), M30_MILLIS)
        .expect("ordinary typed Quant-v3 oracle");
    let ordinary_cumulative_delta = column(&ordinary, "quant_cum_delta_zscore");
    assert_eq!(
        ordinary_cumulative_delta.validity[50],
        FeatureCellValidity::Valid
    );
    assert!(ordinary_cumulative_delta.values[50].is_finite());
}

#[test]
fn vol_ratio_boundary_uses_exact_migrated_log_validity() {
    let bars = positive_subfloor_fixture();
    let legacy = compute_quant_feature_columns_f64(&bars).expect("legacy-validity authority");
    assert!(
        column(&legacy, "quant_vol_ratio").validity[20..]
            .iter()
            .all(|validity| validity.is_valid()),
        "the fixture must expose the legacy/exact-log validity boundary"
    );

    let migrated = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("typed Quant-v3 oracle");
    let subfloor_ratio = column(&migrated, "quant_vol_ratio");
    assert!(
        subfloor_ratio.validity[..20]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::Warmup)
    );
    assert!(
        subfloor_ratio.validity[20..]
            .iter()
            .all(|validity| *validity == FeatureCellValidity::ZeroDenominator)
    );
    assert!(
        subfloor_ratio
            .values
            .iter()
            .all(|value| value.to_bits() == 0x7ff8_0000_0000_0000)
    );

    let ordinary = typed_quant_v3_oracle(&ordinary_m30_fixture(), M30_MILLIS)
        .expect("ordinary typed Quant-v3 oracle");
    let ordinary_ratio = column(&ordinary, "quant_vol_ratio");
    assert_eq!(ordinary_ratio.validity[20], FeatureCellValidity::Valid);
    assert!(ordinary_ratio.values[20].is_finite());
}

#[test]
fn autocorrelation_boundaries_use_exact_migrated_log_validity() {
    let bars = positive_subfloor_fixture();
    let legacy = compute_quant_feature_columns_f64(&bars).expect("legacy-validity authority");
    let migrated = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("typed Quant-v3 oracle");
    let ordinary = typed_quant_v3_oracle(&ordinary_m30_fixture(), M30_MILLIS)
        .expect("ordinary typed Quant-v3 oracle");

    for (name, warmup) in [
        ("quant_autocorr_1", 51_usize),
        ("quant_autocorr_5", 55),
        ("quant_autocorr_10", 60),
    ] {
        assert!(
            column(&legacy, name).validity[warmup..]
                .iter()
                .all(|validity| validity.is_valid()),
            "{name} fixture must expose the legacy/exact-log validity boundary"
        );
        let subfloor = column(&migrated, name);
        assert!(
            subfloor.validity[..warmup]
                .iter()
                .all(|validity| *validity == FeatureCellValidity::Warmup)
        );
        assert!(
            subfloor.validity[warmup..]
                .iter()
                .all(|validity| *validity == FeatureCellValidity::ZeroDenominator)
        );
        assert!(
            subfloor
                .values
                .iter()
                .all(|value| value.to_bits() == 0x7ff8_0000_0000_0000)
        );

        let ordinary_column = column(&ordinary, name);
        assert_eq!(ordinary_column.validity[warmup], FeatureCellValidity::Valid);
        assert!(ordinary_column.values[warmup].is_finite());
    }
}

#[test]
fn remaining_exact_log_derived_validities_use_exact_migrated_intermediates() {
    let bars = positive_subfloor_fixture();
    let legacy = compute_quant_feature_columns_f64(&bars).expect("legacy-validity authority");
    let migrated = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("typed Quant-v3 oracle");
    let ordinary = typed_quant_v3_oracle(&ordinary_m30_fixture(), M30_MILLIS)
        .expect("ordinary typed Quant-v3 oracle");

    for (name, warmup) in [
        ("quant_hurst_100", 100_usize),
        ("quant_skewness_30", 30),
        ("quant_kurtosis_30", 30),
    ] {
        assert!(
            column(&legacy, name).validity[warmup..]
                .iter()
                .all(|validity| validity.is_valid()),
            "{name} fixture must expose the legacy/exact-log validity boundary"
        );
        let subfloor = column(&migrated, name);
        assert!(
            subfloor.validity[..warmup]
                .iter()
                .all(|validity| *validity == FeatureCellValidity::Warmup)
        );
        assert!(
            subfloor.validity[warmup..]
                .iter()
                .all(|validity| *validity == FeatureCellValidity::ZeroDenominator)
        );
        assert!(
            subfloor
                .values
                .iter()
                .all(|value| value.to_bits() == 0x7ff8_0000_0000_0000)
        );

        let ordinary_column = column(&ordinary, name);
        assert_eq!(ordinary_column.validity[warmup], FeatureCellValidity::Valid);
        assert!(ordinary_column.values[warmup].is_finite());
    }
}

#[test]
fn kyle_lambda_keeps_rust_signum_value_denominator_and_zero_delta_validity() {
    let bars = flat_close_transition_fixture();
    let v2 = compute_quant_feature_columns_f64(&bars).expect("flat-close current Quant-v2 oracle");
    let v3 = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("flat-close Quant-v3 oracle");
    assert_bitwise_preserved_v2_routes(&v2, &v3);

    let row = 39;
    let volume = bars.volume.as_deref().expect("flat fixture volume");
    let mut numerator = 0.0;
    let mut value_denominator = 0.0;
    let mut validity_denominator = 0.0;
    for index in (row - 19)..=row {
        let delta = bars.close[index] - bars.close[index - 1];
        let signed_volume = delta.signum() * volume[index];
        numerator += delta * signed_volume;
        value_denominator += signed_volume * signed_volume;
        let validity_direction = if delta > 0.0 {
            1.0
        } else if delta < 0.0 {
            -1.0
        } else {
            0.0
        };
        validity_denominator += (validity_direction * volume[index]).powi(2);
    }
    assert!(value_denominator > validity_denominator);
    let kyle = column(&v3, "quant_kyle_lambda");
    assert_eq!(kyle.validity[row], FeatureCellValidity::Valid);
    assert_eq!(
        kyle.values[row].to_bits(),
        (numerator / value_denominator).to_bits()
    );
}

#[test]
fn utc_day_boundary_fixture_exposes_only_a_completed_prior_day() {
    let columns = typed_quant_v3_oracle(&fixture_with_rows(100), M30_MILLIS).expect("day fixture");
    let previous = column(&columns, "quant_prev_day_h_dist");
    assert_eq!(previous.validity[47], FeatureCellValidity::Warmup);
    assert_eq!(previous.validity[48], FeatureCellValidity::Valid);
}

#[test]
fn five_day_week_boundary_fixture_requires_five_completed_observed_trading_days() {
    let columns = typed_quant_v3_oracle(&fixture_with_rows(300), M30_MILLIS).expect("week fixture");
    let previous = column(&columns, "quant_prev_week_h_dist");
    assert_eq!(previous.validity[239], FeatureCellValidity::Warmup);
    assert_eq!(previous.validity[240], FeatureCellValidity::Valid);
}

#[test]
fn asian_session_orb_boundary_fixture_freezes_first_n_observed_bars_and_resets_daily() {
    let columns = typed_quant_v3_oracle(&fixture_with_rows(100), M30_MILLIS).expect("ORB fixture");
    for (name, first_valid) in [
        ("quant_orb_4", 4_usize),
        ("quant_orb_8", 8),
        ("quant_orb_12", 12),
    ] {
        let orb = column(&columns, name);
        assert_eq!(orb.validity[first_valid - 1], FeatureCellValidity::Warmup);
        assert_eq!(orb.validity[first_valid], FeatureCellValidity::Valid);
        assert_eq!(orb.validity[47], FeatureCellValidity::Valid);
        assert_eq!(orb.validity[48], FeatureCellValidity::Warmup);
    }
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
#[test]
fn rtx_device_fixture_matches_all_quant_v3_value_bits_and_validity_codes() {
    use neoethos_gpu_cuda::resident_quant_v3::run_resident_quant_v3_device_fixture;

    let mut mismatch_census = Vec::new();
    for (fixture_name, bars) in [
        ("ordinary_m30", ordinary_m30_fixture()),
        ("exact_grid_gap", exact_grid_gap_fixture()),
        ("positive_subfloor", positive_subfloor_fixture()),
        ("flat_close_transition", flat_close_transition_fixture()),
    ] {
        let expected = typed_quant_v3_oracle(&bars, M30_MILLIS).expect("typed CPU oracle");
        let actual = run_resident_quant_v3_device_fixture(
            &bars.open,
            &bars.high,
            &bars.low,
            &bars.close,
            bars.volume.as_deref().expect("fixture volume"),
            bars.timestamp.as_deref().expect("fixture timestamps"),
            M30_MILLIS as u64,
        )
        .expect("resident Quant-v3 test-only parity D2H fixture");
        for (index, expected_column) in expected.iter().enumerate() {
            let offset = index * bars.len();
            let mut value_mismatches = Vec::new();
            let mut validity_mismatches = Vec::new();
            for row in 0..bars.len() {
                let expected_bits = expected_column.values[row].to_bits();
                let actual_bits = actual.values[offset + row].to_bits();
                if expected_bits != actual_bits {
                    value_mismatches.push((row, expected_bits, actual_bits));
                }

                let expected_code = expected_column.validity[row].code();
                let actual_code = actual.validity_u8[offset + row];
                if expected_code != actual_code {
                    validity_mismatches.push((row, expected_code, actual_code));
                }
            }

            if let (Some(first), Some(last)) = (value_mismatches.first(), value_mismatches.last()) {
                mismatch_census.push(format!(
                    "fixture={fixture_name};route={};kind=value_bits;count={};first_row={};last_row={};first_expected={:#018x};first_actual={:#018x}",
                    expected_column.name,
                    value_mismatches.len(),
                    first.0,
                    last.0,
                    first.1,
                    first.2,
                ));
            }
            if let (Some(first), Some(last)) =
                (validity_mismatches.first(), validity_mismatches.last())
            {
                mismatch_census.push(format!(
                    "fixture={fixture_name};route={};kind=validity_code;count={};first_row={};last_row={};first_expected={};first_actual={}",
                    expected_column.name,
                    validity_mismatches.len(),
                    first.0,
                    last.0,
                    first.1,
                    first.2,
                ));
            }
        }
    }

    assert!(
        mismatch_census.is_empty(),
        "resident Quant-v3 device parity mismatch census ({} entries):\n{}",
        mismatch_census.len(),
        mismatch_census.join("\n")
    );
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
fn integrated_m30_parent_fixture(base: &Ohlcv) -> Ohlcv {
    let timestamps = base.timestamp.as_deref().expect("base timestamps");
    let volume = base.volume.as_deref().expect("base volume");
    assert_eq!(
        base.len() % 2,
        0,
        "M15 fixture must aggregate exactly to M30"
    );
    let rows = base.len() / 2;
    let mut parent = Ohlcv {
        timestamp: Some(Vec::with_capacity(rows)),
        open: Vec::with_capacity(rows),
        high: Vec::with_capacity(rows),
        low: Vec::with_capacity(rows),
        close: Vec::with_capacity(rows),
        volume: Some(Vec::with_capacity(rows)),
    };
    for row in 0..rows {
        let first = row * 2;
        let second = first + 1;
        parent
            .timestamp
            .as_mut()
            .expect("parent timestamps")
            .push(timestamps[first]);
        parent.open.push(base.open[first]);
        parent.high.push(base.high[first].max(base.high[second]));
        parent.low.push(base.low[first].min(base.low[second]));
        parent.close.push(base.close[second]);
        parent
            .volume
            .as_mut()
            .expect("parent volume")
            .push(volume[first] + volume[second]);
    }
    parent
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
fn integrated_cpu_columns(
    base: &Ohlcv,
    parent: &Ohlcv,
    budget_rows: usize,
) -> Vec<FeatureColumnF64> {
    use neoethos_data::core::features::align_feature_columns_by_ms;

    fn local_columns(bars: &Ohlcv, timeframe_ms: i64, budget_rows: usize) -> Vec<FeatureColumnF64> {
        use neoethos_data::core::hpc_ta::compute_classic_ta_gpu_exact_parity_feature_columns_for_device_fixture_v3;
        use neoethos_data::{
            compute_footprint_feature_columns_f64, compute_regime_feature_columns_f64,
            compute_session_feature_columns_f64, compute_smc_feature_columns_f64,
        };

        let mut columns = compute_smc_feature_columns_f64(bars).expect("SMC CPU oracle");
        columns.extend(
            compute_classic_ta_gpu_exact_parity_feature_columns_for_device_fixture_v3(
                bars,
                budget_rows,
            )
            .expect("Classic exact-parity-subset-v3 CPU oracle"),
        );
        columns.extend(typed_quant_v3_oracle(bars, timeframe_ms).expect("Quant-v3 CPU oracle"));
        columns.extend(compute_session_feature_columns_f64(bars).expect("Session CPU oracle"));
        columns.extend(compute_regime_feature_columns_f64(bars).expect("Regime CPU oracle"));
        columns.extend(compute_footprint_feature_columns_f64(bars).expect("Footprint CPU oracle"));
        columns
    }

    let mut expected = local_columns(base, M15_MILLIS, budget_rows);
    let parent_columns = local_columns(parent, M30_MILLIS, budget_rows);
    let mut aligned = align_feature_columns_by_ms(
        base.timestamp.as_deref().expect("base timestamps"),
        parent.timestamp.as_deref().expect("parent timestamps"),
        &parent_columns,
        true,
        Some(4 * M15_MILLIS),
        M30_MILLIS,
    )
    .expect("M30 CPU causal alignment oracle");
    for column in &mut aligned {
        column.name = format!("M30_{}", column.name);
    }
    expected.extend(aligned);
    expected
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
#[test]
fn integrated_resident_feature_store_v3_matches_all_route_value_bits_and_validity_codes() {
    use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
    use neoethos_data::core::features::FeatureProfile;
    use neoethos_data::{
        BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
        CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef,
        SelectedDatasetGenerationV1, install_data_runtime_overrides,
        materialize_gpu_only_feature_store_v3, pin_exact_canonical_series_v1,
        preflight_gpu_only_feature_workspace_v3, publish_canonical_ohlcv_generation,
    };
    use neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureProducerV3;
    use neoethos_gpu_cuda::acquire_discovery_run_device_admission_v1;
    use neoethos_gpu_cuda::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_v1;

    install_data_runtime_overrides(false);
    let root = tempfile::tempdir().expect("integrated canonical fixture root");
    let base = fixture_with_rows_at_timeframe(1_200, M15_MILLIS);
    let parent = integrated_m30_parent_fixture(&base);
    let namespace = "neoethos-integrated-resident-v3";
    let publish = |timeframe: CanonicalTimeframe, bars: &Ohlcv| {
        let identity = CanonicalDatasetIdentity::external(
            namespace,
            "EURUSD",
            timeframe,
            BarTimestampConvention::BarOpen,
        )
        .expect("integrated fixture identity");
        let provenance = ProducerProvenanceEnvelopeV1::new(
            "neoethos.integrated-resident-fixture.v3",
            format!("{namespace}:{timeframe}").into_bytes(),
        )
        .expect("integrated fixture provenance");
        let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            provenance: &provenance,
            ohlcv: bars,
            volume: CanonicalVolumeRef::Float64(
                bars.volume.as_deref().expect("integrated fixture volume"),
            ),
            rows_per_chunk: 256,
        })
        .expect("publish integrated canonical fixture");
        SelectedDatasetGenerationV1::from_manifest(published.manifest())
            .expect("select integrated generation")
    };
    let base_generation = publish(CanonicalTimeframe::M15, &base);
    let parent_generation = publish(CanonicalTimeframe::M30, &parent);
    let series = CanonicalDatasetSeriesReceiptV1::new(
        base_generation.clone(),
        vec![base_generation, parent_generation],
    )
    .expect("seal integrated direct-timeframe series");
    let pinned = pin_exact_canonical_series_v1(root.path(), series).expect("pin integrated series");
    let workspace = preflight_gpu_only_feature_workspace_v3(
        pinned,
        CanonicalTimeframe::M15,
        FeatureProfile::Standard,
        base.len(),
    )
    .expect("preflight integrated ten-producer workspace");
    assert_eq!(
        workspace.producer_capability_count(),
        ResidentFeatureProducerV3::ALL.len()
    );

    let admitted = seal_test_full_discovery_run_v1(
        acquire_discovery_run_device_admission_v1().expect("acquire exact CUDA run"),
        2 * 1024 * 1024 * 1024,
        512 * 1024 * 1024,
    )
    .expect("seal integrated full-Discovery run");
    let store = materialize_gpu_only_feature_store_v3(workspace, admitted)
        .expect("materialize integrated ten-producer resident store");
    let names = store
        .ordered_feature_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let actual = store
        .copy_bar_major_for_device_fixture_v3()
        .expect("bounded fixture-only final bar-major readback");
    let expected = integrated_cpu_columns(&base, &parent, base.len());
    assert_eq!(actual.rows, base.len());
    assert_eq!(actual.columns, expected.len());
    assert_eq!(
        names,
        expected
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>(),
        "integrated route order"
    );

    let mut mismatch_census = Vec::new();
    for (column, expected_column) in expected.iter().enumerate() {
        let mut value_count = 0_usize;
        let mut validity_count = 0_usize;
        let mut first_value = None;
        let mut first_validity = None;
        let mut last_value = 0_usize;
        let mut last_validity = 0_usize;
        for row in 0..base.len() {
            let cell = row * actual.columns + column;
            let expected_bits = expected_column.values[row].to_bits();
            let actual_bits = actual.values[cell].to_bits();
            if expected_bits != actual_bits {
                first_value.get_or_insert((row, expected_bits, actual_bits));
                last_value = row;
                value_count += 1;
            }
            let expected_code = expected_column.validity[row].code();
            let actual_code = actual.validity_u8[cell];
            if expected_code != actual_code {
                first_validity.get_or_insert((row, expected_code, actual_code));
                last_validity = row;
                validity_count += 1;
            }
        }
        if let Some((row, expected_bits, actual_bits)) = first_value {
            mismatch_census.push(format!(
                "route={};kind=value_bits;count={value_count};first_row={row};last_row={last_value};first_expected={expected_bits:#018x};first_actual={actual_bits:#018x}",
                expected_column.name
            ));
        }
        if let Some((row, expected_code, actual_code)) = first_validity {
            mismatch_census.push(format!(
                "route={};kind=validity_code;count={validity_count};first_row={row};last_row={last_validity};first_expected={expected_code};first_actual={actual_code}",
                expected_column.name
            ));
        }
    }
    assert!(
        mismatch_census.is_empty(),
        "integrated mismatch census ({} entries):\n{}",
        mismatch_census.len(),
        mismatch_census.join("\n")
    );
    eprintln!(
        "integrated_resident_feature_store_v3_parity=GREEN;rows={};columns={};producers={};value_bit_mismatches=0;validity_mismatches=0;classic_mode=gpu_only_exact_parity_subset_v3",
        actual.rows,
        actual.columns,
        ResidentFeatureProducerV3::ALL.len(),
    );
}

#[cfg(feature = "gpu-cuda-device-fixtures")]
#[test]
#[ignore = "required-card nsys performance receipt"]
fn rtx_device_fixture_profiles_the_single_lane_bounded_schedule() {
    use neoethos_gpu_cuda::resident_quant_v3::run_resident_quant_v3_device_perf_fixture;

    let rows = std::env::var("NEOETHOS_QUANT_PERF_ROWS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("positive Quant perf row count")
        })
        .unwrap_or(4_096);
    let repetitions = std::env::var("NEOETHOS_QUANT_PERF_REPETITIONS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("positive Quant perf repetition count")
        })
        .unwrap_or(33);
    assert!(rows >= 600, "Quant perf fixture requires at least 600 rows");
    let bars = fixture_with_rows(rows);
    run_resident_quant_v3_device_perf_fixture(
        &bars.open,
        &bars.high,
        &bars.low,
        &bars.close,
        bars.volume.as_deref().expect("fixture volume"),
        bars.timestamp.as_deref().expect("fixture timestamps"),
        M30_MILLIS as u64,
        repetitions,
    )
    .expect("resident Quant-v3 required-card performance fixture");
    eprintln!(
        "resident_quant_v3_perf_rows={rows};repetitions={repetitions};kernel_schedule=single-lane-fixed-order;feature_columns=63;feature_d2h_bytes=0"
    );
}
