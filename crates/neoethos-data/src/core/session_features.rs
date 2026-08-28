/// Session-Level OHLC Features
///
/// Tracks individual trading session statistics (Asian, London, New York)
/// and provides key institutional reference levels like session VWAP,
/// session open gaps, and session range positions.
use super::super::Ohlcv;
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::timestamps::{
    TimestampUnit, infer_timestamp_unit, timestamp_to_millis,
    validate_canonical_millisecond_timestamps,
};
use anyhow::{Result, ensure};
use chrono::{TimeZone, Timelike, Utc};

#[derive(Default, Clone)]
struct SessionAccum {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    vol_sum: f64,
    vwap_num: f64,
    vwap_den: f64,
    bar_count: usize,
    started: bool,
}

impl SessionAccum {
    fn reset(&mut self, open_price: f64) {
        self.open = open_price;
        self.high = open_price;
        self.low = open_price;
        self.close = open_price;
        self.vol_sum = 0.0;
        self.vwap_num = 0.0;
        self.vwap_den = 0.0;
        self.bar_count = 0;
        self.started = true;
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) {
        if high > self.high {
            self.high = high;
        }
        if low < self.low {
            self.low = low;
        }
        self.close = close;
        self.vol_sum += volume;
        let typical = (high + low + close) / 3.0;
        self.vwap_num += typical * volume;
        self.vwap_den += volume;
        self.bar_count += 1;
    }

    fn vwap(&self) -> f64 {
        if self.vwap_den > 1e-10 {
            self.vwap_num / self.vwap_den
        } else {
            self.close
        }
    }

    fn range(&self) -> f64 {
        self.high - self.low
    }

    fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Compute session-level institutional reference features.
pub fn compute_session_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    if n == 0 {
        return vec![];
    }

    // Session features
    let mut london_open_dist = vec![0.0_f64; n];
    let mut london_high_dist = vec![0.0_f64; n];
    let mut london_low_dist = vec![0.0_f64; n];
    let mut london_range = vec![0.0_f64; n];
    let mut london_vwap_dist = vec![0.0_f64; n];

    let mut ny_open_dist = vec![0.0_f64; n];
    let mut ny_high_dist = vec![0.0_f64; n];
    let mut ny_low_dist = vec![0.0_f64; n];
    let mut ny_range = vec![0.0_f64; n];
    let mut ny_vwap_dist = vec![0.0_f64; n];

    let mut asian_open_dist = vec![0.0_f64; n];
    let mut asian_close_dist = vec![0.0_f64; n];
    let mut asian_range_norm = vec![0.0_f64; n];

    // Session overlap features
    let mut london_ny_overlap = vec![0.0_f64; n]; // Are we in the overlap zone?
    let mut session_volatility_ratio = vec![0.0_f64; n]; // Current session vol vs previous

    // Previous session levels (for gap/continuation analysis)
    let mut prev_session_close_dist = vec![0.0_f64; n];
    let mut session_open_gap = vec![0.0_f64; n];

    // Daily features
    let mut daily_range_pct = vec![0.0_f64; n];
    let mut daily_body_pct = vec![0.0_f64; n];
    let mut daily_position = vec![0.0_f64; n]; // Where in today's range? 0-1
    let mut daily_high_dist = vec![0.0_f64; n];
    let mut daily_low_dist = vec![0.0_f64; n];
    let mut daily_vwap_dist = vec![0.0_f64; n];

    let volume = ohlcv.volume.as_deref();
    let timestamp_unit = ohlcv
        .timestamp
        .as_deref()
        .and_then(infer_timestamp_unit)
        .unwrap_or(TimestampUnit::Milliseconds);

    let mut asian = SessionAccum::default();
    let mut london = SessionAccum::default();
    let mut ny = SessionAccum::default();
    let mut daily = SessionAccum::default();
    let mut prev_session_close = f64::NAN;
    let mut prev_asian_range = 0.0_f64;

    // Running ATR for normalization
    let mut atr_sum = 0.0_f64;
    let mut atr_count = 0_usize;

    for i in 0..n {
        let open = ohlcv.open[i];
        let high = ohlcv.high[i];
        let low = ohlcv.low[i];
        let close = ohlcv.close[i];
        let vol = volume.map(|v| v[i]).unwrap_or(1.0);

        // Running ATR
        if i > 0 {
            let tr = (high - low)
                .max((high - ohlcv.close[i - 1]).abs())
                .max((low - ohlcv.close[i - 1]).abs());
            atr_sum += tr;
            atr_count += 1;
        }
        let atr = if atr_count > 0 {
            atr_sum / atr_count as f64
        } else {
            (high - low).max(1e-10)
        };

        if let Some(raw_ts) = ohlcv.timestamp.as_ref().map(|t| t[i])
            && let Ok(ts_ms) = timestamp_to_millis(raw_ts, timestamp_unit)
            && let chrono::LocalResult::Single(dt) = Utc.timestamp_millis_opt(ts_ms)
        {
            let hour = dt.hour();
            let minute = dt.minute();

            // === Session Boundaries ===
            // Asian: 00:00-08:00 UTC
            // London: 07:00-16:00 UTC
            // NY: 12:00-21:00 UTC
            // Overlap: 12:00-16:00 UTC

            // Asian session
            if hour == 0 && minute == 0 {
                if asian.started {
                    prev_session_close = asian.close;
                    prev_asian_range = asian.range();
                }
                asian.reset(open);
            }
            if hour < 8 && asian.started {
                asian.update(high, low, close, vol);
            }

            // London session
            if hour == 7 && minute == 0 {
                if london.started {
                    prev_session_close = london.close;
                }
                london.reset(open);
                // Session open gap
                if prev_session_close.is_finite() {
                    session_open_gap[i] = (open - prev_session_close) / atr.max(1e-10);
                }
            }
            if (7..16).contains(&hour) && london.started {
                london.update(high, low, close, vol);
            }

            // NY session
            if hour == 12 && minute == 0 {
                if ny.started {
                    prev_session_close = ny.close;
                }
                ny.reset(open);
                if prev_session_close.is_finite() {
                    session_open_gap[i] = (open - prev_session_close) / atr.max(1e-10);
                }
            }
            if (12..21).contains(&hour) && ny.started {
                ny.update(high, low, close, vol);
            }

            // Daily
            if hour == 0 && minute == 0 {
                daily.reset(open);
            }
            daily.update(high, low, close, vol);

            // === Compute Feature Values ===

            // London distances
            if london.started && london.bar_count > 0 {
                london_open_dist[i] = (close - london.open) / atr.max(1e-10);
                london_high_dist[i] = (close - london.high) / atr.max(1e-10);
                london_low_dist[i] = (close - london.low) / atr.max(1e-10);
                london_range[i] = london.range() / atr.max(1e-10);
                london_vwap_dist[i] = (close - london.vwap()) / atr.max(1e-10);
            }

            // NY distances
            if ny.started && ny.bar_count > 0 {
                ny_open_dist[i] = (close - ny.open) / atr.max(1e-10);
                ny_high_dist[i] = (close - ny.high) / atr.max(1e-10);
                ny_low_dist[i] = (close - ny.low) / atr.max(1e-10);
                ny_range[i] = ny.range() / atr.max(1e-10);
                ny_vwap_dist[i] = (close - ny.vwap()) / atr.max(1e-10);
            }

            // Asian distances
            if asian.started && asian.bar_count > 0 {
                asian_open_dist[i] = (close - asian.open) / atr.max(1e-10);
                asian_close_dist[i] = (close - asian.close) / atr.max(1e-10);
                asian_range_norm[i] = asian.range() / atr.max(1e-10);
            }

            // London-NY overlap zone
            if (12..16).contains(&hour) {
                london_ny_overlap[i] = 1.0;
            }

            // Session volatility ratio
            if prev_asian_range > 1e-10 && london.started {
                session_volatility_ratio[i] = london.range() / prev_asian_range;
            }

            // Previous session close distance
            if prev_session_close.is_finite() {
                prev_session_close_dist[i] = (close - prev_session_close) / atr.max(1e-10);
            }

            // Daily features
            if daily.started && daily.bar_count > 0 {
                let dr = daily.range();
                daily_range_pct[i] = if close > 1e-10 { dr / close } else { 0.0 };
                daily_body_pct[i] = if close > 1e-10 {
                    daily.body() / close
                } else {
                    0.0
                };
                daily_position[i] = if dr > 1e-10 {
                    (close - daily.low) / dr
                } else {
                    0.5
                };
                daily_high_dist[i] = (close - daily.high) / atr.max(1e-10);
                daily_low_dist[i] = (close - daily.low) / atr.max(1e-10);
                daily_vwap_dist[i] = (close - daily.vwap()) / atr.max(1e-10);
            }
        }
    }

    vec![
        ("session_london_open_dist".to_string(), london_open_dist),
        ("session_london_high_dist".to_string(), london_high_dist),
        ("session_london_low_dist".to_string(), london_low_dist),
        ("session_london_range".to_string(), london_range),
        ("session_london_vwap_dist".to_string(), london_vwap_dist),
        ("session_ny_open_dist".to_string(), ny_open_dist),
        ("session_ny_high_dist".to_string(), ny_high_dist),
        ("session_ny_low_dist".to_string(), ny_low_dist),
        ("session_ny_range".to_string(), ny_range),
        ("session_ny_vwap_dist".to_string(), ny_vwap_dist),
        ("session_asian_open_dist".to_string(), asian_open_dist),
        ("session_asian_close_dist".to_string(), asian_close_dist),
        ("session_asian_range_norm".to_string(), asian_range_norm),
        ("session_london_ny_overlap".to_string(), london_ny_overlap),
        ("session_vol_ratio".to_string(), session_volatility_ratio),
        (
            "session_prev_close_dist".to_string(),
            prev_session_close_dist,
        ),
        ("session_open_gap".to_string(), session_open_gap),
        ("daily_range_pct".to_string(), daily_range_pct),
        ("daily_body_pct".to_string(), daily_body_pct),
        ("daily_position".to_string(), daily_position),
        ("daily_high_dist".to_string(), daily_high_dist),
        ("daily_low_dist".to_string(), daily_low_dist),
        ("daily_vwap_dist".to_string(), daily_vwap_dist),
    ]
}

/// Explicit-validity f64 session lane used by the atomic Tasks 5B-9
/// migration. The legacy value producer remains the temporary parity bridge;
/// this replay supplies the information that its prefilled zero vectors lost.
/// It accepts canonical i64 milliseconds only and never fabricates VWAP from
/// unit volume when broker volume is absent.
pub fn compute_session_feature_columns_f64(ohlcv: &Ohlcv) -> Result<Vec<FeatureColumnF64>> {
    let n = ohlcv.len();
    ensure!(n > 0, "session features require at least one OHLC row");
    ensure!(
        ohlcv.open.len() == n && ohlcv.high.len() == n && ohlcv.low.len() == n,
        "session OHLC lengths do not match close length {n}"
    );
    for row in 0..n {
        let open = ohlcv.open[row];
        let high = ohlcv.high[row];
        let low = ohlcv.low[row];
        let close = ohlcv.close[row];
        ensure!(
            open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite(),
            "session OHLC row {row} contains a non-finite price"
        );
        ensure!(
            open > 0.0 && high > 0.0 && low > 0.0 && close > 0.0,
            "session OHLC row {row} contains a non-positive price"
        );
        ensure!(
            low <= open.min(close) && high >= open.max(close),
            "session OHLC row {row} violates low <= open/close <= high"
        );
    }

    let volume = if let Some(volume) = ohlcv.volume.as_deref() {
        ensure!(
            volume.len() == n,
            "session volume length {} does not match OHLC length {n}",
            volume.len()
        );
        if let Some((row, value)) = volume
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            anyhow::bail!("session volume row {row} is invalid: {value}");
        }
        Some(volume)
    } else {
        None
    };

    let legacy = compute_session_feature_columns(ohlcv);
    let mut validity_by_name: std::collections::HashMap<String, Vec<FeatureCellValidity>> = legacy
        .iter()
        .map(|(name, _)| (name.clone(), vec![FeatureCellValidity::Warmup; n]))
        .collect();

    let Some(timestamps) = ohlcv.timestamp.as_deref() else {
        for validity in validity_by_name.values_mut() {
            validity.fill(FeatureCellValidity::MissingInput);
        }
        return legacy
            .into_iter()
            .map(|(name, values)| {
                let validity = validity_by_name
                    .remove(&name)
                    .expect("session validity row exists");
                FeatureColumnF64::new(name, values, validity)
            })
            .collect();
    };
    ensure!(
        timestamps.len() == n,
        "session timestamp length {} does not match OHLC length {n}",
        timestamps.len()
    );
    validate_canonical_millisecond_timestamps(timestamps)?;

    let mut asian = SessionAccum::default();
    let mut london = SessionAccum::default();
    let mut ny = SessionAccum::default();
    let mut daily = SessionAccum::default();
    let mut prev_session_close = f64::NAN;
    let mut prev_asian_range = 0.0_f64;
    let mut has_previous_asian = false;
    let mut atr_sum = 0.0_f64;
    let mut atr_count = 0_usize;

    for row in 0..n {
        let open = ohlcv.open[row];
        let high = ohlcv.high[row];
        let low = ohlcv.low[row];
        let close = ohlcv.close[row];
        let row_volume = volume.map_or(1.0, |values| values[row]);

        if row > 0 {
            let true_range = (high - low)
                .max((high - ohlcv.close[row - 1]).abs())
                .max((low - ohlcv.close[row - 1]).abs());
            atr_sum += true_range;
            atr_count += 1;
        }
        let atr = if atr_count > 0 {
            atr_sum / atr_count as f64
        } else {
            high - low
        };
        let atr_validity = if atr > 1e-10 {
            FeatureCellValidity::Valid
        } else {
            FeatureCellValidity::ZeroDenominator
        };

        let dt = Utc
            .timestamp_millis_opt(timestamps[row])
            .single()
            .ok_or_else(|| anyhow::anyhow!("session timestamp row {row} is not a UTC instant"))?;
        let hour = dt.hour();
        let minute = dt.minute();
        let asian_open = hour == 0 && minute == 0;
        let london_open = hour == 7 && minute == 0;
        let ny_open = hour == 12 && minute == 0;

        if asian_open {
            if asian.started {
                prev_session_close = asian.close;
                prev_asian_range = asian.range();
                has_previous_asian = true;
            }
            asian.reset(open);
        }
        if hour < 8 && asian.started {
            asian.update(high, low, close, row_volume);
        }
        if london_open {
            if london.started {
                prev_session_close = london.close;
            }
            london.reset(open);
        }
        if (7..16).contains(&hour) && london.started {
            london.update(high, low, close, row_volume);
        }
        if ny_open {
            if ny.started {
                prev_session_close = ny.close;
            }
            ny.reset(open);
        }
        if (12..21).contains(&hour) && ny.started {
            ny.update(high, low, close, row_volume);
        }
        if asian_open {
            daily.reset(open);
        }
        daily.update(high, low, close, row_volume);

        let mut mark = |name: &str, validity: FeatureCellValidity| {
            validity_by_name
                .get_mut(name)
                .expect("session validity plan covers every output")[row] = validity;
        };

        mark("session_london_ny_overlap", FeatureCellValidity::Valid);
        mark("session_open_gap", FeatureCellValidity::Valid);
        if (london_open || ny_open) && !prev_session_close.is_finite() {
            mark("session_open_gap", FeatureCellValidity::Warmup);
        } else if (london_open || ny_open) && atr_validity != FeatureCellValidity::Valid {
            mark("session_open_gap", atr_validity);
        }

        if london.started && london.bar_count > 0 {
            for name in [
                "session_london_open_dist",
                "session_london_high_dist",
                "session_london_low_dist",
                "session_london_range",
            ] {
                mark(name, atr_validity);
            }
            mark(
                "session_london_vwap_dist",
                if volume.is_none() {
                    FeatureCellValidity::MissingInput
                } else if london.vwap_den <= 1e-10 {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    atr_validity
                },
            );
        }
        if ny.started && ny.bar_count > 0 {
            for name in [
                "session_ny_open_dist",
                "session_ny_high_dist",
                "session_ny_low_dist",
                "session_ny_range",
            ] {
                mark(name, atr_validity);
            }
            mark(
                "session_ny_vwap_dist",
                if volume.is_none() {
                    FeatureCellValidity::MissingInput
                } else if ny.vwap_den <= 1e-10 {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    atr_validity
                },
            );
        }
        if asian.started && asian.bar_count > 0 {
            for name in [
                "session_asian_open_dist",
                "session_asian_close_dist",
                "session_asian_range_norm",
            ] {
                mark(name, atr_validity);
            }
        }

        if london.started && has_previous_asian {
            mark(
                "session_vol_ratio",
                if prev_asian_range > 1e-10 {
                    FeatureCellValidity::Valid
                } else {
                    FeatureCellValidity::ZeroDenominator
                },
            );
        }
        if prev_session_close.is_finite() {
            mark("session_prev_close_dist", atr_validity);
        }

        if daily.started && daily.bar_count > 0 {
            mark("daily_range_pct", FeatureCellValidity::Valid);
            mark("daily_body_pct", FeatureCellValidity::Valid);
            mark(
                "daily_position",
                if daily.range() > 1e-10 {
                    FeatureCellValidity::Valid
                } else {
                    FeatureCellValidity::ZeroDenominator
                },
            );
            mark("daily_high_dist", atr_validity);
            mark("daily_low_dist", atr_validity);
            mark(
                "daily_vwap_dist",
                if volume.is_none() {
                    FeatureCellValidity::MissingInput
                } else if daily.vwap_den <= 1e-10 {
                    FeatureCellValidity::ZeroDenominator
                } else {
                    atr_validity
                },
            );
        }
    }

    legacy
        .into_iter()
        .map(|(name, values)| {
            let validity = validity_by_name
                .remove(&name)
                .ok_or_else(|| anyhow::anyhow!("missing session validity plan for `{name}`"))?;
            FeatureColumnF64::new(name, values, validity)
        })
        .collect()
}
