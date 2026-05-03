/// Session-Level OHLC Features
///
/// Tracks individual trading session statistics (Asian, London, New York)
/// and provides key institutional reference levels like session VWAP,
/// session open gaps, and session range positions.
use super::super::Ohlcv;
use crate::core::timestamps::{TimestampUnit, infer_timestamp_unit, timestamp_to_millis};
use chrono::{TimeZone, Timelike, Utc};

#[derive(Default, Clone)]
#[allow(dead_code)]
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
