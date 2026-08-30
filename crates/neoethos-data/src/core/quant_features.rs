/// Advanced Quantitative & Statistical Feature Generation
///
/// Institutional-grade statistical features for regime detection,
/// market microstructure analysis, and alpha generation.
use super::super::Ohlcv;
use super::features::{FeatureCellValidity, FeatureColumnF64};
#[cfg(feature = "gpu-cuda")]
use super::gpu_resident_quant_v3::RESIDENT_QUANT_COLUMN_NAMES_V3;
#[cfg(feature = "gpu-cuda")]
use super::gpu_resident_temporal_grid_v1::{
    ASIAN_SESSION_MILLIS_V2, UTC_DAY_MILLIS_V2, admit_fixed_intraday_grid_v1,
};
use super::quant_exact_math_v3::quant_log_positive_f64_v3;
use super::timestamps::{
    infer_timestamp_unit, timestamp_to_millis, validate_canonical_millisecond_timestamps,
};
use anyhow::{Result, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;
use std::collections::HashMap;

// Quant-v3's resident owner is feature-gated because its production consumer
// launches CUDA. The CPU oracle itself remains buildable without CUDA, but it
// must execute the exact same temporal admission source rather than copy its
// rules. This path-backed module is compiled only when the resident owner is
// absent; a given build therefore has one temporal-grid implementation.
#[cfg(not(feature = "gpu-cuda"))]
#[path = "gpu_resident_temporal_grid_v1.rs"]
mod quant_v3_temporal_grid_cpu_authority;
#[cfg(not(feature = "gpu-cuda"))]
use quant_v3_temporal_grid_cpu_authority::{
    ASIAN_SESSION_MILLIS_V2, UTC_DAY_MILLIS_V2, admit_fixed_intraday_grid_v1,
};

/// Bars per trading day, derived from the actual timestamp spacing (audit
/// D04). The "previous day / previous week" levels used a hardcoded 24 / 120
/// bars — correct ONLY on H1 (24×H1 = 1 day). On M1 that "day" was 24
/// minutes, on M5 two hours, on D1 twenty-four days — so the feature meant
/// something different on every timeframe. Deriving the count from the median
/// bar period (unit-agnostic, like the alignment fixes) makes
/// "previous day" actually one day on ANY timeframe. Falls back to 24 (the
/// old H1 assumption) only when timestamps are missing/degenerate.
fn bars_per_day(ohlcv: &Ohlcv, n: usize) -> usize {
    const FALLBACK_H1_BARS_PER_DAY: usize = 24;
    let Some(ts) = ohlcv.timestamp.as_deref() else {
        return FALLBACK_H1_BARS_PER_DAY;
    };
    if ts.len() < 2 {
        return FALLBACK_H1_BARS_PER_DAY;
    }
    let Some(unit) = infer_timestamp_unit(ts) else {
        return FALLBACK_H1_BARS_PER_DAY;
    };
    // Median positive spacing = the bar period, in native units.
    let mut steps: Vec<i64> = ts
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0)
        .collect();
    if steps.is_empty() {
        return FALLBACK_H1_BARS_PER_DAY;
    }
    let mid = steps.len() / 2;
    steps.select_nth_unstable(mid);
    // Unit-conversion fix (2026-07-16, found by the source-only re-audit):
    // `scale_to_millis()` is a MULTIPLIER only for seconds — for µs/ns it is
    // the DIVISOR. Multiplying (the old code) inflated an ns step by 1e6×,
    // collapsing bars_per_day to 1 on ns-stamped data. Latent in production
    // (datasets are ms-normalized at load) but wrong; use the canonical
    // converter which applies the correct direction per unit.
    let step_ms = match timestamp_to_millis(steps[mid], unit) {
        Ok(ms) => ms,
        Err(_) => return FALLBACK_H1_BARS_PER_DAY,
    };
    if step_ms <= 0 {
        return FALLBACK_H1_BARS_PER_DAY;
    }
    // One day = 86_400_000 ms. Round to nearest bar, clamp to [1, n].
    let per_day = ((86_400_000_f64 / step_ms as f64).round() as i64).max(1) as usize;
    per_day.clamp(1, n.max(1))
}

/// Compute advanced quantitative features for the genetic discovery engine.
pub fn compute_quant_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    if n == 0 {
        return vec![];
    }

    let close = &ohlcv.close;
    let high = &ohlcv.high;
    let low = &ohlcv.low;
    let open = &ohlcv.open;
    let volume = ohlcv.volume.as_deref();

    let mut cols: Vec<(String, Vec<f64>)> = Vec::new();

    // Canonical raw close input for stateful price models. Keeping it in the
    // shared plan avoids adapter-specific OHLC aliases and hidden reloads.
    cols.push(("quant_close".to_string(), close.clone()));

    // ==========================================
    // 1. Returns at multiple horizons
    // ==========================================
    for &lag in &[1, 2, 3, 5, 8, 13, 21] {
        let mut ret = vec![0.0; n];
        for i in lag..n {
            if close[i - lag].abs() > 1e-10 {
                ret[i] = (close[i] - close[i - lag]) / close[i - lag];
            }
        }
        cols.push((format!("quant_return_{}", lag), ret));
    }

    // ==========================================
    // 2. Log Returns
    // ==========================================
    let mut log_ret = vec![0.0; n];
    for i in 1..n {
        if close[i - 1] > 1e-10 && close[i] > 1e-10 {
            log_ret[i] = (close[i] / close[i - 1]).ln();
        }
    }
    cols.push(("quant_log_return".to_string(), log_ret.clone()));

    // Instantaneous log range used by the HMM regime model. Keep this in the
    // shared feature plan so training and inference consume exactly one
    // validity-aware implementation rather than recomputing OHLC privately.
    let mut log_volatility = vec![0.0; n];
    for i in 0..n {
        let range = high[i] - low[i];
        if range > 1e-10 {
            log_volatility[i] = range.ln();
        }
    }
    cols.push(("quant_log_volatility".to_string(), log_volatility));

    // ==========================================
    // 3. Realized Volatility (multiple windows)
    // ==========================================
    for &window in &[5, 10, 20, 50] {
        let mut rv = vec![0.0; n];
        for (i, rv_value) in rv.iter_mut().enumerate().skip(window) {
            let mut sum_sq = 0.0;
            for &value in log_ret.iter().take(i + 1).skip(i - window + 1) {
                sum_sq += value * value;
            }
            *rv_value = (sum_sq / window as f64).sqrt() * 252.0_f64.sqrt();
        }
        cols.push((format!("quant_realized_vol_{}", window), rv));
    }

    // ==========================================
    // 4. Garman-Klass Volatility (superior to close-close)
    // ==========================================
    for &window in &[10, 20] {
        let mut gk = vec![0.0; n];
        for (i, gk_value) in gk.iter_mut().enumerate().skip(window) {
            let mut sum = 0.0;
            for j in (i - window + 1)..=i {
                if open[j].abs() > 1e-10 {
                    let u = (high[j] / open[j]).ln();
                    let d = (low[j] / open[j]).ln();
                    let c = (close[j] / open[j]).ln();
                    sum += 0.5 * (u - d).powi(2) - (2.0_f64.ln() - 1.0) * c.powi(2);
                }
            }
            *gk_value = (sum / window as f64).abs().sqrt() * 252.0_f64.sqrt();
        }
        cols.push((format!("quant_gk_vol_{}", window), gk));
    }

    // ==========================================
    // 5. Parkinson Volatility (uses High-Low only)
    // ==========================================
    for &window in &[10, 20] {
        let mut pk = vec![0.0; n];
        for (i, pk_value) in pk.iter_mut().enumerate().skip(window) {
            let mut sum = 0.0;
            for j in (i - window + 1)..=i {
                if low[j] > 1e-10 {
                    let hl = (high[j] / low[j]).ln();
                    sum += hl * hl;
                }
            }
            let factor = 1.0 / (4.0 * window as f64 * 2.0_f64.ln());
            *pk_value = (factor * sum).sqrt() * 252.0_f64.sqrt();
        }
        cols.push((format!("quant_parkinson_vol_{}", window), pk));
    }

    // ==========================================
    // 6. Volatility Ratio (short/long vol — regime change detector)
    // ==========================================
    {
        let mut vol_ratio = vec![0.0; n];
        for (i, vol_ratio_value) in vol_ratio.iter_mut().enumerate().skip(20) {
            let mut short_sq = 0.0;
            let mut long_sq = 0.0;
            for &value in log_ret.iter().take(i + 1).skip(i - 4) {
                short_sq += value * value;
            }
            for &value in log_ret.iter().take(i + 1).skip(i - 19) {
                long_sq += value * value;
            }
            let short_v = (short_sq / 5.0).sqrt();
            let long_v = (long_sq / 20.0).sqrt();
            *vol_ratio_value = if long_v > 1e-10 {
                short_v / long_v
            } else {
                1.0
            };
        }
        cols.push(("quant_vol_ratio".to_string(), vol_ratio));
    }

    // ==========================================
    // 7. Hurst Exponent (Rescaled Range method — regime detection)
    // H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk
    // ==========================================
    {
        let window = 100;
        let mut hurst = vec![0.5; n]; // Default to random walk
        for i in window..n {
            let slice = &log_ret[(i - window + 1)..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let mut cumulative_dev = Vec::with_capacity(window);
            let mut running_sum = 0.0;
            for &v in slice {
                running_sum += v - mean;
                cumulative_dev.push(running_sum);
            }
            let r = cumulative_dev
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                - cumulative_dev.iter().cloned().fold(f64::INFINITY, f64::min);
            let s = {
                let var =
                    slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
                var.sqrt()
            };
            if s > 1e-12 && r > 1e-12 {
                hurst[i] = (r / s).ln() / (window as f64).ln();
                hurst[i] = hurst[i].clamp(0.0, 1.0);
            }
        }
        cols.push(("quant_hurst_100".to_string(), hurst));
    }

    // ==========================================
    // 8. Autocorrelation of returns (lag 1, 5, 10)
    // ==========================================
    for &ac_lag in &[1, 5, 10] {
        let window = 50;
        let mut autocorr = vec![0.0; n];
        for i in (window + ac_lag)..n {
            let slice = &log_ret[(i - window + 1)..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let mut num = 0.0;
            let mut den = 0.0;
            for t in ac_lag..window {
                let x = slice[t] - mean;
                let y = slice[t - ac_lag] - mean;
                num += x * y;
                den += x * x;
            }
            autocorr[i] = if den.abs() > 1e-12 { num / den } else { 0.0 };
            autocorr[i] = autocorr[i].clamp(-1.0, 1.0);
        }
        cols.push((format!("quant_autocorr_{}", ac_lag), autocorr));
    }

    // ==========================================
    // 9. Price Efficiency Ratio (Kaufman ER — trend strength)
    // ==========================================
    for &window in &[10, 20] {
        let mut er = vec![0.0; n];
        for i in window..n {
            let direction = (close[i] - close[i - window]).abs();
            let mut volatility = 0.0;
            for j in (i - window + 1)..=i {
                volatility += (close[j] - close[j - 1]).abs();
            }
            er[i] = if volatility > 1e-10 {
                direction / volatility
            } else {
                0.0
            };
        }
        cols.push((format!("quant_efficiency_ratio_{}", window), er));
    }

    // ==========================================
    // 10. Skewness & Kurtosis (rolling)
    // ==========================================
    {
        let window = 30;
        let mut skew = vec![0.0; n];
        let mut kurt = vec![0.0; n];
        for i in window..n {
            let slice = &log_ret[(i - window + 1)..=i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let mut m2 = 0.0;
            let mut m3 = 0.0;
            let mut m4 = 0.0;
            for &v in slice {
                let d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
                m4 += d * d * d * d;
            }
            m2 /= window as f64;
            m3 /= window as f64;
            m4 /= window as f64;
            let std = m2.sqrt();
            if std > 1e-12 {
                skew[i] = m3 / std.powi(3);
                kurt[i] = m4 / std.powi(4) - 3.0; // Excess kurtosis
                skew[i] = skew[i].clamp(-10.0, 10.0);
                kurt[i] = kurt[i].clamp(-10.0, 50.0);
            }
        }
        cols.push(("quant_skewness_30".to_string(), skew));
        cols.push(("quant_kurtosis_30".to_string(), kurt));
    }

    // ==========================================
    // 11. Kyle's Lambda proxy (price impact of volume)
    // ==========================================
    if let Some(vol) = volume {
        let window = 20;
        let mut kyle_lambda = vec![0.0; n];
        for (i, kyle_lambda_value) in kyle_lambda.iter_mut().enumerate().skip(window) {
            let mut sum_dv = 0.0;
            let mut sum_vv = 0.0;
            for j in (i - window + 1)..=i {
                let dp = close[j] - close[j.saturating_sub(1)];
                let signed_vol = dp.signum() * vol[j];
                sum_dv += dp * signed_vol;
                sum_vv += signed_vol * signed_vol;
            }
            *kyle_lambda_value = if sum_vv.abs() > 1e-10 {
                sum_dv / sum_vv
            } else {
                0.0
            };
        }
        cols.push(("quant_kyle_lambda".to_string(), kyle_lambda));
    }

    // ==========================================
    // 12. VPIN (Volume-Synchronized Probability of Informed Trading)
    // ==========================================
    if let Some(vol) = volume {
        let bucket_size = 50; // bars per bucket
        let n_buckets = 10;
        let mut vpin = vec![0.0; n];
        for (i, vpin_value) in vpin.iter_mut().enumerate().skip(bucket_size * n_buckets) {
            let mut buy_vol_sum = 0.0;
            let mut sell_vol_sum = 0.0;
            let mut total_vol = 0.0;
            for j in (i - bucket_size * n_buckets)..i {
                let mid = (high[j] + low[j]) / 2.0;
                let v = vol[j].abs();
                if close[j] > mid {
                    buy_vol_sum += v;
                } else {
                    sell_vol_sum += v;
                }
                total_vol += v;
            }
            *vpin_value = if total_vol > 1e-10 {
                (buy_vol_sum - sell_vol_sum).abs() / total_vol
            } else {
                0.0
            };
        }
        cols.push(("quant_vpin".to_string(), vpin));
    }

    // ==========================================
    // 13. Amihud Illiquidity Ratio
    // ==========================================
    if let Some(vol) = volume {
        let window = 20;
        let mut amihud = vec![0.0; n];
        for (i, amihud_value) in amihud.iter_mut().enumerate().skip(window) {
            let mut sum = 0.0;
            let mut count = 0;
            for j in (i - window + 1)..=i {
                if vol[j].abs() > 1e-10 && j > 0 {
                    let ret = (close[j] - close[j - 1]).abs() / close[j - 1].max(1e-10);
                    sum += ret / vol[j].abs();
                    count += 1;
                }
            }
            *amihud_value = if count > 0 { sum / count as f64 } else { 0.0 };
        }
        cols.push(("quant_amihud_illiquidity".to_string(), amihud));
    }

    // ==========================================
    // 14. Roll Spread Estimate (bid-ask proxy from close prices)
    // ==========================================
    {
        let window = 20;
        let mut roll_spread = vec![0.0; n];
        for (i, roll_spread_value) in roll_spread.iter_mut().enumerate().skip(window + 1) {
            let mut cov_sum = 0.0;
            let mut count = 0;
            for j in (i - window + 1)..=i {
                if j >= 2 {
                    let d1 = close[j] - close[j - 1];
                    let d0 = close[j - 1] - close[j - 2];
                    cov_sum += d1 * d0;
                    count += 1;
                }
            }
            if count > 0 {
                let cov = cov_sum / count as f64;
                // Roll spread = 2 * sqrt(-cov) if cov < 0
                *roll_spread_value = if cov < 0.0 { 2.0 * (-cov).sqrt() } else { 0.0 };
            }
        }
        cols.push(("quant_roll_spread".to_string(), roll_spread));
    }

    // ==========================================
    // 15. Consecutive Directional Bars
    // ==========================================
    {
        let mut consec_up = vec![0.0; n];
        let mut consec_down = vec![0.0; n];
        let mut up_count = 0.0_f64;
        let mut down_count = 0.0_f64;
        for i in 1..n {
            if close[i] > close[i - 1] {
                up_count += 1.0;
                down_count = 0.0;
            } else if close[i] < close[i - 1] {
                down_count += 1.0;
                up_count = 0.0;
            } else {
                up_count = 0.0;
                down_count = 0.0;
            }
            consec_up[i] = up_count;
            consec_down[i] = down_count;
        }
        cols.push(("quant_consec_up".to_string(), consec_up));
        cols.push(("quant_consec_down".to_string(), consec_down));
    }

    // ==========================================
    // 16. Inside Bar / Outside Bar count
    // ==========================================
    {
        let mut inside_bar = vec![0.0; n];
        let mut outside_bar = vec![0.0; n];
        for i in 1..n {
            if high[i] <= high[i - 1] && low[i] >= low[i - 1] {
                inside_bar[i] = 1.0;
            }
            if high[i] > high[i - 1] && low[i] < low[i - 1] {
                outside_bar[i] = 1.0;
            }
        }
        cols.push(("quant_inside_bar".to_string(), inside_bar));
        cols.push(("quant_outside_bar".to_string(), outside_bar));
    }

    // ==========================================
    // 17. Body-to-Range Ratio (candle structure)
    // ==========================================
    {
        let mut body_ratio = vec![0.0; n];
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                body_ratio[i] = (close[i] - open[i]).abs() / range;
            }
        }
        cols.push(("quant_body_ratio".to_string(), body_ratio));
    }

    // ==========================================
    // 18. Upper/Lower Shadow Ratio
    // ==========================================
    {
        let mut upper_shadow = vec![0.0; n];
        let mut lower_shadow = vec![0.0; n];
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                let body_top = close[i].max(open[i]);
                let body_bot = close[i].min(open[i]);
                upper_shadow[i] = (high[i] - body_top) / range;
                lower_shadow[i] = (body_bot - low[i]) / range;
            }
        }
        cols.push(("quant_upper_shadow".to_string(), upper_shadow));
        cols.push(("quant_lower_shadow".to_string(), lower_shadow));
    }

    // ==========================================
    // 19. Previous Day/Week High & Low Distance (normalized)
    // ==========================================
    {
        // Previous-day high/low: one actual trading day of bars on ANY
        // timeframe (audit D04 — was a hardcoded 24, i.e. H1-only).
        let mut prev_day_h_dist = vec![0.0; n];
        let mut prev_day_l_dist = vec![0.0; n];
        let period = bars_per_day(ohlcv, n);
        for i in period..n {
            let mut ph = f64::NEG_INFINITY;
            let mut pl = f64::INFINITY;
            for j in (i - period)..i {
                if high[j] > ph {
                    ph = high[j];
                }
                if low[j] < pl {
                    pl = low[j];
                }
            }
            let atr_proxy = (ph - pl).max(1e-10);
            prev_day_h_dist[i] = (close[i] - ph) / atr_proxy;
            prev_day_l_dist[i] = (close[i] - pl) / atr_proxy;
        }
        cols.push(("quant_prev_day_h_dist".to_string(), prev_day_h_dist));
        cols.push(("quant_prev_day_l_dist".to_string(), prev_day_l_dist));

        // Previous-week high/low: five trading days of bars on ANY
        // timeframe (audit D04 — was a hardcoded 120, i.e. 5×24 H1-only).
        let mut prev_week_h_dist = vec![0.0; n];
        let mut prev_week_l_dist = vec![0.0; n];
        let w_period = (period.saturating_mul(5)).clamp(1, n.max(1));
        for i in w_period..n {
            let mut ph = f64::NEG_INFINITY;
            let mut pl = f64::INFINITY;
            for j in (i - w_period)..i {
                if high[j] > ph {
                    ph = high[j];
                }
                if low[j] < pl {
                    pl = low[j];
                }
            }
            let atr_proxy = (ph - pl).max(1e-10);
            prev_week_h_dist[i] = (close[i] - ph) / atr_proxy;
            prev_week_l_dist[i] = (close[i] - pl) / atr_proxy;
        }
        cols.push(("quant_prev_week_h_dist".to_string(), prev_week_h_dist));
        cols.push(("quant_prev_week_l_dist".to_string(), prev_week_l_dist));
    }

    // ==========================================
    // 20. Opening Range Breakout (ORB) — first N bars of session
    // ==========================================
    {
        for &orb_bars in &[4, 8, 12] {
            // 15min, 30min, 1h on M5
            let mut orb_signal = vec![0.0; n];
            for i in orb_bars..n {
                let mut orb_high = f64::NEG_INFINITY;
                let mut orb_low = f64::INFINITY;
                for j in (i - orb_bars)..i {
                    if high[j] > orb_high {
                        orb_high = high[j];
                    }
                    if low[j] < orb_low {
                        orb_low = low[j];
                    }
                }
                if close[i] > orb_high {
                    orb_signal[i] = 1.0; // Bullish ORB breakout
                } else if close[i] < orb_low {
                    orb_signal[i] = -1.0; // Bearish ORB breakout
                }
            }
            cols.push((format!("quant_orb_{}", orb_bars), orb_signal));
        }
    }

    // ==========================================
    // 21. Power of 3 / AMD (Accumulation → Manipulation → Distribution)
    // ==========================================
    {
        let window = 20;
        let mut amd_phase = vec![0.0; n];
        for i in window..n {
            // Phase detection via range compression then expansion
            let mut ranges: Vec<f64> = Vec::with_capacity(window);
            for j in (i - window)..i {
                ranges.push(high[j] - low[j]);
            }
            let avg_range = ranges.iter().sum::<f64>() / window as f64;
            let recent_range = ranges[window - 1];
            let early_range = ranges.iter().take(window / 3).sum::<f64>() / (window as f64 / 3.0);

            if early_range < avg_range * 0.6 && recent_range > avg_range * 1.5 {
                // Accumulation → Distribution pattern
                amd_phase[i] = if close[i] > open[i] { 1.0 } else { -1.0 };
            } else if early_range < avg_range * 0.7 {
                amd_phase[i] = 0.3; // Accumulation phase (compression)
            }
        }
        cols.push(("quant_amd_phase".to_string(), amd_phase));
    }

    // ==========================================
    // 22. Wyckoff Phase Detection (Spring / Upthrust)
    // ==========================================
    {
        let window = 30;
        let mut wyckoff = vec![0.0; n];
        for i in window..n {
            let mut period_low = f64::INFINITY;
            let mut period_high = f64::NEG_INFINITY;
            for j in (i - window)..i {
                if low[j] < period_low {
                    period_low = low[j];
                }
                if high[j] > period_high {
                    period_high = high[j];
                }
            }
            // Spring: Wick below support, closes above it (bullish reversal)
            if low[i] < period_low && close[i] > period_low {
                wyckoff[i] = 1.0; // Spring (bullish)
            }
            // Upthrust: Wick above resistance, closes below it (bearish reversal)
            if high[i] > period_high && close[i] < period_high {
                wyckoff[i] = -1.0; // Upthrust (bearish)
            }
        }
        cols.push(("quant_wyckoff".to_string(), wyckoff));
    }

    // ==========================================
    // 23. Engulfing Pattern with Volume Confirmation
    // ==========================================
    if let Some(vol) = volume {
        let mut engulfing = vec![0.0; n];
        for i in 1..n {
            let prev_body = (close[i - 1] - open[i - 1]).abs();
            let curr_body = (close[i] - open[i]).abs();
            let vol_increase = vol[i] > vol[i - 1] * 1.2;

            // Bullish engulfing
            if close[i - 1] < open[i - 1]
                && close[i] > open[i]
                && open[i] <= close[i - 1]
                && close[i] >= open[i - 1]
                && curr_body > prev_body
                && vol_increase
            {
                engulfing[i] = 1.0;
            }
            // Bearish engulfing
            if close[i - 1] > open[i - 1]
                && close[i] < open[i]
                && open[i] >= close[i - 1]
                && close[i] <= open[i - 1]
                && curr_body > prev_body
                && vol_increase
            {
                engulfing[i] = -1.0;
            }
        }
        cols.push(("quant_engulfing_vol".to_string(), engulfing));
    }

    // ==========================================
    // 24. Pivot Points (Classic, Fibonacci, Camarilla)
    // ==========================================
    {
        let period = 24; // Rolling daily proxy
        let mut pivot = vec![0.0; n];
        let mut r1 = vec![0.0; n];
        let mut r2 = vec![0.0; n];
        let mut s1 = vec![0.0; n];
        let mut s2 = vec![0.0; n];
        let mut cam_r3 = vec![0.0; n]; // Camarilla R3
        let mut cam_s3 = vec![0.0; n]; // Camarilla S3

        for i in period..n {
            let mut ph = f64::NEG_INFINITY;
            let mut pl = f64::INFINITY;
            let pc = close[i - 1]; // Previous period close
            for j in (i - period)..i {
                if high[j] > ph {
                    ph = high[j];
                }
                if low[j] < pl {
                    pl = low[j];
                }
            }
            let pp = (ph + pl + pc) / 3.0;
            pivot[i] = pp;
            r1[i] = 2.0 * pp - pl;
            r2[i] = pp + (ph - pl);
            s1[i] = 2.0 * pp - ph;
            s2[i] = pp - (ph - pl);

            // Camarilla levels
            let range = ph - pl;
            cam_r3[i] = pc + range * 1.1 / 4.0;
            cam_s3[i] = pc - range * 1.1 / 4.0;
        }
        // Normalize as distance from close
        for i in 0..n {
            let atr_proxy = (high[i] - low[i]).max(1e-10);
            if pivot[i] != 0.0 {
                pivot[i] = (close[i] - pivot[i]) / atr_proxy;
                r1[i] = (close[i] - r1[i]) / atr_proxy;
                r2[i] = (close[i] - r2[i]) / atr_proxy;
                s1[i] = (close[i] - s1[i]) / atr_proxy;
                s2[i] = (close[i] - s2[i]) / atr_proxy;
                cam_r3[i] = (close[i] - cam_r3[i]) / atr_proxy;
                cam_s3[i] = (close[i] - cam_s3[i]) / atr_proxy;
            }
        }
        cols.push(("quant_pivot_dist".to_string(), pivot));
        cols.push(("quant_r1_dist".to_string(), r1));
        cols.push(("quant_r2_dist".to_string(), r2));
        cols.push(("quant_s1_dist".to_string(), s1));
        cols.push(("quant_s2_dist".to_string(), s2));
        cols.push(("quant_cam_r3_dist".to_string(), cam_r3));
        cols.push(("quant_cam_s3_dist".to_string(), cam_s3));
    }

    // ==========================================
    // 25. Z-Score of Price (mean-reversion signal)
    // ==========================================
    for &window in &[20, 50] {
        let mut zscore = vec![0.0; n];
        for i in window..n {
            let slice = &close[(i - window)..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
            let std = var.sqrt();
            zscore[i] = if std > 1e-10 {
                (close[i] - mean) / std
            } else {
                0.0
            };
        }
        cols.push((format!("quant_zscore_{}", window), zscore));
    }

    // ==========================================
    // 26. Fractal Dimension (Box-counting approximation)
    // ==========================================
    {
        let window = 30;
        let mut fd = vec![1.5; n]; // Default = Brownian motion
        for i in window..n {
            let slice = &close[(i - window)..=i];
            let max_p = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_p = slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_p - min_p;
            if range > 1e-10 {
                // Petrosian approximation
                let mut n_sign_changes = 0;
                for j in 2..slice.len() {
                    let d1 = slice[j] - slice[j - 1];
                    let d0 = slice[j - 1] - slice[j - 2];
                    if d1 * d0 < 0.0 {
                        n_sign_changes += 1;
                    }
                }
                let n_sc = n_sign_changes as f64;
                let n_pts = slice.len() as f64;
                fd[i] = (n_pts.ln()) / (n_pts.ln() + (n_pts / (n_pts + 0.4 * n_sc)).ln());
                fd[i] = fd[i].clamp(1.0, 2.0);
            }
        }
        cols.push(("quant_fractal_dim".to_string(), fd));
    }

    // ==========================================
    // 27. Volume Profile: relative volume to N-period average
    // ==========================================
    if let Some(vol) = volume {
        for &window in &[10, 20, 50] {
            let mut rvol = vec![0.0; n];
            for i in window..n {
                let avg = vol[(i - window)..i].iter().sum::<f64>() / window as f64;
                rvol[i] = if avg > 1e-10 { vol[i] / avg } else { 1.0 };
            }
            cols.push((format!("quant_rvol_{}", window), rvol));
        }
    }

    // ==========================================
    // 28. Delta Volume (buy vs sell pressure estimation)
    // ==========================================
    if let Some(vol) = volume {
        let mut delta = vec![0.0; n];
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Estimate buy/sell split using bar position
                let buy_pct = (close[i] - low[i]) / range;
                delta[i] = vol[i] * (2.0 * buy_pct - 1.0); // -1 to +1 scaled by volume
            }
        }
        // Cumulative delta
        let mut cum_delta = vec![0.0; n];
        let mut running = 0.0;
        for i in 0..n {
            running += delta[i];
            cum_delta[i] = running;
        }
        // Normalize cumulative delta (rolling Z-score)
        let window = 50;
        let mut cd_zscore = vec![0.0; n];
        for i in window..n {
            let slice = &cum_delta[(i - window)..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
            let std = var.sqrt();
            cd_zscore[i] = if std > 1e-10 {
                (cum_delta[i] - mean) / std
            } else {
                0.0
            };
        }
        cols.push(("quant_delta_volume".to_string(), delta));
        cols.push(("quant_cum_delta_zscore".to_string(), cd_zscore));
    }

    cols
}

fn quant_validity_mut<'a>(
    validity_by_name: &'a mut HashMap<String, Vec<FeatureCellValidity>>,
    name: &str,
) -> Result<&'a mut Vec<FeatureCellValidity>> {
    validity_by_name
        .get_mut(name)
        .ok_or_else(|| anyhow::anyhow!("missing quantitative validity plan for `{name}`"))
}

fn quant_mark_after_warmup(
    validity_by_name: &mut HashMap<String, Vec<FeatureCellValidity>>,
    name: &str,
    warmup: usize,
) -> Result<()> {
    let validity = quant_validity_mut(validity_by_name, name)?;
    validity.fill(FeatureCellValidity::Valid);
    let warmup = warmup.min(validity.len());
    validity[..warmup].fill(FeatureCellValidity::Warmup);
    Ok(())
}

fn quant_mark_all(
    validity_by_name: &mut HashMap<String, Vec<FeatureCellValidity>>,
    name: &str,
    reason: FeatureCellValidity,
) -> Result<()> {
    quant_validity_mut(validity_by_name, name)?.fill(reason);
    Ok(())
}

/// Explicit-validity f64 quantitative lane used by the atomic Tasks 5B-9
/// migration.
///
/// The legacy value producer remains temporarily so value parity can be
/// measured while the shared `FeatureFrame` is migrated. This boundary
/// replays every formula's information requirements and never exposes its
/// numeric warmup/default sentinels as observations. Outputs whose current
/// formula silently assumes a bar annualization, broker trading session, or
/// timeframe are marked missing until that typed contract reaches the
/// producer; guessing those semantics would make a fast but false feature.
pub fn compute_quant_feature_columns_f64(ohlcv: &Ohlcv) -> Result<Vec<FeatureColumnF64>> {
    compute_quant_feature_columns_f64_with_cumulative_delta_dependency(
        ohlcv,
        CumulativeDeltaValidityDependency::Prefix,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CumulativeDeltaValidityDependency {
    Prefix,
    RollingWindow,
}

fn compute_quant_feature_columns_f64_with_cumulative_delta_dependency(
    ohlcv: &Ohlcv,
    cumulative_delta_dependency: CumulativeDeltaValidityDependency,
) -> Result<Vec<FeatureColumnF64>> {
    const EPS: f64 = 1e-12;

    let n = ohlcv.len();
    ensure!(n > 0, "quantitative features require at least one OHLC row");
    ensure!(
        ohlcv.open.len() == n && ohlcv.high.len() == n && ohlcv.low.len() == n,
        "quantitative OHLC lengths do not match close length {n}"
    );
    for row in 0..n {
        let open = ohlcv.open[row];
        let high = ohlcv.high[row];
        let low = ohlcv.low[row];
        let close = ohlcv.close[row];
        ensure!(
            open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite(),
            "quantitative OHLC row {row} contains a non-finite price"
        );
        ensure!(
            open > 0.0 && high > 0.0 && low > 0.0 && close > 0.0,
            "quantitative OHLC row {row} contains a non-positive price"
        );
        ensure!(
            low <= open.min(close) && high >= open.max(close),
            "quantitative OHLC row {row} violates low <= open/close <= high"
        );
    }

    if let Some(timestamps) = ohlcv.timestamp.as_deref() {
        ensure!(
            timestamps.len() == n,
            "quantitative timestamp length {} does not match OHLC length {n}",
            timestamps.len()
        );
        validate_canonical_millisecond_timestamps(timestamps)?;
    }

    let volume = if let Some(volume) = ohlcv.volume.as_deref() {
        ensure!(
            volume.len() == n,
            "quantitative volume length {} does not match OHLC length {n}",
            volume.len()
        );
        if let Some((row, value)) = volume
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value < 0.0)
        {
            anyhow::bail!("quantitative volume row {row} is invalid: {value}");
        }
        Some(volume)
    } else {
        None
    };

    // The legacy function conditionally omits volume-derived columns. Supply
    // zeroes only to enumerate the stable schema; every such payload is
    // replaced with canonical NaN below when real volume is absent.
    let mut value_input = ohlcv.clone();
    if volume.is_none() {
        value_input.volume = Some(vec![0.0; n]);
    }
    let legacy = compute_quant_feature_columns(&value_input);
    let mut validity_by_name: HashMap<String, Vec<FeatureCellValidity>> = legacy
        .iter()
        .map(|(name, _)| (name.clone(), vec![FeatureCellValidity::ComputeFailure; n]))
        .collect();

    let mut log_returns = vec![0.0; n];
    for row in 1..n {
        log_returns[row] = (ohlcv.close[row] / ohlcv.close[row - 1]).ln();
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_close", 0)?;
    for lag in [1_usize, 2, 3, 5, 8, 13, 21] {
        quant_mark_after_warmup(&mut validity_by_name, &format!("quant_return_{lag}"), lag)?;
    }
    quant_mark_after_warmup(&mut validity_by_name, "quant_log_return", 1)?;
    quant_mark_after_warmup(&mut validity_by_name, "quant_log_volatility", 0)?;
    for row in 0..n {
        if ohlcv.high[row] - ohlcv.low[row] <= EPS {
            quant_validity_mut(&mut validity_by_name, "quant_log_volatility")?[row] =
                FeatureCellValidity::ZeroDenominator;
        }
    }

    // The existing values multiply every bar-frequency estimate by sqrt(252),
    // even when the source is intraday. Without the typed timeframe and
    // annualization convention these are not comparable observations.
    for name in [
        "quant_realized_vol_5",
        "quant_realized_vol_10",
        "quant_realized_vol_20",
        "quant_realized_vol_50",
        "quant_gk_vol_10",
        "quant_gk_vol_20",
        "quant_parkinson_vol_10",
        "quant_parkinson_vol_20",
    ] {
        quant_mark_all(
            &mut validity_by_name,
            name,
            FeatureCellValidity::MissingInput,
        )?;
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_vol_ratio", 20)?;
    for row in 20..n {
        let long_sq: f64 = log_returns[(row - 19)..=row]
            .iter()
            .map(|value| value * value)
            .sum();
        if long_sq <= EPS {
            quant_validity_mut(&mut validity_by_name, "quant_vol_ratio")?[row] =
                FeatureCellValidity::ZeroDenominator;
        }
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_hurst_100", 100)?;
    for row in 100..n {
        let slice = &log_returns[(row - 99)..=row];
        let mean = slice.iter().sum::<f64>() / 100.0;
        let mut running = 0.0;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for value in slice {
            running += value - mean;
            minimum = minimum.min(running);
            maximum = maximum.max(running);
        }
        let variance = slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 99.0;
        if variance.sqrt() <= EPS || maximum - minimum <= EPS {
            quant_validity_mut(&mut validity_by_name, "quant_hurst_100")?[row] =
                FeatureCellValidity::ZeroDenominator;
        }
    }

    for lag in [1_usize, 5, 10] {
        let name = format!("quant_autocorr_{lag}");
        quant_mark_after_warmup(&mut validity_by_name, &name, 50 + lag)?;
        for row in (50 + lag)..n {
            let slice = &log_returns[(row - 49)..=row];
            let mean = slice.iter().sum::<f64>() / 50.0;
            let denominator: f64 = (lag..50).map(|offset| (slice[offset] - mean).powi(2)).sum();
            if denominator <= EPS {
                quant_validity_mut(&mut validity_by_name, &name)?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    }

    for window in [10_usize, 20] {
        let name = format!("quant_efficiency_ratio_{window}");
        quant_mark_after_warmup(&mut validity_by_name, &name, window)?;
        for row in window..n {
            let volatility: f64 = ((row - window + 1)..=row)
                .map(|index| (ohlcv.close[index] - ohlcv.close[index - 1]).abs())
                .sum();
            if volatility <= EPS {
                quant_validity_mut(&mut validity_by_name, &name)?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    }

    for name in ["quant_skewness_30", "quant_kurtosis_30"] {
        quant_mark_after_warmup(&mut validity_by_name, name, 30)?;
    }
    for row in 30..n {
        let slice = &log_returns[(row - 29)..=row];
        let mean = slice.iter().sum::<f64>() / 30.0;
        let variance = slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 30.0;
        if variance.sqrt() <= EPS {
            for name in ["quant_skewness_30", "quant_kurtosis_30"] {
                quant_validity_mut(&mut validity_by_name, name)?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    }

    if let Some(volume) = volume {
        quant_mark_after_warmup(&mut validity_by_name, "quant_kyle_lambda", 20)?;
        for row in 20..n {
            let denominator: f64 = ((row - 19)..=row)
                .map(|index| {
                    let delta = ohlcv.close[index] - ohlcv.close[index - 1];
                    let direction = if delta > 0.0 {
                        1.0
                    } else if delta < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    (direction * volume[index]).powi(2)
                })
                .sum();
            if denominator <= EPS {
                quant_validity_mut(&mut validity_by_name, "quant_kyle_lambda")?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }

        quant_mark_after_warmup(&mut validity_by_name, "quant_vpin", 500)?;
        for row in 500..n {
            let total_volume: f64 = volume[(row - 500)..row].iter().sum();
            if total_volume <= EPS {
                quant_validity_mut(&mut validity_by_name, "quant_vpin")?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }

        quant_mark_after_warmup(&mut validity_by_name, "quant_amihud_illiquidity", 20)?;
        for row in 20..n {
            if volume[(row - 19)..=row].iter().all(|value| *value <= EPS) {
                quant_validity_mut(&mut validity_by_name, "quant_amihud_illiquidity")?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    } else {
        for name in [
            "quant_kyle_lambda",
            "quant_vpin",
            "quant_amihud_illiquidity",
        ] {
            quant_mark_all(
                &mut validity_by_name,
                name,
                FeatureCellValidity::MissingInput,
            )?;
        }
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_roll_spread", 21)?;
    for name in ["quant_consec_up", "quant_consec_down"] {
        quant_mark_after_warmup(&mut validity_by_name, name, 1)?;
    }
    for name in ["quant_inside_bar", "quant_outside_bar"] {
        quant_mark_after_warmup(&mut validity_by_name, name, 1)?;
    }
    for name in [
        "quant_body_ratio",
        "quant_upper_shadow",
        "quant_lower_shadow",
    ] {
        quant_mark_after_warmup(&mut validity_by_name, name, 0)?;
    }
    for row in 0..n {
        if ohlcv.high[row] - ohlcv.low[row] <= EPS {
            for name in [
                "quant_body_ratio",
                "quant_upper_shadow",
                "quant_lower_shadow",
            ] {
                quant_validity_mut(&mut validity_by_name, name)?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    }

    // These columns currently infer a rolling "day" or pretend the last N
    // bars are an opening range. Ohlcv does not carry the broker-session and
    // source-timeframe contract required to make those identities true.
    for name in [
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
    ] {
        quant_mark_all(
            &mut validity_by_name,
            name,
            FeatureCellValidity::MissingInput,
        )?;
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_amd_phase", 20)?;
    for row in 20..n {
        let average_range: f64 = ((row - 20)..row)
            .map(|index| ohlcv.high[index] - ohlcv.low[index])
            .sum::<f64>()
            / 20.0;
        if average_range <= EPS {
            quant_validity_mut(&mut validity_by_name, "quant_amd_phase")?[row] =
                FeatureCellValidity::ZeroDenominator;
        }
    }
    quant_mark_after_warmup(&mut validity_by_name, "quant_wyckoff", 30)?;

    if volume.is_some() {
        quant_mark_after_warmup(&mut validity_by_name, "quant_engulfing_vol", 1)?;
    } else {
        quant_mark_all(
            &mut validity_by_name,
            "quant_engulfing_vol",
            FeatureCellValidity::MissingInput,
        )?;
    }

    for window in [20_usize, 50] {
        let name = format!("quant_zscore_{window}");
        quant_mark_after_warmup(&mut validity_by_name, &name, window)?;
        for row in window..n {
            let slice = &ohlcv.close[(row - window)..row];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let variance = slice
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (window as f64 - 1.0);
            if variance.sqrt() <= EPS {
                quant_validity_mut(&mut validity_by_name, &name)?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    }

    quant_mark_after_warmup(&mut validity_by_name, "quant_fractal_dim", 30)?;
    for row in 30..n {
        let slice = &ohlcv.close[(row - 30)..=row];
        let minimum = slice.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if maximum - minimum <= EPS {
            quant_validity_mut(&mut validity_by_name, "quant_fractal_dim")?[row] =
                FeatureCellValidity::ZeroDenominator;
        }
    }

    if let Some(volume) = volume {
        for window in [10_usize, 20, 50] {
            let name = format!("quant_rvol_{window}");
            quant_mark_after_warmup(&mut validity_by_name, &name, window)?;
            for row in window..n {
                let average = volume[(row - window)..row].iter().sum::<f64>() / window as f64;
                if average <= EPS {
                    quant_validity_mut(&mut validity_by_name, &name)?[row] =
                        FeatureCellValidity::ZeroDenominator;
                }
            }
        }

        quant_mark_after_warmup(&mut validity_by_name, "quant_delta_volume", 0)?;
        let mut delta_validity = vec![FeatureCellValidity::Valid; n];
        let mut cumulative_delta = vec![0.0; n];
        let mut cumulative = 0.0;
        for row in 0..n {
            let range = ohlcv.high[row] - ohlcv.low[row];
            if range <= EPS {
                delta_validity[row] = FeatureCellValidity::ZeroDenominator;
                quant_validity_mut(&mut validity_by_name, "quant_delta_volume")?[row] =
                    FeatureCellValidity::ZeroDenominator;
            } else {
                let buy_fraction = (ohlcv.close[row] - ohlcv.low[row]) / range;
                cumulative += volume[row] * (2.0 * buy_fraction - 1.0);
            }
            cumulative_delta[row] = cumulative;
        }

        quant_mark_after_warmup(&mut validity_by_name, "quant_cum_delta_zscore", 50)?;
        for row in 50..n {
            let dependency_start = match cumulative_delta_dependency {
                CumulativeDeltaValidityDependency::Prefix => 0,
                // A delta at or before the oldest cumulative observation is a
                // common additive offset in every value used by this z-score
                // and cancels. Only the 50 increments through the current row
                // affect relative levels in corrected semantic-v4.
                CumulativeDeltaValidityDependency::RollingWindow => row + 1 - 50,
            };
            if delta_validity[dependency_start..=row]
                .iter()
                .any(|validity| !validity.is_valid())
            {
                quant_validity_mut(&mut validity_by_name, "quant_cum_delta_zscore")?[row] =
                    FeatureCellValidity::ZeroDenominator;
                continue;
            }
            let slice = &cumulative_delta[(row - 50)..row];
            let mean = slice.iter().sum::<f64>() / 50.0;
            let variance = slice
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / 49.0;
            if variance.sqrt() <= EPS {
                quant_validity_mut(&mut validity_by_name, "quant_cum_delta_zscore")?[row] =
                    FeatureCellValidity::ZeroDenominator;
            }
        }
    } else {
        for name in [
            "quant_rvol_10",
            "quant_rvol_20",
            "quant_rvol_50",
            "quant_delta_volume",
            "quant_cum_delta_zscore",
        ] {
            quant_mark_all(
                &mut validity_by_name,
                name,
                FeatureCellValidity::MissingInput,
            )?;
        }
    }

    if let Some((name, row)) = validity_by_name.iter().find_map(|(name, validity)| {
        validity
            .iter()
            .position(|reason| *reason == FeatureCellValidity::ComputeFailure)
            .map(|row| (name.clone(), row))
    }) {
        anyhow::bail!("unclassified quantitative validity for `{name}` row {row}");
    }

    legacy
        .into_iter()
        .map(|(name, values)| {
            let validity = validity_by_name
                .remove(&name)
                .ok_or_else(|| anyhow::anyhow!("missing quantitative validity for `{name}`"))?;
            FeatureColumnF64::new(name, values, validity)
        })
        .collect()
}

const QUANT_V3_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct QuantV3CompletedUtcDay {
    high: f64,
    low: f64,
    close: f64,
}

fn quant_v3_replace_column(
    columns: &mut [FeatureColumnF64],
    name: &str,
    values: Vec<f64>,
    validity: Vec<FeatureCellValidity>,
) -> Result<()> {
    let replacement = FeatureColumnF64::new(name, values, validity)?;
    let slot = columns
        .iter_mut()
        .find(|column| column.name == name)
        .ok_or_else(|| anyhow::anyhow!("Quant-v3 source omitted `{name}`"))?;
    *slot = replacement;
    Ok(())
}

fn quant_v3_fixed_validity(rows: usize, warmup: usize) -> Vec<FeatureCellValidity> {
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

fn quant_v3_cloned_validity(
    columns: &[FeatureColumnF64],
    name: &str,
) -> Result<Vec<FeatureCellValidity>> {
    columns
        .iter()
        .find(|column| column.name == name)
        .map(|column| column.validity.clone())
        .ok_or_else(|| anyhow::anyhow!("Quant-v3 source omitted `{name}`"))
}

fn quant_v3_exact_log(value: f64) -> Result<f64> {
    quant_log_positive_f64_v3(value)
        .ok_or_else(|| anyhow::anyhow!("Quant-v3 exact log rejected {value}"))
}

fn quant_v3_install_exact_log_migrations(
    bars: &Ohlcv,
    bars_per_day: u64,
    annualization_periods_per_year: u64,
    columns: &mut [FeatureColumnF64],
) -> Result<()> {
    ensure!(
        bars_per_day > 0,
        "Quant-v3 admitted grid reported zero bars per UTC day"
    );
    let rows = bars.len();
    let mut log_returns = vec![0.0; rows];
    for row in 1..rows {
        if bars.close[row - 1].abs() > 1e-10 && bars.close[row].abs() > 1e-10 {
            log_returns[row] = quant_v3_exact_log(bars.close[row] / bars.close[row - 1])?;
        }
    }
    quant_v3_replace_column(
        columns,
        "quant_log_return",
        log_returns.clone(),
        quant_v3_cloned_validity(columns, "quant_log_return")?,
    )?;

    let mut log_volatility = vec![0.0; rows];
    for row in 0..rows {
        let range = bars.high[row] - bars.low[row];
        if range > 1e-10 {
            log_volatility[row] = quant_v3_exact_log(range)?;
        }
    }
    quant_v3_replace_column(
        columns,
        "quant_log_volatility",
        log_volatility,
        quant_v3_cloned_validity(columns, "quant_log_volatility")?,
    )?;

    ensure!(
        annualization_periods_per_year >= bars_per_day,
        "Quant-v3 admitted annualization periods are smaller than one UTC trading day"
    );
    let annualization_scale = (annualization_periods_per_year as f64).sqrt();

    for window in [5_usize, 10, 20, 50] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum_squared = 0.0;
            for value in &log_returns[(row - window + 1)..=row] {
                sum_squared += value * value;
            }
            values[row] = (sum_squared / window as f64).sqrt() * annualization_scale;
        }
        quant_v3_replace_column(
            columns,
            &format!("quant_realized_vol_{window}"),
            values,
            quant_v3_fixed_validity(rows, window),
        )?;
    }

    for window in [10_usize, 20] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum = 0.0;
            for index in (row - window + 1)..=row {
                if bars.open[index].abs() > 1e-10 {
                    let up = quant_v3_exact_log(bars.high[index] / bars.open[index])?;
                    let down = quant_v3_exact_log(bars.low[index] / bars.open[index])?;
                    let close = quant_v3_exact_log(bars.close[index] / bars.open[index])?;
                    sum += 0.5 * (up - down).powi(2)
                        - (quant_v3_exact_log(2.0)? - 1.0) * close.powi(2);
                }
            }
            values[row] = (sum / window as f64).abs().sqrt() * annualization_scale;
        }
        quant_v3_replace_column(
            columns,
            &format!("quant_gk_vol_{window}"),
            values,
            quant_v3_fixed_validity(rows, window),
        )?;
    }

    for window in [10_usize, 20] {
        let mut values = vec![0.0; rows];
        for row in window..rows {
            let mut sum = 0.0;
            for index in (row - window + 1)..=row {
                if bars.low[index] > 1e-10 {
                    let high_low = quant_v3_exact_log(bars.high[index] / bars.low[index])?;
                    sum += high_low * high_low;
                }
            }
            let factor = 1.0 / (4.0 * window as f64 * quant_v3_exact_log(2.0)?);
            values[row] = (factor * sum).sqrt() * annualization_scale;
        }
        quant_v3_replace_column(
            columns,
            &format!("quant_parkinson_vol_{window}"),
            values,
            quant_v3_fixed_validity(rows, window),
        )?;
    }

    let mut vol_ratio = vec![0.0; rows];
    let mut vol_ratio_validity = quant_v3_fixed_validity(rows, 20);
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
        if long_squared <= QUANT_V3_EPS {
            vol_ratio_validity[row] = FeatureCellValidity::ZeroDenominator;
        }
        vol_ratio[row] = if long > 1e-10 { short / long } else { 1.0 };
    }
    quant_v3_replace_column(columns, "quant_vol_ratio", vol_ratio, vol_ratio_validity)?;

    let mut hurst = vec![0.5; rows];
    let mut hurst_validity = quant_v3_fixed_validity(rows, 100);
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
        if deviation <= QUANT_V3_EPS || spread <= QUANT_V3_EPS {
            hurst_validity[row] = FeatureCellValidity::ZeroDenominator;
            continue;
        }
        hurst[row] =
            (quant_v3_exact_log(spread / deviation)? / quant_v3_exact_log(100.0)?).clamp(0.0, 1.0);
    }
    quant_v3_replace_column(columns, "quant_hurst_100", hurst, hurst_validity)?;

    for lag in [1_usize, 5, 10] {
        let mut autocorrelation = vec![0.0; rows];
        let mut autocorrelation_validity = quant_v3_fixed_validity(rows, 50 + lag);
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
            if denominator <= QUANT_V3_EPS {
                autocorrelation_validity[row] = FeatureCellValidity::ZeroDenominator;
            }
            autocorrelation[row] = if denominator.abs() > QUANT_V3_EPS {
                numerator / denominator
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
        }
        quant_v3_replace_column(
            columns,
            &format!("quant_autocorr_{lag}"),
            autocorrelation,
            autocorrelation_validity,
        )?;
    }

    let mut skewness = vec![0.0; rows];
    let mut kurtosis = vec![0.0; rows];
    let mut skewness_validity = quant_v3_fixed_validity(rows, 30);
    let mut kurtosis_validity = quant_v3_fixed_validity(rows, 30);
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
        if standard_deviation <= QUANT_V3_EPS {
            skewness_validity[row] = FeatureCellValidity::ZeroDenominator;
            kurtosis_validity[row] = FeatureCellValidity::ZeroDenominator;
            continue;
        }
        skewness[row] = (third / standard_deviation.powi(3)).clamp(-10.0, 10.0);
        kurtosis[row] = (fourth / standard_deviation.powi(4) - 3.0).clamp(-10.0, 50.0);
    }
    quant_v3_replace_column(columns, "quant_skewness_30", skewness, skewness_validity)?;
    quant_v3_replace_column(columns, "quant_kurtosis_30", kurtosis, kurtosis_validity)?;

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
            let numerator = quant_v3_exact_log(points)?;
            let denominator =
                numerator + quant_v3_exact_log(points / (points + 0.4 * sign_changes))?;
            fractal_dimension[row] = (numerator / denominator).clamp(1.0, 2.0);
        }
    }
    quant_v3_replace_column(
        columns,
        "quant_fractal_dim",
        fractal_dimension,
        quant_v3_cloned_validity(columns, "quant_fractal_dim")?,
    )?;
    Ok(())
}

fn quant_v3_install_temporal_migrations(
    bars: &Ohlcv,
    columns: &mut [FeatureColumnF64],
) -> Result<()> {
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
    let timestamps = bars
        .timestamp
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Quant-v3 requires canonical millisecond timestamps"))?;
    let mut values: [Vec<f64>; 14] = std::array::from_fn(|_| vec![0.0; rows]);
    let mut validity: [Vec<FeatureCellValidity>; 14] =
        std::array::from_fn(|_| vec![FeatureCellValidity::Warmup; rows]);

    let mut completed_days: [Option<QuantV3CompletedUtcDay>; 5] = [None; 5];
    let mut completed_day_count = 0_usize;
    let mut current_day_key = timestamps[0].div_euclid(UTC_DAY_MILLIS_V2);
    let mut current_day = QuantV3CompletedUtcDay {
        high: f64::NEG_INFINITY,
        low: f64::INFINITY,
        close: bars.close[0],
    };
    let mut orb_count = 0_usize;
    let mut orb_high = [f64::NEG_INFINITY; 3];
    let mut orb_low = [f64::INFINITY; 3];
    let orb_thresholds = [4_usize, 8, 12];

    for row in 0..rows {
        let day_key = timestamps[row].div_euclid(UTC_DAY_MILLIS_V2);
        if day_key != current_day_key {
            if completed_day_count < completed_days.len() {
                completed_days[completed_day_count] = Some(current_day);
                completed_day_count += 1;
            } else {
                completed_days.rotate_left(1);
                completed_days[completed_days.len() - 1] = Some(current_day);
            }
            current_day_key = day_key;
            current_day = QuantV3CompletedUtcDay {
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
                validity[index][row] = if previous_range <= QUANT_V3_EPS {
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
                validity[index][row] = if current_range <= QUANT_V3_EPS {
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
                validity[index][row] = if week_range <= QUANT_V3_EPS {
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
        let millis_in_day = timestamps[row].rem_euclid(UTC_DAY_MILLIS_V2);
        if millis_in_day < ASIAN_SESSION_MILLIS_V2 {
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
        quant_v3_replace_column(
            columns,
            NAMES[index],
            std::mem::take(&mut values[index]),
            std::mem::take(&mut validity[index]),
        )?;
    }
    Ok(())
}

/// Typed CPU authority for the corrected exact resident Quant semantic-v4 graph.
///
/// Unlike the compatibility `compute_quant_feature_columns_f64` entrypoint,
/// this requires an explicit fixed intraday timeframe, validates every source
/// timestamp against that grid, annualizes per-bar volatility with
/// `sqrt(252 * bars_per_UTC_day)`, and materializes the UTC-day/Asian-session
/// routes. Its operation order is shared with the frozen CPU/CUDA parity
/// oracle; invalid cells retain explicit validity and canonical NaN payloads.
pub fn compute_quant_feature_columns_v4_f64(
    ohlcv: &Ohlcv,
    timeframe: CanonicalTimeframe,
) -> Result<Vec<FeatureColumnF64>> {
    let timeframe_millis = timeframe.fixed_duration_ms().ok_or_else(|| {
        anyhow::anyhow!("Quant-v4 requires a fixed intraday timeframe, got {timeframe}")
    })?;
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Quant-v4 requires canonical millisecond timestamps"))?;
    let admitted_grid = admit_fixed_intraday_grid_v1(timeframe_millis, timestamps)
        .map_err(|error| anyhow::anyhow!("Quant-v4 temporal admission failed: {error}"))?;
    ensure!(
        admitted_grid.timeframe_millis()
            == u64::try_from(timeframe_millis)
                .map_err(|_| anyhow::anyhow!("Quant-v4 timeframe width overflow"))?
            && admitted_grid.bars_per_asian_session() > 0
            && admitted_grid.bars_per_trading_week() >= admitted_grid.bars_per_utc_day(),
        "Quant-v4 temporal admission returned an internally inconsistent grid receipt"
    );
    let mut columns = compute_quant_feature_columns_f64_with_cumulative_delta_dependency(
        ohlcv,
        CumulativeDeltaValidityDependency::RollingWindow,
    )?;
    quant_v3_install_exact_log_migrations(
        ohlcv,
        admitted_grid.bars_per_utc_day(),
        admitted_grid.annualization_periods_per_year(),
        &mut columns,
    )?;
    quant_v3_install_temporal_migrations(ohlcv, &mut columns)?;
    #[cfg(feature = "gpu-cuda")]
    ensure!(
        columns.len() == RESIDENT_QUANT_COLUMN_NAMES_V3.len()
            && columns
                .iter()
                .map(|column| column.name.as_str())
                .eq(RESIDENT_QUANT_COLUMN_NAMES_V3),
        "Quant-v4 CPU authority schema does not match the exact resident 63-column order"
    );
    #[cfg(not(feature = "gpu-cuda"))]
    ensure!(
        columns.len() == 63,
        "Quant-v4 CPU authority schema width is {}, expected 63",
        columns.len()
    );
    Ok(columns)
}

#[cfg(test)]
mod d04_tests {
    use super::*;
    use crate::Ohlcv;

    fn ohlcv_with_step(step_ms: i64, n: usize) -> Ohlcv {
        let ts: Vec<i64> = (0..n as i64)
            .map(|i| 1_700_000_000_000 + i * step_ms)
            .collect();
        Ohlcv {
            timestamp: Some(ts),
            open: vec![1.0; n],
            high: vec![1.0; n],
            low: vec![1.0; n],
            close: vec![1.0; n],
            volume: Some(vec![1.0; n]),
        }
    }

    #[test]
    fn bars_per_day_is_timeframe_aware() {
        // D04: the "previous day" window must be one actual day of bars on
        // every timeframe, not a hardcoded 24 (H1-only).
        let n = 5000;
        // H1 (3_600_000 ms) → 24 bars/day.
        assert_eq!(bars_per_day(&ohlcv_with_step(3_600_000, n), n), 24);
        // M5 (300_000 ms) → 288 bars/day.
        assert_eq!(bars_per_day(&ohlcv_with_step(300_000, n), n), 288);
        // M1 (60_000 ms) → 1440 bars/day.
        assert_eq!(bars_per_day(&ohlcv_with_step(60_000, n), n), 1440);
        // D1 (86_400_000 ms) → 1 bar/day.
        assert_eq!(bars_per_day(&ohlcv_with_step(86_400_000, n), n), 1);
    }

    #[test]
    fn bars_per_day_converts_ns_and_us_units_correctly() {
        // Regression (2026-07-16): scale_to_millis is a divisor for µs/ns;
        // the old multiply collapsed ns-stamped M1 to 1 bar/day.
        let n = 5000;
        // ns-stamped M1: base ~1.7e18 ns, step 60e9 ns → 1440 bars/day.
        let ts_ns: Vec<i64> = (0..n as i64)
            .map(|i| 1_700_000_000_000_000_000 + i * 60_000_000_000)
            .collect();
        let mut o = ohlcv_with_step(60_000, n);
        o.timestamp = Some(ts_ns);
        assert_eq!(bars_per_day(&o, n), 1440, "ns M1 must be 1440 bars/day");
        // µs-stamped M5: step 300e6 µs → 288 bars/day.
        let ts_us: Vec<i64> = (0..n as i64)
            .map(|i| 1_700_000_000_000_000 + i * 300_000_000)
            .collect();
        let mut o = ohlcv_with_step(300_000, n);
        o.timestamp = Some(ts_us);
        assert_eq!(bars_per_day(&o, n), 288, "µs M5 must be 288 bars/day");
    }

    #[test]
    fn bars_per_day_falls_back_without_timestamps() {
        let mut o = ohlcv_with_step(300_000, 100);
        o.timestamp = None;
        assert_eq!(
            bars_per_day(&o, 100),
            24,
            "no timestamps → legacy H1 assumption"
        );
    }

    #[test]
    fn bars_per_day_clamped_to_series_length() {
        // A short D1 series still yields a usable (clamped) window.
        let o = ohlcv_with_step(86_400_000, 3);
        assert!(bars_per_day(&o, 3) >= 1);
    }
}
