/// Advanced Quantitative & Statistical Feature Generation
///
/// Institutional-grade statistical features for regime detection,
/// market microstructure analysis, and alpha generation.
use super::super::Ohlcv;
use super::timestamps::infer_timestamp_unit;

/// Bars per trading day, derived from the actual timestamp spacing (audit
/// D04). The "previous day / previous week" levels used a hardcoded 24 / 120
/// bars — correct ONLY on H1 (24×H1 = 1 day). On M1 that "day" was 24
/// minutes, on M5 two hours, on D1 twenty-four days — so the feature meant
/// something different on every timeframe. Deriving the count from the median
/// bar period (unit-agnostic, like the resample/alignment fixes) makes
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
    let mut steps: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0).collect();
    if steps.is_empty() {
        return FALLBACK_H1_BARS_PER_DAY;
    }
    let mid = steps.len() / 2;
    steps.select_nth_unstable(mid);
    let step_ms = steps[mid].saturating_mul(unit.scale_to_millis());
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

#[cfg(test)]
mod d04_tests {
    use super::*;
    use crate::Ohlcv;

    fn ohlcv_with_step(step_ms: i64, n: usize) -> Ohlcv {
        let ts: Vec<i64> = (0..n as i64).map(|i| 1_700_000_000_000 + i * step_ms).collect();
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
    fn bars_per_day_falls_back_without_timestamps() {
        let mut o = ohlcv_with_step(300_000, 100);
        o.timestamp = None;
        assert_eq!(bars_per_day(&o, 100), 24, "no timestamps → legacy H1 assumption");
    }

    #[test]
    fn bars_per_day_clamped_to_series_length() {
        // A short D1 series still yields a usable (clamped) window.
        let o = ohlcv_with_step(86_400_000, 3);
        assert!(bars_per_day(&o, 3) >= 1);
    }
}
