/// Market Regime Detection Features
///
/// Classifies the market into distinct regimes:
/// - Trending vs Ranging (ADX-based + Hurst-based)
/// - High/Low/Normal volatility (Garman-Klass regime switching)
/// - Risk-On vs Risk-Off (inferred from price behavior)
/// - Compression → Expansion transitions (squeeze detection)
/// - Mean-reversion vs Momentum regime probability
use super::super::Ohlcv;

/// Compute regime classification features for the genetic discovery engine.
pub fn compute_regime_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    if n == 0 {
        return vec![];
    }

    let close = &ohlcv.close;
    let high = &ohlcv.high;
    let low = &ohlcv.low;
    let open = &ohlcv.open;

    let mut cols: Vec<(String, Vec<f64>)> = Vec::new();

    // ==========================================
    // 1. Volatility Regime (3-state: low/normal/high)
    // Uses ratio of short-term vs long-term Garman-Klass volatility
    // ==========================================
    {
        let short_w = 10;
        let long_w = 50;
        let mut vol_regime = vec![0.0; n];
        let mut vol_zscore = vec![0.0; n];

        for i in long_w..n {
            // Short-term GK vol
            let mut short_gk = 0.0;
            for j in (i - short_w + 1)..=i {
                let u = if open[j] > 1e-10 {
                    (high[j] / open[j]).ln()
                } else {
                    0.0
                };
                let d = if open[j] > 1e-10 {
                    (low[j] / open[j]).ln()
                } else {
                    0.0
                };
                let c = if open[j] > 1e-10 {
                    (close[j] / open[j]).ln()
                } else {
                    0.0
                };
                short_gk += 0.5 * (u - d).powi(2) - (2.0_f64.ln() - 1.0) * c.powi(2);
            }
            short_gk = (short_gk / short_w as f64).abs().sqrt();

            // Long-term GK vol
            let mut long_gk = 0.0;
            for j in (i - long_w + 1)..=i {
                let u = if open[j] > 1e-10 {
                    (high[j] / open[j]).ln()
                } else {
                    0.0
                };
                let d = if open[j] > 1e-10 {
                    (low[j] / open[j]).ln()
                } else {
                    0.0
                };
                let c = if open[j] > 1e-10 {
                    (close[j] / open[j]).ln()
                } else {
                    0.0
                };
                long_gk += 0.5 * (u - d).powi(2) - (2.0_f64.ln() - 1.0) * c.powi(2);
            }
            long_gk = (long_gk / long_w as f64).abs().sqrt();

            let ratio = if long_gk > 1e-12 {
                short_gk / long_gk
            } else {
                1.0
            };

            // Classify
            if ratio > 1.5 {
                vol_regime[i] = 1.0; // High volatility regime
            } else if ratio < 0.6 {
                vol_regime[i] = -1.0; // Low volatility / compression
            }
            // else 0.0 = normal

            // Also provide continuous Z-score
            vol_zscore[i] = (ratio - 1.0).clamp(-3.0, 3.0);
        }
        cols.push(("regime_vol_state".to_string(), vol_regime));
        cols.push(("regime_vol_zscore".to_string(), vol_zscore));
    }

    // ==========================================
    // 2. Trending vs Ranging (ADX-like calculation)
    // ==========================================
    {
        let period = 14;
        let mut trend_strength = vec![0.0; n];
        let mut trend_direction = vec![0.0; n];

        if n > period + 1 {
            let mut plus_dm_smooth = 0.0_f64;
            let mut minus_dm_smooth = 0.0_f64;
            let mut tr_smooth = 0.0_f64;

            for i in 1..n {
                let up_move = high[i] - high[i - 1];
                let down_move = low[i - 1] - low[i];
                let plus_dm = if up_move > down_move && up_move > 0.0 {
                    up_move
                } else {
                    0.0
                };
                let minus_dm = if down_move > up_move && down_move > 0.0 {
                    down_move
                } else {
                    0.0
                };
                let tr = (high[i] - low[i])
                    .max((high[i] - close[i - 1]).abs())
                    .max((low[i] - close[i - 1]).abs());

                if i <= period {
                    plus_dm_smooth += plus_dm;
                    minus_dm_smooth += minus_dm;
                    tr_smooth += tr;
                    if i == period {
                        plus_dm_smooth /= period as f64;
                        minus_dm_smooth /= period as f64;
                        tr_smooth /= period as f64;
                    }
                } else {
                    plus_dm_smooth = plus_dm_smooth - plus_dm_smooth / period as f64 + plus_dm;
                    minus_dm_smooth = minus_dm_smooth - minus_dm_smooth / period as f64 + minus_dm;
                    tr_smooth = tr_smooth - tr_smooth / period as f64 + tr;
                }

                if i >= period && tr_smooth > 1e-12 {
                    let plus_di = plus_dm_smooth / tr_smooth;
                    let minus_di = minus_dm_smooth / tr_smooth;
                    let di_sum = plus_di + minus_di;
                    let dx = if di_sum > 1e-12 {
                        ((plus_di - minus_di).abs() / di_sum).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    // Smoothed ADX-like value
                    trend_strength[i] = dx;
                    // Direction: positive means bullish trend
                    trend_direction[i] = if plus_di > minus_di { 1.0 } else { -1.0 };
                }
            }
        }

        // Classify into trending (>0.25) vs ranging (<0.15)
        let mut trend_regime = vec![0.0; n];
        for i in 0..n {
            if trend_strength[i] > 0.25 {
                trend_regime[i] = trend_direction[i]; // +1 bullish trend, -1 bearish trend
            }
            // 0.0 = ranging
        }

        cols.push(("regime_trend_strength".to_string(), trend_strength));
        cols.push(("regime_trend_direction".to_string(), trend_direction));
        cols.push(("regime_trend_state".to_string(), trend_regime));
    }

    // ==========================================
    // 3. Squeeze Detection (Bollinger inside Keltner = compression)
    // ==========================================
    {
        let bb_period = 20;
        let kc_period = 20;
        let mut squeeze = vec![0.0; n];
        let mut squeeze_momentum = vec![0.0; n];

        for i in bb_period.max(kc_period)..n {
            // Bollinger Band width
            let bb_slice = &close[(i - bb_period + 1)..=i];
            let bb_mean = bb_slice.iter().sum::<f64>() / bb_period as f64;
            let bb_var = bb_slice.iter().map(|x| (x - bb_mean).powi(2)).sum::<f64>()
                / (bb_period as f64 - 1.0);
            let bb_std = bb_var.sqrt();
            let bb_upper = bb_mean + 2.0 * bb_std;
            let bb_lower = bb_mean - 2.0 * bb_std;

            // Keltner Channel width (ATR-based)
            let mut atr_sum = 0.0;
            for j in (i - kc_period + 1)..=i {
                if j > 0 {
                    let tr = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    atr_sum += tr;
                }
            }
            let atr = atr_sum / kc_period as f64;
            let kc_upper = bb_mean + 1.5 * atr;
            let kc_lower = bb_mean - 1.5 * atr;

            // Squeeze: BB inside KC
            if bb_upper < kc_upper && bb_lower > kc_lower {
                squeeze[i] = 1.0; // In squeeze (compression)
            } else {
                squeeze[i] = -1.0; // Expansion (released)
            }

            // Momentum during/after squeeze (linear regression of midline deviation)
            let midline = (bb_upper + bb_lower) / 2.0;
            squeeze_momentum[i] = (close[i] - midline) / atr.max(1e-10);
        }
        cols.push(("regime_squeeze".to_string(), squeeze));
        cols.push(("regime_squeeze_momentum".to_string(), squeeze_momentum));
    }

    // ==========================================
    // 4. Mean-Reversion vs Momentum Probability
    // Based on short-term autocorrelation sign
    // ==========================================
    {
        let window = 20;
        let mut mr_vs_mom = vec![0.0; n];
        for (i, mr_vs_mom_value) in mr_vs_mom.iter_mut().enumerate().skip(window + 1) {
            let mut positive_ac = 0;
            let mut negative_ac = 0;
            for j in (i - window + 1)..=i {
                if j >= 2 {
                    let r1 = close[j] - close[j - 1];
                    let r0 = close[j - 1] - close[j - 2];
                    if r1 * r0 > 0.0 {
                        positive_ac += 1; // Same direction = momentum
                    } else if r1 * r0 < 0.0 {
                        negative_ac += 1; // Reversal = mean-reversion
                    }
                }
            }
            let total = (positive_ac + negative_ac) as f64;
            if total > 0.0 {
                // +1 = pure momentum, -1 = pure mean-reversion, 0 = random
                *mr_vs_mom_value = (positive_ac as f64 - negative_ac as f64) / total;
            }
        }
        cols.push(("regime_mr_vs_momentum".to_string(), mr_vs_mom));
    }

    // ==========================================
    // 5. Range Expansion Index (REI)
    // ==========================================
    {
        let period = 8;
        let mut rei = vec![0.0; n];
        for (i, rei_value) in rei.iter_mut().enumerate().skip(period) {
            let mut sum_up = 0.0;
            let mut sum_abs = 0.0;
            for j in (i - period + 1)..=i {
                let range = high[j] - low[j];
                let dir = close[j] - open[j];
                sum_up += dir;
                sum_abs += range;
            }
            *rei_value = if sum_abs > 1e-10 {
                (sum_up / sum_abs).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        cols.push(("regime_rei".to_string(), rei));
    }

    // ==========================================
    // 6. Choppiness Index (ranging indicator, 0-100)
    // ==========================================
    {
        let period = 14;
        let mut chop = vec![50.0; n]; // default = neutral
        for (i, chop_value) in chop.iter_mut().enumerate().skip(period) {
            let mut atr_sum = 0.0;
            let mut period_high = f64::NEG_INFINITY;
            let mut period_low = f64::INFINITY;
            for j in (i - period + 1)..=i {
                if j > 0 {
                    let tr = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    atr_sum += tr;
                }
                if high[j] > period_high {
                    period_high = high[j];
                }
                if low[j] < period_low {
                    period_low = low[j];
                }
            }
            let hl_range = period_high - period_low;
            if hl_range > 1e-10 && atr_sum > 1e-10 {
                *chop_value = 100.0 * (atr_sum / hl_range).ln() / (period as f64).ln();
                *chop_value = chop_value.clamp(0.0, 100.0);
            }
        }
        // Normalize to 0-1
        for v in chop.iter_mut() {
            *v /= 100.0;
        }
        cols.push(("regime_choppiness".to_string(), chop));
    }

    // ==========================================
    // 7. Regime Change Detection (structural break via CUSUM)
    // ==========================================
    {
        let window = 50;
        let threshold = 3.0;
        let mut cusum_up = vec![0.0; n];
        let mut cusum_down = vec![0.0; n];
        let mut regime_change = vec![0.0; n];

        for i in window..n {
            let slice = &close[(i - window)..i];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
            let std = var.sqrt().max(1e-12);

            let z = (close[i] - mean) / std;
            let prev_up = if i > 0 { cusum_up[i - 1] } else { 0.0 };
            let prev_down = if i > 0 { cusum_down[i - 1] } else { 0.0 };

            cusum_up[i] = (prev_up + z - 0.5).max(0.0);
            cusum_down[i] = (prev_down - z - 0.5).max(0.0);

            if cusum_up[i] > threshold {
                regime_change[i] = 1.0; // Upward regime shift
                cusum_up[i] = 0.0;
            } else if cusum_down[i] > threshold {
                regime_change[i] = -1.0; // Downward regime shift
                cusum_down[i] = 0.0;
            }
        }
        cols.push(("regime_cusum_up".to_string(), cusum_up));
        cols.push(("regime_cusum_down".to_string(), cusum_down));
        cols.push(("regime_change_signal".to_string(), regime_change));
    }

    // ==========================================
    // 8. Entropy (Shannon entropy of price changes — randomness measure)
    // ==========================================
    {
        let window = 30;
        let n_bins = 10;
        let mut entropy = vec![0.0; n];
        for (i, entropy_value) in entropy.iter_mut().enumerate().skip(window) {
            let mut returns = Vec::with_capacity(window);
            for j in (i - window + 1)..=i {
                if j > 0 && close[j - 1].abs() > 1e-10 {
                    returns.push((close[j] / close[j - 1]).ln());
                }
            }
            if returns.len() < 5 {
                continue;
            }

            let min_r = returns.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_r = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_r - min_r).max(1e-12);

            let mut bins = vec![0_usize; n_bins];
            for &r in &returns {
                let bin = ((r - min_r) / range * (n_bins as f64 - 0.001)) as usize;
                bins[bin.min(n_bins - 1)] += 1;
            }

            let total = returns.len() as f64;
            let mut h = 0.0;
            for &count in &bins {
                if count > 0 {
                    let p = count as f64 / total;
                    h -= p * p.ln();
                }
            }
            // Normalize by max entropy (uniform distribution)
            let max_entropy = (n_bins as f64).ln();
            *entropy_value = if max_entropy > 1e-12 {
                h / max_entropy
            } else {
                0.0
            };
        }
        cols.push(("regime_entropy".to_string(), entropy));
    }

    cols
}
