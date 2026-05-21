use std::f64::consts::LN_2;

#[derive(Debug, Clone, Copy)]
pub struct VolEnsembleWeights {
    pub yang_zhang: f64,
    pub garman_klass: f64,
    pub rogers_satchell: f64,
    pub parkinson: f64,
}

impl VolEnsembleWeights {
    fn normalize(&self) -> Option<[f64; 4]> {
        let mut vals = [
            self.yang_zhang.max(0.0),
            self.garman_klass.max(0.0),
            self.rogers_satchell.max(0.0),
            self.parkinson.max(0.0),
        ];
        let total: f64 = vals.iter().sum();
        if total <= 0.0 {
            return None;
        }
        for v in &mut vals {
            *v /= total;
        }
        Some(vals)
    }
}

#[derive(Debug, Clone)]
pub struct StopTargetSettings {
    pub vol_estimator: String,
    pub vol_window: usize,
    pub ewma_lambda: f64,
    pub vol_horizon_bars: usize,
    pub tail_window: usize,
    pub tail_alpha: f64,
    pub tail_step: usize,
    pub tail_max_bars: usize,
    pub stop_k_vol: f64,
    pub stop_k_tail: f64,
    pub meta_label_min_dist: f64,
    pub regime_adx_trend: f64,
    pub regime_adx_range: f64,
    pub hurst_window: usize,
    pub hurst_trend: f64,
    pub hurst_range: f64,
    pub rr_trend: f64,
    pub rr_range: f64,
    pub rr_neutral: f64,
    pub min_risk_reward: f64,
    pub atr_stop_multiplier: f64,
    pub stop_target_mode: String,
    pub structure_lookback_bars: usize,
    pub structure_swing_window: usize,
    pub structure_min_atr_mult: f64,
    pub structure_max_atr_mult: f64,
    pub ema_fast_period: usize,
    pub ema_slow_period: usize,
    pub atr_period: usize,
    pub weights: Option<VolEnsembleWeights>,
    pub weights_trend: Option<VolEnsembleWeights>,
    pub weights_range: Option<VolEnsembleWeights>,
}

impl Default for StopTargetSettings {
    fn default() -> Self {
        let rr_trend = 2.5;
        let rr_range = 1.5;
        Self {
            vol_estimator: "yang_zhang".to_string(),
            vol_window: 50,
            ewma_lambda: 0.94,
            vol_horizon_bars: 5,
            tail_window: 100,
            tail_alpha: 0.975,
            tail_step: 5,
            tail_max_bars: 300_000,
            stop_k_vol: 1.0,
            stop_k_tail: 1.25,
            meta_label_min_dist: 0.0,
            regime_adx_trend: 25.0,
            regime_adx_range: 20.0,
            hurst_window: 100,
            hurst_trend: 0.55,
            hurst_range: 0.45,
            rr_trend,
            rr_range,
            rr_neutral: (rr_trend + rr_range) / 2.0,
            min_risk_reward: 2.0,
            atr_stop_multiplier: 1.5,
            stop_target_mode: "blend".to_string(),
            structure_lookback_bars: 120,
            structure_swing_window: 2,
            structure_min_atr_mult: 0.8,
            structure_max_atr_mult: 4.0,
            ema_fast_period: 20,
            ema_slow_period: 50,
            atr_period: 14,
            weights: None,
            weights_trend: None,
            weights_range: None,
        }
    }
}

fn safe_log(v: f64) -> f64 {
    v.max(1e-12).ln()
}

fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 {
        return values.to_vec();
    }
    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i + 1 >= window {
            out[i] = sum / window as f64;
            sum -= values[i + 1 - window];
        }
    }
    out
}

fn rolling_var(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 {
        return vec![0.0; n];
    }
    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for i in 0..n {
        sum += values[i];
        sumsq += values[i] * values[i];
        if i + 1 >= window {
            let mean = sum / window as f64;
            let var = (sumsq - sum * mean) / (window as f64 - 1.0);
            out[i] = var.max(0.0);
            sum -= values[i + 1 - window];
            sumsq -= values[i + 1 - window] * values[i + 1 - window];
        }
    }
    out
}

fn vol_parkinson(high: &[f64], low: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .map(|(h, l)| {
            let hl = safe_log(*h) - safe_log(*l);
            (hl * hl) / (4.0 * LN_2)
        })
        .collect()
}

fn vol_rogers_satchell(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let ho = safe_log(high[i]) - safe_log(open[i]);
        let hc = safe_log(high[i]) - safe_log(close[i]);
        let lo = safe_log(low[i]) - safe_log(open[i]);
        let lc = safe_log(low[i]) - safe_log(close[i]);
        out.push((ho * hc) + (lo * lc));
    }
    out
}

fn vol_garman_klass(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(close.len());
    let c1 = 2.0 * LN_2 - 1.0;
    for i in 0..close.len() {
        let h = safe_log(high[i]);
        let l = safe_log(low[i]);
        let o = safe_log(open[i]);
        let c = safe_log(close[i]);
        let hl = h - l;
        let co = c - o;
        out.push(0.5 * (hl * hl) - c1 * (co * co));
    }
    out
}

fn vol_yang_zhang(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
) -> Vec<f64> {
    let n = close.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut o = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        o.push(safe_log(open[i]));
        c.push(safe_log(close[i]));
        h.push(safe_log(high[i]));
        l.push(safe_log(low[i]));
    }

    let mut o_ret = vec![f64::NAN; n];
    let mut c_ret = vec![f64::NAN; n];
    let mut rs = vec![f64::NAN; n];

    for i in 1..n {
        o_ret[i] = o[i] - c[i - 1];
        c_ret[i] = c[i] - o[i];
        let ho = h[i] - o[i];
        let hc = h[i] - c[i];
        let lo = l[i] - o[i];
        let lc = l[i] - c[i];
        rs[i] = (ho * hc) + (lo * lc);
    }

    let k = if window > 1 {
        0.34 / (1.34 + (window as f64 + 1.0) / (window as f64 - 1.0))
    } else {
        0.0
    };

    let sigma_o2 = rolling_var(&o_ret, window);
    let sigma_c2 = rolling_var(&c_ret, window);
    let sigma_rs2 = rolling_mean(&rs, window);

    let mut sigma = vec![0.0; n];
    for i in 0..n {
        let mut val = sigma_o2[i] + k * sigma_c2[i] + (1.0 - k) * sigma_rs2[i];
        if !val.is_finite() {
            val = 0.0;
        }
        sigma[i] = val.max(0.0).sqrt();
    }
    sigma
}

fn vol_ewma(close: &[f64], window: usize, lam: f64) -> Vec<f64> {
    let n = close.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let lam = if lam <= 0.0 || lam >= 1.0 { 0.94 } else { lam };
    let mut r = Vec::with_capacity(n - 1);
    for i in 1..n {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    let mut var = vec![f64::NAN; n];
    let init = if window <= 1 {
        mean(&r.iter().map(|v| v * v).collect::<Vec<_>>())
    } else {
        let span = window.min(r.len());
        mean(&r[..span].iter().map(|v| v * v).collect::<Vec<_>>())
    };
    if n > 1 {
        var[1] = if init.is_finite() { init } else { 0.0 };
    }
    for i in 1..r.len() {
        let prev = if var[i].is_finite() { var[i] } else { init };
        var[i + 1] = (lam * prev) + ((1.0 - lam) * (r[i] * r[i]));
    }
    let mut sigma = vec![0.0; n];
    let mut last = 0.0;
    for i in 0..n {
        let val = var[i];
        if val.is_finite() && val > 0.0 {
            last = val.sqrt();
            sigma[i] = last;
        } else {
            sigma[i] = last;
        }
    }
    sigma
}

use neoethos_core::utils::{mean, stddev_sample};

fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

fn median_ignore_nan(values: &[f64]) -> f64 {
    neoethos_core::utils::median_ignore_nan(values)
}

pub struct VolatilityEstimateInput<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub window: usize,
    pub method: &'a str,
    pub weights: Option<VolEnsembleWeights>,
    pub ewma_lambda: f64,
}

pub fn estimate_volatility(input: VolatilityEstimateInput<'_>) -> Vec<f64> {
    let VolatilityEstimateInput {
        open,
        high,
        low,
        close,
        window,
        method,
        weights,
        ewma_lambda,
    } = input;
    let method = method.to_lowercase();
    if method == "ensemble" || method == "mix" || method == "blend" {
        let v_pk = vol_parkinson(high, low);
        let sigma_pk = rolling_mean(&v_pk, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect::<Vec<_>>();
        let v_gk = vol_garman_klass(open, high, low, close);
        let sigma_gk = rolling_mean(&v_gk, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect::<Vec<_>>();
        let v_rs = vol_rogers_satchell(open, high, low, close);
        let sigma_rs = rolling_mean(&v_rs, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect::<Vec<_>>();
        let sigma_yz = vol_yang_zhang(open, high, low, close, window);

        let mut out = vec![0.0; close.len()];
        let weights = weights.and_then(|w| w.normalize());
        for i in 0..out.len() {
            let stacked = [sigma_yz[i], sigma_gk[i], sigma_rs[i], sigma_pk[i]];
            if let Some(w) = weights {
                out[i] =
                    stacked[0] * w[0] + stacked[1] * w[1] + stacked[2] * w[2] + stacked[3] * w[3];
            } else {
                let med = median_ignore_nan(&stacked);
                out[i] = if med.is_finite() { med } else { 0.0 };
            }
        }
        return out;
    }
    if method == "parkinson" || method == "park" {
        let v = vol_parkinson(high, low);
        return rolling_mean(&v, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect();
    }
    if method == "garman_klass" || method == "gk" {
        let v = vol_garman_klass(open, high, low, close);
        return rolling_mean(&v, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect();
    }
    if method == "rogers_satchell" || method == "rs" {
        let v = vol_rogers_satchell(open, high, low, close);
        return rolling_mean(&v, window)
            .into_iter()
            .map(|v| v.max(0.0).sqrt())
            .collect();
    }
    if method == "ewma" || method == "riskmetrics" {
        return vol_ewma(close, window, ewma_lambda);
    }
    vol_yang_zhang(open, high, low, close, window)
}

pub fn estimate_expected_shortfall(close: &[f64], window: usize, alpha: f64) -> Option<f64> {
    if window <= 2 || close.len() <= 2 {
        return None;
    }
    let mut r = Vec::with_capacity(close.len() - 1);
    for i in 1..close.len() {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if r.len() < window {
        return None;
    }
    let tail = &r[r.len() - window..];
    let mut sorted = tail.to_vec();
    let q_idx = ((1.0 - alpha).clamp(0.0, 1.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let (_, q_ref, _) = sorted.select_nth_unstable_by(q_idx, cmp_f64);
    let q = *q_ref;
    let losses: Vec<f64> = tail.iter().cloned().filter(|v| *v <= q).collect();
    if losses.is_empty() {
        return None;
    }
    Some(losses.iter().map(|v| v.abs()).sum::<f64>() / losses.len() as f64)
}

pub fn estimate_expected_shortfall_series(
    close: &[f64],
    window: usize,
    alpha: f64,
    step: usize,
    max_bars: usize,
) -> Option<Vec<f64>> {
    if window <= 2 || close.len() <= 2 || close.len() > max_bars {
        return None;
    }
    let mut r = Vec::with_capacity(close.len() - 1);
    for i in 1..close.len() {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if r.len() < window {
        return None;
    }
    let mut es = vec![f64::NAN; r.len()];
    let step = step.max(1);
    let mut i = window - 1;
    while i < r.len() {
        let win = &r[i + 1 - window..=i];
        let mut sorted = win.to_vec();
        let q_idx = ((1.0 - alpha).clamp(0.0, 1.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        let (_, q_ref, _) = sorted.select_nth_unstable_by(q_idx, cmp_f64);
        let q = *q_ref;
        let losses: Vec<f64> = win.iter().cloned().filter(|v| *v <= q).collect();
        if !losses.is_empty() {
            es[i] = losses.iter().map(|v| v.abs()).sum::<f64>() / losses.len() as f64;
        }
        i += step;
    }
    let mut last = 0.0;
    for v in &mut es {
        if v.is_finite() {
            last = *v;
        } else {
            *v = last;
        }
    }
    let mut out = Vec::with_capacity(close.len());
    out.push(f64::NAN);
    out.extend(es);
    Some(out)
}

pub fn estimate_hurst(close: &[f64], window: usize, max_lag: usize) -> Option<f64> {
    if window <= 10 || close.len() <= window {
        return None;
    }
    let start = close.len().saturating_sub(window + 1);
    let mut series = Vec::with_capacity(window);
    for i in (start + 1)..close.len() {
        series.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if series.len() < 20 {
        return None;
    }
    let max_lag = max_lag.min(series.len() / 2).max(2);
    let mut lags = Vec::with_capacity(max_lag.saturating_sub(1));
    let mut tau = Vec::with_capacity(max_lag.saturating_sub(1));
    for lag in 2..=max_lag {
        let mut diffs = Vec::with_capacity(series.len() - lag);
        for i in lag..series.len() {
            diffs.push(series[i] - series[i - lag]);
        }
        let std = stddev(&diffs);
        if std > 0.0 {
            lags.push(lag as f64);
            tau.push(std);
        }
    }
    if lags.len() < 2 {
        return None;
    }
    let log_lags: Vec<f64> = lags.iter().map(|v| v.ln()).collect();
    let log_tau: Vec<f64> = tau.iter().map(|v| v.ln()).collect();
    let slope = linreg_slope(&log_lags, &log_tau)?;
    Some(slope)
}

fn linreg_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    Some((n * sum_xy - sum_x * sum_y) / denom)
}

fn stddev(values: &[f64]) -> f64 {
    let m = mean(values);
    stddev_sample(values, m)
}

fn compute_ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut out = Vec::with_capacity(values.len());
    let mut prev = values[0];
    out.push(prev);
    for &value in values.iter().skip(1) {
        prev = alpha * value + (1.0 - alpha) * prev;
        out.push(prev);
    }
    out
}

fn compute_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr[i] = tr1.max(tr2).max(tr3);
    }
    compute_ema(&tr, period)
}

fn compute_adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Option<f64> {
    let n = close.len();
    if n <= period + 1 {
        return None;
    }
    let mut tr = vec![0.0; n];
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr[i] = tr1.max(tr2).max(tr3);

        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }
    }

    let mut tr_sum: f64 = tr[1..=period].iter().sum();
    let mut plus_sum: f64 = plus_dm[1..=period].iter().sum();
    let mut minus_sum: f64 = minus_dm[1..=period].iter().sum();

    let mut dx = Vec::with_capacity(n.saturating_sub(period + 1));
    for i in (period + 1)..n {
        tr_sum = tr_sum - (tr_sum / period as f64) + tr[i];
        plus_sum = plus_sum - (plus_sum / period as f64) + plus_dm[i];
        minus_sum = minus_sum - (minus_sum / period as f64) + minus_dm[i];

        let plus_di = if tr_sum > 0.0 {
            100.0 * plus_sum / tr_sum
        } else {
            0.0
        };
        let minus_di = if tr_sum > 0.0 {
            100.0 * minus_sum / tr_sum
        } else {
            0.0
        };
        let denom = (plus_di + minus_di).max(1e-9);
        dx.push(100.0 * (plus_di - minus_di).abs() / denom);
    }

    if dx.len() < period {
        return None;
    }

    let mut adx = mean(&dx[..period]);
    for value in dx.iter().skip(period) {
        adx = ((adx * (period as f64 - 1.0)) + *value) / period as f64;
    }
    Some(adx)
}

pub fn infer_regime(
    _open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
) -> String {
    let adx_val = compute_adx(high, low, close, settings.atr_period);
    let hurst_val = estimate_hurst(close, settings.hurst_window, 20);

    if let (Some(adx), Some(hurst)) = (adx_val, hurst_val) {
        if adx >= settings.regime_adx_trend && hurst >= settings.hurst_trend {
            return "trend".to_string();
        }
        if adx <= settings.regime_adx_range && hurst <= settings.hurst_range {
            return "range".to_string();
        }
    }

    if let Some(adx) = adx_val {
        if adx >= settings.regime_adx_trend {
            return "trend".to_string();
        }
        if adx <= settings.regime_adx_range {
            return "range".to_string();
        }
    }

    if let Some(hurst) = hurst_val {
        if hurst >= settings.hurst_trend {
            return "trend".to_string();
        }
        if hurst <= settings.hurst_range {
            return "range".to_string();
        }
    }

    let ema_fast = compute_ema(close, settings.ema_fast_period);
    let ema_slow = compute_ema(close, settings.ema_slow_period);
    let atr = compute_atr(high, low, close, settings.atr_period);
    if let (Some(ef), Some(es), Some(a)) = (ema_fast.last(), ema_slow.last(), atr.last())
        && *a > 0.0
    {
        let spread = (ef - es).abs();
        let strength = spread / a;
        if strength >= 0.6 {
            return "trend".to_string();
        }
        if strength <= 0.3 {
            return "range".to_string();
        }
    }

    "neutral".to_string()
}

fn regime_rr(regime: &str, settings: &StopTargetSettings) -> f64 {
    match regime {
        "trend" => settings.rr_trend,
        "range" => settings.rr_range,
        _ => settings.rr_neutral,
    }
}

fn atr_last_distances(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
    regime: &str,
) -> Option<(f64, f64, f64)> {
    let atr = compute_atr(high, low, close, settings.atr_period)
        .last()
        .copied()
        .unwrap_or(f64::NAN);
    if !atr.is_finite() || atr <= 0.0 {
        return None;
    }
    let sl = (atr * settings.atr_stop_multiplier.max(0.1)).max(settings.meta_label_min_dist);
    if !sl.is_finite() || sl <= 0.0 {
        return None;
    }
    let rr_floor = settings.min_risk_reward.max(1.5);
    let rr = regime_rr(regime, settings).max(rr_floor);
    let tp = (sl * rr).max(settings.meta_label_min_dist);
    if !tp.is_finite() || tp <= 0.0 {
        return None;
    }
    Some((sl, tp, rr))
}

fn swing_levels(
    high: &[f64],
    low: &[f64],
    lookback: usize,
    swing_window: usize,
) -> Option<(f64, f64)> {
    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return None;
    }
    let span = (2 * swing_window.max(1)) + 1;
    let lb = lookback.max(span + 2).min(high.len());
    if lb < span {
        return None;
    }
    let hs = &high[(high.len() - lb)..];
    let ls = &low[(low.len() - lb)..];
    let half = swing_window.max(1);
    let eps = 1e-12;

    let mut swing_highs: Vec<f64> = Vec::new();
    let mut swing_lows: Vec<f64> = Vec::new();
    for i in half..(lb - half) {
        let mut max_w = f64::NEG_INFINITY;
        let mut min_w = f64::INFINITY;
        for j in (i - half)..=(i + half) {
            max_w = max_w.max(hs[j]);
            min_w = min_w.min(ls[j]);
        }
        if hs[i].is_finite() && max_w.is_finite() && hs[i] >= (max_w - eps) {
            swing_highs.push(hs[i]);
        }
        if ls[i].is_finite() && min_w.is_finite() && ls[i] <= (min_w + eps) {
            swing_lows.push(ls[i]);
        }
    }

    let resistance = swing_highs
        .last()
        .copied()
        .unwrap_or_else(|| hs.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let support = swing_lows
        .last()
        .copied()
        .unwrap_or_else(|| ls.iter().copied().fold(f64::INFINITY, f64::min));
    if !resistance.is_finite() || !support.is_finite() {
        return None;
    }
    Some((support, resistance))
}

fn structure_distances(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
    signal: i8,
    regime: &str,
) -> Option<(f64, f64, f64)> {
    let (support, resistance) = swing_levels(
        high,
        low,
        settings.structure_lookback_bars.max(20),
        settings.structure_swing_window.max(1),
    )?;
    let px = *close.last()?;
    if !px.is_finite() || px <= 0.0 {
        return None;
    }

    let (sl_raw, tp_raw) = if signal > 0 {
        (px - support, resistance - px)
    } else if signal < 0 {
        (resistance - px, px - support)
    } else {
        let down = px - support;
        let up = resistance - px;
        (
            down.max(0.0).min(up.max(0.0)),
            down.max(0.0).max(up.max(0.0)),
        )
    };
    if !sl_raw.is_finite() || sl_raw <= 0.0 {
        return None;
    }

    let atr = compute_atr(high, low, close, settings.atr_period)
        .last()
        .copied()
        .unwrap_or(f64::NAN);
    let min_dist = settings.meta_label_min_dist.max(0.0);
    let mut sl = sl_raw.max(min_dist);
    if atr.is_finite() && atr > 0.0 {
        let lo = (atr * settings.structure_min_atr_mult.max(0.1)).max(min_dist);
        let hi = (atr
            * settings
                .structure_max_atr_mult
                .max(settings.structure_min_atr_mult))
        .max(lo);
        sl = sl.clamp(lo, hi);
    }

    let rr_floor = settings.min_risk_reward.max(1.5);
    let rr_regime = regime_rr(regime, settings).max(rr_floor);
    let rr_struct = if tp_raw.is_finite() && tp_raw > 0.0 {
        tp_raw / sl.max(1e-9)
    } else {
        rr_regime
    };
    let rr = rr_struct.max(rr_regime).clamp(rr_floor, 6.0);
    let mut tp = (sl * rr).max(settings.meta_label_min_dist);
    if tp_raw.is_finite() && tp_raw > 0.0 {
        tp = tp.max(tp_raw);
    }
    if !tp.is_finite() || tp <= 0.0 {
        return None;
    }
    Some((sl, tp, rr))
}

pub fn compute_stop_distance_series(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
) -> Option<Vec<f64>> {
    if close.len() < settings.vol_window.max(settings.tail_window).max(5) {
        return None;
    }

    let regime = infer_regime(open, high, low, close, settings);
    let weights = match regime.as_str() {
        "trend" => settings.weights_trend.or(settings.weights),
        "range" => settings.weights_range.or(settings.weights),
        _ => settings.weights,
    };

    let sigma = estimate_volatility(VolatilityEstimateInput {
        open,
        high,
        low,
        close,
        window: settings.vol_window,
        method: &settings.vol_estimator,
        weights,
        ewma_lambda: settings.ewma_lambda,
    });
    let scale = (settings.vol_horizon_bars.max(1) as f64).sqrt();
    let vol_dist: Vec<f64> = close
        .iter()
        .zip(sigma.iter())
        .map(|(c, s)| c * s * scale)
        .collect();

    let es_series = estimate_expected_shortfall_series(
        close,
        settings.tail_window,
        settings.tail_alpha,
        settings.tail_step,
        settings.tail_max_bars,
    );
    let tail_dist = if let Some(es) = es_series {
        close
            .iter()
            .zip(es.iter())
            .map(|(c, s)| c * s * scale)
            .collect::<Vec<_>>()
    } else {
        vec![0.0; close.len()]
    };

    let mut dist = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let base = (settings.stop_k_vol * vol_dist[i]).max(settings.stop_k_tail * tail_dist[i]);
        dist.push(base.max(settings.meta_label_min_dist));
    }

    let med = median_ignore_nan(&dist);
    if !med.is_finite() || med <= 0.0 {
        return None;
    }
    for v in &mut dist {
        if !v.is_finite() {
            *v = med;
        }
    }

    Some(dist)
}

pub fn infer_stop_target_pips(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
    pip_size: f64,
    signal: i8,
) -> Option<(f64, f64, f64)> {
    if close.len() < settings.vol_window.max(settings.tail_window).max(5) {
        return None;
    }

    let regime = infer_regime(open, high, low, close, settings);
    let weights = match regime.as_str() {
        "trend" => settings.weights_trend.or(settings.weights),
        "range" => settings.weights_range.or(settings.weights),
        _ => settings.weights,
    };

    let sigma = estimate_volatility(VolatilityEstimateInput {
        open,
        high,
        low,
        close,
        window: settings.vol_window,
        method: &settings.vol_estimator,
        weights,
        ewma_lambda: settings.ewma_lambda,
    });
    let &sigma_last = sigma.last()?;
    let es = estimate_expected_shortfall(close, settings.tail_window, settings.tail_alpha)?;
    let &price = close.last()?;
    let scale = (settings.vol_horizon_bars.max(1) as f64).sqrt();
    let vol_dist = price * sigma_last * scale;
    let tail_dist = price * es * scale;
    let dist = (settings.stop_k_vol * vol_dist)
        .max(settings.stop_k_tail * tail_dist)
        .max(settings.meta_label_min_dist);
    let base = if dist.is_finite() && dist > 0.0 {
        let rr = regime_rr(&regime, settings);
        Some((dist, dist * rr, rr))
    } else {
        None
    };
    let atr = atr_last_distances(high, low, close, settings, &regime);
    let structure = structure_distances(high, low, close, settings, signal, &regime);

    let mode = settings.stop_target_mode.trim().to_ascii_lowercase();
    let selected = if matches!(mode.as_str(), "structure" | "market_structure" | "swing") {
        structure.or(atr).or(base)
    } else if matches!(mode.as_str(), "atr" | "atr_only") {
        atr.or(base).or(structure)
    } else {
        let base_eff = base.or(atr);
        match (structure, base_eff, atr) {
            (Some(s), Some(b), _) => {
                let w_struct = if regime == "trend" {
                    0.70
                } else if regime == "range" {
                    0.35
                } else {
                    0.55
                };
                let w_atr = 1.0 - w_struct;
                let sl = (w_struct * s.0) + (w_atr * b.0);
                let rr_floor = settings.min_risk_reward.max(1.5);
                let rr = ((w_struct * s.2) + (w_atr * b.2)).max(rr_floor);
                let tp = (sl * rr).max((w_struct * s.1) + (w_atr * b.1));
                Some((sl, tp, rr))
            }
            _ => structure.or(base_eff).or(atr),
        }
    };
    let (sl_dist, tp_dist, rr_selected) = selected?;
    if !sl_dist.is_finite() || !tp_dist.is_finite() || sl_dist <= 0.0 || tp_dist <= 0.0 {
        return None;
    }
    let mut sl_pips = sl_dist / pip_size.max(1e-9);
    let mut tp_pips = tp_dist / pip_size.max(1e-9);
    if !sl_pips.is_finite() || !tp_pips.is_finite() || sl_pips <= 0.0 || tp_pips <= 0.0 {
        return None;
    }
    let rr_floor = settings.min_risk_reward.max(1.5);
    let mut rr_out = (tp_pips / sl_pips.max(1e-9)).max(rr_selected.max(0.0));
    if rr_out < rr_floor {
        tp_pips = sl_pips * rr_floor;
        rr_out = rr_floor;
    }
    // Ensure positive bounded values.
    sl_pips = sl_pips.max(1e-9);
    Some((sl_pips, tp_pips.max(1e-9), rr_out))
}
