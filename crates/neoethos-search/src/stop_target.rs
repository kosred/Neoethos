use std::f64::consts::LN_2;
use std::sync::OnceLock;

/// Why a per-bar stop-distance series could not be built.
///
/// This is deliberately NOT an `Option`. The old `Option` collapsed two
/// unrelated answers into one `None`: "this series is too short to have a
/// stop" (legitimate — fall back to the gene's fixed pips) and "I declined to
/// compute the tail term on a series this long" (not legitimate — the tail
/// term IS part of the stop, so dropping it evaluates a different strategy).
/// The second one was then turned into `tail_dist = 0` by the caller and
/// nothing anywhere said so. Naming the cases is what makes the second one
/// impossible to absorb by accident.
#[derive(Debug, Clone, PartialEq)]
pub enum StopDistanceError {
    /// Fewer bars than the volatility / tail windows need.
    TooShort { bars: usize, needed: usize },
    /// The series is longer than the configured `tail_max_bars`.
    TailCapExceeded { bars: usize, cap: usize },
    /// A caller-supplied scalar was non-finite or non-positive.
    InvalidScalar { name: &'static str, value: f64 },
    /// Every distance came out non-finite / non-positive.
    Degenerate { median: f64 },
}

impl std::fmt::Display for StopDistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { bars, needed } => write!(
                f,
                "stop-distance series needs at least {needed} bars, got {bars}"
            ),
            Self::TailCapExceeded { bars, cap } => write!(
                f,
                "stop-distance series refused: {bars} bars exceeds the configured \
                 models.stop_target_runtime.tail_max_bars = {cap}. The expected-shortfall \
                 tail term is part of the stop — skipping it would score this dataset on a \
                 DIFFERENT stop than the walk-forward and live paths compute. Raise the cap \
                 (0 = no cap, the default) or shorten the series"
            ),
            Self::InvalidScalar { name, value } => {
                write!(f, "stop-distance series got a non-usable {name} = {value}")
            }
            Self::Degenerate { median } => write!(
                f,
                "stop-distance series is degenerate: median distance = {median}"
            ),
        }
    }
}

impl std::error::Error for StopDistanceError {}

/// Process-wide adaptive-stop cost caps — the typed mirror of
/// `neoethos_core::config::StopTargetRuntimeConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopTargetRuntimeOverrides {
    /// `0` = no cap. See [`StopDistanceError::TailCapExceeded`].
    pub tail_max_bars: usize,
    /// `1` = sample the tail every bar (position-invariant). See
    /// [`StopTargetSettings::tail_step`].
    pub tail_step: usize,
}

impl Default for StopTargetRuntimeOverrides {
    fn default() -> Self {
        Self {
            tail_max_bars: 0,
            tail_step: 1,
        }
    }
}

impl StopTargetRuntimeOverrides {
    /// Config-driven constructor. A
    /// `stop_target_from_settings_default_matches_default` test guarantees a
    /// fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        Self {
            tail_max_bars: s.models.stop_target_runtime.tail_max_bars,
            // `0` is meaningless for a stride; fall back to the exact default
            // rather than silently dividing by zero downstream.
            tail_step: s.models.stop_target_runtime.tail_step.max(1),
        }
    }
}

static STOP_TARGET_RUNTIME_OVERRIDES: OnceLock<StopTargetRuntimeOverrides> = OnceLock::new();

/// Install process-wide adaptive-stop caps. First install wins.
pub fn install_stop_target_runtime_overrides(
    overrides: StopTargetRuntimeOverrides,
) -> Result<(), StopTargetRuntimeOverrides> {
    STOP_TARGET_RUNTIME_OVERRIDES.set(overrides)
}

/// Config-driven install. Idempotent — called once at startup from
/// `install_search_runtime_overrides_from_settings`.
pub fn install_stop_target_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = STOP_TARGET_RUNTIME_OVERRIDES.set(StopTargetRuntimeOverrides::from_settings(s));
}

/// The installed caps, or the deterministic defaults when nothing installed.
pub fn current_stop_target_runtime_overrides() -> StopTargetRuntimeOverrides {
    STOP_TARGET_RUNTIME_OVERRIDES.get().copied().unwrap_or_default()
}

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
    /// Sample the rolling expected shortfall every `tail_step` bars and carry
    /// the value forward between samples. `1` = every bar (the default).
    ///
    /// This is NOT only a speed knob. The sampling grid is anchored at the
    /// START of the series (`i = window - 1`, then `+= step`), so with
    /// `step > 1` two callers whose slices differ by an offset that is not a
    /// multiple of `step` sample DIFFERENT bars. Measured on EURUSD M5 with
    /// the old default of 5: the base series over the trailing 300 001 bars
    /// differs from the same bars of the full-series base by up to **86 %**
    /// per bar, while the medians agree to 0.006 % — which is why nothing
    /// noticed. The live loop's rolling buffer shifts its start every bar, so
    /// that divergence fires continuously, not rarely.
    ///
    /// `1` is position-invariant by construction. It cost 574 ms instead of
    /// 206 ms over 1 054 320 bars (once per combo) and moved every median by
    /// under 0.02 %. Raise it only to buy that time back, knowing what it
    /// costs in agreement between the scoring, walk-forward and live paths.
    pub tail_step: usize,
    /// Cost cap on the rolling expected-shortfall series. `0` = no cap.
    ///
    /// Was a hardcoded `300_000` here with no config recipient until
    /// 2026-08-04. See [`neoethos_core::config::StopTargetRuntimeConfig`] for
    /// what that cost: 300 000 bars gave a median EURUSD M5 base stop of
    /// 18.09 pips and 300 001 bars gave 5.81 pips, because the caller turned
    /// "cap hit" into "the tail term is zero". Exceeding the cap is now a
    /// [`StopDistanceError::TailCapExceeded`], never a silent zero.
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
            // Every bar. The old `5` made the tail term depend on where the
            // caller's slice happened to start (see the field doc).
            tail_step: 1,
            // No cap. The old `300_000` silently dropped the tail term above it.
            tail_max_bars: 0,
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

/// Rolling expected shortfall, sampled every `step` bars and carried forward.
///
/// `None` means only "there is not enough data here" — it no longer also means
/// "I declined". The length cap that used to live in this signature moved to
/// [`compute_stop_distance_series`], where it can be reported as a named error
/// instead of being flattened into a zero tail distance by the caller.
pub fn estimate_expected_shortfall_series(
    close: &[f64],
    window: usize,
    alpha: f64,
    step: usize,
) -> Option<Vec<f64>> {
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

/// Per-bar stop distance (price units) = `max(k_vol × vol, k_tail × tail)`.
///
/// Returns a typed [`StopDistanceError`] rather than `None` so a caller cannot
/// treat "too long to compute" the same way it treats "too short to have a
/// stop". Both used to arrive as `None`; only one of them is a reason to fall
/// back to fixed pips.
pub fn compute_stop_distance_series(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
) -> Result<Vec<f64>, StopDistanceError> {
    // `tail_window + 1`: the tail estimator consumes RETURNS, so it needs one
    // bar more than its window. The old guard used `tail_window`, which let a
    // series of exactly `tail_window` bars past the length check and into the
    // `es_series == None` branch — where the caller then substituted a zero
    // tail and returned a vol-only stop. Same hole as the 300 000-bar cliff,
    // one bar wide. `infer_stop_target_pips` already refused at that length,
    // so this also makes the scalar and series paths agree.
    let needed = settings
        .vol_window
        .max(settings.tail_window.saturating_add(1))
        .max(5);
    if close.len() < needed {
        return Err(StopDistanceError::TooShort {
            bars: close.len(),
            needed,
        });
    }
    // The cap is a COST knob. It is checked HERE, before any work, and it
    // refuses by name — it never silently changes the formula. Turning "I
    // cannot compute this" into "the answer is zero" is how the 300 000-bar
    // cliff survived from v0.4.19 to 2026-08-04.
    if settings.tail_max_bars > 0 && close.len() > settings.tail_max_bars {
        return Err(StopDistanceError::TailCapExceeded {
            bars: close.len(),
            cap: settings.tail_max_bars,
        });
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

    // No `else { vec![0.0; ...] }`. A missing tail term is a missing INPUT,
    // not a tail of zero — the `stop_k_tail = 1.25` branch would then never
    // win the `max`, which is exactly the 3.11× stop change measured at the
    // old 300 000-bar boundary. The only remaining reason this can be `None`
    // is too few returns, which the length guard above already rejects, so it
    // reports as `TooShort` rather than being absorbed.
    let es = estimate_expected_shortfall_series(
        close,
        settings.tail_window,
        settings.tail_alpha,
        settings.tail_step,
    )
    .ok_or(StopDistanceError::TooShort {
        bars: close.len(),
        needed,
    })?;
    let tail_dist: Vec<f64> = close
        .iter()
        .zip(es.iter())
        .map(|(c, s)| c * s * scale)
        .collect();

    let mut dist = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let base = (settings.stop_k_vol * vol_dist[i]).max(settings.stop_k_tail * tail_dist[i]);
        dist.push(base.max(settings.meta_label_min_dist));
    }

    let med = median_ignore_nan(&dist);
    if !med.is_finite() || med <= 0.0 {
        return Err(StopDistanceError::Degenerate { median: med });
    }
    for v in &mut dist {
        if !v.is_finite() {
            *v = med;
        }
    }

    Ok(dist)
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

/// Per-bar volatility-adaptive `(sl_pips, tp_pips)` for a whole dataset — the
/// foundation of the adaptive-stop mode (Stage 1, 2026-07-23). It turns the
/// vol/tail stop-distance series ([`compute_stop_distance_series`], price units)
/// into PIP distances the existing fixed-pip backtest/live paths already
/// consume, so wiring it in later changes nothing except that the value now
/// varies per ENTRY bar:
///   `sl_pips[i] = vol_mult * dist[i] / pip_size`,   `tp_pips[i] = rr * sl_pips[i]`.
///
/// `vol_mult` is the gene's searchable knob (how many vol-units the stop sits
/// at); `rr` is the reward:risk multiple kept OUT of the stop so profit stays a
/// fixed multiple of the risk (the operator wants ~2R). Returns a typed
/// [`StopDistanceError`] when the series can't be built or a scalar is
/// non-finite/non-positive; only [`StopDistanceError::TooShort`] means "fall
/// back to the gene's fixed `sl_pips`/`tp_pips`".
///
/// Deterministic and allocation-simple on purpose: the same inputs MUST yield
/// the same series on the CPU backtest and (Stage 3) the GPU kernel and (Stage
/// 4) the live loop, or the adaptive path breaks CPU↔GPU / backtest↔live parity.
pub fn adaptive_sl_tp_pips_series(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
    pip_size: f64,
    vol_mult: f64,
    rr: f64,
) -> Result<(Vec<f64>, Vec<f64>), StopDistanceError> {
    for (name, value) in [("pip_size", pip_size), ("vol_mult", vol_mult), ("rr", rr)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(StopDistanceError::InvalidScalar { name, value });
        }
    }
    let dist = compute_stop_distance_series(open, high, low, close, settings)?;
    let mut sl_pips = Vec::with_capacity(dist.len());
    let mut tp_pips = Vec::with_capacity(dist.len());
    for d in dist {
        let sl = (d * vol_mult / pip_size).max(1e-9);
        sl_pips.push(sl);
        tp_pips.push((sl * rr).max(1e-9));
    }
    Ok((sl_pips, tp_pips))
}

/// The per-bar base stop distance in PIPS (multiplier 1) for adaptive stops,
/// using the OPEN-INDEPENDENT Parkinson high/low range estimator (+ the tail
/// expected-shortfall on close). This matters for parity: the discovery scoring
/// path has the full OHLCV, but the gathered / walk-forward-window validation
/// paths only carry high/low/close — using an estimator that needs `open` would
/// make the validation base series differ from the scoring one and evaluate a
/// different strategy. Parkinson depends only on high/low/close, which every
/// path has, so the base is identical everywhere. A gene's `stop_vol_mult` then
/// scales this shared series.
///
/// This is the ONE production entry point for the shared base, so it is also
/// where the process-wide cost cap is read — every caller therefore sees the
/// same `tail_max_bars`. That is precisely the property the old hardcoded
/// `300_000` broke: the cap fired for the scoring path (full series) and not
/// for the walk-forward or live paths (windows), silently.
pub fn adaptive_base_pips_series(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    pip_size: f64,
) -> Result<Vec<f64>, StopDistanceError> {
    if !(pip_size.is_finite() && pip_size > 0.0) {
        return Err(StopDistanceError::InvalidScalar {
            name: "pip_size",
            value: pip_size,
        });
    }
    let overrides = current_stop_target_runtime_overrides();
    let mut settings = StopTargetSettings::default();
    settings.vol_estimator = "parkinson".to_string();
    settings.tail_max_bars = overrides.tail_max_bars;
    settings.tail_step = overrides.tail_step;
    // `open` is unused by the Parkinson estimator — pass `close` as a harmless
    // placeholder so callers without an open series get the identical result.
    let dist = compute_stop_distance_series(close, high, low, close, &settings)?;
    Ok(dist.iter().map(|d| (d / pip_size).max(1e-9)).collect())
}

/// Whether adaptive volatility-scaled stops are enabled for this process — the
/// Stage-2c dev gate. `NEOETHOS_ADAPTIVE_STOPS=1|true|on`. Default off. When on,
/// gene generation seeds a searchable `stop_vol_mult` and the discovery backtest
/// scales each entry's stop by the bar's volatility. Stage 5 productionizes this
/// to a `Settings` field + auto-enable (mirrors the fused-eval rollout).
pub fn adaptive_stops_enabled() -> bool {
    // Default ON: the operator wants the stop to scale with volatility
    // automatically — no toggle, no env var to remember. `NEOETHOS_ADAPTIVE_STOPS`
    // is now only an emergency ESCAPE HATCH: set it to 0/false/off to fall back to
    // the old fixed-pip stops (e.g. to reproduce a pre-adaptive run). Any other
    // value, or unset, keeps adaptive on.
    match std::env::var("NEOETHOS_ADAPTIVE_STOPS") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

/// Reward:risk multiple for adaptive stops (`TP = rr × stop`). The operator wants
/// ~2R kept out of the stop. `NEOETHOS_ADAPTIVE_STOP_RR`, default 2.0.
pub fn adaptive_stops_rr() -> f64 {
    std::env::var("NEOETHOS_ADAPTIVE_STOP_RR")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or(2.0)
}

#[cfg(test)]
mod adaptive_stop_tests {
    use super::*;

    #[test]
    fn adaptive_sl_tp_series_scales_with_vol_and_holds_rr() {
        // Calm first half, ~6x more volatile second half — the adaptive stop
        // must WIDEN where the market is choppier (the whole point of ATR/vol
        // scaling) while the reward:risk stays pinned.
        let n = 400usize;
        let (mut open, mut high, mut low, mut close) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let amp = if i < n / 2 { 0.0005 } else { 0.0030 };
            let c = 1.10 + amp * ((i as f64) * 0.3).sin();
            open.push(c);
            close.push(c);
            high.push(c + amp);
            low.push(c - amp);
        }
        let s = StopTargetSettings::default();
        let (sl, tp) =
            adaptive_sl_tp_pips_series(&open, &high, &low, &close, &s, 0.0001, 1.0, 2.0)
                .expect("series builds on a long-enough dataset");
        assert_eq!(sl.len(), n);
        assert_eq!(tp.len(), n);
        // Reward:risk held exactly, per bar.
        for i in 0..n {
            assert!((tp[i] - 2.0 * sl[i]).abs() < 1e-9, "TP must be exactly rr*SL");
            assert!(sl[i] > 0.0);
        }
        // The volatile half's median stop must exceed the calm half's.
        let median = |v: &[f64]| {
            let mut w = v.to_vec();
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w[w.len() / 2]
        };
        let calm = median(&sl[60..n / 2]);
        let volatile = median(&sl[n / 2 + 60..]);
        assert!(
            volatile > calm,
            "volatile-regime stop ({volatile:.2}p) should exceed the calm stop ({calm:.2}p)"
        );
        // vol_mult scales the stop linearly.
        let (sl2, _) =
            adaptive_sl_tp_pips_series(&open, &high, &low, &close, &s, 0.0001, 2.0, 2.0).unwrap();
        assert!((sl2[300] - 2.0 * sl[300]).abs() < 1e-6, "2x vol_mult => 2x stop");
        // Bad scalars → a NAMED error, so the caller can tell "no stop here"
        // apart from "I declined to compute the stop".
        assert_eq!(
            adaptive_sl_tp_pips_series(&open, &high, &low, &close, &s, 0.0, 1.0, 2.0),
            Err(StopDistanceError::InvalidScalar {
                name: "pip_size",
                value: 0.0
            })
        );
        assert_eq!(
            adaptive_sl_tp_pips_series(&open, &high, &low, &close, &s, 0.0001, -1.0, 2.0),
            Err(StopDistanceError::InvalidScalar {
                name: "vol_mult",
                value: -1.0
            })
        );
    }

    /// Deterministic bar generator — a fixed LCG so the 301 000-bar series is
    /// reproducible without a data file and without an RNG dependency.
    ///
    /// Calibrated so the TAIL term dominates the stop, which is what real bars
    /// do: on EURUSD M5 the measured base stop is 18.09 pips with the tail term
    /// and 5.81 pips without it. A series where the Parkinson range term always
    /// won the `max()` would make the regression test below pass whether or not
    /// the tail term was computed — i.e. a test with no teeth. Narrow bar ranges
    /// (~0.2-0.4 pip) with fat-tailed close-to-close steps (up to ~4 pips)
    /// reproduce the real ordering.
    fn synthetic_bars(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut high, mut low, mut close) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let mut state: u64 = 0x2026_08_04;
        let mut px = 1.1000_f64;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // [-1, 1)
            let u = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            // Cubing keeps most steps tiny and makes a few of them large — the
            // fat tail the expected-shortfall term exists to measure.
            px = (px + 0.0004 * u * u * u).clamp(0.5, 2.0);
            let rng_amp = 0.00002 * (1.0 + u.abs());
            close.push(px);
            high.push(px + rng_amp);
            low.push(px - rng_amp);
        }
        (high, low, close)
    }

    /// Median of a slice, for the assertions below.
    fn median_of(v: &[f64]) -> f64 {
        let mut w = v.to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w[w.len() / 2]
    }

    /// The base series with an EXPLICIT cap — used to reproduce the pre-fix
    /// behaviour inside a test so the regression test can prove it has teeth.
    fn base_pips_with_cap(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        pip: f64,
        tail_max_bars: usize,
    ) -> Result<Vec<f64>, StopDistanceError> {
        let mut settings = StopTargetSettings::default();
        settings.vol_estimator = "parkinson".to_string();
        settings.tail_max_bars = tail_max_bars;
        let dist = compute_stop_distance_series(close, high, low, close, &settings)?;
        Ok(dist.iter().map(|d| (d / pip).max(1e-9)).collect())
    }

    /// THE regression test for the 300 000-bar expected-shortfall cliff.
    ///
    /// The same bars, read as a 301 000-bar series and as the trailing
    /// 299 000-bar slice of it, must produce the SAME per-bar base stop on the
    /// overlap. Before 2026-08-04 they did not: `tail_max_bars = 300_000`
    /// switched the 1.25× tail term off for the longer series and the caller
    /// substituted zero, so on real EURUSD M5 bars the median base stop went
    /// from 18.09 pips at 300 000 bars to 5.81 pips at 300 001 — a 3.11×
    /// change from one extra bar, with the run's own doc comment promising
    /// "scoring and every validation path compute the IDENTICAL per-bar stop".
    ///
    /// TOLERANCE: exact to 1e-9 relative. Both estimators are rolling, so with
    /// `tail_step = 1` the overlap is bit-identical arithmetic at ANY offset.
    /// Anything above 1e-9 means a position-dependent branch has come back.
    ///
    /// The offset here is deliberately 2 001, NOT 2 000. At the old
    /// `tail_step = 5` the expected-shortfall grid was anchored at the series
    /// start, so a 2 000-bar offset (a multiple of 5) lined up and hid the
    /// problem, while 2 001 did not — on real EURUSD M5 bars that misalignment
    /// showed up as an 86 % per-bar difference against the full-series base
    /// with the medians still agreeing to 0.006 %. A test that only used
    /// aligned offsets would have passed over it.
    #[test]
    fn base_stop_is_length_invariant_across_the_old_300k_cliff() {
        const LONG: usize = 301_000;
        const OFFSET: usize = 2_001; // deliberately NOT a multiple of the old step 5
        const SHORT: usize = LONG - OFFSET;

        let (high, low, close) = synthetic_bars(LONG);
        let pip = 0.0001_f64;

        let long_base = adaptive_base_pips_series(&high, &low, &close, pip)
            .expect("301 000-bar base series must build");
        let short_base = adaptive_base_pips_series(
            &high[OFFSET..],
            &low[OFFSET..],
            &close[OFFSET..],
            pip,
        )
        .expect("299 000-bar base series must build");

        assert_eq!(long_base.len(), LONG);
        assert_eq!(short_base.len(), SHORT);

        // Skip the short series' own warmup (vol_window 50 / tail_window 100),
        // then every remaining overlapping bar must agree.
        let warmup = 200usize;
        let mut worst_rel = 0.0_f64;
        let mut worst_at = 0usize;
        for i in warmup..SHORT {
            let a = long_base[i + OFFSET];
            let b = short_base[i];
            let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-12);
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = i;
            }
        }
        assert!(
            worst_rel < 1e-9,
            "base stop is NOT length-invariant: worst relative difference {worst_rel:.6e} at \
             overlap index {worst_at} ({} pips over {LONG} bars vs {} pips over {SHORT} bars). \
             A length-dependent branch is back in the stop-distance path.",
            long_base[worst_at + OFFSET],
            short_base[worst_at]
        );

        // And the medians — the number the operator actually reads — agree too.
        let m_long = median_of(&long_base[OFFSET + warmup..]);
        let m_short = median_of(&short_base[warmup..]);
        assert!(
            (m_long - m_short).abs() / m_long.max(m_short) < 1e-9,
            "median base stop diverges across the old cliff: {m_long} vs {m_short}"
        );

        // ── This test has teeth ────────────────────────────────────────────
        // Everything above would pass on a series where the tail term never
        // wins the `max()`, whether or not the tail term was computed. Pin the
        // pre-fix behaviour explicitly: with the old hardcoded cap of 300 000,
        // the SAME two calls must diverge, and by a lot. If this half ever
        // stops failing-under-the-old-cap, the half above has become vacuous.
        let capped_long = base_pips_with_cap(&high, &low, &close, pip, 300_000);
        assert_eq!(
            capped_long,
            Err(StopDistanceError::TailCapExceeded {
                bars: LONG,
                cap: 300_000
            }),
            "the old cap must now REFUSE the long series rather than silently \
             dropping the tail term"
        );
        let capped_short =
            base_pips_with_cap(&high[OFFSET..], &low[OFFSET..], &close[OFFSET..], pip, 300_000)
                .expect("299 000 bars sit under the old cap");
        let m_capped_short = median_of(&capped_short[warmup..]);
        assert!(
            (m_capped_short - m_short).abs() / m_short < 1e-9,
            "under the cap the short series must be unchanged: {m_capped_short} vs {m_short}"
        );
        // The tail term is worth this much: dropping it (what the old code did
        // above the cap) shrinks the stop by more than 2x on this series, the
        // same direction and order of magnitude as the 18.09 -> 5.81 pips
        // measured on EURUSD M5.
        let mut no_tail = StopTargetSettings::default();
        no_tail.vol_estimator = "parkinson".to_string();
        no_tail.stop_k_tail = 0.0;
        let vol_only = compute_stop_distance_series(&close, &high, &low, &close, &no_tail)
            .expect("vol-only series builds");
        let m_vol_only = median_of(
            &vol_only[OFFSET + warmup..]
                .iter()
                .map(|d| d / pip)
                .collect::<Vec<_>>(),
        );
        assert!(
            m_long > m_vol_only * 2.0,
            "the tail term must dominate for this regression test to mean anything: \
             with tail {m_long:.4} pips, vol-only {m_vol_only:.4} pips"
        );

        // ── And the second half of the same defect ─────────────────────────
        // At `tail_step = 5` the expected-shortfall grid is anchored at the
        // slice start, so this misaligned offset makes the SAME bars disagree.
        // Pin that, so the `tail_step = 1` default is understood as load-bearing
        // and not mistaken for a speed setting somebody can turn back up.
        let stepped = |h: &[f64], l: &[f64], c: &[f64]| -> Vec<f64> {
            let mut s = StopTargetSettings::default();
            s.vol_estimator = "parkinson".to_string();
            s.tail_step = 5;
            compute_stop_distance_series(c, h, l, c, &s)
                .expect("stepped series builds")
                .iter()
                .map(|d| d / pip)
                .collect()
        };
        let stepped_long = stepped(&high, &low, &close);
        let stepped_short = stepped(&high[OFFSET..], &low[OFFSET..], &close[OFFSET..]);
        let stepped_worst = (warmup..SHORT)
            .map(|i| {
                let (a, b) = (stepped_long[i + OFFSET], stepped_short[i]);
                (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
            })
            .fold(0.0_f64, f64::max);
        assert!(
            stepped_worst > 1e-6,
            "expected `tail_step = 5` to reintroduce a position-dependent tail at a \
             misaligned offset (worst relative difference {stepped_worst:.3e}). If this \
             stopped happening, the grid was made position-invariant and the \
             `tail_step = 1` default can be revisited"
        );
    }

    /// A cap that IS configured must refuse by name, never by returning a
    /// stop built from a zero tail term.
    #[test]
    fn a_configured_tail_cap_fails_loudly_instead_of_zeroing_the_tail() {
        let (high, low, close) = synthetic_bars(2_000);
        let mut settings = StopTargetSettings::default();
        settings.vol_estimator = "parkinson".to_string();

        // Default: no cap, series builds.
        assert_eq!(settings.tail_max_bars, 0, "the default must be NO cap");
        let uncapped = compute_stop_distance_series(&close, &high, &low, &close, &settings)
            .expect("uncapped series builds");

        // Cap below the series length: a named error carrying both numbers.
        settings.tail_max_bars = 1_000;
        assert_eq!(
            compute_stop_distance_series(&close, &high, &low, &close, &settings),
            Err(StopDistanceError::TailCapExceeded {
                bars: 2_000,
                cap: 1_000
            })
        );

        // Cap at or above the length: identical to uncapped, bit for bit.
        settings.tail_max_bars = 2_000;
        let capped_at_len = compute_stop_distance_series(&close, &high, &low, &close, &settings)
            .expect("a cap at the series length must not bite");
        assert_eq!(capped_at_len, uncapped);
    }

    /// The tail term is not decoration: proving it moves the stop is what makes
    /// the old silent `tail_dist = 0` substitution a behaviour change rather
    /// than a rounding detail.
    #[test]
    fn dropping_the_tail_term_materially_changes_the_stop() {
        let (high, low, close) = synthetic_bars(4_000);
        let mut settings = StopTargetSettings::default();
        settings.vol_estimator = "parkinson".to_string();
        let with_tail = compute_stop_distance_series(&close, &high, &low, &close, &settings)
            .expect("series builds");

        // `stop_k_tail = 0` is the honest way to express "no tail term" — the
        // same arithmetic the old cap silently performed.
        settings.stop_k_tail = 0.0;
        let without_tail = compute_stop_distance_series(&close, &high, &low, &close, &settings)
            .expect("series builds");

        let m_with = median_of(&with_tail[200..]);
        let m_without = median_of(&without_tail[200..]);
        assert!(
            m_with > m_without * 1.5,
            "the tail term must dominate the stop for it to be worth guarding: \
             with tail {m_with}, without {m_without}"
        );
    }

    #[test]
    fn stop_target_from_settings_default_matches_default() {
        let s = neoethos_core::Settings::default();
        assert_eq!(
            StopTargetRuntimeOverrides::from_settings(&s),
            StopTargetRuntimeOverrides::default()
        );
        assert_eq!(
            StopTargetRuntimeOverrides::default().tail_max_bars,
            StopTargetSettings::default().tail_max_bars,
            "the config mirror and the settings struct must agree on the cap"
        );
        assert_eq!(
            StopTargetRuntimeOverrides::default().tail_step,
            StopTargetSettings::default().tail_step,
            "the config mirror and the settings struct must agree on the stride"
        );
        // A `0` stride is meaningless; it must not reach the estimator.
        let mut zeroed = neoethos_core::Settings::default();
        zeroed.models.stop_target_runtime.tail_step = 0;
        assert_eq!(StopTargetRuntimeOverrides::from_settings(&zeroed).tail_step, 1);
    }
}
