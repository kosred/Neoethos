//! Time-series statistical helpers.
//!
//! Phase 69 extraction: previously `median_ignore_nan`,
//! `percentile_sorted`, `rolling_mean_f64`, `moving_average_f32`, and
//! `ewma_f32` were spread across forex-search/stop_target,
//! forex-models/forecasting/swarm_impl, and forex-models/anomaly/forest_impl.
//! They now live here so search-side and models-side callers share one
//! tested implementation.

/// Median of `values`, ignoring NaN / ±∞ samples. Returns `f64::NAN`
/// when no finite samples remain.
pub fn median_ignore_nan(values: &[f64]) -> f64 {
    let mut vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        (vals[mid - 1] + vals[mid]) / 2.0
    }
}

/// Median of `values: &[f32]`, requires the input to be pre-sorted.
/// Returns `0.0` for an empty slice. Use this when the caller already
/// has a sorted vector and wants to avoid the copy in
/// [`median_ignore_nan`].
pub fn median_sorted_f32(sorted: &[f32]) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// Linear-interpolated quantile of a pre-sorted slice. `quantile`
/// outside `[0, 1]` is clamped. Returns `0.0` for an empty slice.
pub fn percentile_sorted_f32(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = quantile.clamp(0.0, 1.0);
    let pos = q * ((sorted.len() - 1) as f32);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Trailing arithmetic mean of `values` over `window`. Position `i` of
/// the output is the mean of `values[i.saturating_sub(window-1)..=i]`.
/// Returns the same length as the input.
pub fn rolling_mean_f64(values: &[f64], window: usize) -> Vec<f64> {
    if values.is_empty() || window == 0 {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0_f64;
    for (i, v) in values.iter().enumerate() {
        sum += v;
        if i >= window {
            sum -= values[i - window];
        }
        let denom = (i + 1).min(window) as f64;
        out.push(sum / denom);
    }
    out
}

/// Simple `f32` moving average over a fixed `window`. Same shape as
/// [`rolling_mean_f64`] but takes / returns `f32`.
pub fn moving_average_f32(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() || window == 0 {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0_f32;
    for (i, v) in values.iter().enumerate() {
        sum += v;
        if i >= window {
            sum -= values[i - window];
        }
        let denom = (i + 1).min(window) as f32;
        out.push(sum / denom);
    }
    out
}

/// Exponentially-weighted moving average. `alpha` is the smoothing
/// factor in `(0, 1]`; values outside that range are clamped. The
/// first sample is propagated as-is (no warm-up).
pub fn ewma_f32(values: &[f32], alpha: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let a = alpha.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(values.len());
    out.push(values[0]);
    for v in &values[1..] {
        let prev = *out.last().unwrap();
        out.push(a * v + (1.0 - a) * prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ignore_nan_drops_non_finite_and_handles_even() {
        let values = [3.0, 1.0, 4.0, f64::NAN, 1.0, 5.0, 9.0, f64::INFINITY];
        // Finite sorted: [1, 1, 3, 4, 5, 9]. Even length → average of 3 and 4.
        assert!((median_ignore_nan(&values) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn median_ignore_nan_returns_nan_when_no_finite_samples() {
        let v = [f64::NAN, f64::INFINITY];
        assert!(median_ignore_nan(&v).is_nan());
    }

    #[test]
    fn percentile_sorted_f32_interpolates_linearly() {
        let sorted = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_sorted_f32(&sorted, 0.0) - 1.0).abs() < 1e-6);
        assert!((percentile_sorted_f32(&sorted, 1.0) - 5.0).abs() < 1e-6);
        // 50th percentile of 5 elements (positions 0..=4) → position 2.0 → 3.0
        assert!((percentile_sorted_f32(&sorted, 0.5) - 3.0).abs() < 1e-6);
        // 25th percentile → position 1.0 → 2.0
        assert!((percentile_sorted_f32(&sorted, 0.25) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rolling_mean_f64_warmup_then_window() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = rolling_mean_f64(&v, 3);
        // Pos 0: mean([1])     = 1
        // Pos 1: mean([1,2])   = 1.5
        // Pos 2: mean([1,2,3]) = 2
        // Pos 3: mean([2,3,4]) = 3
        // Pos 4: mean([3,4,5]) = 4
        assert_eq!(out, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ewma_f32_smooths_step_input() {
        let v = [1.0_f32, 1.0, 1.0, 5.0, 5.0, 5.0];
        let out = ewma_f32(&v, 0.5);
        // Step from 1 to 5 should smooth gradually.
        assert!(out[3] > 1.0 && out[3] < 5.0);
        assert!(out[4] > out[3]);
        assert!(out[5] > out[4]);
    }
}
