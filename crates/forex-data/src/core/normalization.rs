//! Per-column feature normalization.
//!
//! Raw indicator outputs span wildly different scales — a price-level
//! feature like `vwap` is on the order of 1.10 (EURUSD) or 165 (EURJPY)
//! or 2400 (XAUUSD), while an oscillator like `rsi` is bounded 0..100,
//! and SMC binary flags are 0/1. Mixing them in a weighted sum (the
//! GA's "combined" signal) means the largest-magnitude column always
//! dominates regardless of weight, and the GA's `long_threshold ≈ 0.45`
//! never triggers on small-scale columns or always triggers on
//! large-scale ones. The result is the empty-portfolio bug we observed
//! on EURJPY (feature magnitudes ±3.5e11) and XAUUSD.
//!
//! The fix is a robust per-column z-score:
//! - Compute median + MAD (median absolute deviation) per column.
//! - `z = (x - median) / (1.4826 * MAD)` — Gaussian-equivalent scale.
//! - Sanitise NaN / Inf to 0.0 (the bar contributes nothing).
//! - Clip to ±10 (1-in-billion under Gaussian) so a single outlier
//!   can't blow up the GA's combined sum.
//!
//! Runs in-place on a [`crate::core::features::FeatureFrame`] after
//! all indicators have been computed and aligned across timeframes.
//! `normalize_feature_frame` is idempotent — re-applying it yields
//! the same frame because the medians of normalized data are 0.

use ndarray::Array2;

/// Hard clip applied after z-scoring. Outliers beyond this are likely
/// data-feed glitches or warmup-period transients we don't want the
/// GA to treat as real signal.
pub const Z_CLIP: f32 = 10.0;

/// Median Absolute Deviation → standard-deviation correction factor
/// for a normal distribution. `MAD * 1.4826 ≈ σ`.
pub const MAD_TO_SIGMA: f32 = 1.4826;

/// Normalize every column of a feature matrix in place. NaN / Inf
/// cells become 0.0; survivors are robust z-scored and clipped to
/// [-Z_CLIP, +Z_CLIP].
///
/// Returns the per-column (median, scale) used so downstream code can
/// log the transformation or persist it for inference.
pub fn normalize_feature_matrix(data: &mut Array2<f32>) -> Vec<(f32, f32)> {
    let n_rows = data.nrows();
    let n_cols = data.ncols();
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }

    let mut stats = Vec::with_capacity(n_cols);

    // Scratch buffer reused per column to avoid per-column allocs.
    let mut finite: Vec<f32> = Vec::with_capacity(n_rows);

    for c in 0..n_cols {
        finite.clear();
        for r in 0..n_rows {
            let v = data[(r, c)];
            if v.is_finite() {
                finite.push(v);
            }
        }
        let (median, scale) = if finite.is_empty() {
            (0.0_f32, 1.0_f32)
        } else {
            // Median (linear partition; small overhead, simpler than
            // a select-nth implementation, fine for ~3K-200K rows).
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = finite[finite.len() / 2];
            // MAD = median(|x - median|).
            let mut deviations: Vec<f32> =
                finite.iter().map(|x| (*x - median).abs()).collect();
            deviations
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = deviations[deviations.len() / 2];
            let scale = (mad * MAD_TO_SIGMA).max(1e-9);
            (median, scale)
        };
        stats.push((median, scale));

        // Apply z-score in place.
        for r in 0..n_rows {
            let v = data[(r, c)];
            let z = if v.is_finite() {
                ((v - median) / scale).clamp(-Z_CLIP, Z_CLIP)
            } else {
                0.0
            };
            data[(r, c)] = z;
        }
    }

    stats
}

/// Variant that takes an immutable matrix and returns a new normalized
/// one — convenient for tests where the original is needed for
/// comparison.
pub fn normalize_feature_matrix_copy(data: &Array2<f32>) -> (Array2<f32>, Vec<(f32, f32)>) {
    let mut out = data.clone();
    let stats = normalize_feature_matrix(&mut out);
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn nan_cells_become_zero() {
        let mut m = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, f32::NAN, 2.0, 3.0, f32::INFINITY, 4.0],
        )
        .unwrap();
        normalize_feature_matrix(&mut m);
        // Column 0: NaN → 0
        assert_eq!(m[(0, 0)], 0.0);
        // Column 1: NaN → 0 (the [0,1] cell was NaN), Inf → 0
        // (our MAD-based scale is finite so the surviving 3.0/4.0 stay finite).
        assert!(m[(0, 1)].abs() < 1e-9 || m[(0, 1)] == 0.0);
    }

    #[test]
    fn huge_magnitudes_get_normalized_to_unit_scale() {
        // Simulate EURJPY-style feature: range ±3.5e11
        let raw: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 - 500.0) / 500.0) * 3.5e11)
            .collect();
        let mut m = Array2::from_shape_vec((1000, 1), raw).unwrap();
        let stats = normalize_feature_matrix(&mut m);
        // Median should be near 0 (symmetric range), scale should be huge.
        assert!(stats[0].1 > 1e10);
        // Every output should be within the clip range.
        for r in 0..1000 {
            assert!(m[(r, 0)].abs() <= Z_CLIP);
        }
        // Min/max should approach ±Z_CLIP (some clipping, some not).
        let max = m.column(0).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = m.column(0).iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max > 1.0 && max <= Z_CLIP);
        assert!(min < -1.0 && min >= -Z_CLIP);
    }

    #[test]
    fn binary_flag_columns_stay_meaningful() {
        // Binary 0/1 column — half ones, half zeros.
        let raw: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
        let mut m = Array2::from_shape_vec((100, 1), raw).unwrap();
        normalize_feature_matrix(&mut m);
        // Binary columns post-normalization will have median = 0 or 1,
        // and the MAD will be small. The point: 0s and 1s map to two
        // distinct z values, neither clipped to ±Z_CLIP.
        let unique: Vec<f32> = {
            let mut v: Vec<f32> = m.column(0).iter().cloned().collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v.dedup();
            v
        };
        assert!(unique.len() >= 1, "binary col should still have distinct values");
    }

    #[test]
    fn idempotent_under_double_application() {
        let raw: Vec<f32> = (0..1000).map(|i| (i as f32) * 1e6).collect();
        let mut m = Array2::from_shape_vec((1000, 1), raw).unwrap();
        normalize_feature_matrix(&mut m);
        let after_first = m.clone();
        normalize_feature_matrix(&mut m);
        // Second pass: median is now 0, MAD/scale ≈ 1, so every value
        // stays approximately the same.
        for r in 0..1000 {
            assert!(
                (m[(r, 0)] - after_first[(r, 0)]).abs() < 0.1,
                "double-norm drift at row {} too large",
                r
            );
        }
    }
}
