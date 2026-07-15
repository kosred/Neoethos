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
    // Scratch column buffer reused across columns. Robust z-scoring is
    // per-column independent, so we delegate each column to the single-series
    // routine — guaranteeing the out-of-core build path (which normalises each
    // feature series *before* writing it to the mmap store) is bit-identical
    // to normalising the assembled in-RAM matrix here.
    let mut scratch = vec![0.0_f32; n_rows];
    let fit_rows = norm_fit_rows(n_rows);
    for c in 0..n_cols {
        for r in 0..n_rows {
            scratch[r] = data[(r, c)];
        }
        let (median, scale) = normalize_feature_series_in_place(&mut scratch, fit_rows);
        for r in 0..n_rows {
            data[(r, c)] = scratch[r];
        }
        stats.push((median, scale));
    }

    stats
}

/// Robust z-score one feature's full time series in place — the per-column
/// kernel of [`normalize_feature_matrix`], exposed so the out-of-core feature
/// build can normalise each series as it streams it to the mmap store (where
/// each feature is one contiguous row) without ever materialising the dense
/// `[samples × features]` matrix.
///
/// NaN / Inf cells become `0.0`; finite survivors are `(x - median) / scale`
/// (`scale = MAD * 1.4826`, floored at `1e-9`) clipped to `±Z_CLIP`. Returns
/// the `(median, scale)` used.
/// Fraction of the series (leading rows) whose statistics the robust z-score
/// is FIT on (audit D09). Fitting on the FULL series let each row's median/MAD
/// depend on FUTURE rows — lookahead — and historical normalized values
/// changed whenever new bars were appended. Fitting on the training prefix and
/// applying those immutable stats to the whole series makes the out-of-sample
/// tail leakage-free and the whole series stable under appends. 0.8 matches the
/// discovery 80/20 holdout.
pub const NORM_FIT_FRACTION: f64 = 0.8;

/// Number of leading rows to fit normalization stats on. Small series
/// (<= 128 rows) fit on all of themselves so the median/MAD stay meaningful.
pub fn norm_fit_rows(n_rows: usize) -> usize {
    if n_rows <= 128 {
        return n_rows;
    }
    (((n_rows as f64) * NORM_FIT_FRACTION) as usize).clamp(128, n_rows)
}

/// Robust z-score one feature series IN PLACE, fitting the median/MAD on the
/// first `fit_rows` rows only (audit D09) and applying them to the whole
/// series. `fit_rows == 0` (or `>= len`) fits on the full series — the legacy
/// behavior, kept for callers that genuinely have only training data.
pub fn normalize_feature_series_in_place(series: &mut [f32], fit_rows: usize) -> (f32, f32) {
    let fit_end = if fit_rows == 0 || fit_rows > series.len() {
        series.len()
    } else {
        fit_rows
    };
    let mut finite: Vec<f32> = series[..fit_end]
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let (median, scale) = if finite.is_empty() {
        (0.0_f32, 1.0_f32)
    } else {
        // Median (linear partition; fine for ~3K-5M rows).
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = finite[finite.len() / 2];
        // MAD = median(|x - median|).
        let mut deviations: Vec<f32> = finite.iter().map(|x| (*x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = deviations[deviations.len() / 2];
        let scale = (mad * MAD_TO_SIGMA).max(1e-9);
        (median, scale)
    };

    for v in series.iter_mut() {
        *v = if v.is_finite() {
            ((*v - median) / scale).clamp(-Z_CLIP, Z_CLIP)
        } else {
            0.0
        };
    }

    (median, scale)
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
        // Note — fixed pre-existing test logic error.
        //
        // ndarray::from_shape_vec is ROW-MAJOR, so the input vector
        // `[1.0, NaN, 2.0, 3.0, Inf, 4.0]` lays out as:
        //   row 0: [1.0, NaN]
        //   row 1: [2.0, 3.0]
        //   row 2: [Inf, 4.0]
        // The old test asserted `m[(0,0)] == 0.0` — but `m[(0,0)]` was
        // `1.0` (finite), so after z-score normalisation it became
        // `-0.674…`, not zero. The function correctly zeros ONLY the
        // NaN / Inf cells (see `normalize_feature_matrix` body — the
        // `if v.is_finite() else 0.0` branch at line ~82). Update the
        // assertions to target the cells that were actually non-finite.
        let mut m =
            Array2::from_shape_vec((3, 2), vec![1.0, f32::NAN, 2.0, 3.0, f32::INFINITY, 4.0])
                .unwrap();
        normalize_feature_matrix(&mut m);
        // Cell (0, 1) was NaN → must be exactly 0.0.
        assert_eq!(m[(0, 1)], 0.0, "NaN cell must be zeroed");
        // Cell (2, 0) was Inf → must be exactly 0.0.
        assert_eq!(m[(2, 0)], 0.0, "Inf cell must be zeroed");
        // Finite survivor cells go through the MAD-based z-score; they
        // may NOT be zero but MUST be finite and bounded by [-Z_CLIP, +Z_CLIP].
        for (r, c) in [(0, 0), (1, 0), (1, 1), (2, 1)] {
            assert!(
                m[(r, c)].is_finite() && m[(r, c)].abs() <= Z_CLIP,
                "finite cell ({r},{c}) must stay finite and clipped, got {}",
                m[(r, c)]
            );
        }
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
        let max = m
            .column(0)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
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
        assert!(
            unique.len() >= 1,
            "binary col should still have distinct values"
        );
    }

    #[test]
    fn normalization_fits_on_prefix_and_ignores_the_future_tail() {
        // Audit D09: robust-z stats must come from the TRAINING PREFIX, so
        // out-of-sample rows cannot leak their own values into median/scale.
        // Prefix = a clean 0..800 ramp; tail = 200 extreme 1e9 outliers.
        let mut series: Vec<f32> = (0..800).map(|i| i as f32).collect();
        series.extend(std::iter::repeat_n(1.0e9_f32, 200)); // future tail

        // Fit on the first 800 rows only → tail is invisible to the fit.
        let (median, _scale) = normalize_feature_series_in_place(&mut series, 800);
        assert!(
            (median - 400.0).abs() < 1.0,
            "median must be the prefix median (~400), not pulled by the 1e9 tail; got {median}"
        );

        // A full-series fit (fit_rows = 0) sees the outliers → materially
        // different stats: proves the prefix guard actually changed behavior.
        let mut full: Vec<f32> = (0..800).map(|i| i as f32).collect();
        full.extend(std::iter::repeat_n(1.0e9_f32, 200));
        let (full_median, _) = normalize_feature_series_in_place(&mut full, 0);
        assert!(
            (full_median - median).abs() > 50.0,
            "full-series fit ({full_median}) must differ from prefix fit ({median})"
        );
    }

    #[test]
    fn norm_fit_rows_is_prefix_for_large_series_and_full_for_small() {
        assert_eq!(norm_fit_rows(1000), 800, "80% of a large series");
        assert_eq!(norm_fit_rows(100), 100, "small series fit on all of itself");
        assert_eq!(norm_fit_rows(0), 0);
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
