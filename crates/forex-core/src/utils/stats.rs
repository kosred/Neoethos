//! Shared statistical helpers for the search / models / app layers.
//!
//! Phase 64 extraction: previously `mean`, `stddev`, `stddev_sample`,
//! and `mean_std` lived in five different files inside
//! `forex-search` (portfolio.rs, quality.rs, stop_target.rs, eval.rs,
//! cubecl_eval.rs). They now live here so every caller agrees on the
//! population vs Bessel-corrected variant and on the NaN-tolerance
//! behavior.

/// Arithmetic mean of `values`. Returns `0.0` for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / (values.len() as f64)
}

/// Population standard deviation (divisor `n`). Returns `0.0` when
/// `values` has fewer than two elements.
pub fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64);
    var.max(0.0).sqrt()
}

/// Sample standard deviation (divisor `n - 1`, Bessel correction).
/// Returns `0.0` when `values` has fewer than two elements.
pub fn stddev_sample(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / ((values.len() - 1) as f64);
    var.max(0.0).sqrt()
}

/// Compute mean + (sample) standard deviation in a single pass.
/// NaN / infinite samples are dropped before the calculation so a
/// single corrupt bar cannot poison a population's metric.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return (0.0, 0.0);
    }
    let m = mean(&finite);
    let s = stddev_sample(&finite, m);
    (m, s)
}

/// Pearson correlation between two equal-length f32 slices. Returns
/// `0.0` when either input has constant variance or the lengths
/// differ.
pub fn pearson_correlation_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 || n != y.len() {
        return 0.0;
    }
    let n_f = n as f32;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sum_xy = 0.0_f32;
    let mut sum_x2 = 0.0_f32;
    let mut sum_y2 = 0.0_f32;
    for i in 0..n {
        let a = x[i];
        let b = y[i];
        sum_x += a;
        sum_y += b;
        sum_xy += a * b;
        sum_x2 += a * a;
        sum_y2 += b * b;
    }
    let num = n_f * sum_xy - sum_x * sum_y;
    let den = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
    if den == 0.0 || !den.is_finite() {
        0.0
    } else {
        num / den
    }
}

/// Element-wise mean of a list of equal-length `Vec<f32>` (e.g. an
/// elite population's genomes for CRFMNES / NEAT). Returns an empty
/// vector when the input is empty.
pub fn mean_vector_f32(elites: &[Vec<f32>]) -> Vec<f32> {
    if elites.is_empty() {
        return Vec::new();
    }
    let dim = elites[0].len();
    let mut sum = vec![0.0_f32; dim];
    let mut count = 0_f32;
    for e in elites {
        if e.len() != dim {
            continue;
        }
        for (i, v) in e.iter().enumerate() {
            sum[i] += *v;
        }
        count += 1.0;
    }
    if count == 0.0 {
        return vec![0.0; dim];
    }
    sum.into_iter().map(|s| s / count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_handles_empty_and_simple() {
        assert_eq!(mean(&[]), 0.0);
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn population_vs_sample_stddev_match_known_values() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&values);
        // population variance = ((1-3)^2 + ... ) / 5 = 2.0 → σ ≈ 1.4142
        assert!((stddev(&values, m) - 1.4142135623730951).abs() < 1e-12);
        // sample variance = 10 / 4 = 2.5 → σ ≈ 1.5811
        assert!((stddev_sample(&values, m) - 1.5811388300841898).abs() < 1e-12);
    }

    #[test]
    fn mean_std_drops_non_finite_samples() {
        let values = [1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0, f64::INFINITY];
        let (m, s) = mean_std(&values);
        assert!((m - 3.0).abs() < 1e-12);
        assert!((s - 1.5811388300841898).abs() < 1e-12);
    }

    #[test]
    fn pearson_correlation_handles_perfect_positive_negative_and_constant() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let y_pos = [10.0_f32, 20.0, 30.0, 40.0];
        let y_neg = [40.0_f32, 30.0, 20.0, 10.0];
        let y_const = [5.0_f32; 4];
        assert!((pearson_correlation_f32(&x, &y_pos) - 1.0).abs() < 1e-6);
        assert!((pearson_correlation_f32(&x, &y_neg) + 1.0).abs() < 1e-6);
        assert_eq!(pearson_correlation_f32(&x, &y_const), 0.0);
    }

    #[test]
    fn mean_vector_f32_averages_equal_length_inputs() {
        let elites = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        let m = mean_vector_f32(&elites);
        assert_eq!(m, vec![2.0, 3.0, 4.0]);

        // mismatched lengths are skipped silently.
        let mixed = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let m = mean_vector_f32(&mixed);
        assert_eq!(m, vec![1.0, 2.0]);
    }
}
