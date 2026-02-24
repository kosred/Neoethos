//! SIMD-optimized CPU validation for HPC mode (AVX2/FMA).
//! 
//! This module provides AVX2-optimized backtesting for CPU validation
//! of strategies that pass initial GPU screening.

use std::arch::x86_64::*;

/// Check if AVX2 is available at runtime
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// SIMD-optimized backtest using AVX2/FMA
/// 
/// Computes equity curve from signals and returns.
/// Returns final profit factor approximation.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn backtest_simd(signals: &[i8], returns: &[f64]) -> f64 {
    if signals.len() != returns.len() || signals.is_empty() {
        return 0.0;
    }

    let n = signals.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Initialize accumulators
    let mut sum_profit = _mm256_setzero_pd();
    let mut sum_loss = _mm256_setzero_pd();

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;

        // Load signals (convert i8 to f64)
        let sig = _mm256_set_pd(
            signals[offset + 3] as f64,
            signals[offset + 2] as f64,
            signals[offset + 1] as f64,
            signals[offset] as f64,
        );

        // Load returns
        let ret = _mm256_loadu_pd(returns.as_ptr().add(offset));

        // Compute trade P&L: signal * return
        let pnl = _mm256_mul_pd(sig, ret);

        // Separate profits and losses
        let zero = _mm256_setzero_pd();
        let is_profit = _mm256_cmp_pd(pnl, zero, _CMP_GT_OQ);

        let profit = _mm256_and_pd(pnl, is_profit);
        let loss = _mm256_and_pd(_mm256_sub_pd(zero, pnl), _mm256_cmp_pd(pnl, zero, _CMP_LT_OQ));

        sum_profit = _mm256_add_pd(sum_profit, profit);
        sum_loss = _mm256_add_pd(sum_loss, loss);
    }

    // Horizontal sum of accumulators
    let hsum_profit = hsum256_pd(sum_profit);
    let hsum_loss = hsum256_pd(sum_loss);

    // Process remainder
    let mut rem_profit = 0.0;
    let mut rem_loss = 0.0;

    for i in (n - remainder)..n {
        let pnl = signals[i] as f64 * returns[i];
        if pnl > 0.0 {
            rem_profit += pnl;
        } else {
            rem_loss -= pnl;
        }
    }

    let total_profit = hsum_profit + rem_profit;
    let total_loss = hsum_loss + rem_loss;

    // Return profit factor
    if total_loss > 1e-12 {
        total_profit / total_loss
    } else {
        total_profit
    }
}

/// Horizontal sum of 4 doubles in AVX2 register
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_pd(v: __m256d) -> f64 {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(lo, hi);
    let unpacked = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, unpacked);
    _mm_cvtsd_f64(sum64)
}

/// SIMD-optimized signal computation
/// 
/// Computes weighted sum of features for multiple genes simultaneously
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn compute_signals_simd(
    features: &[f32],
    weights: &[f32],
    threshold: f32,
) -> i8 {
    let n = features.len().min(weights.len());
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    // Process 8 floats at a time
    for i in 0..chunks {
        let offset = i * 8;
        let f = _mm256_loadu_ps(features.as_ptr().add(offset));
        let w = _mm256_loadu_ps(weights.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(f, w, sum);
    }

    // Horizontal sum
    let mut total = hsum256_ps(sum);

    // Process remainder
    for i in (n - remainder)..n {
        total += features[i] * weights[i];
    }

    // Apply threshold
    if total >= threshold {
        1
    } else if total <= -threshold {
        -1
    } else {
        0
    }
}

/// Horizontal sum of 8 floats in AVX register
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    // Sum pairs: [a,b,c,d,e,f,g,h] -> [a+b, c+d, e+f, g+h, ...]
    let sum1 = _mm256_hadd_ps(v, v);
    // Sum again: [a+b+c+d, e+f+g+h, ...]
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    
    // Extract low and high 128-bit halves
    let low = _mm256_castps256_ps128(sum2);
    let high = _mm256_extractf128_ps(sum2, 1);
    
    // Add halves
    let sum128 = _mm_add_ss(low, high);
    _mm_cvtss_f32(sum128)
}

/// Batch evaluation of multiple strategies using SIMD
/// 
/// This is much faster for CPU validation of candidates
pub fn batch_evaluate_simd(
    features: &[Vec<f32>],
    genes_indices: &[Vec<usize>],
    genes_weights: &[Vec<f32>],
    long_thresholds: &[f32],
    short_thresholds: &[f32],
) -> Vec<i8> {
    if !has_avx2() {
        // Fallback to scalar implementation
        return batch_evaluate_scalar(
            features,
            genes_indices,
            genes_weights,
            long_thresholds,
            short_thresholds,
        );
    }

    let n_samples = features.len();
    let n_genes = genes_indices.len();
    let mut signals = vec![0_i8; n_samples * n_genes];

    // Process each gene
    for (gene_idx, indices) in genes_indices.iter().enumerate() {
        let weights = &genes_weights[gene_idx];
        let long_th = long_thresholds[gene_idx];
        let short_th = short_thresholds[gene_idx];

        // Pre-extract features for this gene
        let gene_features: Vec<Vec<f32>> = features
            .iter()
            .map(|sample| {
                indices.iter().map(|&i| sample[i]).collect()
            })
            .collect();

        // Compute signals
        for (sample_idx, sample_features) in gene_features.iter().enumerate() {
            let signal = unsafe {
                compute_signals_simd(sample_features, weights, long_th)
            };
            signals[sample_idx * n_genes + gene_idx] = signal;
        }
    }

    signals
}

/// Scalar fallback for batch evaluation
fn batch_evaluate_scalar(
    features: &[Vec<f32>],
    genes_indices: &[Vec<usize>],
    genes_weights: &[Vec<f32>],
    long_thresholds: &[f32],
    short_thresholds: &[f32],
) -> Vec<i8> {
    let n_samples = features.len();
    let n_genes = genes_indices.len();
    let mut signals = vec![0_i8; n_samples * n_genes];

    for (gene_idx, indices) in genes_indices.iter().enumerate() {
        let weights = &genes_weights[gene_idx];
        let long_th = long_thresholds[gene_idx];
        let short_th = short_thresholds[gene_idx];

        for (sample_idx, sample) in features.iter().enumerate() {
            let mut sum = 0.0_f32;
            for (i, &idx) in indices.iter().enumerate() {
                sum += sample[idx] * weights[i];
            }

            let signal = if sum >= long_th {
                1
            } else if sum <= short_th {
                -1
            } else {
                0
            };

            signals[sample_idx * n_genes + gene_idx] = signal;
        }
    }

    signals
}

/// Fast Sharpe ratio computation using SIMD
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub unsafe fn sharpe_ratio_simd(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Accumulators
    let mut sum = _mm256_setzero_pd();
    let mut sumsq = _mm256_setzero_pd();

    // Process chunks
    for i in 0..chunks {
        let offset = i * 4;
        let r = _mm256_loadu_pd(returns.as_ptr().add(offset));
        sum = _mm256_add_pd(sum, r);
        sumsq = _mm256_fmadd_pd(r, r, sumsq);
    }

    // Horizontal sum
    let hsum = hsum256_pd(sum);
    let hsumsq = hsum256_pd(sumsq);

    // Process remainder
    let mut rem_sum = 0.0;
    let mut rem_sumsq = 0.0;
    for i in (n - remainder)..n {
        rem_sum += returns[i];
        rem_sumsq += returns[i] * returns[i];
    }

    let total_sum = hsum + rem_sum;
    let total_sumsq = hsumsq + rem_sumsq;

    let mean = total_sum / n as f64;
    let variance = (total_sumsq / n as f64) - mean * mean;
    let std_dev = variance.max(0.0).sqrt();

    if std_dev > 1e-12 {
        mean / std_dev
    } else {
        0.0
    }
}

/// Public wrapper for Sharpe ratio computation
pub fn compute_sharpe_ratio(returns: &[f64]) -> f64 {
    if has_avx2() {
        unsafe { sharpe_ratio_simd(returns) }
    } else {
        compute_sharpe_scalar(returns)
    }
}

fn compute_sharpe_scalar(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let sum: f64 = returns.iter().sum();
    let sumsq: f64 = returns.iter().map(|r| r * r).sum();

    let mean = sum / n;
    let variance = (sumsq / n) - mean * mean;
    let std_dev = variance.max(0.0).sqrt();

    if std_dev > 1e-12 {
        mean / std_dev
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012];
        
        let simd_result = compute_sharpe_ratio(&returns);
        let scalar_result = compute_sharpe_scalar(&returns);
        
        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_backtest_simd() {
        let signals = vec![1, -1, 1, 1, -1, 1, -1, -1];
        let returns = vec![0.01, -0.02, 0.015, 0.008, -0.01, 0.012, -0.005, 0.003];

        if has_avx2() {
            let simd_result = unsafe { backtest_simd(&signals, &returns) };
            assert!(simd_result > 0.0);
        }
    }
}
