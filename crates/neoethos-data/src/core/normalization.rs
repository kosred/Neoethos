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
//! - Preserve invalid cells as typed validity plus canonical NaN; an undefined
//!   indicator value is never silently converted into a valid numeric zero.
//! - Clip to ±10 (1-in-billion under Gaussian) so a single outlier
//!   can't blow up the GA's combined sum.
//!
//! The caller supplies the exact training-row range. This module never infers
//! a train/test split and therefore cannot fit on future rows accidentally.

use std::ops::Range;

use anyhow::{Result, bail};

use crate::core::features::{FeatureCellValidity, FeatureColumnF64};

pub const Z_CLIP_F64: f64 = 10.0;
pub const MAD_TO_SIGMA_F64: f64 = 1.4826;

/// Immutable fitted state for the explicit-validity f64 normalization lane.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustNormalizationFitF64 {
    pub training_rows: Range<usize>,
    pub median: f64,
    pub scale: f64,
    pub valid_training_cells: usize,
    pub degenerate: bool,
}

/// Fit robust normalization only on explicitly valid training cells and apply
/// that immutable fit to the full column.
///
/// No split is inferred here: callers must supply the exact training range.
/// Invalid cells retain their reason and canonical NaN payload. A constant
/// training column is marked degenerate rather than emitted as a zero-valued
/// signal. If MAD is zero but the observations are not constant (for example a
/// sparse binary flag), population standard deviation is the deterministic
/// fallback scale so distinct valid values remain distinct.
pub fn normalize_feature_column_f64(
    column: &mut FeatureColumnF64,
    training_rows: Range<usize>,
) -> Result<RobustNormalizationFitF64> {
    if training_rows.start >= training_rows.end || training_rows.end > column.len() {
        bail!(
            "feature column `{}` normalization range {:?} is outside 0..{}",
            column.name,
            training_rows,
            column.len()
        );
    }

    let mut training_values = Vec::with_capacity(training_rows.len());
    for row in training_rows.clone() {
        if column.validity[row].is_valid() {
            let value = column.values[row];
            if !value.is_finite() {
                bail!(
                    "feature column `{}` row {row} is valid but non-finite before normalization",
                    column.name
                );
            }
            training_values.push(value);
        }
    }
    if training_values.is_empty() {
        bail!(
            "feature column `{}` has no valid cells in normalization training range {:?}",
            column.name,
            training_rows
        );
    }

    training_values.sort_by(f64::total_cmp);
    let median = median_sorted_f64(&training_values);
    let mut deviations: Vec<f64> = training_values
        .iter()
        .map(|value| (value - median).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let mad_scale = median_sorted_f64(&deviations) * MAD_TO_SIGMA_F64;
    let max_abs = training_values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let scale_floor = 32.0 * f64::EPSILON * max_abs.max(1.0);

    let scale = if mad_scale > scale_floor {
        mad_scale
    } else {
        let mean = training_values.iter().sum::<f64>() / training_values.len() as f64;
        let variance = training_values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / training_values.len() as f64;
        variance.sqrt()
    };
    let degenerate = !scale.is_finite() || scale <= scale_floor;

    if degenerate {
        for row in 0..column.len() {
            if column.validity[row].is_valid() {
                column.validity[row] = FeatureCellValidity::Degenerate;
                column.values[row] = f64::NAN;
            }
        }
    } else {
        for row in 0..column.len() {
            if !column.validity[row].is_valid() {
                column.values[row] = f64::NAN;
                continue;
            }
            let normalized = (column.values[row] - median) / scale;
            if normalized.is_finite() {
                column.values[row] = normalized.clamp(-Z_CLIP_F64, Z_CLIP_F64);
            } else {
                column.validity[row] = FeatureCellValidity::NonFinite;
                column.values[row] = f64::NAN;
            }
        }
    }

    Ok(RobustNormalizationFitF64 {
        training_rows,
        median,
        scale,
        valid_training_cells: training_values.len(),
        degenerate,
    })
}

fn median_sorted_f64(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        sorted[mid - 1] * 0.5 + sorted[mid] * 0.5
    } else {
        sorted[mid]
    }
}

/// Version 2 fits only the declared training rows and preserves typed invalid
/// cells instead of rewriting them to numeric zero.
pub const NORMALIZATION_TRANSFORM_SEMANTIC_VERSION: u32 = 2;
