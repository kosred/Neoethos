// Base classes and utilities for machine learning models
//
// Derived from the legacy reference implementation and fully maintained in Rust.
//
// This module provides:
// - EarlyStopper: Universal early stopping for training loops
// - ExpertModel: Abstract trait for all expert models
// - Training utilities for time-series aware data handling

use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tracing::*;

use crate::runtime::artifacts::{
    LabelMapping, RuntimeArtifactMetadata, TrainingSummaryMetadata,
    default_three_class_label_mapping,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::{PredictionMetadata, RuntimePrediction, RuntimePredictionError};

type ModelSaveFn = Box<dyn FnOnce(&Path) -> Result<()>>;

// ============================================================================
// EARLY STOPPING
// ============================================================================

/// Universal Early Stopping utility.
/// Stops training when validation metric stops improving.
///
/// Early-stopping helper for supervised training loops.
pub struct EarlyStopper {
    patience: usize,
    min_delta: f64,
    counter: usize,
    best_loss: Option<f64>,
    pub early_stop: bool,
}

impl EarlyStopper {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: None,
            early_stop: false,
        }
    }

    /// Call with validation loss. Returns true if should stop.
    /// Derived from legacy __call__ method (lines 38-48)
    pub fn check(&mut self, val_loss: f64) -> bool {
        if self.best_loss.is_none() {
            self.best_loss = Some(val_loss);
        } else if let Some(best) = self.best_loss {
            if val_loss > best - self.min_delta {
                self.counter += 1;
                if self.counter >= self.patience {
                    self.early_stop = true;
                }
            } else {
                self.best_loss = Some(val_loss);
                self.counter = 0;
            }
        }
        self.early_stop
    }
}

/// Return (patience, min_delta) with optional env overrides.
/// Derived from legacy get_early_stop_params (lines 51-69)
pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let mut patience = default_patience;
    let mut min_delta = default_min_delta;

    // Try to read env var for patience
    if let Ok(env_pat) = std::env::var("FOREX_BOT_EARLY_STOP_PATIENCE")
        && !env_pat.is_empty()
        && let Ok(val) = env_pat.parse::<usize>()
        && val > 0
    {
        patience = val;
    }

    // Try to read env var for min_delta
    if let Ok(env_delta) = std::env::var("FOREX_BOT_EARLY_STOP_MIN_DELTA")
        && !env_delta.is_empty()
        && let Ok(val) = env_delta.parse::<f64>()
    {
        min_delta = val;
    }

    (patience, min_delta)
}

// ============================================================================
// EXPERT MODEL TRAIT
// ============================================================================

/// Abstract base trait for all expert models.
/// Derived from legacy ExpertModel class (lines 71-127)
pub trait ExpertModel {
    /// Train the model.
    /// Derived from legacy fit method (lines 74-77)
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()>;

    /// Predict probabilities for classes [-1, 0, 1].
    ///
    /// Returns:
    ///     Array2<f32>: Shape (N, 3) where columns map to [neutral, buy, sell]
    ///                  Convention: col 0 -> neutral, col 1 -> buy, col 2 -> sell
    ///
    /// Derived from legacy predict_proba method (lines 79-89)
    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>>;

    /// Save model artifacts to directory.
    /// Derived from legacy save method (lines 91-94)
    fn save(&self, path: &Path) -> Result<()>;

    /// Load model artifacts from directory.
    /// Derived from legacy load method (lines 96-99)
    fn load(&mut self, path: &Path) -> Result<()>;

    /// Helper for atomic model saving with rotation/backup.
    /// Keeps 'model.pt' (current) and 'model.pt.bak' (previous).
    ///
    /// Derived from legacy _atomic_save method (lines 101-126)
    fn atomic_save(&self, save_func: ModelSaveFn, target_path: &Path) -> Result<()> {
        let temp_path = target_path.with_extension("tmp");
        let backup_path = target_path.with_extension("bak");

        // Save to temp file
        save_func(&temp_path)
            .with_context(|| format!("Failed to save to temp file: {}", temp_path.display()))?;

        // Rotate: current -> backup, temp -> current
        if target_path.exists() {
            if backup_path.exists() {
                std::fs::remove_file(&backup_path).with_context(|| {
                    format!("Failed to delete old backup: {}", backup_path.display())
                })?;
            }
            std::fs::rename(target_path, &backup_path).with_context(|| {
                format!("Failed to rotate to backup: {}", backup_path.display())
            })?;
        }

        if target_path.exists() {
            std::fs::remove_file(target_path).with_context(|| {
                format!(
                    "Failed to remove previous target: {}",
                    target_path.display()
                )
            })?;
        }
        std::fs::rename(&temp_path, target_path)
            .with_context(|| format!("Failed to move temp to target: {}", target_path.display()))?;

        Ok(())
    }
}

// ============================================================================
// DATA CONVERSION UTILITIES
// ============================================================================

/// Convert a DataFrame to a float32 ndarray suitable for models.
///
/// Derived from legacy dataframe_to_float32_numpy (lines 129-139)
pub fn dataframe_to_float32_array(df: &DataFrame) -> Result<Array2<f32>> {
    let n_rows = df.height();
    let n_cols = df.width();

    let mut array_data = vec![0.0_f32; n_rows * n_cols];
    for (col_idx, col) in df.get_columns().iter().enumerate() {
        let series_f64 = col
            .cast(&DataType::Float64)
            .with_context(|| format!("Failed to cast column {} to f64", col.name()))?;

        let ca = series_f64
            .f64()
            .with_context(|| format!("Failed to get f64 chunked array for {}", col.name()))?;

        for (row_idx, val) in ca.into_iter().enumerate() {
            let value = val.with_context(|| {
                format!(
                    "Column {} contains null at row {}; model features must be fully materialized",
                    col.name(),
                    row_idx
                )
            })?;
            if !value.is_finite() {
                return Err(anyhow::anyhow!(
                    "Column {} contains non-finite value {} at row {}",
                    col.name(),
                    value,
                    row_idx
                ));
            }

            array_data[row_idx * n_cols + col_idx] = value as f32;
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), array_data)
        .context("Failed to create Array2 from DataFrame")
}

/// Extract a numeric column as strict finite Float64 values.
/// Nulls or non-finite values are treated as structural data errors.
pub fn strict_numeric_column_values(df: &DataFrame, column_name: &str) -> Result<Vec<f64>> {
    let series = df
        .column(column_name)
        .with_context(|| format!("Missing required numeric column {column_name}"))?
        .cast(&DataType::Float64)
        .with_context(|| format!("Failed to cast column {column_name} to Float64"))?;

    series
        .f64()
        .with_context(|| format!("Failed to access column {column_name} as Float64"))?
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            let value = value.with_context(|| {
                format!(
                    "Column {column_name} contains null at row {idx}; downstream models require strict numeric input"
                )
            })?;
            if !value.is_finite() {
                return Err(anyhow::anyhow!(
                    "Column {column_name} contains non-finite value {} at row {}",
                    value,
                    idx
                ));
            }
            Ok(value)
        })
        .collect()
}

/// Extract ordered feature column names from a dataframe.
pub fn feature_columns_from_dataframe(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

/// Return the canonical three-class label mapping used by the runtime contract.
pub fn canonical_three_class_label_mapping() -> Vec<LabelMapping> {
    default_three_class_label_mapping()
}

/// Build runtime artifact metadata from the shared model contract.
pub fn build_runtime_artifact_metadata(
    model_name: impl Into<String>,
    family: ModelFamily,
    state: CapabilityState,
    feature_columns: Vec<String>,
    label_mapping: Vec<LabelMapping>,
    training_summary: TrainingSummaryMetadata,
) -> RuntimeArtifactMetadata {
    try_build_runtime_artifact_metadata(
        model_name,
        family,
        state,
        feature_columns,
        label_mapping,
        training_summary,
    )
    .expect("runtime artifact metadata contract violation")
}

/// Build runtime artifact metadata from the shared model contract without panicking.
pub fn try_build_runtime_artifact_metadata(
    model_name: impl Into<String>,
    family: ModelFamily,
    state: CapabilityState,
    feature_columns: Vec<String>,
    label_mapping: Vec<LabelMapping>,
    training_summary: TrainingSummaryMetadata,
) -> Result<RuntimeArtifactMetadata> {
    let model_name = model_name.into();
    let mut label_mapping = label_mapping;
    if feature_columns.is_empty() {
        bail!("runtime artifact metadata requires at least one feature column");
    }
    if label_mapping.is_empty() {
        warn!(
            "runtime artifact metadata for {} is missing label mapping; defaulting to canonical three-class mapping",
            model_name
        );
        label_mapping = default_three_class_label_mapping();
    }
    if training_summary.dataset_rows == 0 {
        bail!("runtime artifact metadata requires a non-zero dataset row count");
    }
    let training_summary = normalize_training_summary_for_metadata(&model_name, training_summary)?;
    Ok(RuntimeArtifactMetadata::new(
        model_name,
        family,
        state,
        feature_columns,
        label_mapping,
        training_summary,
    ))
}

fn normalize_training_summary_for_metadata(
    model_name: &str,
    mut summary: TrainingSummaryMetadata,
) -> Result<TrainingSummaryMetadata> {
    let current_total = summary.train_rows + summary.val_rows;
    if current_total != summary.dataset_rows {
        if summary.train_rows <= summary.dataset_rows {
            let repaired_val_rows = summary.dataset_rows.saturating_sub(summary.train_rows);
            warn!(
                "runtime artifact metadata train/val mismatch for {}: repairing train_rows={} val_rows={} dataset_rows={} -> val_rows={}",
                model_name,
                summary.train_rows,
                summary.val_rows,
                summary.dataset_rows,
                repaired_val_rows
            );
            summary.val_rows = repaired_val_rows;
        } else if summary.val_rows <= summary.dataset_rows {
            let repaired_train_rows = summary.dataset_rows.saturating_sub(summary.val_rows);
            warn!(
                "runtime artifact metadata train/val mismatch for {}: repairing train_rows={} val_rows={} dataset_rows={} -> train_rows={}",
                model_name,
                summary.train_rows,
                summary.val_rows,
                summary.dataset_rows,
                repaired_train_rows
            );
            summary.train_rows = repaired_train_rows;
        } else {
            bail!(
                "runtime artifact metadata cannot repair split rows: train_rows={} val_rows={} dataset_rows={}",
                summary.train_rows,
                summary.val_rows,
                summary.dataset_rows
            );
        }
    }

    if summary.train_rows == 0 && summary.dataset_rows > 0 {
        warn!(
            "runtime artifact metadata for {} has zero train rows; promoting split to train_rows={} val_rows=0",
            model_name, summary.dataset_rows
        );
        summary.train_rows = summary.dataset_rows;
        summary.val_rows = 0;
    }

    Ok(summary)
}

/// Build runtime prediction output from the shared model contract.
pub fn build_runtime_prediction(
    model_name: impl Into<String>,
    family: ModelFamily,
    state: CapabilityState,
    class_probabilities: [f32; 3],
    confidence: Option<f32>,
    abstain_recommended: Option<bool>,
) -> Result<RuntimePrediction, RuntimePredictionError> {
    RuntimePrediction::try_new(
        class_probabilities,
        confidence,
        abstain_recommended,
        PredictionMetadata::new(model_name, family, state),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_runtime_prediction_with_details(
    model_name: impl Into<String>,
    family: ModelFamily,
    state: CapabilityState,
    class_probabilities: [f32; 3],
    confidence: Option<f32>,
    abstain_recommended: Option<bool>,
    execution_backend: Option<String>,
    degraded_reason: Option<String>,
) -> Result<RuntimePrediction, RuntimePredictionError> {
    RuntimePrediction::try_new(
        class_probabilities,
        confidence,
        abstain_recommended,
        PredictionMetadata::new(model_name, family, state)
            .with_runtime_details(execution_backend, degraded_reason),
    )
}

pub fn three_class_runtime_confidence(row_values: [f32; 3]) -> Result<(f32, bool)> {
    let mut normalized = row_values;
    let mut sum = 0.0_f32;
    for value in &normalized {
        if !value.is_finite() || *value < 0.0 {
            bail!("runtime predictions require finite non-negative probabilities");
        }
        if *value > 1.0 + 1e-4 {
            bail!("runtime predictions require probability rows bounded by 1.0");
        }
        sum += *value;
    }
    if !sum.is_finite() || sum <= f32::EPSILON {
        bail!("runtime predictions require probability rows with positive mass");
    }
    for value in &mut normalized {
        *value /= sum;
    }

    let mut sorted = normalized;
    sorted.sort_by(|left, right| right.total_cmp(left));
    let top = sorted[0];
    let runner_up = sorted[1];
    let margin = (top - runner_up).max(0.0);
    let entropy = normalized
        .iter()
        .copied()
        .filter(|value| *value > 1e-8)
        .map(|value| -value * value.ln())
        .sum::<f32>()
        / (3.0_f32.ln());
    let sharpness = (1.0 - entropy).clamp(0.0, 1.0);
    let confidence = (0.6 * top + 0.25 * margin + 0.15 * sharpness).clamp(0.0, 1.0);
    Ok((confidence, top < 0.5 || confidence < 0.56))
}

// ============================================================================
// TIME-SERIES VALIDATION
// ============================================================================

/// Validate that DataFrame index is monotonically increasing (time-ordered).
///
/// This is critical for time-series models to prevent look-ahead bias.
///
/// Derived from legacy validate_time_ordering (lines 142-184)
pub fn validate_time_ordering(df: &DataFrame, context: &str) -> Result<bool> {
    if df.height() == 0 {
        return Ok(true);
    }

    let time_cols = ["timestamp", "time", "date", "datetime"];
    for col_name in time_cols {
        if let Ok(series) = df.column(col_name)
            && let Ok(ca) = series.cast(&polars::datatypes::DataType::Int64)
            && let Ok(ca_i64) = ca.i64()
        {
            let mut prev = i64::MIN;
            for val in ca_i64.into_iter().flatten() {
                if val < prev {
                    return Err(anyhow::anyhow!(
                        "{}: Datetime column '{}' is NOT monotonically increasing. Look-ahead bias structural risk!",
                        context,
                        col_name
                    ));
                }
                prev = val;
            }
            return Ok(true);
        }
    }

    warn!(
        "{}: No explicit time column found (timestamp/time/date). Assuming data is pre-sorted.",
        context
    );
    Ok(true)
}

/// Splits data for time-series training with an embargo gap.
///
/// Derived from legacy time_series_train_val_split (lines 187-212)
pub fn time_series_train_val_split(
    x: &DataFrame,
    y: &Series,
    val_ratio: f64,
    min_train_samples: usize,
    embargo_samples: usize, // HPC FIX: Guaranteed memory flush
) -> Result<(DataFrame, DataFrame, Series, Series)> {
    let n = x.height();
    let val_size = (n as f64 * val_ratio) as usize;
    let mut train_end = n.saturating_sub(val_size).saturating_sub(embargo_samples);

    if train_end < min_train_samples {
        // If dataset too small, reduce embargo but maintain at least 100 bars
        let reduced_embargo = embargo_samples.min(100.max(n / 10));
        train_end = n.saturating_sub(val_size).saturating_sub(reduced_embargo);
    }

    let x_train = x.slice(0, train_end);
    let y_train = y.slice(0, train_end);

    let val_start = train_end + embargo_samples;
    let val_len = n.saturating_sub(val_start);
    let x_val = x.slice(val_start as i64, val_len);
    let y_val = y.slice(val_start as i64, val_len);

    Ok((x_train, x_val, y_train, y_val))
}

// ============================================================================
// SAMPLING UTILITIES
// ============================================================================

/// Downsample data while preserving class distribution.
///
/// Used to limit memory/compute for large datasets while maintaining
/// representative class balance.
///
/// Derived from legacy stratified_downsample (lines 215-289)
pub fn stratified_downsample(
    x: &DataFrame,
    y: &Series,
    max_samples: usize,
    random_state: u64,
) -> Result<(DataFrame, Series)> {
    let n = x.height();

    if max_samples == 0 || n <= max_samples {
        return Ok((x.clone(), y.clone()));
    }

    use rand::SeedableRng;
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(random_state);

    // Group by class
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, label) in y_ca.into_iter().enumerate() {
        if let Some(lbl) = label {
            class_indices.entry(lbl).or_default().push(idx);
        }
    }

    // Calculate samples per class (proportional)
    let total = n;
    let mut sampled_indices = Vec::new();

    for (_label, indices) in class_indices.iter() {
        // Proportion of this class in original data
        let class_ratio = indices.len() as f64 / total as f64;
        // Target samples for this class
        let target_count = ((max_samples as f64 * class_ratio) as usize).max(1);
        // Actual samples to take
        let take_count = indices.len().min(target_count);

        if take_count > 0 {
            let mut indices_clone = indices.clone();
            indices_clone.shuffle(&mut rng);
            sampled_indices.extend_from_slice(&indices_clone[..take_count]);
        }
    }

    // Trim to max if over
    if sampled_indices.len() > max_samples {
        sampled_indices.shuffle(&mut rng);
        sampled_indices.truncate(max_samples);
    }

    // Sort to maintain temporal order
    sampled_indices.sort_unstable();

    // Create downsampled DataFrame and Series
    // Polars 0.47: take() expects ChunkedArray<UInt32Type>
    let indices: Vec<u32> = sampled_indices.iter().map(|&i| i as u32).collect();
    let indices_ca = Series::new("indices".into(), indices).u32()?.clone();
    let x_out = x.take(&indices_ca)?;
    let y_out = y.take(&indices_ca)?;

    info!(
        "Downsampled from {} to {} samples ({:.1}%)",
        n,
        x_out.height(),
        (x_out.height() as f64 / n as f64) * 100.0
    );

    Ok((x_out, y_out))
}

// ============================================================================
// CLASS WEIGHTING
// ============================================================================

/// Compute balanced class weights for imbalanced classification.
///
/// Uses inverse frequency weighting: rare classes get higher weights.
///
/// Derived from legacy compute_class_weights (lines 292-319)
pub fn compute_class_weights(y: &Series) -> Result<HashMap<i64, f64>> {
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut class_counts: HashMap<i64, usize> = HashMap::new();
    let mut n_samples = 0;

    for label in y_ca.into_iter().flatten() {
        *class_counts.entry(label).or_insert(0) += 1;
        n_samples += 1;
    }

    let n_classes = class_counts.len();
    let mut weights = HashMap::new();

    for (cls, count) in class_counts.iter() {
        if *count > 0 {
            // sklearn-style balanced weight
            weights.insert(*cls, n_samples as f64 / (n_classes as f64 * *count as f64));
        }
    }

    Ok(weights)
}

/// Compute per-sample weights based on class frequency.
///
/// Derived from legacy compute_sample_weights (lines 322-343)
pub fn compute_sample_weights(y: &Series) -> Result<Vec<f32>> {
    let class_weights = compute_class_weights(y)?;
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut sample_weights = Vec::with_capacity(y.len());

    for label in y_ca.into_iter() {
        if let Some(lbl) = label {
            let weight = class_weights.get(&lbl).copied().unwrap_or(1.0);
            sample_weights.push(weight as f32);
        } else {
            sample_weights.push(1.0);
        }
    }

    Ok(sample_weights)
}

// ============================================================================
// FEATURE DRIFT DETECTION
// ============================================================================

/// Detect feature drift between training and validation data.
///
/// Uses Population Stability Index (PSI) or simple mean/std comparison
/// to identify features that have shifted significantly.
///
/// Derived from legacy detect_feature_drift (lines 346-477)
pub fn detect_feature_drift(
    train_df: &DataFrame,
    val_df: &DataFrame,
    _threshold: f64,
    method: &str,
) -> Result<FeatureDriftReport> {
    if train_df.height() == 0 || val_df.height() == 0 {
        return Ok(FeatureDriftReport {
            drifted_features: vec![],
            drift_scores: HashMap::new(),
            summary: "Insufficient data for drift detection".to_string(),
            critical: false,
        });
    }

    // Find common numeric columns
    let train_cols: std::collections::HashSet<_> =
        train_df.get_column_names().iter().copied().collect();
    let val_cols: std::collections::HashSet<_> =
        val_df.get_column_names().iter().copied().collect();
    let common_cols: Vec<_> = train_cols.intersection(&val_cols).copied().collect();

    let numeric_cols: Vec<String> = common_cols
        .iter()
        .filter(|&col_name| {
            if let (Ok(train_col), Ok(val_col)) =
                (train_df.column(col_name), val_df.column(col_name))
            {
                matches!(
                    train_col.dtype(),
                    DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
                ) && matches!(
                    val_col.dtype(),
                    DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
                )
            } else {
                false
            }
        })
        .map(|s| s.to_string())
        .collect();

    if numeric_cols.is_empty() {
        return Ok(FeatureDriftReport {
            drifted_features: vec![],
            drift_scores: HashMap::new(),
            summary: "No numeric features to check".to_string(),
            critical: false,
        });
    }

    // HPC FIX: Regime-Aware Drift Thresholding (lines 405-417)
    let base_threshold = std::env::var("FOREX_BOT_DRIFT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.20);

    let mut vol_scale = 1.0;
    if let Ok(vol_col) = val_df.column("realized_volatility")
        && let Ok(series) = vol_col.cast(&DataType::Float64)
        && let Ok(ca) = series.f64()
    {
        let vol_sum: f64 = ca.into_iter().flatten().sum();
        let count = ca.into_iter().flatten().count();
        if count > 0 && vol_sum > 0.0 {
            vol_scale = ((vol_sum / count as f64) * 1000.0).clamp(0.2, 5.0);
        }
    }
    let threshold = base_threshold * vol_scale;

    let mut drift_scores = HashMap::new();
    let mut drifted_features = Vec::new();

    // HPC: Use parallel processing for drift detection (lines 436-437)
    use rayon::prelude::*;

    let results: Vec<_> = numeric_cols
        .par_iter()
        .filter_map(|col| {
            let train_col = train_df.column(col).ok()?;
            let val_col = val_df.column(col).ok()?;

            // Polars 0.47: Convert Column to Series
            let train_series = train_col.as_materialized_series().clone();
            let val_series = val_col.as_materialized_series().clone();

            let train_vals = extract_numeric_values(&train_series).ok()?;
            let val_vals = extract_numeric_values(&val_series).ok()?;

            if train_vals.len() < 10 || val_vals.len() < 10 {
                return None;
            }

            let score = if method == "psi" {
                compute_psi(&train_vals, &val_vals, 10)
            } else {
                compute_stats_drift(&train_vals, &val_vals)
            };

            Some((col.clone(), score))
        })
        .collect();

    for (col, score) in results {
        drift_scores.insert(col.clone(), score);
        if score >= threshold {
            drifted_features.push(col);
        }
    }

    // Calculate overall drift severity (lines 448-460)
    let critical_threshold = 0.25;
    let critical_count = drift_scores
        .values()
        .filter(|&&s| s >= critical_threshold)
        .count();
    let total_features = drift_scores.len();

    let (critical, summary) = if critical_count > total_features * 3 / 10 {
        (
            true,
            format!(
                "CRITICAL: {}/{} features have significant drift",
                critical_count, total_features
            ),
        )
    } else if drifted_features.len() > total_features * 2 / 10 {
        (
            false,
            format!(
                "WARNING: {}/{} features show drift",
                drifted_features.len(),
                total_features
            ),
        )
    } else {
        (
            false,
            format!(
                "OK: {}/{} features with minor drift",
                drifted_features.len(),
                total_features
            ),
        )
    };

    if !drifted_features.is_empty() {
        let mut sorted_drifted = drifted_features.clone();
        sorted_drifted.sort_by(|a, b| {
            let score_a = drift_scores.get(a).copied().unwrap_or(f64::NEG_INFINITY);
            let score_b = drift_scores.get(b).copied().unwrap_or(f64::NEG_INFINITY);
            score_b.total_cmp(&score_a).then_with(|| a.cmp(b))
        });

        let top_5: Vec<_> = sorted_drifted.iter().take(5).map(|s| s.as_str()).collect();
        let msg = format!(
            "Feature drift detected: {}. Top drifted: {:?}",
            summary, top_5
        );

        if critical || summary.starts_with("WARNING:") {
            warn!("{}", msg);
        } else {
            info!("{}", msg);
        }
    }

    Ok(FeatureDriftReport {
        drifted_features,
        drift_scores,
        summary,
        critical,
    })
}

pub struct FeatureDriftReport {
    pub drifted_features: Vec<String>,
    pub drift_scores: HashMap<String, f64>,
    pub summary: String,
    pub critical: bool,
}

/// Extract numeric values from a Polars series
fn extract_numeric_values(series: &Series) -> Result<Vec<f64>> {
    let series_f64 = series.cast(&DataType::Float64)?;
    let ca = series_f64.f64()?;
    let values: Vec<f64> = ca
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .collect();
    Ok(values)
}

/// Compute Population Stability Index (PSI) between two distributions.
///
/// PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
///
/// Interpretation:
/// - PSI < 0.1: No significant change
/// - 0.1 <= PSI < 0.25: Moderate change
/// - PSI >= 0.25: Significant change
///
/// Derived from legacy _compute_psi (lines 480-535)
pub fn compute_psi(expected: &[f64], actual: &[f64], n_bins: usize) -> f64 {
    let eps = 1e-6;
    let n_bins = n_bins.max(3);

    // Create bins from expected distribution
    let mut breakpoints = compute_percentiles(expected, n_bins + 1);
    breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    breakpoints.dedup();

    if breakpoints.len() < 2 {
        return 0.0;
    }

    let expected_counts = histogram(expected, &breakpoints);
    let actual_counts = histogram(actual, &breakpoints);

    // If bins are too sparse, retry with coarser bins
    if expected_counts.iter().any(|&c| c < 3) || actual_counts.iter().any(|&c| c < 3) {
        let coarse_bins = (3_usize).max((breakpoints.len() - 1).min(5));
        let coarse_breaks = compute_percentiles(expected, coarse_bins + 1);
        if coarse_breaks.len() >= 2 && coarse_breaks.len() < breakpoints.len() {
            let expected_counts = histogram(expected, &coarse_breaks);
            let actual_counts = histogram(actual, &coarse_breaks);
            return compute_psi_from_counts(
                &expected_counts,
                &actual_counts,
                expected.len(),
                actual.len(),
                eps,
            );
        }
    }

    compute_psi_from_counts(
        &expected_counts,
        &actual_counts,
        expected.len(),
        actual.len(),
        eps,
    )
}

fn compute_psi_from_counts(
    expected_counts: &[usize],
    actual_counts: &[usize],
    expected_len: usize,
    actual_len: usize,
    eps: f64,
) -> f64 {
    let expected_pct: Vec<f64> = expected_counts
        .iter()
        .map(|&c| (c as f64 / (expected_len as f64 + eps)).clamp(eps, 1.0))
        .collect();
    let actual_pct: Vec<f64> = actual_counts
        .iter()
        .map(|&c| (c as f64 / (actual_len as f64 + eps)).clamp(eps, 1.0))
        .collect();

    let psi: f64 = expected_pct
        .iter()
        .zip(actual_pct.iter())
        .map(|(&exp, &act)| {
            let diff = act - exp;
            let ratio = (act / exp).ln();
            diff * ratio
        })
        .sum();

    psi
}

fn compute_percentiles(data: &[f64], n: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() || n == 0 {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.total_cmp(b));

    (0..=n)
        .map(|i| {
            let pct = i as f64 / n as f64;
            let idx = ((sorted.len() - 1) as f64 * pct) as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

fn histogram(data: &[f64], breakpoints: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; breakpoints.len().saturating_sub(1)];

    for &val in data {
        for i in 0..breakpoints.len() - 1 {
            if val >= breakpoints[i] && (i == breakpoints.len() - 2 || val < breakpoints[i + 1]) {
                counts[i] += 1;
                break;
            }
        }
    }

    counts
}

/// Fallback drift metric based on mean/std shift.
///
/// Derived from legacy _compute_stats_drift (lines 538-554)
pub fn compute_stats_drift(train_vals: &[f64], val_vals: &[f64]) -> f64 {
    let train_mean = train_vals.iter().sum::<f64>() / train_vals.len() as f64;
    let val_mean = val_vals.iter().sum::<f64>() / val_vals.len() as f64;

    let train_std = {
        let variance = train_vals
            .iter()
            .map(|&x| (x - train_mean).powi(2))
            .sum::<f64>()
            / train_vals.len() as f64;
        variance.sqrt()
    };
    let val_std = {
        let variance = val_vals
            .iter()
            .map(|&x| (x - val_mean).powi(2))
            .sum::<f64>()
            / val_vals.len() as f64;
        variance.sqrt()
    };

    let eps = f64::EPSILON;

    if train_std > eps {
        let mean_shift = (val_mean - train_mean).abs() / train_std.max(eps);
        let std_ratio = val_std / train_std.max(eps);
        mean_shift + (1.0 - std_ratio).abs()
    } else {
        0.0
    }
}

// ============================================================================
// ROBUST SCALING (HPC ADVANCEMENT)
// ============================================================================

/// Robust Scaler that handles NaN and Infinite values efficiently.
/// Derived from legacy advancements in normalization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RobustScaler {
    pub mean: Option<Array2<f32>>,
    pub scale: Option<Array2<f32>>,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            scale: None,
        }
    }

    /// Fit the scaler to data, ignoring NaN/Inf.
    pub fn fit(&mut self, data: &Array2<f32>) -> Result<()> {
        let n_features = data.ncols();
        let mut means = Array2::zeros((1, n_features));
        let mut scales = Array2::zeros((1, n_features));

        for j in 0..n_features {
            let col = data.column(j);
            let valid_values: Vec<f32> = col.iter().filter(|&&x| x.is_finite()).copied().collect();

            if valid_values.is_empty() {
                means[[0, j]] = 0.0;
                scales[[0, j]] = 1.0;
                continue;
            }

            let sum: f32 = valid_values.iter().sum();
            let count = valid_values.len() as f32;
            let mean = sum / count;

            let variance: f32 = valid_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / count;

            let std = variance.sqrt().max(1e-3);

            means[[0, j]] = mean;
            scales[[0, j]] = std;
        }

        self.mean = Some(means);
        self.scale = Some(scales);
        Ok(())
    }

    /// Transform data using fitted parameters.
    pub fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mean = self.mean.as_ref().context("Scaler not fitted")?;
        let scale = self.scale.as_ref().context("Scaler not fitted")?;

        let mut transformed = data.clone();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let val = transformed[[i, j]];
                if val.is_finite() {
                    transformed[[i, j]] = (val - mean[[0, j]]) / scale[[0, j]];
                } else {
                    // Replace NaN/Inf with 0.0 after normalization (mean)
                    transformed[[i, j]] = 0.0;
                }
            }
        }

        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn feature_columns_from_dataframe_preserves_column_order() -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("open".into(), vec![1.0_f64, 2.0]).into(),
            Series::new("high".into(), vec![1.5_f64, 2.5]).into(),
            Series::new("close".into(), vec![1.25_f64, 2.25]).into(),
        ])?;

        let columns = feature_columns_from_dataframe(&df);
        assert_eq!(
            columns,
            vec!["open".to_string(), "high".to_string(), "close".to_string()]
        );
        Ok(())
    }

    #[test]
    fn build_runtime_artifact_metadata_populates_feature_and_label_contracts() {
        let metadata = build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            vec!["rsi".to_string(), "atr".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(12_345, 10_000, 2_345),
        );

        assert_eq!(metadata.model_name, "lightgbm");
        assert_eq!(metadata.family, ModelFamily::Tree);
        assert_eq!(metadata.state, CapabilityState::Implemented);
        assert_eq!(metadata.feature_columns, vec!["rsi", "atr"]);
        assert_eq!(
            metadata.label_mapping,
            vec![
                LabelMapping::new(-1, 2),
                LabelMapping::new(0, 0),
                LabelMapping::new(1, 1),
            ]
        );
        assert_eq!(metadata.training_summary.dataset_rows, 12_345);
        assert_eq!(metadata.training_summary.train_rows, 10_000);
        assert_eq!(metadata.training_summary.val_rows, 2_345);
    }

    #[test]
    #[should_panic(expected = "runtime artifact metadata requires at least one feature column")]
    fn build_runtime_artifact_metadata_rejects_empty_feature_columns() {
        let _ = build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            Vec::new(),
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 8, 2),
        );
    }

    #[test]
    fn try_build_runtime_artifact_metadata_returns_error_for_invalid_contract() {
        let err = try_build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            Vec::new(),
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 8, 2),
        )
        .expect_err("expected contract validation error");

        assert!(
            err.to_string()
                .contains("requires at least one feature column")
        );
    }

    #[test]
    fn try_build_runtime_artifact_metadata_repairs_train_val_mismatch_when_possible() {
        let metadata = try_build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            vec!["rsi".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 8, 1),
        )
        .expect("metadata split should be repaired");

        assert_eq!(metadata.training_summary.dataset_rows, 10);
        assert_eq!(metadata.training_summary.train_rows, 8);
        assert_eq!(metadata.training_summary.val_rows, 2);
    }

    #[test]
    fn try_build_runtime_artifact_metadata_defaults_empty_label_mapping() {
        let metadata = try_build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            vec!["rsi".to_string()],
            Vec::new(),
            TrainingSummaryMetadata::new(10, 8, 2),
        )
        .expect("empty label mapping should be defaulted");

        assert_eq!(
            metadata.label_mapping,
            canonical_three_class_label_mapping()
        );
    }

    #[test]
    fn try_build_runtime_artifact_metadata_rejects_irreparable_train_val_mismatch() {
        let err = try_build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            vec!["rsi".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(10, 12, 12),
        )
        .expect_err("split larger than dataset should remain invalid");

        assert!(err.to_string().contains("cannot repair split rows"));
    }

    #[test]
    fn try_build_runtime_artifact_metadata_promotes_zero_train_rows() {
        let metadata = try_build_runtime_artifact_metadata(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            vec!["rsi".to_string()],
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(7, 0, 7),
        )
        .expect("zero-train split should be promoted");

        assert_eq!(metadata.training_summary.dataset_rows, 7);
        assert_eq!(metadata.training_summary.train_rows, 7);
        assert_eq!(metadata.training_summary.val_rows, 0);
    }

    #[test]
    fn build_runtime_prediction_attaches_metadata_and_validates_probability_shape() -> Result<()> {
        let prediction = build_runtime_prediction(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            [0.1, 0.7, 0.2],
            Some(0.7),
            Some(false),
        )?;

        let (probs, confidence, abstain, metadata) = prediction.parts();
        assert_eq!(probs, [0.1, 0.7, 0.2]);
        assert_eq!(confidence, Some(0.7));
        assert_eq!(abstain, Some(false));
        assert_eq!(metadata.model_name, "lightgbm");
        assert_eq!(metadata.family, ModelFamily::Tree);
        assert_eq!(metadata.state, CapabilityState::Implemented);
        Ok(())
    }

    #[test]
    fn build_runtime_prediction_rejects_invalid_confidence() {
        let err = build_runtime_prediction(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            [0.1, 0.7, 0.2],
            Some(1.5),
            Some(false),
        )
        .expect_err("invalid confidence should fail");

        assert!(err.to_string().contains("invalid confidence"));
    }

    #[test]
    fn build_runtime_prediction_with_details_attaches_backend_and_degraded_reason() -> Result<()> {
        let prediction = build_runtime_prediction_with_details(
            "lightgbm",
            ModelFamily::Tree,
            CapabilityState::Implemented,
            [0.1, 0.7, 0.2],
            Some(0.7),
            Some(false),
            Some("tree_surrogate".to_string()),
            Some("native_lightgbm_unavailable".to_string()),
        )?;

        let (_, _, _, metadata) = prediction.parts();
        assert_eq!(
            metadata.execution_backend.as_deref(),
            Some("tree_surrogate")
        );
        assert_eq!(
            metadata.degraded_reason.as_deref(),
            Some("native_lightgbm_unavailable")
        );
        Ok(())
    }

    #[test]
    fn three_class_runtime_confidence_abstains_on_tight_top_two_margin() -> Result<()> {
        let (confidence, abstain) = three_class_runtime_confidence([0.51, 0.49, 0.0])?;
        assert!(
            confidence < 0.56,
            "tight top-two split should stay low-confidence"
        );
        assert!(abstain, "tight top-two split should recommend abstain");
        Ok(())
    }

    #[test]
    fn three_class_runtime_confidence_accepts_clear_decisive_rows() -> Result<()> {
        let (confidence, abstain) = three_class_runtime_confidence([0.8, 0.1, 0.1])?;
        assert!(
            confidence > 0.56,
            "clear winner should produce strong confidence"
        );
        assert!(!abstain, "clear winner should not recommend abstain");
        Ok(())
    }

    #[test]
    fn three_class_runtime_confidence_normalizes_probability_mass() -> Result<()> {
        let (confidence_a, abstain_a) = three_class_runtime_confidence([0.6, 0.3, 0.1])?;
        let (confidence_b, abstain_b) = three_class_runtime_confidence([0.3, 0.15, 0.05])?;
        assert!((confidence_a - confidence_b).abs() < 1e-6);
        assert_eq!(abstain_a, abstain_b);
        Ok(())
    }

    #[test]
    fn three_class_runtime_confidence_rejects_probabilities_above_one() {
        let err = three_class_runtime_confidence([1.2, 0.1, 0.1])
            .expect_err("probabilities above one must fail");
        assert!(err.to_string().contains("bounded by 1.0"));
    }

    #[test]
    fn dataframe_to_float32_array_still_builds_row_major_contract() -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("a".into(), vec![1.0_f64, 2.0]).into(),
            Series::new("b".into(), vec![3.0_f64, 4.0]).into(),
        ])?;

        let array = dataframe_to_float32_array(&df)?;
        assert_eq!(array, array![[1.0_f32, 3.0_f32], [2.0_f32, 4.0_f32]]);
        Ok(())
    }

    #[test]
    fn dataframe_to_float32_array_rejects_nulls() -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("a".into(), vec![Some(1.0_f64), None]).into(),
            Series::new("b".into(), vec![Some(3.0_f64), Some(4.0)]).into(),
        ])?;

        let err = dataframe_to_float32_array(&df).expect_err("null feature row should fail");
        assert!(err.to_string().contains("contains null"));
        Ok(())
    }

    #[test]
    fn strict_numeric_column_values_rejects_non_finite_values() -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("close".into(), vec![1.0_f64, f64::NAN]).into(),
        ])?;

        let err = strict_numeric_column_values(&df, "close")
            .expect_err("non-finite numeric values should fail");
        assert!(err.to_string().contains("non-finite"));
        Ok(())
    }
}
