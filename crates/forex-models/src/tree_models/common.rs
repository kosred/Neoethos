use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_artifact_metadata, canonical_three_class_label_mapping, compute_sample_weights,
    dataframe_to_float32_array, feature_columns_from_dataframe,
};
use crate::runtime::artifacts::{LabelMapping, RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const XGBOOST_MODEL_FILE_NAME: &str = "model.bin";
pub const LIGHTGBM_MODEL_FILE_NAME: &str = "model.txt";
pub const CATBOOST_MODEL_FILE_NAME: &str = "model.cbm";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeLocalFallbackArtifact {
    pub feature_columns: Vec<String>,
    pub training_summary: TrainingSummaryMetadata,
    pub class_priors: Vec<f32>,
    pub feature_means: Vec<f32>,
    pub feature_scales: Vec<f32>,
    pub class_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    pub class_variances: Vec<Vec<f32>>,
}

pub fn canonical_tree_label_mapping() -> Vec<LabelMapping> {
    canonical_three_class_label_mapping()
}

pub fn remap_labels_to_tree_targets(y: &Series) -> Result<Vec<f32>> {
    let y_i64 = y
        .cast(&DataType::Int64)
        .context("failed to cast labels to Int64 for tree model backend")?;
    let y_ca = y_i64
        .i64()
        .context("failed to access Int64 labels for tree model backend")?;

    y_ca.into_iter()
        .map(|value| match value {
            Some(-1) => Ok(2.0_f32),
            Some(0) => Ok(0.0_f32),
            Some(1) => Ok(1.0_f32),
            Some(other) => bail!("unsupported tree-model label: {other}; expected one of -1, 0, 1"),
            None => bail!("tree-model labels may not contain nulls"),
        })
        .collect()
}

pub fn dataframe_to_row_major_vec(df: &DataFrame) -> Result<(Vec<f32>, usize, usize)> {
    let array = dataframe_to_float32_array(df)?;
    let rows = array.nrows();
    let cols = array.ncols();
    let flat = array
        .as_slice_memory_order()
        .context("tree model feature matrix must be contiguous in memory")?
        .to_vec();
    Ok((flat, rows, cols))
}

pub fn ensure_feature_columns_match(expected: &[String], df: &DataFrame) -> Result<()> {
    if expected.is_empty() {
        bail!("persisted tree model is missing expected feature columns");
    }

    let actual = feature_columns_from_dataframe(df);
    if actual != expected {
        bail!(
            "feature column mismatch for persisted tree model; expected {:?}, got {:?}",
            expected,
            actual
        );
    }

    Ok(())
}

pub fn tree_runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Tree,
        CapabilityState::Implemented,
        feature_columns,
        canonical_tree_label_mapping(),
        training_summary,
    )
}

pub fn default_training_summary(df: &DataFrame) -> TrainingSummaryMetadata {
    TrainingSummaryMetadata::new(df.height(), df.height(), 0)
}

pub fn tree_artifact_paths(root: &Path, model_file_name: &str) -> (PathBuf, PathBuf) {
    (root.join(model_file_name), root.join(METADATA_FILE_NAME))
}

pub fn write_runtime_metadata(path: &Path, metadata: &RuntimeArtifactMetadata) -> Result<()> {
    let payload = serde_json::to_vec_pretty(metadata).context("serialize runtime metadata")?;
    atomic_write(path, &payload)
}

pub fn read_runtime_metadata(path: &Path) -> Result<RuntimeArtifactMetadata> {
    let payload = std::fs::read(path)
        .with_context(|| format!("read runtime metadata from {}", path.display()))?;
    serde_json::from_slice(&payload)
        .with_context(|| format!("deserialize runtime metadata from {}", path.display()))
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create artifact directory {}", parent.display()))?;
    }

    let temp_path = path.with_extension("tmp");
    std::fs::write(&temp_path, bytes)
        .with_context(|| format!("write temporary artifact {}", temp_path.display()))?;
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("rename artifact into {}", path.display()))?;
    Ok(())
}

pub fn set_neutral_probability_row(row: &mut [f32]) {
    for value in row.iter_mut() {
        *value = 0.0;
    }
    if let Some(first) = row.first_mut() {
        *first = 1.0;
    }
}

fn neutral_prior_for_class(class_idx: usize) -> f32 {
    if class_idx == 0 { 0.998 } else { 0.001 }
}

pub fn build_tree_local_fallback_artifact(
    x: &DataFrame,
    y: &Series,
    training_summary: TrainingSummaryMetadata,
) -> Result<TreeLocalFallbackArtifact> {
    if x.height() != y.len() {
        bail!(
            "tree local fallback requires matching feature and label rows: {} features vs {} labels",
            x.height(),
            y.len()
        );
    }

    let (flat_x, rows, cols) = dataframe_to_row_major_vec(x)?;
    let labels = remap_labels_to_tree_targets(y)?;
    if rows == 0 || cols == 0 {
        bail!("tree local fallback requires a non-empty feature matrix");
    }

    let mut feature_means = vec![0.0_f32; cols];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            feature_means[col_idx] += flat_x[row_idx * cols + col_idx];
        }
    }
    for value in &mut feature_means {
        *value /= rows as f32;
    }

    let mut feature_scales = vec![0.0_f32; cols];
    for row_idx in 0..rows {
        for col_idx in 0..cols {
            let centered = flat_x[row_idx * cols + col_idx] - feature_means[col_idx];
            feature_scales[col_idx] += centered * centered;
        }
    }
    for value in &mut feature_scales {
        *value = (*value / rows as f32).sqrt().max(1e-3);
    }

    let sample_weights = compute_sample_weights(y)?;
    let mut class_counts = vec![0.0_f32; 3];
    let mut class_centroid_sums = vec![vec![0.0_f32; cols]; 3];
    for row_idx in 0..rows {
        let class_idx = labels[row_idx].round().clamp(0.0, 2.0) as usize;
        let weight = sample_weights.get(row_idx).copied().unwrap_or(1.0).max(0.0);
        class_counts[class_idx] += weight;
        for col_idx in 0..cols {
            class_centroid_sums[class_idx][col_idx] += flat_x[row_idx * cols + col_idx] * weight;
        }
    }

    let mut class_centroids = Vec::with_capacity(3);
    let global_centroid = feature_means.clone();
    for class_idx in 0..3 {
        if class_counts[class_idx] <= f32::EPSILON {
            class_centroids.push(global_centroid.clone());
        } else {
            class_centroids.push(
                class_centroid_sums[class_idx]
                    .iter()
                    .map(|value| *value / class_counts[class_idx])
                    .collect(),
            );
        }
    }

    let global_variances = feature_scales
        .iter()
        .map(|scale| scale * scale)
        .collect::<Vec<_>>();
    let mut class_variance_sums = vec![vec![0.0_f32; cols]; 3];
    for row_idx in 0..rows {
        let class_idx = labels[row_idx].round().clamp(0.0, 2.0) as usize;
        let weight = sample_weights.get(row_idx).copied().unwrap_or(1.0).max(0.0);
        let centroid = &class_centroids[class_idx];
        for col_idx in 0..cols {
            let diff = flat_x[row_idx * cols + col_idx] - centroid[col_idx];
            class_variance_sums[class_idx][col_idx] += diff * diff * weight;
        }
    }

    let mut class_variances = Vec::with_capacity(3);
    for class_idx in 0..3 {
        if class_counts[class_idx] <= f32::EPSILON {
            class_variances.push(global_variances.clone());
            continue;
        }

        let variances = class_variance_sums[class_idx]
            .iter()
            .enumerate()
            .map(|(col_idx, sum)| {
                let class_variance = *sum / class_counts[class_idx].max(1e-6);
                let floor = global_variances[col_idx].max(1e-6);
                let shrinkage = 0.15 * floor;
                (0.85 * class_variance + shrinkage).max(1e-6)
            })
            .collect::<Vec<_>>();
        class_variances.push(variances);
    }

    let total = class_counts.iter().copied().sum::<f32>().max(rows as f32);
    let class_priors = class_counts
        .into_iter()
        .map(|count| (count + 1.0) / (total + 3.0))
        .collect::<Vec<_>>();

    Ok(TreeLocalFallbackArtifact {
        feature_columns: feature_columns_from_dataframe(x),
        training_summary,
        class_priors,
        feature_means,
        feature_scales,
        class_centroids,
        class_variances,
    })
}

pub fn predict_tree_local_fallback(
    artifact: &TreeLocalFallbackArtifact,
    x: &DataFrame,
) -> Result<Array2<f32>> {
    ensure_feature_columns_match(&artifact.feature_columns, x)?;
    let (flat_x, rows, cols) = dataframe_to_row_major_vec(x)?;
    if rows == 0 {
        return Ok(Array2::zeros((0, 3)));
    }

    let mut probabilities = Array2::zeros((rows, 3));
    for row_idx in 0..rows {
        let mut scores = [0.0_f32; 3];
        for (class_idx, score_slot) in scores.iter_mut().enumerate() {
            let prior = artifact
                .class_priors
                .get(class_idx)
                .copied()
                .unwrap_or_else(|| neutral_prior_for_class(class_idx))
                .max(1e-6);
            let mut score = prior.ln();
            let centroid = artifact
                .class_centroids
                .get(class_idx)
                .unwrap_or(&artifact.feature_means);
            let class_variances = artifact.class_variances.get(class_idx);
            for col_idx in 0..cols {
                let variance = class_variances
                    .and_then(|values| values.get(col_idx))
                    .copied()
                    .unwrap_or_else(|| {
                        let scale = artifact
                            .feature_scales
                            .get(col_idx)
                            .copied()
                            .unwrap_or(1.0)
                            .max(1e-3);
                        scale * scale
                    })
                    .max(1e-6);
                let diff = flat_x[row_idx * cols + col_idx] - centroid[col_idx];
                score -= 0.5 * (variance.ln() + (diff * diff) / variance);
            }
            *score_slot = score;
        }

        let max_score = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
        let row = probabilities.row_mut(row_idx);
        let row = row
            .into_slice()
            .context("tree local fallback row is not contiguous")?;
        let mut sum = 0.0_f32;
        for class_idx in 0..3 {
            let value = (scores[class_idx] - max_score).exp();
            row[class_idx] = value;
            sum += value;
        }

        if sum <= f32::EPSILON {
            set_neutral_probability_row(row);
        } else {
            for value in row.iter_mut() {
                *value /= sum;
            }
        }
    }

    Ok(probabilities)
}

pub fn reshape_three_class_probabilities(
    probabilities: Vec<f32>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>> {
    if cols != 3 {
        bail!("expected 3 probability columns, got {cols}");
    }

    Array2::from_shape_vec((rows, cols), probabilities)
        .context("reshape tree-model probabilities into Array2")
}

pub fn remap_labels_to_contiguous(y: &Series) -> Result<(Array1<i32>, HashMap<i64, i32>)> {
    let y_i64 = y
        .cast(&DataType::Int64)
        .context("cast labels to Int64 for contiguous remapping")?;
    let y_ca = y_i64
        .i64()
        .context("access Int64 labels for contiguous remapping")?;

    let unique = y_ca.into_iter().flatten().collect::<BTreeSet<_>>();

    let mapping = unique
        .into_iter()
        .enumerate()
        .map(|(idx, label)| (label, idx as i32))
        .collect::<HashMap<_, _>>();

    let remapped = y_i64
        .i64()
        .context("re-open Int64 labels for contiguous remapping")?
        .into_iter()
        .map(|value| match value {
            Some(label) => mapping
                .get(&label)
                .copied()
                .context("missing contiguous class mapping"),
            None => bail!("labels may not contain nulls during contiguous remapping"),
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((Array1::from(remapped), mapping))
}

pub fn reorder_to_neutral_buy_sell(
    probabilities: Array2<f32>,
    class_order: Option<Vec<i32>>,
) -> Array2<f32> {
    match probabilities.ncols() {
        3 => {
            if let Some(order) = class_order {
                let mut reordered = Array2::zeros((probabilities.nrows(), 3));
                for (target_col, class_id) in [0_i32, 1_i32, 2_i32].into_iter().enumerate() {
                    if let Some(source_col) = order.iter().position(|value| *value == class_id) {
                        reordered
                            .column_mut(target_col)
                            .assign(&probabilities.column(source_col));
                    }
                }
                reordered
            } else {
                probabilities
            }
        }
        2 => {
            let mut reordered = Array2::zeros((probabilities.nrows(), 3));
            reordered.column_mut(0).assign(&probabilities.column(0));
            reordered.column_mut(1).assign(&probabilities.column(1));
            reordered
        }
        _ => probabilities,
    }
}

pub fn augment_time_features(mut df: DataFrame) -> Result<DataFrame> {
    let close = df
        .column("close")
        .context("augment_time_features requires a close column")?
        .cast(&DataType::Float64)
        .context("cast close column to Float64")?;
    let close_values = close
        .f64()
        .context("access Float64 close column")?
        .into_iter()
        .map(|value| value.unwrap_or(0.0))
        .collect::<Vec<_>>();

    let mut ret1 = Vec::with_capacity(close_values.len());
    ret1.push(0.0);
    for window in close_values.windows(2) {
        let previous = window[0];
        let current = window[1];
        let value = if previous.abs() > f64::EPSILON {
            (current - previous) / previous
        } else {
            0.0
        };
        ret1.push(value);
    }

    let mut ret1_lag1 = Vec::with_capacity(ret1.len());
    ret1_lag1.push(0.0);
    ret1_lag1.extend(ret1.iter().copied().take(ret1.len().saturating_sub(1)));

    df.with_column(Series::new("ret1".into(), ret1.clone()))
        .context("append ret1 feature")?;
    df.with_column(Series::new("ret1_lag1".into(), ret1_lag1))
        .context("append ret1_lag1 feature")?;

    if ret1.len() >= 14 {
        let mut vol14 = vec![0.0; ret1.len()];
        for end in 13..ret1.len() {
            let window = &ret1[end + 1 - 14..=end];
            let mean = window.iter().copied().sum::<f64>() / window.len() as f64;
            let variance = window
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f64>()
                / window.len() as f64;
            vol14[end] = variance.sqrt();
        }
        df.with_column(Series::new("vol14".into(), vol14))
            .context("append vol14 feature")?;
    }

    Ok(df)
}
