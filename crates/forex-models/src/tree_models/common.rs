use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction_with_details,
    canonical_three_class_label_mapping, compute_sample_weights, dataframe_to_float32_array,
    feature_columns_from_dataframe, three_class_runtime_confidence,
};
use crate::runtime::artifacts::{LabelMapping, RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const XGBOOST_MODEL_FILE_NAME: &str = "model.bin";
pub const LIGHTGBM_MODEL_FILE_NAME: &str = "model.txt";
pub const CATBOOST_MODEL_FILE_NAME: &str = "model.cbm";
pub const TREE_LOCAL_SURROGATE_KIND: &str = "gaussian_centroid_surrogate";

fn default_tree_surrogate_kind() -> String {
    TREE_LOCAL_SURROGATE_KIND.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeLocalFallbackArtifact {
    #[serde(default = "default_tree_surrogate_kind")]
    pub surrogate_kind: String,
    pub feature_columns: Vec<String>,
    pub training_summary: TrainingSummaryMetadata,
    pub class_priors: Vec<f32>,
    #[serde(default)]
    pub class_support: Vec<f32>,
    pub feature_means: Vec<f32>,
    pub feature_scales: Vec<f32>,
    #[serde(default)]
    pub feature_salience: Vec<f32>,
    pub class_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    pub class_variances: Vec<Vec<f32>>,
    #[serde(default)]
    pub distance_location: Vec<f32>,
    #[serde(default)]
    pub distance_scale: Vec<f32>,
}

pub fn validate_tree_local_fallback_artifact(
    artifact: &TreeLocalFallbackArtifact,
    expected_feature_columns: &[String],
) -> Result<()> {
    if expected_feature_columns.is_empty() {
        bail!("tree local fallback is missing expected feature columns");
    }
    if artifact.feature_columns.is_empty() {
        bail!("tree local fallback artifact is missing feature columns");
    }
    if artifact.feature_columns != expected_feature_columns {
        bail!(
            "tree local fallback feature mismatch: expected {:?}, got {:?}",
            expected_feature_columns,
            artifact.feature_columns
        );
    }
    if artifact.surrogate_kind != TREE_LOCAL_SURROGATE_KIND {
        bail!(
            "tree local fallback surrogate kind mismatch: expected `{}`, got `{}`",
            TREE_LOCAL_SURROGATE_KIND,
            artifact.surrogate_kind
        );
    }
    if artifact.training_summary.dataset_rows
        != artifact.training_summary.train_rows + artifact.training_summary.val_rows
    {
        bail!("tree local fallback artifact contains an inconsistent training summary");
    }
    if artifact.training_summary.dataset_rows == 0 {
        bail!("tree local fallback artifact requires at least one training row");
    }

    if artifact.class_priors.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 class priors, got {}",
            artifact.class_priors.len()
        );
    }
    if artifact.class_centroids.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 class centroids, got {}",
            artifact.class_centroids.len()
        );
    }
    if artifact.class_variances.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 class variance rows, got {}",
            artifact.class_variances.len()
        );
    }
    if !artifact.class_support.is_empty() && artifact.class_support.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 class support values, got {}",
            artifact.class_support.len()
        );
    }
    if !artifact.distance_location.is_empty() && artifact.distance_location.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 distance location values, got {}",
            artifact.distance_location.len()
        );
    }
    if !artifact.distance_scale.is_empty() && artifact.distance_scale.len() != 3 {
        bail!(
            "tree local fallback artifact expected 3 distance scale values, got {}",
            artifact.distance_scale.len()
        );
    }

    let feature_count = expected_feature_columns.len();
    if artifact.feature_means.len() != feature_count {
        bail!(
            "tree local fallback feature means mismatch: expected {}, got {}",
            feature_count,
            artifact.feature_means.len()
        );
    }
    if artifact.feature_scales.len() != feature_count {
        bail!(
            "tree local fallback feature scales mismatch: expected {}, got {}",
            feature_count,
            artifact.feature_scales.len()
        );
    }
    if !artifact.feature_salience.is_empty() && artifact.feature_salience.len() != feature_count {
        bail!(
            "tree local fallback feature salience mismatch: expected {}, got {}",
            feature_count,
            artifact.feature_salience.len()
        );
    }

    let priors_sum = artifact
        .class_priors
        .iter()
        .try_fold(0.0_f32, |acc, value| {
            if !value.is_finite() || *value < 0.0 {
                bail!("tree local fallback class priors must be finite and non-negative");
            }
            Ok(acc + *value)
        })?;
    if priors_sum <= f32::EPSILON {
        bail!("tree local fallback class priors must have positive mass");
    }
    if (priors_sum - 1.0).abs() > 1e-3 {
        bail!(
            "tree local fallback class priors must sum to 1.0 within tolerance, got {}",
            priors_sum
        );
    }

    for value in artifact
        .feature_means
        .iter()
        .chain(artifact.feature_scales.iter())
        .chain(artifact.feature_salience.iter())
        .chain(artifact.class_support.iter())
        .chain(artifact.distance_location.iter())
        .chain(artifact.distance_scale.iter())
    {
        if !value.is_finite() {
            bail!("tree local fallback feature statistics must be finite");
        }
    }
    if artifact.class_support.iter().any(|value| *value < 0.0) {
        bail!("tree local fallback class support must be non-negative");
    }
    if artifact.distance_scale.iter().any(|value| *value <= 0.0) {
        bail!("tree local fallback distance scale must be positive");
    }
    for centroid in &artifact.class_centroids {
        if centroid.len() != feature_count {
            bail!(
                "tree local fallback centroid width mismatch: expected {}, got {}",
                feature_count,
                centroid.len()
            );
        }
        if centroid.iter().any(|value| !value.is_finite()) {
            bail!("tree local fallback centroids must be finite");
        }
    }
    for variances in &artifact.class_variances {
        if variances.len() != feature_count {
            bail!(
                "tree local fallback variance width mismatch: expected {}, got {}",
                feature_count,
                variances.len()
            );
        }
        if variances
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("tree local fallback variances must be finite and positive");
        }
    }

    Ok(())
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
    if path.exists() {
        let backup_path = path.with_extension("bak");
        if backup_path.exists() {
            std::fs::remove_file(&backup_path)
                .with_context(|| format!("remove stale backup {}", backup_path.display()))?;
        }
        std::fs::rename(path, &backup_path)
            .with_context(|| format!("move existing artifact to {}", backup_path.display()))?;
        if let Err(rename_err) = std::fs::rename(&temp_path, path) {
            let _ = std::fs::rename(&backup_path, path);
            return Err(rename_err)
                .with_context(|| format!("rename artifact into {}", path.display()));
        }
        if backup_path.exists() {
            std::fs::remove_file(&backup_path)
                .with_context(|| format!("remove backup artifact {}", backup_path.display()))?;
        }
    } else {
        std::fs::rename(&temp_path, path)
            .with_context(|| format!("rename artifact into {}", path.display()))?;
    }
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

fn fallback_anchor_distribution(artifact: &TreeLocalFallbackArtifact) -> [f32; 3] {
    let mut support_distribution = [1.0_f32 / 3.0; 3];
    let support_sum = artifact.class_support.iter().copied().sum::<f32>();
    if support_sum > f32::EPSILON && artifact.class_support.len() == 3 {
        for (class_idx, slot) in support_distribution.iter_mut().enumerate() {
            *slot = artifact.class_support[class_idx].max(0.0) / support_sum;
        }
    }

    let mut anchor = [0.0_f32; 3];
    let mut sum = 0.0_f32;
    for (class_idx, slot) in anchor.iter_mut().enumerate() {
        let prior = artifact
            .class_priors
            .get(class_idx)
            .copied()
            .unwrap_or(1.0_f32 / 3.0)
            .max(1e-6);
        let support = support_distribution[class_idx].max(1e-6);
        *slot = 0.7 * prior + 0.2 * support + 0.1 * (1.0 / 3.0);
        sum += *slot;
    }
    if sum <= f32::EPSILON {
        [1.0, 0.0, 0.0]
    } else {
        for value in &mut anchor {
            *value /= sum;
        }
        anchor
    }
}

fn probability_row_confidence(row_values: [f32; 3]) -> Result<(f32, bool)> {
    three_class_runtime_confidence(row_values)
}

pub fn tree_runtime_backend_details(
    native_loaded: bool,
    native_backend: &str,
    fallback: Option<&TreeLocalFallbackArtifact>,
    fallback_reason: &str,
    unknown_backend: &str,
) -> (Option<String>, Option<String>) {
    if native_loaded {
        (Some(native_backend.to_string()), None)
    } else if let Some(fallback) = fallback {
        (
            Some(format!("tree_surrogate:{}", fallback.surrogate_kind)),
            Some(fallback_reason.to_string()),
        )
    } else {
        (
            Some(unknown_backend.to_string()),
            Some("tree_runtime_backend_unavailable".to_string()),
        )
    }
}

pub fn build_tree_runtime_predictions(
    model_name: &str,
    probabilities: &Array2<f32>,
    native_loaded: bool,
    native_backend: &str,
    fallback: Option<&TreeLocalFallbackArtifact>,
    fallback_reason: &str,
    unknown_backend: &str,
) -> Result<Vec<RuntimePrediction>> {
    if probabilities.ncols() != 3 {
        bail!(
            "tree runtime predictions require exactly 3 probability columns, got {}",
            probabilities.ncols()
        );
    }

    let (execution_backend, degraded_reason) = tree_runtime_backend_details(
        native_loaded,
        native_backend,
        fallback,
        fallback_reason,
        unknown_backend,
    );

    let mut predictions = Vec::with_capacity(probabilities.nrows());
    for row in probabilities.outer_iter() {
        let row_values = [row[0], row[1], row[2]];
        let (confidence, should_abstain) = probability_row_confidence(row_values)?;
        predictions.push(build_runtime_prediction_with_details(
            model_name,
            ModelFamily::Tree,
            CapabilityState::Implemented,
            row_values,
            Some(confidence),
            Some(should_abstain),
            execution_backend.clone(),
            degraded_reason.clone(),
        )?);
    }

    Ok(predictions)
}

pub fn calibrate_three_class_probabilities(
    probabilities: Array2<f32>,
    temperature: f32,
    context_name: &str,
) -> Result<Array2<f32>> {
    if probabilities.ncols() != 3 {
        bail!(
            "{} probability calibration requires exactly 3 columns, got {}",
            context_name,
            probabilities.ncols()
        );
    }
    if !temperature.is_finite() || (temperature - 1.0).abs() < f32::EPSILON {
        return Ok(probabilities);
    }

    let mut calibrated = probabilities;
    let temperature = temperature.max(1e-6);
    for row_idx in 0..calibrated.nrows() {
        let mut max_logit = f32::NEG_INFINITY;
        let mut logits = [0.0_f32; 3];
        for col_idx in 0..calibrated.ncols() {
            let probability = calibrated[(row_idx, col_idx)];
            if !probability.is_finite() {
                bail!("{context_name} calibrated a non-finite probability");
            }
            let probability = probability.clamp(1e-12, 1.0);
            let logit = probability.ln() / temperature;
            logits[col_idx] = logit;
            max_logit = max_logit.max(logit);
        }

        let mut sum = 0.0_f32;
        for (col_idx, logit) in logits.into_iter().enumerate() {
            let value = (logit - max_logit).exp();
            calibrated[(row_idx, col_idx)] = value;
            sum += value;
        }

        if sum > f32::EPSILON {
            for col_idx in 0..calibrated.ncols() {
                calibrated[(row_idx, col_idx)] /= sum;
            }
        }
    }

    Ok(calibrated)
}

pub fn normalize_three_class_probabilities(
    probabilities: Array2<f32>,
    context_name: &str,
) -> Result<Array2<f32>> {
    if probabilities.ncols() != 3 {
        bail!(
            "{} probability normalization requires exactly 3 columns, got {}",
            context_name,
            probabilities.ncols()
        );
    }

    let mut normalized = probabilities;
    for row_idx in 0..normalized.nrows() {
        let mut sum = 0.0_f32;
        for col_idx in 0..normalized.ncols() {
            let value = normalized[(row_idx, col_idx)];
            if !value.is_finite() {
                bail!(
                    "{} probability normalization encountered non-finite value at row {} col {}",
                    context_name,
                    row_idx,
                    col_idx
                );
            }
            let clamped = value.max(0.0);
            normalized[(row_idx, col_idx)] = clamped;
            sum += clamped;
        }

        if sum <= f32::EPSILON {
            if let Some(row) = normalized.row_mut(row_idx).into_slice() {
                set_neutral_probability_row(row);
            } else {
                bail!("{context_name} probability row is not contiguous");
            }
            continue;
        }

        for col_idx in 0..normalized.ncols() {
            normalized[(row_idx, col_idx)] /= sum;
        }
    }

    Ok(normalized)
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
    if flat_x.iter().any(|value| !value.is_finite()) {
        bail!("tree local fallback requires finite feature values");
    }

    let sample_weights = compute_sample_weights(y)?;
    let total_weight = sample_weights
        .iter()
        .copied()
        .map(|weight| weight.max(0.0))
        .sum::<f32>()
        .max(rows as f32);

    let mut feature_means = vec![0.0_f32; cols];
    for row_idx in 0..rows {
        let row_weight = sample_weights.get(row_idx).copied().unwrap_or(1.0).max(0.0);
        for col_idx in 0..cols {
            feature_means[col_idx] += flat_x[row_idx * cols + col_idx] * row_weight;
        }
    }
    for value in &mut feature_means {
        *value /= total_weight;
    }

    let mut feature_scales = vec![0.0_f32; cols];
    for row_idx in 0..rows {
        let row_weight = sample_weights.get(row_idx).copied().unwrap_or(1.0).max(0.0);
        for col_idx in 0..cols {
            let centered = flat_x[row_idx * cols + col_idx] - feature_means[col_idx];
            feature_scales[col_idx] += centered * centered * row_weight;
        }
    }
    for value in &mut feature_scales {
        *value = (*value / total_weight).sqrt().max(1e-3);
    }

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

    let mut feature_salience = vec![1.0_f32; cols];
    for col_idx in 0..cols {
        let centroid_values = class_centroids
            .iter()
            .map(|centroid| centroid[col_idx])
            .collect::<Vec<_>>();
        let centroid_mean =
            centroid_values.iter().copied().sum::<f32>() / centroid_values.len().max(1) as f32;
        let between_class = centroid_values
            .iter()
            .map(|value| {
                let centered = *value - centroid_mean;
                centered * centered
            })
            .sum::<f32>()
            / centroid_values.len().max(1) as f32;
        let within_class = class_variances
            .iter()
            .map(|variance| variance[col_idx])
            .sum::<f32>()
            / class_variances.len().max(1) as f32;
        let normalized_signal = (between_class.sqrt() / within_class.sqrt().max(1e-6)).max(0.0);
        feature_salience[col_idx] = (0.35 + normalized_signal).clamp(0.35, 3.0);
    }

    let mut distance_weight_sums = [0.0_f32; 3];
    let mut distance_sums = [0.0_f32; 3];
    let mut distance_sq_sums = [0.0_f32; 3];
    for row_idx in 0..rows {
        let class_idx = labels[row_idx].round().clamp(0.0, 2.0) as usize;
        let weight = sample_weights.get(row_idx).copied().unwrap_or(1.0).max(0.0);
        let centroid = &class_centroids[class_idx];
        let mut normalized_distance = 0.0_f32;
        for col_idx in 0..cols {
            let salience = feature_salience
                .get(col_idx)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.1, 4.0);
            let variance = class_variances[class_idx][col_idx].max(1e-6);
            let diff = flat_x[row_idx * cols + col_idx] - centroid[col_idx];
            normalized_distance += salience * (diff * diff) / variance;
        }
        normalized_distance /= cols.max(1) as f32;
        distance_weight_sums[class_idx] += weight;
        distance_sums[class_idx] += normalized_distance * weight;
        distance_sq_sums[class_idx] += normalized_distance * normalized_distance * weight;
    }

    let total = class_counts.iter().copied().sum::<f32>().max(rows as f32);
    let class_priors = class_counts
        .iter()
        .copied()
        .map(|count| (count + 1.0) / (total + 3.0))
        .collect::<Vec<_>>();
    let class_support = class_counts.clone();

    let global_distance_weight = distance_weight_sums.iter().copied().sum::<f32>().max(1e-6);
    let global_distance_location =
        distance_sums.iter().copied().sum::<f32>() / global_distance_weight;
    let global_distance_variance = (distance_sq_sums.iter().copied().sum::<f32>()
        / global_distance_weight
        - global_distance_location * global_distance_location)
        .max(0.0);
    let global_distance_scale = global_distance_variance.sqrt().max(0.35);

    let mut distance_location = Vec::with_capacity(3);
    let mut distance_scale = Vec::with_capacity(3);
    for class_idx in 0..3 {
        if distance_weight_sums[class_idx] <= f32::EPSILON {
            distance_location.push(global_distance_location.max(1e-3));
            distance_scale.push(global_distance_scale);
            continue;
        }
        let location = distance_sums[class_idx] / distance_weight_sums[class_idx];
        let variance = (distance_sq_sums[class_idx] / distance_weight_sums[class_idx]
            - location * location)
            .max(0.0);
        distance_location.push(location.max(1e-3));
        distance_scale.push((variance.sqrt() + 0.15 * global_distance_scale).max(0.25));
    }

    let feature_columns = feature_columns_from_dataframe(x);
    let artifact = TreeLocalFallbackArtifact {
        surrogate_kind: default_tree_surrogate_kind(),
        feature_columns,
        training_summary,
        class_priors,
        class_support,
        feature_means,
        feature_scales,
        feature_salience,
        class_centroids,
        class_variances,
        distance_location,
        distance_scale,
    };
    validate_tree_local_fallback_artifact(&artifact, &artifact.feature_columns)?;
    Ok(artifact)
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
    if flat_x.iter().any(|value| !value.is_finite()) {
        bail!("tree local fallback prediction requires finite feature values");
    }

    let mut probabilities = Array2::zeros((rows, 3));
    let anchor_distribution = fallback_anchor_distribution(artifact);
    for row_idx in 0..rows {
        let mut scores = [0.0_f32; 3];
        let mut normalized_distances = [0.0_f32; 3];
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
                let salience = artifact
                    .feature_salience
                    .get(col_idx)
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(0.1, 4.0);
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
                score -= 0.5 * salience * (variance.ln() + (diff * diff) / variance);
                normalized_distances[class_idx] += salience * (diff * diff) / variance;
            }
            if !score.is_finite() {
                bail!("tree local fallback produced a non-finite score");
            }
            normalized_distances[class_idx] /= cols.max(1) as f32;
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
        let mut best_class_idx = 0_usize;
        let mut best_probability = f32::NEG_INFINITY;
        for class_idx in 0..3 {
            let value = (scores[class_idx] - max_score).exp();
            row[class_idx] = value;
            sum += value;
            if value > best_probability {
                best_probability = value;
                best_class_idx = class_idx;
            }
        }

        if sum <= f32::EPSILON {
            set_neutral_probability_row(row);
        } else {
            for value in row.iter_mut() {
                *value /= sum;
            }

            let support = artifact
                .class_support
                .get(best_class_idx)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let support_ratio = (support
                / artifact
                    .training_summary
                    .train_rows
                    .max(artifact.training_summary.dataset_rows)
                    .max(1) as f32)
                .clamp(0.0, 1.0);
            let distance_location = artifact
                .distance_location
                .get(best_class_idx)
                .copied()
                .unwrap_or(1.0)
                .max(1e-3);
            let distance_scale = artifact
                .distance_scale
                .get(best_class_idx)
                .copied()
                .unwrap_or(1.0)
                .max(1e-3);
            let standardized_distance =
                ((normalized_distances[best_class_idx] - distance_location) / distance_scale)
                    .max(0.0);
            let distance_reliability = 1.0 / (1.0 + ((standardized_distance - 0.75) / 0.9).exp());
            let support_reliability = (0.45 + 0.55 * support_ratio.sqrt()).clamp(0.45, 1.0);
            let reliability = (distance_reliability * support_reliability).clamp(0.05, 1.0);

            if reliability <= 0.08 {
                set_neutral_probability_row(row);
            } else {
                let mut blended_sum = 0.0_f32;
                for class_idx in 0..3 {
                    row[class_idx] = reliability * row[class_idx]
                        + (1.0 - reliability) * anchor_distribution[class_idx];
                    blended_sum += row[class_idx];
                }
                if blended_sum <= f32::EPSILON {
                    set_neutral_probability_row(row);
                } else {
                    for value in row.iter_mut() {
                        *value /= blended_sum;
                    }
                }
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
            for row_idx in 0..probabilities.nrows() {
                reordered[(row_idx, 0)] = 1.0;
            }
            reordered
        }
        _ => probabilities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fallback_artifact() -> TreeLocalFallbackArtifact {
        TreeLocalFallbackArtifact {
            surrogate_kind: TREE_LOCAL_SURROGATE_KIND.to_string(),
            feature_columns: vec!["momentum".into(), "trend".into()],
            training_summary: TrainingSummaryMetadata::new(12, 12, 0),
            class_priors: vec![0.2, 0.5, 0.3],
            class_support: vec![2.0, 5.0, 3.0],
            feature_means: vec![0.0, 0.0],
            feature_scales: vec![1.0, 1.0],
            feature_salience: vec![1.0, 1.0],
            class_centroids: vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![-1.0, -1.0]],
            class_variances: vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
            distance_location: vec![1.0, 1.0, 1.0],
            distance_scale: vec![1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn validate_tree_local_fallback_artifact_rejects_unnormalized_class_priors() {
        let mut artifact = sample_fallback_artifact();
        artifact.class_priors = vec![0.2, 0.5, 0.5];

        let err =
            validate_tree_local_fallback_artifact(&artifact, &["momentum".into(), "trend".into()])
                .expect_err("unnormalized priors should fail");
        assert!(err.to_string().contains("sum to 1.0"));
    }

    #[test]
    fn normalize_three_class_probabilities_rejects_non_finite_values() {
        let probabilities =
            Array2::from_shape_vec((1, 3), vec![0.2, f32::NAN, 0.8]).expect("array");
        let err = normalize_three_class_probabilities(probabilities, "tree-test")
            .expect_err("non-finite row should fail");
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn tree_runtime_backend_details_marks_missing_backend_as_degraded() {
        let (backend, degraded_reason) = tree_runtime_backend_details(
            false,
            "lightgbm_native",
            None,
            "native_lightgbm_unavailable",
            "lightgbm_unknown",
        );
        assert_eq!(backend.as_deref(), Some("lightgbm_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("tree_runtime_backend_unavailable")
        );
    }

    #[test]
    fn reorder_to_neutral_buy_sell_two_columns_returns_neutral_rows() {
        let probabilities =
            Array2::from_shape_vec((2, 2), vec![0.3, 0.7, 0.4, 0.6]).expect("array");
        let reordered = reorder_to_neutral_buy_sell(probabilities, None);
        assert_eq!(reordered.ncols(), 3);
        assert_eq!(reordered[(0, 0)], 1.0);
        assert_eq!(reordered[(0, 1)], 0.0);
        assert_eq!(reordered[(0, 2)], 0.0);
        assert_eq!(reordered[(1, 0)], 1.0);
        assert_eq!(reordered[(1, 1)], 0.0);
        assert_eq!(reordered[(1, 2)], 0.0);
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
        .enumerate()
        .map(|(idx, value)| {
            let value = value.with_context(|| {
                format!(
                    "augment_time_features close column contains null at row {}; engineered tree features require fully materialized prices",
                    idx
                )
            })?;
            if !value.is_finite() {
                bail!(
                    "augment_time_features close column contains non-finite value {} at row {}",
                    value,
                    idx
                );
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;

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
