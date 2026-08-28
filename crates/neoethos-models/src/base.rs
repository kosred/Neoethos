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
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
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
// EXPERT MODEL TRAIT
// ============================================================================

/// Abstract base trait for all expert models.
/// Derived from legacy ExpertModel class (lines 71-127)
pub trait ExpertModel {
    /// Train the model.
    /// Derived from legacy fit method (lines 74-77)
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()>;

    /// Train the model using an explicit validation frame for early stopping
    /// or `eval_set`-style monitoring (M5/M6/M7 audit fixes). The default
    /// implementation ignores the validation data and falls back to plain
    /// `fit`, which preserves the legacy contract for models that have not
    /// opted in. Implementations that genuinely support a validation set
    /// (Burn deep learners, gradient boosters, anomaly detectors) override
    /// this method so the HPO val frame flows through end-to-end and we
    /// stop relying on tail-of-train internal splits.
    fn fit_with_validation(
        &mut self,
        x: &FeatureFrame,
        y: &[i32],
        _val_x: Option<&FeatureFrame>,
        _val_y: Option<&[i32]>,
        lease: &CpuLease,
    ) -> Result<()> {
        self.fit(x, y, lease)
    }

    /// Predict probabilities for classes [-1, 0, 1].
    ///
    /// Returns:
    ///     Array2<f64>: Shape (N, 3) where columns map to [neutral, buy, sell]
    ///                  Convention: col 0 -> neutral, col 1 -> buy, col 2 -> sell
    ///
    /// Derived from legacy predict_proba method (lines 79-89)
    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>>;

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
// TYPED MODEL INPUT UTILITIES
// ============================================================================

/// Materialize the exact shared f64 values for a model that requires a dense
/// matrix. Invalid cells are structural input errors here: callers that want
/// to exclude rows must construct and record the eligible row set before
/// calling a model, never silently drop or zero-fill them in this function.
pub fn feature_frame_to_f64_array(frame: &FeatureFrame) -> Result<Array2<f64>> {
    let dense = frame
        .to_dense_samples_major()
        .context("materialize typed f64 model frame")?;
    for row in 0..dense.values.nrows() {
        for column in 0..dense.values.ncols() {
            let validity = dense.validity[(row, column)];
            if !validity.is_valid() {
                bail!(
                    "model feature `{}` row {} is ineligible: {}",
                    frame.names[column],
                    row,
                    validity.as_str()
                );
            }
            let value = dense.values[(row, column)];
            if !value.is_finite() {
                bail!(
                    "model feature `{}` row {} is marked valid with non-finite value {}",
                    frame.names[column],
                    row,
                    value
                );
            }
        }
    }
    Ok(dense.values)
}

/// Extract one named f64 feature column without losing validity information.
pub fn strict_feature_column_values(frame: &FeatureFrame, column_name: &str) -> Result<Vec<f64>> {
    let index = frame
        .names
        .iter()
        .position(|name| name == column_name)
        .with_context(|| format!("missing required feature column `{column_name}`"))?;
    let column = frame.feature_column(index)?;
    for (row, validity) in column.validity.iter().copied().enumerate() {
        if !validity.is_valid() {
            bail!(
                "model feature `{column_name}` row {row} is ineligible: {}",
                validity.as_str()
            );
        }
    }
    Ok(column.values.clone())
}

/// Extract ordered feature names from the typed shared frame.
pub fn feature_columns_from_frame(frame: &FeatureFrame) -> Vec<String> {
    frame.names.clone()
}

/// Validate the typed label boundary before any model-specific conversion.
pub fn validate_model_labels(labels: &[i32], expected_rows: usize) -> Result<()> {
    if labels.len() != expected_rows {
        bail!(
            "model label count mismatch: {} labels for {expected_rows} feature rows",
            labels.len()
        );
    }
    for (row, &label) in labels.iter().enumerate() {
        if !matches!(label, -1 | 0 | 1) {
            bail!("model label row {row} has unsupported class {label}; expected -1, 0, or 1");
        }
    }
    Ok(())
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
    class_probabilities: [f64; 3],
    confidence: Option<f64>,
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
    class_probabilities: [f64; 3],
    confidence: Option<f64>,
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

pub fn three_class_runtime_confidence(row_values: [f64; 3]) -> Result<(f64, bool)> {
    let mut normalized = row_values;
    let mut sum = 0.0_f64;
    for value in &normalized {
        if !value.is_finite() || *value < 0.0 {
            bail!("runtime predictions require finite non-negative probabilities");
        }
        if *value > 1.0 + 1e-4 {
            bail!("runtime predictions require probability rows bounded by 1.0");
        }
        sum += *value;
    }
    if !sum.is_finite() || sum <= f64::EPSILON {
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
        .sum::<f64>()
        / (3.0_f64.ln());
    let sharpness = (1.0 - entropy).clamp(0.0, 1.0);
    let confidence = (0.6 * top + 0.25 * margin + 0.15 * sharpness).clamp(0.0, 1.0);
    Ok((confidence, top < 0.5 || confidence < 0.56))
}

// ============================================================================
// CLASS WEIGHTING
// ============================================================================

/// Compute balanced class weights for imbalanced classification.
///
/// Uses inverse frequency weighting: rare classes get higher weights.
///
/// Derived from legacy compute_class_weights (lines 292-319)
pub fn compute_class_weights(labels: &[i32]) -> Result<HashMap<i32, f64>> {
    validate_model_labels(labels, labels.len())?;
    if labels.is_empty() {
        bail!("class weighting requires at least one model label");
    }

    let mut class_counts: HashMap<i32, usize> = HashMap::new();
    for &label in labels {
        *class_counts.entry(label).or_insert(0) += 1;
    }

    let n_samples = labels.len();

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
pub fn compute_sample_weights(labels: &[i32]) -> Result<Vec<f64>> {
    let class_weights = compute_class_weights(labels)?;
    labels
        .iter()
        .map(|label| {
            class_weights
                .get(label)
                .copied()
                .with_context(|| format!("missing class weight for model label {label}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

    fn sample_frame() -> FeatureFrame {
        let columns = [
            ("open", vec![1.0, 2.0]),
            ("high", vec![1.5, 2.5]),
            ("close", vec![1.25, 2.25]),
        ]
        .into_iter()
        .map(|(name, values)| {
            FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; 2])
                .expect("valid typed feature column")
        })
        .collect();
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(2),
            columns,
        )
        .expect("typed feature frame")
    }

    #[test]
    fn sample_weights_use_typed_labels_and_preserve_f64() -> Result<()> {
        let labels = [-1, 0, 0, 1, 1, 1];
        let weights: Vec<f64> = compute_sample_weights(&labels)?;

        assert_eq!(weights.len(), labels.len());
        assert_eq!(weights[0], 2.0);
        assert_eq!(weights[1], 1.0);
        assert_eq!(weights[3], 2.0 / 3.0);
        Ok(())
    }

    #[test]
    fn sample_weights_reject_unknown_model_labels() {
        let error = compute_sample_weights(&[-1, 0, 2])
            .expect_err("the shared model label contract must reject class 2");
        assert!(error.to_string().contains("unsupported class 2"));
    }

    #[test]
    fn feature_columns_from_frame_preserves_column_order() {
        let columns = feature_columns_from_frame(&sample_frame());
        assert_eq!(
            columns,
            vec!["open".to_string(), "high".to_string(), "close".to_string()]
        );
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
            TrainingSummaryMetadata::raw_for_validation(10, 8, 1),
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
            TrainingSummaryMetadata::raw_for_validation(10, 12, 12),
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
            TrainingSummaryMetadata::raw_for_validation(7, 0, 7),
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
    fn strict_feature_column_values_rejects_invalid_cells() -> Result<()> {
        let mut columns = vec![FeatureColumnF64::new(
            "close",
            vec![1.0, 2.0],
            vec![FeatureCellValidity::Valid; 2],
        )?];
        columns[0].invalidate(1, FeatureCellValidity::Gap)?;
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(2),
            columns,
        )?;

        let err = strict_feature_column_values(&frame, "close")
            .expect_err("invalid typed feature cells must fail closed");
        assert!(err.to_string().contains("row 1") && err.to_string().contains("gap"));
        Ok(())
    }
}
