use anyhow::{Context, Result, bail};
use ndarray::Array2;
use neoethos_core::storage::json::{
    JsonBackupWriteConfig, read_json as read_json_artifact,
    write_json_with_backup as write_json_artifact_with_backup,
};
use neoethos_data::FeatureFrame;
use serde::{Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    feature_columns_from_frame, feature_frame_to_f64_array, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata, validate_model_labels,
};
use crate::runtime::artifacts::{LabelMapping, RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const XGBOOST_MODEL_FILE_NAME: &str = "model.ubj";
pub const LIGHTGBM_MODEL_FILE_NAME: &str = "model.txt";
pub const CATBOOST_MODEL_FILE_NAME: &str = "model.cbm";

pub fn canonical_tree_label_mapping() -> Vec<LabelMapping> {
    canonical_three_class_label_mapping()
}

/// Convert typed NeoEthos labels into the class ids required by native tree
/// libraries. This is an adapter-local f32 boundary: shared labels remain
/// typed integers everywhere else.
pub fn remap_labels_to_tree_targets(labels: &[i32]) -> Result<Vec<f32>> {
    validate_model_labels(labels, labels.len())?;
    labels
        .iter()
        .map(|label| match label {
            -1 => Ok(2.0_f32),
            0 => Ok(0.0_f32),
            1 => Ok(1.0_f32),
            other => bail!("unsupported tree-model label: {other}; expected one of -1, 0, 1"),
        })
        .collect()
}

/// Materialize one checked row-major f32 buffer for a native tree backend.
///
/// `FeatureFrame` remains f64+validity. Narrowing exists only inside this
/// named adapter because the native XGBoost/LightGBM/CatBoost interfaces used
/// by this crate accept f32 buffers. Values that overflow or underflow the
/// backend representation fail instead of silently becoming infinity/zero.
pub fn feature_frame_to_tree_f32_row_major(
    frame: &FeatureFrame,
) -> Result<(Vec<f32>, usize, usize)> {
    let array = feature_frame_to_f64_array(frame)?;
    let rows = array.nrows();
    let cols = array.ncols();
    let source = array
        .as_slice_memory_order()
        .context("tree model feature matrix must be contiguous in memory")?;
    let mut narrowed = Vec::with_capacity(source.len());
    for (flat_index, value) in source.iter().copied().enumerate() {
        if value.abs() > f32::MAX as f64 {
            bail!(
                "tree backend f32 adapter cannot represent feature value {value} at flat index {flat_index}"
            );
        }
        let converted = value as f32;
        if !converted.is_finite() {
            bail!("tree backend f32 adapter produced non-finite value at flat index {flat_index}");
        }
        if value != 0.0 && converted == 0.0 {
            bail!(
                "tree backend f32 adapter underflowed non-zero feature value {value} at flat index {flat_index}"
            );
        }
        narrowed.push(converted);
    }
    Ok((narrowed, rows, cols))
}

pub fn ensure_feature_columns_match(expected: &[String], frame: &FeatureFrame) -> Result<()> {
    if expected.is_empty() {
        bail!("persisted tree model is missing expected feature columns");
    }

    let actual = feature_columns_from_frame(frame);
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
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Tree,
        CapabilityState::Implemented,
        feature_columns,
        canonical_tree_label_mapping(),
        training_summary,
    )
}

pub fn default_training_summary(frame: &FeatureFrame) -> TrainingSummaryMetadata {
    TrainingSummaryMetadata::new(frame.n_samples(), frame.n_samples(), 0)
}

pub fn tree_artifact_paths(root: &Path, model_file_name: &str) -> (PathBuf, PathBuf) {
    (root.join(model_file_name), root.join(METADATA_FILE_NAME))
}

pub fn write_runtime_metadata(path: &Path, metadata: &RuntimeArtifactMetadata) -> Result<()> {
    write_tree_json_artifact(path, metadata, "tree runtime metadata")
}

pub fn read_runtime_metadata(path: &Path) -> Result<RuntimeArtifactMetadata> {
    read_json_artifact(path, "tree runtime metadata")
}

pub fn write_tree_json_artifact<T: Serialize + ?Sized>(
    path: &Path,
    value: &T,
    artifact_label: &'static str,
) -> Result<()> {
    write_json_artifact_with_backup(
        path,
        value,
        JsonBackupWriteConfig {
            artifact_label,
            temp_extension: "tmp",
            backup_extension: "bak",
        },
    )
}

pub fn read_tree_json_artifact<T: DeserializeOwned>(
    path: &Path,
    artifact_label: &str,
) -> Result<T> {
    read_json_artifact(path, artifact_label)
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

pub fn build_tree_runtime_predictions(
    model_name: &str,
    probabilities: &Array2<f64>,
    native_backend: &str,
) -> Result<Vec<RuntimePrediction>> {
    if probabilities.ncols() != 3 {
        bail!(
            "tree runtime predictions require exactly 3 probability columns, got {}",
            probabilities.ncols()
        );
    }

    let mut predictions = Vec::with_capacity(probabilities.nrows());
    for row in probabilities.outer_iter() {
        let row_values = [row[0], row[1], row[2]];
        let (confidence, should_abstain) = three_class_runtime_confidence(row_values)?;
        predictions.push(build_runtime_prediction_with_details(
            model_name,
            ModelFamily::Tree,
            CapabilityState::Implemented,
            row_values,
            Some(confidence),
            Some(should_abstain),
            Some(native_backend.to_string()),
            None,
        )?);
    }
    Ok(predictions)
}

pub fn calibrate_three_class_probabilities(
    probabilities: Array2<f64>,
    temperature: f64,
    context_name: &str,
) -> Result<Array2<f64>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        bail!("{context_name} probability temperature must be finite and positive");
    }
    let mut calibrated = normalize_three_class_probabilities(probabilities, context_name)?;
    if (temperature - 1.0).abs() < f64::EPSILON {
        return Ok(calibrated);
    }

    for row_idx in 0..calibrated.nrows() {
        let mut max_logit = f64::NEG_INFINITY;
        let mut logits = [0.0_f64; 3];
        for col_idx in 0..3 {
            let probability = calibrated[(row_idx, col_idx)];
            let logit = probability.max(f64::MIN_POSITIVE).ln() / temperature;
            logits[col_idx] = logit;
            max_logit = max_logit.max(logit);
        }

        let mut sum = 0.0_f64;
        for (col_idx, logit) in logits.into_iter().enumerate() {
            let value = (logit - max_logit).exp();
            if !value.is_finite() {
                bail!("{context_name} calibration produced a non-finite probability");
            }
            calibrated[(row_idx, col_idx)] = value;
            sum += value;
        }
        if !sum.is_finite() || sum <= f64::EPSILON {
            bail!("{context_name} calibration produced no positive probability mass");
        }
        for col_idx in 0..3 {
            calibrated[(row_idx, col_idx)] /= sum;
        }
    }
    Ok(calibrated)
}

pub fn normalize_three_class_probabilities(
    probabilities: Array2<f64>,
    context_name: &str,
) -> Result<Array2<f64>> {
    if probabilities.ncols() != 3 {
        bail!(
            "{} probability normalization requires exactly 3 columns, got {}",
            context_name,
            probabilities.ncols()
        );
    }

    let mut normalized = probabilities;
    for row_idx in 0..normalized.nrows() {
        let mut sum = 0.0_f64;
        for col_idx in 0..3 {
            let value = normalized[(row_idx, col_idx)];
            if !value.is_finite() {
                bail!(
                    "{} probability normalization encountered non-finite value at row {} col {}",
                    context_name,
                    row_idx,
                    col_idx
                );
            }
            if value < 0.0 {
                bail!(
                    "{} probability normalization encountered negative value {} at row {} col {}",
                    context_name,
                    value,
                    row_idx,
                    col_idx
                );
            }
            if value > 1.0 {
                bail!(
                    "{} probability normalization encountered value above 1.0 at row {} col {}",
                    context_name,
                    row_idx,
                    col_idx
                );
            }
            sum += value;
        }
        if !sum.is_finite() || sum <= f64::EPSILON {
            bail!(
                "{} probability row {} must contain positive mass",
                context_name,
                row_idx
            );
        }
        for col_idx in 0..3 {
            normalized[(row_idx, col_idx)] /= sum;
        }
    }
    Ok(normalized)
}

/// Widen the native f32 probability buffer immediately into the shared f64
/// model contract. No shared artifact or downstream decision retains f32.
pub fn reshape_three_class_probabilities(
    probabilities: Vec<f32>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>> {
    if cols != 3 {
        bail!("expected 3 probability columns, got {cols}");
    }
    let expected_len = rows
        .checked_mul(cols)
        .context("tree-model probability shape overflow")?;
    if probabilities.len() != expected_len {
        bail!(
            "tree-model probability length mismatch: expected {expected_len}, got {}",
            probabilities.len()
        );
    }
    let widened = probabilities.into_iter().map(f64::from).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), widened)
        .context("reshape tree-model probabilities into Array2")
}

pub fn reorder_to_neutral_buy_sell(
    probabilities: Array2<f64>,
    class_order: Option<Vec<i32>>,
) -> Result<Array2<f64>> {
    if probabilities.ncols() != 3 {
        bail!(
            "tree probability reorder requires exactly 3 columns, got {}",
            probabilities.ncols()
        );
    }
    let Some(order) = class_order else {
        return Ok(probabilities);
    };
    if order.len() != 3 {
        bail!(
            "tree class order requires exactly 3 entries, got {}",
            order.len()
        );
    }
    for class_id in [0_i32, 1_i32, 2_i32] {
        if order.iter().filter(|value| **value == class_id).count() != 1 {
            bail!("tree class order must contain each class id 0, 1, 2 exactly once");
        }
    }

    let mut reordered = Array2::zeros((probabilities.nrows(), 3));
    for (target_col, class_id) in [0_i32, 1_i32, 2_i32].into_iter().enumerate() {
        let source_col = order
            .iter()
            .position(|value| *value == class_id)
            .context("validated tree class id disappeared")?;
        reordered
            .column_mut(target_col)
            .assign(&probabilities.column(source_col));
    }
    Ok(reordered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_three_class_probabilities_rejects_non_finite_values() {
        let probabilities =
            Array2::from_shape_vec((1, 3), vec![0.2_f64, f64::NAN, 0.8]).expect("array");
        let err = normalize_three_class_probabilities(probabilities, "tree-test")
            .expect_err("non-finite row should fail");
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn normalize_three_class_probabilities_rejects_zero_mass() {
        let probabilities = Array2::zeros((1, 3));
        let err = normalize_three_class_probabilities(probabilities, "tree-test")
            .expect_err("zero probability mass must not become a fabricated neutral row");
        assert!(err.to_string().contains("positive mass"));
    }

    #[test]
    fn normalize_three_class_probabilities_rejects_negative_values() {
        let probabilities =
            Array2::from_shape_vec((1, 3), vec![0.6_f64, -0.1, 0.5]).expect("array");
        let err = normalize_three_class_probabilities(probabilities, "tree-test")
            .expect_err("negative probabilities must not be clamped into a different result");
        assert!(err.to_string().contains("negative"));
    }

    #[test]
    fn reorder_to_neutral_buy_sell_rejects_two_columns() {
        let probabilities =
            Array2::from_shape_vec((2, 2), vec![0.3_f64, 0.7, 0.4, 0.6]).expect("array");
        let err = reorder_to_neutral_buy_sell(probabilities, None)
            .expect_err("a two-column backend result cannot fabricate a neutral class");
        assert!(err.to_string().contains("exactly 3"));
    }

    #[test]
    fn reshape_three_class_probabilities_widens_native_values_to_f64() -> Result<()> {
        let native = vec![0.1_f32, 0.2_f32, 0.7_f32];
        let widened = reshape_three_class_probabilities(native.clone(), 1, 3)?;
        let _: f64 = widened[(0, 0)];
        for (actual, expected) in widened.iter().zip(native) {
            assert_eq!(actual.to_bits(), (expected as f64).to_bits());
        }
        Ok(())
    }
}
