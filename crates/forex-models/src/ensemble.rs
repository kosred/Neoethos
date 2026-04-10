use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_prediction_with_details, three_class_runtime_confidence, ExpertModel,
};
use crate::runtime::artifacts::{default_three_class_label_mapping, RuntimeArtifactMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    ensure_feature_columns_match, meta_runtime_metadata, read_json, write_json, METADATA_FILE_NAME,
};
use crate::tree_models::XGBoostExpert;

const META_BLENDER_FILE_NAME: &str = "meta_blender.json";
const CALIBRATOR_FILE_NAME: &str = "probability_calibrator.json";
const CONFORMAL_FILE_NAME: &str = "conformal_gate.json";
const META_STACK_FILE_NAME: &str = "meta_stack.json";
const CALIBRATION_EXPERT_FILE_NAME: &str = "probability_calibration_expert.json";
const CONFORMAL_EXPERT_FILE_NAME: &str = "conformal_prediction_expert.json";
const BLENDER_DIR_NAME: &str = "blender_model";
const BLENDER_BACKEND_DIR_NAME: &str = "xgboost_backend";
const CALIBRATION_BACKEND_DIR_NAME: &str = "calibration_backend";
const CONFORMAL_BACKEND_DIR_NAME: &str = "conformal_backend";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    Identity,
    Platt,
    Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationModel {
    Constant(f32),
    Platt { a: f32, b: f32 },
    Temperature { temperature: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaBlenderArtifact {
    feature_columns: Vec<String>,
    fitted: bool,
    #[serde(default)]
    training_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProbabilityCalibratorArtifact {
    method: CalibrationMethod,
    fitted: bool,
    models: Vec<CalibrationModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConformalGateArtifact {
    alpha: f32,
    qhat: f32,
    fitted: bool,
    n_calib: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaDecisionStackArtifact {
    fitted: bool,
    feature_columns: Vec<String>,
    #[serde(default)]
    training_rows: usize,
    #[serde(default = "default_calibration_method")]
    method: CalibrationMethod,
    #[serde(default = "default_conformal_alpha")]
    alpha: f32,
    min_prediction_set: usize,
    min_fit_rows: usize,
}

fn default_calibration_method() -> CalibrationMethod {
    CalibrationMethod::Platt
}

fn default_conformal_alpha() -> f32 {
    0.10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProbabilityCalibrationExpertArtifact {
    fitted: bool,
    feature_columns: Vec<String>,
    #[serde(default)]
    training_rows: usize,
    method: CalibrationMethod,
    min_fit_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConformalPredictionExpertArtifact {
    fitted: bool,
    feature_columns: Vec<String>,
    #[serde(default)]
    training_rows: usize,
    alpha: f32,
    method: CalibrationMethod,
    min_prediction_set: usize,
    min_fit_rows: usize,
}

fn series_labels(y: &Series) -> Result<Vec<i32>> {
    let labels = y
        .cast(&DataType::Int32)
        .context("cast meta labels to Int32")?;
    labels
        .i32()
        .context("access meta labels as Int32")?
        .into_iter()
        .map(|value| value.context("meta labels may not contain nulls"))
        .collect()
}

fn label_to_class_index(label: i32) -> Result<usize> {
    match label {
        -1 => Ok(2),
        0 => Ok(0),
        1 => Ok(1),
        other => bail!("unsupported meta label {other}; expected one of -1, 0, 1"),
    }
}

fn clamp_probability(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(1e-6, 1.0 - 1e-6)
    } else {
        0.5
    }
}

fn set_neutral_probability_row(probabilities: &mut Array2<f32>, row_idx: usize) {
    for col_idx in 0..probabilities.ncols() {
        probabilities[(row_idx, col_idx)] = 0.0;
    }
    if probabilities.ncols() > 0 {
        probabilities[(row_idx, 0)] = 1.0;
    }
}

fn renormalize_rows(probabilities: &Array2<f32>) -> Array2<f32> {
    let mut normalized = probabilities.clone();
    for row_idx in 0..normalized.nrows() {
        let mut sum = 0.0_f32;
        for col_idx in 0..normalized.ncols() {
            let value = normalized[(row_idx, col_idx)];
            let clamped = if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            };
            normalized[(row_idx, col_idx)] = clamped;
            sum += clamped;
        }

        if sum <= f32::EPSILON {
            set_neutral_probability_row(&mut normalized, row_idx);
            continue;
        }

        for col_idx in 0..normalized.ncols() {
            normalized[(row_idx, col_idx)] /= sum;
        }
    }
    normalized
}

fn logit(probability: f32) -> f32 {
    let p = clamp_probability(probability);
    (p / (1.0 - p)).ln()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn validate_meta_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_model_name: &str,
) -> Result<()> {
    if metadata.model_name != expected_model_name {
        bail!(
            "meta artifact model mismatch: expected {}, got {}",
            expected_model_name,
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Meta {
        bail!(
            "meta artifact family mismatch: expected {:?}, got {:?}",
            ModelFamily::Meta,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "meta artifact state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != default_three_class_label_mapping() {
        bail!("meta artifact label mapping mismatch");
    }
    if metadata.feature_columns.is_empty() {
        bail!("meta artifact metadata must contain at least one feature column");
    }
    if metadata.training_summary.dataset_rows == 0 {
        bail!("meta artifact training summary must persist a non-zero dataset row count");
    }
    if metadata.training_summary.dataset_rows
        != metadata.training_summary.train_rows + metadata.training_summary.val_rows
    {
        bail!("meta artifact training summary is inconsistent");
    }
    Ok(())
}

fn staged_meta_artifact_dir(path: &Path) -> PathBuf {
    path.with_extension("tmp_meta_artifact")
}

fn backup_meta_artifact_dir(path: &Path) -> PathBuf {
    path.with_extension("bak_meta_artifact")
}

fn staged_meta_file(path: &Path) -> PathBuf {
    path.with_extension("tmp_meta_file")
}

fn backup_meta_file(path: &Path) -> PathBuf {
    path.with_extension("bak_meta_file")
}

fn cleanup_meta_artifact_dir(path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("remove staged meta artifact {}", path.display()))?;
    }
    Ok(())
}

fn replace_meta_artifact_dir(staged_path: &Path, target_path: &Path) -> Result<()> {
    let backup_path = backup_meta_artifact_dir(target_path);
    cleanup_meta_artifact_dir(&backup_path)?;
    if target_path.exists() {
        std::fs::rename(target_path, &backup_path).with_context(|| {
            format!(
                "move previous meta artifact into backup {}",
                backup_path.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(staged_path, target_path) {
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, target_path);
        }
        bail!(
            "rename staged meta artifact into {} failed: {}",
            target_path.display(),
            error
        );
    }
    cleanup_meta_artifact_dir(&backup_path)?;
    Ok(())
}

fn with_staged_meta_artifact_dir<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    let staged_path = staged_meta_artifact_dir(path);
    cleanup_meta_artifact_dir(&staged_path)?;
    std::fs::create_dir_all(&staged_path)
        .with_context(|| format!("create staged meta artifact dir {}", staged_path.display()))?;
    if let Err(error) = writer(&staged_path) {
        let _ = cleanup_meta_artifact_dir(&staged_path);
        return Err(error);
    }
    replace_meta_artifact_dir(&staged_path, path)
}

fn write_json_with_backup<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create artifact directory {}", parent.display()))?;
    }

    let staged_path = staged_meta_file(path);
    let backup_path = backup_meta_file(path);
    if staged_path.exists() {
        std::fs::remove_file(&staged_path)
            .with_context(|| format!("remove stale staged meta file {}", staged_path.display()))?;
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path)
            .with_context(|| format!("remove stale backup meta file {}", backup_path.display()))?;
    }

    let payload = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", path.display()))?;
    std::fs::write(&staged_path, payload)
        .with_context(|| format!("write staged meta file {}", staged_path.display()))?;

    if path.exists() {
        std::fs::rename(path, &backup_path)
            .with_context(|| format!("backup current meta file {}", path.display()))?;
    }

    if let Err(error) = std::fs::rename(&staged_path, path) {
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, path);
        } else if staged_path.exists() {
            let _ = std::fs::remove_file(&staged_path);
        }
        bail!(
            "promote staged meta file into {} failed: {}",
            path.display(),
            error
        );
    }

    if backup_path.exists() {
        std::fs::remove_file(&backup_path)
            .with_context(|| format!("remove backup meta file {}", backup_path.display()))?;
    }

    Ok(())
}

fn join_degraded_reasons<I>(reasons: I) -> Option<String>
where
    I: IntoIterator<Item = String>,
{
    let reasons = reasons
        .into_iter()
        .filter(|reason| !reason.trim().is_empty())
        .collect::<Vec<_>>();
    if reasons.is_empty() {
        None
    } else {
        Some(reasons.join("; "))
    }
}

fn validate_calibrator_artifact(artifact: &ProbabilityCalibratorArtifact) -> Result<()> {
    match artifact.method {
        CalibrationMethod::Identity => {
            if !artifact.models.is_empty() {
                bail!("identity calibrator should not persist trained models");
            }
        }
        CalibrationMethod::Temperature => {
            if artifact.models.len() != 1 {
                bail!(
                    "temperature calibrator must persist exactly one model, found {}",
                    artifact.models.len()
                );
            }
            match artifact.models.first() {
                Some(CalibrationModel::Temperature { temperature })
                    if temperature.is_finite() && *temperature > 0.0 => {}
                Some(_) => bail!("temperature calibrator stored a non-temperature model"),
                None => bail!("temperature calibrator model payload missing"),
            }
        }
        CalibrationMethod::Platt => {
            if artifact.models.len() != 3 {
                bail!(
                    "platt calibrator must persist exactly three binary models, found {}",
                    artifact.models.len()
                );
            }
            for model in &artifact.models {
                match model {
                    CalibrationModel::Constant(probability) => {
                        if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                            bail!("platt calibrator stored invalid constant probability");
                        }
                    }
                    CalibrationModel::Platt { a, b } => {
                        if !a.is_finite() || !b.is_finite() {
                            bail!("platt calibrator stored non-finite coefficients");
                        }
                    }
                    CalibrationModel::Temperature { .. } => {
                        bail!("platt calibrator stored a temperature model")
                    }
                }
            }
        }
    }

    if !artifact.fitted && !artifact.models.is_empty() {
        bail!("unfitted calibrator should not persist trained models");
    }

    Ok(())
}

fn validate_meta_blender_save_state(state: &MetaBlender) -> Result<()> {
    let model = state.model.as_ref().context("MetaBlender not fitted")?;
    if !state.fitted {
        bail!("MetaBlender must be marked as fitted before save");
    }
    if state.feature_columns.is_empty() {
        bail!("MetaBlender must persist at least one feature column before save");
    }
    if state.training_rows == 0 {
        bail!("MetaBlender must persist a non-zero training row count before save");
    }
    if model.feature_columns.is_empty() {
        bail!("MetaBlender backend is missing feature columns");
    }
    if model.feature_columns != state.feature_columns {
        bail!(
            "MetaBlender backend feature-column mismatch between state {:?} and backend {:?}",
            state.feature_columns,
            model.feature_columns
        );
    }
    Ok(())
}

fn validate_probability_calibrator_live_state(state: &ProbabilityCalibrator) -> Result<()> {
    if !state.fitted {
        bail!("probability calibrator is not fitted");
    }
    validate_calibrator_artifact(&ProbabilityCalibratorArtifact {
        method: state.method,
        fitted: state.fitted,
        models: state.models.clone(),
    })
}

fn validate_conformal_gate_live_state(state: &ConformalGate) -> Result<()> {
    if !state.fitted {
        bail!("conformal gate is not fitted");
    }
    validate_conformal_artifact(&ConformalGateArtifact {
        alpha: state.alpha,
        qhat: state.qhat,
        fitted: state.fitted,
        n_calib: state.n_calib,
    })
}

fn validate_probability_calibration_expert_artifact(
    artifact: &ProbabilityCalibrationExpertArtifact,
) -> Result<()> {
    if artifact.feature_columns.is_empty() {
        bail!("probability calibration artifact must contain at least one feature column");
    }
    if artifact.training_rows == 0 {
        bail!("probability calibration artifact must persist a non-zero training row count");
    }
    if artifact.min_fit_rows < 32 {
        bail!(
            "probability calibration artifact min_fit_rows must be at least 32, got {}",
            artifact.min_fit_rows
        );
    }
    if !artifact.fitted {
        bail!("probability calibration expert artifact is marked as unfitted");
    }
    Ok(())
}

fn validate_probability_calibration_expert_save_state(
    state: &ProbabilityCalibrationExpert,
) -> Result<()> {
    validate_meta_blender_save_state(&state.backend)?;
    validate_probability_calibrator_live_state(&state.calibrator)?;
    if !state.fitted {
        bail!("probability calibration expert is not fitted");
    }
    if state.feature_columns.is_empty() {
        bail!("probability calibration expert must persist feature columns before save");
    }
    if state.training_rows == 0 {
        bail!("probability calibration expert must persist training rows before save");
    }
    if state.feature_columns != state.backend.feature_columns {
        bail!(
            "probability calibration expert feature-column mismatch between state {:?} and backend {:?}",
            state.feature_columns,
            state.backend.feature_columns
        );
    }
    if state.training_rows != state.backend.training_rows {
        bail!(
            "probability calibration expert training row mismatch between state {} and backend {}",
            state.training_rows,
            state.backend.training_rows
        );
    }
    if state.min_fit_rows < 32 {
        bail!(
            "probability calibration expert min_fit_rows must be at least 32, got {}",
            state.min_fit_rows
        );
    }
    Ok(())
}

fn validate_conformal_artifact(artifact: &ConformalGateArtifact) -> Result<()> {
    if !artifact.alpha.is_finite() || !(0.0..1.0).contains(&artifact.alpha) {
        bail!("conformal gate alpha must be finite and strictly between 0 and 1");
    }
    if !artifact.qhat.is_finite() || !(0.0..=1.0).contains(&artifact.qhat) {
        bail!("conformal gate qhat must be finite and between 0 and 1");
    }
    if artifact.fitted && artifact.n_calib < 32 {
        bail!(
            "fitted conformal gate must retain at least 32 calibration rows, got {}",
            artifact.n_calib
        );
    }
    if !artifact.fitted && artifact.n_calib != 0 {
        bail!("unfitted conformal gate should not persist calibration row count");
    }
    Ok(())
}

fn validate_conformal_prediction_expert_save_state(
    state: &ConformalPredictionExpert,
) -> Result<()> {
    validate_meta_blender_save_state(&state.backend)?;
    validate_probability_calibrator_live_state(&state.calibrator)?;
    validate_conformal_gate_live_state(&state.conformal_gate)?;
    if !state.fitted {
        bail!("conformal prediction expert is not fitted");
    }
    if state.feature_columns.is_empty() {
        bail!("conformal prediction expert must persist feature columns before save");
    }
    if state.training_rows == 0 {
        bail!("conformal prediction expert must persist training rows before save");
    }
    if state.feature_columns != state.backend.feature_columns {
        bail!(
            "conformal prediction expert feature-column mismatch between state {:?} and backend {:?}",
            state.feature_columns,
            state.backend.feature_columns
        );
    }
    if state.training_rows != state.backend.training_rows {
        bail!(
            "conformal prediction expert training row mismatch between state {} and backend {}",
            state.training_rows,
            state.backend.training_rows
        );
    }
    if !(1..=3).contains(&state.min_prediction_set) {
        bail!(
            "conformal prediction expert min_prediction_set must be between 1 and 3, got {}",
            state.min_prediction_set
        );
    }
    if state.min_fit_rows < 32 {
        bail!(
            "conformal prediction expert min_fit_rows must be at least 32, got {}",
            state.min_fit_rows
        );
    }
    Ok(())
}

fn validate_conformal_prediction_expert_artifact(
    artifact: &ConformalPredictionExpertArtifact,
) -> Result<()> {
    if artifact.feature_columns.is_empty() {
        bail!("conformal prediction artifact must contain at least one feature column");
    }
    if artifact.training_rows == 0 {
        bail!("conformal prediction artifact must persist a non-zero training row count");
    }
    if !artifact.alpha.is_finite() || !(0.0..1.0).contains(&artifact.alpha) {
        bail!("conformal prediction artifact alpha must be finite and strictly between 0 and 1");
    }
    if !(1..=3).contains(&artifact.min_prediction_set) {
        bail!(
            "conformal prediction artifact min_prediction_set must be between 1 and 3, got {}",
            artifact.min_prediction_set
        );
    }
    if artifact.min_fit_rows < 32 {
        bail!(
            "conformal prediction artifact min_fit_rows must be at least 32, got {}",
            artifact.min_fit_rows
        );
    }
    if !artifact.fitted {
        bail!("conformal prediction artifact is marked as unfitted");
    }
    Ok(())
}

fn validate_meta_stack_save_state(state: &MetaDecisionStack) -> Result<()> {
    validate_meta_blender_save_state(&state.blender)?;
    validate_probability_calibrator_live_state(&state.calibrator)?;
    validate_conformal_gate_live_state(&state.conformal_gate)?;
    if !state.fitted {
        bail!("meta decision stack is not fitted");
    }
    if state.feature_columns.is_empty() {
        bail!("meta decision stack must persist feature columns before save");
    }
    if state.training_rows == 0 {
        bail!("meta decision stack must persist training rows before save");
    }
    if state.feature_columns != state.blender.feature_columns {
        bail!(
            "meta decision stack feature-column mismatch between state {:?} and blender {:?}",
            state.feature_columns,
            state.blender.feature_columns
        );
    }
    if state.training_rows != state.blender.training_rows {
        bail!(
            "meta decision stack training row mismatch between state {} and blender {}",
            state.training_rows,
            state.blender.training_rows
        );
    }
    if !(1..=3).contains(&state.min_prediction_set) {
        bail!(
            "meta decision stack min_prediction_set must be between 1 and 3, got {}",
            state.min_prediction_set
        );
    }
    if state.min_fit_rows < 32 {
        bail!(
            "meta decision stack min_fit_rows must be at least 32, got {}",
            state.min_fit_rows
        );
    }
    Ok(())
}

fn validate_meta_stack_artifact(artifact: &MetaDecisionStackArtifact) -> Result<()> {
    if artifact.feature_columns.is_empty() {
        bail!("meta stack artifact must contain at least one feature column");
    }
    if artifact.training_rows == 0 {
        bail!("meta stack artifact must persist a non-zero training row count");
    }
    if !artifact.alpha.is_finite() || !(0.0..1.0).contains(&artifact.alpha) {
        bail!("meta stack artifact alpha must be finite and strictly between 0 and 1");
    }
    if !(1..=3).contains(&artifact.min_prediction_set) {
        bail!(
            "meta stack artifact min_prediction_set must be between 1 and 3, got {}",
            artifact.min_prediction_set
        );
    }
    if artifact.min_fit_rows < 32 {
        bail!(
            "meta stack artifact min_fit_rows must be at least 32, got {}",
            artifact.min_fit_rows
        );
    }
    if !artifact.fitted {
        bail!("meta decision stack artifact is marked as unfitted");
    }
    Ok(())
}

fn fit_binary_logistic(xs: &[f32], ys: &[f32]) -> CalibrationModel {
    if xs.is_empty() || ys.is_empty() || xs.len() != ys.len() {
        return CalibrationModel::Constant(0.5);
    }

    let positive_rate = ys.iter().copied().sum::<f32>() / ys.len() as f32;
    if !(1e-4..=1.0 - 1e-4).contains(&positive_rate) {
        return CalibrationModel::Constant(positive_rate.clamp(1e-4, 1.0 - 1e-4));
    }

    let mut a = 1.0_f32;
    let mut b = 0.0_f32;
    let learning_rate = 0.05_f32;
    let l2 = 1e-3_f32;

    for _ in 0..300 {
        let mut grad_a = 0.0_f32;
        let mut grad_b = 0.0_f32;

        for (x, y) in xs.iter().copied().zip(ys.iter().copied()) {
            let prediction = sigmoid(a * x + b);
            let error = prediction - y;
            grad_a += error * x;
            grad_b += error;
        }

        grad_a = grad_a / xs.len() as f32 + l2 * a;
        grad_b /= xs.len() as f32;

        a -= learning_rate * grad_a;
        b -= learning_rate * grad_b;
    }

    CalibrationModel::Platt { a, b }
}

fn select_temperature(probabilities: &Array2<f32>, labels: &[i32]) -> Result<f32> {
    if probabilities.nrows() != labels.len() {
        bail!(
            "temperature calibration row mismatch: {} rows vs {} labels",
            probabilities.nrows(),
            labels.len()
        );
    }

    let mut best_temperature = 1.0_f32;
    let mut best_loss = f32::INFINITY;

    for step in 10..=120 {
        let temperature = step as f32 / 20.0;
        let mut loss = 0.0_f32;

        for (row_idx, label) in labels.iter().copied().enumerate() {
            let class_idx = label_to_class_index(label)?;
            let row = [
                clamp_probability(probabilities[(row_idx, 0)]),
                clamp_probability(probabilities[(row_idx, 1)]),
                clamp_probability(probabilities[(row_idx, 2)]),
            ];
            let logits = [row[0].ln(), row[1].ln(), row[2].ln()];
            let max_logit = logits
                .iter()
                .map(|value| *value / temperature)
                .fold(f32::NEG_INFINITY, f32::max);

            let mut exp_sum = 0.0_f32;
            let mut scaled = [0.0_f32; 3];
            for idx in 0..3 {
                let value = ((logits[idx] / temperature) - max_logit).exp();
                scaled[idx] = value;
                exp_sum += value;
            }
            for value in &mut scaled {
                *value /= exp_sum.max(f32::EPSILON);
            }

            loss -= clamp_probability(scaled[class_idx]).ln();
        }

        loss /= labels.len().max(1) as f32;
        if loss < best_loss {
            best_loss = loss;
            best_temperature = temperature;
        }
    }

    Ok(best_temperature)
}

fn build_meta_runtime_prediction(
    model_name: &str,
    row: [f32; 3],
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, _) = conformal_gate.should_abstain(&row, min_prediction_set);
    let degraded_reason = join_degraded_reasons(
        [
            shared_abstain.then(|| "meta runtime confidence gate recommended abstain".to_string()),
            conformal_abstain.then(|| "meta conformal gate recommended abstain".to_string()),
        ]
        .into_iter()
        .flatten(),
    );
    build_runtime_prediction_with_details(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        row,
        Some(confidence),
        Some(shared_abstain || conformal_abstain),
        Some("xgboost_meta_blender+conformal_gate".to_string()),
        degraded_reason,
    )
}

fn calibration_method_name(method: CalibrationMethod) -> &'static str {
    match method {
        CalibrationMethod::Identity => "identity",
        CalibrationMethod::Platt => "platt",
        CalibrationMethod::Temperature => "temperature",
    }
}

fn build_probability_calibration_runtime_prediction(
    row: [f32; 3],
    calibration_method: CalibrationMethod,
) -> Result<RuntimePrediction> {
    let (confidence, abstain) = three_class_runtime_confidence(row)?;
    let degraded_reason = if abstain {
        Some("shared three-class confidence gate recommended abstain".to_string())
    } else {
        None
    };
    build_runtime_prediction_with_details(
        "probability_calibrator",
        ModelFamily::Meta,
        CapabilityState::Implemented,
        row,
        Some(confidence),
        Some(abstain),
        Some(format!(
            "xgboost_meta_blender+{}_calibration",
            calibration_method_name(calibration_method)
        )),
        degraded_reason,
    )
}

fn build_conformal_runtime_prediction(
    row: [f32; 3],
    calibration_method: CalibrationMethod,
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, prediction_set_size) =
        conformal_gate.should_abstain(&row, min_prediction_set);
    let degraded_reason = join_degraded_reasons(
        [
            shared_abstain
                .then(|| "shared three-class confidence gate recommended abstain".to_string()),
            conformal_abstain.then(|| {
                format!(
                    "conformal prediction set size {} reached abstain threshold {}",
                    prediction_set_size,
                    min_prediction_set.max(1)
                )
            }),
        ]
        .into_iter()
        .flatten(),
    );

    build_runtime_prediction_with_details(
        "conformal_gate",
        ModelFamily::Meta,
        CapabilityState::Implemented,
        row,
        Some(confidence),
        Some(shared_abstain || conformal_abstain),
        Some(format!(
            "xgboost_meta_blender+{}_calibration+conformal_gate",
            calibration_method_name(calibration_method)
        )),
        degraded_reason,
    )
}

fn build_meta_stack_runtime_prediction(
    row: [f32; 3],
    calibration_method: CalibrationMethod,
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, prediction_set_size) =
        conformal_gate.should_abstain(&row, min_prediction_set);
    let mut degraded_reasons = Vec::new();
    if shared_abstain {
        degraded_reasons.push("shared three-class confidence gate recommended abstain".to_string());
    }
    if conformal_abstain {
        degraded_reasons.push(format!(
            "conformal prediction set size {} reached abstain threshold {}",
            prediction_set_size,
            min_prediction_set.max(1)
        ));
    }
    build_runtime_prediction_with_details(
        "meta_stack",
        ModelFamily::Meta,
        CapabilityState::Implemented,
        row,
        Some(confidence),
        Some(shared_abstain || conformal_abstain),
        Some(format!(
            "xgboost_meta_blender+{}_calibration+conformal_gate",
            calibration_method_name(calibration_method)
        )),
        if degraded_reasons.is_empty() {
            None
        } else {
            Some(degraded_reasons.join("; "))
        },
    )
}

pub struct MetaBlender {
    pub model: Option<XGBoostExpert>,
    pub feature_columns: Vec<String>,
    pub fitted: bool,
    pub training_rows: usize,
}

impl MetaBlender {
    pub fn new() -> Self {
        Self {
            model: None,
            feature_columns: Vec::new(),
            fitted: false,
            training_rows: 0,
        }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let mut model = XGBoostExpert::new(0, None);
        model.fit(x, y)?;
        self.model = Some(model);
        self.feature_columns = x
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        self.training_rows = x.height();
        self.fitted = true;
        Ok(())
    }

    pub fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("MetaBlender is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        model.predict_proba(x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        validate_meta_blender_save_state(self)?;
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        let artifact = MetaBlenderArtifact {
            feature_columns: self.feature_columns.clone(),
            fitted: self.fitted,
            training_rows: self.training_rows,
        };
        with_staged_meta_artifact_dir(path, |staged_path| {
            write_json(
                &staged_path.join(METADATA_FILE_NAME),
                &meta_runtime_metadata(
                    "meta_blender",
                    self.feature_columns.clone(),
                    self.training_rows,
                ),
            )?;
            write_json(&staged_path.join(META_BLENDER_FILE_NAME), &artifact)?;
            model.save(&staged_path.join(BLENDER_BACKEND_DIR_NAME))
        })
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "meta_blender")?;
        let artifact: MetaBlenderArtifact = read_json(&path.join(META_BLENDER_FILE_NAME))?;
        if artifact.feature_columns.is_empty() {
            bail!("meta blender artifact must contain at least one feature column");
        }
        if artifact.feature_columns != metadata.feature_columns {
            bail!(
                "meta blender feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                artifact.feature_columns
            );
        }
        if !artifact.fitted {
            bail!("meta blender artifact is marked as unfitted");
        }
        if artifact.training_rows == 0 {
            bail!("meta blender artifact must persist a non-zero training row count");
        }
        if metadata.training_summary.dataset_rows != artifact.training_rows {
            bail!(
                "meta blender training row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.training_rows
            );
        }
        let mut model = XGBoostExpert::new(0, None);
        model.load(&path.join(BLENDER_BACKEND_DIR_NAME))?;
        if model.feature_columns != artifact.feature_columns {
            bail!(
                "meta blender backend feature-column mismatch between artifact {:?} and backend {:?}",
                artifact.feature_columns,
                model.feature_columns
            );
        }
        self.model = Some(model);
        self.feature_columns = artifact.feature_columns;
        self.fitted = artifact.fitted;
        self.training_rows = artifact.training_rows;
        Ok(())
    }
}

impl Default for MetaBlender {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for MetaBlender {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        MetaBlender::fit(self, x, y)
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        MetaBlender::predict_proba(self, x)
    }

    fn save(&self, path: &Path) -> Result<()> {
        MetaBlender::save(self, path)
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        MetaBlender::load(self, path)
    }
}

#[derive(Debug, Clone)]
pub struct ProbabilityCalibrator {
    pub method: CalibrationMethod,
    pub fitted: bool,
    pub models: Vec<CalibrationModel>,
}

impl ProbabilityCalibrator {
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            fitted: false,
            models: Vec::new(),
        }
    }

    pub fn fit_probabilities(&mut self, probabilities: &Array2<f32>, labels: &[i32]) -> Result<()> {
        if probabilities.nrows() != labels.len() {
            bail!(
                "calibration row mismatch: {} rows vs {} labels",
                probabilities.nrows(),
                labels.len()
            );
        }
        if probabilities.ncols() != 3 {
            bail!(
                "probability calibration requires exactly 3 classes, received {}",
                probabilities.ncols()
            );
        }

        self.models.clear();

        match self.method {
            CalibrationMethod::Identity => {}
            CalibrationMethod::Temperature => {
                let temperature = select_temperature(probabilities, labels)?;
                self.models
                    .push(CalibrationModel::Temperature { temperature });
            }
            CalibrationMethod::Platt => {
                for cls in 0..3 {
                    let mut x_cls = Vec::with_capacity(labels.len());
                    let mut y_cls = Vec::with_capacity(labels.len());
                    for row_idx in 0..labels.len() {
                        x_cls.push(logit(probabilities[(row_idx, cls)]));
                        let target = if label_to_class_index(labels[row_idx])? == cls {
                            1.0_f32
                        } else {
                            0.0_f32
                        };
                        y_cls.push(target);
                    }
                    self.models.push(fit_binary_logistic(&x_cls, &y_cls));
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    pub fn predict_proba(&self, probabilities: &Array2<f32>) -> Result<Array2<f32>> {
        if probabilities.ncols() != 3 {
            bail!(
                "probability calibration requires exactly 3 classes, received {}",
                probabilities.ncols()
            );
        }

        if !self.fitted {
            bail!("probability calibrator is not fitted");
        }

        if matches!(self.method, CalibrationMethod::Identity) {
            return Ok(renormalize_rows(probabilities));
        }

        match self.method {
            CalibrationMethod::Identity => Ok(renormalize_rows(probabilities)),
            CalibrationMethod::Temperature => {
                let CalibrationModel::Temperature { temperature } = self
                    .models
                    .first()
                    .cloned()
                    .context("temperature calibration model missing")?
                else {
                    bail!("temperature calibrator stored invalid model payload");
                };

                let mut calibrated = Array2::<f32>::zeros((probabilities.nrows(), 3));
                for row_idx in 0..probabilities.nrows() {
                    let logits = [
                        clamp_probability(probabilities[(row_idx, 0)]).ln(),
                        clamp_probability(probabilities[(row_idx, 1)]).ln(),
                        clamp_probability(probabilities[(row_idx, 2)]).ln(),
                    ];
                    let max_logit = logits
                        .iter()
                        .map(|value| *value / temperature)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0_f32;
                    for col_idx in 0..3 {
                        let value = ((logits[col_idx] / temperature) - max_logit).exp();
                        calibrated[(row_idx, col_idx)] = value;
                        exp_sum += value;
                    }
                    for col_idx in 0..3 {
                        calibrated[(row_idx, col_idx)] /= exp_sum.max(f32::EPSILON);
                    }
                }
                Ok(calibrated)
            }
            CalibrationMethod::Platt => {
                if self.models.len() != 3 {
                    bail!(
                        "platt calibration requires 3 binary calibrators, found {}",
                        self.models.len()
                    );
                }

                let mut calibrated = Array2::<f32>::zeros((probabilities.nrows(), 3));
                for row_idx in 0..probabilities.nrows() {
                    for cls in 0..3 {
                        let value = match self.models.get(cls).context("platt model missing")? {
                            CalibrationModel::Constant(probability) => {
                                clamp_probability(*probability)
                            }
                            CalibrationModel::Platt { a, b } => {
                                sigmoid(a * logit(probabilities[(row_idx, cls)]) + b)
                            }
                            CalibrationModel::Temperature { .. } => {
                                bail!("unexpected temperature model inside platt calibrator")
                            }
                        };
                        calibrated[(row_idx, cls)] = value;
                    }
                }
                Ok(renormalize_rows(&calibrated))
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("probability calibrator is not fitted");
        }
        let artifact = ProbabilityCalibratorArtifact {
            method: self.method,
            fitted: self.fitted,
            models: self.models.clone(),
        };
        validate_calibrator_artifact(&artifact)?;
        write_json_with_backup(&path.join(CALIBRATOR_FILE_NAME), &artifact)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: ProbabilityCalibratorArtifact = read_json(&path.join(CALIBRATOR_FILE_NAME))?;
        validate_calibrator_artifact(&artifact)?;
        if !artifact.fitted {
            bail!("probability calibrator artifact is marked as unfitted");
        }
        self.method = artifact.method;
        self.fitted = artifact.fitted;
        self.models = artifact.models;
        Ok(())
    }
}

impl Default for ProbabilityCalibrator {
    fn default() -> Self {
        Self::new(CalibrationMethod::Platt)
    }
}

pub struct ProbabilityCalibrationExpert {
    pub backend: MetaBlender,
    pub calibrator: ProbabilityCalibrator,
    pub min_fit_rows: usize,
    fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
}

impl ProbabilityCalibrationExpert {
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            backend: MetaBlender::new(),
            calibrator: ProbabilityCalibrator::new(method),
            min_fit_rows: 300,
            fitted: false,
            feature_columns: Vec::new(),
            training_rows: 0,
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)]),
                clamp_probability(probabilities[(row_idx, 1)]),
                clamp_probability(probabilities[(row_idx, 2)]),
            ];
            runtime_predictions.push(build_probability_calibration_runtime_prediction(
                row,
                self.calibrator.method,
            )?);
        }

        Ok(runtime_predictions)
    }
}

impl Default for ProbabilityCalibrationExpert {
    fn default() -> Self {
        Self::new(CalibrationMethod::Platt)
    }
}

impl ExpertModel for ProbabilityCalibrationExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        if x.height() < self.min_fit_rows {
            bail!(
                "probability calibration requires at least {} rows, received {}",
                self.min_fit_rows,
                x.height()
            );
        }
        self.backend.fit(x, y)?;
        let raw_probabilities = self.backend.predict_proba(x)?;
        let labels = series_labels(y)?;
        self.calibrator
            .fit_probabilities(&raw_probabilities, &labels)?;
        self.feature_columns = self.backend.feature_columns.clone();
        self.training_rows = x.height();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("probability calibration expert is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.backend.predict_proba(x)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_probability_calibration_expert_save_state(self)?;
        let artifact = ProbabilityCalibrationExpertArtifact {
            fitted: self.fitted,
            feature_columns: self.feature_columns.clone(),
            training_rows: self.training_rows,
            method: self.calibrator.method,
            min_fit_rows: self.min_fit_rows,
        };
        validate_probability_calibration_expert_artifact(&artifact)?;
        with_staged_meta_artifact_dir(path, |staged_path| {
            write_json(
                &staged_path.join(METADATA_FILE_NAME),
                &meta_runtime_metadata(
                    "probability_calibrator",
                    self.feature_columns.clone(),
                    self.training_rows,
                ),
            )?;
            write_json(&staged_path.join(CALIBRATION_EXPERT_FILE_NAME), &artifact)?;
            self.backend
                .save(&staged_path.join(CALIBRATION_BACKEND_DIR_NAME))?;
            self.calibrator.save(staged_path)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "probability_calibrator")?;
        let artifact: ProbabilityCalibrationExpertArtifact =
            read_json(&path.join(CALIBRATION_EXPERT_FILE_NAME))?;
        validate_probability_calibration_expert_artifact(&artifact)?;
        if metadata.training_summary.dataset_rows != artifact.training_rows {
            bail!(
                "probability calibration training row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.training_rows
            );
        }
        let mut next_state = Self::new(artifact.method);
        next_state
            .backend
            .load(&path.join(CALIBRATION_BACKEND_DIR_NAME))?;
        next_state.calibrator.load(path)?;
        if next_state.backend.feature_columns != metadata.feature_columns {
            bail!(
                "probability calibrator backend feature-column mismatch between metadata {:?} and backend {:?}",
                metadata.feature_columns,
                next_state.backend.feature_columns
            );
        }
        if next_state.calibrator.method != artifact.method {
            bail!(
                "probability calibrator method mismatch between artifact {:?} and calibrator {:?}",
                artifact.method,
                next_state.calibrator.method
            );
        }
        next_state.min_fit_rows = artifact.min_fit_rows.max(32);
        next_state.feature_columns = artifact.feature_columns;
        next_state.training_rows = artifact.training_rows;
        if next_state.feature_columns != metadata.feature_columns {
            bail!(
                "probability calibrator feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                next_state.feature_columns
            );
        }
        next_state.fitted = artifact.fitted;
        *self = next_state;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ConformalGate {
    pub alpha: f32,
    pub qhat: f32,
    pub fitted: bool,
    pub n_calib: usize,
}

impl ConformalGate {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 0.99),
            qhat: 1.0,
            fitted: false,
            n_calib: 0,
        }
    }

    pub fn fit_probabilities(&mut self, probabilities: &Array2<f32>, labels: &[i32]) -> Result<()> {
        if probabilities.nrows() != labels.len() {
            bail!(
                "conformal row mismatch: {} rows vs {} labels",
                probabilities.nrows(),
                labels.len()
            );
        }
        if probabilities.ncols() != 3 {
            bail!(
                "conformal gate requires exactly 3 classes, received {}",
                probabilities.ncols()
            );
        }
        if probabilities.nrows() < 32 {
            bail!(
                "conformal gate requires at least 32 calibration rows, received {}",
                probabilities.nrows()
            );
        }

        let alpha = self.alpha.clamp(1e-6, 0.99);
        let n = probabilities.nrows();
        let q_level = ((((n + 1) as f32) * (1.0 - alpha)).ceil() / n as f32).clamp(0.0, 1.0);

        let mut scores = Vec::with_capacity(n);
        for row_idx in 0..n {
            let label_idx = label_to_class_index(labels[row_idx])?;
            scores.push(1.0 - clamp_probability(probabilities[(row_idx, label_idx)]));
        }

        scores.sort_by(|left, right| left.total_cmp(right));
        let idx = ((q_level * n as f32).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        self.qhat = scores[idx].clamp(0.0, 1.0);
        self.fitted = true;
        self.n_calib = n;
        Ok(())
    }

    pub fn prediction_set(&self, row: &[f32; 3]) -> Vec<usize> {
        let mut keep = row
            .iter()
            .enumerate()
            .filter_map(|(idx, probability)| {
                if (1.0 - clamp_probability(*probability)) <= self.qhat {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if keep.is_empty() {
            let best_idx = row
                .iter()
                .copied()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            keep.push(best_idx);
        }

        keep
    }

    pub fn should_abstain(&self, row: &[f32; 3], min_set_size: usize) -> (bool, usize) {
        if !self.fitted {
            return (true, row.len().max(min_set_size.max(1)));
        }

        let keep = self.prediction_set(row);
        let size = keep.len();
        (size >= min_set_size.max(1), size)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("conformal gate is not fitted");
        }
        let artifact = ConformalGateArtifact {
            alpha: self.alpha,
            qhat: self.qhat,
            fitted: self.fitted,
            n_calib: self.n_calib,
        };
        validate_conformal_artifact(&artifact)?;
        write_json_with_backup(&path.join(CONFORMAL_FILE_NAME), &artifact)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: ConformalGateArtifact = read_json(&path.join(CONFORMAL_FILE_NAME))?;
        validate_conformal_artifact(&artifact)?;
        if !artifact.fitted {
            bail!("conformal gate artifact is marked as unfitted");
        }
        self.alpha = artifact.alpha;
        self.qhat = artifact.qhat;
        self.fitted = artifact.fitted;
        self.n_calib = artifact.n_calib;
        Ok(())
    }
}

impl Default for ConformalGate {
    fn default() -> Self {
        Self::new(0.10)
    }
}

pub struct ConformalPredictionExpert {
    pub backend: MetaBlender,
    pub calibrator: ProbabilityCalibrator,
    pub conformal_gate: ConformalGate,
    pub min_prediction_set: usize,
    pub min_fit_rows: usize,
    fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
}

impl ConformalPredictionExpert {
    pub fn new(method: CalibrationMethod, alpha: f32) -> Self {
        Self {
            backend: MetaBlender::new(),
            calibrator: ProbabilityCalibrator::new(method),
            conformal_gate: ConformalGate::new(alpha),
            min_prediction_set: 2,
            min_fit_rows: 300,
            fitted: false,
            feature_columns: Vec::new(),
            training_rows: 0,
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)]),
                clamp_probability(probabilities[(row_idx, 1)]),
                clamp_probability(probabilities[(row_idx, 2)]),
            ];
            runtime_predictions.push(build_conformal_runtime_prediction(
                row,
                self.calibrator.method,
                &self.conformal_gate,
                self.min_prediction_set,
            )?);
        }

        Ok(runtime_predictions)
    }
}

impl Default for ConformalPredictionExpert {
    fn default() -> Self {
        Self::new(CalibrationMethod::Platt, 0.10)
    }
}

impl ExpertModel for ConformalPredictionExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        if x.height() < self.min_fit_rows {
            bail!(
                "conformal gate requires at least {} rows, received {}",
                self.min_fit_rows,
                x.height()
            );
        }
        self.backend.fit(x, y)?;
        let raw_probabilities = self.backend.predict_proba(x)?;
        let labels = series_labels(y)?;
        self.calibrator
            .fit_probabilities(&raw_probabilities, &labels)?;
        let calibrated = self.calibrator.predict_proba(&raw_probabilities)?;
        self.conformal_gate
            .fit_probabilities(&calibrated, &labels)?;
        self.feature_columns = self.backend.feature_columns.clone();
        self.training_rows = x.height();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("conformal prediction expert is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.backend.predict_proba(x)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_conformal_prediction_expert_save_state(self)?;
        let artifact = ConformalPredictionExpertArtifact {
            fitted: self.fitted,
            feature_columns: self.feature_columns.clone(),
            training_rows: self.training_rows,
            alpha: self.conformal_gate.alpha,
            method: self.calibrator.method,
            min_prediction_set: self.min_prediction_set,
            min_fit_rows: self.min_fit_rows,
        };
        validate_conformal_prediction_expert_artifact(&artifact)?;
        with_staged_meta_artifact_dir(path, |staged_path| {
            write_json(
                &staged_path.join(METADATA_FILE_NAME),
                &meta_runtime_metadata(
                    "conformal_gate",
                    self.feature_columns.clone(),
                    self.training_rows,
                ),
            )?;
            write_json(&staged_path.join(CONFORMAL_EXPERT_FILE_NAME), &artifact)?;
            self.backend
                .save(&staged_path.join(CONFORMAL_BACKEND_DIR_NAME))?;
            self.calibrator.save(staged_path)?;
            self.conformal_gate.save(staged_path)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "conformal_gate")?;
        let artifact: ConformalPredictionExpertArtifact =
            read_json(&path.join(CONFORMAL_EXPERT_FILE_NAME))?;
        validate_conformal_prediction_expert_artifact(&artifact)?;
        if metadata.training_summary.dataset_rows != artifact.training_rows {
            bail!(
                "conformal prediction training row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.training_rows
            );
        }
        let mut next_state = Self::new(artifact.method, artifact.alpha);
        next_state
            .backend
            .load(&path.join(CONFORMAL_BACKEND_DIR_NAME))?;
        next_state.calibrator.load(path)?;
        next_state.conformal_gate.load(path)?;
        if next_state.backend.feature_columns != metadata.feature_columns {
            bail!(
                "conformal expert backend feature-column mismatch between metadata {:?} and backend {:?}",
                metadata.feature_columns,
                next_state.backend.feature_columns
            );
        }
        if next_state.calibrator.method != artifact.method {
            bail!(
                "conformal expert calibrator method mismatch between artifact {:?} and calibrator {:?}",
                artifact.method,
                next_state.calibrator.method
            );
        }
        if (next_state.conformal_gate.alpha - artifact.alpha).abs() > 1e-6 {
            bail!(
                "conformal expert alpha mismatch between artifact {} and gate {}",
                artifact.alpha,
                next_state.conformal_gate.alpha
            );
        }
        next_state.feature_columns = artifact.feature_columns;
        next_state.training_rows = artifact.training_rows;
        if next_state.feature_columns != metadata.feature_columns {
            bail!(
                "conformal expert feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                next_state.feature_columns
            );
        }
        next_state.min_prediction_set = artifact.min_prediction_set.max(1);
        next_state.min_fit_rows = artifact.min_fit_rows.max(32);
        next_state.fitted = artifact.fitted;
        *self = next_state;
        Ok(())
    }
}

pub struct MetaDecisionStack {
    pub blender: MetaBlender,
    pub calibrator: ProbabilityCalibrator,
    pub conformal_gate: ConformalGate,
    pub min_prediction_set: usize,
    pub min_fit_rows: usize,
    pub fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
}

impl MetaDecisionStack {
    pub fn new(method: CalibrationMethod, alpha: f32) -> Self {
        Self {
            blender: MetaBlender::new(),
            calibrator: ProbabilityCalibrator::new(method),
            conformal_gate: ConformalGate::new(alpha),
            min_prediction_set: 2,
            min_fit_rows: 300,
            fitted: false,
            feature_columns: Vec::new(),
            training_rows: 0,
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)]),
                clamp_probability(probabilities[(row_idx, 1)]),
                clamp_probability(probabilities[(row_idx, 2)]),
            ];
            runtime_predictions.push(build_meta_stack_runtime_prediction(
                row,
                self.calibrator.method,
                &self.conformal_gate,
                self.min_prediction_set,
            )?);
        }

        Ok(runtime_predictions)
    }
}

impl Default for MetaDecisionStack {
    fn default() -> Self {
        Self::new(CalibrationMethod::Platt, 0.10)
    }
}

impl ExpertModel for MetaDecisionStack {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        if x.height() < self.min_fit_rows {
            bail!(
                "meta stack requires at least {} rows, received {}",
                self.min_fit_rows,
                x.height()
            );
        }
        self.blender.fit(x, y)?;
        let raw_probabilities = self.blender.predict_proba(x)?;
        let labels = series_labels(y)?;

        self.calibrator
            .fit_probabilities(&raw_probabilities, &labels)?;
        let calibrated = self.calibrator.predict_proba(&raw_probabilities)?;
        self.conformal_gate
            .fit_probabilities(&calibrated, &labels)?;

        self.feature_columns = self.blender.feature_columns.clone();
        self.training_rows = x.height();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if !self.fitted {
            bail!("meta decision stack is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.blender.predict_proba(x)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_meta_stack_save_state(self)?;
        let artifact = MetaDecisionStackArtifact {
            fitted: self.fitted,
            feature_columns: self.feature_columns.clone(),
            training_rows: self.training_rows,
            method: self.calibrator.method,
            alpha: self.conformal_gate.alpha,
            min_prediction_set: self.min_prediction_set,
            min_fit_rows: self.min_fit_rows,
        };
        validate_meta_stack_artifact(&artifact)?;
        with_staged_meta_artifact_dir(path, |staged_path| {
            write_json(
                &staged_path.join(METADATA_FILE_NAME),
                &meta_runtime_metadata(
                    "meta_stack",
                    self.feature_columns.clone(),
                    self.training_rows,
                ),
            )?;
            write_json(&staged_path.join(META_STACK_FILE_NAME), &artifact)?;
            self.blender.save(&staged_path.join(BLENDER_DIR_NAME))?;
            self.calibrator.save(staged_path)?;
            self.conformal_gate.save(staged_path)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "meta_stack")?;
        let artifact: MetaDecisionStackArtifact = read_json(&path.join(META_STACK_FILE_NAME))?;
        validate_meta_stack_artifact(&artifact)?;
        if metadata.training_summary.dataset_rows != artifact.training_rows {
            bail!(
                "meta stack training row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.training_rows
            );
        }

        let mut next_state = Self::new(artifact.method, artifact.alpha);
        next_state.blender.load(&path.join(BLENDER_DIR_NAME))?;
        next_state.calibrator.load(path)?;
        next_state.conformal_gate.load(path)?;
        if next_state.blender.feature_columns != metadata.feature_columns {
            bail!(
                "meta stack blender feature-column mismatch between metadata {:?} and blender {:?}",
                metadata.feature_columns,
                next_state.blender.feature_columns
            );
        }
        if next_state.calibrator.method != artifact.method {
            bail!(
                "meta stack calibrator method mismatch between artifact {:?} and calibrator {:?}",
                artifact.method,
                next_state.calibrator.method
            );
        }
        if (next_state.conformal_gate.alpha - artifact.alpha).abs() > 1e-6 {
            bail!(
                "meta stack alpha mismatch between artifact {} and gate {}",
                artifact.alpha,
                next_state.conformal_gate.alpha
            );
        }
        next_state.fitted = artifact.fitted;
        next_state.feature_columns = artifact.feature_columns;
        next_state.training_rows = artifact.training_rows;
        if next_state.feature_columns != metadata.feature_columns {
            bail!(
                "meta stack feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                next_state.feature_columns
            );
        }
        next_state.min_prediction_set = artifact.min_prediction_set.max(1);
        next_state.min_fit_rows = artifact.min_fit_rows.max(32);
        *self = next_state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_models::XGBoostExpert;

    fn fitted_temperature_calibrator() -> ProbabilityCalibrator {
        ProbabilityCalibrator {
            method: CalibrationMethod::Temperature,
            fitted: true,
            models: vec![CalibrationModel::Temperature { temperature: 1.0 }],
        }
    }

    fn fitted_conformal_gate(alpha: f32) -> ConformalGate {
        ConformalGate {
            alpha,
            qhat: 0.55,
            fitted: true,
            n_calib: 128,
        }
    }

    #[test]
    fn validate_meta_metadata_rejects_inconsistent_training_summary() {
        let metadata = RuntimeArtifactMetadata::new(
            "meta_stack",
            ModelFamily::Meta,
            CapabilityState::Implemented,
            vec!["feature".to_string()],
            default_three_class_label_mapping(),
            crate::runtime::artifacts::TrainingSummaryMetadata::new(12, 8, 1),
        );

        let err = validate_meta_metadata(&metadata, "meta_stack")
            .expect_err("inconsistent meta training summary must fail");
        assert!(err.to_string().contains("training summary is inconsistent"));
    }

    #[test]
    fn meta_runtime_prediction_uses_shared_three_class_confidence_gate() -> Result<()> {
        let gate = ConformalGate::new(0.10);
        let row = [0.51_f32, 0.49, 0.0];

        let prediction = build_meta_runtime_prediction("meta_stack", row, &gate, 2)?;
        let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

        assert_eq!(prediction.confidence(), Some(expected_confidence));
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        Ok(())
    }

    #[test]
    fn conformal_prediction_artifact_rejects_invalid_prediction_set() {
        let err =
            validate_conformal_prediction_expert_artifact(&ConformalPredictionExpertArtifact {
                fitted: true,
                feature_columns: vec!["f1".to_string()],
                training_rows: 128,
                alpha: 0.10,
                method: CalibrationMethod::Platt,
                min_prediction_set: 4,
                min_fit_rows: 300,
            })
            .unwrap_err()
            .to_string();

        assert!(err.contains("min_prediction_set"));
    }

    #[test]
    fn conformal_prediction_runtime_uses_expert_metadata_and_backend_details() -> Result<()> {
        let mut expert = ConformalPredictionExpert::new(CalibrationMethod::Temperature, 0.10);
        expert.fitted = true;
        expert.feature_columns = vec!["feature".to_string()];
        expert.training_rows = 128;
        expert.conformal_gate.fitted = true;
        expert.conformal_gate.n_calib = 128;
        expert.conformal_gate.qhat = 0.20;

        let frame = DataFrame::new(vec![Series::new("feature".into(), &[1.0_f64]).into()])?;
        expert.backend = MetaBlender {
            model: None,
            feature_columns: vec!["feature".to_string()],
            fitted: true,
            training_rows: 128,
        };
        expert.calibrator.fitted = true;
        expert.calibrator.method = CalibrationMethod::Temperature;
        expert.calibrator.models = vec![CalibrationModel::Temperature { temperature: 1.0 }];

        let predictions = expert.predict_runtime(&frame);
        assert!(
            predictions.is_err(),
            "cold backend should still fail prediction"
        );

        let backend = format!(
            "xgboost_meta_blender+{}_calibration+conformal_gate",
            calibration_method_name(CalibrationMethod::Temperature)
        );
        assert_eq!(
            backend,
            "xgboost_meta_blender+temperature_calibration+conformal_gate"
        );
        Ok(())
    }

    #[test]
    fn probability_calibration_artifact_rejects_missing_feature_columns() {
        let err = validate_probability_calibration_expert_artifact(
            &ProbabilityCalibrationExpertArtifact {
                fitted: true,
                feature_columns: Vec::new(),
                training_rows: 128,
                method: CalibrationMethod::Platt,
                min_fit_rows: 300,
            },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("feature column"));
    }

    #[test]
    fn probability_calibration_runtime_uses_shared_confidence_and_backend_details() -> Result<()> {
        let row = [0.52_f32, 0.33, 0.15];
        let prediction =
            build_probability_calibration_runtime_prediction(row, CalibrationMethod::Temperature)?;
        let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

        assert_eq!(prediction.confidence(), Some(expected_confidence));
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        assert_eq!(
            prediction.metadata().execution_backend.as_deref(),
            Some("xgboost_meta_blender+temperature_calibration")
        );
        Ok(())
    }

    #[test]
    fn probability_calibration_runtime_surfaces_shared_abstain_reason() -> Result<()> {
        let row = [0.50_f32, 0.49, 0.01];
        let prediction =
            build_probability_calibration_runtime_prediction(row, CalibrationMethod::Temperature)?;

        assert_eq!(prediction.abstain_recommended(), Some(true));
        assert!(prediction
            .metadata()
            .degraded_reason
            .as_deref()
            .unwrap_or_default()
            .contains("shared three-class confidence gate recommended abstain"));
        Ok(())
    }

    #[test]
    fn meta_stack_artifact_rejects_invalid_prediction_set() {
        let err = validate_meta_stack_artifact(&MetaDecisionStackArtifact {
            fitted: true,
            feature_columns: vec!["f1".to_string()],
            training_rows: 128,
            method: CalibrationMethod::Platt,
            alpha: 0.10,
            min_prediction_set: 5,
            min_fit_rows: 300,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("min_prediction_set"));
    }

    #[test]
    fn meta_stack_runtime_uses_backend_details_and_shared_confidence() -> Result<()> {
        let gate = ConformalGate {
            alpha: 0.10,
            qhat: 0.20,
            fitted: true,
            n_calib: 128,
        };
        let row = [0.52_f32, 0.33, 0.15];
        let prediction =
            build_meta_stack_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
        let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row)?;

        assert_eq!(prediction.confidence(), Some(expected_confidence));
        assert_eq!(
            prediction.abstain_recommended(),
            Some(expected_abstain || gate.should_abstain(&row, 2).0)
        );
        assert_eq!(
            prediction.metadata().execution_backend.as_deref(),
            Some("xgboost_meta_blender+temperature_calibration+conformal_gate")
        );
        Ok(())
    }

    #[test]
    fn meta_stack_runtime_surfaces_combined_abstain_reasons() -> Result<()> {
        let gate = fitted_conformal_gate(0.10);
        let row = [0.50_f32, 0.49, 0.01];
        let prediction =
            build_meta_stack_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
        let degraded_reason = prediction
            .metadata()
            .degraded_reason
            .as_deref()
            .unwrap_or_default()
            .to_string();

        assert!(degraded_reason.contains("shared three-class confidence gate recommended abstain"));
        assert!(degraded_reason.contains("conformal prediction set size"));
        Ok(())
    }

    #[test]
    fn conformal_runtime_surfaces_shared_and_conformal_abstain_reasons() -> Result<()> {
        let gate = fitted_conformal_gate(0.10);
        let row = [0.50_f32, 0.49, 0.01];
        let prediction =
            build_conformal_runtime_prediction(row, CalibrationMethod::Temperature, &gate, 2)?;
        let degraded_reason = prediction
            .metadata()
            .degraded_reason
            .as_deref()
            .unwrap_or_default()
            .to_string();

        assert!(degraded_reason.contains("shared three-class confidence gate recommended abstain"));
        assert!(degraded_reason.contains("conformal prediction set size"));
        Ok(())
    }

    #[test]
    fn meta_blender_save_state_rejects_backend_feature_drift() {
        let mut backend = XGBoostExpert::new(0, None);
        backend.feature_columns = vec!["backend".to_string()];
        let blender = MetaBlender {
            model: Some(backend),
            feature_columns: vec!["state".to_string()],
            fitted: true,
            training_rows: 128,
        };

        let err = validate_meta_blender_save_state(&blender)
            .expect_err("feature-column drift must fail")
            .to_string();
        assert!(err.contains("feature-column mismatch"));
    }

    #[test]
    fn probability_calibration_save_state_rejects_backend_training_row_drift() {
        let mut backend = XGBoostExpert::new(0, None);
        backend.feature_columns = vec!["feature".to_string()];
        let expert = ProbabilityCalibrationExpert {
            backend: MetaBlender {
                model: Some(backend),
                feature_columns: vec!["feature".to_string()],
                fitted: true,
                training_rows: 64,
            },
            calibrator: fitted_temperature_calibrator(),
            min_fit_rows: 300,
            fitted: true,
            feature_columns: vec!["feature".to_string()],
            training_rows: 128,
        };

        let err = validate_probability_calibration_expert_save_state(&expert)
            .expect_err("backend/state training-row drift must fail")
            .to_string();
        assert!(err.contains("training row mismatch"));
    }

    #[test]
    fn meta_stack_save_state_rejects_blender_feature_drift() {
        let mut backend = XGBoostExpert::new(0, None);
        backend.feature_columns = vec!["backend".to_string()];
        let stack = MetaDecisionStack {
            blender: MetaBlender {
                model: Some(backend),
                feature_columns: vec!["backend".to_string()],
                fitted: true,
                training_rows: 128,
            },
            calibrator: fitted_temperature_calibrator(),
            conformal_gate: fitted_conformal_gate(0.10),
            min_prediction_set: 2,
            min_fit_rows: 300,
            fitted: true,
            feature_columns: vec!["state".to_string()],
            training_rows: 128,
        };

        let err = validate_meta_stack_save_state(&stack)
            .expect_err("feature-column drift must fail")
            .to_string();
        assert!(err.contains("feature-column mismatch"));
    }
}
