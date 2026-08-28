use anyhow::{Context, Result, bail};
use ndarray::Array2;
use neoethos_core::storage::json::{
    DirBackupWriteConfig, JsonBackupWriteConfig, write_dir_with_backup,
    write_json_with_backup as write_json_artifact_with_backup,
};
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, three_class_runtime_confidence,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, default_three_class_label_mapping};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    METADATA_FILE_NAME, ensure_feature_columns_match, meta_runtime_metadata, read_json, write_json,
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
const META_F64_ARTIFACT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    Identity,
    Platt,
    Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationModel {
    Constant(f64),
    Platt { a: f64, b: f64 },
    Temperature { temperature: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetaBlenderArtifact {
    schema_version: u32,
    feature_columns: Vec<String>,
    fitted: bool,
    training_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilityCalibratorArtifact {
    schema_version: u32,
    method: CalibrationMethod,
    fitted: bool,
    models: Vec<CalibrationModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConformalGateArtifact {
    schema_version: u32,
    alpha: f64,
    qhat: f64,
    fitted: bool,
    n_calib: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetaDecisionStackArtifact {
    schema_version: u32,
    fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
    method: CalibrationMethod,
    alpha: f64,
    min_prediction_set: usize,
    min_fit_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilityCalibrationExpertArtifact {
    schema_version: u32,
    fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
    method: CalibrationMethod,
    min_fit_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConformalPredictionExpertArtifact {
    schema_version: u32,
    fitted: bool,
    feature_columns: Vec<String>,
    training_rows: usize,
    alpha: f64,
    method: CalibrationMethod,
    min_prediction_set: usize,
    min_fit_rows: usize,
}

fn validate_meta_f64_schema(schema_version: u32, artifact: &str) -> Result<()> {
    if schema_version != META_F64_ARTIFACT_SCHEMA_VERSION {
        bail!(
            "{artifact} schema version {schema_version} is unsupported; expected f64 schema version {META_F64_ARTIFACT_SCHEMA_VERSION}"
        );
    }
    Ok(())
}

fn label_to_class_index(label: i32) -> Result<usize> {
    match label {
        -1 => Ok(2),
        0 => Ok(0),
        1 => Ok(1),
        other => bail!("unsupported meta label {other}; expected one of -1, 0, 1"),
    }
}

fn validate_probability_row(values: &[f64], context: &str) -> Result<()> {
    if values.is_empty() {
        bail!("{context} probability row may not be empty");
    }
    let mut sum = 0.0_f64;
    for (column, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            bail!("{context} probability at column {column} must be finite");
        }
        if !(0.0..=1.0 + 1e-4).contains(&value) {
            bail!("{context} probability at column {column} must be between 0 and 1, got {value}");
        }
        sum += value;
    }
    if !sum.is_finite() || sum <= f64::EPSILON {
        bail!("{context} probability row must have finite positive mass");
    }
    Ok(())
}

fn validate_probability_matrix(probabilities: &Array2<f64>, context: &str) -> Result<()> {
    for row_idx in 0..probabilities.nrows() {
        let row = probabilities.row(row_idx);
        validate_probability_row(
            row.as_slice()
                .context("probability matrix row must be contiguous")?,
            &format!("{context} row {row_idx}"),
        )?;
    }
    Ok(())
}

fn clamp_probability(value: f64) -> Result<f64> {
    if !value.is_finite() {
        bail!("scalar probability must be finite");
    }
    if !(0.0..=1.0 + 1e-4).contains(&value) {
        bail!("scalar probability must be between 0 and 1, got {value}");
    }
    Ok(value.clamp(1e-6, 1.0 - 1e-6))
}

fn renormalize_rows(probabilities: &Array2<f64>) -> Result<Array2<f64>> {
    validate_probability_matrix(probabilities, "normalization input")?;
    let mut normalized = probabilities.clone();
    for row_idx in 0..normalized.nrows() {
        let mut sum = 0.0_f64;
        for col_idx in 0..normalized.ncols() {
            let value = normalized[(row_idx, col_idx)];
            sum += value;
        }

        for col_idx in 0..normalized.ncols() {
            normalized[(row_idx, col_idx)] /= sum;
        }
    }
    Ok(normalized)
}

fn logit(probability: f64) -> Result<f64> {
    let p = clamp_probability(probability)?;
    Ok((p / (1.0 - p)).ln())
}

fn sigmoid(value: f64) -> f64 {
    neoethos_core::utils::stable_sigmoid_f64(value)
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

/// GROUP E remediation 2026-05-25: the hand-rolled
/// `staged_meta_artifact_dir` / `backup_meta_artifact_dir` /
/// `cleanup_meta_artifact_dir` / `replace_meta_artifact_dir` /
/// `with_staged_meta_artifact_dir` quintet was replaced with a single
/// call to the canonical `neoethos_core::storage::json::write_dir_with_backup`
/// helper. Saves ~80 LOC of duplicate staged-tmp+backup logic per file
/// (this file is one of 4 — ensemble/bayesian/linear/training-orchestrator).
fn with_staged_meta_artifact_dir<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    write_dir_with_backup(
        path,
        DirBackupWriteConfig {
            artifact_label: "meta artifact",
            temp_extension: "tmp_meta_artifact",
            backup_extension: "bak_meta_artifact",
        },
        writer,
    )
}

fn write_json_with_backup<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    write_json_artifact_with_backup(
        path,
        value,
        JsonBackupWriteConfig {
            artifact_label: "meta-model artifact",
            temp_extension: "tmp_meta_file",
            backup_extension: "bak_meta_file",
        },
    )
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
    validate_meta_f64_schema(artifact.schema_version, "probability calibrator")?;
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
        schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
        schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
        alpha: state.alpha,
        qhat: state.qhat,
        fitted: state.fitted,
        n_calib: state.n_calib,
    })
}

fn validate_probability_calibration_expert_artifact(
    artifact: &ProbabilityCalibrationExpertArtifact,
) -> Result<()> {
    validate_meta_f64_schema(artifact.schema_version, "probability calibration expert")?;
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
    validate_meta_f64_schema(artifact.schema_version, "conformal gate")?;
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
    validate_meta_f64_schema(artifact.schema_version, "conformal prediction expert")?;
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
    validate_meta_f64_schema(artifact.schema_version, "meta decision stack")?;
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

fn fit_binary_logistic(xs: &[f64], ys: &[f64]) -> CalibrationModel {
    if xs.is_empty() || ys.is_empty() || xs.len() != ys.len() {
        return CalibrationModel::Constant(0.5);
    }

    let positive_rate = ys.iter().copied().sum::<f64>() / ys.len() as f64;
    if !(1e-4..=1.0 - 1e-4).contains(&positive_rate) {
        return CalibrationModel::Constant(positive_rate.clamp(1e-4, 1.0 - 1e-4));
    }

    let mut a = 1.0_f64;
    let mut b = 0.0_f64;
    let learning_rate = 0.05_f64;
    let l2 = 1e-3_f64;

    for _ in 0..300 {
        let mut grad_a = 0.0_f64;
        let mut grad_b = 0.0_f64;

        for (x, y) in xs.iter().copied().zip(ys.iter().copied()) {
            let prediction = sigmoid(a * x + b);
            let error = prediction - y;
            grad_a += error * x;
            grad_b += error;
        }

        grad_a = grad_a / xs.len() as f64 + l2 * a;
        grad_b /= xs.len() as f64;

        a -= learning_rate * grad_a;
        b -= learning_rate * grad_b;
    }

    CalibrationModel::Platt { a, b }
}

fn select_temperature(probabilities: &Array2<f64>, labels: &[i32]) -> Result<f64> {
    if probabilities.nrows() != labels.len() {
        bail!(
            "temperature calibration row mismatch: {} rows vs {} labels",
            probabilities.nrows(),
            labels.len()
        );
    }
    validate_probability_matrix(probabilities, "temperature calibration")?;

    let mut best_temperature = 1.0_f64;
    let mut best_loss = f64::INFINITY;

    for step in 10..=120 {
        let temperature = step as f64 / 20.0;
        let mut loss = 0.0_f64;

        for (row_idx, label) in labels.iter().copied().enumerate() {
            let class_idx = label_to_class_index(label)?;
            let row = [
                clamp_probability(probabilities[(row_idx, 0)])?,
                clamp_probability(probabilities[(row_idx, 1)])?,
                clamp_probability(probabilities[(row_idx, 2)])?,
            ];
            let logits = [row[0].ln(), row[1].ln(), row[2].ln()];
            let max_logit = logits
                .iter()
                .map(|value| *value / temperature)
                .fold(f64::NEG_INFINITY, f64::max);

            let mut exp_sum = 0.0_f64;
            let mut scaled = [0.0_f64; 3];
            for idx in 0..3 {
                let value = ((logits[idx] / temperature) - max_logit).exp();
                scaled[idx] = value;
                exp_sum += value;
            }
            for value in &mut scaled {
                *value /= exp_sum.max(f64::EPSILON);
            }

            loss -= clamp_probability(scaled[class_idx])?.ln();
        }

        loss /= labels.len().max(1) as f64;
        if loss < best_loss {
            best_loss = loss;
            best_temperature = temperature;
        }
    }

    Ok(best_temperature)
}

#[cfg(test)]
fn build_meta_runtime_prediction(
    model_name: &str,
    row: [f64; 3],
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, _) = conformal_gate.should_abstain(&row, min_prediction_set)?;
    let degraded_reason = join_degraded_reasons(
        [
            shared_abstain.then(|| "meta runtime confidence gate recommended abstain".to_string()),
            conformal_abstain.then(|| "meta conformal gate recommended abstain".to_string()),
        ]
        .into_iter()
        .flatten(),
    );
    Ok(build_runtime_prediction_with_details(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        row,
        Some(confidence),
        Some(shared_abstain || conformal_abstain),
        Some("xgboost_meta_blender+conformal_gate".to_string()),
        degraded_reason,
    )?)
}

fn calibration_method_name(method: CalibrationMethod) -> &'static str {
    match method {
        CalibrationMethod::Identity => "identity",
        CalibrationMethod::Platt => "platt",
        CalibrationMethod::Temperature => "temperature",
    }
}

fn build_probability_calibration_runtime_prediction(
    row: [f64; 3],
    calibration_method: CalibrationMethod,
) -> Result<RuntimePrediction> {
    let (confidence, abstain) = three_class_runtime_confidence(row)?;
    let degraded_reason = if abstain {
        Some("shared three-class confidence gate recommended abstain".to_string())
    } else {
        None
    };
    Ok(build_runtime_prediction_with_details(
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
    )?)
}

fn build_conformal_runtime_prediction(
    row: [f64; 3],
    calibration_method: CalibrationMethod,
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, prediction_set_size) =
        conformal_gate.should_abstain(&row, min_prediction_set)?;
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

    Ok(build_runtime_prediction_with_details(
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
    )?)
}

fn build_meta_stack_runtime_prediction(
    row: [f64; 3],
    calibration_method: CalibrationMethod,
    conformal_gate: &ConformalGate,
    min_prediction_set: usize,
) -> Result<RuntimePrediction> {
    let (confidence, shared_abstain) = three_class_runtime_confidence(row)?;
    let (conformal_abstain, prediction_set_size) =
        conformal_gate.should_abstain(&row, min_prediction_set)?;
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
    Ok(build_runtime_prediction_with_details(
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
    )?)
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

    pub fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        let mut model = XGBoostExpert::new(0, None);
        model.fit(x, y, lease)?;
        self.model = Some(model);
        self.feature_columns = x.names.clone();
        self.training_rows = x.n_samples();
        self.fitted = true;
        Ok(())
    }

    pub fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        if !self.fitted {
            bail!("MetaBlender is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        model.predict_proba(x, lease)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        validate_meta_blender_save_state(self)?;
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        let artifact = MetaBlenderArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
                )?,
            )?;
            write_json(&staged_path.join(META_BLENDER_FILE_NAME), &artifact)?;
            model.save(&staged_path.join(BLENDER_BACKEND_DIR_NAME))
        })
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "meta_blender")?;
        let artifact: MetaBlenderArtifact = read_json(&path.join(META_BLENDER_FILE_NAME))?;
        validate_meta_f64_schema(artifact.schema_version, "meta blender")?;
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
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        MetaBlender::fit(self, x, y, lease)
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        MetaBlender::predict_proba(self, x, lease)
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

    pub fn fit_probabilities(&mut self, probabilities: &Array2<f64>, labels: &[i32]) -> Result<()> {
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
        validate_probability_matrix(probabilities, "calibration fit")?;

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
                        x_cls.push(logit(probabilities[(row_idx, cls)])?);
                        let target = if label_to_class_index(labels[row_idx])? == cls {
                            1.0_f64
                        } else {
                            0.0_f64
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

    pub fn predict_proba(&self, probabilities: &Array2<f64>) -> Result<Array2<f64>> {
        if probabilities.ncols() != 3 {
            bail!(
                "probability calibration requires exactly 3 classes, received {}",
                probabilities.ncols()
            );
        }

        if !self.fitted {
            bail!("probability calibrator is not fitted");
        }
        validate_probability_matrix(probabilities, "calibration prediction")?;

        if matches!(self.method, CalibrationMethod::Identity) {
            return renormalize_rows(probabilities);
        }

        match self.method {
            CalibrationMethod::Identity => renormalize_rows(probabilities),
            CalibrationMethod::Temperature => {
                let CalibrationModel::Temperature { temperature } = self
                    .models
                    .first()
                    .cloned()
                    .context("temperature calibration model missing")?
                else {
                    bail!("temperature calibrator stored invalid model payload");
                };

                let mut calibrated = Array2::<f64>::zeros((probabilities.nrows(), 3));
                for row_idx in 0..probabilities.nrows() {
                    let logits = [
                        clamp_probability(probabilities[(row_idx, 0)])?.ln(),
                        clamp_probability(probabilities[(row_idx, 1)])?.ln(),
                        clamp_probability(probabilities[(row_idx, 2)])?.ln(),
                    ];
                    let max_logit = logits
                        .iter()
                        .map(|value| *value / temperature)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut exp_sum = 0.0_f64;
                    for col_idx in 0..3 {
                        let value = ((logits[col_idx] / temperature) - max_logit).exp();
                        calibrated[(row_idx, col_idx)] = value;
                        exp_sum += value;
                    }
                    for col_idx in 0..3 {
                        calibrated[(row_idx, col_idx)] /= exp_sum.max(f64::EPSILON);
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

                let mut calibrated = Array2::<f64>::zeros((probabilities.nrows(), 3));
                for row_idx in 0..probabilities.nrows() {
                    for cls in 0..3 {
                        let value = match self.models.get(cls).context("platt model missing")? {
                            CalibrationModel::Constant(probability) => {
                                clamp_probability(*probability)?
                            }
                            CalibrationModel::Platt { a, b } => {
                                sigmoid(a * logit(probabilities[(row_idx, cls)])? + b)
                            }
                            CalibrationModel::Temperature { .. } => {
                                bail!("unexpected temperature model inside platt calibrator")
                            }
                        };
                        calibrated[(row_idx, cls)] = value;
                    }
                }
                renormalize_rows(&calibrated)
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("probability calibrator is not fitted");
        }
        let artifact = ProbabilityCalibratorArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)])?,
                clamp_probability(probabilities[(row_idx, 1)])?,
                clamp_probability(probabilities[(row_idx, 2)])?,
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
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        if x.n_samples() < self.min_fit_rows {
            bail!(
                "probability calibration requires at least {} rows, received {}",
                self.min_fit_rows,
                x.n_samples()
            );
        }
        self.backend.fit(x, y, lease)?;
        let raw_probabilities = self.backend.predict_proba(x, lease)?;
        self.calibrator.fit_probabilities(&raw_probabilities, y)?;
        self.feature_columns = self.backend.feature_columns.clone();
        self.training_rows = x.n_samples();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        if !self.fitted {
            bail!("probability calibration expert is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.backend.predict_proba(x, lease)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_probability_calibration_expert_save_state(self)?;
        let artifact = ProbabilityCalibrationExpertArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
                )?,
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
    pub alpha: f64,
    pub qhat: f64,
    pub fitted: bool,
    pub n_calib: usize,
}

impl ConformalGate {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 0.99),
            qhat: 1.0,
            fitted: false,
            n_calib: 0,
        }
    }

    pub fn fit_probabilities(&mut self, probabilities: &Array2<f64>, labels: &[i32]) -> Result<()> {
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
        validate_probability_matrix(probabilities, "conformal calibration")?;

        let alpha = self.alpha.clamp(1e-6, 0.99);
        let n = probabilities.nrows();
        let q_level = ((((n + 1) as f64) * (1.0 - alpha)).ceil() / n as f64).clamp(0.0, 1.0);

        let mut scores = Vec::with_capacity(n);
        for row_idx in 0..n {
            let label_idx = label_to_class_index(labels[row_idx])?;
            scores.push(1.0 - clamp_probability(probabilities[(row_idx, label_idx)])?);
        }

        scores.sort_by(|left, right| left.total_cmp(right));
        let idx = ((q_level * n as f64).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        self.qhat = scores[idx].clamp(0.0, 1.0);
        self.fitted = true;
        self.n_calib = n;
        Ok(())
    }

    pub fn prediction_set(&self, row: &[f64; 3]) -> Result<Vec<usize>> {
        validate_probability_row(row, "conformal prediction")?;
        let mut keep = row
            .iter()
            .enumerate()
            .map(|(idx, probability)| {
                Ok(((1.0 - clamp_probability(*probability)?) <= self.qhat).then_some(idx))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
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

        Ok(keep)
    }

    pub fn should_abstain(&self, row: &[f64; 3], min_set_size: usize) -> Result<(bool, usize)> {
        validate_probability_row(row, "conformal abstention")?;
        if !self.fitted {
            return Ok((true, row.len().max(min_set_size.max(1))));
        }

        let keep = self.prediction_set(row)?;
        let size = keep.len();
        Ok((size >= min_set_size.max(1), size))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            bail!("conformal gate is not fitted");
        }
        let artifact = ConformalGateArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
    pub fn new(method: CalibrationMethod, alpha: f64) -> Self {
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

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)])?,
                clamp_probability(probabilities[(row_idx, 1)])?,
                clamp_probability(probabilities[(row_idx, 2)])?,
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
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        if x.n_samples() < self.min_fit_rows {
            bail!(
                "conformal gate requires at least {} rows, received {}",
                self.min_fit_rows,
                x.n_samples()
            );
        }
        self.backend.fit(x, y, lease)?;
        let raw_probabilities = self.backend.predict_proba(x, lease)?;
        self.calibrator.fit_probabilities(&raw_probabilities, y)?;
        let calibrated = self.calibrator.predict_proba(&raw_probabilities)?;
        self.conformal_gate.fit_probabilities(&calibrated, y)?;
        self.feature_columns = self.backend.feature_columns.clone();
        self.training_rows = x.n_samples();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        if !self.fitted {
            bail!("conformal prediction expert is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.backend.predict_proba(x, lease)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_conformal_prediction_expert_save_state(self)?;
        let artifact = ConformalPredictionExpertArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
                )?,
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
    pub fn new(method: CalibrationMethod, alpha: f64) -> Self {
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

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        let mut runtime_predictions = Vec::with_capacity(probabilities.nrows());

        for row_idx in 0..probabilities.nrows() {
            let row = [
                clamp_probability(probabilities[(row_idx, 0)])?,
                clamp_probability(probabilities[(row_idx, 1)])?,
                clamp_probability(probabilities[(row_idx, 2)])?,
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
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        if x.n_samples() < self.min_fit_rows {
            bail!(
                "meta stack requires at least {} rows, received {}",
                self.min_fit_rows,
                x.n_samples()
            );
        }
        self.blender.fit(x, y, lease)?;
        let raw_probabilities = self.blender.predict_proba(x, lease)?;

        self.calibrator.fit_probabilities(&raw_probabilities, y)?;
        let calibrated = self.calibrator.predict_proba(&raw_probabilities)?;
        self.conformal_gate.fit_probabilities(&calibrated, y)?;

        self.feature_columns = self.blender.feature_columns.clone();
        self.training_rows = x.n_samples();
        self.fitted = true;
        Ok(())
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        if !self.fitted {
            bail!("meta decision stack is not fitted");
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let raw_probabilities = self.blender.predict_proba(x, lease)?;
        self.calibrator.predict_proba(&raw_probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        validate_meta_stack_save_state(self)?;
        let artifact = MetaDecisionStackArtifact {
            schema_version: META_F64_ARTIFACT_SCHEMA_VERSION,
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
                )?,
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
#[path = "ensemble_tests.rs"]
mod tests;
