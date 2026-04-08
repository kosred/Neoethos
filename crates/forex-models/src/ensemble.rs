use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{build_runtime_prediction, ExpertModel};
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
    Ok(())
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

fn max_probability(row: &[f32; 3]) -> f32 {
    row.iter().copied().fold(0.0_f32, f32::max)
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
        std::fs::create_dir_all(path)
            .with_context(|| format!("create meta blender directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &meta_runtime_metadata(
                "meta_blender",
                self.feature_columns.clone(),
                self.training_rows,
            ),
        )?;
        write_json(
            &path.join(META_BLENDER_FILE_NAME),
            &MetaBlenderArtifact {
                feature_columns: self.feature_columns.clone(),
                fitted: self.fitted,
                training_rows: self.training_rows,
            },
        )?;
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        model.save(&path.join(BLENDER_BACKEND_DIR_NAME))?;
        Ok(())
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
        write_json(&path.join(CALIBRATOR_FILE_NAME), &artifact)
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
        if !self.fitted {
            bail!("probability calibration expert is not fitted");
        }
        std::fs::create_dir_all(path).with_context(|| {
            format!("create probability calibrator directory {}", path.display())
        })?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &meta_runtime_metadata(
                "probability_calibrator",
                self.feature_columns.clone(),
                self.training_rows,
            ),
        )?;
        write_json(
            &path.join(CALIBRATION_EXPERT_FILE_NAME),
            &ProbabilityCalibrationExpertArtifact {
                fitted: self.fitted,
                feature_columns: self.feature_columns.clone(),
                training_rows: self.training_rows,
                method: self.calibrator.method,
                min_fit_rows: self.min_fit_rows,
            },
        )?;
        self.backend
            .save(&path.join(CALIBRATION_BACKEND_DIR_NAME))?;
        self.calibrator.save(path)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "probability_calibrator")?;
        let artifact: ProbabilityCalibrationExpertArtifact =
            read_json(&path.join(CALIBRATION_EXPERT_FILE_NAME))?;
        if !artifact.fitted {
            bail!("probability calibration expert artifact is marked as unfitted");
        }
        if artifact.training_rows == 0 {
            bail!("probability calibration artifact must persist a non-zero training row count");
        }
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
        write_json(&path.join(CONFORMAL_FILE_NAME), &artifact)
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
            let confidence = max_probability(&row);
            let (abstain, _) = self
                .conformal_gate
                .should_abstain(&row, self.min_prediction_set);
            runtime_predictions.push(build_runtime_prediction(
                "conformal_gate",
                crate::runtime::capabilities::ModelFamily::Meta,
                crate::runtime::capabilities::CapabilityState::Implemented,
                row,
                Some(confidence),
                Some(abstain),
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
        if !self.fitted {
            bail!("conformal prediction expert is not fitted");
        }
        std::fs::create_dir_all(path)
            .with_context(|| format!("create conformal expert directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &meta_runtime_metadata(
                "conformal_gate",
                self.feature_columns.clone(),
                self.training_rows,
            ),
        )?;
        write_json(
            &path.join(CONFORMAL_EXPERT_FILE_NAME),
            &ConformalPredictionExpertArtifact {
                fitted: self.fitted,
                feature_columns: self.feature_columns.clone(),
                training_rows: self.training_rows,
                alpha: self.conformal_gate.alpha,
                method: self.calibrator.method,
                min_prediction_set: self.min_prediction_set,
                min_fit_rows: self.min_fit_rows,
            },
        )?;
        self.backend.save(&path.join(CONFORMAL_BACKEND_DIR_NAME))?;
        self.calibrator.save(path)?;
        self.conformal_gate.save(path)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "conformal_gate")?;
        let artifact: ConformalPredictionExpertArtifact =
            read_json(&path.join(CONFORMAL_EXPERT_FILE_NAME))?;
        if !artifact.fitted {
            bail!("conformal prediction expert artifact is marked as unfitted");
        }
        if artifact.training_rows == 0 {
            bail!("conformal prediction artifact must persist a non-zero training row count");
        }
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
            let confidence = max_probability(&row);
            let (abstain, _) = self
                .conformal_gate
                .should_abstain(&row, self.min_prediction_set);
            runtime_predictions.push(build_runtime_prediction(
                "meta_stack",
                crate::runtime::capabilities::ModelFamily::Meta,
                crate::runtime::capabilities::CapabilityState::Implemented,
                row,
                Some(confidence),
                Some(abstain),
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
        if !self.fitted {
            bail!("meta decision stack is not fitted");
        }
        std::fs::create_dir_all(path)
            .with_context(|| format!("create meta stack directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &meta_runtime_metadata(
                "meta_stack",
                self.feature_columns.clone(),
                self.training_rows,
            ),
        )?;
        write_json(
            &path.join(META_STACK_FILE_NAME),
            &MetaDecisionStackArtifact {
                fitted: self.fitted,
                feature_columns: self.feature_columns.clone(),
                training_rows: self.training_rows,
                method: self.calibrator.method,
                alpha: self.conformal_gate.alpha,
                min_prediction_set: self.min_prediction_set,
                min_fit_rows: self.min_fit_rows,
            },
        )?;
        self.blender.save(&path.join(BLENDER_DIR_NAME))?;
        self.calibrator.save(path)?;
        self.conformal_gate.save(path)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_meta_metadata(&metadata, "meta_stack")?;
        let artifact: MetaDecisionStackArtifact = read_json(&path.join(META_STACK_FILE_NAME))?;
        if !artifact.fitted {
            bail!("meta decision stack artifact is marked as unfitted");
        }
        if artifact.training_rows == 0 {
            bail!("meta decision stack artifact must persist a non-zero training row count");
        }
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
