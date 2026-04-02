use anyhow::{Context, Result, bail};
#[cfg(feature = "adaptive-models")]
use irithyll::ensemble::config::DriftDetectorType;
#[cfg(feature = "adaptive-models")]
use irithyll::loss::logistic::LogisticLoss;
#[cfg(feature = "adaptive-models")]
use irithyll::serde_support::{load_model, save_model_with};
#[cfg(feature = "adaptive-models")]
use irithyll::{DynSGBT, LossType, SGBT, SGBTConfig, Sample};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::base::{
    ExpertModel, build_runtime_artifact_metadata, canonical_three_class_label_mapping,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, MODEL_FILE_NAME, ensure_feature_columns_match,
    feature_matrix_from_dataframe, read_json, remap_three_class_labels, softmax_rows, write_json,
};
#[cfg(feature = "adaptive-models")]
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum AdaptiveModelKind {
    PassiveAggressive,
    Hoeffding,
}

impl AdaptiveModelKind {
    fn model_name(self) -> &'static str {
        match self {
            Self::PassiveAggressive => "online_pa",
            Self::Hoeffding => "online_hoeffding",
        }
    }
}

fn adaptive_runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Adaptive,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn validate_adaptive_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_model_name: &str,
) -> Result<()> {
    if metadata.model_name != expected_model_name {
        bail!(
            "adaptive artifact model mismatch: expected {}, got {}",
            expected_model_name,
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Adaptive {
        bail!(
            "adaptive artifact family mismatch: expected {:?}, got {:?}",
            ModelFamily::Adaptive,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "adaptive artifact state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("adaptive artifact label mapping mismatch");
    }
    if metadata.feature_columns.is_empty() {
        bail!("adaptive artifact metadata must contain at least one feature column");
    }
    Ok(())
}

fn usize_param(params: &HashMap<String, String>, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn float_param(params: &HashMap<String, String>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn float_param_f32(params: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn labels_to_binary_targets(labels: &[usize], positive_class: usize) -> Vec<f64> {
    labels
        .iter()
        .map(|label| if *label == positive_class { 1.0 } else { 0.0 })
        .collect()
}

fn balanced_class_weights(labels: &[usize], classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; classes.max(1)];
    for &label in labels {
        if let Some(slot) = counts.get_mut(label) {
            *slot += 1;
        }
    }

    let total = labels.len().max(1) as f32;
    counts
        .into_iter()
        .map(|count| {
            let count = count.max(1) as f32;
            (total / (classes.max(1) as f32 * count)).clamp(0.5, 4.0)
        })
        .collect()
}

fn array2_from_logits(rows: usize, cols: usize, logits: Vec<f32>) -> Result<Array2<f32>> {
    Array2::from_shape_vec((rows, cols), logits).context("build adaptive model logits")
}

fn fallback_logits(
    features: &Array2<f32>,
    scaler: &FeatureScaler,
    weights: &Array2<f32>,
    bias: &Array1<f32>,
) -> Result<Array2<f32>> {
    let features = scaler.transform(features)?;
    let mut logits = Vec::with_capacity(features.nrows() * 3);
    for row in 0..features.nrows() {
        for class_idx in 0..3 {
            logits.push(
                weights
                    .row(class_idx)
                    .iter()
                    .zip(features.row(row).iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    + bias[class_idx],
            );
        }
    }

    array2_from_logits(features.nrows(), 3, logits)
}

#[cfg(feature = "adaptive-models")]
fn committee_output_to_logit(output: f64) -> f32 {
    if !output.is_finite() {
        return 0.0;
    }
    let probability = output.clamp(1e-6, 1.0 - 1e-6);
    (probability / (1.0 - probability)).ln() as f32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PassiveAggressiveArtifact {
    model_name: String,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    scaler: FeatureScaler,
    weights: Array2<f32>,
    bias: Array1<f32>,
    aggressiveness: f32,
    epochs: usize,
}

#[derive(Debug, Clone)]
pub struct OnlinePassiveAggressiveExpert {
    pub aggressiveness: f32,
    pub epochs: usize,
    pub feature_columns: Vec<String>,
    pub dataset_rows: usize,
    pub scaler: Option<FeatureScaler>,
    pub weights: Option<Array2<f32>>,
    pub bias: Option<Array1<f32>>,
}

impl OnlinePassiveAggressiveExpert {
    pub fn new(aggressiveness: f32, epochs: usize) -> Self {
        Self {
            aggressiveness: aggressiveness.max(1e-4),
            epochs: epochs.max(1),
            feature_columns: Vec::new(),
            dataset_rows: 0,
            scaler: None,
            weights: None,
            bias: None,
        }
    }

    pub fn from_params(params: Option<HashMap<String, String>>) -> Self {
        let params = params.unwrap_or_default();
        Self::new(
            float_param_f32(&params, "c", 1.0),
            usize_param(&params, "epochs", 4),
        )
    }

    fn artifact(&self) -> Result<PassiveAggressiveArtifact> {
        Ok(PassiveAggressiveArtifact {
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            feature_columns: self.feature_columns.clone(),
            dataset_rows: self.dataset_rows,
            scaler: self.scaler.clone().context("missing PA scaler")?,
            weights: self.weights.clone().context("missing PA weights")?,
            bias: self.bias.clone().context("missing PA bias")?,
            aggressiveness: self.aggressiveness,
            epochs: self.epochs,
        })
    }
}

impl Default for OnlinePassiveAggressiveExpert {
    fn default() -> Self {
        Self::new(1.0, 4)
    }
}

impl ExpertModel for OnlinePassiveAggressiveExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
        let labels = remap_three_class_labels(y)?;
        let scaler = FeatureScaler::fit(&features)?;
        let features = scaler.transform(&features)?;
        let sample_weights = balanced_class_weights(&labels, 3);

        let n_rows = features.nrows();
        let n_cols = features.ncols();
        let mut weights = Array2::<f32>::zeros((3, n_cols));
        let mut bias = Array1::<f32>::zeros(3);

        for _ in 0..self.epochs {
            for (row, target_class) in labels.iter().enumerate().take(n_rows) {
                let x_row = features.row(row);
                let norm_sq = x_row.iter().map(|value| value * value).sum::<f32>() + 1e-6;

                let mut scores = [0.0_f32; 3];
                for class_idx in 0..3 {
                    scores[class_idx] = weights
                        .row(class_idx)
                        .iter()
                        .zip(x_row.iter())
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>()
                        + bias[class_idx];
                }

                let predicted_class = scores
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(class_idx, _)| class_idx)
                    .unwrap_or(*target_class);
                if predicted_class == *target_class {
                    continue;
                }

                let margin = scores[predicted_class] - scores[*target_class] + 1.0;
                if margin <= 0.0 {
                    continue;
                }

                let tau = (margin * sample_weights[*target_class] / (2.0 * norm_sq))
                    .min(self.aggressiveness);
                for col in 0..n_cols {
                    weights[(*target_class, col)] += tau * x_row[col];
                    weights[(predicted_class, col)] -= tau * x_row[col];
                }
                bias[*target_class] += tau;
                bias[predicted_class] -= tau;
            }
        }

        self.feature_columns = feature_columns;
        self.dataset_rows = n_rows;
        self.scaler = Some(scaler);
        self.weights = Some(weights);
        self.bias = Some(bias);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let weights = self
            .weights
            .as_ref()
            .context("online_pa model not fitted")?;
        let bias = self.bias.as_ref().context("online_pa model not fitted")?;
        let scaler = self.scaler.as_ref().context("online_pa scaler missing")?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let features = scaler.transform(&features)?;

        let mut logits = Vec::with_capacity(features.nrows() * 3);
        for row in 0..features.nrows() {
            for class_idx in 0..3 {
                let score = weights
                    .row(class_idx)
                    .iter()
                    .zip(features.row(row).iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    + bias[class_idx];
                logits.push(score);
            }
        }

        Ok(softmax_rows(&array2_from_logits(
            features.nrows(),
            3,
            logits,
        )?))
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create online_pa directory {}", path.display()))?;
        write_json(
            &path.join(METADATA_FILE_NAME),
            &adaptive_runtime_metadata(
                AdaptiveModelKind::PassiveAggressive.model_name(),
                self.feature_columns.clone(),
                self.dataset_rows,
            ),
        )?;
        write_json(&path.join(MODEL_FILE_NAME), &self.artifact()?)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_adaptive_metadata(&metadata, AdaptiveModelKind::PassiveAggressive.model_name())?;
        let artifact: PassiveAggressiveArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if artifact.model_name != AdaptiveModelKind::PassiveAggressive.model_name() {
            bail!("expected online_pa artifact, got {}", artifact.model_name);
        }
        if artifact.feature_columns != metadata.feature_columns {
            bail!(
                "online_pa feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.dataset_rows != metadata.training_summary.dataset_rows {
            bail!(
                "online_pa dataset-row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.dataset_rows
            );
        }

        self.aggressiveness = artifact.aggressiveness;
        self.epochs = artifact.epochs;
        self.feature_columns = artifact.feature_columns;
        self.dataset_rows = artifact.dataset_rows;
        self.scaler = Some(artifact.scaler);
        self.weights = Some(artifact.weights);
        self.bias = Some(artifact.bias);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HoeffdingArtifact {
    model_name: String,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    params: HashMap<String, String>,
    committee_json: Vec<String>,
    #[serde(default)]
    fallback_scaler: Option<FeatureScaler>,
    #[serde(default)]
    fallback_weights: Option<Array2<f32>>,
    #[serde(default)]
    fallback_bias: Option<Array1<f32>>,
}

fn fit_fallback_online_committee(
    features: &Array2<f32>,
    labels: &[usize],
    params: &HashMap<String, String>,
) -> Result<(FeatureScaler, Array2<f32>, Array1<f32>)> {
    let scaler = FeatureScaler::fit(features)?;
    let features = scaler.transform(features)?;
    let rows = features.nrows();
    let cols = features.ncols();
    if rows == 0 || cols == 0 {
        bail!("online_hoeffding fallback requires a non-empty feature matrix");
    }

    let lr = float_param_f32(params, "learning_rate", 0.03).max(1e-4);
    let epochs = usize_param(params, "epochs", 4).max(1);
    let l2 = float_param_f32(params, "l2", 1e-4).max(0.0);
    let sample_weights = balanced_class_weights(labels, 3);
    let mut weights = Array2::<f32>::zeros((3, cols));
    let mut bias = Array1::<f32>::zeros(3);

    for _ in 0..epochs {
        for row in 0..rows {
            let x_row = features.row(row);
            let mut logits = Array2::<f32>::zeros((1, 3));
            for class_idx in 0..3 {
                logits[(0, class_idx)] = weights
                    .row(class_idx)
                    .iter()
                    .zip(x_row.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    + bias[class_idx];
            }
            let probabilities = softmax_rows(&logits);
            let sample_weight = sample_weights[labels[row]];
            for class_idx in 0..3 {
                let target = if labels[row] == class_idx { 1.0 } else { 0.0 };
                let error = (probabilities[(0, class_idx)] - target) * sample_weight;
                for col in 0..cols {
                    weights[(class_idx, col)] -=
                        lr * (error * x_row[col] + l2 * weights[(class_idx, col)]);
                }
                bias[class_idx] -= lr * error;
            }
        }
    }

    Ok((scaler, weights, bias))
}

pub struct OnlineHoeffdingExpert {
    params: HashMap<String, String>,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    committee_json: Vec<String>,
    fallback_scaler: Option<FeatureScaler>,
    fallback_weights: Option<Array2<f32>>,
    fallback_bias: Option<Array1<f32>>,
    #[cfg(feature = "adaptive-models")]
    committees: Vec<DynSGBT>,
}

impl OnlineHoeffdingExpert {
    pub fn new(params: Option<HashMap<String, String>>) -> Self {
        Self {
            params: params.unwrap_or_default(),
            feature_columns: Vec::new(),
            dataset_rows: 0,
            committee_json: Vec::new(),
            fallback_scaler: None,
            fallback_weights: None,
            fallback_bias: None,
            #[cfg(feature = "adaptive-models")]
            committees: Vec::new(),
        }
    }

    #[cfg(feature = "adaptive-models")]
    fn drift_detector(&self) -> DriftDetectorType {
        match self
            .params
            .get("drift_detector")
            .map(|value| value.trim().to_ascii_lowercase())
        {
            Some(kind) if kind == "adwin" => DriftDetectorType::Adwin {
                delta: float_param(&self.params, "drift_delta", 0.002),
            },
            Some(kind) if kind == "ddm" => DriftDetectorType::Ddm {
                warning_level: float_param(&self.params, "warning_level", 2.0),
                drift_level: float_param(&self.params, "drift_level", 3.0),
                min_instances: usize_param(&self.params, "min_instances", 30) as u64,
            },
            _ => DriftDetectorType::PageHinkley {
                delta: float_param(&self.params, "drift_delta", 0.005),
                lambda: float_param(&self.params, "drift_lambda", 50.0),
            },
        }
    }

    #[cfg(feature = "adaptive-models")]
    fn config(&self) -> Result<SGBTConfig> {
        SGBTConfig::builder()
            .n_steps(usize_param(&self.params, "n_steps", 24))
            .learning_rate(float_param(&self.params, "learning_rate", 0.05))
            .feature_subsample_rate(float_param(&self.params, "feature_subsample_rate", 0.8))
            .max_depth(usize_param(&self.params, "max_depth", 5))
            .n_bins(usize_param(&self.params, "n_bins", 32))
            .grace_period(usize_param(&self.params, "grace_period", 32))
            .delta(float_param(&self.params, "delta", 1e-7))
            .lambda(float_param(&self.params, "lambda", 1.0))
            .gamma(float_param(&self.params, "gamma", 0.0))
            .drift_detector(self.drift_detector())
            .build()
            .map_err(|err| anyhow::anyhow!("invalid online_hoeffding config: {err}"))
    }

    fn artifact(&self) -> HoeffdingArtifact {
        HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: self.feature_columns.clone(),
            dataset_rows: self.dataset_rows,
            params: self.params.clone(),
            committee_json: self.committee_json.clone(),
            fallback_scaler: self.fallback_scaler.clone(),
            fallback_weights: self.fallback_weights.clone(),
            fallback_bias: self.fallback_bias.clone(),
        }
    }

    fn has_persistable_fallback(&self) -> bool {
        self.fallback_scaler.is_some()
            && self.fallback_weights.is_some()
            && self.fallback_bias.is_some()
    }

    #[cfg(feature = "adaptive-models")]
    fn restore_committees(&mut self) -> Result<()> {
        self.committees.clear();
        let mut failures = 0usize;
        for payload in &self.committee_json {
            match load_model(payload) {
                Ok(model) => self.committees.push(model),
                Err(err) => {
                    failures += 1;
                    warn!(
                        "failed to restore online_hoeffding committee from artifact: {}",
                        err
                    );
                }
            }
        }
        if !self.committee_json.is_empty() && self.committees.len() != self.committee_json.len() {
            if self.fallback_scaler.is_none()
                || self.fallback_weights.is_none()
                || self.fallback_bias.is_none()
            {
                bail!(
                    "restored {} of {} online_hoeffding committees and no fallback committee is available",
                    self.committees.len(),
                    self.committee_json.len()
                );
            }
            warn!(
                "restored {} of {} online_hoeffding committees; using persisted linear fallback path for integrity",
                self.committees.len(),
                self.committee_json.len()
            );
            self.committees.clear();
        }
        if failures > 0 && self.committee_json.is_empty() {
            bail!(
                "online_hoeffding restore reported committee failures with no committee payloads"
            );
        }
        Ok(())
    }
}

impl Default for OnlineHoeffdingExpert {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ExpertModel for OnlineHoeffdingExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        #[cfg(not(feature = "adaptive-models"))]
        {
            let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
            let labels = remap_three_class_labels(y)?;
            let (scaler, weights, bias) =
                fit_fallback_online_committee(&features, &labels, &self.params)?;
            self.params
                .entry("classes".to_string())
                .or_insert_with(|| "3".to_string());
            self.feature_columns = feature_columns;
            self.dataset_rows = features.nrows();
            self.committee_json.clear();
            self.fallback_scaler = Some(scaler);
            self.fallback_weights = Some(weights);
            self.fallback_bias = Some(bias);
            Ok(())
        }

        #[cfg(feature = "adaptive-models")]
        {
            let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
            let labels = remap_three_class_labels(y)?;
            let config = self.config()?;
            let rows = (0..features.nrows())
                .map(|row_idx| {
                    features
                        .row(row_idx)
                        .iter()
                        .map(|value| *value as f64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut committee_json = Vec::with_capacity(3);
            let mut committees = Vec::with_capacity(3);

            for class_idx in 0..3 {
                let binary_targets = labels_to_binary_targets(&labels, class_idx);
                let mut committee = SGBT::with_loss(config.clone(), LogisticLoss);
                for (features, target) in rows.iter().zip(binary_targets.iter()) {
                    committee.train_one(&Sample::new(features.clone(), *target));
                }

                let payload = save_model_with(&committee, LossType::Logistic)
                    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
                let runtime =
                    load_model(&payload).map_err(|err| anyhow::anyhow!(err.to_string()))?;
                committee_json.push(payload);
                committees.push(runtime);
            }

            let (fallback_scaler, fallback_weights, fallback_bias) =
                fit_fallback_online_committee(&features, &labels, &self.params)?;

            self.params
                .entry("classes".to_string())
                .or_insert_with(|| "3".to_string());
            self.feature_columns = feature_columns;
            self.dataset_rows = features.nrows();
            self.committee_json = committee_json;
            self.fallback_scaler = Some(fallback_scaler);
            self.fallback_weights = Some(fallback_weights);
            self.fallback_bias = Some(fallback_bias);
            self.committees = committees;
            Ok(())
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(not(feature = "adaptive-models"))]
        {
            ensure_feature_columns_match(&self.feature_columns, x)?;
            let (features, _) = feature_matrix_from_dataframe(x)?;
            let scaler = self
                .fallback_scaler
                .as_ref()
                .context("online_hoeffding fallback scaler missing")?;
            let weights = self
                .fallback_weights
                .as_ref()
                .context("online_hoeffding fallback weights missing")?;
            let bias = self
                .fallback_bias
                .as_ref()
                .context("online_hoeffding fallback bias missing")?;
            Ok(softmax_rows(&fallback_logits(
                &features, scaler, weights, bias,
            )?))
        }

        #[cfg(feature = "adaptive-models")]
        {
            ensure_feature_columns_match(&self.feature_columns, x)?;
            let (features, _) = feature_matrix_from_dataframe(x)?;
            let rows = (0..features.nrows())
                .map(|row_idx| {
                    features
                        .row(row_idx)
                        .iter()
                        .map(|value| *value as f64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let fallback = self
                .fallback_scaler
                .as_ref()
                .zip(self.fallback_weights.as_ref())
                .zip(self.fallback_bias.as_ref())
                .map(|((scaler, weights), bias)| fallback_logits(&features, scaler, weights, bias))
                .transpose()?;

            if self.committees.len() == 3 {
                let mut committee_logits = Array2::<f32>::zeros((rows.len(), 3));
                for (row_idx, features) in rows.iter().enumerate() {
                    for (class_idx, committee) in self.committees.iter().enumerate() {
                        committee_logits[(row_idx, class_idx)] =
                            committee_output_to_logit(committee.predict(features));
                    }
                }
                if let Some(fallback) = fallback {
                    let mut blended = committee_logits;
                    for row in 0..blended.nrows() {
                        for class_idx in 0..blended.ncols() {
                            blended[(row, class_idx)] =
                                0.7 * blended[(row, class_idx)] + 0.3 * fallback[(row, class_idx)];
                        }
                    }
                    Ok(softmax_rows(&blended))
                } else {
                    Ok(softmax_rows(&committee_logits))
                }
            } else {
                let fallback = fallback.context("online_hoeffding fallback model missing")?;
                Ok(softmax_rows(&fallback))
            }
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create online_hoeffding directory {}", path.display()))?;
        if self.committee_json.is_empty() && !self.has_persistable_fallback() {
            bail!("online_hoeffding cannot be saved without committees or a fallback model");
        }
        write_json(
            &path.join(METADATA_FILE_NAME),
            &adaptive_runtime_metadata(
                AdaptiveModelKind::Hoeffding.model_name(),
                self.feature_columns.clone(),
                self.dataset_rows,
            ),
        )?;
        write_json(&path.join(MODEL_FILE_NAME), &self.artifact())?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_adaptive_metadata(&metadata, AdaptiveModelKind::Hoeffding.model_name())?;
        let artifact: HoeffdingArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if artifact.model_name != AdaptiveModelKind::Hoeffding.model_name() {
            bail!(
                "expected online_hoeffding artifact, got {}",
                artifact.model_name
            );
        }
        if artifact.feature_columns != metadata.feature_columns {
            bail!(
                "online_hoeffding feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.dataset_rows != metadata.training_summary.dataset_rows {
            bail!(
                "online_hoeffding dataset-row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.dataset_rows
            );
        }

        self.params = artifact.params;
        self.feature_columns = artifact.feature_columns;
        self.dataset_rows = artifact.dataset_rows;
        self.committee_json = artifact.committee_json;
        self.fallback_scaler = artifact.fallback_scaler;
        self.fallback_weights = artifact.fallback_weights;
        self.fallback_bias = artifact.fallback_bias;
        if self.committee_json.is_empty() && !self.has_persistable_fallback() {
            bail!("online_hoeffding artifact has neither committees nor a fallback model");
        }
        #[cfg(feature = "adaptive-models")]
        self.restore_committees()?;
        Ok(())
    }
}

pub struct AdaptiveGradientBooster {
    #[cfg(feature = "adaptive-models")]
    inner: SGBT,
    #[cfg(not(feature = "adaptive-models"))]
    weights: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    bias: f64,
    #[cfg(not(feature = "adaptive-models"))]
    learning_rate: f64,
    #[cfg(not(feature = "adaptive-models"))]
    feature_mean: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    feature_m2: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    observations: usize,
}

impl AdaptiveGradientBooster {
    pub fn new() -> Self {
        #[cfg(feature = "adaptive-models")]
        {
            let config = SGBTConfig::builder()
                .n_steps(24)
                .learning_rate(0.05)
                .feature_subsample_rate(0.8)
                .max_depth(5)
                .n_bins(32)
                .grace_period(32)
                .build()
                .expect("default adaptive booster config should be valid");
            Self {
                inner: SGBT::new(config),
            }
        }

        #[cfg(not(feature = "adaptive-models"))]
        {
            Self {
                weights: Vec::new(),
                bias: 0.0,
                learning_rate: 0.05,
                feature_mean: Vec::new(),
                feature_m2: Vec::new(),
                observations: 0,
            }
        }
    }

    #[cfg(not(feature = "adaptive-models"))]
    fn update_feature_statistics(&mut self, x: &[f64]) {
        if self.feature_mean.len() != x.len() {
            self.feature_mean = vec![0.0; x.len()];
            self.feature_m2 = vec![0.0; x.len()];
            self.observations = 0;
        }

        self.observations += 1;
        let count = self.observations as f64;
        for (idx, value) in x.iter().enumerate() {
            let delta = *value - self.feature_mean[idx];
            self.feature_mean[idx] += delta / count;
            let delta2 = *value - self.feature_mean[idx];
            self.feature_m2[idx] += delta * delta2;
        }
    }

    #[cfg(not(feature = "adaptive-models"))]
    fn normalize_features(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.weights.is_empty() {
            return Ok(x.to_vec());
        }
        if self.weights.len() != x.len()
            || self.feature_mean.len() != x.len()
            || self.feature_m2.len() != x.len()
        {
            bail!(
                "adaptive gradient booster expected {} features, got {}",
                self.weights.len(),
                x.len()
            );
        }

        Ok(x.iter()
            .enumerate()
            .map(|(idx, value)| {
                let variance = if self.observations > 1 {
                    self.feature_m2[idx] / (self.observations as f64 - 1.0)
                } else {
                    1.0
                };
                let scale = variance.abs().sqrt().max(1e-6);
                (*value - self.feature_mean[idx]) / scale
            })
            .collect())
    }

    pub fn learn_one(&mut self, x: Vec<f64>, y: f64) -> Result<()> {
        #[cfg(feature = "adaptive-models")]
        {
            self.inner.train_one(&Sample::new(x, y));
            Ok(())
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            if !self.weights.is_empty() && self.weights.len() != x.len() {
                bail!(
                    "adaptive gradient booster expected {} features, got {}",
                    self.weights.len(),
                    x.len()
                );
            }
            self.update_feature_statistics(&x);
            let x = self.normalize_features(&x)?;
            if self.weights.is_empty() {
                self.weights = vec![0.0; x.len()];
            }

            let prediction = self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + self.bias;
            let error = prediction - y;
            for (weight, value) in self.weights.iter_mut().zip(x.iter()) {
                *weight -= self.learning_rate * error * *value;
            }
            self.bias -= self.learning_rate * error;
            Ok(())
        }
    }

    pub fn predict_one(&self, x: &[f64]) -> Result<f64> {
        #[cfg(feature = "adaptive-models")]
        {
            Ok(self.inner.predict(x))
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            if self.weights.is_empty() {
                bail!("adaptive gradient booster has not observed any samples");
            }
            let x = self.normalize_features(x)?;
            Ok(self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + self.bias)
        }
    }
}

impl Default for AdaptiveGradientBooster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "adaptive-models"))]
mod tests {
    use super::*;
    use polars::prelude::{DataFrame, Series};
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_models_{name}_{}_{}",
            std::process::id(),
            stamp
        ))
    }

    #[test]
    fn online_hoeffding_loads_fallback_when_committees_are_corrupt() -> Result<()> {
        let path = temp_model_dir("online_hoeffding");
        std::fs::create_dir_all(&path)?;

        write_json(
            &path.join(METADATA_FILE_NAME),
            &adaptive_runtime_metadata(
                AdaptiveModelKind::Hoeffding.model_name(),
                vec!["f1".to_string(), "f2".to_string()],
                2,
            ),
        )?;

        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::new(),
            committee_json: vec!["not-json".to_string()],
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::from_shape_vec(
                (3, 2),
                vec![0.0, 0.0, 1.2, 0.0, -1.2, 0.0],
            )?),
            fallback_bias: Some(Array1::from(vec![0.0, 0.0, 0.0])),
        };
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;

        let mut expert = OnlineHoeffdingExpert::new(None);
        expert.load(&path)?;

        let frame = DataFrame::new(vec![
            Series::new("f1".into(), vec![1.0_f64, -1.0]).into(),
            Series::new("f2".into(), vec![0.5_f64, 0.25]).into(),
        ])?;
        let probabilities = expert.predict_proba(&frame)?;

        assert_eq!(probabilities.nrows(), 2);
        assert_eq!(probabilities.ncols(), 3);
        for row in 0..probabilities.nrows() {
            let sum = probabilities.row(row).iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }
}
