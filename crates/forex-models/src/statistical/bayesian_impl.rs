use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

use super::common::{
    FeatureScaler, METADATA_FILE_NAME, MODEL_FILE_NAME, ensure_feature_columns_match,
    feature_matrix_from_dataframe, read_json, remap_three_class_labels,
    runtime_backend_with_gpu_fallback, softmax_rows, write_json,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BayesianClassPosterior {
    weights: Array1<f32>,
    bias: f32,
    variance_diag: Array1<f32>,
    bias_variance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BayesianOneVsRestArtifact {
    model_name: String,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    scaler: FeatureScaler,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
    prior_precision: f32,
    learning_rate: f32,
    epochs: usize,
    classes: Vec<BayesianClassPosterior>,
}

fn sigmoid(value: f32) -> f32 {
    forex_core::utils::stable_sigmoid_f32(value)
}

fn split_train_val_indices(rows: usize) -> (Vec<usize>, Vec<usize>) {
    if rows <= 6 {
        return ((0..rows).collect(), Vec::new());
    }

    let val_rows = ((rows as f32) * 0.2).round() as usize;
    let val_rows = val_rows.clamp(1, rows.saturating_sub(2));
    let embargo_rows = if rows >= 20 {
        ((rows as f32) * 0.02).round() as usize
    } else {
        0
    };
    let embargo_rows = embargo_rows.clamp(0, rows.saturating_sub(val_rows + 1));
    let train_rows = rows.saturating_sub(val_rows + embargo_rows);

    if train_rows == 0 {
        return ((0..rows).collect(), Vec::new());
    }

    let train = (0..train_rows).collect::<Vec<_>>();
    let val = (train_rows + embargo_rows..rows).collect::<Vec<_>>();
    (train, val)
}

fn staged_bayesian_artifact_dir(path: &Path) -> PathBuf {
    path.with_extension("tmp_bayesian_artifact")
}

fn backup_bayesian_artifact_dir(path: &Path) -> PathBuf {
    path.with_extension("bak_bayesian_artifact")
}

fn cleanup_bayesian_artifact_dir(path: &Path) -> Result<()> {
    if path.exists() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("remove bayesian artifact directory {}", path.display()))?;
    }
    Ok(())
}

fn replace_bayesian_artifact_dir(staged_path: &Path, target_path: &Path) -> Result<()> {
    let backup_path = backup_bayesian_artifact_dir(target_path);
    cleanup_bayesian_artifact_dir(&backup_path)?;
    if target_path.exists() {
        std::fs::rename(target_path, &backup_path).with_context(|| {
            format!(
                "move previous bayesian artifact into backup {}",
                backup_path.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(staged_path, target_path) {
        if backup_path.exists() {
            if let Err(restore_err) = std::fs::rename(&backup_path, target_path) {
                tracing::error!(
                    target: "forex_models::artifact",
                    backup = %backup_path.display(),
                    target = %target_path.display(),
                    error = %restore_err,
                    "failed to restore backup after staged-rename failure;                      artifact directory may be in an inconsistent state"
                );
            }
        }
        bail!(
            "rename staged bayesian artifact into {} failed: {}",
            target_path.display(),
            error
        );
    }
    cleanup_bayesian_artifact_dir(&backup_path)?;
    Ok(())
}

fn with_staged_bayesian_artifact_dir<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    let staged_path = staged_bayesian_artifact_dir(path);
    cleanup_bayesian_artifact_dir(&staged_path)?;
    std::fs::create_dir_all(&staged_path).with_context(|| {
        format!(
            "create staged bayesian artifact directory {}",
            staged_path.display()
        )
    })?;
    if let Err(error) = writer(&staged_path) {
        let _ = cleanup_bayesian_artifact_dir(&staged_path);
        return Err(error);
    }
    replace_bayesian_artifact_dir(&staged_path, path)
}

fn runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    train_rows: usize,
    val_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new_unchecked(dataset_rows, train_rows, val_rows),
    )
}

fn normalize_bayesian_params(
    prior_precision: f32,
    learning_rate: f32,
    epochs: usize,
) -> Result<(f32, f32, usize)> {
    if !prior_precision.is_finite() {
        bail!("bayesian prior_precision must be finite");
    }
    if !learning_rate.is_finite() {
        bail!("bayesian learning_rate must be finite");
    }
    Ok((
        prior_precision.max(1e-6),
        learning_rate.max(1e-4),
        epochs.max(1),
    ))
}

fn fit_binary_posterior(
    train_features: &Array2<f32>,
    train_labels: &[f32],
    val_features: Option<&Array2<f32>>,
    val_labels: Option<&[f32]>,
    prior_precision: f32,
    learning_rate: f32,
    epochs: usize,
) -> Result<BayesianClassPosterior> {
    let rows = train_features.nrows();
    let cols = train_features.ncols();
    if rows == 0 || cols == 0 {
        bail!("bayesian logistic regression requires a non-empty feature matrix");
    }
    if train_labels.len() != rows {
        bail!(
            "bayesian logistic regression label mismatch: {} rows vs {} labels",
            rows,
            train_labels.len()
        );
    }
    if let Some(val_features) = val_features
        && let Some(val_labels) = val_labels
        && val_features.nrows() != val_labels.len()
    {
        bail!(
            "bayesian validation mismatch: {} rows vs {} labels",
            val_features.nrows(),
            val_labels.len()
        );
    }

    let prior = prior_precision.max(1e-6);
    let lr = learning_rate.max(1e-4);
    let mut weights = Array1::<f32>::zeros(cols);
    let mut bias = 0.0_f32;
    let mut best_weights = weights.clone();
    let mut best_bias = bias;
    let mut best_val_loss = f32::INFINITY;
    let mut stale_epochs = 0usize;
    let patience = 25usize;

    for _ in 0..epochs.max(1) {
        let mut grad_w = Array1::<f32>::zeros(cols);
        let mut grad_b = 0.0_f32;

        for (row, label) in train_labels.iter().enumerate().take(rows) {
            let x_row = train_features.row(row);
            let logit = weights
                .iter()
                .zip(x_row.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + bias;
            let probability = sigmoid(logit);
            let error = probability - *label;

            for col in 0..cols {
                grad_w[col] += error * x_row[col];
            }
            grad_b += error;
        }

        for col in 0..cols {
            grad_w[col] = grad_w[col] / rows as f32 + prior * weights[col];
            weights[col] -= lr * grad_w[col];
        }
        grad_b /= rows as f32;
        bias -= lr * grad_b;

        if let (Some(val_features), Some(val_labels)) = (val_features, val_labels)
            && val_features.nrows() > 0
        {
            let mut val_loss = 0.0_f32;
            for (row, target) in val_labels.iter().enumerate().take(val_features.nrows()) {
                let x_row = val_features.row(row);
                let logit = weights
                    .iter()
                    .zip(x_row.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    + bias;
                let probability = sigmoid(logit).clamp(1e-6, 1.0 - 1e-6);
                val_loss -= *target * probability.ln() + (1.0 - *target) * (1.0 - probability).ln();
            }
            val_loss /= val_features.nrows() as f32;

            if val_loss + 1e-6 < best_val_loss {
                best_val_loss = val_loss;
                best_weights = weights.clone();
                best_bias = bias;
                stale_epochs = 0;
            } else {
                stale_epochs += 1;
                if stale_epochs >= patience {
                    break;
                }
            }
        }
    }

    if best_val_loss.is_finite() {
        weights = best_weights;
        bias = best_bias;
    }

    let mut variance_diag = Array1::<f32>::zeros(cols);
    let mut bias_hessian = 0.0_f32;
    for row in 0..rows {
        let x_row = train_features.row(row);
        let logit = weights
            .iter()
            .zip(x_row.iter())
            .map(|(weight, value)| weight * value)
            .sum::<f32>()
            + bias;
        let probability = sigmoid(logit);
        let curvature = (probability * (1.0 - probability)).max(1e-6);
        bias_hessian += curvature;
        for col in 0..cols {
            variance_diag[col] += curvature * x_row[col] * x_row[col];
        }
    }

    for col in 0..cols {
        let diagonal = prior + variance_diag[col] / rows as f32;
        variance_diag[col] = 1.0 / diagonal.max(1e-6);
    }
    let bias_variance = 1.0 / (bias_hessian / rows as f32 + 1e-6);

    Ok(BayesianClassPosterior {
        weights,
        bias,
        variance_diag,
        bias_variance,
    })
}

fn predictive_logit(class_model: &BayesianClassPosterior, features: &[f32]) -> Result<f32> {
    let mean = class_model
        .weights
        .iter()
        .zip(features.iter())
        .map(|(weight, value)| weight * value)
        .sum::<f32>()
        + class_model.bias;
    let variance = class_model
        .variance_diag
        .iter()
        .zip(features.iter())
        .map(|(variance, value)| variance * value * value)
        .sum::<f32>()
        + class_model.bias_variance;
    if !mean.is_finite() || !variance.is_finite() {
        bail!("bayesian logistic regression produced non-finite posterior moments");
    }
    let correction = (1.0 + std::f32::consts::PI * variance.max(0.0) / 8.0).sqrt();
    Ok(mean / correction.max(1e-6))
}

fn runtime_predictions(
    model_name: &str,
    probabilities: &Array2<f32>,
) -> Result<Vec<RuntimePrediction>> {
    let cpu_backend = format!("{model_name}_bayesian_ovr_cpu");
    let (execution_backend, degraded_reason) =
        runtime_backend_with_gpu_fallback(model_name, &cpu_backend);
    let mut predictions = Vec::with_capacity(probabilities.nrows());
    for row in probabilities.outer_iter() {
        let row_values = [row[0], row[1], row[2]];
        let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
        let reason = degraded_reason.clone().or_else(|| {
            abstain_recommended
                .then(|| "shared three-class confidence gate recommended abstain".to_string())
        });
        predictions.push(build_runtime_prediction_with_details(
            model_name,
            ModelFamily::Meta,
            CapabilityState::Implemented,
            row_values,
            Some(confidence),
            Some(abstain_recommended),
            execution_backend.clone(),
            reason,
        )?);
    }

    Ok(predictions)
}

fn validate_runtime_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_model_name: &str,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if expected_feature_columns.is_empty() {
        bail!("persisted {expected_model_name} artifact is missing feature columns");
    }
    if metadata.model_name != expected_model_name {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected model name {expected_model_name}, got {}",
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Meta {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected family {:?}, got {:?}",
            ModelFamily::Meta,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected state {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("runtime metadata mismatch for {expected_model_name}: unexpected label mapping");
    }
    if metadata.feature_columns != expected_feature_columns {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected feature columns {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: expected {} dataset rows, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows == 0 {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: training rows must be non-zero"
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        bail!(
            "runtime metadata mismatch for {expected_model_name}: training rows {} + validation rows {} must equal dataset rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }

    Ok(())
}

fn resolve_runtime_metadata_from_artifact(
    path: &Path,
    artifact: &BayesianOneVsRestArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_runtime_metadata(
                &metadata,
                "bayes_logit",
                &artifact.feature_columns,
                artifact.dataset_rows,
            )
            .with_context(|| {
                format!(
                    "runtime metadata sidecar mismatch with embedded bayes_logit metadata at {}",
                    metadata_path.display()
                )
            })?;
            if let Some(embedded) = artifact.runtime_metadata.as_ref()
                && (embedded.model_name != metadata.model_name
                    || embedded.family != metadata.family
                    || embedded.state != metadata.state
                    || embedded.feature_columns != metadata.feature_columns
                    || embedded.label_mapping != metadata.label_mapping
                    || embedded.training_summary.dataset_rows
                        != metadata.training_summary.dataset_rows
                    || embedded.training_summary.train_rows != metadata.training_summary.train_rows
                    || embedded.training_summary.val_rows != metadata.training_summary.val_rows)
            {
                bail!(
                    "runtime metadata sidecar mismatch with embedded bayes_logit metadata at {}",
                    metadata_path.display()
                );
            }
            Ok(metadata)
        }
        Err(file_err) => {
            let fallback = artifact
                .runtime_metadata
                .clone()
                .with_context(|| format!("missing runtime metadata file {} and artifact has no embedded metadata: {file_err}", metadata_path.display()))?;
            validate_runtime_metadata(
                &fallback,
                "bayes_logit",
                &artifact.feature_columns,
                artifact.dataset_rows,
            )?;
            tracing::warn!(
                path = %metadata_path.display(),
                error = %file_err,
                model = "bayes_logit",
                "bayesian metadata sidecar missing/unreadable; using embedded runtime metadata"
            );
            Ok(fallback)
        }
    }
}

fn validate_bayesian_artifact(artifact: &BayesianOneVsRestArtifact) -> Result<()> {
    if artifact.model_name != "bayes_logit" {
        bail!(
            "unexpected bayesian artifact model name {}",
            artifact.model_name
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("bayesian artifact must contain at least one feature column");
    }
    if artifact.dataset_rows == 0 {
        bail!("bayesian artifact must persist a non-zero dataset row count");
    }
    if artifact.classes.len() != 3 {
        bail!(
            "bayesian artifact must persist exactly three class models, found {}",
            artifact.classes.len()
        );
    }
    if artifact.scaler.means.len() != artifact.feature_columns.len()
        || artifact.scaler.stds.len() != artifact.feature_columns.len()
    {
        bail!(
            "bayesian artifact scaler dimension mismatch: means {}, stds {}, features {}",
            artifact.scaler.means.len(),
            artifact.scaler.stds.len(),
            artifact.feature_columns.len()
        );
    }
    if artifact.runtime_metadata.is_none() {
        bail!("bayesian artifact must persist runtime metadata");
    }
    validate_runtime_metadata(
        artifact
            .runtime_metadata
            .as_ref()
            .expect("checked runtime metadata presence"),
        "bayes_logit",
        &artifact.feature_columns,
        artifact.dataset_rows,
    )?;
    if !artifact.prior_precision.is_finite() || artifact.prior_precision <= 0.0 {
        bail!("bayesian artifact prior_precision must be finite and positive");
    }
    if !artifact.learning_rate.is_finite() || artifact.learning_rate <= 0.0 {
        bail!("bayesian artifact learning_rate must be finite and positive");
    }
    if artifact.epochs == 0 {
        bail!("bayesian artifact epochs must be positive");
    }
    if artifact.scaler.means.iter().any(|value| !value.is_finite())
        || artifact.scaler.stds.iter().any(|value| !value.is_finite())
        || artifact.scaler.stds.iter().any(|value| *value <= 0.0)
        || artifact.classes.iter().any(|class_model| {
            class_model.weights.len() != artifact.feature_columns.len()
                || class_model.variance_diag.len() != artifact.feature_columns.len()
                || class_model.weights.iter().any(|value| !value.is_finite())
                || !class_model.bias.is_finite()
                || class_model
                    .variance_diag
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                || !class_model.bias_variance.is_finite()
                || class_model.bias_variance <= 0.0
        })
    {
        bail!("bayesian artifact contains invalid posterior parameters");
    }

    Ok(())
}

pub struct BayesianLogitExpert {
    model: Option<BayesianOneVsRestArtifact>,
    pub prior_precision: f32,
    pub learning_rate: f32,
    pub epochs: usize,
}

impl BayesianLogitExpert {
    pub fn new() -> Self {
        Self {
            model: None,
            prior_precision: 0.05,
            learning_rate: 0.05,
            epochs: 250,
        }
    }

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        match &self.model {
            Some(m) => &m.feature_columns,
            None => &[],
        }
    }
}

impl Default for BayesianLogitExpert {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for BayesianLogitExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let (prior_precision, learning_rate, epochs) =
            normalize_bayesian_params(self.prior_precision, self.learning_rate, self.epochs)?;
        let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
        let rows = features.nrows();
        if y.len() != rows {
            bail!(
                "bayes_logit requires matching feature and label rows: {} features vs {} labels",
                rows,
                y.len()
            );
        }
        let labels = remap_three_class_labels(y)?;
        let (train_indices, val_indices) = split_train_val_indices(rows);
        let train_labels = train_indices
            .iter()
            .map(|idx| labels[*idx])
            .collect::<Vec<_>>();
        let val_labels = val_indices
            .iter()
            .map(|idx| labels[*idx] as f32)
            .collect::<Vec<_>>();
        let train_features = features.select(Axis(0), &train_indices);
        let val_features = if val_indices.is_empty() {
            None
        } else {
            Some(features.select(Axis(0), &val_indices))
        };
        let scaler = FeatureScaler::fit(&train_features)?;
        let train_features = scaler.transform(&train_features)?;
        let val_features = if let Some(val_features) = val_features {
            Some(scaler.transform(&val_features)?)
        } else {
            None
        };

        let mut classes = Vec::with_capacity(3);
        for class_idx in 0..3usize {
            let binary = train_labels
                .iter()
                .map(|label| if *label == class_idx { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let val_binary = if val_labels.is_empty() {
                None
            } else {
                Some(
                    val_labels
                        .iter()
                        .map(|label| {
                            if *label as usize == class_idx {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            };
            classes.push(fit_binary_posterior(
                &train_features,
                &binary,
                val_features.as_ref(),
                val_binary.as_deref(),
                prior_precision,
                learning_rate,
                epochs,
            )?);
        }

        let runtime_metadata = runtime_metadata(
            "bayes_logit",
            feature_columns.clone(),
            rows,
            train_labels.len(),
            val_indices.len(),
        )?;
        self.model = Some(BayesianOneVsRestArtifact {
            model_name: "bayes_logit".to_string(),
            feature_columns,
            dataset_rows: rows,
            scaler,
            runtime_metadata: Some(runtime_metadata),
            prior_precision,
            learning_rate,
            epochs,
            classes,
        });
        self.prior_precision = prior_precision;
        self.learning_rate = learning_rate;
        self.epochs = epochs;
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let artifact = self
            .model
            .as_ref()
            .context("BayesianLogitExpert not trained")?;
        ensure_feature_columns_match(&artifact.feature_columns, x)?;
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let features = artifact.scaler.transform(&features)?;
        let mut logits = Vec::with_capacity(features.nrows() * 3);

        for row in 0..features.nrows() {
            let row_values = features.row(row).to_vec();
            for class_model in &artifact.classes {
                logits.push(predictive_logit(class_model, &row_values)?);
            }
        }

        Ok(softmax_rows(
            &Array2::from_shape_vec((features.nrows(), 3), logits)
                .context("shape bayesian probabilities")?,
        ))
    }

    fn save(&self, path: &Path) -> Result<()> {
        let artifact = self
            .model
            .as_ref()
            .context("BayesianLogitExpert not trained")?;
        validate_bayesian_artifact(artifact)?;
        let runtime_metadata = artifact
            .runtime_metadata
            .as_ref()
            .context("BayesianLogitExpert artifact missing runtime metadata")?;
        validate_runtime_metadata(
            runtime_metadata,
            "bayes_logit",
            &artifact.feature_columns,
            artifact.dataset_rows,
        )?;
        with_staged_bayesian_artifact_dir(path, |staged_path| {
            write_json(&staged_path.join(MODEL_FILE_NAME), artifact)?;
            write_json(&staged_path.join(METADATA_FILE_NAME), &runtime_metadata)
        })
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let mut artifact: BayesianOneVsRestArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        validate_bayesian_artifact(&artifact)?;
        let runtime_metadata = resolve_runtime_metadata_from_artifact(path, &artifact)?;
        artifact.runtime_metadata = Some(runtime_metadata);
        self.prior_precision = artifact.prior_precision;
        self.learning_rate = artifact.learning_rate;
        self.epochs = artifact.epochs;
        self.model = Some(artifact);
        Ok(())
    }
}

impl BayesianLogitExpert {
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let artifact = self
            .model
            .as_ref()
            .context("BayesianLogitExpert not trained")?;
        let runtime_metadata = artifact
            .runtime_metadata
            .as_ref()
            .context("BayesianLogitExpert runtime metadata missing")?;
        validate_runtime_metadata(
            runtime_metadata,
            &artifact.model_name,
            &artifact.feature_columns,
            artifact.dataset_rows,
        )?;
        let probabilities = self.predict_proba(x)?;
        runtime_predictions(&artifact.model_name, &probabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::three_class_runtime_confidence;

    fn sample_dataframe() -> DataFrame {
        DataFrame::new(vec![
            Series::new("open".into(), vec![1.0_f64, 1.1, 1.2, 1.3, 1.4, 1.5]).into(),
            Series::new("high".into(), vec![1.2_f64, 1.3, 1.4, 1.5, 1.6, 1.7]).into(),
            Series::new("low".into(), vec![0.9_f64, 1.0, 1.1, 1.2, 1.3, 1.4]).into(),
            Series::new("close".into(), vec![1.05_f64, 1.15, 1.25, 1.35, 1.45, 1.55]).into(),
        ])
        .expect("sample dataframe")
    }

    fn sample_labels() -> Series {
        Series::new("label".into(), vec![-1_i32, 0, 1, -1, 0, 1])
    }

    #[test]
    fn bayesian_logit_rejects_label_row_mismatch() {
        let df = sample_dataframe();
        let y = Series::new("label".into(), vec![-1_i32, 0, 1]);
        let mut model = BayesianLogitExpert::new();

        let err = model
            .fit(&df, &y)
            .expect_err("mismatched labels should fail");
        assert!(err.to_string().contains("matching feature and label rows"));
    }

    #[test]
    fn runtime_predictions_use_shared_three_class_confidence_gate() -> Result<()> {
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.58_f32, 0.20, 0.22])?;
        let predictions = runtime_predictions("bayes_logit", &probabilities)?;
        let prediction = predictions
            .first()
            .expect("one runtime prediction should be produced");
        let (expected_confidence, expected_abstain) =
            three_class_runtime_confidence([0.58, 0.20, 0.22])?;

        assert!((prediction.confidence().expect("confidence") - expected_confidence).abs() < 1e-6);
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        Ok(())
    }

    #[test]
    fn runtime_predictions_persist_bayesian_backend_details() -> Result<()> {
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.58_f32, 0.20, 0.22])?;
        let prediction = runtime_predictions("bayes_logit", &probabilities)?
            .into_iter()
            .next()
            .expect("one runtime prediction");

        assert_eq!(
            prediction.metadata().execution_backend.as_deref(),
            Some("bayes_logit_bayesian_ovr_cpu")
        );
        Ok(())
    }

    #[test]
    fn split_train_val_indices_leaves_temporal_embargo_gap() {
        let (train, val) = split_train_val_indices(50);
        assert!(!val.is_empty(), "validation split should be present");
        let last_train = *train.last().expect("train rows");
        let first_val = *val.first().expect("val rows");
        assert!(
            first_val > last_train + 1,
            "expected embargo gap between train and val"
        );
    }

    #[test]
    fn validate_bayesian_artifact_rejects_missing_runtime_metadata() {
        let artifact = BayesianOneVsRestArtifact {
            model_name: "bayes_logit".to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            runtime_metadata: None,
            prior_precision: 0.05,
            learning_rate: 0.05,
            epochs: 32,
            classes: vec![
                BayesianClassPosterior {
                    weights: Array1::zeros(2),
                    bias: 0.0,
                    variance_diag: Array1::ones(2),
                    bias_variance: 1.0,
                };
                3
            ],
        };

        let err = validate_bayesian_artifact(&artifact)
            .expect_err("artifact without runtime metadata should fail");
        assert!(err.to_string().contains("runtime metadata"));
    }

    #[test]
    fn validate_bayesian_artifact_rejects_non_positive_scaler_stds() {
        let artifact = BayesianOneVsRestArtifact {
            model_name: "bayes_logit".to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 0.0],
            },
            runtime_metadata: Some(
                runtime_metadata(
                    "bayes_logit",
                    vec!["f1".to_string(), "f2".to_string()],
                    8,
                    6,
                    2,
                )
                .expect("build metadata"),
            ),
            prior_precision: 0.05,
            learning_rate: 0.05,
            epochs: 32,
            classes: vec![
                BayesianClassPosterior {
                    weights: Array1::zeros(2),
                    bias: 0.0,
                    variance_diag: Array1::ones(2),
                    bias_variance: 1.0,
                };
                3
            ],
        };

        let err = validate_bayesian_artifact(&artifact)
            .expect_err("artifact with non-positive scaler std should fail");
        assert!(err.to_string().contains("invalid posterior parameters"));
    }

    #[test]
    fn validate_runtime_metadata_rejects_zero_train_rows() {
        let metadata = RuntimeArtifactMetadata {
            model_name: "bayes_logit".to_string(),
            family: ModelFamily::Meta,
            state: CapabilityState::Implemented,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            label_mapping: canonical_three_class_label_mapping(),
            training_summary: TrainingSummaryMetadata::raw_for_validation(8, 0, 8),
        };

        let err = validate_runtime_metadata(
            &metadata,
            "bayes_logit",
            &["f1".to_string(), "f2".to_string()],
            8,
        )
        .expect_err("zero train rows must fail");
        assert!(err.to_string().contains("training rows must be non-zero"));
    }

    #[test]
    fn bayesian_load_uses_embedded_runtime_metadata_when_metadata_file_missing() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let y = sample_labels();
        let mut model = BayesianLogitExpert::new();
        model.fit(&df, &y)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("forex-models-bayes-embed-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;
        std::fs::remove_file(artifact_dir.join(METADATA_FILE_NAME)).expect("remove metadata file");

        let mut reloaded = BayesianLogitExpert::new();
        reloaded.load(&artifact_dir)?;
        assert!(reloaded.model.is_some());

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn bayesian_load_rejects_sidecar_drift_against_embedded_metadata() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let y = sample_labels();
        let mut model = BayesianLogitExpert::new();
        model.fit(&df, &y)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("forex-models-bayes-drift-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;
        let metadata_path = artifact_dir.join(METADATA_FILE_NAME);
        let mut drifted: RuntimeArtifactMetadata =
            read_json(&metadata_path).expect("read saved metadata");
        drifted.training_summary.dataset_rows += 1;
        write_json(&metadata_path, &drifted).expect("write drifted metadata");

        let mut reloaded = BayesianLogitExpert::new();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("drifted sidecar metadata should fail load");
        assert!(err.to_string().contains("sidecar mismatch"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }
}
