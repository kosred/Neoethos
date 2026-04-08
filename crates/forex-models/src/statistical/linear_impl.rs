use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction, canonical_three_class_label_mapping,
    three_class_runtime_confidence, ExpertModel,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

use super::common::{
    ensure_feature_columns_match, feature_matrix_from_dataframe, read_json,
    remap_three_class_labels, softmax_rows, write_json, FeatureScaler, METADATA_FILE_NAME,
    MODEL_FILE_NAME,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearSoftmaxArtifact {
    weights: Array2<f32>,
    bias: Array1<f32>,
    scaler: FeatureScaler,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
    alpha: f32,
    l1_ratio: f32,
    learning_rate: f32,
    epochs: usize,
    model_name: String,
}

fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn split_train_val_indices(rows: usize) -> (Vec<usize>, Vec<usize>) {
    if rows <= 4 {
        return ((0..rows).collect(), Vec::new());
    }

    let val_rows = ((rows as f32) * 0.2).round() as usize;
    let val_rows = val_rows.clamp(1, rows.saturating_sub(1));
    let train_rows = rows - val_rows;

    let train = (0..train_rows).collect::<Vec<_>>();
    let val = (train_rows..rows).collect::<Vec<_>>();
    (train, val)
}

fn logits_from_features(
    features: &Array2<f32>,
    weights: &Array2<f32>,
    bias: &Array1<f32>,
) -> Result<Array2<f32>> {
    if features.ncols() != weights.nrows() {
        bail!(
            "feature dimension mismatch: {} features vs {} weights",
            features.ncols(),
            weights.nrows()
        );
    }
    if weights.ncols() != bias.len() {
        bail!(
            "class dimension mismatch: {} weights cols vs {} bias terms",
            weights.ncols(),
            bias.len()
        );
    }

    let mut logits = features.dot(weights);
    for row in 0..logits.nrows() {
        for class_idx in 0..bias.len() {
            logits[(row, class_idx)] += bias[class_idx];
        }
    }
    if logits.iter().any(|value| !value.is_finite()) {
        bail!("linear model produced non-finite logits");
    }

    Ok(logits)
}

fn cross_entropy_loss(probabilities: &Array2<f32>, labels: &[usize]) -> Result<f32> {
    if probabilities.nrows() != labels.len() {
        bail!(
            "validation label mismatch: {} rows vs {} labels",
            probabilities.nrows(),
            labels.len()
        );
    }

    let mut loss = 0.0_f32;
    for (row_idx, class_idx) in labels.iter().copied().enumerate() {
        let probability = probabilities[(row_idx, class_idx)].clamp(1e-6, 1.0 - 1e-6);
        loss -= probability.ln();
    }

    Ok(loss / labels.len().max(1) as f32)
}

fn runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    train_rows: usize,
    val_rows: usize,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, train_rows, val_rows),
    )
}

fn runtime_predictions(
    model_name: &str,
    probabilities: &Array2<f32>,
) -> Result<Vec<RuntimePrediction>> {
    let mut predictions = Vec::with_capacity(probabilities.nrows());
    for row in probabilities.outer_iter() {
        let row_values = [row[0], row[1], row[2]];
        let (confidence, abstain_recommended) = three_class_runtime_confidence(row_values)?;
        predictions.push(build_runtime_prediction(
            model_name.to_string(),
            ModelFamily::Meta,
            CapabilityState::Implemented,
            row_values,
            Some(confidence),
            Some(abstain_recommended),
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

fn validate_linear_artifact(artifact: &LinearSoftmaxArtifact) -> Result<()> {
    if artifact.model_name != "elasticnet" && artifact.model_name != "logistic" {
        bail!(
            "unexpected linear artifact model name {}",
            artifact.model_name
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("linear artifact must contain at least one feature column");
    }
    if artifact.weights.nrows() != artifact.feature_columns.len() {
        bail!(
            "linear artifact feature-column mismatch: {} weights rows vs {} feature columns",
            artifact.weights.nrows(),
            artifact.feature_columns.len()
        );
    }
    if artifact.weights.ncols() != 3 || artifact.bias.len() != 3 {
        bail!(
            "linear artifact must persist exactly three classes, found {} weight columns and {} bias terms",
            artifact.weights.ncols(),
            artifact.bias.len()
        );
    }
    if artifact.scaler.means.len() != artifact.feature_columns.len()
        || artifact.scaler.stds.len() != artifact.feature_columns.len()
    {
        bail!(
            "linear artifact scaler dimension mismatch: means {}, stds {}, features {}",
            artifact.scaler.means.len(),
            artifact.scaler.stds.len(),
            artifact.feature_columns.len()
        );
    }
    if artifact.weights.iter().any(|value| !value.is_finite())
        || artifact.bias.iter().any(|value| !value.is_finite())
        || artifact.scaler.means.iter().any(|value| !value.is_finite())
        || artifact.scaler.stds.iter().any(|value| !value.is_finite())
    {
        bail!("linear artifact contains non-finite parameters");
    }
    if artifact.dataset_rows == 0 {
        bail!("linear artifact must persist a non-zero dataset row count");
    }

    Ok(())
}

fn fit_linear_softmax(
    model_name: &str,
    x: &DataFrame,
    y: &Series,
    alpha: f32,
    l1_ratio: f32,
    learning_rate: f32,
    epochs: usize,
) -> Result<LinearSoftmaxArtifact> {
    let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
    let rows = features.nrows();
    let cols = features.ncols();
    let n_classes = 3usize;

    if y.len() != rows {
        bail!(
            "{model_name} requires matching feature and label rows: {} features vs {} labels",
            rows,
            y.len()
        );
    }

    let labels = remap_three_class_labels(y)?;

    if rows == 0 || cols == 0 {
        bail!("{model_name} requires a non-empty feature matrix");
    }

    let (train_indices, val_indices) = split_train_val_indices(rows);
    let train_labels = train_indices
        .iter()
        .map(|idx| labels[*idx])
        .collect::<Vec<_>>();
    let val_labels = val_indices
        .iter()
        .map(|idx| labels[*idx])
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

    let mut weights = Array2::<f32>::zeros((cols, n_classes));
    let mut bias = Array1::<f32>::zeros(n_classes);
    let lr = learning_rate.max(1e-5);
    let regularization = alpha.max(0.0);
    let mut best_weights = weights.clone();
    let mut best_bias = bias.clone();
    let mut best_val_loss = f32::INFINITY;
    let mut stale_epochs = 0usize;
    let patience = 25usize;

    for _ in 0..epochs.max(1) {
        let logits = logits_from_features(&train_features, &weights, &bias)?;
        let probabilities = softmax_rows(&logits);
        let mut error = probabilities;
        for (row_idx, class_idx) in train_labels.iter().copied().enumerate() {
            error[(row_idx, class_idx)] -= 1.0;
        }

        let mut grad_w = train_features.t().dot(&error) / train_features.nrows() as f32;
        let grad_b = error.sum_axis(Axis(0)) / train_features.nrows() as f32;

        if regularization > 0.0 {
            for feature_idx in 0..cols {
                for class_idx in 0..n_classes {
                    let weight = weights[(feature_idx, class_idx)];
                    let l2 = (1.0 - l1_ratio.clamp(0.0, 1.0)) * weight;
                    let l1 = l1_ratio.clamp(0.0, 1.0) * sign(weight);
                    grad_w[(feature_idx, class_idx)] += regularization * (l2 + l1);
                }
            }
        }

        for feature_idx in 0..cols {
            for class_idx in 0..n_classes {
                weights[(feature_idx, class_idx)] -= lr * grad_w[(feature_idx, class_idx)];
            }
        }
        for class_idx in 0..n_classes {
            bias[class_idx] -= lr * grad_b[class_idx];
        }

        if let Some(val_features) = val_features.as_ref() {
            let val_logits = logits_from_features(val_features, &weights, &bias)?;
            let val_probabilities = softmax_rows(&val_logits);
            let val_loss = cross_entropy_loss(&val_probabilities, &val_labels)?;
            if val_loss + 1e-6 < best_val_loss {
                best_val_loss = val_loss;
                best_weights = weights.clone();
                best_bias = bias.clone();
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

    let train_rows = train_labels.len();
    let val_rows = val_labels.len();
    let runtime_metadata = runtime_metadata(
        model_name,
        feature_columns.clone(),
        rows,
        train_rows,
        val_rows,
    );

    Ok(LinearSoftmaxArtifact {
        weights,
        bias,
        scaler,
        feature_columns,
        dataset_rows: rows,
        runtime_metadata: Some(runtime_metadata),
        alpha,
        l1_ratio,
        learning_rate,
        epochs,
        model_name: model_name.to_string(),
    })
}

fn predict_linear_softmax(artifact: &LinearSoftmaxArtifact, x: &DataFrame) -> Result<Array2<f32>> {
    ensure_feature_columns_match(&artifact.feature_columns, x)?;
    let (features, _) = feature_matrix_from_dataframe(x)?;
    let features = artifact.scaler.transform(&features)?;
    let logits = logits_from_features(&features, &artifact.weights, &artifact.bias)?;
    Ok(softmax_rows(&logits))
}

pub struct ElasticNetExpert {
    model: Option<LinearSoftmaxArtifact>,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub learning_rate: f32,
    pub epochs: usize,
}

impl ElasticNetExpert {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            model: None,
            alpha,
            l1_ratio,
            learning_rate: 0.05,
            epochs: 300,
        }
    }

    pub fn ranked_feature_importance(&self) -> Result<Vec<(String, f32)>> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;

        let mut ranked = model
            .feature_columns
            .iter()
            .enumerate()
            .map(|(feature_idx, name)| {
                let importance = model
                    .weights
                    .row(feature_idx)
                    .iter()
                    .map(|weight| weight.abs())
                    .sum::<f32>()
                    / model.weights.ncols().max(1) as f32;
                (name.clone(), importance)
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        Ok(ranked)
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;
        let probabilities = predict_linear_softmax(model, x)?;
        runtime_predictions(&model.model_name, &probabilities)
    }
}

impl ExpertModel for ElasticNetExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        self.model = Some(fit_linear_softmax(
            "elasticnet",
            x,
            y,
            self.alpha as f32,
            self.l1_ratio as f32,
            self.learning_rate,
            self.epochs,
        )?);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;
        predict_linear_softmax(model, x)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .context("ElasticNetExpert not trained")?;
        validate_linear_artifact(model)?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("ElasticNetExpert artifact missing runtime metadata")?;
        validate_runtime_metadata(
            runtime_metadata,
            "elasticnet",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        write_json(&path.join(MODEL_FILE_NAME), model)?;
        write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let mut model: LinearSoftmaxArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if model.model_name != "elasticnet" {
            bail!("expected elasticnet artifact, got {}", model.model_name);
        }
        validate_linear_artifact(&model)?;
        let runtime_metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_runtime_metadata(
            &runtime_metadata,
            "elasticnet",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        if model.runtime_metadata.as_ref() != Some(&runtime_metadata) {
            bail!("runtime metadata file does not match elasticnet artifact");
        }
        model.runtime_metadata = Some(runtime_metadata);
        self.alpha = model.alpha as f64;
        self.l1_ratio = model.l1_ratio as f64;
        self.learning_rate = model.learning_rate;
        self.epochs = model.epochs;
        self.model = Some(model);
        Ok(())
    }
}

pub struct LogisticExpert {
    model: Option<LinearSoftmaxArtifact>,
    pub alpha: f32,
    pub learning_rate: f32,
    pub epochs: usize,
}

impl LogisticExpert {
    pub fn new() -> Self {
        Self {
            model: None,
            alpha: 0.01,
            learning_rate: 0.05,
            epochs: 250,
        }
    }
}

impl Default for LogisticExpert {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for LogisticExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        self.model = Some(fit_linear_softmax(
            "logistic",
            x,
            y,
            self.alpha,
            0.0,
            self.learning_rate,
            self.epochs,
        )?);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let model = self.model.as_ref().context("LogisticExpert not trained")?;
        predict_linear_softmax(model, x)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model = self.model.as_ref().context("LogisticExpert not trained")?;
        validate_linear_artifact(model)?;
        let runtime_metadata = model
            .runtime_metadata
            .as_ref()
            .context("LogisticExpert artifact missing runtime metadata")?;
        validate_runtime_metadata(
            runtime_metadata,
            "logistic",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        write_json(&path.join(MODEL_FILE_NAME), model)?;
        write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let mut model: LinearSoftmaxArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        if model.model_name != "logistic" {
            bail!("expected logistic artifact, got {}", model.model_name);
        }
        validate_linear_artifact(&model)?;
        let runtime_metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        validate_runtime_metadata(
            &runtime_metadata,
            "logistic",
            &model.feature_columns,
            model.dataset_rows,
        )?;
        if model.runtime_metadata.as_ref() != Some(&runtime_metadata) {
            bail!("runtime metadata file does not match logistic artifact");
        }
        model.runtime_metadata = Some(runtime_metadata);
        self.alpha = model.alpha;
        self.learning_rate = model.learning_rate;
        self.epochs = model.epochs;
        self.model = Some(model);
        Ok(())
    }
}

impl LogisticExpert {
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let model = self.model.as_ref().context("LogisticExpert not trained")?;
        let probabilities = predict_linear_softmax(model, x)?;
        runtime_predictions(&model.model_name, &probabilities)
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
    fn logistic_expert_rejects_label_row_mismatch() {
        let df = sample_dataframe();
        let y = Series::new("label".into(), vec![-1_i32, 0, 1]);
        let mut model = LogisticExpert::new();

        let err = model
            .fit(&df, &y)
            .expect_err("mismatched labels should fail");
        assert!(err.to_string().contains("matching feature and label rows"));
    }

    #[test]
    fn logistic_expert_trains_and_persists_runtime_metadata() -> Result<()> {
        let df = sample_dataframe();
        let y = sample_labels();
        let mut model = LogisticExpert::new();

        model.fit(&df, &y)?;

        let artifact = model.model.as_ref().expect("trained model");
        let metadata = artifact
            .runtime_metadata
            .as_ref()
            .expect("runtime metadata to be recorded");

        assert_eq!(metadata.model_name, "logistic");
        assert_eq!(metadata.family, ModelFamily::Meta);
        assert_eq!(metadata.state, CapabilityState::Implemented);
        assert_eq!(metadata.training_summary.dataset_rows, 6);
        assert_eq!(
            metadata.training_summary.train_rows + metadata.training_summary.val_rows,
            6
        );

        let runtime_predictions = model.predict_runtime(&df)?;
        assert_eq!(runtime_predictions.len(), 6);
        Ok(())
    }

    #[test]
    fn elasticnet_runtime_predictions_validate_probability_contract() -> Result<()> {
        let df = sample_dataframe();
        let y = sample_labels();
        let mut model = ElasticNetExpert::new(0.01, 0.5);
        model.fit(&df, &y)?;

        let probabilities = model.predict_proba(&df)?;
        assert_eq!(probabilities.ncols(), 3);

        let runtime_predictions = model.predict_runtime(&df)?;
        assert_eq!(runtime_predictions.len(), 6);
        Ok(())
    }

    #[test]
    fn runtime_predictions_use_shared_three_class_confidence_gate() -> Result<()> {
        let probabilities = Array2::from_shape_vec((1, 3), vec![0.58_f32, 0.20, 0.22])?;
        let predictions = runtime_predictions("logistic", &probabilities)?;
        let prediction = predictions
            .first()
            .expect("one runtime prediction should be produced");
        let (expected_confidence, expected_abstain) =
            three_class_runtime_confidence([0.58, 0.20, 0.22])?;

        assert!(
            (prediction.confidence().expect("confidence") - expected_confidence).abs() < 1e-6
        );
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        Ok(())
    }

    #[test]
    fn logistic_expert_rejects_tampered_metadata_on_load() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let y = sample_labels();
        let mut model = LogisticExpert::new();
        model.fit(&df, &y)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("forex-models-logistic-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;

        let metadata_path = artifact_dir.join(METADATA_FILE_NAME);
        let mut metadata: RuntimeArtifactMetadata =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("parse metadata");
        metadata.model_name = "tampered".to_string();
        std::fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write tampered metadata");

        let mut reloaded = LogisticExpert::new();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("tampered metadata should fail");
        assert!(err.to_string().contains("runtime metadata"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }
}
