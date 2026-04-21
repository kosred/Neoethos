use anyhow::{Context, Result, bail};
#[cfg(feature = "anomaly-detection")]
use extended_isolation_forest::{Forest, ForestOptions};
use ndarray::Array2;
use polars::prelude::*;
use seq_macro::seq;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, append_runtime_degraded_reason, gpu_policy_cpu_fallback_reason,
};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    METADATA_FILE_NAME, MODEL_FILE_NAME, ensure_feature_columns_match,
    feature_matrix_from_dataframe, read_json, write_json,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationForestArtifact {
    model_name: String,
    #[serde(default)]
    backend_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    n_trees: usize,
    sample_size: usize,
    extension_level: usize,
    max_tree_depth: Option<usize>,
    anomaly_threshold: f32,
    score_mean: f32,
    score_std: f32,
    #[serde(default)]
    score_median: f32,
    #[serde(default = "default_score_mad")]
    score_mad: f32,
    model_json: String,
    #[serde(default)]
    fallback_means: Vec<f32>,
    #[serde(default)]
    fallback_stds: Vec<f32>,
}

fn default_score_mad() -> f32 {
    1.0
}

#[cfg(feature = "anomaly-detection")]
trait ForestBackend: Send + Sync {
    fn score_row(&self, values: &[f64]) -> Result<f64>;
    fn to_json(&self) -> Result<String>;
}

#[cfg(feature = "anomaly-detection")]
struct ForestBackendImpl<const N: usize> {
    forest: Forest<f64, N>,
}

#[cfg(feature = "anomaly-detection")]
impl<const N: usize> ForestBackend for ForestBackendImpl<N> {
    fn score_row(&self, values: &[f64]) -> Result<f64> {
        if values.len() != N {
            bail!(
                "extended isolation forest expected {} features, received {}",
                N,
                values.len()
            );
        }

        let row = std::array::from_fn(|idx| values[idx]);
        Ok(self.forest.score(&row))
    }

    fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.forest).context("serialize extended isolation forest")
    }
}

#[cfg(feature = "anomaly-detection")]
fn build_forest_backend<const N: usize>(
    rows: &[Vec<f64>],
    options: &ForestOptions,
) -> Result<Box<dyn ForestBackend>> {
    let training_rows = rows
        .iter()
        .map(|row| {
            if row.len() != N {
                bail!(
                    "extended isolation forest expected {} columns, got {}",
                    N,
                    row.len()
                );
            }
            Ok(std::array::from_fn(|idx| row[idx]))
        })
        .collect::<Result<Vec<[f64; N]>>>()?;

    let forest = Forest::from_slice(training_rows.as_slice(), options)
        .map_err(|err| anyhow::anyhow!("build extended isolation forest: {err}"))?;
    Ok(Box::new(ForestBackendImpl::<N> { forest }))
}

#[cfg(feature = "anomaly-detection")]
fn load_forest_backend<const N: usize>(payload: &str) -> Result<Box<dyn ForestBackend>> {
    let forest: Forest<f64, N> =
        serde_json::from_str(payload).context("deserialize extended isolation forest")?;
    Ok(Box::new(ForestBackendImpl::<N> { forest }))
}

#[cfg(feature = "anomaly-detection")]
fn dispatch_forest_builder(
    feature_count: usize,
    rows: &[Vec<f64>],
    options: &ForestOptions,
) -> Result<Box<dyn ForestBackend>> {
    seq!(N in 1..=128 {
        match feature_count {
            #(N => build_forest_backend::<N>(rows, options),)*
            _ => bail!(
                "extended isolation forest currently supports 1..=128 feature columns, got {}",
                feature_count
            ),
        }
    })
}

#[cfg(feature = "anomaly-detection")]
fn dispatch_forest_loader(feature_count: usize, payload: &str) -> Result<Box<dyn ForestBackend>> {
    seq!(N in 1..=128 {
        match feature_count {
            #(N => load_forest_backend::<N>(payload),)*
            _ => bail!(
                "extended isolation forest currently supports 1..=128 feature columns, got {}",
                feature_count
            ),
        }
    })
}

fn anomaly_runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Anomaly,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn validate_runtime_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if metadata.family != ModelFamily::Anomaly {
        bail!(
            "runtime metadata mismatch for isolation_forest: expected family {:?}, got {:?}",
            ModelFamily::Anomaly,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "runtime metadata mismatch for isolation_forest: expected state {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("runtime metadata mismatch for isolation_forest: label mapping mismatch");
    }
    if expected_feature_columns.is_empty() {
        bail!("persisted isolation_forest artifact is missing feature columns");
    }
    if metadata.model_name != "isolation_forest" {
        bail!(
            "runtime metadata mismatch for isolation_forest: expected model name isolation_forest, got {}",
            metadata.model_name
        );
    }
    if metadata.feature_columns != expected_feature_columns {
        bail!(
            "runtime metadata mismatch for isolation_forest: expected feature columns {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        bail!(
            "runtime metadata mismatch for isolation_forest: expected {} dataset rows, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        bail!(
            "runtime metadata mismatch for isolation_forest: training rows {} + validation rows {} must equal dataset rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }

    Ok(())
}

fn resolve_runtime_metadata_from_artifact(
    path: &Path,
    artifact: &IsolationForestArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_runtime_metadata(&metadata, &artifact.feature_columns, artifact.dataset_rows)?;
            if let Some(embedded) = artifact.runtime_metadata.as_ref() {
                if embedded.model_name != metadata.model_name
                    || embedded.family != metadata.family
                    || embedded.state != metadata.state
                    || embedded.feature_columns != metadata.feature_columns
                    || embedded.label_mapping != metadata.label_mapping
                    || embedded.training_summary.dataset_rows
                        != metadata.training_summary.dataset_rows
                    || embedded.training_summary.train_rows != metadata.training_summary.train_rows
                    || embedded.training_summary.val_rows != metadata.training_summary.val_rows
                {
                    bail!(
                        "runtime metadata sidecar mismatch with embedded isolation_forest metadata at {}",
                        metadata_path.display()
                    );
                }
            }
            Ok(metadata)
        }
        Err(file_err) => {
            let fallback = artifact
                .runtime_metadata
                .clone()
                .with_context(|| format!("missing runtime metadata file {} and isolation artifact has no embedded metadata: {file_err}", metadata_path.display()))?;
            validate_runtime_metadata(&fallback, &artifact.feature_columns, artifact.dataset_rows)?;
            tracing::warn!(
                path = %metadata_path.display(),
                error = %file_err,
                "isolation_forest metadata sidecar missing/unreadable; using embedded runtime metadata"
            );
            Ok(fallback)
        }
    }
}

fn validate_isolation_forest_artifact(artifact: &IsolationForestArtifact) -> Result<()> {
    if artifact.feature_columns.is_empty() {
        bail!("isolation_forest artifact must contain feature columns");
    }
    let runtime_metadata = artifact
        .runtime_metadata
        .as_ref()
        .context("isolation_forest artifact must persist runtime metadata")?;
    validate_runtime_metadata(
        runtime_metadata,
        &artifact.feature_columns,
        artifact.dataset_rows,
    )?;
    if artifact.backend_kind.trim().is_empty() {
        bail!("isolation_forest artifact must declare a backend kind");
    }
    if artifact.dataset_rows == 0 {
        bail!("isolation_forest artifact must contain at least one training row");
    }
    if !artifact.anomaly_threshold.is_finite() || artifact.anomaly_threshold < 0.0 {
        bail!("isolation_forest anomaly_threshold must be finite and non-negative");
    }
    if !artifact.score_mean.is_finite()
        || !artifact.score_std.is_finite()
        || artifact.score_std <= 0.0
        || !artifact.score_median.is_finite()
        || !artifact.score_mad.is_finite()
        || artifact.score_mad <= 0.0
    {
        bail!(
            "isolation_forest score statistics must be finite and score_std/score_mad must be positive"
        );
    }
    if artifact
        .fallback_means
        .iter()
        .chain(artifact.fallback_stds.iter())
        .any(|value| !value.is_finite())
    {
        bail!("isolation_forest fallback statistics must be finite");
    }
    if artifact.fallback_stds.iter().any(|value| *value <= 0.0) {
        bail!("isolation_forest fallback scales must be positive");
    }
    if artifact.fallback_means.len() != artifact.feature_columns.len()
        || artifact.fallback_stds.len() != artifact.feature_columns.len()
    {
        bail!(
            "isolation_forest fallback statistics mismatch: expected {} features, received means {} and stds {}",
            artifact.feature_columns.len(),
            artifact.fallback_means.len(),
            artifact.fallback_stds.len()
        );
    }
    match artifact.backend_kind.as_str() {
        "extended_isolation_forest" => {
            if artifact.model_json.trim().is_empty() {
                bail!("extended_isolation_forest artifact must contain serialized backend payload");
            }
        }
        "diagonal_profile" => {
            if !artifact.model_json.trim().is_empty() {
                bail!("diagonal_profile artifact must not carry serialized backend payload");
            }
        }
        other => bail!("unsupported isolation forest backend kind: {other}"),
    }
    Ok(())
}

fn validate_probability(value: f32) -> Result<f32> {
    if !value.is_finite() {
        bail!("isolation_forest probability projection produced a non-finite value");
    }
    if !(0.0..=1.0).contains(&value) {
        bail!(
            "isolation_forest probability projection produced an out-of-range value {}",
            value
        );
    }
    Ok(value)
}

fn quantile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.5;
    }

    let clamped = fraction.clamp(0.0, 1.0);
    let index = ((values.len().saturating_sub(1) as f32) * clamped).round() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

fn score_statistics(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let std = variance.sqrt();
    let median_value = median(values.to_vec());
    let mad = median(
        values
            .iter()
            .map(|value| (*value - median_value).abs())
            .collect(),
    );
    let std = if std.is_finite() && std > 1e-6 {
        std
    } else {
        1.0
    };
    let mad = if mad.is_finite() && mad > 1e-6 {
        mad
    } else {
        1.0
    };
    (mean, std, median_value, mad)
}

fn anomaly_probabilities(
    scores: &[f32],
    threshold: f32,
    score_std: f32,
    score_median: f32,
    score_mad: f32,
) -> Result<Array2<f32>> {
    let mut probabilities = Vec::with_capacity(scores.len() * 3);
    let robust_scale = (score_mad * 1.4826).max(1e-4);
    let normalizer = robust_scale.max(score_std * 0.1).max(1e-4);

    for score in scores {
        if !score.is_finite() {
            bail!("isolation_forest produced a non-finite anomaly score");
        }
        let centered_score = *score - score_median;
        let adjusted_threshold = threshold - score_median;
        let anomaly_logit = (centered_score - adjusted_threshold) / normalizer;
        let anomaly_probability = validate_probability(1.0 / (1.0 + (-anomaly_logit).exp()))?;
        let directional_probability = (1.0 - anomaly_probability) * 0.5;
        let directional_probability = validate_probability(directional_probability)?;
        probabilities.push(anomaly_probability);
        probabilities.push(directional_probability);
        probabilities.push(directional_probability);
    }

    Array2::from_shape_vec((scores.len(), 3), probabilities).context("shape anomaly probabilities")
}

fn feature_rows(features: &Array2<f32>) -> Vec<Vec<f64>> {
    (0..features.nrows())
        .map(|row_idx| {
            features
                .row(row_idx)
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn fallback_profile(features: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let cols = features.ncols();
    let mut centers = vec![0.0; cols];
    let mut scales = vec![1.0; cols];

    for col in 0..cols {
        let column = (0..features.nrows())
            .map(|row| features[(row, col)])
            .collect::<Vec<_>>();
        let center = median(column.clone());
        centers[col] = center;

        let mut deviations = column
            .into_iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>();
        let mad = median(std::mem::take(&mut deviations));
        let scale = (mad * 1.4826).max(1e-6);
        scales[col] = if scale.is_finite() { scale } else { 1.0 };
    }

    (centers, scales)
}

fn fallback_score_row(values: &[f32], centers: &[f32], scales: &[f32]) -> Result<f32> {
    if values.len() != centers.len() || values.len() != scales.len() {
        bail!(
            "fallback anomaly profile expected {} features, got {}",
            centers.len(),
            values.len()
        );
    }

    let score = values
        .iter()
        .zip(centers.iter().zip(scales.iter()))
        .map(|(value, (center, scale))| {
            let z = (value - center) / scale.max(1e-6);
            z * z
        })
        .sum::<f32>()
        / values.len().max(1) as f32;
    Ok(score)
}

pub struct IsolationForestExpert {
    #[cfg(feature = "anomaly-detection")]
    backend: Option<Box<dyn ForestBackend>>,
    #[cfg(not(feature = "anomaly-detection"))]
    backend: Option<()>,
    pub n_trees: usize,
    pub sample_size: usize,
    pub extension_level: usize,
    pub max_tree_depth: Option<usize>,
    pub backend_kind: String,
    pub feature_columns: Vec<String>,
    pub dataset_rows: usize,
    pub anomaly_threshold: f32,
    pub score_mean: f32,
    pub score_std: f32,
    pub score_median: f32,
    pub score_mad: f32,
    pub fallback_means: Vec<f32>,
    pub fallback_stds: Vec<f32>,
}

impl IsolationForestExpert {
    pub fn new(n_trees: usize, sample_size: usize) -> Self {
        Self {
            backend: None,
            n_trees: n_trees.max(64),
            sample_size: sample_size.max(64),
            extension_level: 0,
            max_tree_depth: None,
            backend_kind: String::new(),
            feature_columns: Vec::new(),
            dataset_rows: 0,
            anomaly_threshold: 0.5,
            score_mean: 0.0,
            score_std: 1.0,
            score_median: 0.0,
            score_mad: 1.0,
            fallback_means: Vec::new(),
            fallback_stds: Vec::new(),
        }
    }

    fn fit_diagonal_profile(
        &mut self,
        features: &Array2<f32>,
        feature_columns: Vec<String>,
    ) -> Result<()> {
        let (means, stds) = fallback_profile(features);
        let mut scores = (0..features.nrows())
            .map(|row_idx| fallback_score_row(&features.row(row_idx).to_vec(), &means, &stds))
            .collect::<Result<Vec<_>>>()?;
        scores.sort_by(|left, right| left.total_cmp(right));

        let (score_mean, score_std, score_median, score_mad) = score_statistics(&scores);
        self.backend = None;
        self.feature_columns = feature_columns;
        self.dataset_rows = features.nrows();
        self.fallback_means = means;
        self.fallback_stds = stds;
        self.backend_kind = "diagonal_profile".to_string();
        self.anomaly_threshold = quantile(&scores, 0.95).max(0.5);
        self.score_mean = score_mean;
        self.score_std = score_std;
        self.score_median = score_median;
        self.score_mad = score_mad;
        Ok(())
    }

    fn artifact(&self) -> Result<IsolationForestArtifact> {
        #[cfg(feature = "anomaly-detection")]
        {
            Ok(IsolationForestArtifact {
                model_name: "isolation_forest".to_string(),
                backend_kind: self.backend_kind.clone(),
                runtime_metadata: Some(anomaly_runtime_metadata(
                    "isolation_forest",
                    self.feature_columns.clone(),
                    self.dataset_rows,
                )?),
                feature_columns: self.feature_columns.clone(),
                dataset_rows: self.dataset_rows,
                n_trees: self.n_trees,
                sample_size: self.sample_size,
                extension_level: self.extension_level,
                max_tree_depth: self.max_tree_depth,
                anomaly_threshold: self.anomaly_threshold,
                score_mean: self.score_mean,
                score_std: self.score_std,
                score_median: self.score_median,
                score_mad: self.score_mad,
                model_json: if self.backend_kind == "diagonal_profile" {
                    String::new()
                } else {
                    self.backend
                        .as_ref()
                        .context("isolation forest backend missing")?
                        .to_json()?
                },
                fallback_means: self.fallback_means.clone(),
                fallback_stds: self.fallback_stds.clone(),
            })
        }

        #[cfg(not(feature = "anomaly-detection"))]
        {
            Ok(IsolationForestArtifact {
                model_name: "isolation_forest".to_string(),
                backend_kind: self.backend_kind.clone(),
                runtime_metadata: Some(anomaly_runtime_metadata(
                    "isolation_forest",
                    self.feature_columns.clone(),
                    self.dataset_rows,
                )?),
                feature_columns: self.feature_columns.clone(),
                dataset_rows: self.dataset_rows,
                n_trees: self.n_trees,
                sample_size: self.sample_size,
                extension_level: self.extension_level,
                max_tree_depth: self.max_tree_depth,
                anomaly_threshold: self.anomaly_threshold,
                score_mean: self.score_mean,
                score_std: self.score_std,
                score_median: self.score_median,
                score_mad: self.score_mad,
                model_json: String::new(),
                fallback_means: self.fallback_means.clone(),
                fallback_stds: self.fallback_stds.clone(),
            })
        }
    }
}

impl Default for IsolationForestExpert {
    fn default() -> Self {
        Self::new(128, 256)
    }
}

impl ExpertModel for IsolationForestExpert {
    fn fit(&mut self, x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(not(feature = "anomaly-detection"))]
        {
            let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
            if features.nrows() < 8 {
                bail!(
                    "isolation forest requires at least 8 rows, received {}",
                    features.nrows()
                );
            }
            if features.ncols() == 0 {
                bail!("isolation forest cannot train with zero feature columns");
            }

            self.fit_diagonal_profile(&features, feature_columns)
        }

        #[cfg(feature = "anomaly-detection")]
        {
            let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
            if features.nrows() < 8 {
                bail!(
                    "isolation forest requires at least 8 rows, received {}",
                    features.nrows()
                );
            }
            if features.ncols() == 0 {
                bail!("isolation forest cannot train with zero feature columns");
            }
            if features.ncols() > 128 {
                return self.fit_diagonal_profile(&features, feature_columns);
            }

            let training_rows = feature_rows(&features);
            let sample_size = self.sample_size.min(training_rows.len()).max(8);
            let extension_level = if self.extension_level == 0 {
                features.ncols().saturating_sub(1)
            } else {
                self.extension_level.min(features.ncols().saturating_sub(1))
            };
            let (means, stds) = fallback_profile(&features);

            let options = ForestOptions {
                n_trees: self.n_trees.max(32),
                sample_size,
                max_tree_depth: self.max_tree_depth,
                extension_level,
            };

            let backend = dispatch_forest_builder(features.ncols(), &training_rows, &options)?;
            let mut training_scores = training_rows
                .iter()
                .map(|row| backend.score_row(row).map(|score| score as f32))
                .collect::<Result<Vec<_>>>()?;
            training_scores.sort_by(|left, right| left.total_cmp(right));

            let (score_mean, score_std, score_median, score_mad) =
                score_statistics(&training_scores);
            self.backend = Some(backend);
            self.n_trees = options.n_trees;
            self.sample_size = options.sample_size;
            self.extension_level = extension_level;
            self.backend_kind = "extended_isolation_forest".to_string();
            self.feature_columns = feature_columns;
            self.dataset_rows = features.nrows();
            self.anomaly_threshold = quantile(&training_scores, 0.95).max(0.5);
            self.score_mean = score_mean;
            self.score_std = score_std;
            self.score_median = score_median;
            self.score_mad = score_mad;
            self.fallback_means = means;
            self.fallback_stds = stds;
            Ok(())
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        ensure_feature_columns_match(&self.feature_columns, x)?;

        #[cfg(not(feature = "anomaly-detection"))]
        {
            let (features, _) = feature_matrix_from_dataframe(x)?;
            let scores = (0..features.nrows())
                .map(|row_idx| {
                    fallback_score_row(
                        &features.row(row_idx).to_vec(),
                        &self.fallback_means,
                        &self.fallback_stds,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            anomaly_probabilities(
                &scores,
                self.anomaly_threshold,
                self.score_std,
                self.score_median,
                self.score_mad,
            )
        }

        #[cfg(feature = "anomaly-detection")]
        {
            let (features, _) = feature_matrix_from_dataframe(x)?;
            let scores = match self.backend_kind.as_str() {
                "extended_isolation_forest" => {
                    let backend = self
                        .backend
                        .as_ref()
                        .context("isolation forest backend missing for extended artifact")?;
                    feature_rows(&features)
                        .iter()
                        .map(|row| backend.score_row(row).map(|score| score as f32))
                        .collect::<Result<Vec<_>>>()?
                }
                "diagonal_profile" => {
                    if self.fallback_means.len() != features.ncols()
                        || self.fallback_stds.len() != features.ncols()
                    {
                        bail!(
                            "diagonal-profile anomaly artifact is missing feature statistics: expected {} columns, got means {} and stds {}",
                            features.ncols(),
                            self.fallback_means.len(),
                            self.fallback_stds.len()
                        );
                    }
                    (0..features.nrows())
                        .map(|row_idx| {
                            fallback_score_row(
                                &features.row(row_idx).to_vec(),
                                &self.fallback_means,
                                &self.fallback_stds,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                other => bail!("unsupported isolation forest backend kind: {other}"),
            };

            anomaly_probabilities(
                &scores,
                self.anomaly_threshold,
                self.score_std,
                self.score_median,
                self.score_mad,
            )
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create isolation forest directory {}", path.display()))?;
        match self.backend_kind.as_str() {
            "extended_isolation_forest" => {
                self.backend
                    .as_ref()
                    .context("extended isolation forest backend missing")?;
            }
            "diagonal_profile" => {
                if self.fallback_means.len() != self.feature_columns.len()
                    || self.fallback_stds.len() != self.feature_columns.len()
                {
                    bail!(
                        "diagonal-profile anomaly artifact is missing feature statistics: expected {} columns, got means {} and stds {}",
                        self.feature_columns.len(),
                        self.fallback_means.len(),
                        self.fallback_stds.len()
                    );
                }
            }
            other => bail!("unsupported isolation forest backend kind: {other}"),
        }
        let runtime_metadata = anomaly_runtime_metadata(
            "isolation_forest",
            self.feature_columns.clone(),
            self.dataset_rows,
        )?;
        validate_runtime_metadata(&runtime_metadata, &self.feature_columns, self.dataset_rows)?;
        write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
        let artifact = self.artifact()?;
        validate_isolation_forest_artifact(&artifact)?;
        if artifact.runtime_metadata.as_ref() != Some(&runtime_metadata) {
            bail!("runtime metadata file does not match isolation_forest artifact");
        }
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: IsolationForestArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        validate_isolation_forest_artifact(&artifact)?;
        if artifact.model_name != "isolation_forest" {
            bail!(
                "expected isolation_forest artifact, got {}",
                artifact.model_name
            );
        }
        resolve_runtime_metadata_from_artifact(path, &artifact)?;

        let mut next_state = Self::new(artifact.n_trees, artifact.sample_size);
        next_state.extension_level = artifact.extension_level;
        next_state.max_tree_depth = artifact.max_tree_depth;
        next_state.backend_kind = artifact.backend_kind;
        next_state.feature_columns = artifact.feature_columns;
        next_state.dataset_rows = artifact.dataset_rows;
        next_state.anomaly_threshold = artifact.anomaly_threshold;
        next_state.score_mean = artifact.score_mean;
        next_state.score_std = artifact.score_std;
        next_state.score_median = artifact.score_median;
        next_state.score_mad = artifact.score_mad;
        next_state.fallback_means = artifact.fallback_means;
        next_state.fallback_stds = artifact.fallback_stds;

        #[cfg(not(feature = "anomaly-detection"))]
        {
            if next_state.backend_kind != "diagonal_profile" {
                bail!(
                    "isolation_forest artifact requires backend `{}` but this build only supports `diagonal_profile`",
                    next_state.backend_kind
                );
            }
            if next_state.fallback_means.len() != next_state.feature_columns.len()
                || next_state.fallback_stds.len() != next_state.feature_columns.len()
            {
                bail!(
                    "diagonal-profile anomaly artifact is missing feature statistics: expected {} columns, got means {} and stds {}",
                    next_state.feature_columns.len(),
                    next_state.fallback_means.len(),
                    next_state.fallback_stds.len()
                );
            }
        }

        #[cfg(feature = "anomaly-detection")]
        {
            next_state.backend = match next_state.backend_kind.as_str() {
                "extended_isolation_forest" => Some(dispatch_forest_loader(
                    next_state.feature_columns.len(),
                    &artifact.model_json,
                )?),
                "diagonal_profile" => {
                    if next_state.fallback_means.len() != next_state.feature_columns.len()
                        || next_state.fallback_stds.len() != next_state.feature_columns.len()
                    {
                        bail!(
                            "diagonal-profile anomaly artifact is missing feature statistics: expected {} columns, got means {} and stds {}",
                            next_state.feature_columns.len(),
                            next_state.fallback_means.len(),
                            next_state.fallback_stds.len()
                        );
                    }
                    None
                }
                other => bail!("unsupported isolation forest backend kind: {other}"),
            };
        }

        *self = next_state;
        Ok(())
    }
}

impl IsolationForestExpert {
    fn has_minimal_runtime_context(&self) -> bool {
        self.dataset_rows > 0 && !self.feature_columns.is_empty()
    }

    fn has_diagonal_profile_runtime(&self) -> bool {
        self.has_minimal_runtime_context()
            && self.fallback_means.len() == self.feature_columns.len()
            && self.fallback_stds.len() == self.feature_columns.len()
            && self
                .fallback_stds
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.fallback_means.iter().all(|value| value.is_finite())
    }

    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let gpu_cpu_fallback = gpu_policy_cpu_fallback_reason("isolation_forest");
        if !self.has_minimal_runtime_context() {
            return (
                Some("isolation_forest_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("anomaly_runtime_state_incomplete".to_string()),
                    gpu_cpu_fallback,
                ),
            );
        }

        match self.backend_kind.as_str() {
            "extended_isolation_forest" => {
                #[cfg(feature = "anomaly-detection")]
                {
                    if self.backend.is_some() {
                        (
                            Some("extended_isolation_forest".to_string()),
                            gpu_cpu_fallback.clone(),
                        )
                    } else {
                        let degraded_reason = if self.has_diagonal_profile_runtime() {
                            "extended_anomaly_backend_missing"
                        } else {
                            "anomaly_runtime_state_incomplete"
                        };
                        (
                            Some("extended_isolation_forest".to_string()),
                            append_runtime_degraded_reason(
                                Some(degraded_reason.to_string()),
                                gpu_cpu_fallback.clone(),
                            ),
                        )
                    }
                }
                #[cfg(not(feature = "anomaly-detection"))]
                {
                    let degraded_reason = if self.has_diagonal_profile_runtime() {
                        "extended_anomaly_backend_unavailable"
                    } else {
                        "anomaly_runtime_state_incomplete"
                    };
                    (
                        Some("extended_isolation_forest".to_string()),
                        append_runtime_degraded_reason(
                            Some(degraded_reason.to_string()),
                            gpu_cpu_fallback.clone(),
                        ),
                    )
                }
            }
            "diagonal_profile" => {
                if self.has_diagonal_profile_runtime() {
                    (
                        Some("diagonal_profile".to_string()),
                        append_runtime_degraded_reason(
                            Some("anomaly_backend_degraded_to_diagonal_profile".to_string()),
                            gpu_cpu_fallback.clone(),
                        ),
                    )
                } else {
                    (
                        Some("isolation_forest_unknown".to_string()),
                        append_runtime_degraded_reason(
                            Some("anomaly_runtime_state_incomplete".to_string()),
                            gpu_cpu_fallback.clone(),
                        ),
                    )
                }
            }
            _ => (
                Some("isolation_forest_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some("anomaly_backend_unknown".to_string()),
                    gpu_cpu_fallback,
                ),
            ),
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                "isolation_forest",
                ModelFamily::Anomaly,
                CapabilityState::Implemented,
                row_values,
                Some(confidence),
                Some(abstain),
                execution_backend.clone(),
                degraded_reason.clone(),
            )?);
        }
        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataframe() -> DataFrame {
        DataFrame::new(vec![
            Series::new(
                "open".into(),
                vec![1.0_f64, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            )
            .into(),
            Series::new(
                "high".into(),
                vec![1.2_f64, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            )
            .into(),
            Series::new(
                "low".into(),
                vec![0.9_f64, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            )
            .into(),
            Series::new(
                "close".into(),
                vec![1.05_f64, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
            )
            .into(),
        ])
        .expect("sample dataframe")
    }

    #[test]
    fn isolation_forest_rejects_tampered_backend_kind_on_load() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let mut model = IsolationForestExpert::default();
        model.fit(&df, &Series::new("label".into(), vec![0_i32; 8]))?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir = std::env::temp_dir().join(format!("forex-models-forest-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;

        let model_path = artifact_dir.join(MODEL_FILE_NAME);
        let mut artifact: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&model_path).expect("read model"))
                .expect("parse model");
        artifact["backend_kind"] = serde_json::Value::String(String::new());
        std::fs::write(
            &model_path,
            serde_json::to_vec_pretty(&artifact).expect("serialize tampered model"),
        )
        .expect("write tampered model");

        let mut reloaded = IsolationForestExpert::default();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("blank backend kind should fail");
        assert!(err.to_string().contains("backend kind"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn isolation_forest_rejects_diagonal_profile_with_backend_payload() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let mut model = IsolationForestExpert::default();
        model.fit(&df, &Series::new("label".into(), vec![0_i32; 8]))?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("forex-models-forest-payload-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;

        let model_path = artifact_dir.join(MODEL_FILE_NAME);
        let mut artifact: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&model_path).expect("read model"))
                .expect("parse model");
        artifact["backend_kind"] = serde_json::Value::String("diagonal_profile".to_string());
        artifact["model_json"] = serde_json::Value::String("{\"tampered\":true}".to_string());
        std::fs::write(
            &model_path,
            serde_json::to_vec_pretty(&artifact).expect("serialize tampered model"),
        )
        .expect("write tampered model");

        let mut reloaded = IsolationForestExpert::default();
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("diagonal profile with backend payload should fail");
        assert!(
            err.to_string()
                .contains("must not carry serialized backend payload")
        );

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn predict_runtime_uses_shared_three_class_confidence_gate() -> Result<()> {
        let df = sample_dataframe();
        let expert = IsolationForestExpert {
            backend_kind: "diagonal_profile".to_string(),
            feature_columns: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
            ],
            dataset_rows: df.height(),
            anomaly_threshold: 0.5,
            score_std: 1.0,
            score_median: 0.0,
            score_mad: 1.0,
            fallback_means: vec![0.0; 4],
            fallback_stds: vec![1.0; 4],
            ..IsolationForestExpert::default()
        };

        let probabilities = expert.predict_proba(&df)?;
        let predictions = expert.predict_runtime(&df)?;
        let first_row = probabilities.row(0);
        let row_values = [first_row[0], first_row[1], first_row[2]];
        let (expected_confidence, expected_abstain) = three_class_runtime_confidence(row_values)?;

        assert_eq!(predictions.len(), df.height());
        assert_eq!(predictions[0].confidence(), Some(expected_confidence));
        assert_eq!(predictions[0].abstain_recommended(), Some(expected_abstain));
        Ok(())
    }

    #[test]
    fn robust_score_profile_stays_centered_under_single_large_outlier() {
        let (mean, std, median, mad) = score_statistics(&[0.9, 1.0, 1.1, 8.0]);
        assert!(mean > median, "mean should be pulled by the outlier");
        assert!(
            (median - 1.05).abs() < 0.1,
            "median should stay near the dense cluster"
        );
        assert!(
            mad < std,
            "robust dispersion should stay tighter than std under outliers"
        );
    }

    #[test]
    fn isolation_forest_artifact_requires_runtime_metadata() {
        let artifact = IsolationForestArtifact {
            model_name: "isolation_forest".to_string(),
            backend_kind: "diagonal_profile".to_string(),
            runtime_metadata: None,
            feature_columns: vec!["f1".to_string()],
            dataset_rows: 8,
            n_trees: 64,
            sample_size: 8,
            extension_level: 0,
            max_tree_depth: None,
            anomaly_threshold: 0.5,
            score_mean: 0.0,
            score_std: 1.0,
            score_median: 0.0,
            score_mad: 1.0,
            model_json: String::new(),
            fallback_means: vec![0.0],
            fallback_stds: vec![1.0],
        };

        let err = validate_isolation_forest_artifact(&artifact)
            .expect_err("artifact without runtime metadata should fail");
        assert!(err.to_string().contains("runtime metadata"));
    }

    #[test]
    fn isolation_forest_load_uses_embedded_runtime_metadata_when_metadata_file_missing()
    -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let labels = sample_labels();
        let mut model = IsolationForestExpert::new(64, 16);
        model.fit(&df, &labels)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("forex-models-isolation-embed-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;
        std::fs::remove_file(artifact_dir.join(METADATA_FILE_NAME)).expect("remove metadata file");

        let mut reloaded = IsolationForestExpert::new(64, 16);
        reloaded.load(&artifact_dir)?;
        assert!(!reloaded.feature_columns.is_empty());

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn isolation_forest_load_rejects_sidecar_drift_against_embedded_metadata() -> Result<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let df = sample_dataframe();
        let labels = sample_labels();
        let mut model = IsolationForestExpert::new(64, 16);
        model.fit(&df, &labels)?;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("forex-models-isolation-drift-{unique}"));
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        model.save(&artifact_dir)?;
        let metadata_path = artifact_dir.join(METADATA_FILE_NAME);
        let mut drifted: RuntimeArtifactMetadata =
            read_json(&metadata_path).expect("read saved metadata");
        drifted.training_summary.dataset_rows += 1;
        write_json(&metadata_path, &drifted).expect("write drifted metadata");

        let mut reloaded = IsolationForestExpert::new(64, 16);
        let err = reloaded
            .load(&artifact_dir)
            .expect_err("drifted sidecar metadata should fail load");
        assert!(err.to_string().contains("sidecar mismatch"));

        std::fs::remove_dir_all(&artifact_dir).expect("cleanup artifact dir");
        Ok(())
    }

    #[test]
    fn isolation_forest_runtime_details_mark_diagonal_profile_as_degraded() {
        let expert = IsolationForestExpert {
            backend_kind: "diagonal_profile".to_string(),
            feature_columns: vec!["open".to_string(), "high".to_string()],
            dataset_rows: 8,
            fallback_means: vec![0.0, 0.0],
            fallback_stds: vec![1.0, 1.0],
            ..IsolationForestExpert::default()
        };

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("diagonal_profile"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("anomaly_backend_degraded_to_diagonal_profile")
        );
    }

    #[test]
    fn isolation_forest_runtime_details_mark_incomplete_diagonal_profile_state_as_unknown() {
        let expert = IsolationForestExpert {
            backend_kind: "diagonal_profile".to_string(),
            feature_columns: vec!["open".to_string(), "high".to_string()],
            dataset_rows: 8,
            fallback_means: vec![0.0],
            fallback_stds: vec![1.0, 1.0],
            ..IsolationForestExpert::default()
        };

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("isolation_forest_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("anomaly_runtime_state_incomplete")
        );
    }

    #[test]
    fn isolation_forest_runtime_details_mark_missing_feature_schema_as_incomplete() {
        let expert = IsolationForestExpert {
            backend_kind: "extended_isolation_forest".to_string(),
            dataset_rows: 8,
            ..IsolationForestExpert::default()
        };

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("isolation_forest_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("anomaly_runtime_state_incomplete")
        );
    }
}
