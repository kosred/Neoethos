#[cfg(any(test, feature = "anomaly-detection"))]
use anyhow::Context;
use anyhow::{Result, bail};
#[cfg(feature = "anomaly-detection")]
use extended_isolation_forest::{Forest, ForestOptions};
use ndarray::Array2;
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
#[cfg(feature = "anomaly-detection")]
use seq_macro::seq;
#[cfg(any(test, feature = "anomaly-detection"))]
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::ExpertModel;
#[cfg(any(test, feature = "anomaly-detection"))]
use crate::base::canonical_three_class_label_mapping;
#[cfg(feature = "anomaly-detection")]
use crate::base::{
    build_runtime_prediction_with_details, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata, validate_model_labels,
};
#[cfg(any(test, feature = "anomaly-detection"))]
use crate::runtime::artifacts::RuntimeArtifactMetadata;
#[cfg(feature = "anomaly-detection")]
use crate::runtime::artifacts::TrainingSummaryMetadata;
#[cfg(any(test, feature = "anomaly-detection"))]
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::capabilities::{
    normalize_runtime_device_policy, requested_runtime_device_policy,
};
use crate::runtime::prediction::RuntimePrediction;
#[cfg(feature = "anomaly-detection")]
use crate::statistical::common::{METADATA_FILE_NAME, read_json};
#[cfg(feature = "anomaly-detection")]
use crate::statistical::common::{
    MODEL_FILE_NAME, ensure_feature_columns_match, feature_matrix_from_frame, write_json,
};

#[cfg(any(test, feature = "anomaly-detection"))]
const ISOLATION_FOREST_PRECISION_SCHEMA: &str = "neoethos.isolation_forest.f64.v2";
const ISOLATION_FOREST_CPU_BACKEND: &str = "extended_isolation_forest_cpu";
#[cfg(any(test, feature = "anomaly-detection"))]
const MAX_COMPILED_FEATURES: usize = 128;

#[cfg(any(test, feature = "anomaly-detection"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationForestArtifact {
    precision_schema: String,
    model_name: String,
    backend_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_metadata: Option<RuntimeArtifactMetadata>,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    n_trees: usize,
    sample_size: usize,
    extension_level: usize,
    max_tree_depth: Option<usize>,
    anomaly_threshold: f64,
    score_mean: f64,
    score_std: f64,
    score_median: f64,
    score_mad: f64,
    model_json: String,
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

#[cfg(feature = "anomaly-detection")]
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

#[cfg(any(test, feature = "anomaly-detection"))]
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

#[cfg(feature = "anomaly-detection")]
fn resolve_runtime_metadata_from_artifact(
    path: &Path,
    artifact: &IsolationForestArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_runtime_metadata(&metadata, &artifact.feature_columns, artifact.dataset_rows)
                .with_context(|| {
                    format!(
                        "runtime metadata sidecar mismatch with embedded isolation_forest metadata at {}",
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
                    "runtime metadata sidecar mismatch with embedded isolation_forest metadata at {}",
                    metadata_path.display()
                );
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

#[cfg(any(test, feature = "anomaly-detection"))]
fn validate_isolation_forest_artifact(artifact: &IsolationForestArtifact) -> Result<()> {
    if artifact.precision_schema != ISOLATION_FOREST_PRECISION_SCHEMA {
        bail!(
            "isolation_forest precision schema mismatch: expected {}, got {}",
            ISOLATION_FOREST_PRECISION_SCHEMA,
            artifact.precision_schema
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("isolation_forest artifact must contain feature columns");
    }
    if artifact.feature_columns.len() > MAX_COMPILED_FEATURES {
        bail!(
            "isolation_forest artifact has {} features; this build supports 1..={MAX_COMPILED_FEATURES}",
            artifact.feature_columns.len()
        );
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
    if artifact.backend_kind != ISOLATION_FOREST_CPU_BACKEND {
        bail!(
            "unsupported isolation forest backend kind: {}",
            artifact.backend_kind
        );
    }
    if artifact.model_json.trim().is_empty() {
        bail!("extended isolation forest artifact must contain serialized backend payload");
    }
    Ok(())
}

#[cfg(any(test, feature = "anomaly-detection"))]
fn validate_probability(value: f64) -> Result<f64> {
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

#[cfg(feature = "anomaly-detection")]
fn quantile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.5;
    }

    let clamped = fraction.clamp(0.0, 1.0);
    let index = ((values.len().saturating_sub(1) as f64) * clamped).round() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

#[cfg(any(test, feature = "anomaly-detection"))]
fn score_statistics(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
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

#[cfg(any(test, feature = "anomaly-detection"))]
fn anomaly_probabilities(
    scores: &[f64],
    threshold: f64,
    score_std: f64,
    score_median: f64,
    score_mad: f64,
) -> Result<Array2<f64>> {
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

#[cfg(feature = "anomaly-detection")]
fn feature_rows(features: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..features.nrows())
        .map(|row_idx| features.row(row_idx).iter().copied().collect::<Vec<_>>())
        .collect()
}

#[cfg(any(test, feature = "anomaly-detection"))]
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

pub struct IsolationForestExpert {
    #[cfg(feature = "anomaly-detection")]
    backend: Option<Box<dyn ForestBackend>>,
    pub n_trees: usize,
    pub sample_size: usize,
    pub extension_level: usize,
    pub max_tree_depth: Option<usize>,
    pub backend_kind: String,
    pub feature_columns: Vec<String>,
    pub dataset_rows: usize,
    pub anomaly_threshold: f64,
    pub score_mean: f64,
    pub score_std: f64,
    pub score_median: f64,
    pub score_mad: f64,
}

impl IsolationForestExpert {
    pub fn new(n_trees: usize, sample_size: usize) -> Self {
        Self {
            #[cfg(feature = "anomaly-detection")]
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
        }
    }

    #[cfg(feature = "anomaly-detection")]
    fn artifact(&self) -> Result<IsolationForestArtifact> {
        Ok(IsolationForestArtifact {
            precision_schema: ISOLATION_FOREST_PRECISION_SCHEMA.to_string(),
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
            model_json: self
                .backend
                .as_ref()
                .context("isolation forest backend missing")?
                .to_json()?,
        })
    }
}

impl Default for IsolationForestExpert {
    fn default() -> Self {
        Self::new(128, 256)
    }
}

fn anomaly_cpu_backend_for_policy(requested: &str) -> Result<&'static str> {
    let normalized = normalize_runtime_device_policy(requested);
    match normalized.as_str() {
        "auto" | "cpu" => Ok(ISOLATION_FOREST_CPU_BACKEND),
        "gpu" => bail!(
            "GpuOnly anomaly request has no isolation_forest GPU implementation; CPU execution was not started"
        ),
        policy if policy.starts_with("gpu:") => bail!(
            "GpuOnly anomaly request '{policy}' has no isolation_forest GPU implementation; CPU execution was not started"
        ),
        other => bail!("unsupported isolation_forest device policy '{other}'"),
    }
}

#[cfg(any(test, feature = "anomaly-detection"))]
fn validate_feature_count(feature_count: usize) -> Result<()> {
    if !(1..=MAX_COMPILED_FEATURES).contains(&feature_count) {
        bail!(
            "extended isolation forest supports 1..={MAX_COMPILED_FEATURES} compiled feature dimensions, got {feature_count}"
        );
    }
    Ok(())
}

impl IsolationForestExpert {
    #[cfg(feature = "anomaly-detection")]
    fn fit_cpu(&mut self, x: &FeatureFrame, y: &[i32], runtime_backend: &str) -> Result<()> {
        validate_model_labels(y, x.n_samples())?;
        let (features, feature_columns) = feature_matrix_from_frame(x)?;
        if features.nrows() < 8 {
            bail!(
                "isolation forest requires at least 8 rows, received {}",
                features.nrows()
            );
        }
        validate_feature_count(features.ncols())?;

        let training_rows = feature_rows(&features);
        let sample_size = self.sample_size.min(training_rows.len()).max(8);
        let extension_level = if self.extension_level == 0 {
            features.ncols().saturating_sub(1)
        } else {
            self.extension_level.min(features.ncols().saturating_sub(1))
        };
        let options = ForestOptions {
            n_trees: self.n_trees.max(32),
            sample_size,
            max_tree_depth: self.max_tree_depth,
            extension_level,
        };

        let backend = dispatch_forest_builder(features.ncols(), &training_rows, &options)?;
        let mut training_scores = training_rows
            .iter()
            .map(|row| backend.score_row(row))
            .collect::<Result<Vec<_>>>()?;
        training_scores.sort_by(|left, right| left.total_cmp(right));

        let (score_mean, score_std, score_median, score_mad) = score_statistics(&training_scores);
        self.backend = Some(backend);
        self.n_trees = options.n_trees;
        self.sample_size = options.sample_size;
        self.extension_level = extension_level;
        self.backend_kind = runtime_backend.to_string();
        self.feature_columns = feature_columns;
        self.dataset_rows = features.nrows();
        self.anomaly_threshold = quantile(&training_scores, 0.95).max(0.5);
        self.score_mean = score_mean;
        self.score_std = score_std;
        self.score_median = score_median;
        self.score_mad = score_mad;
        Ok(())
    }

    #[cfg(feature = "anomaly-detection")]
    fn predict_proba_cpu(&self, x: &FeatureFrame) -> Result<Array2<f64>> {
        if self.backend_kind != ISOLATION_FOREST_CPU_BACKEND {
            bail!(
                "isolation_forest backend mismatch: expected {ISOLATION_FOREST_CPU_BACKEND}, got {}",
                self.backend_kind
            );
        }
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let (features, _) = feature_matrix_from_frame(x)?;
        validate_feature_count(features.ncols())?;
        let backend = self
            .backend
            .as_ref()
            .context("isolation forest CPU backend is not loaded")?;
        let scores = feature_rows(&features)
            .iter()
            .map(|row| backend.score_row(row))
            .collect::<Result<Vec<_>>>()?;

        anomaly_probabilities(
            &scores,
            self.anomaly_threshold,
            self.score_std,
            self.score_median,
            self.score_mad,
        )
    }
}

impl ExpertModel for IsolationForestExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        let runtime_backend =
            anomaly_cpu_backend_for_policy(&requested_runtime_device_policy("isolation_forest"))?;

        #[cfg(not(feature = "anomaly-detection"))]
        {
            let _ = (x, y, lease, runtime_backend);
            bail!(
                "isolation_forest requires the anomaly-detection feature; no substitute model is permitted"
            )
        }

        #[cfg(feature = "anomaly-detection")]
        {
            lease.scope(|| self.fit_cpu(x, y, runtime_backend))
        }
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        anomaly_cpu_backend_for_policy(&requested_runtime_device_policy("isolation_forest"))?;

        #[cfg(not(feature = "anomaly-detection"))]
        {
            let _ = (x, lease);
            bail!(
                "isolation_forest requires the anomaly-detection feature; no substitute model is permitted"
            )
        }

        #[cfg(feature = "anomaly-detection")]
        {
            lease.scope(|| self.predict_proba_cpu(x))
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "anomaly-detection"))]
        {
            let _ = path;
            bail!("isolation_forest requires the anomaly-detection feature; no model can be saved")
        }

        #[cfg(feature = "anomaly-detection")]
        {
            if self.backend_kind != ISOLATION_FOREST_CPU_BACKEND {
                bail!(
                    "unsupported isolation forest backend kind: {}",
                    self.backend_kind
                );
            }
            self.backend
                .as_ref()
                .context("extended isolation forest CPU backend missing")?;
            std::fs::create_dir_all(path)
                .with_context(|| format!("create isolation forest directory {}", path.display()))?;
            let runtime_metadata = anomaly_runtime_metadata(
                "isolation_forest",
                self.feature_columns.clone(),
                self.dataset_rows,
            )?;
            validate_runtime_metadata(&runtime_metadata, &self.feature_columns, self.dataset_rows)?;
            let artifact = self.artifact()?;
            validate_isolation_forest_artifact(&artifact)?;
            if artifact.runtime_metadata.as_ref() != Some(&runtime_metadata) {
                bail!("runtime metadata file does not match isolation_forest artifact");
            }
            write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
            write_json(&path.join(MODEL_FILE_NAME), &artifact)
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "anomaly-detection"))]
        {
            let _ = path;
            bail!(
                "isolation_forest requires the anomaly-detection feature; artifact loading is unavailable"
            )
        }

        #[cfg(feature = "anomaly-detection")]
        {
            let artifact: IsolationForestArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
            validate_isolation_forest_artifact(&artifact)?;
            if artifact.model_name != "isolation_forest" {
                bail!(
                    "expected isolation_forest artifact, got {}",
                    artifact.model_name
                );
            }
            resolve_runtime_metadata_from_artifact(path, &artifact)?;
            let backend =
                dispatch_forest_loader(artifact.feature_columns.len(), &artifact.model_json)?;

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
            next_state.backend = Some(backend);

            *self = next_state;
            Ok(())
        }
    }
}

impl IsolationForestExpert {
    #[cfg(feature = "anomaly-detection")]
    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        #[cfg(not(feature = "anomaly-detection"))]
        {
            return (
                Some("isolation_forest_unavailable".to_string()),
                Some("anomaly_backend_not_compiled".to_string()),
            );
        }

        #[cfg(feature = "anomaly-detection")]
        {
            if self.dataset_rows == 0
                || self.feature_columns.is_empty()
                || self.backend.is_none()
                || self.backend_kind != ISOLATION_FOREST_CPU_BACKEND
            {
                return (
                    Some("isolation_forest_unknown".to_string()),
                    Some("anomaly_runtime_state_incomplete".to_string()),
                );
            }
            (Some(self.backend_kind.clone()), None)
        }
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        anomaly_cpu_backend_for_policy(&requested_runtime_device_policy("isolation_forest"))?;

        #[cfg(not(feature = "anomaly-detection"))]
        {
            let _ = (x, lease);
            bail!(
                "isolation_forest requires the anomaly-detection feature; prediction is unavailable"
            )
        }

        #[cfg(feature = "anomaly-detection")]
        {
            lease.scope(|| {
                let probabilities = self.predict_proba_cpu(x)?;
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
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_probability_projection_preserves_sub_f32_delta() -> Result<()> {
        let probabilities =
            anomaly_probabilities(&[0.500_000_000_001], 0.5, 1.0e-12, 0.5, 1.0e-12)?;
        assert!(
            probabilities[(0, 0)] > 0.5,
            "the f64 score delta was rounded away"
        );
        assert!((probabilities.row(0).sum() - 1.0).abs() <= 1.0e-15);
        Ok(())
    }

    #[test]
    fn gpu_policy_is_rejected_before_cpu_backend_selection() {
        let error = anomaly_cpu_backend_for_policy("cuda:1")
            .expect_err("GpuOnly must not execute isolation forest on CPU");
        assert!(error.to_string().contains("GpuOnly"));
        assert_eq!(
            anomaly_cpu_backend_for_policy("cpu").expect("CpuOnly backend"),
            ISOLATION_FOREST_CPU_BACKEND
        );
    }

    #[test]
    fn unsupported_dimension_is_not_replaced_by_another_model() {
        let error = validate_feature_count(MAX_COMPILED_FEATURES + 1)
            .expect_err("unsupported feature width must fail closed");
        assert!(error.to_string().contains("1..=128"));
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
    fn isolation_forest_artifact_requires_f64_schema_and_runtime_metadata() {
        let mut artifact = IsolationForestArtifact {
            precision_schema: ISOLATION_FOREST_PRECISION_SCHEMA.to_string(),
            model_name: "isolation_forest".to_string(),
            backend_kind: ISOLATION_FOREST_CPU_BACKEND.to_string(),
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
            model_json: "{}".to_string(),
        };

        let error = validate_isolation_forest_artifact(&artifact)
            .expect_err("artifact without runtime metadata must fail");
        assert!(error.to_string().contains("runtime metadata"));

        artifact.precision_schema = "neoethos.isolation_forest.f32.v1".to_string();
        let error = validate_isolation_forest_artifact(&artifact)
            .expect_err("legacy f32 artifact must fail");
        assert!(error.to_string().contains("precision schema"));
    }
}
