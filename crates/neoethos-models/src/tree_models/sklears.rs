use anyhow::{Context, Result, bail};
use ndarray::Array2;
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::Path;

use crate::base::{
    ExpertModel, feature_columns_from_frame, feature_frame_to_f64_array, validate_model_labels,
};
use crate::runtime::artifacts::TrainingSummaryMetadata;
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;
use crate::tree_models::common::{
    build_tree_runtime_predictions, default_training_summary, ensure_feature_columns_match,
    read_runtime_metadata, read_tree_json_artifact, tree_artifact_paths, tree_runtime_metadata,
    write_runtime_metadata, write_tree_json_artifact,
};

const MODEL_FILE_NAME: &str = "model.json";
const SKLEARS_RUNTIME_FILE_NAME: &str = "runtime.json";
const SKLEARS_F64_ARTIFACT_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TreeNode {
    Leaf {
        class_counts: [usize; 3],
        probabilities: [f64; 3],
    },
    Split {
        feature_index: usize,
        threshold: f64,
        probabilities: [f64; 3],
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionTreeArtifact {
    schema_version: u32,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_thresholds_per_feature: usize,
    root: TreeNode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SklearsRuntimeArtifact {
    schema_version: u32,
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
}

fn validate_probability_vector(probabilities: &[f64; 3]) -> Result<()> {
    let mut sum = 0.0_f64;
    for value in probabilities {
        if !value.is_finite() || *value < 0.0 {
            bail!("sklears-tree probabilities must be finite and non-negative");
        }
        sum += *value;
    }
    if sum <= f64::EPSILON {
        bail!("sklears-tree probabilities must have positive mass");
    }
    if (sum - 1.0).abs() > 1e-3 {
        bail!(
            "sklears-tree probabilities must sum to 1.0 within tolerance, got {}",
            sum
        );
    }
    Ok(())
}

fn validate_tree_node(node: &TreeNode, feature_count: usize) -> Result<()> {
    match node {
        TreeNode::Leaf {
            class_counts,
            probabilities,
        } => {
            if class_counts.iter().sum::<usize>() == 0 {
                bail!("sklears-tree leaf must contain at least one observed class");
            }
            validate_probability_vector(probabilities)?;
        }
        TreeNode::Split {
            feature_index,
            threshold,
            probabilities,
            left,
            right,
        } => {
            if *feature_index >= feature_count {
                bail!(
                    "sklears-tree split feature index {} is out of bounds for {} features",
                    feature_index,
                    feature_count
                );
            }
            if !threshold.is_finite() {
                bail!("sklears-tree split threshold must be finite");
            }
            validate_probability_vector(probabilities)?;
            validate_tree_node(left, feature_count)?;
            validate_tree_node(right, feature_count)?;
        }
    }
    Ok(())
}

fn validate_tree_artifact(artifact: &DecisionTreeArtifact, feature_count: usize) -> Result<()> {
    if artifact.max_depth == 0 {
        bail!("sklears-tree artifact must have positive max_depth");
    }
    if artifact.min_samples_split == 0 || artifact.min_samples_leaf == 0 {
        bail!("sklears-tree artifact must have positive sample thresholds");
    }
    if artifact.max_thresholds_per_feature == 0 {
        bail!("sklears-tree artifact must have positive threshold budget");
    }
    validate_tree_node(&artifact.root, feature_count)
}

#[derive(Debug, Clone)]
pub struct SklearsTreeExpert {
    root: Option<TreeNode>,
    feature_columns: Vec<String>,
    training_summary: Option<TrainingSummaryMetadata>,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_thresholds_per_feature: usize,
}

impl SklearsTreeExpert {
    fn read_runtime_artifact(path: &Path) -> Result<Option<SklearsRuntimeArtifact>> {
        let runtime_path = path.join(SKLEARS_RUNTIME_FILE_NAME);
        if !runtime_path.exists() {
            return Ok(None);
        }
        let artifact = read_tree_json_artifact(&runtime_path, "sklears-tree runtime artifact")?;
        Ok(Some(artifact))
    }

    pub fn new() -> Self {
        Self {
            root: None,
            feature_columns: Vec::new(),
            training_summary: None,
            max_depth: 6,
            min_samples_split: 32,
            min_samples_leaf: 16,
            max_thresholds_per_feature: 32,
        }
    }

    fn labels_from_slice(labels: &[i32], expected_rows: usize) -> Result<Vec<usize>> {
        validate_model_labels(labels, expected_rows)?;
        labels
            .iter()
            .map(|value| match value {
                0 => Ok(0usize),
                1 => Ok(1usize),
                -1 => Ok(2usize),
                other => {
                    bail!("unsupported sklears-tree label: {other}; expected one of -1, 0, 1")
                }
            })
            .collect()
    }

    fn class_counts(labels: &[usize], rows: &[usize]) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for row in rows {
            counts[labels[*row]] += 1;
        }
        counts
    }

    fn probabilities_from_counts(counts: [usize; 3]) -> Result<[f64; 3]> {
        let total = counts.iter().sum::<usize>() as f64;
        if total <= f64::EPSILON {
            bail!("sklears-tree node cannot derive probabilities from zero observations");
        }
        Ok([
            counts[0] as f64 / total,
            counts[1] as f64 / total,
            counts[2] as f64 / total,
        ])
    }

    fn is_pure(counts: [usize; 3]) -> bool {
        counts.iter().filter(|count| **count > 0).count() <= 1
    }

    fn gini_from_counts(counts: [usize; 3]) -> f64 {
        let total = counts.iter().sum::<usize>() as f64;
        if total <= f64::EPSILON {
            return 0.0;
        }
        1.0 - counts
            .iter()
            .map(|count| {
                let prob = *count as f64 / total;
                prob * prob
            })
            .sum::<f64>()
    }

    fn candidate_thresholds(
        &self,
        features: &Array2<f64>,
        rows: &[usize],
        feature_idx: usize,
    ) -> Vec<f64> {
        let mut values = rows
            .iter()
            .map(|row| features[(*row, feature_idx)])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
        if values.len() < 2 {
            return Vec::new();
        }

        let midpoints = values
            .windows(2)
            .map(|window| (window[0] + window[1]) * 0.5)
            .collect::<Vec<_>>();
        if midpoints.len() <= self.max_thresholds_per_feature {
            return midpoints;
        }

        let step = ((midpoints.len() as f64) / (self.max_thresholds_per_feature as f64))
            .ceil()
            .max(1.0) as usize;
        midpoints
            .into_iter()
            .step_by(step)
            .take(self.max_thresholds_per_feature)
            .collect()
    }

    fn split_rows(
        features: &Array2<f64>,
        rows: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for row in rows {
            if features[(*row, feature_idx)] <= threshold {
                left.push(*row);
            } else {
                right.push(*row);
            }
        }
        (left, right)
    }

    fn best_split(
        &self,
        features: &Array2<f64>,
        labels: &[usize],
        rows: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let parent_counts = Self::class_counts(labels, rows);
        let parent_gini = Self::gini_from_counts(parent_counts);
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        for feature_idx in 0..features.ncols() {
            for threshold in self.candidate_thresholds(features, rows, feature_idx) {
                let (left_rows, right_rows) =
                    Self::split_rows(features, rows, feature_idx, threshold);
                if left_rows.len() < self.min_samples_leaf
                    || right_rows.len() < self.min_samples_leaf
                {
                    continue;
                }

                let left_counts = Self::class_counts(labels, &left_rows);
                let right_counts = Self::class_counts(labels, &right_rows);
                let left_weight = left_rows.len() as f64 / rows.len() as f64;
                let right_weight = right_rows.len() as f64 / rows.len() as f64;
                let gain = parent_gini
                    - (left_weight * Self::gini_from_counts(left_counts))
                    - (right_weight * Self::gini_from_counts(right_counts));

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_rows, right_rows));
                }
            }
        }

        if best_gain > 1e-6 { best_split } else { None }
    }

    fn build_node(
        &self,
        features: &Array2<f64>,
        labels: &[usize],
        rows: &[usize],
        depth: usize,
    ) -> Result<TreeNode> {
        let counts = Self::class_counts(labels, rows);
        let probabilities = Self::probabilities_from_counts(counts)?;
        if depth >= self.max_depth || rows.len() < self.min_samples_split || Self::is_pure(counts) {
            return Ok(TreeNode::Leaf {
                class_counts: counts,
                probabilities,
            });
        }

        if let Some((feature_index, threshold, left_rows, right_rows)) =
            self.best_split(features, labels, rows)
        {
            return Ok(TreeNode::Split {
                feature_index,
                threshold,
                probabilities,
                left: Box::new(self.build_node(features, labels, &left_rows, depth + 1)?),
                right: Box::new(self.build_node(features, labels, &right_rows, depth + 1)?),
            });
        }

        Ok(TreeNode::Leaf {
            class_counts: counts,
            probabilities,
        })
    }

    fn predict_row_probabilities(
        node: &TreeNode,
        features: &Array2<f64>,
        row_idx: usize,
    ) -> [f64; 3] {
        match node {
            TreeNode::Leaf { probabilities, .. } => *probabilities,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if features[(row_idx, *feature_index)] <= *threshold {
                    Self::predict_row_probabilities(left, features, row_idx)
                } else {
                    Self::predict_row_probabilities(right, features, row_idx)
                }
            }
        }
    }

    fn stored_training_summary(&self) -> TrainingSummaryMetadata {
        self.training_summary
            .clone()
            .unwrap_or_else(|| TrainingSummaryMetadata::new(0, 0, 0))
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        let root = self
            .root
            .as_ref()
            .context("sklears-tree model not fitted")?;
        if self.feature_columns.is_empty() {
            bail!("sklears-tree model is missing feature columns");
        }
        let summary = self
            .training_summary
            .as_ref()
            .context("sklears-tree model is missing training summary metadata")?;
        if summary.dataset_rows == 0 {
            bail!("sklears-tree training summary must record non-zero dataset_rows");
        }
        if summary.dataset_rows != summary.train_rows + summary.val_rows {
            bail!(
                "sklears-tree training summary is inconsistent: dataset_rows={} train_rows={} val_rows={}",
                summary.dataset_rows,
                summary.train_rows,
                summary.val_rows
            );
        }
        validate_tree_node(root, self.feature_columns.len())?;
        Ok(())
    }
}

impl Default for SklearsTreeExpert {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for SklearsTreeExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], _lease: &CpuLease) -> Result<()> {
        let features =
            feature_frame_to_f64_array(x).context("build f64 sklears-tree feature matrix")?;
        if features.nrows() == 0 || features.ncols() == 0 {
            bail!("sklears-tree requires a non-empty feature matrix");
        }
        let labels = Self::labels_from_slice(y, features.nrows())?;

        let rows = (0..features.nrows()).collect::<Vec<_>>();
        self.root = Some(self.build_node(&features, &labels, &rows, 0)?);
        self.feature_columns = feature_columns_from_frame(x);
        self.training_summary = Some(default_training_summary(x));
        Ok(())
    }

    fn predict_proba(&self, x: &FeatureFrame, _lease: &CpuLease) -> Result<Array2<f64>> {
        let root = self
            .root
            .as_ref()
            .context("sklears-tree model not fitted")?;
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let features =
            feature_frame_to_f64_array(x).context("build f64 sklears-tree inference matrix")?;
        let mut probabilities = Array2::zeros((features.nrows(), 3));
        for row_idx in 0..features.nrows() {
            let row_probs = Self::predict_row_probabilities(root, &features, row_idx);
            probabilities[(row_idx, 0)] = row_probs[0];
            probabilities[(row_idx, 1)] = row_probs[1];
            probabilities[(row_idx, 2)] = row_probs[2];
        }
        Ok(probabilities)
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.ensure_runtime_state_ready()?;
        let root = self
            .root
            .as_ref()
            .context("sklears-tree model not fitted")?;
        let artifact = DecisionTreeArtifact {
            schema_version: SKLEARS_F64_ARTIFACT_SCHEMA_VERSION,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_thresholds_per_feature: self.max_thresholds_per_feature,
            root: root.clone(),
        };
        let (model_path, metadata_path) = tree_artifact_paths(path, MODEL_FILE_NAME);
        write_tree_json_artifact(&model_path, &artifact, "sklears-tree artifact")?;
        let runtime_artifact = SklearsRuntimeArtifact {
            schema_version: SKLEARS_F64_ARTIFACT_SCHEMA_VERSION,
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
        };
        write_tree_json_artifact(
            &path.join(SKLEARS_RUNTIME_FILE_NAME),
            &runtime_artifact,
            "sklears-tree runtime artifact",
        )?;
        write_runtime_metadata(
            &metadata_path,
            &tree_runtime_metadata(
                "sklears_tree",
                self.feature_columns.clone(),
                self.stored_training_summary(),
            )?,
        )?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let (model_path, metadata_path) = tree_artifact_paths(path, MODEL_FILE_NAME);
        let artifact: DecisionTreeArtifact =
            read_tree_json_artifact(&model_path, "sklears-tree artifact")?;
        if artifact.schema_version != SKLEARS_F64_ARTIFACT_SCHEMA_VERSION {
            bail!(
                "sklears-tree artifact schema {} is unsupported; expected f64 schema {}",
                artifact.schema_version,
                SKLEARS_F64_ARTIFACT_SCHEMA_VERSION
            );
        }
        let runtime_artifact = Self::read_runtime_artifact(path)?;
        if let Some(runtime) = runtime_artifact.as_ref()
            && runtime.schema_version != SKLEARS_F64_ARTIFACT_SCHEMA_VERSION
        {
            bail!(
                "sklears-tree runtime artifact schema {} is unsupported; expected f64 schema {}",
                runtime.schema_version,
                SKLEARS_F64_ARTIFACT_SCHEMA_VERSION
            );
        }
        let metadata = if metadata_path.exists() {
            let metadata = read_runtime_metadata(&metadata_path)?;
            if metadata.model_name != "sklears_tree" || metadata.family != ModelFamily::Tree {
                bail!(
                    "sklears-tree runtime metadata mismatch: expected tree/sklears_tree, got {}/{}",
                    metadata.family,
                    metadata.model_name
                );
            }
            if metadata.feature_columns.is_empty() {
                bail!("sklears-tree runtime metadata must contain at least one feature column");
            }
            metadata
        } else if let Some(runtime_artifact) = runtime_artifact {
            let metadata = tree_runtime_metadata(
                "sklears_tree",
                runtime_artifact.feature_columns,
                runtime_artifact.training_summary,
            )?;
            tracing::warn!(
                path = %path.display(),
                "sklears-tree metadata sidecar missing; reconstructing from runtime artifact"
            );
            metadata
        } else {
            bail!(
                "sklears-tree metadata sidecar missing and runtime artifact missing at {}",
                path.display()
            );
        };
        if metadata.training_summary.dataset_rows == 0 {
            bail!("sklears-tree runtime metadata must record non-zero dataset_rows");
        }
        if metadata.training_summary.dataset_rows
            != metadata.training_summary.train_rows + metadata.training_summary.val_rows
        {
            bail!("sklears-tree runtime metadata training summary is inconsistent");
        }
        validate_tree_artifact(&artifact, metadata.feature_columns.len())?;
        self.max_depth = artifact.max_depth;
        self.min_samples_split = artifact.min_samples_split;
        self.min_samples_leaf = artifact.min_samples_leaf;
        self.max_thresholds_per_feature = artifact.max_thresholds_per_feature;
        self.root = Some(artifact.root);
        self.training_summary = Some(metadata.training_summary);
        self.feature_columns = metadata.feature_columns;
        Ok(())
    }
}

impl SklearsTreeExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`]
    /// adapter so the registry / aggregator can detect column-layout
    /// drift after a retraining session.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_state_ready()?;
        let probabilities = self.predict_proba(x, lease)?;
        build_tree_runtime_predictions("sklears_tree", &probabilities, "sklears_tree_f64")
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertModel, SklearsTreeExpert, TreeNode};
    use crate::runtime::artifacts::TrainingSummaryMetadata;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
    use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn frame(columns: Vec<(&str, Vec<f64>)>) -> FeatureFrame {
        let rows = columns.first().map(|(_, values)| values.len()).unwrap_or(0);
        let columns = columns
            .into_iter()
            .map(|(name, values)| {
                FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; rows])
                    .expect("valid feature column")
            })
            .collect::<Vec<_>>();
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
        .expect("valid feature frame")
    }

    fn sample_three_class_dataset() -> (FeatureFrame, Vec<i32>) {
        let x = frame(vec![
            (
                "momentum",
                vec![0.96, 0.93, 0.89, 0.07, 0.03, 0.11, -0.94, -0.91, -0.88],
            ),
            (
                "trend",
                vec![0.87, 0.91, 0.86, 0.01, -0.02, 0.04, -0.9, -0.86, -0.93],
            ),
            (
                "volatility",
                vec![0.62, 0.58, 0.6, 0.2, 0.18, 0.23, 0.69, 0.66, 0.64],
            ),
        ]);
        (x, vec![1, 1, 1, 0, 0, 0, -1, -1, -1])
    }

    fn single_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("single-worker lease")
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{nonce}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn sklears_save_rejects_missing_training_summary() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("sklears-missing-summary");
        let lease = single_worker_lease();

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.training_summary = None;

        let err = expert
            .save(&artifact_dir)
            .expect_err("save should fail without training summary");
        assert!(err.to_string().contains("training summary"));
    }

    #[test]
    fn sklears_predict_runtime_returns_runtime_predictions() {
        let (x, y) = sample_three_class_dataset();
        let lease = single_worker_lease();

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y, &lease).expect("fit should succeed");

        let predictions = expert
            .predict_runtime(&x, &lease)
            .expect("runtime prediction should succeed");
        assert_eq!(predictions.len(), x.n_samples());
        assert!(predictions.iter().all(|prediction| {
            prediction.class_probabilities().len() == 3
                && prediction
                    .class_probabilities()
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
        }));
    }

    #[test]
    fn sklears_load_rejects_inconsistent_training_summary() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("sklears-bad-summary");
        let lease = single_worker_lease();

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");

        let metadata_path = artifact_dir.join("metadata.json");
        let mut metadata: crate::runtime::artifacts::RuntimeArtifactMetadata =
            serde_json::from_slice(&std::fs::read(&metadata_path).expect("read metadata"))
                .expect("deserialize metadata");
        metadata.training_summary = TrainingSummaryMetadata::raw_for_validation(9, 8, 0);
        std::fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let mut loaded = SklearsTreeExpert::new();
        let err = loaded
            .load(&artifact_dir)
            .expect_err("inconsistent training summary should fail");
        assert!(err.to_string().contains("training summary"));
    }

    #[test]
    fn sklears_load_uses_runtime_artifact_when_metadata_sidecar_missing() {
        let (x, y) = sample_three_class_dataset();
        let artifact_dir = unique_temp_dir("sklears-metadata-missing");
        let lease = single_worker_lease();

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y, &lease).expect("fit should succeed");
        expert.save(&artifact_dir).expect("save should succeed");

        let metadata_path = artifact_dir.join("metadata.json");
        assert!(
            metadata_path.exists(),
            "expected metadata sidecar at {}",
            metadata_path.display()
        );
        std::fs::remove_file(&metadata_path)
            .expect("remove metadata sidecar to trigger reconstruction");

        let mut loaded = SklearsTreeExpert::new();
        loaded
            .load(&artifact_dir)
            .expect("load should reconstruct metadata from runtime artifact");
        let probabilities = loaded
            .predict_proba(&x, &lease)
            .expect("prediction should succeed after metadata reconstruction");
        assert_eq!(probabilities.dim(), (x.n_samples(), 3));
    }

    #[test]
    fn sklears_keeps_split_information_beyond_f32_precision() {
        let values = (0..8)
            .map(|index| 1.0_f64 + index as f64 * 1.0e-9)
            .collect::<Vec<_>>();
        assert!(
            values.iter().all(|value| *value as f32 == 1.0_f32),
            "fixture must collapse when narrowed to f32"
        );
        let x = frame(vec![("sub_f32_resolution", values)]);
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let lease = single_worker_lease();
        let mut expert = SklearsTreeExpert::new();
        expert.min_samples_leaf = 1;
        expert.min_samples_split = 2;
        expert.max_thresholds_per_feature = 16;

        expert.fit(&x, &y, &lease).expect("fit should succeed");

        assert!(
            matches!(expert.root, Some(TreeNode::Split { .. })),
            "the project-owned tree must retain f64-only separation instead of collapsing samples"
        );
    }
}
