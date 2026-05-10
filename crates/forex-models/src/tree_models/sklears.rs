use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::Path;

use crate::base::{ExpertModel, dataframe_to_float32_array, feature_columns_from_dataframe};
use crate::runtime::artifacts::TrainingSummaryMetadata;
use crate::runtime::capabilities::ModelFamily;
use crate::runtime::prediction::RuntimePrediction;
use crate::tree_models::common::{
    atomic_write, build_tree_runtime_predictions, default_training_summary,
    ensure_feature_columns_match, read_runtime_metadata, tree_artifact_paths,
    tree_runtime_metadata, write_runtime_metadata,
};

const MODEL_FILE_NAME: &str = "model.json";
const SKLEARS_RUNTIME_FILE_NAME: &str = "runtime.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TreeNode {
    Leaf {
        class_counts: [usize; 3],
        probabilities: [f32; 3],
    },
    Split {
        feature_index: usize,
        threshold: f32,
        probabilities: [f32; 3],
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTreeArtifact {
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_thresholds_per_feature: usize,
    root: TreeNode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SklearsRuntimeArtifact {
    feature_columns: Vec<String>,
    training_summary: TrainingSummaryMetadata,
}

fn validate_probability_vector(probabilities: &[f32; 3]) -> Result<()> {
    let mut sum = 0.0_f32;
    for value in probabilities {
        if !value.is_finite() || *value < 0.0 {
            bail!("sklears-tree probabilities must be finite and non-negative");
        }
        sum += *value;
    }
    if sum <= f32::EPSILON {
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
        let payload = std::fs::read(&runtime_path).with_context(|| {
            format!(
                "read sklears-tree runtime artifact {}",
                runtime_path.display()
            )
        })?;
        let artifact = serde_json::from_slice(&payload).with_context(|| {
            format!(
                "deserialize sklears-tree runtime artifact {}",
                runtime_path.display()
            )
        })?;
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

    fn labels_from_series(y: &Series) -> Result<Vec<usize>> {
        let labels = y
            .cast(&DataType::Int32)
            .context("cast sklears-tree labels to Int32")?;
        labels
            .i32()
            .context("access sklears-tree labels as Int32")?
            .into_iter()
            .map(|value| match value {
                Some(0) => Ok(0usize),
                Some(1) => Ok(1usize),
                Some(-1) => Ok(2usize),
                Some(other) => {
                    bail!("unsupported sklears-tree label: {other}; expected one of -1, 0, 1")
                }
                None => bail!("sklears-tree labels may not contain nulls"),
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

    fn probabilities_from_counts(counts: [usize; 3]) -> [f32; 3] {
        let total = counts.iter().sum::<usize>() as f32;
        if total <= f32::EPSILON {
            return [1.0, 0.0, 0.0];
        }
        [
            counts[0] as f32 / total,
            counts[1] as f32 / total,
            counts[2] as f32 / total,
        ]
    }

    fn is_pure(counts: [usize; 3]) -> bool {
        counts.iter().filter(|count| **count > 0).count() <= 1
    }

    fn gini_from_counts(counts: [usize; 3]) -> f32 {
        let total = counts.iter().sum::<usize>() as f32;
        if total <= f32::EPSILON {
            return 0.0;
        }
        1.0 - counts
            .iter()
            .map(|count| {
                let prob = *count as f32 / total;
                prob * prob
            })
            .sum::<f32>()
    }

    fn candidate_thresholds(
        &self,
        features: &Array2<f32>,
        rows: &[usize],
        feature_idx: usize,
    ) -> Vec<f32> {
        let mut values = rows
            .iter()
            .map(|row| features[(*row, feature_idx)])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() <= f32::EPSILON);
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

        let step = ((midpoints.len() as f32) / (self.max_thresholds_per_feature as f32))
            .ceil()
            .max(1.0) as usize;
        midpoints
            .into_iter()
            .step_by(step)
            .take(self.max_thresholds_per_feature)
            .collect()
    }

    fn split_rows(
        features: &Array2<f32>,
        rows: &[usize],
        feature_idx: usize,
        threshold: f32,
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
        features: &Array2<f32>,
        labels: &[usize],
        rows: &[usize],
    ) -> Option<(usize, f32, Vec<usize>, Vec<usize>)> {
        let parent_counts = Self::class_counts(labels, rows);
        let parent_gini = Self::gini_from_counts(parent_counts);
        let mut best_gain = f32::NEG_INFINITY;
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
                let left_weight = left_rows.len() as f32 / rows.len() as f32;
                let right_weight = right_rows.len() as f32 / rows.len() as f32;
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
        features: &Array2<f32>,
        labels: &[usize],
        rows: &[usize],
        depth: usize,
    ) -> TreeNode {
        let counts = Self::class_counts(labels, rows);
        let probabilities = Self::probabilities_from_counts(counts);
        if depth >= self.max_depth || rows.len() < self.min_samples_split || Self::is_pure(counts) {
            return TreeNode::Leaf {
                class_counts: counts,
                probabilities,
            };
        }

        if let Some((feature_index, threshold, left_rows, right_rows)) =
            self.best_split(features, labels, rows)
        {
            return TreeNode::Split {
                feature_index,
                threshold,
                probabilities,
                left: Box::new(self.build_node(features, labels, &left_rows, depth + 1)),
                right: Box::new(self.build_node(features, labels, &right_rows, depth + 1)),
            };
        }

        TreeNode::Leaf {
            class_counts: counts,
            probabilities,
        }
    }

    fn predict_row_probabilities(
        node: &TreeNode,
        features: &Array2<f32>,
        row_idx: usize,
    ) -> [f32; 3] {
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
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let features =
            dataframe_to_float32_array(x).context("build sklears-tree feature matrix")?;
        if features.nrows() == 0 || features.ncols() == 0 {
            bail!("sklears-tree requires a non-empty feature matrix");
        }
        let labels = Self::labels_from_series(y)?;
        if labels.len() != features.nrows() {
            bail!(
                "sklears-tree row/label mismatch: {} rows vs {} labels",
                features.nrows(),
                labels.len()
            );
        }

        let rows = (0..features.nrows()).collect::<Vec<_>>();
        self.root = Some(self.build_node(&features, &labels, &rows, 0));
        self.feature_columns = feature_columns_from_dataframe(x);
        self.training_summary = Some(default_training_summary(x));
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let root = self
            .root
            .as_ref()
            .context("sklears-tree model not fitted")?;
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let features =
            dataframe_to_float32_array(x).context("build sklears-tree inference matrix")?;
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
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_thresholds_per_feature: self.max_thresholds_per_feature,
            root: root.clone(),
        };
        let (model_path, metadata_path) = tree_artifact_paths(path, MODEL_FILE_NAME);
        let payload =
            serde_json::to_vec_pretty(&artifact).context("serialize sklears-tree artifact")?;
        atomic_write(&model_path, &payload)?;
        let runtime_artifact = SklearsRuntimeArtifact {
            feature_columns: self.feature_columns.clone(),
            training_summary: self.stored_training_summary(),
        };
        let runtime_payload = serde_json::to_vec_pretty(&runtime_artifact)
            .context("serialize sklears-tree runtime artifact")?;
        atomic_write(&path.join(SKLEARS_RUNTIME_FILE_NAME), &runtime_payload)?;
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
        let payload = std::fs::read(&model_path)
            .with_context(|| format!("read sklears-tree artifact {}", model_path.display()))?;
        let artifact: DecisionTreeArtifact =
            serde_json::from_slice(&payload).with_context(|| {
                format!("deserialize sklears-tree artifact {}", model_path.display())
            })?;
        let runtime_artifact = Self::read_runtime_artifact(path)?;
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
    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_state_ready()?;
        let probabilities = self.predict_proba(x)?;
        build_tree_runtime_predictions(
            "sklears_tree",
            &probabilities,
            true,
            "sklears_tree_native",
            None,
            "native_sklears_tree_unavailable",
            "sklears_tree_unknown",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertModel, SklearsTreeExpert};
    use crate::runtime::artifacts::TrainingSummaryMetadata;
    use polars::df;
    use polars::prelude::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_three_class_dataset() -> (DataFrame, Series) {
        let x = df![
            "momentum" => &[0.96, 0.93, 0.89, 0.07, 0.03, 0.11, -0.94, -0.91, -0.88],
            "trend" => &[0.87, 0.91, 0.86, 0.01, -0.02, 0.04, -0.9, -0.86, -0.93],
            "volatility" => &[0.62, 0.58, 0.6, 0.2, 0.18, 0.23, 0.69, 0.66, 0.64],
        ]
        .expect("build training dataframe");
        let y = Series::new("label".into(), &[1_i32, 1, 1, 0, 0, 0, -1, -1, -1]);
        (x, y)
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

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y).expect("fit should succeed");
        expert.training_summary = None;

        let err = expert
            .save(&artifact_dir)
            .expect_err("save should fail without training summary");
        assert!(err.to_string().contains("training summary"));
    }

    #[test]
    fn sklears_predict_runtime_returns_runtime_predictions() {
        let (x, y) = sample_three_class_dataset();

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y).expect("fit should succeed");

        let predictions = expert
            .predict_runtime(&x)
            .expect("runtime prediction should succeed");
        assert_eq!(predictions.len(), x.height());
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

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y).expect("fit should succeed");
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

        let mut expert = SklearsTreeExpert::new();
        expert.fit(&x, &y).expect("fit should succeed");
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
            .predict_proba(&x)
            .expect("prediction should succeed after metadata reconstruction");
        assert_eq!(probabilities.dim(), (x.height(), 3));
    }
}
