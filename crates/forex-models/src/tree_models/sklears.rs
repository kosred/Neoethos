use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::Path;

use crate::base::{dataframe_to_float32_array, feature_columns_from_dataframe, ExpertModel};
use crate::runtime::artifacts::TrainingSummaryMetadata;
use crate::runtime::capabilities::ModelFamily;
use crate::tree_models::common::{
    atomic_write, ensure_feature_columns_match, read_runtime_metadata, tree_artifact_paths,
    tree_runtime_metadata, write_runtime_metadata,
};

const MODEL_FILE_NAME: &str = "model.json";

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
    training_rows: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_thresholds_per_feature: usize,
}

impl SklearsTreeExpert {
    pub fn new() -> Self {
        Self {
            root: None,
            feature_columns: Vec::new(),
            training_rows: 0,
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

        if best_gain > 1e-6 {
            best_split
        } else {
            None
        }
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
        self.training_rows = features.nrows();
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
        let root = self
            .root
            .as_ref()
            .context("sklears-tree model not fitted")?;
        if self.feature_columns.is_empty() {
            bail!("sklears-tree model is missing feature columns");
        }
        validate_tree_node(root, self.feature_columns.len())?;
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
        write_runtime_metadata(
            &metadata_path,
            &tree_runtime_metadata(
                "sklears_tree",
                self.feature_columns.clone(),
                TrainingSummaryMetadata::new(self.training_rows, self.training_rows, 0),
            ),
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
        validate_tree_artifact(&artifact, metadata.feature_columns.len())?;
        self.max_depth = artifact.max_depth;
        self.min_samples_split = artifact.min_samples_split;
        self.min_samples_leaf = artifact.min_samples_leaf;
        self.max_thresholds_per_feature = artifact.max_thresholds_per_feature;
        self.root = Some(artifact.root);
        self.training_rows = metadata.training_summary.dataset_rows;
        self.feature_columns = metadata.feature_columns;
        Ok(())
    }
}
