use anyhow::{Context, Result, bail};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;

use crate::base::{
    build_runtime_artifact_metadata, dataframe_to_float32_array, feature_columns_from_dataframe,
};
use crate::runtime::artifacts::{
    RuntimeArtifactMetadata, TrainingSummaryMetadata, default_three_class_label_mapping,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const MODEL_FILE_NAME: &str = "model.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScaler {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
}

impl FeatureScaler {
    pub fn fit(features: &Array2<f32>) -> Result<Self> {
        if features.nrows() == 0 || features.ncols() == 0 {
            bail!("feature scaler requires a non-empty feature matrix");
        }

        let rows = features.nrows();
        let cols = features.ncols();
        let mut means = vec![0.0; cols];
        let mut stds = vec![1.0; cols];

        for col in 0..cols {
            let mut sum = 0.0_f32;
            for row in 0..features.nrows() {
                sum += features[(row, col)];
            }
            let mean = sum / rows as f32;
            means[col] = mean;

            let mut variance = 0.0_f32;
            for row in 0..features.nrows() {
                let centered = features[(row, col)] - mean;
                variance += centered * centered;
            }
            let std = (variance / rows as f32).sqrt();
            stds[col] = if std.is_finite() && std > 1e-8 {
                std
            } else {
                1.0
            };
        }

        Ok(Self { means, stds })
    }

    pub fn transform(&self, features: &Array2<f32>) -> Result<Array2<f32>> {
        if features.ncols() != self.means.len() || features.ncols() != self.stds.len() {
            bail!(
                "feature scaler dimension mismatch: {} cols vs means {} / stds {}",
                features.ncols(),
                self.means.len(),
                self.stds.len()
            );
        }

        let mut scaled = features.clone();
        for row in 0..scaled.nrows() {
            for col in 0..scaled.ncols() {
                scaled[(row, col)] = (scaled[(row, col)] - self.means[col]) / self.stds[col];
            }
        }
        Ok(scaled)
    }
}

pub fn feature_matrix_from_dataframe(df: &DataFrame) -> Result<(Array2<f32>, Vec<String>)> {
    let features = dataframe_to_float32_array(df)?;
    let columns = feature_columns_from_dataframe(df);
    Ok((features, columns))
}

pub fn remap_three_class_labels(y: &Series) -> Result<Vec<usize>> {
    let labels = y
        .cast(&DataType::Int32)
        .context("cast statistical labels to Int32")?;
    labels
        .i32()
        .context("access statistical labels as Int32")?
        .into_iter()
        .map(|value| match value {
            Some(-1) => Ok(2usize),
            Some(0) => Ok(0usize),
            Some(1) => Ok(1usize),
            Some(other) => {
                bail!("unsupported statistical-model label: {other}; expected one of -1, 0, 1")
            }
            None => bail!("statistical-model labels may not contain nulls"),
        })
        .collect()
}

pub fn ensure_feature_columns_match(expected: &[String], df: &DataFrame) -> Result<()> {
    if expected.is_empty() {
        bail!("persisted statistical model is missing feature columns");
    }

    let actual = feature_columns_from_dataframe(df);
    if expected != actual {
        bail!(
            "feature column mismatch for persisted statistical model; expected {:?}, got {:?}",
            expected,
            actual
        );
    }

    Ok(())
}

pub fn softmax_rows(logits: &Array2<f32>) -> Array2<f32> {
    let mut probabilities = logits.clone();
    for row in 0..probabilities.nrows() {
        let mut max_logit = f32::NEG_INFINITY;
        for col in 0..probabilities.ncols() {
            max_logit = max_logit.max(probabilities[(row, col)]);
        }

        let mut normalizer = 0.0_f32;
        for col in 0..probabilities.ncols() {
            let value = (probabilities[(row, col)] - max_logit).exp();
            probabilities[(row, col)] = value;
            normalizer += value;
        }

        if normalizer <= f32::EPSILON {
            for col in 0..probabilities.ncols() {
                probabilities[(row, col)] = 0.0;
            }
            if probabilities.ncols() > 0 {
                probabilities[(row, 0)] = 1.0;
            }
            continue;
        }

        for col in 0..probabilities.ncols() {
            probabilities[(row, col)] /= normalizer;
        }
    }

    probabilities
}

pub fn meta_runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        feature_columns,
        default_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create artifact directory {}", parent.display()))?;
    }

    let temp_path = path.with_extension("tmp");
    let payload = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", path.display()))?;
    std::fs::write(&temp_path, payload)
        .with_context(|| format!("write temporary artifact {}", temp_path.display()))?;
    std::fs::rename(&temp_path, path)
        .with_context(|| format!("rename artifact into {}", path.display()))?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let payload =
        std::fs::read(path).with_context(|| format!("read model artifact {}", path.display()))?;
    serde_json::from_slice(&payload)
        .with_context(|| format!("deserialize model artifact {}", path.display()))
}
