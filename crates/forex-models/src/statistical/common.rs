use anyhow::{Context, Result, bail};
use forex_core::storage::json::{
    JsonBackupWriteConfig, read_json as read_json_artifact,
    write_json_with_backup as write_json_artifact_with_backup,
};
use ndarray::Array2;
use polars::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;

use crate::base::{
    dataframe_to_float32_array, feature_columns_from_dataframe, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{
    RuntimeArtifactMetadata, TrainingSummaryMetadata, default_three_class_label_mapping,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const MODEL_FILE_NAME: &str = "model.json";

pub fn normalize_statistical_device_policy(policy: &str) -> String {
    crate::common::normalize_vendor_device_policy(policy, &[])
}

pub fn runtime_backend_with_gpu_fallback(
    model_name: &str,
    cpu_backend: &str,
) -> (Option<String>, Option<String>) {
    let model_key = format!(
        "FOREX_BOT_{}_DEVICE",
        model_name.trim().to_ascii_uppercase().replace('-', "_")
    );
    let requested = std::env::var(&model_key)
        .or_else(|_| std::env::var("FOREX_BOT_META_DEVICE"))
        .unwrap_or_else(|_| "auto".to_string());
    let normalized = normalize_statistical_device_policy(&requested);
    let degraded_reason = if normalized == "gpu" || normalized.starts_with("gpu:") {
        Some(format!(
            "requested device policy `{normalized}`; statistical backend currently executes on CPU"
        ))
    } else {
        None
    };
    (Some(cpu_backend.to_string()), degraded_reason)
}

fn ensure_finite_matrix(values: &Array2<f32>, context: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{context} contains non-finite values");
    }
    Ok(())
}

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
        ensure_finite_matrix(features, "feature scaler input")?;

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
        ensure_finite_matrix(features, "feature scaler transform input")?;

        let mut scaled = features.clone();
        for row in 0..scaled.nrows() {
            for col in 0..scaled.ncols() {
                scaled[(row, col)] = (scaled[(row, col)] - self.means[col]) / self.stds[col];
            }
        }
        ensure_finite_matrix(&scaled, "feature scaler output")?;
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
        let mut row_is_valid = true;
        for col in 0..probabilities.ncols() {
            let value = probabilities[(row, col)];
            if !value.is_finite() {
                row_is_valid = false;
                break;
            }
            max_logit = max_logit.max(value);
        }

        if !row_is_valid || !max_logit.is_finite() {
            for col in 0..probabilities.ncols() {
                probabilities[(row, col)] = 0.0;
            }
            if probabilities.ncols() > 0 {
                probabilities[(row, 0)] = 1.0;
            }
            continue;
        }

        let mut normalizer = 0.0_f32;
        for col in 0..probabilities.ncols() {
            let value = (probabilities[(row, col)] - max_logit).exp();
            if !value.is_finite() {
                row_is_valid = false;
                break;
            }
            probabilities[(row, col)] = value;
            normalizer += value;
        }

        if !row_is_valid || !normalizer.is_finite() || normalizer <= f32::EPSILON {
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
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Meta,
        CapabilityState::Implemented,
        feature_columns,
        default_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    write_json_artifact_with_backup(
        path,
        value,
        JsonBackupWriteConfig {
            artifact_label: "statistical model artifact",
            temp_extension: "tmp",
            backup_extension: "bak",
        },
    )
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    read_json_artifact(path, "statistical model")
}

#[cfg(test)]
mod tests {
    use super::{normalize_statistical_device_policy, runtime_backend_with_gpu_fallback};

    #[test]
    fn normalize_statistical_device_policy_accepts_vendor_aliases() {
        assert_eq!(normalize_statistical_device_policy("cuda:1"), "gpu:1");
        assert_eq!(normalize_statistical_device_policy("rocm:2"), "gpu:2");
        assert_eq!(normalize_statistical_device_policy("metal"), "gpu");
        assert_eq!(normalize_statistical_device_policy("vulkan:0"), "gpu:0");
    }

    #[test]
    fn runtime_backend_marks_gpu_request_as_cpu_fallback() {
        unsafe {
            std::env::set_var("FOREX_BOT_META_DEVICE", "gpu:0");
        }
        let (backend, degraded_reason) =
            runtime_backend_with_gpu_fallback("elasticnet", "cpu_backend");
        assert_eq!(backend.as_deref(), Some("cpu_backend"));
        assert!(
            degraded_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("currently executes on CPU"))
        );
        unsafe {
            std::env::remove_var("FOREX_BOT_META_DEVICE");
        }
    }
}
