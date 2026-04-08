use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::capabilities::{CapabilityState, ModelFamily};

pub const ONNX_EXPORT_STATUS_FILE_NAME: &str = "onnx_export_status.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnnxExportDisposition {
    Requested,
    Exported,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxExportStatus {
    pub model_name: String,
    pub capability_family: ModelFamily,
    pub capability_state: CapabilityState,
    pub requested: bool,
    pub disposition: OnnxExportDisposition,
    pub exporter: String,
    pub artifact_dir: PathBuf,
    pub output_path: Option<PathBuf>,
    pub feature_count: usize,
    pub sample_rows: usize,
    pub reason: Option<String>,
}

impl OnnxExportStatus {
    #[allow(clippy::too_many_arguments)]
    pub fn skipped(
        model_name: impl Into<String>,
        capability_family: ModelFamily,
        capability_state: CapabilityState,
        exporter: impl Into<String>,
        artifact_dir: PathBuf,
        feature_count: usize,
        sample_rows: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            capability_family,
            capability_state,
            requested: true,
            disposition: OnnxExportDisposition::Skipped,
            exporter: exporter.into(),
            artifact_dir,
            output_path: None,
            feature_count,
            sample_rows,
            reason: Some(reason.into()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn exported(
        model_name: impl Into<String>,
        capability_family: ModelFamily,
        capability_state: CapabilityState,
        exporter: impl Into<String>,
        artifact_dir: PathBuf,
        output_path: PathBuf,
        feature_count: usize,
        sample_rows: usize,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            capability_family,
            capability_state,
            requested: true,
            disposition: OnnxExportDisposition::Exported,
            exporter: exporter.into(),
            artifact_dir,
            output_path: Some(output_path),
            feature_count,
            sample_rows,
            reason: None,
        }
    }
}

pub fn write_onnx_export_status(path: &Path, status: &OnnxExportStatus) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create ONNX export dir {}", parent.display()))?;
    }

    let payload = serde_json::to_vec_pretty(status).context("serialize ONNX export status")?;
    std::fs::write(path, payload)
        .with_context(|| format!("write ONNX export status to {}", path.display()))
}
