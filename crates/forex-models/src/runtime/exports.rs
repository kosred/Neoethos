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

fn validate_onnx_export_status(status: &OnnxExportStatus) -> Result<()> {
    if !status.requested {
        anyhow::bail!("ONNX export status requested flag must remain true");
    }
    if status.model_name.trim().is_empty() {
        anyhow::bail!("ONNX export status model_name must not be empty");
    }
    if status.exporter.trim().is_empty() {
        anyhow::bail!("ONNX export status exporter must not be empty");
    }
    if status.artifact_dir.as_os_str().is_empty() {
        anyhow::bail!("ONNX export status artifact_dir must not be empty");
    }
    if status.feature_count == 0 {
        anyhow::bail!("ONNX export status feature_count must be non-zero");
    }
    if status.sample_rows == 0 {
        anyhow::bail!("ONNX export status sample_rows must be non-zero");
    }
    match status.disposition {
        OnnxExportDisposition::Requested => {
            if status.output_path.is_some() {
                anyhow::bail!("requested ONNX export status must not contain output_path");
            }
        }
        OnnxExportDisposition::Exported => {
            if status.reason.is_some() {
                anyhow::bail!("exported ONNX export status must not contain skip reason");
            }
            if status
                .output_path
                .as_ref()
                .is_none_or(|path| path.as_os_str().is_empty())
            {
                anyhow::bail!("exported ONNX export status must contain non-empty output_path");
            }
        }
        OnnxExportDisposition::Skipped => {
            if status.output_path.is_some() {
                anyhow::bail!("skipped ONNX export status must not contain output_path");
            }
            if status
                .reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
            {
                anyhow::bail!("skipped ONNX export status must contain non-empty reason");
            }
        }
    }
    Ok(())
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
        let status = Self {
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
        };
        validate_onnx_export_status(&status).expect("skipped ONNX export status must be valid");
        status
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
        let status = Self {
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
        };
        validate_onnx_export_status(&status).expect("exported ONNX export status must be valid");
        status
    }
}

pub fn write_onnx_export_status(path: &Path, status: &OnnxExportStatus) -> Result<()> {
    validate_onnx_export_status(status)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create ONNX export dir {}", parent.display()))?;
    }

    let temp_path = path.with_extension("tmp_onnx_status");
    let backup_path = path.with_extension("bak_onnx_status");
    let payload = serde_json::to_vec_pretty(status).context("serialize ONNX export status")?;
    if temp_path.exists() {
        std::fs::remove_file(&temp_path).with_context(|| {
            format!(
                "remove stale staged ONNX export status {}",
                temp_path.display()
            )
        })?;
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove stale backup ONNX export status {}",
                backup_path.display()
            )
        })?;
    }
    std::fs::write(&temp_path, payload)
        .with_context(|| format!("write staged ONNX export status to {}", temp_path.display()))?;
    if path.exists() {
        std::fs::rename(path, &backup_path)
            .with_context(|| format!("backup current ONNX export status {}", path.display()))?;
    }
    if let Err(error) = std::fs::rename(&temp_path, path) {
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, path);
        } else if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }
        anyhow::bail!(
            "write ONNX export status to {} failed: {}",
            path.display(),
            error
        );
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path).with_context(|| {
            format!("remove backup ONNX export status {}", backup_path.display())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_status_rejects_zero_feature_count() {
        let status = OnnxExportStatus {
            model_name: "mlp".to_string(),
            capability_family: ModelFamily::Deep,
            capability_state: CapabilityState::Implemented,
            requested: true,
            disposition: OnnxExportDisposition::Exported,
            exporter: "native".to_string(),
            artifact_dir: PathBuf::from("artifacts"),
            output_path: Some(PathBuf::from("model.onnx")),
            feature_count: 0,
            sample_rows: 32,
            reason: None,
        };

        let err = validate_onnx_export_status(&status)
            .expect_err("zero feature_count must fail")
            .to_string();
        assert!(err.contains("feature_count"));
    }

    #[test]
    fn skipped_status_requires_reason() {
        let status = OnnxExportStatus {
            model_name: "mlp".to_string(),
            capability_family: ModelFamily::Deep,
            capability_state: CapabilityState::Implemented,
            requested: true,
            disposition: OnnxExportDisposition::Skipped,
            exporter: "native".to_string(),
            artifact_dir: PathBuf::from("artifacts"),
            output_path: None,
            feature_count: 12,
            sample_rows: 32,
            reason: Some("".to_string()),
        };

        let err = validate_onnx_export_status(&status)
            .expect_err("blank skip reason must fail")
            .to_string();
        assert!(err.contains("reason"));
    }

    #[test]
    fn export_status_requires_requested_flag() {
        let status = OnnxExportStatus {
            model_name: "mlp".to_string(),
            capability_family: ModelFamily::Deep,
            capability_state: CapabilityState::Implemented,
            requested: false,
            disposition: OnnxExportDisposition::Requested,
            exporter: "native".to_string(),
            artifact_dir: PathBuf::from("artifacts"),
            output_path: None,
            feature_count: 12,
            sample_rows: 32,
            reason: None,
        };

        let err = validate_onnx_export_status(&status)
            .expect_err("requested=false must fail")
            .to_string();
        assert!(err.contains("requested flag"));
    }
}
