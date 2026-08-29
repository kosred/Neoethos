use anyhow::{Result, bail};
use ndarray::Array2;
use neoethos_core::storage::json::{
    JsonBackupWriteConfig, read_json as read_json_artifact,
    write_json_with_backup as write_json_artifact_with_backup,
};
use neoethos_data::FeatureFrame;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;

use crate::base::{
    feature_columns_from_frame, feature_frame_to_f64_array, try_build_runtime_artifact_metadata,
    validate_model_labels,
};
use crate::runtime::artifacts::{
    RuntimeArtifactMetadata, TrainingSummaryMetadata, default_three_class_label_mapping,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

pub const METADATA_FILE_NAME: &str = "metadata.json";
pub const MODEL_FILE_NAME: &str = "model.json";

pub fn normalize_statistical_device_policy(policy: &str) -> String {
    match crate::common::parse_cuda_device_policy(policy) {
        Ok(crate::common::CudaDevicePolicy::Auto) => "auto".to_string(),
        Ok(crate::common::CudaDevicePolicy::Cpu) => "cpu".to_string(),
        Ok(crate::common::CudaDevicePolicy::Gpu { ordinal: 0 }) if !policy.trim().contains(':') => {
            "gpu".to_string()
        }
        Ok(crate::common::CudaDevicePolicy::Gpu { ordinal }) => format!("gpu:{ordinal}"),
        Err(_) => policy.trim().to_ascii_lowercase(),
    }
}

/// Process-wide device policy for the statistical models, installed once at
/// startup from `models.statistical_device`.
///
/// Until 2026-08-02 the fallback here was the hard-coded literal `"auto"`,
/// which `cuda_kernel_enabled` treats as NOT-gpu — so the CUDA softmax kernel
/// could only ever be reached by setting an env var by hand. That was
/// consistent with the build, in which `statistical-gpu` was enabled by
/// nothing and `linear_gpu.rs` was not compiled at all. Now that the feature
/// is in the CUDA aggregate, the policy needs a real home in config.
static STATISTICAL_DEVICE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Install ONLY this registry. `pub(crate)` so the single crate-wide
/// installer in `runtime::install` is the one entry point callers see.
pub(crate) fn set_statistical_device(settings: &neoethos_core::Settings) {
    let configured = settings.models.statistical_device.trim();
    let _ = STATISTICAL_DEVICE.set(if configured.is_empty() {
        "cpu".to_string()
    } else {
        configured.to_string()
    });
}

/// Install the statistical device policy from `Settings`. Call once at
/// startup, before any model training; the first install wins. Called from
/// `neoethos_app::install_runtime_overrides_from_settings` (app + desktop)
/// and from the CLI's `main`.
///
/// Kept under its historical name because those callers already use it; it now
/// installs EVERY `neoethos-models` registry via
/// [`crate::runtime::install::install_model_runtime_from_settings`] and emits
/// the retired-env-var report.
pub fn install_statistical_runtime_from_settings(settings: &neoethos_core::Settings) {
    crate::runtime::install::install_model_runtime_from_settings(settings);
}

/// The configured policy (defaults to `"cpu"` when never installed — e.g. in
/// unit tests — which is the behaviour every build has had to date).
pub fn configured_statistical_device() -> &'static str {
    STATISTICAL_DEVICE.get_or_init(|| "cpu".to_string())
}

/// The device policy for one statistical model. An explicit
/// `models.model_param_overrides.<model>.device` wins; otherwise the shared
/// `models.statistical_device` policy applies. This lets CUDA-capable linear
/// models run on the card while a deliberately CPU-only statistical model is
/// pinned to CPU without any fallback at the execution boundary.
///
/// ## 2026-08-10 — the two env overrides are deleted
///
/// This used to read `NEOETHOS_BOT_<MODEL>_DEVICE`, then
/// `NEOETHOS_BOT_META_DEVICE`, and only then the config field — i.e. the field
/// the operator could see was the LOWEST-priority input, and the two that
/// outranked it appeared in no config file, no knob catalogue and no run
/// artifact. A shell export could put ElasticNet on the CUDA softmax kernel,
/// whose fitted weights differ from the CPU path, and the artifact would record
/// `cpu`.
///
pub fn statistical_device_policy(model_name: &str) -> String {
    let overrides = crate::runtime::capabilities::current_model_device_overrides();
    if overrides.per_model.contains_key(model_name) {
        return crate::runtime::capabilities::requested_runtime_device_policy(model_name);
    }
    configured_statistical_device().to_string()
}

/// Select the CPU implementation only for an explicitly CPU-resolved request.
/// A GPU or unresolved automatic request must be handled by its own complete
/// execution lane and may never degrade into this backend.
pub fn cpu_backend_for_policy(requested: &str, cpu_backend: &str) -> Result<String> {
    if matches!(
        crate::common::parse_cuda_device_policy(requested)?,
        crate::common::CudaDevicePolicy::Cpu
    ) {
        return Ok(cpu_backend.to_string());
    }
    bail!(
        "GpuOnly statistical request `{requested}` has no permission to execute CPU backend `{cpu_backend}`"
    )
}

fn ensure_finite_matrix(values: &Array2<f64>, context: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{context} contains non-finite values");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl FeatureScaler {
    pub fn fit(features: &Array2<f64>) -> Result<Self> {
        if features.nrows() == 0 || features.ncols() == 0 {
            bail!("feature scaler requires a non-empty feature matrix");
        }
        ensure_finite_matrix(features, "feature scaler input")?;

        let rows = features.nrows();
        let cols = features.ncols();
        let mut means = vec![0.0; cols];
        let mut stds = vec![1.0; cols];

        for col in 0..cols {
            let mut sum = 0.0_f64;
            for row in 0..features.nrows() {
                sum += features[(row, col)];
            }
            let mean = sum / rows as f64;
            means[col] = mean;

            let mut variance = 0.0_f64;
            for row in 0..features.nrows() {
                let centered = features[(row, col)] - mean;
                variance += centered * centered;
            }
            let std = (variance / rows as f64).sqrt();
            stds[col] = if std.is_finite() && std > 1e-12 {
                std
            } else {
                1.0
            };
        }

        Ok(Self { means, stds })
    }

    pub fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
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

pub fn feature_matrix_from_frame(frame: &FeatureFrame) -> Result<(Array2<f64>, Vec<String>)> {
    let features = feature_frame_to_f64_array(frame)?;
    let columns = feature_columns_from_frame(frame);
    Ok((features, columns))
}

pub fn remap_three_class_labels(labels: &[i32]) -> Result<Vec<usize>> {
    validate_model_labels(labels, labels.len())?;
    labels
        .iter()
        .copied()
        .map(|value| match value {
            -1 => Ok(2usize),
            0 => Ok(0usize),
            1 => Ok(1usize),
            other => {
                bail!("unsupported statistical-model label: {other}; expected one of -1, 0, 1")
            }
        })
        .collect()
}

pub fn ensure_feature_columns_match(expected: &[String], frame: &FeatureFrame) -> Result<()> {
    if expected.is_empty() {
        bail!("persisted statistical model is missing feature columns");
    }

    let actual = feature_columns_from_frame(frame);
    if expected != actual {
        bail!(
            "feature column mismatch for persisted statistical model; expected {:?}, got {:?}",
            expected,
            actual
        );
    }

    Ok(())
}

pub fn softmax_rows(logits: &Array2<f64>) -> Result<Array2<f64>> {
    let mut probabilities = logits.clone();
    for row in 0..probabilities.nrows() {
        let mut max_logit = f64::NEG_INFINITY;
        for col in 0..probabilities.ncols() {
            let value = probabilities[(row, col)];
            if !value.is_finite() {
                bail!("softmax row {row} contains non-finite logit at column {col}");
            }
            max_logit = max_logit.max(value);
        }

        if !max_logit.is_finite() {
            bail!("softmax row {row} has no finite maximum logit");
        }

        let mut normalizer = 0.0_f64;
        for col in 0..probabilities.ncols() {
            let value = (probabilities[(row, col)] - max_logit).exp();
            if !value.is_finite() {
                bail!("softmax row {row} overflowed at column {col}");
            }
            probabilities[(row, col)] = value;
            normalizer += value;
        }

        if !normalizer.is_finite() || normalizer <= f64::EPSILON {
            bail!("softmax row {row} has invalid normalization mass {normalizer}");
        }

        for col in 0..probabilities.ncols() {
            probabilities[(row, col)] /= normalizer;
        }
    }

    Ok(probabilities)
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
    use super::{
        configured_statistical_device, cpu_backend_for_policy,
        install_statistical_runtime_from_settings, normalize_statistical_device_policy,
        softmax_rows, statistical_device_policy,
    };
    use ndarray::Array2;

    /// The default keeps the statistical models on the CPU.
    ///
    /// Compiling `statistical/linear_gpu.rs` is a capability change and is
    /// unconditional in a CUDA build; letting a run USE it changes the fitted
    /// weights, so it waits for `models.statistical_device`. A fresh
    /// `Settings` must reproduce the uninstalled default, or the shipped
    /// config and the code would disagree about what "unset" means.
    #[test]
    fn statistical_device_defaults_to_cpu() {
        let settings = neoethos_core::Settings::default();
        assert_eq!(settings.models.statistical_device, "cpu");
        install_statistical_runtime_from_settings(&settings);
        assert_eq!(configured_statistical_device(), "cpu");
    }

    #[test]
    fn normalize_statistical_device_policy_preserves_rejected_vendor_input() {
        assert_eq!(normalize_statistical_device_policy("cuda:1"), "gpu:1");
        assert_eq!(normalize_statistical_device_policy("rocm:2"), "rocm:2");
        assert_eq!(normalize_statistical_device_policy("metal"), "metal");
        assert_eq!(normalize_statistical_device_policy("vulkan:0"), "vulkan:0");
        for rejected in ["rocm:2", "metal", "vulkan:0"] {
            assert!(
                crate::common::parse_cuda_device_policy(&normalize_statistical_device_policy(
                    rejected
                ))
                .is_err(),
                "non-CUDA vendor policy `{rejected}` must remain fail-closed"
            );
        }
    }

    #[test]
    fn cpu_backend_rejects_gpu_policy_instead_of_falling_back() {
        let error = cpu_backend_for_policy("cuda:0", "cpu_backend")
            .expect_err("a GPU request must never execute the CPU statistical backend");
        assert!(error.to_string().contains("GpuOnly"));

        assert_eq!(
            cpu_backend_for_policy("cpu", "cpu_backend").expect("CpuOnly backend"),
            "cpu_backend"
        );
    }

    #[test]
    fn softmax_rejects_non_finite_logits_instead_of_fabricating_neutral() {
        let logits =
            Array2::from_shape_vec((1, 3), vec![0.0_f64, f64::NAN, 1.0]).expect("shape logits");
        let error = softmax_rows(&logits)
            .expect_err("non-finite logits must not become a synthetic probability row");
        assert!(error.to_string().contains("non-finite"));
    }

    /// The retired per-model / subsystem device env vars must not move the
    /// statistical policy any more.
    #[test]
    fn statistical_device_policy_ignores_retired_env_overrides() {
        unsafe {
            std::env::set_var("NEOETHOS_BOT_META_DEVICE", "gpu:0");
            std::env::set_var("NEOETHOS_BOT_ELASTICNET_DEVICE", "gpu:3");
        }
        let policy = statistical_device_policy("elasticnet");
        unsafe {
            std::env::remove_var("NEOETHOS_BOT_META_DEVICE");
            std::env::remove_var("NEOETHOS_BOT_ELASTICNET_DEVICE");
        }
        assert_eq!(policy, configured_statistical_device());
    }
}
