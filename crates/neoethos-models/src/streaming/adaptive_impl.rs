use anyhow::{Context, Result, bail};
#[cfg(feature = "adaptive-models")]
use irithyll::ensemble::config::DriftDetectorType;
#[cfg(feature = "adaptive-models")]
use irithyll::loss::logistic::LogisticLoss;
#[cfg(feature = "adaptive-models")]
use irithyll::serde_support::{load_model, save_model_with};
#[cfg(feature = "adaptive-models")]
use irithyll::{DynSGBT, LossType, SGBT, SGBTConfig, Sample};
use ndarray::{Array1, Array2};
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::base::{
    ExpertModel, build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, append_runtime_degraded_reason, gpu_policy_cpu_fallback_reason,
    requested_runtime_device_policy,
};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, MODEL_FILE_NAME, ensure_feature_columns_match,
    feature_matrix_from_frame, read_json, remap_three_class_labels, softmax_rows, write_json,
};
#[cfg(feature = "adaptive-models")]
use tracing::warn;

#[cfg(feature = "statistical-gpu")]
use super::adaptive_gpu::{
    try_fit_passive_aggressive_cuda_full_pipeline,
    try_predict_passive_aggressive_cuda_full_pipeline,
    validate_passive_aggressive_cuda_device_identity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum AdaptiveModelKind {
    PassiveAggressive,
    Hoeffding,
}

impl AdaptiveModelKind {
    fn model_name(self) -> &'static str {
        match self {
            Self::PassiveAggressive => "online_pa",
            Self::Hoeffding => "online_hoeffding",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HoeffdingFallbackBasis {
    Linear,
    Quadratic,
}

impl HoeffdingFallbackBasis {
    fn expanded_dim(self, feature_dim: usize) -> usize {
        match self {
            Self::Linear => feature_dim,
            Self::Quadratic => feature_dim.saturating_mul(2),
        }
    }
}

fn hoeffding_fallback_basis(params: &HashMap<String, String>) -> HoeffdingFallbackBasis {
    match params
        .get("fallback_basis")
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("quadratic" | "poly2") => HoeffdingFallbackBasis::Quadratic,
        _ => HoeffdingFallbackBasis::Linear,
    }
}

fn validate_hoeffding_fallback_basis_param(params: &HashMap<String, String>) -> Result<()> {
    if let Some(value) = params.get("fallback_basis") {
        let normalized = value.trim().to_ascii_lowercase();
        if !normalized.is_empty()
            && !matches!(normalized.as_str(), "linear" | "quadratic" | "poly2")
        {
            bail!(
                "online_hoeffding fallback_basis `{}` is not supported; expected linear or quadratic",
                value.trim()
            );
        }
    }
    Ok(())
}

fn ensure_label_count(model_name: &str, rows: usize, labels: usize) -> Result<()> {
    if rows != labels {
        bail!("{model_name} row/label mismatch: {rows} feature rows vs {labels} labels");
    }
    Ok(())
}

fn expand_hoeffding_fallback_features(values: &[f64], basis: HoeffdingFallbackBasis) -> Vec<f64> {
    match basis {
        HoeffdingFallbackBasis::Linear => values.to_vec(),
        HoeffdingFallbackBasis::Quadratic => {
            let mut expanded = Vec::with_capacity(values.len().saturating_mul(2));
            expanded.extend_from_slice(values);
            expanded.extend(values.iter().map(|value| {
                let clamped = value.clamp(-1.0e9_f64, 1.0e9_f64);
                clamped * clamped
            }));
            expanded
        }
    }
}

fn expand_hoeffding_fallback_matrix(
    features: &Array2<f64>,
    basis: HoeffdingFallbackBasis,
) -> Result<Array2<f64>> {
    match basis {
        HoeffdingFallbackBasis::Linear => Ok(features.clone()),
        HoeffdingFallbackBasis::Quadratic => {
            let cols = features.ncols();
            let mut expanded = Vec::with_capacity(features.nrows() * cols.saturating_mul(2));
            for row in 0..features.nrows() {
                let row_values = features.row(row).iter().copied().collect::<Vec<_>>();
                expanded.extend(expand_hoeffding_fallback_features(&row_values, basis));
            }
            Array2::from_shape_vec((features.nrows(), cols.saturating_mul(2)), expanded)
                .context("shape quadratic online_hoeffding fallback features")
        }
    }
}

fn adaptive_runtime_metadata(
    model_name: &str,
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        model_name,
        ModelFamily::Adaptive,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn resolve_adaptive_runtime_metadata(
    path: &Path,
    model_name: &str,
    feature_columns: &[String],
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    let reconstructed =
        adaptive_runtime_metadata(model_name, feature_columns.to_vec(), dataset_rows)?;
    validate_adaptive_metadata(&reconstructed, model_name)?;
    if reconstructed.feature_columns != feature_columns {
        bail!(
            "{} reconstructed feature columns mismatch: expected {:?}, got {:?}",
            model_name,
            feature_columns,
            reconstructed.feature_columns
        );
    }
    if reconstructed.training_summary.dataset_rows != dataset_rows {
        bail!(
            "{} reconstructed dataset rows mismatch: expected {}, got {}",
            model_name,
            dataset_rows,
            reconstructed.training_summary.dataset_rows
        );
    }

    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_adaptive_metadata(&metadata, model_name).with_context(|| {
                format!(
                    "{} metadata sidecar mismatch with reconstructed metadata at {}",
                    model_name,
                    metadata_path.display()
                )
            })?;
            if metadata.model_name != reconstructed.model_name
                || metadata.family != reconstructed.family
                || metadata.state != reconstructed.state
                || metadata.feature_columns != reconstructed.feature_columns
                || metadata.label_mapping != reconstructed.label_mapping
                || metadata.training_summary.dataset_rows
                    != reconstructed.training_summary.dataset_rows
                || metadata.training_summary.train_rows != reconstructed.training_summary.train_rows
                || metadata.training_summary.val_rows != reconstructed.training_summary.val_rows
            {
                bail!(
                    "{} metadata sidecar mismatch with reconstructed metadata at {}",
                    model_name,
                    metadata_path.display()
                );
            }
            Ok(metadata)
        }
        Err(file_err) => {
            tracing::warn!(
                model = %model_name,
                path = %metadata_path.display(),
                error = %file_err,
                "adaptive metadata sidecar missing/unreadable; using reconstructed metadata"
            );
            Ok(reconstructed)
        }
    }
}

fn staged_adaptive_file(path: &Path, file_name: &str) -> PathBuf {
    path.join(format!("{file_name}.tmp"))
}

fn backup_adaptive_file(path: &Path, file_name: &str) -> PathBuf {
    path.join(format!("{file_name}.bak"))
}

fn cleanup_adaptive_temp_file(path: &Path) {
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
}

fn stage_adaptive_target(target: &Path, backup: &Path, staged: Option<&Path>) -> Result<()> {
    if backup.exists() {
        std::fs::remove_file(backup)
            .with_context(|| format!("remove adaptive backup {}", backup.display()))?;
    }
    if target.exists() {
        std::fs::rename(target, backup).with_context(|| {
            format!(
                "rotate adaptive artifact target {} to backup {}",
                target.display(),
                backup.display()
            )
        })?;
    }
    if let Some(staged_path) = staged {
        std::fs::rename(staged_path, target).with_context(|| {
            format!(
                "promote adaptive staged file {} to {}",
                staged_path.display(),
                target.display()
            )
        })?;
    }
    Ok(())
}

fn restore_adaptive_backup(target: &Path, backup: &Path) {
    if target.exists() {
        let _ = std::fs::remove_file(target);
    }
    if backup.exists() {
        let _ = std::fs::rename(backup, target);
    }
}

fn validate_adaptive_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_model_name: &str,
) -> Result<()> {
    if metadata.model_name != expected_model_name {
        bail!(
            "adaptive artifact model mismatch: expected {}, got {}",
            expected_model_name,
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Adaptive {
        bail!(
            "adaptive artifact family mismatch: expected {:?}, got {:?}",
            ModelFamily::Adaptive,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "adaptive artifact state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("adaptive artifact label mapping mismatch");
    }
    if metadata.feature_columns.is_empty() {
        bail!("adaptive artifact metadata must contain at least one feature column");
    }
    if metadata.training_summary.dataset_rows == 0 {
        bail!("adaptive artifact metadata must record non-zero training rows");
    }
    if metadata.training_summary.dataset_rows
        != metadata.training_summary.train_rows + metadata.training_summary.val_rows
    {
        bail!("adaptive artifact metadata training summary is inconsistent");
    }
    Ok(())
}

fn validate_passive_aggressive_artifact(artifact: &PassiveAggressiveArtifact) -> Result<()> {
    if artifact.model_name != AdaptiveModelKind::PassiveAggressive.model_name() {
        bail!(
            "online_pa artifact model mismatch: expected {}, got {}",
            AdaptiveModelKind::PassiveAggressive.model_name(),
            artifact.model_name
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("online_pa artifact must contain at least one feature column");
    }
    if artifact.dataset_rows == 0 {
        bail!("online_pa artifact must contain at least one training row");
    }

    let expected_features = artifact.feature_columns.len();
    if artifact.scaler.means.len() != expected_features
        || artifact.scaler.stds.len() != expected_features
    {
        bail!(
            "online_pa scaler mismatch: expected {} features, received means {} and stds {}",
            expected_features,
            artifact.scaler.means.len(),
            artifact.scaler.stds.len()
        );
    }
    if artifact.weights.nrows() != 3 || artifact.weights.ncols() != expected_features {
        bail!(
            "online_pa weight matrix mismatch: expected 3x{}, received {:?}",
            expected_features,
            artifact.weights.dim()
        );
    }
    if artifact.bias.len() != 3 {
        bail!(
            "online_pa bias mismatch: expected 3 entries, received {}",
            artifact.bias.len()
        );
    }
    if artifact
        .scaler
        .means
        .iter()
        .chain(artifact.scaler.stds.iter())
        .any(|value| !value.is_finite())
    {
        bail!("online_pa scaler contains non-finite values");
    }
    if artifact.scaler.stds.iter().any(|value| *value <= 0.0) {
        bail!("online_pa scaler stds must stay strictly positive");
    }
    if artifact.weights.iter().any(|value| !value.is_finite()) {
        bail!("online_pa weights contain non-finite values");
    }
    if artifact.bias.iter().any(|value| !value.is_finite()) {
        bail!("online_pa bias contains non-finite values");
    }
    if !artifact.aggressiveness.is_finite() || artifact.aggressiveness <= 0.0 {
        bail!("online_pa aggressiveness must be finite and positive");
    }
    if artifact.epochs == 0 {
        bail!("online_pa epochs must stay positive");
    }
    match artifact.training_semantics_schema.as_str() {
        ONLINE_PA_LEGACY_TRAINING_SEMANTICS_V1 => {
            if artifact.training_backend != "online_pa_cpu_legacy_v1"
                || artifact.effective_device_policy != "cpu"
                || artifact.class_slack_weight_policy
                    != ONLINE_PA_LEGACY_CLASS_SLACK_WEIGHT_POLICY_V1
                || artifact.class_slack_weights.is_some()
                || artifact.cuda_training_evidence.is_some()
            {
                bail!(
                    "online_pa legacy-v1 artifact must bind its implicit loss-multiplier policy, CPU training, and no CUDA evidence"
                );
            }
        }
        ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2 => {
            let ordinal =
                match crate::common::parse_cuda_device_policy(&artifact.effective_device_policy)? {
                    crate::common::CudaDevicePolicy::Gpu { ordinal } => ordinal,
                    _ => bail!("online_pa CUDA-v2 artifact must bind an explicit GPU ordinal"),
                };
            if artifact.class_slack_weight_policy != ONLINE_PA_CUDA_CLASS_SLACK_WEIGHT_POLICY_V1 {
                bail!(
                    "online_pa CUDA-v2 class-slack weight policy mismatch: expected {}, got {}",
                    ONLINE_PA_CUDA_CLASS_SLACK_WEIGHT_POLICY_V1,
                    artifact.class_slack_weight_policy
                );
            }
            let class_slack_weights = artifact
                .class_slack_weights
                .context("online_pa CUDA-v2 artifact is missing class-slack weights")?;
            if class_slack_weights
                .iter()
                .any(|weight| !weight.is_finite() || !(0.5..=4.0).contains(weight))
            {
                bail!(
                    "online_pa CUDA-v2 class-slack weights must be finite within the bound policy [0.5, 4.0]"
                );
            }
            let evidence = artifact
                .cuda_training_evidence
                .as_ref()
                .context("online_pa CUDA-v2 artifact is missing CUDA evidence")?;
            let rows = u64::try_from(artifact.dataset_rows)
                .context("online_pa dataset rows exceed evidence width")?;
            let cols = u64::try_from(expected_features)
                .context("online_pa feature count exceeds evidence width")?;
            let epochs =
                u64::try_from(artifact.epochs).context("online_pa epochs exceed evidence width")?;
            let feature_bytes = rows
                .checked_mul(cols)
                .and_then(|value| value.checked_mul(8))
                .context("online_pa CUDA-v2 feature byte evidence overflow")?;
            let label_bytes = rows
                .checked_mul(4)
                .context("online_pa CUDA-v2 label byte evidence overflow")?;
            let weight_bytes = cols
                .checked_mul(3)
                .and_then(|value| value.checked_mul(8))
                .context("online_pa CUDA-v2 weight byte evidence overflow")?;
            let bias_bytes = 3_u64 * 8;
            let class_weight_bytes = 3_u64 * 8;
            let arithmetic_status_bytes = 4_u64;
            let legacy_training_only_h2d = feature_bytes
                .checked_add(label_bytes)
                .and_then(|value| value.checked_add(class_weight_bytes))
                .and_then(|value| value.checked_add(weight_bytes))
                .and_then(|value| value.checked_add(bias_bytes))
                .and_then(|value| value.checked_add(arithmetic_status_bytes))
                .context("online_pa CUDA-v2 H2D evidence overflow")?;
            let training_d2h = weight_bytes
                .checked_add(bias_bytes)
                .and_then(|value| value.checked_add(arithmetic_status_bytes))
                .context("online_pa CUDA-v2 D2H evidence overflow")?;
            let expected_visits = rows
                .checked_mul(epochs)
                .context("online_pa CUDA-v2 visit evidence overflow")?;
            let expected_training_chunks_per_epoch =
                rows.div_ceil(ONLINE_PA_CUDA_TRAINING_ROWS_PER_LAUNCH_V3);
            let expected_training_launches = expected_training_chunks_per_epoch
                .checked_mul(epochs)
                .context("online_pa CUDA-v3 training launch evidence overflow")?;
            let expected_whole_fit_launches = expected_training_launches
                .checked_add(9)
                .context("online_pa CUDA-v3 whole-fit launch evidence overflow")?;
            if evidence.ordered_sample_visits != expected_visits {
                bail!("online_pa CUDA-v2 training evidence does not match artifact dimensions");
            }

            match evidence.full_pipeline.as_ref() {
                None => {
                    let expected_backend = format!("online_pa_cuda[cuda:{ordinal}]");
                    let launch_count_matches_schema = match evidence.evidence_scope_schema.as_str()
                    {
                        // Read-only compatibility for pre-r3 training-only
                        // artifacts. No production path can infer from these.
                        "neoethos.online_pa.cuda_evidence.training_stage.v1" => {
                            evidence.kernel_launch_count == 1
                        }
                        "neoethos.online_pa.cuda_evidence.training_stage.v2" => {
                            evidence.kernel_launch_count == expected_training_launches
                        }
                        _ => false,
                    };
                    if artifact.training_backend != expected_backend
                        || !launch_count_matches_schema
                        || evidence.host_to_device_bytes != legacy_training_only_h2d
                        || evidence.device_to_host_bytes != training_d2h
                    {
                        bail!(
                            "online_pa CUDA-v2 training-only evidence/backend mismatch; expected {expected_backend}"
                        );
                    }
                }
                Some(full_pipeline) => {
                    let expected_backend = format!("online_pa_cuda_full_pipeline[cuda:{ordinal}]");
                    let expected_inference_backend =
                        format!("online_pa_cuda_fused_inference[cuda:{ordinal}]");
                    let expected_effective = format!("gpu:{ordinal}");
                    if artifact.training_backend != expected_backend
                        || full_pipeline.training_backend != expected_backend
                        || full_pipeline.bound_inference_backend != expected_inference_backend
                        || full_pipeline.preprocessing_backend
                            != ONLINE_PA_CUDA_PREPROCESSING_BACKEND_V2
                        || full_pipeline.execution_pipeline_schema
                            != ONLINE_PA_CUDA_FULL_PIPELINE_SCHEMA_V3
                        || full_pipeline.evidence_scope_schema
                            != "neoethos.online_pa.cuda_pipeline_stages.v3"
                        || full_pipeline.loss_cost_policy != "rho(y,r)=1; PA-I cap=C*w_y"
                        || full_pipeline.residency_scope != "call_scoped"
                        || full_pipeline.persistent_model_buffers
                        || full_pipeline.effective_device_policy != expected_effective
                        || full_pipeline.effective_device_policy != artifact.effective_device_policy
                    {
                        bail!(
                            "online_pa full CUDA pipeline backend/schema/device provenance mismatch"
                        );
                    }
                    match crate::common::parse_cuda_device_policy(
                        &full_pipeline.requested_device_policy,
                    )? {
                        crate::common::CudaDevicePolicy::Auto => {}
                        crate::common::CudaDevicePolicy::Gpu {
                            ordinal: requested_ordinal,
                        } if requested_ordinal == ordinal => {}
                        _ => bail!(
                            "online_pa full CUDA requested policy does not bind effective ordinal {ordinal}"
                        ),
                    }
                    if full_pipeline.device_identity.ordinal
                        != u32::try_from(ordinal).context("online_pa CUDA ordinal exceeds u32")?
                        || full_pipeline.device_identity.name.trim().is_empty()
                        || full_pipeline
                            .device_identity
                            .uuid
                            .iter()
                            .all(|byte| *byte == 0)
                        || full_pipeline.device_identity.pci_domain < 0
                        || full_pipeline.device_identity.pci_bus < 0
                        || full_pipeline.device_identity.pci_device < 0
                        || full_pipeline.device_identity.compute_capability_major <= 0
                        || full_pipeline.device_identity.compute_capability_minor < 0
                        || full_pipeline.device_identity.multiprocessor_count <= 0
                        || full_pipeline.device_identity.total_memory_bytes == 0
                    {
                        bail!("online_pa full CUDA physical-device identity is incomplete");
                    }
                    let expected_rows = u32::try_from(artifact.dataset_rows)
                        .context("online_pa full CUDA rows exceed device counter width")?;
                    let received_rows = full_pipeline
                        .class_counts
                        .iter()
                        .map(|count| u64::from(*count))
                        .sum::<u64>();
                    if full_pipeline.class_counts.iter().any(|count| *count == 0)
                        || received_rows != u64::from(expected_rows)
                    {
                        bail!("online_pa full CUDA class-count receipt is inconsistent");
                    }
                    let expected_class_weights = full_pipeline.class_counts.map(|count| {
                        (f64::from(expected_rows) / (3.0 * f64::from(count))).clamp(0.5, 4.0)
                    });
                    if expected_class_weights != class_slack_weights {
                        bail!("online_pa full CUDA class weights do not match device class counts");
                    }
                    let scaler_bytes = cols
                        .checked_mul(8)
                        .context("online_pa full CUDA scaler byte evidence overflow")?;
                    let class_count_bytes = 3_u64 * 4;
                    let expected_artifact_d2h = arithmetic_status_bytes
                        .checked_add(class_count_bytes)
                        .and_then(|value| value.checked_add(class_weight_bytes))
                        .and_then(|value| value.checked_add(scaler_bytes))
                        .and_then(|value| value.checked_add(scaler_bytes))
                        .and_then(|value| value.checked_add(weight_bytes))
                        .and_then(|value| value.checked_add(bias_bytes))
                        .context("online_pa full CUDA artifact D2H evidence overflow")?;
                    let expected_whole_fit_h2d = feature_bytes
                        .checked_add(label_bytes)
                        .context("online_pa full CUDA whole-fit H2D evidence overflow")?;
                    if evidence.evidence_scope_schema
                        != "neoethos.online_pa.cuda_evidence.whole_fit_call.v3"
                        || evidence.kernel_launch_count != expected_whole_fit_launches
                        || evidence.host_to_device_bytes != expected_whole_fit_h2d
                        || evidence.device_to_host_bytes != expected_artifact_d2h
                        || full_pipeline.kernel_launch_count != expected_whole_fit_launches
                        || full_pipeline.initialization_launch_count != 1
                        || full_pipeline.label_map_launch_count != 1
                        || full_pipeline.label_count_partial_launch_count != 1
                        || full_pipeline.label_count_weight_finalize_launch_count != 1
                        || full_pipeline.scaler_partial_launch_count != 1
                        || full_pipeline.scaler_finalize_launch_count != 1
                        || full_pipeline.scaler_transform_launch_count != 1
                        || full_pipeline.transform_fault_partial_launch_count != 1
                        || full_pipeline.preprocess_fault_finalize_launch_count != 1
                        || full_pipeline.training_rows_per_launch
                            != ONLINE_PA_CUDA_TRAINING_ROWS_PER_LAUNCH_V3
                        || full_pipeline.training_row_chunk_count_per_epoch
                            != expected_training_chunks_per_epoch
                        || full_pipeline.training_epoch_count != epochs
                        || full_pipeline.training_launch_count != expected_training_launches
                        || full_pipeline.training_interchunk_device_to_host_bytes != 0
                        || full_pipeline.raw_feature_h2d_bytes != feature_bytes
                        || full_pipeline.original_label_h2d_bytes != label_bytes
                        || full_pipeline.scaled_feature_h2d_bytes != 0
                        || full_pipeline.remapped_label_h2d_bytes != 0
                        || full_pipeline.class_slack_weight_h2d_bytes != 0
                        || full_pipeline.parameter_initialization_h2d_bytes != 0
                        || full_pipeline.artifact_d2h_bytes != expected_artifact_d2h
                    {
                        bail!("online_pa full CUDA pipeline launch/byte evidence is inconsistent");
                    }
                }
            }
        }
        other => bail!("unsupported online_pa training semantics schema `{other}`"),
    }
    Ok(())
}

fn validate_hoeffding_artifact(artifact: &HoeffdingArtifact) -> Result<()> {
    if artifact.model_name != AdaptiveModelKind::Hoeffding.model_name() {
        bail!(
            "online_hoeffding artifact model mismatch: expected {}, got {}",
            AdaptiveModelKind::Hoeffding.model_name(),
            artifact.model_name
        );
    }
    if artifact.feature_columns.is_empty() {
        bail!("online_hoeffding artifact must contain at least one feature column");
    }
    if artifact.dataset_rows == 0 {
        bail!("online_hoeffding artifact must contain at least one training row");
    }
    validate_hoeffding_fallback_basis_param(&artifact.params)?;

    match (
        artifact.fallback_scaler.as_ref(),
        artifact.fallback_weights.as_ref(),
        artifact.fallback_bias.as_ref(),
    ) {
        (None, None, None) => {}
        (Some(scaler), Some(weights), Some(bias)) => {
            let expected_features = artifact.feature_columns.len();
            let fallback_basis = hoeffding_fallback_basis(&artifact.params);
            let expected_state_dim = fallback_basis.expanded_dim(expected_features);
            if scaler.means.len() != expected_features || scaler.stds.len() != expected_features {
                bail!(
                    "online_hoeffding fallback scaler mismatch: expected {} features, received means {} and stds {}",
                    expected_features,
                    scaler.means.len(),
                    scaler.stds.len()
                );
            }
            if weights.nrows() != 3 || weights.ncols() != expected_state_dim {
                bail!(
                    "online_hoeffding fallback weight matrix mismatch: expected 3x{}, received {:?}",
                    expected_state_dim,
                    weights.dim()
                );
            }
            if bias.len() != 3 {
                bail!(
                    "online_hoeffding fallback bias mismatch: expected 3 entries, received {}",
                    bias.len()
                );
            }
            if scaler
                .means
                .iter()
                .chain(scaler.stds.iter())
                .any(|value| !value.is_finite())
            {
                bail!("online_hoeffding fallback scaler contains non-finite values");
            }
            if weights.iter().any(|value| !value.is_finite()) {
                bail!("online_hoeffding fallback weights contain non-finite values");
            }
            if bias.iter().any(|value| !value.is_finite()) {
                bail!("online_hoeffding fallback bias contains non-finite values");
            }
        }
        _ => {
            bail!(
                "online_hoeffding artifact fallback components must either all be present or all be absent"
            );
        }
    }

    let has_fallback = artifact.fallback_scaler.is_some();
    if artifact
        .committee_json
        .iter()
        .any(|payload| payload.trim().is_empty())
    {
        bail!("online_hoeffding committee payloads must not be blank");
    }
    if artifact.committee_json.is_empty() && !has_fallback {
        bail!("online_hoeffding artifact has neither committees nor a fallback model");
    }
    if let Some(mode) = artifact.params.get("artifact_mode") {
        let expected_mode =
            derived_hoeffding_artifact_mode(!artifact.committee_json.is_empty(), has_fallback);
        if mode.trim() != expected_mode {
            bail!(
                "online_hoeffding artifact_mode mismatch: expected `{}`, got `{}`",
                expected_mode,
                mode.trim()
            );
        }
    }

    let fallback_blend_weight = hoeffding_fallback_blend_weight(&artifact.params)?;
    if artifact.committee_json.is_empty() && fallback_blend_weight <= f64::EPSILON && !has_fallback
    {
        bail!("online_hoeffding artifact cannot be empty and blend-less at the same time");
    }

    Ok(())
}

fn derived_hoeffding_artifact_mode(has_committees: bool, has_fallback: bool) -> &'static str {
    match (has_committees, has_fallback) {
        (true, true) => "committee_hybrid",
        (true, false) => "committee_only",
        (false, true) => "fallback_only",
        (false, false) => "invalid",
    }
}

fn usize_param(params: &HashMap<String, String>, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn float_param(params: &HashMap<String, String>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn hoeffding_fallback_blend_weight(params: &HashMap<String, String>) -> Result<f64> {
    match params.get("fallback_blend_weight") {
        Some(raw_value) => {
            let value = raw_value.trim().parse::<f64>().with_context(|| {
                format!(
                    "online_hoeffding fallback_blend_weight `{}` is not a valid number",
                    raw_value
                )
            })?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!(
                    "online_hoeffding fallback_blend_weight must be finite and within [0.0, 1.0], got {}",
                    value
                );
            }
            Ok(value)
        }
        None => Ok(0.3),
    }
}

#[cfg(feature = "adaptive-models")]
fn labels_to_binary_targets(labels: &[usize], positive_class: usize) -> Vec<f64> {
    labels
        .iter()
        .map(|label| if *label == positive_class { 1.0 } else { 0.0 })
        .collect()
}

fn balanced_class_weights(labels: &[usize], classes: usize) -> Vec<f64> {
    let mut counts = vec![0usize; classes.max(1)];
    for &label in labels {
        if let Some(slot) = counts.get_mut(label) {
            *slot += 1;
        }
    }

    let total = labels.len().max(1) as f64;
    counts
        .into_iter()
        .map(|count| {
            let count = count.max(1) as f64;
            (total / (classes.max(1) as f64 * count)).clamp(0.5, 4.0)
        })
        .collect()
}

#[cfg(test)]
pub(super) fn clamped_balanced_class_slack_weights_v1(labels: &[usize]) -> Result<[f64; 3]> {
    let mut counts = [0_u32; 3];
    for &label in labels {
        let count = counts
            .get_mut(label)
            .with_context(|| format!("online_pa CUDA-v2 label {label} is outside 0..3"))?;
        *count = count
            .checked_add(1)
            .context("online_pa CUDA-v2 class count overflow")?;
    }
    for (class, count) in counts.iter().enumerate() {
        if *count == 0 {
            bail!(
                "online_pa CUDA-v2 class-slack weights require all three classes; class {class} is absent"
            );
        }
    }
    let total =
        u32::try_from(labels.len()).context("online_pa CUDA-v2 label count exceeds u32")? as f64;
    Ok(counts.map(|count| (total / (3.0 * f64::from(count))).clamp(0.5, 4.0)))
}

fn array2_from_logits(rows: usize, cols: usize, logits: Vec<f64>) -> Result<Array2<f64>> {
    Array2::from_shape_vec((rows, cols), logits).context("build adaptive model logits")
}

fn fallback_logits(
    features: &Array2<f64>,
    scaler: &FeatureScaler,
    weights: &Array2<f64>,
    bias: &Array1<f64>,
    basis: HoeffdingFallbackBasis,
) -> Result<Array2<f64>> {
    let features = scaler.transform(features)?;
    let features = expand_hoeffding_fallback_matrix(&features, basis)?;
    let mut logits = Vec::with_capacity(features.nrows() * 3);
    for row in 0..features.nrows() {
        for class_idx in 0..3 {
            logits.push(
                weights
                    .row(class_idx)
                    .iter()
                    .zip(features.row(row).iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    + bias[class_idx],
            );
        }
    }

    array2_from_logits(features.nrows(), 3, logits)
}

#[cfg(feature = "adaptive-models")]
fn committee_output_to_logit(output: f64) -> Result<f64> {
    if !output.is_finite() {
        bail!("online_hoeffding committee produced a non-finite probability");
    }
    let probability = output.clamp(1e-6, 1.0 - 1e-6);
    Ok((probability / (1.0 - probability)).ln())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PassiveAggressiveArtifact {
    model_name: String,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    scaler: FeatureScaler,
    weights: Array2<f64>,
    bias: Array1<f64>,
    aggressiveness: f64,
    epochs: usize,
    #[serde(default = "legacy_online_pa_class_slack_weight_policy")]
    class_slack_weight_policy: String,
    #[serde(default)]
    class_slack_weights: Option<[f64; 3]>,
    #[serde(default = "legacy_online_pa_training_semantics")]
    training_semantics_schema: String,
    #[serde(default = "legacy_online_pa_training_backend")]
    training_backend: String,
    #[serde(default = "legacy_online_pa_effective_device_policy")]
    effective_device_policy: String,
    #[serde(default)]
    cuda_training_evidence: Option<PassiveAggressiveCudaEvidenceV1>,
}

const ONLINE_PA_LEGACY_TRAINING_SEMANTICS_V1: &str =
    "neoethos.online_pa.legacy_weighted_argmax.epsilon_norm.f64.v1";
pub(super) const ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2: &str =
    "neoethos.online_pa.prediction_based.class_weighted_slack_cap.bias_augmented.f64.v2";
pub(super) const ONLINE_PA_CUDA_FULL_PIPELINE_SCHEMA_V3: &str = "neoethos.online_pa.cuda_full_pipeline.raw_f64.original_i32.parallel_ddof0.pb_weighted_slack_v2.epoch_major_chunks.v3";
pub(super) const ONLINE_PA_CUDA_PREPROCESSING_BACKEND_V2: &str =
    "online_pa_cuda_preprocessing_parallel_f64_v2";
const ONLINE_PA_CUDA_TRAINING_ROWS_PER_LAUNCH_V3: u64 = 1_024;
const ONLINE_PA_LEGACY_CLASS_SLACK_WEIGHT_POLICY_V1: &str =
    "neoethos.online_pa.legacy_implicit_loss_multiplier_clamped_[0.5,4].v1";
const ONLINE_PA_CUDA_CLASS_SLACK_WEIGHT_POLICY_V1: &str =
    "neoethos.online_pa.clamped_balanced_class_slack_weight_[0.5,4].v1";

fn legacy_online_pa_training_semantics() -> String {
    ONLINE_PA_LEGACY_TRAINING_SEMANTICS_V1.to_string()
}

fn legacy_online_pa_training_backend() -> String {
    "online_pa_cpu_legacy_v1".to_string()
}

fn legacy_online_pa_class_slack_weight_policy() -> String {
    ONLINE_PA_LEGACY_CLASS_SLACK_WEIGHT_POLICY_V1.to_string()
}

fn legacy_online_pa_effective_device_policy() -> String {
    "cpu".to_string()
}

fn legacy_online_pa_cuda_evidence_scope() -> String {
    "neoethos.online_pa.cuda_evidence.training_stage.v1".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct PassiveAggressiveCudaEvidenceV1 {
    #[serde(default = "legacy_online_pa_cuda_evidence_scope")]
    pub(super) evidence_scope_schema: String,
    pub(super) kernel_launch_count: u64,
    pub(super) host_to_device_bytes: u64,
    pub(super) device_to_host_bytes: u64,
    pub(super) ordered_sample_visits: u64,
    #[serde(default)]
    pub(super) full_pipeline: Option<PassiveAggressiveCudaPipelineEvidenceV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct PassiveAggressiveCudaDeviceIdentityV1 {
    pub(super) ordinal: u32,
    pub(super) uuid: [u8; 16],
    pub(super) pci_domain: i32,
    pub(super) pci_bus: i32,
    pub(super) pci_device: i32,
    pub(super) compute_capability_major: i32,
    pub(super) compute_capability_minor: i32,
    pub(super) multiprocessor_count: i32,
    pub(super) total_memory_bytes: u64,
    pub(super) name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct PassiveAggressiveCudaPipelineEvidenceV1 {
    pub(super) execution_pipeline_schema: String,
    pub(super) evidence_scope_schema: String,
    pub(super) requested_device_policy: String,
    pub(super) effective_device_policy: String,
    pub(super) preprocessing_backend: String,
    pub(super) training_backend: String,
    pub(super) bound_inference_backend: String,
    pub(super) device_identity: PassiveAggressiveCudaDeviceIdentityV1,
    pub(super) loss_cost_policy: String,
    pub(super) residency_scope: String,
    pub(super) persistent_model_buffers: bool,
    pub(super) class_counts: [u32; 3],
    pub(super) kernel_launch_count: u64,
    pub(super) initialization_launch_count: u64,
    pub(super) label_map_launch_count: u64,
    pub(super) label_count_partial_launch_count: u64,
    pub(super) label_count_weight_finalize_launch_count: u64,
    pub(super) scaler_partial_launch_count: u64,
    pub(super) scaler_finalize_launch_count: u64,
    pub(super) scaler_transform_launch_count: u64,
    pub(super) transform_fault_partial_launch_count: u64,
    pub(super) preprocess_fault_finalize_launch_count: u64,
    #[serde(default)]
    pub(super) training_rows_per_launch: u64,
    #[serde(default)]
    pub(super) training_row_chunk_count_per_epoch: u64,
    #[serde(default)]
    pub(super) training_epoch_count: u64,
    pub(super) training_launch_count: u64,
    #[serde(default)]
    pub(super) training_interchunk_device_to_host_bytes: u64,
    pub(super) raw_feature_h2d_bytes: u64,
    pub(super) original_label_h2d_bytes: u64,
    pub(super) scaled_feature_h2d_bytes: u64,
    pub(super) remapped_label_h2d_bytes: u64,
    pub(super) class_slack_weight_h2d_bytes: u64,
    pub(super) parameter_initialization_h2d_bytes: u64,
    pub(super) artifact_d2h_bytes: u64,
}

#[cfg(feature = "statistical-gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PassiveAggressiveCudaInferenceEvidenceV1 {
    pub(super) evidence_scope_schema: String,
    pub(super) requested_device_policy: String,
    pub(super) effective_device_policy: String,
    pub(super) device_identity: PassiveAggressiveCudaDeviceIdentityV1,
    pub(super) residency_scope: String,
    pub(super) persistent_model_buffers: bool,
    pub(super) kernel_launch_count: u64,
    pub(super) host_to_device_bytes: u64,
    pub(super) device_to_host_bytes: u64,
    pub(super) raw_feature_h2d_bytes: u64,
    pub(super) scaler_parameter_h2d_bytes: u64,
    pub(super) model_parameter_h2d_bytes: u64,
    pub(super) probability_d2h_bytes: u64,
    pub(super) status_d2h_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct OnlinePassiveAggressiveExpert {
    pub aggressiveness: f64,
    pub epochs: usize,
    pub feature_columns: Vec<String>,
    pub dataset_rows: usize,
    pub scaler: Option<FeatureScaler>,
    pub weights: Option<Array2<f64>>,
    pub bias: Option<Array1<f64>>,
    class_slack_weight_policy: String,
    class_slack_weights: Option<[f64; 3]>,
    training_semantics_schema: String,
    training_backend: String,
    effective_device_policy: String,
    cuda_training_evidence: Option<PassiveAggressiveCudaEvidenceV1>,
}

impl OnlinePassiveAggressiveExpert {
    pub fn new(aggressiveness: impl Into<f64>, epochs: usize) -> Self {
        let aggressiveness = aggressiveness.into();
        Self {
            aggressiveness: aggressiveness.clamp(1e-4_f64, 100.0_f64),
            epochs: epochs.max(1),
            feature_columns: Vec::new(),
            dataset_rows: 0,
            scaler: None,
            weights: None,
            bias: None,
            class_slack_weight_policy: legacy_online_pa_class_slack_weight_policy(),
            class_slack_weights: None,
            training_semantics_schema: legacy_online_pa_training_semantics(),
            training_backend: legacy_online_pa_training_backend(),
            effective_device_policy: legacy_online_pa_effective_device_policy(),
            cuda_training_evidence: None,
        }
    }

    pub fn from_params(params: Option<HashMap<String, String>>) -> Self {
        let params = params.unwrap_or_default();
        Self::new(
            float_param(&params, "c", 1.0),
            usize_param(&params, "epochs", 4),
        )
    }

    fn artifact(&self) -> Result<PassiveAggressiveArtifact> {
        Ok(PassiveAggressiveArtifact {
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            feature_columns: self.feature_columns.clone(),
            dataset_rows: self.dataset_rows,
            scaler: self.scaler.clone().context("missing PA scaler")?,
            weights: self.weights.clone().context("missing PA weights")?,
            bias: self.bias.clone().context("missing PA bias")?,
            aggressiveness: self.aggressiveness,
            epochs: self.epochs,
            class_slack_weight_policy: self.class_slack_weight_policy.clone(),
            class_slack_weights: self.class_slack_weights,
            training_semantics_schema: self.training_semantics_schema.clone(),
            training_backend: self.training_backend.clone(),
            effective_device_policy: self.effective_device_policy.clone(),
            cuda_training_evidence: self.cuda_training_evidence.clone(),
        })
    }
}

impl Default for OnlinePassiveAggressiveExpert {
    fn default() -> Self {
        Self::new(1.0, 4)
    }
}

fn resolved_online_pa_device_policy(requested: &str) -> Result<String> {
    match crate::common::resolve_cuda_device_policy(
        requested,
        crate::tree_models::config::nvidia_gpu_count(),
    )? {
        crate::common::ResolvedCudaDevicePolicy::Cpu => Ok("cpu".to_string()),
        crate::common::ResolvedCudaDevicePolicy::Cuda { ordinal } => {
            #[cfg(feature = "statistical-gpu")]
            {
                Ok(format!("gpu:{ordinal}"))
            }
            #[cfg(not(feature = "statistical-gpu"))]
            {
                bail!(
                    "online_pa resolved CUDA ordinal {ordinal} from policy `{requested}`, but this binary was built without `statistical-gpu`"
                )
            }
        }
    }
}

impl ExpertModel for OnlinePassiveAggressiveExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        let requested_device_policy = requested_runtime_device_policy("online_pa");
        let resolved_device_policy = resolved_online_pa_device_policy(&requested_device_policy)?;
        // Host-side FeatureFrame conversion is unavoidable input decoding and
        // row-major packing.  It performs no PA preprocessing or model math.
        let (raw_features, feature_columns) = feature_matrix_from_frame(x)?;
        ensure_label_count("online_pa", raw_features.nrows(), y.len())?;

        #[cfg(feature = "statistical-gpu")]
        if crate::common::cuda_kernel_enabled(&resolved_device_policy)? {
            // ONLINE_PA_FULL_GPU_FIT_BEGIN
            let fit = try_fit_passive_aggressive_cuda_full_pipeline(
                &requested_device_policy,
                &resolved_device_policy,
                &raw_features,
                y,
                self.aggressiveness,
                self.epochs,
            )
            .context("fit online_pa through the required full CUDA pipeline")?;
            let pipeline_receipt = fit
                .evidence
                .full_pipeline
                .as_ref()
                .context("online_pa full CUDA fit omitted its pipeline receipt")?;
            if pipeline_receipt.device_identity != fit.device_identity
                || pipeline_receipt.class_counts != fit.class_counts
            {
                bail!("online_pa full CUDA fit returned inconsistent device/count provenance");
            }
            self.feature_columns = feature_columns;
            self.dataset_rows = raw_features.nrows();
            self.scaler = Some(FeatureScaler {
                means: fit.scaler_means,
                stds: fit.scaler_stds,
            });
            self.weights = Some(fit.weights);
            self.bias = Some(fit.bias);
            self.class_slack_weight_policy =
                ONLINE_PA_CUDA_CLASS_SLACK_WEIGHT_POLICY_V1.to_string();
            self.class_slack_weights = Some(fit.class_slack_weights);
            self.training_semantics_schema = fit.training_semantics_schema;
            self.training_backend = fit.runtime_backend;
            self.effective_device_policy = fit.effective_device_policy;
            self.cuda_training_evidence = Some(fit.evidence);
            // ONLINE_PA_FULL_GPU_FIT_END
            return Ok(());
        }

        if resolved_device_policy != "cpu" {
            bail!(
                "online_pa resolved device `{resolved_device_policy}` without an executable CUDA lane"
            );
        }

        lease.scope(|| {
            let labels = remap_three_class_labels(y)?;
            let scaler = FeatureScaler::fit(&raw_features)?;
            let features = scaler.transform(&raw_features)?;
            let sample_weights = balanced_class_weights(&labels, 3);
            let n_rows = features.nrows();
            let n_cols = features.ncols();
            let mut weights = Array2::<f64>::zeros((3, n_cols));
            let mut bias = Array1::<f64>::zeros(3);

            for _ in 0..self.epochs {
                for (row, target_class) in labels.iter().enumerate().take(n_rows) {
                    let x_row = features.row(row);
                    let norm_sq = x_row.iter().map(|value| value * value).sum::<f64>() + 1e-6;

                    let mut scores = [0.0_f64; 3];
                    for class_idx in 0..3 {
                        scores[class_idx] = weights
                            .row(class_idx)
                            .iter()
                            .zip(x_row.iter())
                            .map(|(weight, value)| weight * value)
                            .sum::<f64>()
                            + bias[class_idx];
                    }

                    if scores.iter().any(|s| !s.is_finite()) {
                        tracing::warn!(
                            target: "online_learner",
                            "non-finite scores in argmax; skipping update"
                        );
                        continue;
                    }
                    let predicted_class = scores
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .map(|(class_idx, _)| class_idx)
                        .unwrap_or(*target_class);
                    if predicted_class == *target_class {
                        continue;
                    }

                    let margin = scores[predicted_class] - scores[*target_class] + 1.0;
                    if margin <= 0.0 {
                        continue;
                    }

                    let tau = (margin * sample_weights[*target_class] / (2.0 * norm_sq))
                        .min(self.aggressiveness);
                    if !tau.is_finite() || tau < 0.0 {
                        tracing::warn!(
                            target: "online_learner",
                            "OPA tau non-finite or negative ({tau}); skipping update"
                        );
                        continue;
                    }
                    for col in 0..n_cols {
                        weights[(*target_class, col)] += tau * x_row[col];
                        weights[(predicted_class, col)] -= tau * x_row[col];
                    }
                    bias[*target_class] += tau;
                    bias[predicted_class] -= tau;
                }
            }

            self.feature_columns = feature_columns;
            self.dataset_rows = n_rows;
            self.scaler = Some(scaler);
            self.weights = Some(weights);
            self.bias = Some(bias);
            self.class_slack_weight_policy = legacy_online_pa_class_slack_weight_policy();
            self.class_slack_weights = None;
            self.training_semantics_schema = legacy_online_pa_training_semantics();
            self.training_backend = legacy_online_pa_training_backend();
            self.effective_device_policy = legacy_online_pa_effective_device_policy();
            self.cuda_training_evidence = None;
            Ok(())
        })
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        ensure_feature_columns_match(&self.feature_columns, x)?;
        let weights = self
            .weights
            .as_ref()
            .context("online_pa model not fitted")?;
        let bias = self.bias.as_ref().context("online_pa model not fitted")?;
        let scaler = self.scaler.as_ref().context("online_pa scaler missing")?;
        // This conversion is host I/O/packing; all numerical inference below
        // stays on the backend bound into the artifact.
        let (raw_features, _) = feature_matrix_from_frame(x)?;
        let live_requested_device_policy = requested_runtime_device_policy("online_pa");
        let live_effective_device_policy =
            resolved_online_pa_device_policy(&live_requested_device_policy)?;

        if let Some(full_pipeline) = self
            .cuda_training_evidence
            .as_ref()
            .and_then(|evidence| evidence.full_pipeline.as_ref())
        {
            #[cfg(feature = "statistical-gpu")]
            {
                if live_requested_device_policy != full_pipeline.requested_device_policy {
                    bail!(
                        "online_pa full CUDA requested-policy drift: artifact `{}`, current `{}`",
                        full_pipeline.requested_device_policy,
                        live_requested_device_policy
                    );
                }
                let current_effective = live_effective_device_policy.clone();
                if current_effective != self.effective_device_policy
                    || current_effective != full_pipeline.effective_device_policy
                {
                    bail!(
                        "online_pa full CUDA inference device drift: artifact {}, current {}",
                        self.effective_device_policy,
                        current_effective
                    );
                }
                // ONLINE_PA_FULL_GPU_PREDICT_BEGIN
                let prediction = try_predict_passive_aggressive_cuda_full_pipeline(
                    &full_pipeline.requested_device_policy,
                    &current_effective,
                    &full_pipeline.device_identity,
                    &raw_features,
                    &scaler.means,
                    &scaler.stds,
                    weights,
                    bias,
                )
                .context("predict online_pa through fused raw CUDA inference")?;
                // ONLINE_PA_FULL_GPU_PREDICT_END
                let expected_raw_h2d = u64::try_from(raw_features.len())
                    .ok()
                    .and_then(|elements| elements.checked_mul(8))
                    .context("online_pa inference raw-byte evidence overflow")?;
                let expected_scaler_h2d = u64::try_from(scaler.means.len())
                    .ok()
                    .and_then(|elements| elements.checked_mul(16))
                    .context("online_pa inference scaler-byte evidence overflow")?;
                let expected_model_h2d = u64::try_from(weights.len())
                    .ok()
                    .and_then(|elements| elements.checked_add(u64::try_from(bias.len()).ok()?))
                    .and_then(|elements| elements.checked_mul(8))
                    .context("online_pa inference model-byte evidence overflow")?;
                let expected_probability_d2h = u64::try_from(raw_features.nrows())
                    .ok()
                    .and_then(|rows| rows.checked_mul(3 * 8))
                    .context("online_pa inference probability-byte evidence overflow")?;
                let expected_status_d2h = u64::try_from(raw_features.nrows())
                    .ok()
                    .and_then(|rows| rows.checked_mul(4))
                    .context("online_pa inference status-byte evidence overflow")?;
                let expected_whole_h2d = expected_raw_h2d
                    .checked_add(expected_scaler_h2d)
                    .and_then(|bytes| bytes.checked_add(expected_model_h2d))
                    .context("online_pa inference whole-call H2D evidence overflow")?;
                let expected_whole_d2h = expected_probability_d2h
                    .checked_add(expected_status_d2h)
                    .context("online_pa inference whole-call D2H evidence overflow")?;
                if prediction.runtime_backend != full_pipeline.bound_inference_backend
                    || prediction.effective_device_policy != current_effective
                    || prediction.device_identity != full_pipeline.device_identity
                    || prediction.evidence.requested_device_policy
                        != full_pipeline.requested_device_policy
                    || prediction.evidence.effective_device_policy != current_effective
                    || prediction.evidence.device_identity != full_pipeline.device_identity
                    || prediction.evidence.evidence_scope_schema
                        != "neoethos.online_pa.cuda_evidence.whole_predict_call.v2"
                    || prediction.evidence.residency_scope != "call_scoped"
                    || prediction.evidence.persistent_model_buffers
                    || prediction.evidence.kernel_launch_count != 1
                    || prediction.evidence.host_to_device_bytes != expected_whole_h2d
                    || prediction.evidence.device_to_host_bytes != expected_whole_d2h
                    || prediction.evidence.raw_feature_h2d_bytes != expected_raw_h2d
                    || prediction.evidence.scaler_parameter_h2d_bytes != expected_scaler_h2d
                    || prediction.evidence.model_parameter_h2d_bytes != expected_model_h2d
                    || prediction.evidence.probability_d2h_bytes != expected_probability_d2h
                    || prediction.evidence.status_d2h_bytes != expected_status_d2h
                {
                    bail!("online_pa fused CUDA inference receipt is inconsistent");
                }
                return Ok(prediction.probabilities);
            }
            #[cfg(not(feature = "statistical-gpu"))]
            {
                bail!(
                    "online_pa full CUDA artifact schema `{}` cannot infer because this binary lacks `statistical-gpu`",
                    full_pipeline.execution_pipeline_schema
                );
            }
        }

        if self.effective_device_policy != "cpu" {
            bail!(
                "online_pa CUDA training-only artifact has no full-GPU inference receipt; refusing CPU fallback"
            );
        }
        if live_effective_device_policy != "cpu" {
            bail!(
                "online_pa CPU artifact cannot infer under CUDA-required policy `{live_requested_device_policy}`; retrain through the full CUDA pipeline"
            );
        }

        lease.scope(|| {
            let features = scaler.transform(&raw_features)?;

            let mut logits = Vec::with_capacity(features.nrows() * 3);
            for row in 0..features.nrows() {
                for class_idx in 0..3 {
                    let score = weights
                        .row(class_idx)
                        .iter()
                        .zip(features.row(row).iter())
                        .map(|(weight, value)| weight * value)
                        .sum::<f64>()
                        + bias[class_idx];
                    logits.push(score);
                }
            }

            softmax_rows(&array2_from_logits(features.nrows(), 3, logits)?)
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create online_pa directory {}", path.display()))?;
        let artifact = self.artifact()?;
        validate_passive_aggressive_artifact(&artifact)?;
        let runtime_metadata = adaptive_runtime_metadata(
            AdaptiveModelKind::PassiveAggressive.model_name(),
            artifact.feature_columns.clone(),
            artifact.dataset_rows,
        )?;
        let metadata_path = path.join(METADATA_FILE_NAME);
        let model_path = path.join(MODEL_FILE_NAME);
        let metadata_tmp = staged_adaptive_file(path, METADATA_FILE_NAME);
        let model_tmp = staged_adaptive_file(path, MODEL_FILE_NAME);
        write_json(&metadata_tmp, &runtime_metadata)?;
        if let Err(err) = write_json(&model_tmp, &artifact) {
            cleanup_adaptive_temp_file(&metadata_tmp);
            cleanup_adaptive_temp_file(&model_tmp);
            return Err(err)
                .with_context(|| format!("write online_pa artifact to {}", path.display()));
        }
        let metadata_backup = backup_adaptive_file(path, METADATA_FILE_NAME);
        let model_backup = backup_adaptive_file(path, MODEL_FILE_NAME);
        if let Err(err) =
            stage_adaptive_target(&metadata_path, &metadata_backup, Some(&metadata_tmp))
        {
            cleanup_adaptive_temp_file(&model_tmp);
            return Err(err);
        }
        if let Err(err) = stage_adaptive_target(&model_path, &model_backup, Some(&model_tmp)) {
            restore_adaptive_backup(&metadata_path, &metadata_backup);
            return Err(err);
        }
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: PassiveAggressiveArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        validate_passive_aggressive_artifact(&artifact)?;
        if artifact.model_name != AdaptiveModelKind::PassiveAggressive.model_name() {
            bail!("expected online_pa artifact, got {}", artifact.model_name);
        }
        if let Some(full_pipeline) = artifact
            .cuda_training_evidence
            .as_ref()
            .and_then(|evidence| evidence.full_pipeline.as_ref())
        {
            let current_requested = requested_runtime_device_policy("online_pa");
            if current_requested != full_pipeline.requested_device_policy {
                bail!(
                    "online_pa full CUDA artifact requested-policy drift: recorded `{}`, current `{}`",
                    full_pipeline.requested_device_policy,
                    current_requested
                );
            }
            let current_effective = resolved_online_pa_device_policy(&current_requested)?;
            if current_effective != artifact.effective_device_policy
                || current_effective != full_pipeline.effective_device_policy
            {
                bail!(
                    "online_pa full CUDA artifact device drift: recorded {}, current {}",
                    artifact.effective_device_policy,
                    current_effective
                );
            }
            #[cfg(feature = "statistical-gpu")]
            validate_passive_aggressive_cuda_device_identity(
                &current_effective,
                &full_pipeline.device_identity,
            )?;
            #[cfg(not(feature = "statistical-gpu"))]
            bail!(
                "online_pa full CUDA artifact cannot load because this binary lacks `statistical-gpu`"
            );
        }
        let metadata = resolve_adaptive_runtime_metadata(
            path,
            AdaptiveModelKind::PassiveAggressive.model_name(),
            &artifact.feature_columns,
            artifact.dataset_rows,
        )?;
        if artifact.feature_columns != metadata.feature_columns {
            bail!(
                "online_pa feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.dataset_rows != metadata.training_summary.dataset_rows {
            bail!(
                "online_pa dataset-row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.dataset_rows
            );
        }
        let mut next_state = Self::new(artifact.aggressiveness, artifact.epochs);
        next_state.feature_columns = artifact.feature_columns;
        next_state.dataset_rows = artifact.dataset_rows;
        next_state.scaler = Some(artifact.scaler);
        next_state.weights = Some(artifact.weights);
        next_state.bias = Some(artifact.bias);
        next_state.class_slack_weight_policy = artifact.class_slack_weight_policy;
        next_state.class_slack_weights = artifact.class_slack_weights;
        next_state.training_semantics_schema = artifact.training_semantics_schema;
        next_state.training_backend = artifact.training_backend;
        next_state.effective_device_policy = artifact.effective_device_policy;
        next_state.cuda_training_evidence = artifact.cuda_training_evidence;
        *self = next_state;
        Ok(())
    }
}

impl OnlinePassiveAggressiveExpert {
    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let has_runtime_state =
            self.scaler.is_some() && self.weights.is_some() && self.bias.is_some();
        if self.dataset_rows == 0 {
            return (
                Some("online_pa_unknown".to_string()),
                Some("adaptive_policy_unavailable".to_string()),
            );
        }
        if self.feature_columns.is_empty() {
            return (
                Some("online_pa_unknown".to_string()),
                Some("pa_feature_schema_missing".to_string()),
            );
        }
        if !has_runtime_state {
            return (
                Some("online_pa_unknown".to_string()),
                Some("pa_runtime_state_incomplete".to_string()),
            );
        }

        if let Some(full_pipeline) = self
            .cuda_training_evidence
            .as_ref()
            .and_then(|evidence| evidence.full_pipeline.as_ref())
        {
            return (Some(full_pipeline.bound_inference_backend.clone()), None);
        }
        if self.effective_device_policy != "cpu" {
            return (
                Some("online_pa_cuda_inference_unavailable".to_string()),
                Some("pa_full_gpu_inference_receipt_missing".to_string()),
            );
        }

        let gpu_cpu_fallback = gpu_policy_cpu_fallback_reason("online_pa");
        (Some("online_pa_cpu".to_string()), gpu_cpu_fallback)
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                AdaptiveModelKind::PassiveAggressive.model_name(),
                ModelFamily::Adaptive,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HoeffdingArtifact {
    model_name: String,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    params: HashMap<String, String>,
    committee_json: Vec<String>,
    #[serde(default)]
    fallback_scaler: Option<FeatureScaler>,
    #[serde(default)]
    fallback_weights: Option<Array2<f64>>,
    #[serde(default)]
    fallback_bias: Option<Array1<f64>>,
}

fn fit_fallback_online_committee(
    features: &Array2<f64>,
    labels: &[usize],
    params: &HashMap<String, String>,
) -> Result<(FeatureScaler, Array2<f64>, Array1<f64>)> {
    ensure_label_count("online_hoeffding fallback", features.nrows(), labels.len())?;
    validate_hoeffding_fallback_basis_param(params)?;
    let scaler = FeatureScaler::fit(features)?;
    let features = scaler.transform(features)?;
    let basis = hoeffding_fallback_basis(params);
    let features = expand_hoeffding_fallback_matrix(&features, basis)?;
    let rows = features.nrows();
    let cols = features.ncols();
    if rows == 0 || cols == 0 {
        bail!("online_hoeffding fallback requires a non-empty feature matrix");
    }

    let lr = float_param(params, "learning_rate", 0.03).max(1e-4);
    let epochs = usize_param(params, "epochs", 4).max(1);
    let l2 = float_param(params, "l2", 1e-4).max(0.0);
    let sample_weights = balanced_class_weights(labels, 3);
    let mut weights = Array2::<f64>::zeros((3, cols));
    let mut bias = Array1::<f64>::zeros(3);

    for _ in 0..epochs {
        for row in 0..rows {
            let x_row = features.row(row);
            let mut logits = Array2::<f64>::zeros((1, 3));
            for class_idx in 0..3 {
                logits[(0, class_idx)] = weights
                    .row(class_idx)
                    .iter()
                    .zip(x_row.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f64>()
                    + bias[class_idx];
            }
            let probabilities = softmax_rows(&logits)?;
            let sample_weight = sample_weights[labels[row]];
            for class_idx in 0..3 {
                let target = if labels[row] == class_idx { 1.0 } else { 0.0 };
                let error = (probabilities[(0, class_idx)] - target) * sample_weight;
                if !error.is_finite() {
                    tracing::warn!(
                        target: "online_learner",
                        "Hoeffding error non-finite; skipping update"
                    );
                    continue;
                }
                for col in 0..cols {
                    weights[(class_idx, col)] -=
                        lr * (error * x_row[col] + l2 * weights[(class_idx, col)]);
                }
                bias[class_idx] -= lr * error;
            }
        }
    }

    Ok((scaler, weights, bias))
}

pub struct OnlineHoeffdingExpert {
    params: HashMap<String, String>,
    feature_columns: Vec<String>,
    dataset_rows: usize,
    committee_json: Vec<String>,
    committee_runtime_ready: bool,
    load_degraded_reason: Option<String>,
    fallback_scaler: Option<FeatureScaler>,
    fallback_weights: Option<Array2<f64>>,
    fallback_bias: Option<Array1<f64>>,
    #[cfg(feature = "adaptive-models")]
    committees: Vec<DynSGBT>,
}

impl OnlineHoeffdingExpert {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn new(params: Option<HashMap<String, String>>) -> Self {
        Self {
            params: params.unwrap_or_default(),
            feature_columns: Vec::new(),
            dataset_rows: 0,
            committee_json: Vec::new(),
            committee_runtime_ready: false,
            load_degraded_reason: None,
            fallback_scaler: None,
            fallback_weights: None,
            fallback_bias: None,
            #[cfg(feature = "adaptive-models")]
            committees: Vec::new(),
        }
    }

    #[cfg(feature = "adaptive-models")]
    fn drift_detector(&self) -> DriftDetectorType {
        match self
            .params
            .get("drift_detector")
            .map(|value| value.trim().to_ascii_lowercase())
        {
            Some(kind) if kind == "adwin" => DriftDetectorType::Adwin {
                delta: float_param(&self.params, "drift_delta", 0.002),
            },
            Some(kind) if kind == "ddm" => DriftDetectorType::Ddm {
                warning_level: float_param(&self.params, "warning_level", 2.0),
                drift_level: float_param(&self.params, "drift_level", 3.0),
                min_instances: usize_param(&self.params, "min_instances", 30) as u64,
            },
            _ => DriftDetectorType::PageHinkley {
                delta: float_param(&self.params, "drift_delta", 0.005),
                lambda: float_param(&self.params, "drift_lambda", 50.0),
            },
        }
    }

    #[cfg(feature = "adaptive-models")]
    fn config(&self) -> Result<SGBTConfig> {
        SGBTConfig::builder()
            .n_steps(usize_param(&self.params, "n_steps", 24))
            .learning_rate(float_param(&self.params, "learning_rate", 0.05))
            .feature_subsample_rate(float_param(&self.params, "feature_subsample_rate", 0.8))
            .max_depth(usize_param(&self.params, "max_depth", 5))
            .n_bins(usize_param(&self.params, "n_bins", 32))
            .grace_period(usize_param(&self.params, "grace_period", 32))
            .delta(float_param(&self.params, "delta", 1e-7))
            .lambda(float_param(&self.params, "lambda", 1.0))
            .gamma(float_param(&self.params, "gamma", 0.0))
            .drift_detector(self.drift_detector())
            .build()
            .map_err(|err| anyhow::anyhow!("invalid online_hoeffding config: {err}"))
    }

    fn artifact(&self) -> HoeffdingArtifact {
        let mut params = self.params.clone();
        params.insert(
            "artifact_mode".to_string(),
            derived_hoeffding_artifact_mode(
                !self.committee_json.is_empty(),
                self.has_persistable_fallback(),
            )
            .to_string(),
        );
        HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: self.feature_columns.clone(),
            dataset_rows: self.dataset_rows,
            params,
            committee_json: self.committee_json.clone(),
            fallback_scaler: self.fallback_scaler.clone(),
            fallback_weights: self.fallback_weights.clone(),
            fallback_bias: self.fallback_bias.clone(),
        }
    }

    fn has_persistable_fallback(&self) -> bool {
        self.fallback_scaler.is_some()
            && self.fallback_weights.is_some()
            && self.fallback_bias.is_some()
    }

    fn has_live_runtime_committees(&self) -> bool {
        #[cfg(feature = "adaptive-models")]
        {
            self.committee_runtime_ready
                && !self.committees.is_empty()
                && self.committees.len() == self.committee_json.len()
        }

        #[cfg(not(feature = "adaptive-models"))]
        {
            false
        }
    }

    #[cfg(feature = "adaptive-models")]
    fn restore_committees(&mut self) -> Result<()> {
        self.committee_runtime_ready = false;
        self.committees.clear();
        let mut failures = 0usize;
        for payload in &self.committee_json {
            match load_model(payload) {
                Ok(model) => self.committees.push(model),
                Err(err) => {
                    failures += 1;
                    warn!(
                        "failed to restore online_hoeffding committee from artifact: {}",
                        err
                    );
                }
            }
        }
        if !self.committee_json.is_empty() && self.committees.len() != self.committee_json.len() {
            bail!(
                "restored {} of {} online_hoeffding committees; partial committee recovery is not accepted because it would silently degrade inference",
                self.committees.len(),
                self.committee_json.len()
            );
        }
        if failures > 0 && self.committee_json.is_empty() {
            bail!(
                "online_hoeffding restore reported committee failures with no committee payloads"
            );
        }
        self.committee_runtime_ready = !self.committee_json.is_empty();
        Ok(())
    }
}

impl Default for OnlineHoeffdingExpert {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ExpertModel for OnlineHoeffdingExpert {
    fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease) -> Result<()> {
        lease.scope(|| {
            #[cfg(not(feature = "adaptive-models"))]
            {
                let (features, feature_columns) = feature_matrix_from_frame(x)?;
                let labels = remap_three_class_labels(y)?;
                ensure_label_count("online_hoeffding", features.nrows(), labels.len())?;
                // Establish the fallback basis/blend/classes params BEFORE fitting so
                // the fitted weight matrix matches the basis the runtime validator
                // recomputes from these params (otherwise it is built linear (3×F)
                // but validated against the quadratic expansion (3×2F) → mismatch).
                self.params
                    .entry("fallback_blend_weight".to_string())
                    .or_insert_with(|| "0.3".to_string());
                self.params
                    .entry("fallback_basis".to_string())
                    .or_insert_with(|| "quadratic".to_string());
                self.params
                    .entry("classes".to_string())
                    .or_insert_with(|| "3".to_string());
                let (scaler, weights, bias) =
                    fit_fallback_online_committee(&features, &labels, &self.params)?;
                self.params
                    .insert("artifact_mode".to_string(), "fallback_only".to_string());
                self.feature_columns = feature_columns;
                self.dataset_rows = features.nrows();
                self.committee_json.clear();
                self.committee_runtime_ready = false;
                self.fallback_scaler = Some(scaler);
                self.fallback_weights = Some(weights);
                self.fallback_bias = Some(bias);
                Ok(())
            }

            #[cfg(feature = "adaptive-models")]
            {
                let (features, feature_columns) = feature_matrix_from_frame(x)?;
                let labels = remap_three_class_labels(y)?;
                ensure_label_count("online_hoeffding", features.nrows(), labels.len())?;
                validate_hoeffding_fallback_basis_param(&self.params)?;
                let config = self.config()?;
                let rows = (0..features.nrows())
                    .map(|row_idx| features.row(row_idx).iter().copied().collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                let mut committee_json = Vec::with_capacity(3);
                let mut committees = Vec::with_capacity(3);

                for class_idx in 0..3 {
                    let binary_targets = labels_to_binary_targets(&labels, class_idx);
                    let mut committee = SGBT::with_loss(config.clone(), LogisticLoss);
                    for (features, target) in rows.iter().zip(binary_targets.iter()) {
                        committee.train_one(&Sample::new(features.clone(), *target));
                    }

                    let payload = save_model_with(&committee, LossType::Logistic)
                        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
                    let runtime =
                        load_model(&payload).map_err(|err| anyhow::anyhow!(err.to_string()))?;
                    committee_json.push(payload);
                    committees.push(runtime);
                }

                // Establish the fallback basis/blend/classes params BEFORE fitting so
                // the fitted weight matrix matches the basis the runtime validator
                // recomputes from these params (otherwise it is built linear (3×F)
                // but validated against the quadratic expansion (3×2F) → mismatch).
                self.params
                    .entry("fallback_blend_weight".to_string())
                    .or_insert_with(|| "0.3".to_string());
                self.params
                    .entry("fallback_basis".to_string())
                    .or_insert_with(|| "quadratic".to_string());
                self.params
                    .entry("classes".to_string())
                    .or_insert_with(|| "3".to_string());

                let (fallback_scaler, fallback_weights, fallback_bias) =
                    fit_fallback_online_committee(&features, &labels, &self.params)?;

                self.params
                    .insert("artifact_mode".to_string(), "committee_hybrid".to_string());
                self.feature_columns = feature_columns;
                self.dataset_rows = features.nrows();
                self.committee_json = committee_json;
                self.committee_runtime_ready = true;
                self.fallback_scaler = Some(fallback_scaler);
                self.fallback_weights = Some(fallback_weights);
                self.fallback_bias = Some(fallback_bias);
                self.committees = committees;
                Ok(())
            }
        })
    }

    fn predict_proba(&self, x: &FeatureFrame, lease: &CpuLease) -> Result<Array2<f64>> {
        lease.scope(|| {
        #[cfg(not(feature = "adaptive-models"))]
        {
            let fallback_basis = hoeffding_fallback_basis(&self.params);
            ensure_feature_columns_match(&self.feature_columns, x)?;
            let (features, _) = feature_matrix_from_frame(x)?;
            let scaler = self
                .fallback_scaler
                .as_ref()
                .context("online_hoeffding fallback scaler missing")?;
            let weights = self
                .fallback_weights
                .as_ref()
                .context("online_hoeffding fallback weights missing")?;
            let bias = self
                .fallback_bias
                .as_ref()
                .context("online_hoeffding fallback bias missing")?;
            softmax_rows(&fallback_logits(
                &features,
                scaler,
                weights,
                bias,
                fallback_basis,
            )?)
        }

        #[cfg(feature = "adaptive-models")]
        {
            let fallback_basis = hoeffding_fallback_basis(&self.params);
            ensure_feature_columns_match(&self.feature_columns, x)?;
            let fallback_blend_weight = hoeffding_fallback_blend_weight(&self.params)?;
            let (features, _) = feature_matrix_from_frame(x)?;
            let rows = (0..features.nrows())
                .map(|row_idx| {
                    features
                        .row(row_idx)
                        .iter()
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let fallback = self
                .fallback_scaler
                .as_ref()
                .zip(self.fallback_weights.as_ref())
                .zip(self.fallback_bias.as_ref())
                .map(|((scaler, weights), bias)| {
                    fallback_logits(&features, scaler, weights, bias, fallback_basis)
                })
                .transpose()?;

            if self.committees.len() == 3 {
                let mut committee_logits = Array2::<f64>::zeros((rows.len(), 3));
                let mut committee_failed = None;
                for (row_idx, features) in rows.iter().enumerate() {
                    for (class_idx, committee) in self.committees.iter().enumerate() {
                        match committee_output_to_logit(committee.predict(features)) {
                            Ok(logit) => committee_logits[(row_idx, class_idx)] = logit,
                            Err(err) => {
                                committee_failed = Some(err);
                                break;
                            }
                        }
                    }
                    if committee_failed.is_some() {
                        break;
                    }
                }
                if let Some(err) = committee_failed {
                    if let Some(fallback) = fallback {
                        warn!(
                            "online_hoeffding committee inference degraded to fallback model: {}",
                            err
                        );
                        return softmax_rows(&fallback);
                    }
                    return Err(err);
                }
                if let Some(fallback) = fallback {
                    if fallback_blend_weight <= f64::EPSILON {
                        softmax_rows(&committee_logits)
                    } else if fallback_blend_weight >= 1.0 - f64::EPSILON {
                        warn!(
                            "online_hoeffding fallback_blend_weight is 1.0; using fallback logits only"
                        );
                        softmax_rows(&fallback)
                    } else {
                        let committee_weight = 1.0 - fallback_blend_weight;
                        let mut blended = committee_logits;
                        for row in 0..blended.nrows() {
                            for class_idx in 0..blended.ncols() {
                                blended[(row, class_idx)] = committee_weight
                                    * blended[(row, class_idx)]
                                    + fallback_blend_weight * fallback[(row, class_idx)];
                            }
                        }
                        softmax_rows(&blended)
                    }
                } else {
                    softmax_rows(&committee_logits)
                }
            } else {
                let fallback = fallback.context("online_hoeffding fallback model missing")?;
                softmax_rows(&fallback)
            }
        }
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create online_hoeffding directory {}", path.display()))?;
        let artifact = self.artifact();
        validate_hoeffding_artifact(&artifact)?;
        let metadata = adaptive_runtime_metadata(
            AdaptiveModelKind::Hoeffding.model_name(),
            artifact.feature_columns.clone(),
            artifact.dataset_rows,
        )?;
        let metadata_path = path.join(METADATA_FILE_NAME);
        let model_path = path.join(MODEL_FILE_NAME);
        let metadata_tmp = staged_adaptive_file(path, METADATA_FILE_NAME);
        let model_tmp = staged_adaptive_file(path, MODEL_FILE_NAME);
        write_json(&metadata_tmp, &metadata)?;
        if let Err(err) = write_json(&model_tmp, &artifact) {
            cleanup_adaptive_temp_file(&metadata_tmp);
            cleanup_adaptive_temp_file(&model_tmp);
            return Err(err)
                .with_context(|| format!("write online_hoeffding artifact to {}", path.display()));
        }
        let metadata_backup = backup_adaptive_file(path, METADATA_FILE_NAME);
        let model_backup = backup_adaptive_file(path, MODEL_FILE_NAME);
        if let Err(err) =
            stage_adaptive_target(&metadata_path, &metadata_backup, Some(&metadata_tmp))
        {
            cleanup_adaptive_temp_file(&model_tmp);
            return Err(err);
        }
        if let Err(err) = stage_adaptive_target(&model_path, &model_backup, Some(&model_tmp)) {
            restore_adaptive_backup(&metadata_path, &metadata_backup);
            return Err(err);
        }
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let artifact: HoeffdingArtifact = read_json(&path.join(MODEL_FILE_NAME))?;
        validate_hoeffding_artifact(&artifact)?;
        if artifact.model_name != AdaptiveModelKind::Hoeffding.model_name() {
            bail!(
                "expected online_hoeffding artifact, got {}",
                artifact.model_name
            );
        }
        let metadata = resolve_adaptive_runtime_metadata(
            path,
            AdaptiveModelKind::Hoeffding.model_name(),
            &artifact.feature_columns,
            artifact.dataset_rows,
        )?;
        if artifact.feature_columns != metadata.feature_columns {
            bail!(
                "online_hoeffding feature-column mismatch between metadata {:?} and artifact {:?}",
                metadata.feature_columns,
                artifact.feature_columns
            );
        }
        if artifact.dataset_rows != metadata.training_summary.dataset_rows {
            bail!(
                "online_hoeffding dataset-row mismatch between metadata {} and artifact {}",
                metadata.training_summary.dataset_rows,
                artifact.dataset_rows
            );
        }

        let mut next_state = Self::new(Some(artifact.params.clone()));
        next_state.params = artifact.params;
        next_state
            .params
            .entry("artifact_mode".to_string())
            .or_insert_with(|| {
                derived_hoeffding_artifact_mode(
                    !artifact.committee_json.is_empty(),
                    artifact.fallback_scaler.is_some(),
                )
                .to_string()
            });
        next_state.feature_columns = artifact.feature_columns;
        next_state.dataset_rows = artifact.dataset_rows;
        next_state.committee_json = artifact.committee_json;
        next_state.committee_runtime_ready = false;
        next_state.load_degraded_reason = None;
        next_state.fallback_scaler = artifact.fallback_scaler;
        next_state.fallback_weights = artifact.fallback_weights;
        next_state.fallback_bias = artifact.fallback_bias;
        if next_state.committee_json.is_empty() && !next_state.has_persistable_fallback() {
            bail!("online_hoeffding artifact has neither committees nor a fallback model");
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            if !next_state.has_persistable_fallback() {
                bail!(
                    "online_hoeffding artifacts require a persisted fallback model when adaptive-models support is disabled"
                );
            }
            if !next_state.committee_json.is_empty() {
                next_state.committee_runtime_ready = false;
                next_state.load_degraded_reason = Some("committee_backend_unavailable".to_string());
            }
        }
        #[cfg(feature = "adaptive-models")]
        {
            if let Err(err) = next_state.restore_committees() {
                if next_state.has_persistable_fallback() {
                    warn!(
                        "online_hoeffding committee restore degraded to persisted fallback model: {}",
                        err
                    );
                    next_state.committees.clear();
                    next_state.committee_runtime_ready = false;
                    next_state.load_degraded_reason = Some("committee_restore_failed".to_string());
                } else {
                    return Err(err);
                }
            }
        }
        *self = next_state;
        Ok(())
    }
}

impl OnlineHoeffdingExpert {
    fn runtime_details(&self) -> (Option<String>, Option<String>) {
        let gpu_cpu_fallback = gpu_policy_cpu_fallback_reason("online_hoeffding");
        let has_runtime_committees = self.has_live_runtime_committees();
        let has_persisted_committees = !self.committee_json.is_empty();
        let has_fallback = self.has_persistable_fallback();
        let fallback_blend_weight = hoeffding_fallback_blend_weight(&self.params).unwrap_or(0.3);
        let persisted_committee_reason = if has_persisted_committees && !has_runtime_committees {
            Some(
                self.load_degraded_reason
                    .clone()
                    .unwrap_or_else(|| "committee_backend_unavailable".to_string()),
            )
        } else {
            None
        };
        match (has_runtime_committees, has_fallback) {
            (true, true) if fallback_blend_weight >= 1.0 - f64::EPSILON => (
                Some("online_hoeffding_fallback".to_string()),
                append_runtime_degraded_reason(
                    Some("committee_blend_disabled_by_weight".to_string()),
                    gpu_cpu_fallback.clone(),
                ),
            ),
            (true, true) if fallback_blend_weight <= f64::EPSILON => (
                Some("online_hoeffding_committee".to_string()),
                gpu_cpu_fallback.clone(),
            ),
            (true, true) => (
                Some("online_hoeffding_committee_hybrid".to_string()),
                append_runtime_degraded_reason(
                    Some("committee_predictions_blended_with_fallback".to_string()),
                    gpu_cpu_fallback.clone(),
                ),
            ),
            (true, false) => (
                Some("online_hoeffding_committee".to_string()),
                gpu_cpu_fallback.clone(),
            ),
            (false, true) if fallback_blend_weight >= 1.0 - f64::EPSILON => (
                Some("online_hoeffding_fallback".to_string()),
                append_runtime_degraded_reason(
                    Some("committee_blend_disabled_by_weight".to_string()),
                    gpu_cpu_fallback.clone(),
                ),
            ),
            (false, true) => (
                Some("online_hoeffding_fallback".to_string()),
                append_runtime_degraded_reason(
                    persisted_committee_reason,
                    gpu_cpu_fallback.clone(),
                ),
            ),
            (false, false) => (
                Some("online_hoeffding_unknown".to_string()),
                append_runtime_degraded_reason(
                    Some(if let Some(reason) = persisted_committee_reason {
                        reason
                    } else {
                        "adaptive_policy_unavailable".to_string()
                    }),
                    gpu_cpu_fallback,
                ),
            ),
        }
    }

    pub fn predict_runtime(
        &self,
        x: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<RuntimePrediction>> {
        let probabilities = self.predict_proba(x, lease)?;
        let (execution_backend, degraded_reason) = self.runtime_details();
        let mut predictions = Vec::with_capacity(probabilities.nrows());
        for row in probabilities.outer_iter() {
            let row_values = [row[0], row[1], row[2]];
            let (confidence, abstain) = three_class_runtime_confidence(row_values)?;
            predictions.push(build_runtime_prediction_with_details(
                AdaptiveModelKind::Hoeffding.model_name(),
                ModelFamily::Adaptive,
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

pub struct AdaptiveGradientBooster {
    #[cfg(feature = "adaptive-models")]
    inner: SGBT,
    #[cfg(not(feature = "adaptive-models"))]
    weights: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    bias: f64,
    #[cfg(not(feature = "adaptive-models"))]
    learning_rate: f64,
    #[cfg(not(feature = "adaptive-models"))]
    feature_mean: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    feature_m2: Vec<f64>,
    #[cfg(not(feature = "adaptive-models"))]
    observations: usize,
}

impl AdaptiveGradientBooster {
    pub fn new() -> Self {
        #[cfg(feature = "adaptive-models")]
        {
            let config = SGBTConfig::builder()
                .n_steps(24)
                .learning_rate(0.05)
                .feature_subsample_rate(0.8)
                .max_depth(5)
                .n_bins(32)
                .grace_period(32)
                .build()
                .expect("default adaptive booster config should be valid");
            Self {
                inner: SGBT::new(config),
            }
        }

        #[cfg(not(feature = "adaptive-models"))]
        {
            Self {
                weights: Vec::new(),
                bias: 0.0,
                learning_rate: 0.05,
                feature_mean: Vec::new(),
                feature_m2: Vec::new(),
                observations: 0,
            }
        }
    }

    #[cfg(not(feature = "adaptive-models"))]
    fn update_feature_statistics(&mut self, x: &[f64]) {
        if self.feature_mean.len() != x.len() {
            self.feature_mean = vec![0.0; x.len()];
            self.feature_m2 = vec![0.0; x.len()];
            self.observations = 0;
        }

        self.observations += 1;
        let count = self.observations as f64;
        for (idx, value) in x.iter().enumerate() {
            let delta = *value - self.feature_mean[idx];
            self.feature_mean[idx] += delta / count;
            let delta2 = *value - self.feature_mean[idx];
            self.feature_m2[idx] += delta * delta2;
        }
    }

    #[cfg(not(feature = "adaptive-models"))]
    fn normalize_features(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.weights.is_empty() {
            return Ok(x.to_vec());
        }
        if self.weights.len() != x.len()
            || self.feature_mean.len() != x.len()
            || self.feature_m2.len() != x.len()
        {
            bail!(
                "adaptive gradient booster expected {} features, got {}",
                self.weights.len(),
                x.len()
            );
        }

        Ok(x.iter()
            .enumerate()
            .map(|(idx, value)| {
                let variance = if self.observations > 1 {
                    self.feature_m2[idx] / (self.observations as f64 - 1.0)
                } else {
                    1.0
                };
                let scale = variance.abs().sqrt().max(1e-6);
                (*value - self.feature_mean[idx]) / scale
            })
            .collect())
    }

    pub fn learn_one(&mut self, x: Vec<f64>, y: f64) -> Result<()> {
        #[cfg(feature = "adaptive-models")]
        {
            self.inner.train_one(&Sample::new(x, y));
            Ok(())
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            if !self.weights.is_empty() && self.weights.len() != x.len() {
                bail!(
                    "adaptive gradient booster expected {} features, got {}",
                    self.weights.len(),
                    x.len()
                );
            }
            self.update_feature_statistics(&x);
            let x = self.normalize_features(&x)?;
            if self.weights.is_empty() {
                self.weights = vec![0.0; x.len()];
            }

            let prediction = self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + self.bias;
            let error = prediction - y;
            for (weight, value) in self.weights.iter_mut().zip(x.iter()) {
                *weight -= self.learning_rate * error * *value;
            }
            self.bias -= self.learning_rate * error;
            Ok(())
        }
    }

    pub fn predict_one(&self, x: &[f64]) -> Result<f64> {
        #[cfg(feature = "adaptive-models")]
        {
            Ok(self.inner.predict(x))
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            if self.weights.is_empty() {
                bail!("adaptive gradient booster has not observed any samples");
            }
            let x = self.normalize_features(x)?;
            Ok(self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + self.bias)
        }
    }
}

impl Default for AdaptiveGradientBooster {
    fn default() -> Self {
        Self::new()
    }
}

// TODO(real-data): every typed feature frame / committee weight vector
// in this test module is synthetic (zero means, unit stds, hand-picked
// f64 sequences like vec![1.0, -1.0]). Replace each fixture with a
// cTrader historical sample (e.g. EURUSD M15 z-scored features) and
// run training summaries against the actual broker data shape so the
// asserted runtime-fallback / committee paths fire on real noise.
#[cfg(all(test, feature = "adaptive-models"))]
mod tests {
    use super::*;
    use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn typed_frame(columns: Vec<(&str, Vec<f64>)>) -> Result<FeatureFrame> {
        let rows = columns.first().map_or(0, |(_, values)| values.len());
        let columns = columns
            .into_iter()
            .map(|(name, values)| {
                FeatureColumnF64::new(name, values, vec![FeatureCellValidity::Valid; rows])
            })
            .collect::<Result<Vec<_>>>()?;
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
    }

    fn one_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker is valid");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("isolated adaptive-model test lease")
    }

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_models_{name}_{}_{}",
            std::process::id(),
            stamp
        ))
    }

    fn valid_cuda_v2_pa_artifact() -> PassiveAggressiveArtifact {
        PassiveAggressiveArtifact {
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 9,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            weights: Array2::zeros((3, 2)),
            bias: Array1::zeros(3),
            aggressiveness: 1.0,
            epochs: 4,
            class_slack_weight_policy: ONLINE_PA_CUDA_CLASS_SLACK_WEIGHT_POLICY_V1.to_string(),
            class_slack_weights: Some([1.0, 1.0, 1.0]),
            training_semantics_schema: ONLINE_PA_CUDA_TRAINING_SEMANTICS_V2.to_string(),
            training_backend: "online_pa_cuda[cuda:0]".to_string(),
            effective_device_policy: "gpu:0".to_string(),
            cuda_training_evidence: Some(PassiveAggressiveCudaEvidenceV1 {
                evidence_scope_schema: "neoethos.online_pa.cuda_evidence.training_stage.v2"
                    .to_string(),
                kernel_launch_count: 4,
                host_to_device_bytes: 280,
                device_to_host_bytes: 76,
                ordered_sample_visits: 36,
                full_pipeline: None,
            }),
        }
    }

    fn valid_cuda_full_pipeline_pa_artifact() -> PassiveAggressiveArtifact {
        let mut artifact = valid_cuda_v2_pa_artifact();
        artifact.training_backend = "online_pa_cuda_full_pipeline[cuda:0]".to_string();
        artifact.cuda_training_evidence = Some(PassiveAggressiveCudaEvidenceV1 {
            evidence_scope_schema: "neoethos.online_pa.cuda_evidence.whole_fit_call.v3".to_string(),
            kernel_launch_count: 13,
            host_to_device_bytes: 180,
            device_to_host_bytes: 144,
            ordered_sample_visits: 36,
            full_pipeline: Some(PassiveAggressiveCudaPipelineEvidenceV1 {
                execution_pipeline_schema: ONLINE_PA_CUDA_FULL_PIPELINE_SCHEMA_V3.to_string(),
                evidence_scope_schema: "neoethos.online_pa.cuda_pipeline_stages.v3".to_string(),
                requested_device_policy: "gpu:0".to_string(),
                effective_device_policy: "gpu:0".to_string(),
                preprocessing_backend: ONLINE_PA_CUDA_PREPROCESSING_BACKEND_V2.to_string(),
                training_backend: "online_pa_cuda_full_pipeline[cuda:0]".to_string(),
                bound_inference_backend: "online_pa_cuda_fused_inference[cuda:0]".to_string(),
                device_identity: PassiveAggressiveCudaDeviceIdentityV1 {
                    ordinal: 0,
                    uuid: [1; 16],
                    pci_domain: 0,
                    pci_bus: 1,
                    pci_device: 0,
                    compute_capability_major: 12,
                    compute_capability_minor: 0,
                    multiprocessor_count: 1,
                    total_memory_bytes: 1,
                    name: "test CUDA device".to_string(),
                },
                loss_cost_policy: "rho(y,r)=1; PA-I cap=C*w_y".to_string(),
                residency_scope: "call_scoped".to_string(),
                persistent_model_buffers: false,
                class_counts: [3, 3, 3],
                kernel_launch_count: 13,
                initialization_launch_count: 1,
                label_map_launch_count: 1,
                label_count_partial_launch_count: 1,
                label_count_weight_finalize_launch_count: 1,
                scaler_partial_launch_count: 1,
                scaler_finalize_launch_count: 1,
                scaler_transform_launch_count: 1,
                transform_fault_partial_launch_count: 1,
                preprocess_fault_finalize_launch_count: 1,
                training_rows_per_launch: 1_024,
                training_row_chunk_count_per_epoch: 1,
                training_epoch_count: 4,
                training_launch_count: 4,
                training_interchunk_device_to_host_bytes: 0,
                raw_feature_h2d_bytes: 144,
                original_label_h2d_bytes: 36,
                scaled_feature_h2d_bytes: 0,
                remapped_label_h2d_bytes: 0,
                class_slack_weight_h2d_bytes: 0,
                parameter_initialization_h2d_bytes: 0,
                artifact_d2h_bytes: 144,
            }),
        });
        artifact
    }

    #[test]
    fn online_pa_cuda_v2_class_slack_weights_are_explicit_and_require_all_classes() -> Result<()> {
        let labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!(
            clamped_balanced_class_slack_weights_v1(&labels)?,
            [0.5, 4.0, 4.0]
        );

        let error = clamped_balanced_class_slack_weights_v1(&[0, 1, 0, 1])
            .expect_err("CUDA-v2 class-slack weights must not fabricate an absent class");
        assert!(
            error.to_string().contains("class 2 is absent"),
            "unexpected absent-class rejection: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn online_pa_cuda_v2_provenance_is_exact_and_mutation_closed() -> Result<()> {
        let artifact = valid_cuda_v2_pa_artifact();
        validate_passive_aggressive_artifact(&artifact)?;

        let mut wrong_backend = artifact.clone();
        wrong_backend.training_backend = "online_pa_cpu_legacy_v1".to_string();
        assert!(validate_passive_aggressive_artifact(&wrong_backend).is_err());

        let mut wrong_policy = artifact.clone();
        wrong_policy.effective_device_policy = "cpu".to_string();
        assert!(validate_passive_aggressive_artifact(&wrong_policy).is_err());

        let mut wrong_slack_policy = artifact.clone();
        wrong_slack_policy.class_slack_weight_policy = "online_pa.ambient".to_string();
        assert!(validate_passive_aggressive_artifact(&wrong_slack_policy).is_err());

        let mut wrong_slack_weights = artifact.clone();
        wrong_slack_weights.class_slack_weights = Some([0.49, 4.0, 4.0]);
        assert!(validate_passive_aggressive_artifact(&wrong_slack_weights).is_err());

        let mut missing_slack_weights = artifact.clone();
        missing_slack_weights.class_slack_weights = None;
        assert!(validate_passive_aggressive_artifact(&missing_slack_weights).is_err());

        let mut wrong_evidence = artifact.clone();
        wrong_evidence
            .cuda_training_evidence
            .as_mut()
            .expect("fixture has CUDA evidence")
            .host_to_device_bytes += 1;
        assert!(validate_passive_aggressive_artifact(&wrong_evidence).is_err());

        let mut unknown_schema = artifact;
        unknown_schema.training_semantics_schema = "online_pa.future".to_string();
        assert!(validate_passive_aggressive_artifact(&unknown_schema).is_err());
        Ok(())
    }

    #[test]
    fn online_pa_full_cuda_pipeline_receipt_is_structurally_exact_and_mutation_closed() -> Result<()>
    {
        let artifact = valid_cuda_full_pipeline_pa_artifact();
        validate_passive_aggressive_artifact(&artifact)?;

        let mut wrong_requested = artifact.clone();
        wrong_requested
            .cuda_training_evidence
            .as_mut()
            .unwrap()
            .full_pipeline
            .as_mut()
            .unwrap()
            .requested_device_policy = "gpu:1".to_string();
        assert!(validate_passive_aggressive_artifact(&wrong_requested).is_err());

        let mut wrong_counts = artifact.clone();
        wrong_counts
            .cuda_training_evidence
            .as_mut()
            .unwrap()
            .full_pipeline
            .as_mut()
            .unwrap()
            .class_counts = [4, 3, 3];
        assert!(validate_passive_aggressive_artifact(&wrong_counts).is_err());

        let mut wrong_identity = artifact.clone();
        wrong_identity
            .cuda_training_evidence
            .as_mut()
            .unwrap()
            .full_pipeline
            .as_mut()
            .unwrap()
            .device_identity
            .uuid = [0; 16];
        assert!(validate_passive_aggressive_artifact(&wrong_identity).is_err());

        let mut wrong_bytes = artifact.clone();
        wrong_bytes
            .cuda_training_evidence
            .as_mut()
            .unwrap()
            .full_pipeline
            .as_mut()
            .unwrap()
            .artifact_d2h_bytes += 1;
        assert!(validate_passive_aggressive_artifact(&wrong_bytes).is_err());
        Ok(())
    }

    #[test]
    fn online_pa_pre_v2_json_is_read_only_as_explicit_legacy_semantics() -> Result<()> {
        let mut legacy_json = serde_json::to_value(valid_cuda_v2_pa_artifact())?;
        let object = legacy_json
            .as_object_mut()
            .context("serialized PA artifact must be an object")?;
        object.remove("class_slack_weight_policy");
        object.remove("class_slack_weights");
        object.remove("training_semantics_schema");
        object.remove("training_backend");
        object.remove("effective_device_policy");
        object.remove("cuda_training_evidence");
        let artifact: PassiveAggressiveArtifact = serde_json::from_value(legacy_json)?;
        assert_eq!(
            artifact.training_semantics_schema,
            ONLINE_PA_LEGACY_TRAINING_SEMANTICS_V1
        );
        assert_eq!(artifact.training_backend, "online_pa_cpu_legacy_v1");
        assert_eq!(artifact.effective_device_policy, "cpu");
        assert_eq!(
            artifact.class_slack_weight_policy,
            ONLINE_PA_LEGACY_CLASS_SLACK_WEIGHT_POLICY_V1
        );
        assert!(artifact.class_slack_weights.is_none());
        assert!(artifact.cuda_training_evidence.is_none());
        validate_passive_aggressive_artifact(&artifact)
    }

    #[test]
    fn online_hoeffding_loads_fallback_when_committees_are_corrupt() -> Result<()> {
        let path = temp_model_dir("online_hoeffding");
        std::fs::create_dir_all(&path)?;

        write_json(
            &path.join(METADATA_FILE_NAME),
            &adaptive_runtime_metadata(
                AdaptiveModelKind::Hoeffding.model_name(),
                vec!["f1".to_string(), "f2".to_string()],
                2,
            )?,
        )?;

        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::new(),
            committee_json: vec!["not-json".to_string()],
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::from_shape_vec(
                (3, 2),
                vec![0.0, 0.0, 1.2, 0.0, -1.2, 0.0],
            )?),
            fallback_bias: Some(Array1::from(vec![0.0, 0.0, 0.0])),
        };
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;

        let mut expert = OnlineHoeffdingExpert::new(None);
        expert.load(&path)?;
        assert_eq!(expert.committee_json.len(), 1);
        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert!(
            degraded_reason.is_some(),
            "degraded load should preserve committee provenance"
        );

        let frame = typed_frame(vec![("f1", vec![1.0, -1.0]), ("f2", vec![0.5, 0.25])])?;
        let probabilities = expert.predict_proba(&frame, &one_worker_lease())?;

        assert_eq!(probabilities.nrows(), 2);
        assert_eq!(probabilities.ncols(), 3);
        for row in 0..probabilities.nrows() {
            let sum = probabilities.row(row).iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn online_hoeffding_load_reconstructs_metadata_when_sidecar_missing() -> Result<()> {
        let path = temp_model_dir("online_hoeffding_missing_metadata");
        std::fs::create_dir_all(&path)?;

        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::new(),
            committee_json: vec!["not-json".to_string()],
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: Some(Array1::zeros(3)),
        };
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;

        let mut expert = OnlineHoeffdingExpert::new(None);
        expert.load(&path)?;
        assert_eq!(expert.feature_columns, artifact.feature_columns);

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn online_pa_load_rejects_metadata_sidecar_drift() -> Result<()> {
        let path = temp_model_dir("online_pa_sidecar_drift");
        std::fs::create_dir_all(&path)?;

        let artifact = PassiveAggressiveArtifact {
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            weights: Array2::zeros((3, 2)),
            bias: Array1::zeros(3),
            aggressiveness: 1.0,
            epochs: 4,
            class_slack_weight_policy: legacy_online_pa_class_slack_weight_policy(),
            class_slack_weights: None,
            training_semantics_schema: legacy_online_pa_training_semantics(),
            training_backend: legacy_online_pa_training_backend(),
            effective_device_policy: legacy_online_pa_effective_device_policy(),
            cuda_training_evidence: None,
        };
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;
        let mut drifted = adaptive_runtime_metadata(
            AdaptiveModelKind::PassiveAggressive.model_name(),
            artifact.feature_columns.clone(),
            artifact.dataset_rows,
        )?;
        drifted.training_summary.dataset_rows += 1;
        write_json(&path.join(METADATA_FILE_NAME), &drifted)?;

        let mut expert = OnlinePassiveAggressiveExpert::new(1.0, 4);
        let err = expert
            .load(&path)
            .expect_err("drifted metadata sidecar should fail");
        assert!(err.to_string().contains("sidecar mismatch"));

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn validate_hoeffding_artifact_rejects_unknown_fallback_basis() {
        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::from([("fallback_basis".to_string(), "cubic".to_string())]),
            committee_json: Vec::new(),
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err =
            validate_hoeffding_artifact(&artifact).expect_err("unknown fallback basis must fail");
        assert!(err.to_string().contains("fallback_basis"));
    }

    #[cfg(not(feature = "adaptive-models"))]
    #[test]
    fn online_hoeffding_load_preserves_committee_provenance_without_backend() -> Result<()> {
        let path = temp_model_dir("online_hoeffding_no_backend_mode");
        std::fs::create_dir_all(&path)?;

        write_json(
            &path.join(METADATA_FILE_NAME),
            &adaptive_runtime_metadata(
                AdaptiveModelKind::Hoeffding.model_name(),
                vec!["f1".to_string(), "f2".to_string()],
                2,
            )?,
        )?;

        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::from([
                ("artifact_mode".to_string(), "committee_hybrid".to_string()),
                ("fallback_blend_weight".to_string(), "0.3".to_string()),
            ]),
            committee_json: vec!["{}".to_string()],
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: Some(Array1::zeros(3)),
        };
        write_json(&path.join(MODEL_FILE_NAME), &artifact)?;

        let mut expert = OnlineHoeffdingExpert::new(None);
        expert.load(&path)?;

        assert_eq!(expert.committee_json.len(), 1);
        assert_eq!(
            expert.params.get("artifact_mode").map(String::as_str),
            Some("committee_hybrid")
        );
        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn quadratic_hoeffding_fallback_basis_appends_squared_terms() {
        let expanded =
            expand_hoeffding_fallback_features(&[0.5, -0.25], HoeffdingFallbackBasis::Quadratic);
        assert_eq!(expanded, vec![0.5, -0.25, 0.25, 0.0625]);
    }

    #[test]
    fn online_pa_save_rejects_missing_feature_schema() -> Result<()> {
        let path = temp_model_dir("online_pa_missing_schema");
        std::fs::create_dir_all(&path)?;

        let mut expert = OnlinePassiveAggressiveExpert::new(1.0, 4);
        expert.dataset_rows = 8;
        expert.scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        expert.weights = Some(Array2::zeros((3, 2)));
        expert.bias = Some(Array1::zeros(3));

        let err = expert
            .save(&path)
            .expect_err("saving without feature columns should fail");
        assert!(
            err.to_string().contains("feature column"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&path);
        Ok(())
    }

    #[test]
    fn online_pa_runtime_details_mark_missing_schema_as_degraded() {
        let mut expert = OnlinePassiveAggressiveExpert::new(1.0, 4);
        expert.dataset_rows = 8;
        expert.scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        expert.weights = Some(Array2::zeros((3, 2)));
        expert.bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_pa_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("pa_feature_schema_missing")
        );
    }

    #[test]
    fn online_pa_runtime_details_mark_partial_runtime_state_as_degraded() {
        let mut expert = OnlinePassiveAggressiveExpert::new(1.0, 4);
        expert.dataset_rows = 8;
        expert.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        expert.scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        expert.weights = Some(Array2::zeros((3, 2)));

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_pa_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("pa_runtime_state_incomplete")
        );
    }

    #[test]
    fn validate_passive_aggressive_artifact_rejects_zero_epochs() {
        let artifact = PassiveAggressiveArtifact {
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            weights: Array2::zeros((3, 2)),
            bias: Array1::zeros(3),
            aggressiveness: 1.0,
            epochs: 0,
            class_slack_weight_policy: legacy_online_pa_class_slack_weight_policy(),
            class_slack_weights: None,
            training_semantics_schema: legacy_online_pa_training_semantics(),
            training_backend: legacy_online_pa_training_backend(),
            effective_device_policy: legacy_online_pa_effective_device_policy(),
            cuda_training_evidence: None,
        };

        let err = validate_passive_aggressive_artifact(&artifact)
            .expect_err("zero epochs should be rejected");
        assert!(err.to_string().contains("epochs"));
    }

    #[test]
    fn validate_passive_aggressive_artifact_rejects_wrong_model_name() {
        let artifact = PassiveAggressiveArtifact {
            model_name: "not_online_pa".to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 8,
            scaler: FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            },
            weights: Array2::zeros((3, 2)),
            bias: Array1::zeros(3),
            aggressiveness: 1.0,
            epochs: 4,
            class_slack_weight_policy: legacy_online_pa_class_slack_weight_policy(),
            class_slack_weights: None,
            training_semantics_schema: legacy_online_pa_training_semantics(),
            training_backend: legacy_online_pa_training_backend(),
            effective_device_policy: legacy_online_pa_effective_device_policy(),
            cuda_training_evidence: None,
        };

        let err = validate_passive_aggressive_artifact(&artifact)
            .expect_err("wrong model name should be rejected");
        assert!(err.to_string().contains("model mismatch"));
    }

    #[test]
    fn validate_adaptive_metadata_rejects_inconsistent_training_summary() {
        let metadata = RuntimeArtifactMetadata {
            schema_version: crate::runtime::artifacts::RUNTIME_ARTIFACT_METADATA_SCHEMA_VERSION,
            model_name: AdaptiveModelKind::PassiveAggressive
                .model_name()
                .to_string(),
            family: ModelFamily::Adaptive,
            state: CapabilityState::Implemented,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            label_mapping: canonical_three_class_label_mapping(),
            training_summary: TrainingSummaryMetadata::raw_for_validation(8, 7, 0),
        };

        let err = validate_adaptive_metadata(
            &metadata,
            AdaptiveModelKind::PassiveAggressive.model_name(),
        )
        .expect_err("inconsistent training summary should be rejected");
        assert!(err.to_string().contains("inconsistent"));
    }

    #[test]
    fn online_hoeffding_runtime_details_follow_effective_blend_weight() {
        let mut fallback_only = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "fallback_blend_weight".to_string(),
            "1.0".to_string(),
        )])));
        fallback_only.committee_json = vec!["committee-a".to_string()];
        fallback_only.committee_runtime_ready = true;
        fallback_only.fallback_scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        fallback_only.fallback_weights = Some(Array2::zeros((3, 2)));
        fallback_only.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = fallback_only.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_blend_disabled_by_weight")
        );

        let mut committee_only = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "fallback_blend_weight".to_string(),
            "0.0".to_string(),
        )])));
        committee_only.committee_json = vec!["committee-a".to_string()];
        committee_only.committee_runtime_ready = true;
        committee_only.fallback_scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        committee_only.fallback_weights = Some(Array2::zeros((3, 2)));
        committee_only.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = committee_only.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );
    }

    #[test]
    fn online_hoeffding_runtime_details_mark_hybrid_blend_as_degraded() {
        let mut hybrid = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "fallback_blend_weight".to_string(),
            "0.35".to_string(),
        )])));
        hybrid.committee_json = vec!["committee-a".to_string()];
        hybrid.committee_runtime_ready = true;
        hybrid.fallback_scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        hybrid.fallback_weights = Some(Array2::zeros((3, 2)));
        hybrid.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = hybrid.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );
    }

    #[test]
    fn online_hoeffding_runtime_details_mark_persisted_committees_without_runtime_backend_as_degraded()
     {
        let mut expert = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "fallback_blend_weight".to_string(),
            "0.35".to_string(),
        )])));
        expert.committee_json = vec!["committee-a".to_string()];
        expert.committee_runtime_ready = false;
        expert.fallback_scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        expert.fallback_weights = Some(Array2::zeros((3, 2)));
        expert.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );
    }

    #[test]
    fn online_hoeffding_runtime_details_mark_committee_only_artifact_without_runtime_backend_as_unavailable()
     {
        let mut expert = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "artifact_mode".to_string(),
            "committee_only".to_string(),
        )])));
        expert.committee_json = vec!["committee-a".to_string()];
        expert.committee_runtime_ready = false;

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_unknown"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );
    }

    #[test]
    fn validate_hoeffding_artifact_rejects_blank_committee_payload() {
        let artifact = HoeffdingArtifact {
            model_name: AdaptiveModelKind::Hoeffding.model_name().to_string(),
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            dataset_rows: 2,
            params: HashMap::from([("fallback_blend_weight".to_string(), "0.3".to_string())]),
            committee_json: vec!["  ".to_string()],
            fallback_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = validate_hoeffding_artifact(&artifact).expect_err("blank committee should fail");
        assert!(err.to_string().contains("committee payloads"));
    }

    #[cfg(feature = "adaptive-models")]
    #[test]
    fn online_hoeffding_runtime_details_use_live_committees_not_just_runtime_flag() {
        let mut expert = OnlineHoeffdingExpert::new(Some(HashMap::from([(
            "fallback_blend_weight".to_string(),
            "0.0".to_string(),
        )])));
        expert.committee_json = vec!["committee-a".to_string()];
        expert.committee_runtime_ready = true;
        expert.committees.clear();
        expert.fallback_scaler = Some(FeatureScaler {
            means: vec![0.0, 0.0],
            stds: vec![1.0, 1.0],
        });
        expert.fallback_weights = Some(Array2::zeros((3, 2)));
        expert.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = expert.runtime_details();
        assert_eq!(backend.as_deref(), Some("online_hoeffding_fallback"));
        assert_eq!(
            degraded_reason.as_deref(),
            Some("committee_backend_unavailable")
        );
    }
}
