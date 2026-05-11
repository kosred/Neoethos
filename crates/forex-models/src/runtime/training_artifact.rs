use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::base::feature_columns_from_dataframe;
use crate::parallel_trainer::{ModelConfig, TrainingPayload};
use crate::runtime::capabilities::{
    ModelFamily, runtime_backend_kind_from_label, runtime_mode_from_details,
    typed_runtime_degraded_reason,
};
use crate::runtime::profile::{
    TRAINING_MODEL_ARTIFACT_FILE_NAME, TrainingRuntimeProfile, write_training_model_artifact,
};
use forex_core::storage::json::stable_json_hash;
use forex_core::system::HardwareProbe;
use forex_core::utils::{fnv1a64, fnv1a64_update};
use forex_core::{
    ArtifactKind, ArtifactProvenance, BackendKind, DeterminismPolicy, DeviceAssignment,
    RuntimeMode, TrainingModelArtifact,
};

pub fn write_training_model_artifact_contract_sidecar(
    artifact_dir: &Path,
    settings: &forex_core::Settings,
    config: &ModelConfig,
    payload: &TrainingPayload,
    profile: &TrainingRuntimeProfile,
) -> Result<()> {
    let artifact = build_training_model_artifact_contract(settings, config, payload, profile)?;
    write_training_model_artifact(
        &artifact_dir.join(TRAINING_MODEL_ARTIFACT_FILE_NAME),
        &artifact,
    )
}

fn build_training_model_artifact_contract(
    settings: &forex_core::Settings,
    config: &ModelConfig,
    payload: &TrainingPayload,
    profile: &TrainingRuntimeProfile,
) -> Result<TrainingModelArtifact<TrainingRuntimeProfile>> {
    let backend_label = training_runtime_backend_label(profile, config);
    let backend_kind =
        runtime_backend_kind_from_label(Some(&backend_label)).unwrap_or(BackendKind::Unavailable);
    let device_assignment = training_device_assignment(backend_kind, profile);
    let degraded_message = if backend_kind.is_degraded() {
        Some(format!(
            "training runtime backend `{backend_label}` is degraded or unavailable"
        ))
    } else {
        None
    };
    let runtime_mode = runtime_mode_from_details(Some(backend_kind), degraded_message.as_deref())
        .unwrap_or(RuntimeMode::Canonical);
    let runtime_degraded_reason = typed_runtime_degraded_reason(degraded_message.as_deref());
    let feature_columns = feature_columns_from_dataframe(&payload.frame);

    let provenance = ArtifactProvenance::new(
        ArtifactKind::TrainingModel,
        stable_json_hash(&serde_json::json!({
            "feature_columns": feature_columns,
            "feature_count": payload.frame.width(),
        }))?,
        training_dataset_fingerprint(payload),
        stable_json_hash(&serde_json::json!({
            "symbols": [profile.symbol.as_str()],
        }))?,
        stable_json_hash(&serde_json::json!({
            "base_timeframe": profile.base_timeframe.as_str(),
            "higher_timeframes": &profile.higher_timeframes,
        }))?,
        stable_json_hash(&serde_json::json!({
            "policy": "training_payload_frame_ordered_rows",
            "base_timeframe": profile.base_timeframe.as_str(),
            "dataset_rows": profile.dataset_rows,
        }))?,
        stable_json_hash(&serde_json::json!({
            "multi_resolution_enabled": profile.multi_resolution_enabled,
            "base_features_prefixed": profile.base_features_prefixed,
            "base_signal_filter_enabled": profile.base_signal_filter_enabled,
            "feature_columns": feature_columns_from_dataframe(&payload.frame),
        }))?,
        stable_json_hash(&serde_json::json!({
            "label_horizon_bars": profile.label_horizon_bars,
            "effective_label_horizon_bars": profile.effective_label_horizon_bars,
            "meta_label_max_hold_bars": profile.meta_label_max_hold_bars,
            "label_use_triple_barrier": profile.label_use_triple_barrier,
            "label_histogram": training_label_histogram(payload.labels.as_slice()),
        }))?,
        stable_json_hash(&serde_json::json!({
            "model_name": config.name.as_str(),
            "model_type": format!("{:?}", config.model_type),
            "capability_family": config.capability_family.to_string(),
            "capability_state": config.capability_state.to_string(),
            "params": sorted_training_params(&config.params),
            "train_years": profile.train_years,
            "val_years": profile.val_years,
            "holdout_pct": profile.holdout_pct,
            "embargo_minutes": profile.embargo_minutes,
            "requested_hpo_backend": profile.requested_hpo_backend.as_str(),
            "requested_hpo_trials": profile.requested_hpo_trials,
        }))?,
        stable_json_hash(&serde_json::json!({
            "scope": "not_applicable",
            "producer": "training_orchestrator",
        }))?,
        stable_json_hash(&serde_json::json!({
            "backend_label": backend_label.as_str(),
            "backend_kind": format!("{:?}", backend_kind),
            "device": device_assignment.device.as_str(),
            "device_ids": &device_assignment.device_ids,
            "requested_backend": profile.requested_backend.as_deref(),
            "requested_device": profile.requested_device.as_deref(),
            "planned_backend": profile.planned_backend.as_deref(),
            "planned_device": profile.planned_device.as_deref(),
            "planned_precision": profile.planned_precision.as_deref(),
            "ddp_enabled": profile.ddp_enabled,
            "fsdp_enabled": profile.fsdp_enabled,
            "ddp_world_size": profile.ddp_world_size,
        }))?,
        stable_json_hash(&serde_json::json!({
            "meta_label_max_hold_bars": settings.risk.meta_label_max_hold_bars,
            "triple_barrier_max_bars": settings.risk.triple_barrier_max_bars,
            "vol_horizon_bars": settings.risk.vol_horizon_bars,
            "meta_label_min_dist": settings.risk.meta_label_min_dist,
            "conformal_enabled": settings.risk.conformal_enabled,
        }))?,
        DeterminismPolicy::BestEffort,
        training_hardware_profile_id(),
        device_assignment,
        backend_kind,
        runtime_mode,
        runtime_degraded_reason,
        training_source_commit(),
    )
    .context("build training model artifact provenance")?;

    TrainingModelArtifact::new(provenance, profile.clone())
        .context("build training model artifact contract envelope")
}

fn sorted_training_params(params: &HashMap<String, String>) -> BTreeMap<String, String> {
    params
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn training_label_histogram(labels: &[i32]) -> BTreeMap<i32, usize> {
    let mut histogram = BTreeMap::new();
    for label in labels {
        *histogram.entry(*label).or_insert(0) += 1;
    }
    histogram
}

fn training_dataset_fingerprint(payload: &TrainingPayload) -> String {
    let mut hash = fnv1a64(b"training-dataset-v1");
    hash = fnv1a64_update(hash, &(payload.frame.height() as u64).to_le_bytes());
    hash = fnv1a64_update(hash, &(payload.frame.width() as u64).to_le_bytes());
    for column_name in feature_columns_from_dataframe(&payload.frame) {
        hash = fnv1a64_update(hash, column_name.as_bytes());
        hash = fnv1a64_update(hash, b"\0");
    }
    for value in payload.dense_features.iter() {
        hash = fnv1a64_update(hash, &value.to_le_bytes());
    }
    for label in payload.labels.iter() {
        hash = fnv1a64_update(hash, &label.to_le_bytes());
    }
    format!("fnv64:{hash:016x}")
}

fn training_runtime_backend_label(
    profile: &TrainingRuntimeProfile,
    config: &ModelConfig,
) -> String {
    let raw_backend = profile
        .planned_backend
        .as_deref()
        .or(profile.requested_backend.as_deref())
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase();
    let fallback = match config.capability_family {
        ModelFamily::Tree => "tree_cpu",
        ModelFamily::Deep | ModelFamily::Exit => "burn_cpu",
        ModelFamily::Forecasting
        | ModelFamily::Meta
        | ModelFamily::Evolutionary
        | ModelFamily::Adaptive
        | ModelFamily::Anomaly
        | ModelFamily::Rl => "native_cpu",
    };
    if raw_backend.is_empty() || raw_backend == "auto" {
        return fallback.to_string();
    }

    match config.capability_family {
        ModelFamily::Tree if raw_backend == "cpu" => "tree_cpu".to_string(),
        ModelFamily::Tree
            if raw_backend == "gpu"
                || raw_backend == "cuda"
                || raw_backend.starts_with("gpu:")
                || raw_backend.starts_with("cuda:") =>
        {
            "tree_gpu".to_string()
        }
        ModelFamily::Deep | ModelFamily::Exit if raw_backend == "cpu" => "burn_cpu".to_string(),
        ModelFamily::Deep | ModelFamily::Exit
            if matches!(
                raw_backend.as_str(),
                "wgpu" | "vulkan" | "metal" | "dx12" | "rocm"
            ) =>
        {
            "wgpu".to_string()
        }
        _ => raw_backend,
    }
}

fn training_device_assignment(
    backend_kind: BackendKind,
    profile: &TrainingRuntimeProfile,
) -> DeviceAssignment {
    let fallback_device = default_training_device_for_backend(backend_kind);
    let device = profile
        .planned_device
        .as_deref()
        .or(profile.requested_device.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_device)
        .to_string();
    let device_ids = training_device_ids(&device);
    DeviceAssignment {
        backend: backend_kind,
        device,
        device_ids,
    }
}

fn default_training_device_for_backend(backend_kind: BackendKind) -> &'static str {
    match backend_kind {
        BackendKind::NativeCuda | BackendKind::CudaKernel => "cuda:0",
        BackendKind::NativeTreeGpu => "gpu:0",
        BackendKind::BurnWgpu => "wgpu:0",
        BackendKind::NativeCpu
        | BackendKind::BurnCpu
        | BackendKind::NativeTreeCpu
        | BackendKind::CpuReference
        | BackendKind::LocalSurrogateFallback
        | BackendKind::ExternalRuntime
        | BackendKind::Unavailable => "cpu",
    }
}

fn training_device_ids(device: &str) -> Vec<usize> {
    device
        .split([',', ';'])
        .filter_map(|token| {
            let token = token.trim();
            let candidate = token
                .rsplit_once(':')
                .map(|(_, suffix)| suffix)
                .unwrap_or(token);
            candidate.parse::<usize>().ok()
        })
        .collect()
}

fn training_hardware_profile_id() -> String {
    HardwareProbe::new().detect().stable_id()
}

fn training_source_commit() -> String {
    std::env::var("FOREX_AI_SOURCE_COMMIT")
        .or_else(|_| std::env::var("GIT_COMMIT_HASH"))
        .or_else(|_| std::env::var("GITHUB_SHA"))
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown-local-source".to_string())
}
