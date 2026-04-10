use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use super::capabilities::{CapabilityState, ModelFamily};

pub const TRAINING_RUNTIME_PROFILE_FILE_NAME: &str = "training_profile.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRuntimeProfile {
    pub model_name: String,
    pub capability_family: ModelFamily,
    pub capability_state: CapabilityState,
    pub symbol: String,
    pub base_timeframe: String,
    pub feature_count: usize,
    pub dataset_rows: usize,
    pub row_budget_applied: Option<usize>,
    pub label_horizon_bars: usize,
    pub effective_label_horizon_bars: usize,
    pub meta_label_max_hold_bars: usize,
    pub label_use_triple_barrier: bool,
    pub higher_timeframes: Vec<String>,
    pub multi_resolution_enabled: bool,
    pub base_features_prefixed: bool,
    pub base_signal_filter_enabled: bool,
    pub l1_feature_selection_enabled: bool,
    pub requested_backend: Option<String>,
    pub requested_device: Option<String>,
    pub checkpoint_path: Option<PathBuf>,
    pub async_requested: bool,
    pub async_wait_requested: bool,
    pub train_years: usize,
    pub val_years: usize,
    pub requested_hpo_backend: String,
    pub requested_hpo_trials: usize,
    pub holdout_pct: f64,
    pub embargo_minutes: usize,
    pub export_onnx_requested: bool,
    pub rllib_requested: bool,
    pub rllib_num_workers: usize,
    pub ray_tune_max_concurrency: usize,
    pub ddp_enabled: bool,
    pub fsdp_enabled: bool,
    pub ddp_world_size: usize,
    pub symbol_hash_buckets: usize,
    pub notes: Vec<String>,
}

fn validate_training_runtime_profile(profile: &TrainingRuntimeProfile) -> Result<()> {
    if profile.model_name.trim().is_empty() {
        anyhow::bail!("training runtime profile model_name must not be empty");
    }
    if profile.symbol.trim().is_empty() {
        anyhow::bail!("training runtime profile symbol must not be empty");
    }
    if profile.base_timeframe.trim().is_empty() {
        anyhow::bail!("training runtime profile base_timeframe must not be empty");
    }
    if profile.feature_count == 0 {
        anyhow::bail!("training runtime profile feature_count must be non-zero");
    }
    if profile.dataset_rows == 0 {
        anyhow::bail!("training runtime profile dataset_rows must be non-zero");
    }
    if let Some(row_budget_applied) = profile.row_budget_applied {
        if row_budget_applied == 0 {
            anyhow::bail!("training runtime profile row_budget_applied must be non-zero when set");
        }
        if row_budget_applied > profile.dataset_rows {
            anyhow::bail!(
                "training runtime profile row_budget_applied must not exceed dataset_rows"
            );
        }
    }
    if profile.effective_label_horizon_bars < profile.label_horizon_bars {
        anyhow::bail!(
            "training runtime profile effective_label_horizon_bars must be >= label_horizon_bars"
        );
    }
    if !(0.0..1.0).contains(&profile.holdout_pct) {
        anyhow::bail!("training runtime profile holdout_pct must be inside [0, 1)");
    }
    if profile.requested_hpo_backend.trim().is_empty() {
        anyhow::bail!("training runtime profile requested_hpo_backend must not be empty");
    }
    if profile.requested_hpo_trials == 0 {
        anyhow::bail!("training runtime profile requested_hpo_trials must be non-zero");
    }
    if profile.ddp_enabled && profile.ddp_world_size == 0 {
        anyhow::bail!("training runtime profile ddp_world_size must be non-zero when ddp_enabled");
    }
    if !profile.ddp_enabled && profile.ddp_world_size > 1 {
        anyhow::bail!(
            "training runtime profile ddp_world_size > 1 requires ddp_enabled to be true"
        );
    }
    if profile
        .requested_backend
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        anyhow::bail!("training runtime profile requested_backend must not be blank");
    }
    if profile
        .requested_device
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        anyhow::bail!("training runtime profile requested_device must not be blank");
    }
    Ok(())
}

pub fn write_training_runtime_profile(path: &Path, profile: &TrainingRuntimeProfile) -> Result<()> {
    validate_training_runtime_profile(profile)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create training profile dir {}", parent.display()))?;
    }

    let temp_path = path.with_extension("tmp_training_profile");
    let backup_path = path.with_extension("bak_training_profile");
    let payload =
        serde_json::to_vec_pretty(profile).context("serialize training runtime profile")?;
    if temp_path.exists() {
        std::fs::remove_file(&temp_path).with_context(|| {
            format!(
                "remove stale staged training profile {}",
                temp_path.display()
            )
        })?;
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove stale backup training profile {}",
                backup_path.display()
            )
        })?;
    }
    std::fs::write(&temp_path, payload).with_context(|| {
        format!(
            "write staged training runtime profile to {}",
            temp_path.display()
        )
    })?;
    if path.exists() {
        std::fs::rename(path, &backup_path)
            .with_context(|| format!("backup training runtime profile {}", path.display()))?;
    }
    if let Err(error) = std::fs::rename(&temp_path, path) {
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, path);
        } else if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }
        anyhow::bail!(
            "write training runtime profile to {} failed: {}",
            path.display(),
            error
        );
    }
    if backup_path.exists() {
        std::fs::remove_file(&backup_path)
            .with_context(|| format!("remove backup training profile {}", backup_path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> TrainingRuntimeProfile {
        TrainingRuntimeProfile {
            model_name: "lightgbm".to_string(),
            capability_family: ModelFamily::Tree,
            capability_state: CapabilityState::Implemented,
            symbol: "EURUSD".to_string(),
            base_timeframe: "M1".to_string(),
            feature_count: 32,
            dataset_rows: 1_000,
            row_budget_applied: Some(800),
            label_horizon_bars: 8,
            effective_label_horizon_bars: 12,
            meta_label_max_hold_bars: 12,
            label_use_triple_barrier: true,
            higher_timeframes: vec!["H1".to_string()],
            multi_resolution_enabled: true,
            base_features_prefixed: true,
            base_signal_filter_enabled: false,
            l1_feature_selection_enabled: false,
            requested_backend: Some("lightgbm".to_string()),
            requested_device: Some("cuda:0".to_string()),
            checkpoint_path: None,
            async_requested: false,
            async_wait_requested: false,
            train_years: 2,
            val_years: 1,
            requested_hpo_backend: "optuna".to_string(),
            requested_hpo_trials: 20,
            holdout_pct: 0.2,
            embargo_minutes: 60,
            export_onnx_requested: false,
            rllib_requested: false,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            ddp_enabled: false,
            fsdp_enabled: false,
            ddp_world_size: 1,
            symbol_hash_buckets: 64,
            notes: Vec::new(),
        }
    }

    #[test]
    fn training_runtime_profile_rejects_blank_model_name() {
        let mut profile = sample_profile();
        profile.model_name = " ".to_string();
        let err = validate_training_runtime_profile(&profile)
            .expect_err("blank model_name must fail")
            .to_string();
        assert!(err.contains("model_name"));
    }

    #[test]
    fn training_runtime_profile_rejects_zero_hpo_trials() {
        let mut profile = sample_profile();
        profile.requested_hpo_trials = 0;
        let err = validate_training_runtime_profile(&profile)
            .expect_err("zero HPO trials must fail")
            .to_string();
        assert!(err.contains("requested_hpo_trials"));
    }
}
