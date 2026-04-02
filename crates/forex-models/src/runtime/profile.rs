use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const TRAINING_RUNTIME_PROFILE_FILE_NAME: &str = "training_profile.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRuntimeProfile {
    pub model_name: String,
    pub symbol: String,
    pub base_timeframe: String,
    pub feature_count: usize,
    pub dataset_rows: usize,
    pub row_budget_applied: Option<usize>,
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

pub fn write_training_runtime_profile(path: &Path, profile: &TrainingRuntimeProfile) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create training profile dir {}", parent.display()))?;
    }

    let payload =
        serde_json::to_vec_pretty(profile).context("serialize training runtime profile")?;
    std::fs::write(path, payload)
        .with_context(|| format!("write training runtime profile to {}", path.display()))
}
