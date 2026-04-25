use anyhow::{Context, Result, bail};
#[cfg(feature = "reinforcement-learning")]
use candle_core::{DType, Device, Module};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
#[cfg(feature = "reinforcement-learning")]
use rlkit::network::NeuralNetwork;
#[cfg(feature = "reinforcement-learning")]
use rlkit::policies::EpsilonGreedy;
#[cfg(feature = "reinforcement-learning")]
use rlkit::types::{Action, EnvTrait, Reward, Status};
#[cfg(feature = "reinforcement-learning")]
use rlkit::{Algorithm, DNQStateMode, DQN, TrainArgs};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::base::{
    build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{
    CapabilityState, ModelFamily, normalize_training_precision_policy,
    requested_training_precision_policy,
};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, feature_matrix_from_dataframe, read_json,
    remap_three_class_labels, write_json,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingAction {
    Hold,
    Buy,
    Sell,
}

impl TradingAction {
    pub fn all() -> [Self; 3] {
        [Self::Hold, Self::Buy, Self::Sell]
    }

    pub fn as_index(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Buy => 1,
            Self::Sell => 2,
        }
    }

    pub fn from_index(index: usize) -> Result<Self> {
        match index {
            0 => Ok(Self::Hold),
            1 => Ok(Self::Buy),
            2 => Ok(Self::Sell),
            other => bail!("unsupported trading action index {other}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingTransition {
    pub state: Vec<f32>,
    pub next_state: Vec<f32>,
    pub rewards: [f32; 3],
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingEpisode {
    pub transitions: Vec<TradingTransition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingStateEncoding {
    Normalized,
    Naive,
    OneHot,
}

impl TradingStateEncoding {
    #[cfg(feature = "reinforcement-learning")]
    fn as_rlkit(self) -> DNQStateMode {
        match self {
            Self::Normalized => DNQStateMode::Normalized,
            Self::Naive => DNQStateMode::Naive,
            Self::OneHot => DNQStateMode::OneHot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TradingFallbackBasis {
    #[default]
    Linear,
    Quadratic,
}

impl TradingFallbackBasis {
    fn expanded_dim(self, state_dim: usize) -> usize {
        match self {
            Self::Linear => state_dim,
            Self::Quadratic => state_dim.saturating_mul(2),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TradingRlArtifact {
    pub state_dim: usize,
    #[serde(default)]
    pub feature_columns: Vec<String>,
    #[serde(default)]
    pub train_rows: usize,
    pub hidden_dims: Vec<usize>,
    pub state_encoding: TradingStateEncoding,
    pub state_bins: u16,
    pub state_mins: Vec<f32>,
    pub state_maxs: Vec<f32>,
    pub buffer_capacity: usize,
    pub epochs: usize,
    pub max_steps: usize,
    pub update_interval: usize,
    pub update_freq: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_device_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_device_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_precision: Option<String>,
    pub backend: String,
    pub device_policy: String,
    pub parallel_envs: usize,
    pub eval_episodes: usize,
    pub rllib_num_workers: usize,
    pub ray_tune_max_concurrency: usize,
    pub reward_horizon: usize,
    pub episode_len: usize,
    #[serde(default)]
    pub training_report: Option<TradingRlTrainingReport>,
    #[serde(default)]
    pub feature_scaler: Option<FeatureScaler>,
    #[serde(default)]
    pub fallback_basis: TradingFallbackBasis,
    #[serde(default)]
    pub fallback_weights: Option<Array2<f32>>,
    #[serde(default)]
    pub fallback_bias: Option<ndarray::Array1<f32>>,
}

impl Default for TradingRlArtifact {
    fn default() -> Self {
        Self {
            state_dim: 0,
            feature_columns: Vec::new(),
            train_rows: 0,
            hidden_dims: vec![256, 256],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: Vec::new(),
            state_maxs: Vec::new(),
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: None,
            requested_device_policy: None,
            effective_backend: None,
            effective_device_policy: None,
                        network_precision: None,
            backend: "rlkit_cpu".to_string(),
            device_policy: "auto".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: None,
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Linear,
            fallback_weights: None,
            fallback_bias: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TradingRlTrainingReport {
    pub train_rows: usize,
    pub episode_count: usize,
    pub state_dim: usize,
    pub reward_horizon: usize,
    pub episode_len: usize,
    pub backend: String,
    pub device_policy: String,
    pub average_hold_reward: f32,
    pub average_buy_reward: f32,
    pub average_sell_reward: f32,
    pub used_network_snapshot: bool,
    pub used_fallback_q: bool,
    #[serde(default)]
    pub used_feature_scaler: bool,
}

#[derive(Debug, Clone)]
struct FeatureBounds {
    mins: Vec<f32>,
    maxs: Vec<f32>,
}

fn validate_q_values(q_values: Vec<f32>) -> Result<Vec<f32>> {
    if q_values.len() != 3 {
        bail!(
            "RL policy returned {} Q-values instead of 3",
            q_values.len()
        );
    }
    if q_values.iter().any(|value| !value.is_finite()) {
        bail!("RL policy returned non-finite Q-values");
    }

    Ok(q_values)
}

fn softmax_q_values(q_values: &[f32]) -> Result<[f32; 3]> {
    if q_values.len() != 3 {
        bail!(
            "RL probability projection requires exactly 3 Q-values, received {}",
            q_values.len()
        );
    }
    if q_values.iter().any(|value| !value.is_finite()) {
        bail!("RL probability projection received non-finite Q-values");
    }

    let max_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut logits = [0.0_f32; 3];
    let mut normalizer = 0.0_f32;
    for (index, value) in q_values.iter().copied().enumerate() {
        let projected = (value - max_q).exp();
        if !projected.is_finite() {
            bail!("RL probability projection overflowed during softmax");
        }
        logits[index] = projected;
        normalizer += projected;
    }

    if !normalizer.is_finite() || normalizer <= f32::EPSILON {
        bail!("RL probability projection produced an invalid softmax normalizer");
    }

    for value in &mut logits {
        *value /= normalizer;
    }
    Ok(logits)
}

fn default_rl_feature_columns(state_dim: usize) -> Vec<String> {
    (0..state_dim)
        .map(|idx| format!("state_{idx:03}"))
        .collect()
}

fn expand_fallback_basis(values: &[f32], basis: TradingFallbackBasis) -> Vec<f32> {
    match basis {
        TradingFallbackBasis::Linear => values.to_vec(),
        TradingFallbackBasis::Quadratic => {
            let mut expanded = Vec::with_capacity(values.len().saturating_mul(2));
            expanded.extend_from_slice(values);
            expanded.extend(values.iter().map(|value| value * value));
            expanded
        }
    }
}

fn fallback_backend_name(basis: TradingFallbackBasis) -> &'static str {
    match basis {
        TradingFallbackBasis::Linear => "linear_q_cpu",
        TradingFallbackBasis::Quadratic => "quadratic_q_cpu",
    }
}

fn resolve_rl_training_precision_with_capability(
    requested: Option<&str>,
    backend: &str,
    device_policy: &str,
    bf16_supported: Option<bool>,
) -> (String, Option<String>) {
    let requested = requested
        .map(normalize_training_precision_policy)
        .unwrap_or_else(|| requested_training_precision_policy("dqn"));
    let effective_backend = backend.trim().to_ascii_lowercase();
    let effective_backend = if effective_backend.is_empty() {
        "rlkit_cpu".to_string()
    } else {
        effective_backend
    };
    let effective_device_policy = normalize_rl_device_policy(device_policy);
    let rlkit_cuda_runtime =
        effective_backend == "rlkit_cuda" && effective_device_policy.starts_with("cuda:");
    let rlkit_cpu_runtime = effective_backend == "rlkit_cpu" && effective_device_policy == "cpu";
    let bf16_available = if rlkit_cpu_runtime {
        true
    } else if rlkit_cuda_runtime {
        bf16_supported.unwrap_or(true)
    } else {
        false
    };

    match requested.as_str() {
        "auto" if bf16_available => return ("bf16".to_string(), None),
        "auto" | "fp32" => return ("fp32".to_string(), None),
        "bf16" if bf16_available => return ("bf16".to_string(), None),
        _ => {}
    }

    let mut reasons = vec![format!("requested_rl_precision_unavailable({requested})")];

    match requested.as_str() {
        "bf16" => {
            if !(rlkit_cuda_runtime || rlkit_cpu_runtime) {
                reasons.push(format!(
                    "rl_backend_precision_limit({effective_backend}->fp32)"
                ));
            } else if bf16_supported == Some(false) {
                reasons.push(format!(
                    "rl_device_bf16_unavailable({effective_device_policy})"
                ));
            } else {
                reasons.push(format!(
                    "rl_device_bf16_probe_unavailable({effective_device_policy})"
                ));
            }
        }
        "fp8" | "bf4" => {
            let degraded_to = if bf16_available { "bf16" } else { "fp32" };
            reasons.push(format!("rl_precision_degraded_to_{degraded_to}"));
            return (degraded_to.to_string(), Some(reasons.join("; ")));
        }
        _ => {}
    }

    ("fp32".to_string(), Some(reasons.join("; ")))
}

fn is_known_rl_requested_backend(value: &str) -> bool {
    matches!(value, "native" | "rlkit" | "rlkit_cpu" | "rlkit_cuda")
        || value.starts_with("linear_q_")
        || value.starts_with("quadratic_q_")
}

fn is_known_rl_effective_backend(value: &str) -> bool {
    matches!(value, "rlkit_cpu" | "rlkit_cuda")
        || value.starts_with("linear_q_")
        || value.starts_with("quadratic_q_")
}

fn is_known_rl_requested_device_policy(value: &str) -> bool {
    let normalized = normalize_rl_device_policy(value);
    matches!(normalized.as_str(), "cpu" | "auto" | "gpu")
        || normalized.starts_with("gpu:")
        || normalized.starts_with("cuda:")
}

fn is_known_rl_effective_device_policy(value: &str) -> bool {
    let normalized = normalize_rl_device_policy(value);
    normalized == "cpu" || normalized.starts_with("cuda:") || normalized.starts_with("gpu:")
}

fn artifact_requested_backend(artifact: &TradingRlArtifact) -> String {
    artifact
        .requested_backend
        .clone()
        .unwrap_or_else(|| artifact.backend.clone())
}

fn artifact_requested_device_policy(artifact: &TradingRlArtifact) -> String {
    artifact
        .requested_device_policy
        .clone()
        .unwrap_or_else(|| artifact.device_policy.clone())
}

fn artifact_effective_backend(artifact: &TradingRlArtifact) -> String {
    artifact
        .effective_backend
        .clone()
        .unwrap_or_else(|| artifact.backend.clone())
}

fn artifact_effective_device_policy(artifact: &TradingRlArtifact) -> String {
    artifact
        .effective_device_policy
        .clone()
        .unwrap_or_else(|| artifact.device_policy.clone())
}

fn staged_rl_file(path: &Path, file_name: &str) -> PathBuf {
    path.join(format!("{file_name}.tmp"))
}

fn backup_rl_file(path: &Path, file_name: &str) -> PathBuf {
    path.join(format!("{file_name}.bak"))
}

fn rl_runtime_metadata(
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        "dqn",
        ModelFamily::Rl,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn requested_gpu_device_policy(policy: &str) -> bool {
    let normalized = normalize_rl_device_policy(policy);
    normalized == "gpu" || normalized.starts_with("gpu:")
}

fn normalize_rl_network_precision(value: &str) -> Result<String> {
    let normalized = normalize_training_precision_policy(value);
    if matches!(normalized.as_str(), "fp32" | "bf16") {
        Ok(normalized)
    } else {
        bail!(
            "RL network_precision `{}` is not supported; expected fp32 or bf16",
            value.trim()
        );
    }
}

#[cfg(feature = "reinforcement-learning")]
fn dtype_to_rl_network_precision(dtype: DType) -> Result<String> {
    match dtype {
        DType::F32 => Ok("fp32".to_string()),
        DType::BF16 => Ok("bf16".to_string()),
        other => bail!("RL network dtype {:?} is not supported for persistence", other),
    }
}

fn artifact_network_precision(artifact: &TradingRlArtifact) -> Result<Option<String>> {
    artifact
        .network_precision
        .as_deref()
        .map(normalize_rl_network_precision)
        .transpose()
}

#[cfg(all(
    feature = "reinforcement-learning",
    feature = "reinforcement-learning-cuda"
))]
fn probe_runtime_rl_bf16_support(backend: &str, device_policy: &str) -> Option<bool> {
    if backend.eq_ignore_ascii_case("rlkit_cpu")
        && normalize_rl_device_policy(device_policy) == "cpu"
    {
        return Some(true);
    }
    if !backend.eq_ignore_ascii_case("rlkit_cuda") {
        return Some(false);
    }

    let ordinal = normalize_rl_device_policy(device_policy)
        .strip_prefix("cuda:")
        .and_then(|value| value.parse::<usize>().ok())?;
    Device::new_cuda(ordinal)
        .ok()
        .map(|device| device.supports_bf16())
}

#[cfg(not(all(
    feature = "reinforcement-learning",
    feature = "reinforcement-learning-cuda"
)))]
fn probe_runtime_rl_bf16_support(backend: &str, device_policy: &str) -> Option<bool> {
    if backend.eq_ignore_ascii_case("rlkit_cpu")
        && normalize_rl_device_policy(device_policy) == "cpu"
    {
        Some(true)
    } else {
        Some(false)
    }
}

fn validate_rl_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if metadata.model_name != "dqn" {
        bail!(
            "RL metadata model mismatch: expected dqn, got {}",
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Rl {
        bail!(
            "RL metadata family mismatch: expected {:?}, got {:?}",
            ModelFamily::Rl,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "RL metadata state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("RL metadata label mapping mismatch");
    }
    if expected_feature_columns.is_empty() {
        bail!("RL metadata validation requires non-empty feature columns");
    }
    if metadata.feature_columns != expected_feature_columns {
        bail!(
            "RL metadata feature-column mismatch: expected {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        bail!(
            "RL metadata dataset-row mismatch: expected {}, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        bail!(
            "RL metadata rows are inconsistent: train_rows {} + val_rows {} != dataset_rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }
    Ok(())
}

fn resolve_rl_runtime_metadata(
    path: &Path,
    artifact: &TradingRlArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    let reconstructed = rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
    validate_rl_metadata(
        &reconstructed,
        &artifact.feature_columns,
        artifact.train_rows,
    )?;
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_rl_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
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
                    "RL metadata sidecar mismatch with reconstructed runtime metadata at {}",
                    metadata_path.display()
                );
            }
            Ok(metadata)
        }
        Err(file_err) => {
            tracing::warn!(
                path = %metadata_path.display(),
                error = %file_err,
                "RL metadata sidecar missing/unreadable; using reconstructed runtime metadata from artifact"
            );
            Ok(reconstructed)
        }
    }
}

fn cleanup_rl_temp_file(path: &Path) {
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
}

fn stage_rl_target(
    final_path: &Path,
    backup_path: &Path,
    staged_path: Option<&Path>,
) -> Result<bool> {
    if backup_path.exists() {
        std::fs::remove_file(backup_path).with_context(|| {
            format!("remove stale RL backup artifact {}", backup_path.display())
        })?;
    }
    let replaced_existing = final_path.exists();
    if replaced_existing {
        std::fs::rename(final_path, backup_path).with_context(|| {
            format!(
                "backup existing RL artifact {} to {}",
                final_path.display(),
                backup_path.display()
            )
        })?;
    }
    if let Some(staged_path) = staged_path {
        if let Err(err) = std::fs::rename(staged_path, final_path) {
            if replaced_existing && backup_path.exists() {
                let _ = std::fs::rename(backup_path, final_path);
            }
            cleanup_rl_temp_file(staged_path);
            return Err(err).with_context(|| {
                format!(
                    "promote staged RL artifact {} to {}",
                    staged_path.display(),
                    final_path.display()
                )
            });
        }
    }
    if replaced_existing && backup_path.exists() {
        let _ = std::fs::remove_file(backup_path);
    }
    Ok(replaced_existing)
}

fn restore_rl_backup(final_path: &Path, backup_path: &Path) {
    if backup_path.exists() {
        if final_path.exists() {
            let _ = std::fs::remove_file(final_path);
        }
        let _ = std::fs::rename(backup_path, final_path);
    }
}

fn rollback_rl_target(final_path: &Path, backup_path: &Path) {
    if backup_path.exists() {
        restore_rl_backup(final_path, backup_path);
    } else if final_path.exists() {
        let _ = std::fs::remove_file(final_path);
    }
}

fn build_reward_triplet(labels: &[usize], idx: usize, horizon: usize) -> [f32; 3] {
    let end = (idx + horizon).min(labels.len());
    let window = &labels[idx..end];
    if window.is_empty() {
        return [0.05, 0.0, 0.0];
    }

    let mut bullish = 0.0_f32;
    let mut bearish = 0.0_f32;
    let mut neutral = 0.0_f32;
    for (offset, label) in window.iter().copied().enumerate() {
        let weight = 1.0 / (offset + 1) as f32;
        match label {
            1 => bullish += weight,
            2 => bearish += weight,
            _ => neutral += weight,
        }
    }

    let directional_gap = (bullish - bearish).abs();
    let hold_reward = (neutral + 0.15 * (1.0 - directional_gap)).max(0.01);
    [
        hold_reward,
        bullish - bearish * 0.35 - neutral * 0.10,
        bearish - bullish * 0.35 - neutral * 0.10,
    ]
}

fn build_training_episodes(
    features: &Array2<f32>,
    labels: &[usize],
    episode_len: usize,
    horizon: usize,
) -> Result<Vec<TradingEpisode>> {
    if features.nrows() != labels.len() {
        bail!(
            "RL training row mismatch: {} feature rows vs {} labels",
            features.nrows(),
            labels.len()
        );
    }
    if features.nrows() < episode_len.max(16) + 2 {
        bail!(
            "RL training requires at least {} rows, received {}",
            episode_len.max(16) + 2,
            features.nrows()
        );
    }

    let mut episodes = Vec::new();
    let step = (episode_len / 2).max(8);
    let mut start = 0usize;
    while start + 2 < features.nrows() {
        let end = (start + episode_len).min(features.nrows() - 1);
        let mut transitions = Vec::with_capacity(end - start);
        for row_idx in start..end {
            let state = features.row(row_idx).iter().copied().collect::<Vec<_>>();
            let next_state = features
                .row((row_idx + 1).min(features.nrows() - 1))
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let done = row_idx + 1 >= end;
            transitions.push(TradingTransition {
                state,
                next_state,
                rewards: build_reward_triplet(labels, row_idx, horizon),
                done,
            });
        }
        if !transitions.is_empty() {
            episodes.push(TradingEpisode { transitions });
        }
        if end >= features.nrows() - 1 {
            break;
        }
        start += step;
    }

    if episodes.is_empty() {
        bail!("RL training could not build any episodes from the provided series");
    }

    Ok(episodes)
}

impl FeatureBounds {
    fn fit(episodes: &[TradingEpisode], state_dim: usize) -> Result<Self> {
        let mut mins = vec![f32::INFINITY; state_dim];
        let mut maxs = vec![f32::NEG_INFINITY; state_dim];

        for episode in episodes {
            for transition in &episode.transitions {
                if transition.state.len() != state_dim || transition.next_state.len() != state_dim {
                    bail!(
                        "RL transition state dimension mismatch; expected {}, got {} / {}",
                        state_dim,
                        transition.state.len(),
                        transition.next_state.len()
                    );
                }

                for (idx, value) in transition
                    .state
                    .iter()
                    .chain(transition.next_state.iter())
                    .enumerate()
                {
                    if !value.is_finite() {
                        bail!(
                            "RL feature bounds encountered non-finite state value {} at feature {}",
                            value,
                            idx % state_dim
                        );
                    }
                    let feature_idx = idx % state_dim;
                    mins[feature_idx] = mins[feature_idx].min(*value);
                    maxs[feature_idx] = maxs[feature_idx].max(*value);
                }
            }
        }

        for idx in 0..state_dim {
            if !mins[idx].is_finite() || !maxs[idx].is_finite() {
                bail!("RL feature bounds contain non-finite extrema at feature {idx}");
            }
            if (maxs[idx] - mins[idx]).abs() < 1e-6 {
                maxs[idx] = mins[idx] + 1.0;
            }
        }

        Ok(Self { mins, maxs })
    }

    fn discretize(&self, values: &[f32], bins: u16) -> Result<Vec<u16>> {
        if values.len() != self.mins.len() {
            bail!(
                "RL state dimension mismatch during discretization: expected {}, got {}",
                self.mins.len(),
                values.len()
            );
        }

        let upper = bins.saturating_sub(1).max(1) as f32;
        values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if !value.is_finite() {
                    bail!(
                        "RL state contains non-finite value {} at feature {} during discretization",
                        value,
                        idx
                    );
                }
                let min = self.mins[idx];
                let max = self.maxs[idx];
                let scaled = ((*value - min) / (max - min)).clamp(0.0, 1.0);
                Ok((scaled * upper).round() as u16)
            })
            .collect::<Result<Vec<_>>>()
    }

    fn normalize(&self, values: &[f32]) -> Result<Vec<f32>> {
        if values.len() != self.mins.len() {
            bail!(
                "RL state dimension mismatch during normalization: expected {}, got {}",
                self.mins.len(),
                values.len()
            );
        }

        values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if !value.is_finite() {
                    bail!(
                        "RL state contains non-finite value {} at feature {} during normalization",
                        value,
                        idx
                    );
                }
                let min = self.mins[idx];
                let max = self.maxs[idx];
                Ok(((*value - min) / (max - min)).clamp(0.0, 1.0))
            })
            .collect::<Result<Vec<_>>>()
    }
}

#[cfg(feature = "reinforcement-learning")]
fn normalize_rl_device_policy(policy: &str) -> String {
    let normalized = policy.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return "auto".to_string();
    }
    if matches!(
        normalized.as_str(),
        "nvidia" | "cuda" | "rocm" | "metal" | "vulkan"
    ) {
        return "gpu".to_string();
    }
    if let Some(index) = normalized
        .strip_prefix("cuda:")
        .or_else(|| normalized.strip_prefix("rocm:"))
        .or_else(|| normalized.strip_prefix("metal:"))
        .or_else(|| normalized.strip_prefix("vulkan:"))
        .or_else(|| normalized.strip_prefix("wgpu:"))
    {
        return format!("gpu:{index}");
    }
    if matches!(normalized.as_str(), "cpu" | "auto" | "gpu") || normalized.starts_with("gpu:") {
        return normalized;
    }
    "auto".to_string()
}

#[cfg(all(
    feature = "reinforcement-learning",
    feature = "reinforcement-learning-cuda"
))]
fn requested_cuda_ordinal(policy: &str) -> Option<usize> {
    normalize_rl_device_policy(policy)
        .strip_prefix("gpu:")
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(feature = "reinforcement-learning")]
fn resolve_rl_training_device(policy: &str) -> Result<(Device, String, String)> {
    let normalized = normalize_rl_device_policy(policy);
    let explicit_gpu = requested_gpu_device_policy(&normalized);

    #[cfg(feature = "reinforcement-learning-cuda")]
    {
        let ordinal = requested_cuda_ordinal(&normalized).unwrap_or(0);
        match normalized.as_str() {
            "cpu" => return Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string())),
            "auto" => match Device::new_cuda(ordinal) {
                Ok(device) => {
                    return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()));
                }
                Err(_) => return Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string())),
            },
            "gpu" => {
                let device = Device::new_cuda(ordinal)
                    .map_err(|err| anyhow::anyhow!("initialize RL CUDA device {ordinal}: {err}"))?;
                return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()));
            }
            value if value.starts_with("gpu:") => {
                let device = Device::new_cuda(ordinal)
                    .map_err(|err| anyhow::anyhow!("initialize RL CUDA device {ordinal}: {err}"))?;
                return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()));
            }
            _ => {}
        }
    }

    if explicit_gpu {
        return Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string()));
    }

    Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string()))
}

#[cfg(feature = "reinforcement-learning")]
fn resolve_rl_inference_device(policy: &str) -> (Device, String, String) {
    let normalized = normalize_rl_device_policy(policy);

    #[cfg(feature = "reinforcement-learning-cuda")]
    {
        let ordinal = requested_cuda_ordinal(&normalized).unwrap_or(0);
        if matches!(normalized.as_str(), "auto" | "gpu") || normalized.starts_with("gpu:") {
            if let Ok(device) = Device::new_cuda(ordinal) {
                return (device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string());
            }
        }
    }

    (Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string())
}

fn q_value_for_action(
    weights: &Array2<f32>,
    bias: &ndarray::Array1<f32>,
    state: &[f32],
    action: usize,
) -> f32 {
    weights
        .row(action)
        .iter()
        .zip(state.iter())
        .map(|(weight, value)| weight * value)
        .sum::<f32>()
        + bias[action]
}

fn sync_linear_q_target(
    source_weights: &Array2<f32>,
    source_bias: &ndarray::Array1<f32>,
    target_weights: &mut Array2<f32>,
    target_bias: &mut ndarray::Array1<f32>,
) {
    target_weights.assign(source_weights);
    target_bias.assign(source_bias);
}

#[cfg(feature = "reinforcement-learning")]
#[derive(Debug, Clone)]
struct EncodedTransition {
    state: Vec<u16>,
    next_state: Vec<u16>,
    rewards: [f32; 3],
    done: bool,
}

#[cfg(feature = "reinforcement-learning")]
#[derive(Debug, Clone)]
struct EncodedEpisode {
    transitions: Vec<EncodedTransition>,
}

#[cfg(feature = "reinforcement-learning")]
struct TradingEpisodeEnv {
    episodes: Vec<EncodedEpisode>,
    state_uppers: Vec<u16>,
    action_uppers: Vec<u16>,
    active_episode: usize,
    next_episode: usize,
    current_step: usize,
}

#[cfg(feature = "reinforcement-learning")]
impl TradingEpisodeEnv {
    fn new(episodes: Vec<EncodedEpisode>, state_dim: usize, state_bins: u16) -> Result<Self> {
        if episodes.is_empty() {
            bail!("RL training requires at least one episode");
        }
        if episodes
            .iter()
            .any(|episode| episode.transitions.is_empty())
        {
            bail!("RL training episodes may not be empty");
        }

        Ok(Self {
            episodes,
            state_uppers: vec![state_bins.max(2); state_dim],
            action_uppers: vec![3],
            active_episode: 0,
            next_episode: 0,
            current_step: 0,
        })
    }

    fn current_transition(&self) -> &EncodedTransition {
        &self.episodes[self.active_episode].transitions[self.current_step]
    }
}

#[cfg(feature = "reinforcement-learning")]
impl EnvTrait<u16, u16> for TradingEpisodeEnv {
    fn step(&mut self, _state: &Status<u16>, action: &Action<u16>) -> (Status<u16>, Reward, bool) {
        let transition = self.current_transition().clone();
        let action_idx = action.as_slice().first().copied().unwrap_or(0).min(2) as usize;
        let reward = transition.rewards[action_idx];

        let reached_end =
            self.current_step + 1 >= self.episodes[self.active_episode].transitions.len();
        let done = transition.done || reached_end;
        let next_state = if done {
            transition.next_state
        } else {
            self.current_step += 1;
            self.current_transition().state.clone()
        };

        (
            Status::new(next_state, self.state_uppers.clone()),
            Reward(reward),
            done,
        )
    }

    fn reset(&mut self) -> Status<u16> {
        self.current_step = 0;
        self.active_episode = self.next_episode;
        let state = self.episodes[self.active_episode].transitions[0]
            .state
            .clone();
        self.next_episode = (self.next_episode + 1) % self.episodes.len();
        Status::new(state, self.state_uppers.clone())
    }

    fn action_space(&self) -> &[u16] {
        &self.action_uppers
    }

    fn state_space(&self) -> &[u16] {
        &self.state_uppers
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct TradingReinforcementLearner {
    #[cfg(feature = "reinforcement-learning")]
    inference_network: Option<NeuralNetwork>,
    #[cfg(not(feature = "reinforcement-learning"))]
    inference_network: Option<()>,
    hidden_dims: Vec<usize>,
    state_encoding: TradingStateEncoding,
    state_bins: u16,
    buffer_capacity: usize,
    train_args: TradingRlArtifact,
    bounds: Option<FeatureBounds>,
    fallback_weights: Option<Array2<f32>>,
    fallback_bias: Option<ndarray::Array1<f32>>,
    feature_scaler: Option<FeatureScaler>,
    feature_columns: Vec<String>,
    training_report: Option<TradingRlTrainingReport>,
    runtime_effective_backend: Option<String>,
    runtime_effective_device_policy: Option<String>,
    persisted_network_snapshot_present: bool,
}

impl TradingReinforcementLearner {
    pub fn new() -> Self {
        let hidden_dims = vec![256, 256];
        let state_encoding = TradingStateEncoding::Normalized;
        let state_bins = 255;
        let buffer_capacity = 50_000;
        Self {
            inference_network: None,
            hidden_dims: hidden_dims.clone(),
            state_encoding,
            state_bins,
            buffer_capacity,
            train_args: TradingRlArtifact {
                hidden_dims,
                state_encoding,
                state_bins,
                buffer_capacity,
                ..TradingRlArtifact::default()
            },
            bounds: None,
            fallback_weights: None,
            fallback_bias: None,
            feature_scaler: None,
            feature_columns: Vec::new(),
            training_report: None,
            runtime_effective_backend: None,
            runtime_effective_device_policy: None,
            persisted_network_snapshot_present: false,
        }
    }

    pub fn with_hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        if !hidden_dims.is_empty() {
            self.hidden_dims = hidden_dims.clone();
            self.train_args.hidden_dims = hidden_dims;
        }
        self
    }

    pub fn with_state_bins(mut self, bins: u16) -> Self {
        self.state_bins = bins.max(4);
        self.train_args.state_bins = self.state_bins;
        self
    }

    pub fn with_encoding(mut self, encoding: TradingStateEncoding) -> Self {
        self.state_encoding = encoding;
        self.train_args.state_encoding = encoding;
        self
    }

    pub fn with_encoding_name(self, encoding: &str) -> Self {
        let parsed = match encoding.trim().to_ascii_lowercase().as_str() {
            "naive" => TradingStateEncoding::Naive,
            "onehot" | "one_hot" => TradingStateEncoding::OneHot,
            _ => TradingStateEncoding::Normalized,
        };
        self.with_encoding(parsed)
    }

    pub fn with_train_schedule(
        mut self,
        epochs: usize,
        max_steps: usize,
        batch_size: usize,
    ) -> Self {
        self.train_args.epochs = epochs.max(1);
        self.train_args.max_steps = max_steps.max(1);
        self.train_args.batch_size = batch_size.max(8);
        self
    }

    pub fn with_update_schedule(mut self, update_interval: usize, update_freq: usize) -> Self {
        self.train_args.update_interval = update_interval.max(1);
        self.train_args.update_freq = update_freq.max(1);
        self
    }

    pub fn with_optimizer(mut self, learning_rate: f64, gamma: f32) -> Self {
        if learning_rate.is_finite() && learning_rate > 0.0 {
            self.train_args.learning_rate = learning_rate;
        }
        if gamma.is_finite() {
            self.train_args.gamma = gamma.clamp(0.01, 0.9999);
        }
        self
    }

    pub fn with_exploration_schedule(
        mut self,
        epsilon_start: f32,
        epsilon_end: f32,
        epsilon_decay: f32,
    ) -> Self {
        if epsilon_start.is_finite() {
            self.train_args.epsilon_start = epsilon_start.clamp(0.0, 1.0);
        }
        if epsilon_end.is_finite() {
            self.train_args.epsilon_end = epsilon_end.clamp(0.0, 1.0);
        }
        if self.train_args.epsilon_end > self.train_args.epsilon_start {
            self.train_args.epsilon_end = self.train_args.epsilon_start;
        }
        if epsilon_decay.is_finite() {
            self.train_args.epsilon_decay = epsilon_decay.clamp(0.90, 0.99999);
        }
        self
    }

    pub fn with_buffer_capacity(mut self, buffer_capacity: usize) -> Self {
        self.buffer_capacity = buffer_capacity.max(512);
        self.train_args.buffer_capacity = self.buffer_capacity;
        self
    }

    pub fn with_runtime_hints(
        mut self,
        backend: impl Into<String>,
        device_policy: impl Into<String>,
        parallel_envs: usize,
        eval_episodes: usize,
        rllib_num_workers: usize,
        ray_tune_max_concurrency: usize,
    ) -> Self {
        let requested_backend = backend.into();
        let requested_device_policy = device_policy.into();
        self.train_args.backend = if requested_backend.trim().is_empty() {
            self.train_args.backend.clone()
        } else {
            requested_backend.trim().to_ascii_lowercase()
        };
        self.train_args.device_policy = if requested_device_policy.trim().is_empty() {
            self.train_args.device_policy.clone()
        } else {
            requested_device_policy.trim().to_ascii_lowercase()
        };
        self.train_args.requested_backend = Some(self.train_args.backend.clone());
        self.train_args.requested_device_policy = Some(self.train_args.device_policy.clone());
        self.train_args.parallel_envs = parallel_envs.max(1);
        self.train_args.eval_episodes = eval_episodes.max(1);
        self.train_args.rllib_num_workers = rllib_num_workers;
        self.train_args.ray_tune_max_concurrency = ray_tune_max_concurrency.max(1);
        self
    }

    pub fn with_episode_layout(mut self, reward_horizon: usize, episode_len: usize) -> Self {
        self.train_args.reward_horizon = reward_horizon;
        self.train_args.episode_len = episode_len;
        self
    }

    fn train_linear_q_fallback(&mut self, episodes: &[TradingEpisode]) -> Result<()> {
        if episodes.is_empty() {
            bail!("RL training requires at least one episode");
        }
        let first_state = episodes
            .iter()
            .find_map(|episode| {
                episode
                    .transitions
                    .first()
                    .map(|transition| transition.state.len())
            })
            .context("RL episodes do not contain any transitions")?;
        if first_state == 0 {
            bail!("RL state dimension may not be zero");
        }

        let bounds = FeatureBounds::fit(episodes, first_state)?;
        let fallback_basis = TradingFallbackBasis::Quadratic;
        let fallback_state_dim = fallback_basis.expanded_dim(first_state);
        let mut weights = Array2::<f32>::zeros((3, fallback_state_dim));
        let mut bias = ndarray::Array1::<f32>::zeros(3);
        let mut target_weights = weights.clone();
        let mut target_bias = bias.clone();
        let lr = self.train_args.learning_rate as f32;
        let gamma = self.train_args.gamma;
        let update_interval = self.train_args.update_interval.max(1);
        let gradient_clip = 5.0_f32;
        let l2 = 1e-4_f32;
        let mut updates = 0usize;

        for _ in 0..self.train_args.epochs.max(1) {
            for episode in episodes {
                for transition in &episode.transitions {
                    let state = expand_fallback_basis(
                        &bounds.normalize(&transition.state)?,
                        fallback_basis,
                    );
                    let next_state = expand_fallback_basis(
                        &bounds.normalize(&transition.next_state)?,
                        fallback_basis,
                    );

                    let next_best = if transition.done {
                        0.0
                    } else {
                        (0..3)
                            .map(|action| {
                                q_value_for_action(
                                    &target_weights,
                                    &target_bias,
                                    &next_state,
                                    action,
                                )
                            })
                            .max_by(|left, right| left.total_cmp(right))
                            .expect("three-action DQN target selection should always produce a max")
                    };

                    for action in 0..3 {
                        let prediction = q_value_for_action(&weights, &bias, &state, action);
                        let target = transition.rewards[action]
                            + if transition.done {
                                0.0
                            } else {
                                gamma * next_best
                            };
                        let td_error = (prediction - target).clamp(-gradient_clip, gradient_clip);
                        for (feature_idx, value) in state.iter().enumerate() {
                            let grad = td_error * *value + l2 * weights[(action, feature_idx)];
                            weights[(action, feature_idx)] -= lr * grad;
                        }
                        bias[action] -= lr * td_error;
                    }

                    updates += 1;
                    if updates.is_multiple_of(update_interval) {
                        sync_linear_q_target(
                            &weights,
                            &bias,
                            &mut target_weights,
                            &mut target_bias,
                        );
                    }
                }
            }
        }

        sync_linear_q_target(&weights, &bias, &mut target_weights, &mut target_bias);
        self.bounds = Some(bounds.clone());
        self.train_args.state_dim = first_state;
        if self.feature_columns.is_empty() {
            self.feature_columns = default_rl_feature_columns(first_state);
        }
        self.train_args.feature_columns = self.feature_columns.clone();
        self.train_args.state_mins = bounds.mins;
        self.train_args.state_maxs = bounds.maxs;
        self.train_args.fallback_basis = fallback_basis;
        self.train_args.fallback_weights = Some(weights.clone());
        self.train_args.fallback_bias = Some(bias.clone());
        self.train_args
            .requested_backend
            .get_or_insert_with(|| self.train_args.backend.clone());
        self.train_args
            .requested_device_policy
            .get_or_insert_with(|| self.train_args.device_policy.clone());
        let fallback_backend = fallback_backend_name(fallback_basis).to_string();
        self.train_args.effective_backend = Some(fallback_backend.clone());
        self.train_args.effective_device_policy = Some("cpu".to_string());
        self.train_args.network_precision = None;
        self.train_args.backend = fallback_backend;
        self.train_args.device_policy = "cpu".to_string();
        self.runtime_effective_backend = self.train_args.effective_backend.clone();
        self.runtime_effective_device_policy = self.train_args.effective_device_policy.clone();
        self.persisted_network_snapshot_present = false;
        self.fallback_weights = Some(weights);
        self.fallback_bias = Some(bias);
        Ok(())
    }

    fn build_training_report(&self, episodes: &[TradingEpisode]) -> TradingRlTrainingReport {
        let mut hold_reward_sum = 0.0_f32;
        let mut buy_reward_sum = 0.0_f32;
        let mut sell_reward_sum = 0.0_f32;
        let mut transition_count = 0usize;

        for episode in episodes {
            for transition in &episode.transitions {
                hold_reward_sum += transition.rewards[0];
                buy_reward_sum += transition.rewards[1];
                sell_reward_sum += transition.rewards[2];
                transition_count += 1;
            }
        }

        let denom = transition_count.max(1) as f32;
        TradingRlTrainingReport {
            train_rows: self.train_args.train_rows,
            episode_count: episodes.len(),
            state_dim: self.train_args.state_dim,
            reward_horizon: self.train_args.reward_horizon,
            episode_len: self.train_args.episode_len,
            backend: self
                .train_args
                .effective_backend
                .clone()
                .unwrap_or_else(|| self.train_args.backend.clone()),
            device_policy: self
                .train_args
                .effective_device_policy
                .clone()
                .unwrap_or_else(|| self.train_args.device_policy.clone()),
            average_hold_reward: hold_reward_sum / denom,
            average_buy_reward: buy_reward_sum / denom,
            average_sell_reward: sell_reward_sum / denom,
            used_network_snapshot: self.inference_network.is_some(),
            used_fallback_q: self.fallback_weights.is_some() && self.fallback_bias.is_some(),
            used_feature_scaler: self.feature_scaler.is_some(),
        }
    }

    #[cfg(feature = "reinforcement-learning")]
    pub fn train_on_episodes(&mut self, episodes: &[TradingEpisode]) -> Result<()> {
        let total_transitions = episodes
            .iter()
            .map(|episode| episode.transitions.len())
            .sum();
        if total_transitions == 0 {
            bail!("RL training requires at least one transition");
        }
        self.train_args.train_rows = total_transitions;
        self.train_linear_q_fallback(episodes)?;
        let first_state = self.train_args.state_dim;
        let bounds = self.bounds.as_ref().context("RL feature bounds missing")?;
        let encoded_episodes = episodes
            .iter()
            .map(|episode| {
                let transitions = episode
                    .transitions
                    .iter()
                    .map(|transition| {
                        Ok(EncodedTransition {
                            state: bounds.discretize(&transition.state, self.state_bins)?,
                            next_state: bounds
                                .discretize(&transition.next_state, self.state_bins)?,
                            rewards: transition.rewards,
                            done: transition.done,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(EncodedEpisode { transitions })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut env = TradingEpisodeEnv::new(encoded_episodes, first_state, self.state_bins)?;
        let (device, effective_policy, effective_backend) =
            resolve_rl_training_device(&self.train_args.device_policy)?;
        let requested_precision = requested_training_precision_policy("dqn");
        let (effective_precision, _precision_degraded_reason) =
            resolve_rl_training_precision_with_capability(
                Some(&requested_precision),
                &effective_backend,
                &effective_policy,
                probe_runtime_rl_bf16_support(&effective_backend, &effective_policy),
            );
        let model_dtype = match effective_precision.as_str() {
            "bf16" => DType::BF16,
            _ => DType::F32,
        };
        self.train_args
            .requested_backend
            .get_or_insert_with(|| self.train_args.backend.clone());
        self.train_args
            .requested_device_policy
            .get_or_insert_with(|| self.train_args.device_policy.clone());
        let mut model = DQN::new_with_dtype(
            &env,
            self.buffer_capacity,
            &self.hidden_dims,
            self.state_encoding.as_rlkit(),
            model_dtype,
            &device,
        )
        .map_err(|err| anyhow::anyhow!("create DQN model: {err}"))?;
        let mut policy = EpsilonGreedy::new(
            self.train_args
                .epsilon_start
                .max(self.train_args.epsilon_end),
            self.train_args.epsilon_end,
            self.train_args.epsilon_decay,
        );

        let train_args = TrainArgs {
            epochs: self.train_args.epochs,
            max_steps: self.train_args.max_steps,
            update_interval: self.train_args.update_interval,
            update_freq: self.train_args.update_freq,
            batch_size: self.train_args.batch_size,
            learning_rate: self.train_args.learning_rate,
            gamma: self.train_args.gamma,
        };

        model
            .train(&mut env, &mut policy, train_args)
            .map_err(|err| anyhow::anyhow!("train DQN policy: {err}"))?;

        let network = <DQN<u16, u16> as Algorithm<TradingEpisodeEnv, u16, u16>>::vars_any(&model)
            .downcast::<NeuralNetwork>()
            .map_err(|_| anyhow::anyhow!("extract DQN network snapshot"))?;

        self.inference_network = Some(*network);
        self.train_args.effective_backend = Some(effective_backend.clone());
        self.train_args.effective_device_policy = Some(effective_policy.clone());
        self.train_args.network_precision = Some(effective_precision);
        self.train_args.backend = effective_backend;
        self.train_args.device_policy = effective_policy;
        self.runtime_effective_backend = self.train_args.effective_backend.clone();
        self.runtime_effective_device_policy = self.train_args.effective_device_policy.clone();
        self.persisted_network_snapshot_present = true;
        let training_report = self.build_training_report(episodes);
        self.train_args.training_report = Some(training_report.clone());
        self.training_report = Some(training_report);
        Ok(())
    }

    #[cfg(not(feature = "reinforcement-learning"))]
    pub fn train_on_episodes(&mut self, episodes: &[TradingEpisode]) -> Result<()> {
        let total_transitions = episodes
            .iter()
            .map(|episode| episode.transitions.len())
            .sum();
        if total_transitions == 0 {
            bail!("RL training requires at least one transition");
        }
        self.train_args.train_rows = total_transitions;
        self.train_linear_q_fallback(episodes)?;
        let training_report = self.build_training_report(episodes);
        self.train_args.training_report = Some(training_report.clone());
        self.training_report = Some(training_report);
        Ok(())
    }

    pub fn train_on_frame(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let (features, _) = feature_matrix_from_dataframe(x)?;
        let feature_columns = x
            .get_column_names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        let scaler = FeatureScaler::fit(&features)?;
        let scaled = scaler.transform(&features)?;
        let labels = remap_three_class_labels(y)?;
        let horizon = if self.train_args.reward_horizon > 0 {
            self.train_args.reward_horizon.clamp(2, 128)
        } else {
            (scaled.nrows() / 24).clamp(6, 32)
        };
        let episode_len = if self.train_args.episode_len > 0 {
            self.train_args.episode_len.clamp(horizon.max(8), 512)
        } else {
            (scaled.nrows() / 12).clamp(24, 128)
        };
        let episodes = build_training_episodes(&scaled, &labels, episode_len, horizon)?;
        self.train_args.feature_columns = feature_columns.clone();
        self.feature_columns = feature_columns;
        self.feature_scaler = Some(scaler.clone());
        self.train_args.feature_scaler = Some(scaler);
        self.train_args.train_rows = scaled.nrows();
        self.train_on_episodes(&episodes)
    }

    fn artifact(&self) -> Result<TradingRlArtifact> {
        if self.train_args.state_dim == 0 || self.bounds.is_none() {
            bail!("RL artifact is unavailable before the learner is trained");
        }
        let mut artifact = self.train_args.clone();
        if !self.feature_columns.is_empty() {
            artifact.feature_columns = self.feature_columns.clone();
        }
        artifact.feature_scaler = self.feature_scaler.clone();
        #[cfg(feature = "reinforcement-learning")]
        {
            artifact.network_precision = self
                .inference_network
                .as_ref()
                .map(|network| dtype_to_rl_network_precision(network.dtype()))
                .transpose()?;
        }
        #[cfg(not(feature = "reinforcement-learning"))]
        {
            artifact.network_precision = None;
        }
        let persisted_report = self
            .train_args
            .training_report
            .as_ref()
            .context("RL artifact state is missing the persisted training_report")?;
        let training_report = self
            .training_report
            .as_ref()
            .context("RL artifact is missing a persisted training_report")?;
        if persisted_report != training_report {
            bail!("RL live training_report drifted from the persisted artifact report");
        }
        artifact.training_report = Some(persisted_report.clone());
        Self::validate_artifact(&artifact)?;
        Ok(artifact)
    }

    fn validate_artifact(artifact: &TradingRlArtifact) -> Result<()> {
        if artifact.state_dim == 0 {
            bail!("RL artifact is missing the trained state dimension");
        }
        if artifact.train_rows == 0 {
            bail!("RL artifact is missing training-row metadata");
        }
        if artifact.hidden_dims.is_empty() || artifact.hidden_dims.iter().any(|dim| *dim == 0) {
            bail!("RL artifact hidden_dims must contain only positive layer widths");
        }
        if artifact.state_bins < 2 {
            bail!("RL artifact state_bins must be at least 2");
        }
        if artifact.batch_size == 0
            || artifact.buffer_capacity == 0
            || artifact.epochs == 0
            || artifact.max_steps == 0
            || artifact.update_interval == 0
            || artifact.update_freq == 0
            || artifact.parallel_envs == 0
            || artifact.eval_episodes == 0
            || artifact.ray_tune_max_concurrency == 0
        {
            bail!("RL artifact contains zero-valued training parameters that must stay positive");
        }
        if !artifact.learning_rate.is_finite() || artifact.learning_rate <= 0.0 {
            bail!("RL artifact learning_rate must be finite and positive");
        }
        if !artifact.gamma.is_finite() || !(0.0..1.0).contains(&artifact.gamma) {
            bail!("RL artifact gamma must be finite and inside (0, 1)");
        }
        if !artifact.epsilon_start.is_finite() || !(0.0..=1.0).contains(&artifact.epsilon_start) {
            bail!("RL artifact epsilon_start must be finite and inside [0, 1]");
        }
        if !artifact.epsilon_end.is_finite() || !(0.0..=1.0).contains(&artifact.epsilon_end) {
            bail!("RL artifact epsilon_end must be finite and inside [0, 1]");
        }
        if artifact.epsilon_start < artifact.epsilon_end {
            bail!(
                "RL artifact epsilon_start {} must be >= epsilon_end {}",
                artifact.epsilon_start,
                artifact.epsilon_end
            );
        }
        if !artifact.epsilon_decay.is_finite() || !(0.0..=1.0).contains(&artifact.epsilon_decay) {
            bail!("RL artifact epsilon_decay must be finite and inside [0, 1]");
        }
        if !artifact.feature_columns.is_empty()
            && artifact.feature_columns.len() != artifact.state_dim
        {
            bail!(
                "RL artifact feature-column mismatch: expected {} columns, received {}",
                artifact.state_dim,
                artifact.feature_columns.len()
            );
        }
        if let Some(scaler) = artifact.feature_scaler.as_ref() {
            if scaler.means.len() != artifact.state_dim || scaler.stds.len() != artifact.state_dim {
                bail!(
                    "RL artifact feature_scaler mismatch: expected {} dimensions, received means {} / stds {}",
                    artifact.state_dim,
                    scaler.means.len(),
                    scaler.stds.len()
                );
            }
            if scaler
                .means
                .iter()
                .chain(scaler.stds.iter())
                .any(|value| !value.is_finite())
            {
                bail!("RL artifact feature_scaler contains non-finite values");
            }
            if scaler.stds.iter().any(|value| *value <= f32::EPSILON) {
                bail!("RL artifact feature_scaler contains non-positive standard deviations");
            }
        }
        if artifact.fallback_weights.is_some() != artifact.fallback_bias.is_some() {
            bail!("RL artifact fallback weights and bias must be persisted together");
        }
        for (field, value) in [
            ("backend", artifact.backend.as_str()),
            ("device_policy", artifact.device_policy.as_str()),
            (
                "requested_backend",
                artifact.requested_backend.as_deref().unwrap_or_default(),
            ),
            (
                "requested_device_policy",
                artifact
                    .requested_device_policy
                    .as_deref()
                    .unwrap_or_default(),
            ),
            (
                "effective_backend",
                artifact.effective_backend.as_deref().unwrap_or_default(),
            ),
            (
                "effective_device_policy",
                artifact
                    .effective_device_policy
                    .as_deref()
                    .unwrap_or_default(),
            ),
            (
                "network_precision",
                artifact.network_precision.as_deref().unwrap_or_default(),
            ),
        ] {
            if !value.is_empty() && value.trim().is_empty() {
                bail!("RL artifact {} may not be blank", field);
            }
        }
        if let Some(network_precision) = artifact_network_precision(artifact)? {
            let effective_backend = artifact_effective_backend(artifact);
            if effective_backend.starts_with("linear_q_") || effective_backend.starts_with("quadratic_q_") {
                bail!(
                    "RL artifact network_precision {} requires a neural runtime backend, got {}",
                    network_precision,
                    effective_backend
                );
            }
        }
        for (field, value) in [
            ("backend", artifact.backend.as_str()),
            (
                "requested_backend",
                artifact.requested_backend.as_deref().unwrap_or_default(),
            ),
        ] {
            if !value.is_empty() && !is_known_rl_requested_backend(value) {
                bail!(
                    "RL artifact {} `{}` is not a supported backend",
                    field,
                    value
                );
            }
        }
        for (field, value) in [(
            "effective_backend",
            artifact.effective_backend.as_deref().unwrap_or_default(),
        )] {
            if !value.is_empty() && !is_known_rl_effective_backend(value) {
                bail!(
                    "RL artifact {} `{}` is not a supported runtime backend",
                    field,
                    value
                );
            }
        }
        for (field, value) in [
            ("device_policy", artifact.device_policy.as_str()),
            (
                "requested_device_policy",
                artifact
                    .requested_device_policy
                    .as_deref()
                    .unwrap_or_default(),
            ),
        ] {
            if !value.is_empty() && !is_known_rl_requested_device_policy(value) {
                bail!(
                    "RL artifact {} `{}` is not a supported device policy",
                    field,
                    value
                );
            }
        }
        for (field, value) in [(
            "effective_device_policy",
            artifact
                .effective_device_policy
                .as_deref()
                .unwrap_or_default(),
        )] {
            if !value.is_empty() && !is_known_rl_effective_device_policy(value) {
                bail!(
                    "RL artifact {} `{}` is not a supported effective device policy",
                    field,
                    value
                );
            }
        }
        let runtime_identity_count = [
            artifact.requested_backend.as_ref(),
            artifact.requested_device_policy.as_ref(),
            artifact.effective_backend.as_ref(),
            artifact.effective_device_policy.as_ref(),
        ]
        .iter()
        .filter(|value| value.is_some())
        .count();
        if runtime_identity_count != 4 {
            bail!(
                "RL artifact must persist requested/effective backend and device policy together"
            );
        }
        if let Some(effective_backend) = artifact.effective_backend.as_ref() {
            if artifact.backend != *effective_backend {
                bail!(
                    "RL artifact legacy backend {} conflicts with effective_backend {}",
                    artifact.backend,
                    effective_backend
                );
            }
        }
        if let Some(effective_device_policy) = artifact.effective_device_policy.as_ref() {
            let normalized_legacy = normalize_rl_device_policy(&artifact.device_policy);
            let normalized_effective = normalize_rl_device_policy(effective_device_policy);
            if normalized_legacy != normalized_effective {
                bail!(
                    "RL artifact legacy device_policy {} conflicts with effective_device_policy {}",
                    artifact.device_policy,
                    effective_device_policy
                );
            }
        }
        let effective_backend = artifact_effective_backend(artifact);
        let effective_device_policy = artifact_effective_device_policy(artifact);
        if effective_backend == "rlkit_cpu"
            && normalize_rl_device_policy(&effective_device_policy) != "cpu"
        {
            bail!(
                "RL effective backend {} requires cpu device policy, got {}",
                effective_backend,
                effective_device_policy
            );
        }
        if effective_backend == "rlkit_cuda"
            && !normalize_rl_device_policy(&effective_device_policy).starts_with("cuda:")
        {
            bail!(
                "RL effective backend {} requires cuda:<ordinal> device policy, got {}",
                effective_backend,
                effective_device_policy
            );
        }
        if (effective_backend.starts_with("linear_q_")
            || effective_backend.starts_with("quadratic_q_"))
            && normalize_rl_device_policy(&effective_device_policy) != "cpu"
        {
            bail!(
                "RL fallback backend {} requires cpu device policy, got {}",
                effective_backend,
                effective_device_policy
            );
        }
        if artifact.state_mins.len() != artifact.state_dim {
            bail!(
                "RL artifact state_mins mismatch: expected {}, received {}",
                artifact.state_dim,
                artifact.state_mins.len()
            );
        }
        if artifact.state_maxs.len() != artifact.state_dim {
            bail!(
                "RL artifact state_maxs mismatch: expected {}, received {}",
                artifact.state_dim,
                artifact.state_maxs.len()
            );
        }
        if let Some(weights) = artifact.fallback_weights.as_ref() {
            let fallback_state_dim = artifact.fallback_basis.expanded_dim(artifact.state_dim);
            if weights.nrows() != 3 || weights.ncols() != fallback_state_dim {
                bail!(
                    "RL fallback weights mismatch: expected 3x{}, received {:?}",
                    fallback_state_dim,
                    weights.dim()
                );
            }
            if weights.iter().any(|value| !value.is_finite()) {
                bail!("RL fallback weights contain non-finite values");
            }
        }
        if let Some(bias) = artifact.fallback_bias.as_ref() {
            if bias.len() != 3 {
                bail!(
                    "RL fallback bias mismatch: expected 3 entries, received {}",
                    bias.len()
                );
            }
            if bias.iter().any(|value| !value.is_finite()) {
                bail!("RL fallback bias contains non-finite values");
            }
        }
        let report = artifact
            .training_report
            .as_ref()
            .context("RL artifact is missing training_report")?;
        if report.train_rows != artifact.train_rows {
            bail!(
                "RL training report rows {} do not match artifact train_rows {}",
                report.train_rows,
                artifact.train_rows
            );
        }
        if report.state_dim != artifact.state_dim {
            bail!(
                "RL training report state_dim {} does not match artifact state_dim {}",
                report.state_dim,
                artifact.state_dim
            );
        }
        if report.reward_horizon != artifact.reward_horizon {
            bail!(
                "RL training report reward_horizon {} does not match artifact reward_horizon {}",
                report.reward_horizon,
                artifact.reward_horizon
            );
        }
        if report.episode_len != artifact.episode_len {
            bail!(
                "RL training report episode_len {} does not match artifact episode_len {}",
                report.episode_len,
                artifact.episode_len
            );
        }
        if report.backend.trim().is_empty() || report.device_policy.trim().is_empty() {
            bail!("RL training report must persist non-empty backend and device_policy");
        }
        for value in [
            report.average_hold_reward,
            report.average_buy_reward,
            report.average_sell_reward,
        ] {
            if !value.is_finite() {
                bail!("RL training report contains non-finite reward statistics");
            }
        }
        if report.backend.trim().is_empty() {
            bail!("RL training report backend may not be blank");
        }
        if report.device_policy.trim().is_empty() {
            bail!("RL training report device_policy may not be blank");
        }
        if report.backend != artifact_effective_backend(artifact) {
            bail!(
                "RL training report backend {} does not match effective artifact backend {}",
                report.backend,
                artifact_effective_backend(artifact)
            );
        }
        if report.device_policy != artifact_effective_device_policy(artifact) {
            bail!(
                "RL training report device_policy {} does not match effective artifact device policy {}",
                report.device_policy,
                artifact_effective_device_policy(artifact)
            );
        }
        if report.used_fallback_q
            && (artifact.fallback_weights.is_none() || artifact.fallback_bias.is_none())
        {
            bail!(
                "RL training report marks fallback Q as used but artifact does not contain fallback parameters"
            );
        }
        if !report.used_fallback_q
            && (artifact_effective_backend(artifact).starts_with("linear_q")
                || artifact_effective_backend(artifact).starts_with("quadratic_q"))
        {
            bail!(
                "RL training report marks fallback Q as unused but effective artifact backend {} is a fallback backend",
                artifact_effective_backend(artifact)
            );
        }
        if report.used_network_snapshot
            && artifact_effective_backend(artifact).starts_with("linear_q")
            || report.used_network_snapshot
                && artifact_effective_backend(artifact).starts_with("quadratic_q")
        {
            bail!(
                "RL training report marks network snapshot as used but effective artifact backend {} is a fallback backend",
                artifact_effective_backend(artifact)
            );
        }
        if report.used_feature_scaler != artifact.feature_scaler.is_some() {
            bail!(
                "RL training report feature_scaler flag {} does not match artifact scaler presence {}",
                report.used_feature_scaler,
                artifact.feature_scaler.is_some()
            );
        }
        if !report.used_network_snapshot && artifact.network_precision.is_some() {
            bail!(
                "RL artifact persists network_precision without a persisted network snapshot"
            );
        }
        Ok(())
    }

    fn live_runtime_identity(&self) -> (String, String, bool, bool) {
        let used_network_snapshot = self.inference_network.is_some();
        let used_fallback_q = self.fallback_weights.is_some() && self.fallback_bias.is_some();
        let persisted_effective_backend = self
            .runtime_effective_backend
            .clone()
            .or_else(|| self.train_args.effective_backend.clone())
            .unwrap_or_else(|| self.train_args.backend.clone());
        let persisted_effective_device_policy = self
            .runtime_effective_device_policy
            .clone()
            .or_else(|| self.train_args.effective_device_policy.clone())
            .unwrap_or_else(|| self.train_args.device_policy.clone());
        if used_network_snapshot {
            (
                persisted_effective_backend,
                persisted_effective_device_policy,
                true,
                used_fallback_q,
            )
        } else if used_fallback_q {
            (
                fallback_backend_name(self.train_args.fallback_basis).to_string(),
                "cpu".to_string(),
                false,
                true,
            )
        } else {
            (
                persisted_effective_backend,
                persisted_effective_device_policy,
                false,
                false,
            )
        }
    }

    fn ensure_runtime_state_ready(&self) -> Result<()> {
        let artifact = self.artifact()?;
        Self::validate_artifact(&artifact)?;
        if artifact.feature_columns.is_empty() {
            bail!("RL runtime feature schema is unavailable");
        }
        if self.bounds.is_none() {
            bail!("RL runtime feature bounds are unavailable; load or train the learner first");
        }
        if self.inference_network.is_none()
            && (self.fallback_weights.is_none() || self.fallback_bias.is_none())
        {
            bail!("RL runtime policy is unavailable");
        }
        Ok(())
    }

    fn bounds_from_artifact(artifact: &TradingRlArtifact) -> Result<FeatureBounds> {
        Self::validate_artifact(artifact)?;
        let mut bounds = FeatureBounds {
            mins: artifact.state_mins.clone(),
            maxs: artifact.state_maxs.clone(),
        };

        for idx in 0..artifact.state_dim {
            let min = bounds.mins[idx];
            let max = bounds.maxs[idx];
            if !min.is_finite() || !max.is_finite() {
                bail!("RL artifact contains non-finite feature bounds");
            }
            if (max - min).abs() < 1e-6 {
                bounds.maxs[idx] = min + 1.0;
            }
        }

        Ok(bounds)
    }

    fn preprocess_runtime_state(&self, state: &[f32]) -> Result<Vec<f32>> {
        if let Some(scaler) = self.feature_scaler.as_ref() {
            if state.len() != scaler.means.len() || state.len() != scaler.stds.len() {
                bail!(
                    "RL runtime scaler dimension mismatch: expected {}, got {}",
                    scaler.means.len(),
                    state.len()
                );
            }
            let features = Array2::from_shape_vec((1, state.len()), state.to_vec())
                .context("shape RL runtime state for scaling")?;
            let scaled = scaler.transform(&features)?;
            Ok(scaled.row(0).iter().copied().collect())
        } else {
            Ok(state.to_vec())
        }
    }

    #[cfg(feature = "reinforcement-learning")]
    fn discretize_state(&self, state: &[f32]) -> Result<Status<u16>> {
        let bounds = self.bounds.as_ref().context("RL feature bounds missing")?;
        let values = bounds.discretize(state, self.state_bins)?;
        Ok(Status::new(
            values,
            vec![self.state_bins.max(2); self.train_args.state_dim],
        ))
    }

    #[cfg(feature = "reinforcement-learning")]
    pub fn predict_q_values(&self, state: &[f32]) -> Result<Vec<f32>> {
        self.ensure_runtime_state_ready()?;
        if let Some(network) = self.inference_network.as_ref() {
            let scaled = self.preprocess_runtime_state(state)?;
            let status = self.discretize_state(&scaled)?;
            let tensor = match self.state_encoding {
                TradingStateEncoding::OneHot => status
                    .to_one_hot_flat(
                        &vec![self.state_bins.max(2); self.train_args.state_dim],
                        &network.device(),
                    )
                    .map_err(|err| anyhow::anyhow!("encode RL state as one-hot: {err}"))?,
                TradingStateEncoding::Naive => status
                    .to_tensor(&network.device())
                    .map_err(|err| anyhow::anyhow!("encode RL state: {err}"))?,
                TradingStateEncoding::Normalized => status
                    .to_tensor_normalized(
                        &vec![self.state_bins.max(2); self.train_args.state_dim],
                        &network.device(),
                    )
                    .map_err(|err| anyhow::anyhow!("encode normalized RL state: {err}"))?,
            }
            .unsqueeze(0)
            .map_err(|err| anyhow::anyhow!("prepare RL state batch: {err}"))?;
            let tensor = if tensor.dtype() == network.dtype() {
                tensor
            } else {
                tensor
                    .to_dtype(network.dtype())
                    .map_err(|err| anyhow::anyhow!("cast RL state batch dtype: {err}"))?
            };

            network
                .forward(&tensor)
                .map_err(|err| anyhow::anyhow!("forward RL policy network: {err}"))?
                .squeeze(0)
                .map_err(|err| anyhow::anyhow!("squeeze RL policy output: {err}"))?
                .to_vec1::<f32>()
                .map_err(|err| anyhow::anyhow!("collect RL Q-values: {err}"))
                .and_then(validate_q_values)
        } else {
            let weights = self
                .fallback_weights
                .as_ref()
                .context("RL fallback weights missing")?;
            let bias = self
                .fallback_bias
                .as_ref()
                .context("RL fallback bias missing")?;
            let bounds = self.bounds.as_ref().context("RL feature bounds missing")?;
            let scaled = self.preprocess_runtime_state(state)?;
            let normalized = bounds.normalize(&scaled)?;
            let fallback_state = expand_fallback_basis(&normalized, self.train_args.fallback_basis);
            validate_q_values(
                (0..3)
                    .map(|action| {
                        weights
                            .row(action)
                            .iter()
                            .zip(fallback_state.iter())
                            .map(|(weight, value)| weight * value)
                            .sum::<f32>()
                            + bias[action]
                    })
                    .collect(),
            )
        }
    }

    #[cfg(not(feature = "reinforcement-learning"))]
    pub fn predict_q_values(&self, state: &[f32]) -> Result<Vec<f32>> {
        self.ensure_runtime_state_ready()?;
        let weights = self
            .fallback_weights
            .as_ref()
            .context("RL fallback weights missing")?;
        let bias = self
            .fallback_bias
            .as_ref()
            .context("RL fallback bias missing")?;
        let bounds = self.bounds.as_ref().context("RL feature bounds missing")?;
        let scaled = self.preprocess_runtime_state(state)?;
        let normalized = bounds.normalize(&scaled)?;
        let fallback_state = expand_fallback_basis(&normalized, self.train_args.fallback_basis);
        validate_q_values(
            (0..3)
                .map(|action| {
                    weights
                        .row(action)
                        .iter()
                        .zip(fallback_state.iter())
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>()
                        + bias[action]
                })
                .collect(),
        )
    }

    pub fn select_action(&self, state: &[f32]) -> Result<TradingAction> {
        let q_values = self.predict_q_values(state)?;
        let (best_idx, _) = q_values
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .context("RL policy returned no Q-values")?;
        TradingAction::from_index(best_idx)
    }

    fn runtime_backend_details(&self) -> (Option<String>, Option<String>) {
        let (effective_backend, effective_device_policy, used_network_snapshot, used_fallback_q) =
            self.live_runtime_identity();
        let mut reasons = Vec::new();
        let requested_precision = requested_training_precision_policy("dqn");
        let (effective_precision, precision_degraded_reason) =
            resolve_rl_training_precision_with_capability(
                Some(&requested_precision),
                &effective_backend,
                &effective_device_policy,
                probe_runtime_rl_bf16_support(&effective_backend, &effective_device_policy),
            );
        if let Some(precision_degraded_reason) = precision_degraded_reason {
            reasons.push(precision_degraded_reason);
        }
        #[cfg(feature = "reinforcement-learning")]
        if used_network_snapshot {
            let runtime_network_precision = self
                .inference_network
                .as_ref()
                .map(|network| dtype_to_rl_network_precision(network.dtype()))
                .transpose()
                .ok()
                .flatten();
            if let Some(runtime_network_precision) = runtime_network_precision {
                if runtime_network_precision != effective_precision {
                    reasons.push(format!(
                        "rl_runtime_network_precision_drift({runtime_network_precision}!={effective_precision})"
                    ));
                }
                if let Some(persisted_network_precision) = self
                    .train_args
                    .network_precision
                    .as_deref()
                    .map(normalize_rl_network_precision)
                    .transpose()
                    .ok()
                    .flatten()
                {
                    if persisted_network_precision != runtime_network_precision {
                        reasons.push("rl_persisted_network_precision_drift".to_string());
                    }
                }
            } else {
                reasons.push("rl_runtime_network_precision_unknown".to_string());
            }
        }
        if let Some(persisted_backend) = self.train_args.effective_backend.as_ref() {
            if persisted_backend != &effective_backend {
                reasons.push("rl_persisted_runtime_backend_drift".to_string());
            }
        }
        if let Some(persisted_policy) = self.train_args.effective_device_policy.as_ref() {
            if persisted_policy != &effective_device_policy {
                reasons.push("rl_persisted_runtime_device_drift".to_string());
            }
        }
        if let Some(gap_reason) = self
            .train_args
            .requested_device_policy
            .as_ref()
            .filter(|policy| requested_gpu_device_policy(policy))
            .and_then(|_| {
                (!effective_device_policy
                    .trim()
                    .to_ascii_lowercase()
                    .starts_with("cuda"))
                .then(|| "requested_rl_device_unavailable".to_string())
            })
            .or_else(|| {
                self.train_args
                    .requested_backend
                    .as_ref()
                    .filter(|backend| !backend.trim().is_empty())
                    .and_then(|backend| {
                        (!backend.eq_ignore_ascii_case(&effective_backend))
                            .then(|| "requested_rl_backend_unavailable".to_string())
                    })
            })
        {
            reasons.push(gap_reason);
        }
        if self.persisted_network_snapshot_present && !used_network_snapshot {
            reasons.push("persisted_rl_network_snapshot_unavailable".to_string());
        }
        if self.train_args.train_rows > 0 && self.training_report.is_none() {
            reasons.push("rl_training_report_missing".to_string());
        }
        if self.train_args.train_rows > 0
            && self.training_report.is_some()
            && self.train_args.training_report.is_none()
        {
            reasons.push("rl_persisted_training_report_missing".to_string());
        }
        if let (Some(persisted_report), Some(runtime_report)) = (
            self.train_args.training_report.as_ref(),
            self.training_report.as_ref(),
        ) {
            if persisted_report != runtime_report {
                reasons.push("rl_training_report_state_drift".to_string());
            }
        }
        if let Some(report) = self.training_report.as_ref() {
            if report.backend != effective_backend {
                reasons.push("rl_training_report_backend_drift".to_string());
            }
            if report.device_policy != effective_device_policy {
                reasons.push("rl_training_report_device_drift".to_string());
            }
            if report.used_network_snapshot != used_network_snapshot {
                reasons.push("rl_training_report_network_drift".to_string());
            }
            if report.used_fallback_q != used_fallback_q {
                reasons.push("rl_training_report_fallback_drift".to_string());
            }
            if report.used_feature_scaler != self.feature_scaler.is_some() {
                reasons.push("rl_training_report_scaler_drift".to_string());
            }
            if report.used_feature_scaler && self.feature_scaler.is_none() {
                reasons.push("rl_feature_scaler_missing".to_string());
            }
        }
        if used_network_snapshot {
            if effective_backend.starts_with("linear_q")
                || effective_backend.starts_with("quadratic_q")
            {
                reasons.push("rl_runtime_identity_inconsistent".to_string());
            }
            (
                Some(effective_backend.clone()),
                (!reasons.is_empty()).then(|| reasons.join("; ")),
            )
        } else if used_fallback_q {
            reasons.push("rl_network_unavailable".to_string());
            if !self
                .train_args
                .requested_backend
                .as_deref()
                .unwrap_or_default()
                .is_empty()
            {
                reasons.push("rl_backend_degraded_to_fallback_q".to_string());
            }
            (Some(effective_backend), Some(reasons.join("; ")))
        } else {
            (
                Some("rl_unknown".to_string()),
                Some(if reasons.is_empty() {
                    "rl_policy_unavailable".to_string()
                } else {
                    format!("{}; rl_policy_unavailable", reasons.join("; "))
                }),
            )
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_state_ready()?;
        let (features, columns) = feature_matrix_from_dataframe(x)?;
        if !self.feature_columns.is_empty() && self.feature_columns != columns {
            bail!(
                "RL runtime feature-column mismatch: expected {:?}, got {:?}",
                self.feature_columns,
                columns
            );
        }

        let (execution_backend, degraded_reason) = self.runtime_backend_details();
        let mut predictions = Vec::with_capacity(features.nrows());
        for row in features.outer_iter() {
            let state = row.iter().copied().collect::<Vec<_>>();
            let q_values = self.predict_q_values(&state)?;
            let probabilities = softmax_q_values(&q_values)?;
            let (confidence, abstain) = three_class_runtime_confidence(probabilities)?;
            predictions.push(build_runtime_prediction_with_details(
                "dqn",
                ModelFamily::Rl,
                CapabilityState::Implemented,
                probabilities,
                Some(confidence),
                Some(abstain),
                execution_backend.clone(),
                degraded_reason.clone(),
            )?);
        }

        Ok(predictions)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        #[cfg(not(feature = "reinforcement-learning"))]
        {
            self.ensure_runtime_state_ready()?;
            std::fs::create_dir_all(path)
                .with_context(|| format!("create RL model directory {}", path.display()))?;
            if self.fallback_weights.is_none() || self.fallback_bias.is_none() {
                bail!("RL model has neither a trained network nor fallback Q-parameters");
            }
            let mut artifact = self.artifact()?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            let runtime_metadata =
                rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
            validate_rl_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            let metadata_path = path.join(METADATA_FILE_NAME);
            let config_path = path.join("rl_config.json");
            let network_path = path.join("q_network.safetensors");
            let metadata_tmp = staged_rl_file(path, METADATA_FILE_NAME);
            let config_tmp = staged_rl_file(path, "rl_config.json");
            write_json(&metadata_tmp, &runtime_metadata)?;
            if let Err(err) = write_json(&config_tmp, &artifact) {
                cleanup_rl_temp_file(&metadata_tmp);
                cleanup_rl_temp_file(&config_tmp);
                return Err(err).with_context(|| format!("write RL config to {}", path.display()));
            }
            let metadata_backup = backup_rl_file(path, METADATA_FILE_NAME);
            let config_backup = backup_rl_file(path, "rl_config.json");
            let network_backup = backup_rl_file(path, "q_network.safetensors");
            if let Err(err) = stage_rl_target(&metadata_path, &metadata_backup, Some(&metadata_tmp))
            {
                cleanup_rl_temp_file(&config_tmp);
                return Err(err);
            }
            if let Err(err) = stage_rl_target(&config_path, &config_backup, Some(&config_tmp)) {
                rollback_rl_target(&metadata_path, &metadata_backup);
                return Err(err);
            }
            if let Err(err) = stage_rl_target(&network_path, &network_backup, None) {
                rollback_rl_target(&config_path, &config_backup);
                rollback_rl_target(&metadata_path, &metadata_backup);
                return Err(err);
            }
            Ok(())
        }

        #[cfg(feature = "reinforcement-learning")]
        {
            self.ensure_runtime_state_ready()?;
            std::fs::create_dir_all(path)
                .with_context(|| format!("create RL model directory {}", path.display()))?;
            let has_fallback = self.fallback_weights.is_some() && self.fallback_bias.is_some();
            let network_path = path.join("q_network.safetensors");
            let metadata_path = path.join(METADATA_FILE_NAME);
            let config_path = path.join("rl_config.json");
            let metadata_tmp = staged_rl_file(path, METADATA_FILE_NAME);
            let config_tmp = staged_rl_file(path, "rl_config.json");
            let network_tmp = staged_rl_file(path, "q_network.safetensors");
            if let Some(network) = self.inference_network.as_ref() {
                if let Err(err) = network.save(network_tmp.to_string_lossy().as_ref()) {
                    cleanup_rl_temp_file(&metadata_tmp);
                    cleanup_rl_temp_file(&config_tmp);
                    cleanup_rl_temp_file(&network_tmp);
                    return Err(anyhow::anyhow!("save RL network: {err}"));
                }
            } else {
                if !has_fallback {
                    bail!("RL model has neither a trained network nor fallback Q-parameters");
                }
                cleanup_rl_temp_file(&network_tmp);
            }
            let mut artifact = self.artifact()?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            let runtime_metadata =
                rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
            validate_rl_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            write_json(&metadata_tmp, &runtime_metadata)?;
            if let Err(err) = write_json(&config_tmp, &artifact) {
                cleanup_rl_temp_file(&metadata_tmp);
                cleanup_rl_temp_file(&config_tmp);
                cleanup_rl_temp_file(&network_tmp);
                return Err(err).with_context(|| format!("write RL config to {}", path.display()));
            }
            let metadata_backup = backup_rl_file(path, METADATA_FILE_NAME);
            let config_backup = backup_rl_file(path, "rl_config.json");
            let network_backup = backup_rl_file(path, "q_network.safetensors");
            if let Err(err) = stage_rl_target(&metadata_path, &metadata_backup, Some(&metadata_tmp))
            {
                cleanup_rl_temp_file(&config_tmp);
                cleanup_rl_temp_file(&network_tmp);
                return Err(err);
            }
            if let Err(err) = stage_rl_target(&config_path, &config_backup, Some(&config_tmp)) {
                cleanup_rl_temp_file(&network_tmp);
                rollback_rl_target(&metadata_path, &metadata_backup);
                return Err(err);
            }
            let staged_network = self
                .inference_network
                .as_ref()
                .map(|_| network_tmp.as_path());
            if let Err(err) = stage_rl_target(&network_path, &network_backup, staged_network) {
                rollback_rl_target(&config_path, &config_backup);
                rollback_rl_target(&metadata_path, &metadata_backup);
                return Err(err);
            }
            Ok(())
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        #[cfg(not(feature = "reinforcement-learning"))]
        {
            let mut artifact: TradingRlArtifact = read_json(&path.join("rl_config.json"))?;
            Self::validate_artifact(&artifact)?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            let metadata = resolve_rl_runtime_metadata(path, &artifact)?;
            validate_rl_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            let bounds = Self::bounds_from_artifact(&artifact)?;
            if artifact.fallback_weights.is_none() || artifact.fallback_bias.is_none() {
                bail!("RL artifact does not contain fallback Q-parameters");
            }
            let persisted_network_snapshot_present = path.join("q_network.safetensors").exists();
            let runtime_effective_backend =
                fallback_backend_name(artifact.fallback_basis).to_string();
            let runtime_effective_device_policy = "cpu".to_string();
            let training_report = artifact
                .training_report
                .clone()
                .context("RL artifact is missing training_report")?;
            Ok(Self {
                inference_network: None,
                hidden_dims: artifact.hidden_dims.clone(),
                state_encoding: artifact.state_encoding,
                state_bins: artifact.state_bins,
                buffer_capacity: artifact.buffer_capacity,
                bounds: Some(bounds),
                fallback_weights: artifact.fallback_weights.clone(),
                fallback_bias: artifact.fallback_bias.clone(),
                feature_scaler: artifact.feature_scaler.clone(),
                feature_columns: artifact.feature_columns.clone(),
                training_report: Some(training_report),
                runtime_effective_backend: Some(runtime_effective_backend),
                runtime_effective_device_policy: Some(runtime_effective_device_policy),
                persisted_network_snapshot_present,
                train_args: artifact,
            })
        }

        #[cfg(feature = "reinforcement-learning")]
        {
            let mut artifact: TradingRlArtifact = read_json(&path.join("rl_config.json"))?;
            Self::validate_artifact(&artifact)?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            let metadata = resolve_rl_runtime_metadata(path, &artifact)?;
            validate_rl_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            let bounds = Self::bounds_from_artifact(&artifact)?;
            let requested_device_policy = artifact_requested_device_policy(&artifact);
            let (device, effective_policy, effective_backend) =
                resolve_rl_inference_device(&requested_device_policy);
            let network_path = path.join("q_network.safetensors");
            let persisted_network_snapshot_present = network_path.exists();
            let network_precision = if persisted_network_snapshot_present {
                artifact_network_precision(&artifact)?.unwrap_or_else(|| "fp32".to_string())
            } else {
                "fp32".to_string()
            };
            let inference_network = if network_path.exists() {
                Some(
                    NeuralNetwork::load_with_dtype(
                        network_path.to_string_lossy().as_ref(),
                        artifact.state_dim,
                        &artifact.hidden_dims,
                        3,
                        match network_precision.as_str() {
                            "bf16" => DType::BF16,
                            _ => DType::F32,
                        },
                        &device,
                    )
                    .map_err(|err| anyhow::anyhow!("load RL network: {err}"))?,
                )
            } else {
                None
            };
            if let Some(network) = inference_network.as_ref() {
                artifact.network_precision = Some(dtype_to_rl_network_precision(network.dtype())?);
            } else {
                artifact.network_precision = None;
            }
            if inference_network.is_none()
                && (artifact.fallback_weights.is_none() || artifact.fallback_bias.is_none())
            {
                bail!("RL artifact does not contain a network snapshot or fallback Q-parameters");
            }
            let used_network_snapshot = inference_network.is_some();
            let runtime_effective_backend = if used_network_snapshot {
                effective_backend
            } else {
                fallback_backend_name(artifact.fallback_basis).to_string()
            };
            let runtime_effective_device_policy = if used_network_snapshot {
                effective_policy
            } else {
                "cpu".to_string()
            };
            let training_report = artifact
                .training_report
                .clone()
                .context("RL artifact is missing training_report")?;
            if training_report.used_network_snapshot && !persisted_network_snapshot_present {
                bail!(
                    "RL artifact training_report claims a network snapshot but q_network.safetensors is missing"
                );
            }

            Ok(Self {
                inference_network,
                hidden_dims: artifact.hidden_dims.clone(),
                state_encoding: artifact.state_encoding,
                state_bins: artifact.state_bins,
                buffer_capacity: artifact.buffer_capacity,
                bounds: Some(bounds),
                fallback_weights: artifact.fallback_weights.clone(),
                fallback_bias: artifact.fallback_bias.clone(),
                feature_scaler: artifact.feature_scaler.clone(),
                feature_columns: artifact.feature_columns.clone(),
                training_report: Some(training_report),
                runtime_effective_backend: Some(runtime_effective_backend),
                runtime_effective_device_policy: Some(runtime_effective_device_policy),
                persisted_network_snapshot_present,
                train_args: artifact,
            })
        }
    }
}

impl Default for TradingReinforcementLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use std::path::PathBuf;

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{stamp}-{}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp directory");
        path
    }

    #[test]
    fn rollback_rl_target_removes_partial_file_without_backup() {
        let path = unique_temp_dir("rl-rollback-target");
        let final_path = path.join("partial.json");
        let backup_path = path.join("partial.json.bak");
        std::fs::write(&final_path, b"partial").expect("write partial artifact");

        rollback_rl_target(&final_path, &backup_path);

        assert!(
            !final_path.exists(),
            "rollback should remove partial final artifact when no backup exists"
        );
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn fallback_only_artifact_round_trips_without_network_file() {
        let mut learner = TradingReinforcementLearner::new();
        learner.train_args.state_dim = 2;
        learner.train_args.train_rows = 16;
        learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        learner.train_args.hidden_dims = vec![4, 4];
        learner.train_args.state_mins = vec![0.0, 0.0];
        learner.train_args.state_maxs = vec![1.0, 1.0];
        learner.bounds = Some(FeatureBounds {
            mins: vec![0.0, 0.0],
            maxs: vec![1.0, 1.0],
        });
        learner.feature_columns = learner.train_args.feature_columns.clone();
        learner.train_args.backend = "linear_q_cpu".to_string();
        learner.train_args.device_policy = "cpu".to_string();
        learner.train_args.requested_backend = Some("linear_q_cpu".to_string());
        learner.train_args.requested_device_policy = Some("cpu".to_string());
        learner.train_args.effective_backend = Some("linear_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());

        let weights = Array2::<f32>::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5])
            .expect("shape fallback weights");
        let bias = Array1::<f32>::from_vec(vec![0.1_f32, 0.2, -0.1]);
        learner.train_args.fallback_weights = Some(weights.clone());
        learner.train_args.fallback_bias = Some(bias.clone());
        learner.train_args.training_report = Some(TradingRlTrainingReport {
            train_rows: 16,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        });
        learner.fallback_weights = Some(weights);
        learner.fallback_bias = Some(bias);
        learner.training_report = learner.train_args.training_report.clone();

        let path = unique_temp_dir("rl-fallback-only");
        learner
            .save(&path)
            .expect("save should succeed without a network file");

        assert!(
            path.join("rl_config.json").exists(),
            "artifact config should be written"
        );
        assert!(
            !path.join("q_network.safetensors").exists(),
            "fallback-only save should not create a network snapshot"
        );

        let loaded = TradingReinforcementLearner::load(&path)
            .expect("load should accept a fallback-only artifact");
        let q_values = loaded
            .predict_q_values(&[0.25_f32, 0.75_f32])
            .expect("fallback inference should work after load");

        assert_eq!(q_values.len(), 3);
        assert!((q_values[0] - 0.35).abs() < 1e-6);
        assert!((q_values[1] - 0.95).abs() < 1e-6);
        assert!((q_values[2] - 0.4).abs() < 1e-6);
        assert_eq!(
            loaded
                .training_report
                .as_ref()
                .expect("training report should round-trip")
                .backend,
            "linear_q_cpu"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[cfg(feature = "reinforcement-learning")]
    #[test]
    fn runtime_hints_are_normalized() {
        let learner =
            TradingReinforcementLearner::new().with_runtime_hints("RLKIT", "CUDA:0", 2, 4, 0, 1);

        assert_eq!(learner.train_args.backend, "rlkit");
        assert_eq!(learner.train_args.device_policy, "cuda:0");
    }

    #[cfg(feature = "reinforcement-learning")]
    #[test]
    fn auto_policy_uses_cpu_when_cuda_backend_is_unavailable() {
        let (device, effective_policy, effective_backend) =
            resolve_rl_training_device("auto").expect("auto policy should resolve");

        #[cfg(not(feature = "reinforcement-learning-cuda"))]
        {
            assert!(matches!(device, Device::Cpu));
            assert_eq!(effective_policy, "cpu");
            assert_eq!(effective_backend, "rlkit_cpu");
        }

        #[cfg(feature = "reinforcement-learning-cuda")]
        {
            let _ = device;
            assert!(
                matches!(effective_policy.as_str(), "cpu") || effective_policy.starts_with("cuda:")
            );
            assert!(matches!(
                effective_backend.as_str(),
                "rlkit_cpu" | "rlkit_cuda"
            ));
        }
    }

    #[cfg(all(
        feature = "reinforcement-learning",
        not(feature = "reinforcement-learning-cuda")
    ))]
    #[test]
    fn explicit_gpu_policy_falls_back_to_cpu_without_cuda_support() {
        let (device, policy, backend) =
            resolve_rl_training_device("rocm:0").expect("gpu policy should degrade to cpu");
        assert!(matches!(device, Device::Cpu));
        assert_eq!(policy, "cpu");
        assert_eq!(backend, "rlkit_cpu");
    }

    #[test]
    fn normalize_rl_device_policy_accepts_vendor_neutral_gpu_tokens() {
        assert_eq!(normalize_rl_device_policy("CUDA:1"), "gpu:1");
        assert_eq!(normalize_rl_device_policy("rocm:2"), "gpu:2");
        assert_eq!(normalize_rl_device_policy("metal:0"), "gpu:0");
        assert_eq!(normalize_rl_device_policy("vulkan:3"), "gpu:3");
        assert_eq!(normalize_rl_device_policy("nvidia"), "gpu");
    }

    #[test]
    fn validate_artifact_rejects_partial_fallback_parameters() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("linear_q_cpu".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("linear_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "linear_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.0,
                average_buy_reward: 0.0,
                average_sell_reward: 0.0,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Linear,
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: None,
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("partial fallback parameters should be rejected");
        assert!(
            err.to_string().contains("persisted together"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_q_values_rejects_non_finite_rows() {
        let err = validate_q_values(vec![0.1, f32::NAN, 0.2])
            .expect_err("non-finite q-values should be rejected");
        assert!(
            err.to_string().contains("non-finite"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn quadratic_fallback_basis_appends_squared_terms() {
        let expanded = expand_fallback_basis(&[0.5, -0.25], TradingFallbackBasis::Quadratic);
        assert_eq!(expanded, vec![0.5, -0.25, 0.25, 0.0625]);
    }

    #[test]
    fn runtime_backend_details_reflect_quadratic_fallback_basis() {
        let mut learner = TradingReinforcementLearner::new();
        learner.train_args.effective_backend = Some("linear_q_cpu".to_string());
        learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
        learner.fallback_weights = Some(Array2::zeros((3, 4)));
        learner.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = learner.runtime_backend_details();
        assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
        assert_eq!(degraded_reason.as_deref(), Some("rl_network_unavailable"));
    }

    #[test]
    fn runtime_backend_details_explain_requested_gpu_fallback_to_cpu() {
        let mut learner = TradingReinforcementLearner::new();
        learner.train_args.requested_backend = Some("rlkit".to_string());
        learner.train_args.requested_device_policy = Some("cuda:0".to_string());
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
        learner.fallback_weights = Some(Array2::zeros((3, 4)));
        learner.fallback_bias = Some(Array1::zeros(3));

        let (backend, degraded_reason) = learner.runtime_backend_details();
        assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
        let degraded_reason = degraded_reason.expect("fallback should be degraded");
        assert!(degraded_reason.contains("requested_rl_device_unavailable"));
        assert!(degraded_reason.contains("rl_network_unavailable"));
        assert!(degraded_reason.contains("rl_backend_degraded_to_fallback_q"));
    }

    #[test]
    fn runtime_backend_details_explain_requested_precision_when_unavailable() {
        std::env::set_var("FOREX_BOT_DQN_TRAIN_PRECISION", "bf16");
        let learner = TradingReinforcementLearner::new();
        let (_backend, degraded_reason) = learner.runtime_backend_details();
        std::env::remove_var("FOREX_BOT_DQN_TRAIN_PRECISION");

        let degraded_reason =
            degraded_reason.expect("precision request should appear in degraded reason");
        assert!(degraded_reason.contains("requested_rl_precision_unavailable(bf16)"));
    }

    #[test]
    fn rl_precision_resolution_uses_bf16_on_supported_cuda_runtime() {
        let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
            Some("bf16"),
            "rlkit_cuda",
            "cuda:0",
            Some(true),
        );

        assert_eq!(effective_precision, "bf16");
        assert!(degraded_reason.is_none());
    }

    #[test]
    fn rl_precision_resolution_uses_bf16_on_cpu_runtime() {
        let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
            Some("bf16"),
            "rlkit_cpu",
            "cpu",
            Some(true),
        );

        assert_eq!(effective_precision, "bf16");
        assert!(degraded_reason.is_none());
    }

    #[test]
    fn rl_precision_resolution_explains_cpu_backend_limit() {
        let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
            Some("bf16"),
            "quadratic_q_cpu",
            "cpu",
            None,
        );

        assert_eq!(effective_precision, "fp32");
        let degraded_reason = degraded_reason.expect("bf16 request should degrade");
        assert!(degraded_reason.contains("requested_rl_precision_unavailable(bf16)"));
        assert!(degraded_reason.contains("rl_backend_precision_limit(quadratic_q_cpu->fp32)"));
    }

    #[test]
    fn rl_precision_resolution_degrades_lower_precision_requests_to_bf16_when_available() {
        let (effective_precision, degraded_reason) = resolve_rl_training_precision_with_capability(
            Some("fp8"),
            "rlkit_cuda",
            "cuda:0",
            Some(true),
        );

        assert_eq!(effective_precision, "bf16");
        let degraded_reason = degraded_reason.expect("fp8 request should degrade");
        assert!(degraded_reason.contains("requested_rl_precision_unavailable(fp8)"));
        assert!(degraded_reason.contains("rl_precision_degraded_to_bf16"));
    }

    #[test]
    fn validate_artifact_rejects_network_precision_on_fallback_backend() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("linear_q_cpu".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("linear_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
            network_precision: Some("bf16".to_string()),
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "linear_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.0,
                average_buy_reward: 0.0,
                average_sell_reward: 0.0,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Linear,
            fallback_weights: Some(Array2::zeros((3, 2))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("fallback backend should reject neural network precision metadata");
        assert!(err.to_string().contains("network_precision"));
    }

    #[test]
    fn validate_artifact_rejects_training_report_claiming_missing_fallback() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("rlkit_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "rlkit_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: None,
            fallback_bias: None,
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("training report should not claim fallback without fallback parameters");
        assert!(err.to_string().contains("fallback"));
    }

    #[test]
    fn validate_artifact_rejects_training_report_underreporting_fallback_backend() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: false,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("fallback backend must not under-report fallback usage");
        assert!(err.to_string().contains("fallback Q as unused"));
    }

    #[test]
    fn validate_artifact_rejects_training_report_claiming_network_on_fallback_backend() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cuda:0".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "cuda:0".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: true,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("network-on-fallback report should be rejected");
        assert!(err.to_string().contains("network snapshot"));
    }

    #[test]
    fn validate_artifact_rejects_missing_training_report() {
        let mut artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("linear_q_cpu".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("linear_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "linear_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: None,
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };
        artifact.training_report = None;

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("missing training_report should be rejected");
        assert!(err.to_string().contains("missing training_report"));
    }

    #[test]
    fn validate_artifact_rejects_zero_training_parameters() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 0,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("auto".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "auto".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("zero epochs must be rejected");
        assert!(err.to_string().contains("zero-valued training parameters"));
    }

    #[test]
    fn runtime_backend_details_include_missing_training_report_for_trained_state() {
        let mut learner = TradingReinforcementLearner::default();
        learner.train_args.train_rows = 128;
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.fallback_weights = Some(Array2::zeros((3, 2)));
        learner.fallback_bias = Some(Array1::zeros(3));
        learner.training_report = None;

        let (backend, degraded_reason) = learner.runtime_backend_details();

        assert_eq!(backend.as_deref(), Some("quadratic_q_cpu"));
        let degraded_reason = degraded_reason.expect("trained fallback state should be degraded");
        assert!(degraded_reason.contains("rl_training_report_missing"));
        assert!(degraded_reason.contains("rl_network_unavailable"));
    }

    #[test]
    fn runtime_backend_details_flag_missing_persisted_training_report() {
        let mut learner = TradingReinforcementLearner::default();
        learner.train_args.train_rows = 128;
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.fallback_weights = Some(Array2::zeros((3, 2)));
        learner.fallback_bias = Some(Array1::zeros(3));
        learner.training_report = Some(TradingRlTrainingReport {
            train_rows: 128,
            episode_count: 4,
            state_dim: 2,
            reward_horizon: 4,
            episode_len: 16,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.1,
            average_buy_reward: 0.2,
            average_sell_reward: -0.1,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        });
        learner.train_args.training_report = None;

        let (_, degraded_reason) = learner.runtime_backend_details();
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("rl_persisted_training_report_missing")
        );
    }

    #[test]
    fn validate_artifact_rejects_partial_runtime_identity_fields() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 8,
            max_steps: 128,
            update_interval: 8,
            update_freq: 2,
            batch_size: 32,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: None,
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "auto".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("partial runtime identity should be rejected");
        assert!(err.to_string().contains("requested/effective backend"));
    }

    #[test]
    fn validate_artifact_rejects_unknown_effective_backend() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 8,
            max_steps: 128,
            update_interval: 8,
            update_freq: 2,
            batch_size: 32,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("auto".to_string()),
            effective_backend: Some("mystery_backend".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "auto".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "mystery_backend".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("unknown effective backend should be rejected");
        assert!(err.to_string().contains("supported runtime backend"));
    }

    #[test]
    fn validate_artifact_rejects_legacy_effective_runtime_drift() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 8,
            max_steps: 128,
            update_interval: 8,
            update_freq: 2,
            batch_size: 32,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cuda:0".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit_cuda".to_string(),
            device_policy: "cuda:0".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("legacy effective runtime drift should be rejected");
        assert!(err.to_string().contains("legacy backend"));
    }

    #[test]
    fn runtime_backend_details_include_training_report_backend_drift() {
        let mut learner = TradingReinforcementLearner::default();
        learner.train_args.train_rows = 128;
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.fallback_weights = Some(Array2::zeros((3, 2)));
        learner.fallback_bias = Some(Array1::zeros(3));
        learner.training_report = Some(TradingRlTrainingReport {
            backend: "rlkit_cuda".to_string(),
            device_policy: "cpu".to_string(),
            used_fallback_q: true,
            used_feature_scaler: false,
            ..TradingRlTrainingReport::default()
        });

        let (_, degraded_reason) = learner.runtime_backend_details();
        assert!(
            degraded_reason
                .as_deref()
                .unwrap_or_default()
                .contains("rl_training_report_backend_drift")
        );
    }

    #[test]
    fn runtime_backend_details_include_persisted_runtime_drift_and_missing_network_snapshot() {
        let mut learner = TradingReinforcementLearner::default();
        learner.train_args.train_rows = 128;
        learner.train_args.requested_backend = Some("rlkit".to_string());
        learner.train_args.requested_device_policy = Some("cuda:0".to_string());
        learner.train_args.effective_backend = Some("rlkit_cuda".to_string());
        learner.train_args.effective_device_policy = Some("cuda:0".to_string());
        learner.runtime_effective_backend = Some("quadratic_q_cpu".to_string());
        learner.runtime_effective_device_policy = Some("cpu".to_string());
        learner.persisted_network_snapshot_present = true;
        learner.fallback_weights = Some(Array2::zeros((3, 2)));
        learner.fallback_bias = Some(Array1::zeros(3));

        let (_, degraded_reason) = learner.runtime_backend_details();
        let degraded_reason = degraded_reason.expect("runtime should be degraded");
        assert!(degraded_reason.contains("rl_persisted_runtime_backend_drift"));
        assert!(degraded_reason.contains("rl_persisted_runtime_device_drift"));
        assert!(degraded_reason.contains("persisted_rl_network_snapshot_unavailable"));
    }

    #[test]
    fn validate_artifact_rejects_misaligned_feature_scaler() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 8,
            max_steps: 128,
            update_interval: 8,
            update_freq: 2,
            batch_size: 32,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: true,
            }),
            feature_scaler: Some(FeatureScaler {
                means: vec![0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("misaligned feature scaler should be rejected");
        assert!(err.to_string().contains("feature_scaler mismatch"));
    }

    #[test]
    fn validate_artifact_rejects_training_report_scaler_drift() {
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 8,
            max_steps: 128,
            update_interval: 8,
            update_freq: 2,
            batch_size: 32,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("quadratic_q_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 4,
            episode_len: 16,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 4,
                state_dim: 2,
                reward_horizon: 4,
                episode_len: 16,
                backend: "quadratic_q_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: Some(FeatureScaler {
                means: vec![0.0, 0.0],
                stds: vec![1.0, 1.0],
            }),
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        let err = TradingReinforcementLearner::validate_artifact(&artifact)
            .expect_err("training report scaler drift should be rejected");
        assert!(err.to_string().contains("feature_scaler flag"));
    }

    #[test]
    fn artifact_rejects_live_training_report_drift() {
        let mut learner = TradingReinforcementLearner::new();
        learner.train_args.state_dim = 2;
        learner.train_args.train_rows = 16;
        learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        learner.train_args.backend = "quadratic_q_cpu".to_string();
        learner.train_args.device_policy = "cpu".to_string();
        learner.train_args.requested_backend = Some("rlkit".to_string());
        learner.train_args.requested_device_policy = Some("cuda:0".to_string());
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
        learner.train_args.training_report = Some(TradingRlTrainingReport {
            train_rows: 16,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.0,
            average_buy_reward: 0.0,
            average_sell_reward: 0.0,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: true,
        });
        learner.bounds = Some(FeatureBounds {
            mins: vec![0.0, 0.0],
            maxs: vec![1.0, 1.0],
        });
        learner.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        learner.fallback_weights = Some(Array2::zeros((3, 4)));
        learner.fallback_bias = Some(Array1::zeros(3));
        learner.feature_scaler = Some(FeatureScaler {
            means: vec![0.25, 0.75],
            stds: vec![0.5, 0.25],
        });
        learner.training_report = Some(TradingRlTrainingReport {
            train_rows: 16,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "stale_backend".to_string(),
            device_policy: "stale_device".to_string(),
            average_hold_reward: 0.0,
            average_buy_reward: 0.0,
            average_sell_reward: 0.0,
            used_network_snapshot: true,
            used_fallback_q: false,
            used_feature_scaler: true,
        });

        let err = learner
            .artifact()
            .expect_err("live/persisted training report drift should be rejected");
        assert!(err.to_string().contains("training_report drifted"));
    }

    #[test]
    fn artifact_rejects_missing_persisted_training_report() {
        let mut learner = TradingReinforcementLearner::new();
        learner.train_args.state_dim = 2;
        learner.train_args.train_rows = 16;
        learner.train_args.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        learner.train_args.backend = "quadratic_q_cpu".to_string();
        learner.train_args.device_policy = "cpu".to_string();
        learner.train_args.requested_backend = Some("rlkit".to_string());
        learner.train_args.requested_device_policy = Some("cuda:0".to_string());
        learner.train_args.effective_backend = Some("quadratic_q_cpu".to_string());
        learner.train_args.effective_device_policy = Some("cpu".to_string());
        learner.train_args.fallback_basis = TradingFallbackBasis::Quadratic;
        learner.train_args.training_report = None;
        learner.bounds = Some(FeatureBounds {
            mins: vec![0.0, 0.0],
            maxs: vec![1.0, 1.0],
        });
        learner.feature_columns = vec!["f1".to_string(), "f2".to_string()];
        learner.fallback_weights = Some(Array2::zeros((3, 4)));
        learner.fallback_bias = Some(Array1::zeros(3));
        learner.training_report = Some(TradingRlTrainingReport {
            train_rows: 16,
            episode_count: 2,
            state_dim: 2,
            reward_horizon: 0,
            episode_len: 0,
            backend: "quadratic_q_cpu".to_string(),
            device_policy: "cpu".to_string(),
            average_hold_reward: 0.0,
            average_buy_reward: 0.0,
            average_sell_reward: 0.0,
            used_network_snapshot: false,
            used_fallback_q: true,
            used_feature_scaler: false,
        });

        let err = learner
            .artifact()
            .expect_err("missing persisted training report should be rejected");
        assert!(
            err.to_string()
                .contains("missing the persisted training_report")
        );
    }

    #[test]
    fn load_rejects_missing_network_snapshot_when_report_claims_one() {
        let path = unique_temp_dir("rl-missing-network-snapshot");
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("rlkit_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "rlkit_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: true,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };
        let metadata = rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)
            .expect("build metadata");
        write_json(&path.join(METADATA_FILE_NAME), &metadata).expect("write metadata");
        write_json(&path.join("rl_config.json"), &artifact).expect("write config");

        let err = TradingReinforcementLearner::load(&path)
            .expect_err("load should reject missing network snapshot");
        assert!(err.to_string().contains("claims a network snapshot"));

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn load_reconstructs_runtime_metadata_when_sidecar_missing() {
        let path = unique_temp_dir("rl-missing-metadata-sidecar");
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("rlkit_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "rlkit_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };

        write_json(&path.join("rl_config.json"), &artifact).expect("write config");

        let loaded = TradingReinforcementLearner::load(&path)
            .expect("load should reconstruct metadata from artifact fields");
        assert_eq!(loaded.feature_columns, artifact.feature_columns);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn load_rejects_metadata_sidecar_drift_against_reconstructed_runtime_metadata() {
        let path = unique_temp_dir("rl-sidecar-drift");
        let artifact = TradingRlArtifact {
            state_dim: 2,
            feature_columns: vec!["f1".to_string(), "f2".to_string()],
            train_rows: 32,
            hidden_dims: vec![4, 4],
            state_encoding: TradingStateEncoding::Normalized,
            state_bins: 255,
            state_mins: vec![0.0, 0.0],
            state_maxs: vec![1.0, 1.0],
            buffer_capacity: 50_000,
            epochs: 64,
            max_steps: 512,
            update_interval: 32,
            update_freq: 4,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay: 0.995,
            requested_backend: Some("rlkit".to_string()),
            requested_device_policy: Some("cpu".to_string()),
            effective_backend: Some("rlkit_cpu".to_string()),
            effective_device_policy: Some("cpu".to_string()),
                        network_precision: None,
            backend: "rlkit_cpu".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: Some(TradingRlTrainingReport {
                train_rows: 32,
                episode_count: 2,
                state_dim: 2,
                reward_horizon: 0,
                episode_len: 0,
                backend: "rlkit_cpu".to_string(),
                device_policy: "cpu".to_string(),
                average_hold_reward: 0.1,
                average_buy_reward: 0.2,
                average_sell_reward: -0.1,
                used_network_snapshot: false,
                used_fallback_q: true,
                used_feature_scaler: false,
            }),
            feature_scaler: None,
            fallback_basis: TradingFallbackBasis::Quadratic,
            fallback_weights: Some(Array2::zeros((3, 4))),
            fallback_bias: Some(Array1::zeros(3)),
        };
        let mut drifted_metadata =
            rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)
                .expect("build metadata");
        drifted_metadata.training_summary.train_rows = 31;
        drifted_metadata.training_summary.val_rows = 1;
        drifted_metadata.training_summary.dataset_rows = 33;

        write_json(&path.join("rl_config.json"), &artifact).expect("write config");
        write_json(&path.join(METADATA_FILE_NAME), &drifted_metadata).expect("write metadata");

        let err =
            TradingReinforcementLearner::load(&path).expect_err("drifted sidecar should fail load");
        assert!(err.to_string().contains("metadata sidecar mismatch"));

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn preprocess_runtime_state_applies_persisted_feature_scaler() {
        let mut learner = TradingReinforcementLearner::new();
        learner.feature_scaler = Some(FeatureScaler {
            means: vec![1.0, 2.0],
            stds: vec![2.0, 4.0],
        });

        let scaled = learner
            .preprocess_runtime_state(&[5.0, 10.0])
            .expect("runtime preprocessing should use persisted scaler");
        assert_eq!(scaled, vec![2.0, 2.0]);
    }

    #[test]
    fn predict_runtime_rejects_missing_bounds_before_inference() -> Result<()> {
        let learner = TradingReinforcementLearner::new();
        let df = DataFrame::new(vec![Series::new("f1".into(), vec![0.0_f64]).into()])?;

        let err = learner
            .predict_runtime(&df)
            .expect_err("missing bounds should fail early");
        assert!(err.to_string().contains("feature bounds"));
        Ok(())
    }
}
