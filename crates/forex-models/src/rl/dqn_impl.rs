use anyhow::{bail, Context, Result};
#[cfg(feature = "reinforcement-learning")]
use candle_core::{Device, Module};
use ndarray::Array2;
use polars::prelude::{DataFrame, Series};
#[cfg(feature = "reinforcement-learning")]
use rlkit::network::NeuralNetwork;
#[cfg(feature = "reinforcement-learning")]
use rlkit::policies::EpsilonGreedy;
#[cfg(feature = "reinforcement-learning")]
use rlkit::types::{Action, EnvTrait, Reward, Status};
#[cfg(feature = "reinforcement-learning")]
use rlkit::{Algorithm, DNQStateMode, TrainArgs, DQN};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction_with_details,
    canonical_three_class_label_mapping, three_class_runtime_confidence,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    feature_matrix_from_dataframe, read_json, remap_three_class_labels, write_json, FeatureScaler,
    METADATA_FILE_NAME,
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
            backend: "rlkit_cpu".to_string(),
            device_policy: "auto".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: None,
            fallback_basis: TradingFallbackBasis::Linear,
            fallback_weights: None,
            fallback_bias: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

fn rl_runtime_metadata(
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        "dqn",
        ModelFamily::Rl,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
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
        "auto".to_string()
    } else {
        normalized
    }
}

#[cfg(all(
    feature = "reinforcement-learning",
    feature = "reinforcement-learning-cuda"
))]
fn requested_cuda_ordinal(policy: &str) -> Option<usize> {
    normalize_rl_device_policy(policy)
        .strip_prefix("cuda:")
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(feature = "reinforcement-learning")]
fn resolve_rl_training_device(policy: &str) -> Result<(Device, String, String)> {
    let normalized = normalize_rl_device_policy(policy);
    let explicit_gpu =
        matches!(normalized.as_str(), "gpu" | "cuda" | "nvidia") || normalized.starts_with("cuda:");

    #[cfg(feature = "reinforcement-learning-cuda")]
    {
        let ordinal = requested_cuda_ordinal(&normalized).unwrap_or(0);
        match normalized.as_str() {
            "cpu" => return Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string())),
            "auto" => match Device::new_cuda(ordinal) {
                Ok(device) => {
                    return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()))
                }
                Err(_) => return Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string())),
            },
            "gpu" | "cuda" | "nvidia" => {
                let device = Device::new_cuda(ordinal)
                    .map_err(|err| anyhow::anyhow!("initialize RL CUDA device {ordinal}: {err}"))?;
                return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()));
            }
            value if value.starts_with("cuda:") => {
                let device = Device::new_cuda(ordinal)
                    .map_err(|err| anyhow::anyhow!("initialize RL CUDA device {ordinal}: {err}"))?;
                return Ok((device, format!("cuda:{ordinal}"), "rlkit_cuda".to_string()));
            }
            _ => {}
        }
    }

    if explicit_gpu {
        bail!(
            "RL device policy `{normalized}` requested CUDA, but this build does not include reinforcement-learning-cuda support"
        );
    }

    Ok((Device::Cpu, "cpu".to_string(), "rlkit_cpu".to_string()))
}

#[cfg(feature = "reinforcement-learning")]
fn resolve_rl_inference_device(policy: &str) -> (Device, String, String) {
    let _normalized = normalize_rl_device_policy(policy);

    #[cfg(feature = "reinforcement-learning-cuda")]
    {
        let ordinal = requested_cuda_ordinal(&normalized).unwrap_or(0);
        if matches!(normalized.as_str(), "auto" | "gpu" | "cuda" | "nvidia")
            || normalized.starts_with("cuda:")
        {
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
    feature_columns: Vec<String>,
    training_report: Option<TradingRlTrainingReport>,
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
            feature_columns: Vec::new(),
            training_report: None,
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
        self.train_args.backend = fallback_backend;
        self.train_args.device_policy = "cpu".to_string();
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
        self.train_args
            .requested_backend
            .get_or_insert_with(|| self.train_args.backend.clone());
        self.train_args
            .requested_device_policy
            .get_or_insert_with(|| self.train_args.device_policy.clone());
        let mut model = DQN::new(
            &env,
            self.buffer_capacity,
            &self.hidden_dims,
            self.state_encoding.as_rlkit(),
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
        self.train_args.backend = effective_backend;
        self.train_args.device_policy = effective_policy;
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
        self.train_args.train_rows = scaled.nrows();
        self.train_on_episodes(&episodes)
    }

    fn artifact(&self) -> Result<TradingRlArtifact> {
        if self.train_args.state_dim == 0 || self.bounds.is_none() {
            bail!("RL artifact is unavailable before the learner is trained");
        }
        Ok(self.train_args.clone())
    }

    fn validate_artifact(artifact: &TradingRlArtifact) -> Result<()> {
        if artifact.state_dim == 0 {
            bail!("RL artifact is missing the trained state dimension");
        }
        if artifact.train_rows == 0 {
            bail!("RL artifact is missing training-row metadata");
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
        ] {
            if !value.is_empty() && value.trim().is_empty() {
                bail!("RL artifact {} may not be blank", field);
            }
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
        if let Some(report) = artifact.training_report.as_ref() {
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
            for value in [
                report.average_hold_reward,
                report.average_buy_reward,
                report.average_sell_reward,
            ] {
                if !value.is_finite() {
                    bail!("RL training report contains non-finite reward statistics");
                }
            }
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
        if let Some(network) = self.inference_network.as_ref() {
            let status = self.discretize_state(state)?;
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
            let normalized = bounds.normalize(state)?;
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
        let weights = self
            .fallback_weights
            .as_ref()
            .context("RL fallback weights missing")?;
        let bias = self
            .fallback_bias
            .as_ref()
            .context("RL fallback bias missing")?;
        let bounds = self.bounds.as_ref().context("RL feature bounds missing")?;
        let normalized = bounds.normalize(state)?;
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
        let effective_backend = self
            .train_args
            .effective_backend
            .clone()
            .unwrap_or_else(|| self.train_args.backend.clone());
        if self.inference_network.is_some() {
            (
                Some(effective_backend.clone()),
                if effective_backend.starts_with("linear_q") {
                    Some("rl_network_unavailable".to_string())
                } else {
                    None
                },
            )
        } else if self.fallback_weights.is_some() && self.fallback_bias.is_some() {
            let backend = if effective_backend.trim().is_empty()
                || effective_backend.starts_with("linear_q")
                || effective_backend.starts_with("quadratic_q")
            {
                fallback_backend_name(self.train_args.fallback_basis).to_string()
            } else {
                effective_backend
            };
            let degraded_reason = if backend.ends_with("_q_cpu") {
                Some("rl_network_unavailable".to_string())
            } else {
                Some("rl_backend_degraded_to_fallback_q".to_string())
            };
            (Some(backend), degraded_reason)
        } else {
            (
                Some("rl_unknown".to_string()),
                Some("rl_policy_unavailable".to_string()),
            )
        }
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
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
                rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows);
            validate_rl_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
            write_json(&path.join("rl_config.json"), &artifact)
                .with_context(|| format!("write RL config to {}", path.display()))?;
            let network_path = path.join("q_network.safetensors");
            if network_path.exists() {
                std::fs::remove_file(&network_path).with_context(|| {
                    format!("remove stale RL network snapshot from {}", path.display())
                })?;
            }
            Ok(())
        }

        #[cfg(feature = "reinforcement-learning")]
        {
            std::fs::create_dir_all(path)
                .with_context(|| format!("create RL model directory {}", path.display()))?;
            let has_fallback = self.fallback_weights.is_some() && self.fallback_bias.is_some();
            let network_path = path.join("q_network.safetensors");
            if let Some(network) = self.inference_network.as_ref() {
                network
                    .save(network_path.to_string_lossy().as_ref())
                    .map_err(|err| anyhow::anyhow!("save RL network: {err}"))?;
            } else {
                if !has_fallback {
                    bail!("RL model has neither a trained network nor fallback Q-parameters");
                }
                if network_path.exists() {
                    std::fs::remove_file(&network_path).with_context(|| {
                        format!("remove stale RL network snapshot from {}", path.display())
                    })?;
                }
            }
            let mut artifact = self.artifact()?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            let runtime_metadata =
                rl_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows);
            validate_rl_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
            write_json(&path.join("rl_config.json"), &artifact)
                .with_context(|| format!("write RL config to {}", path.display()))?;
            Ok(())
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        #[cfg(not(feature = "reinforcement-learning"))]
        {
            let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
            let mut artifact: TradingRlArtifact = read_json(&path.join("rl_config.json"))?;
            Self::validate_artifact(&artifact)?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            artifact
                .requested_backend
                .get_or_insert_with(|| artifact.backend.clone());
            artifact
                .requested_device_policy
                .get_or_insert_with(|| artifact.device_policy.clone());
            artifact
                .effective_backend
                .get_or_insert_with(|| artifact.backend.clone());
            artifact
                .effective_device_policy
                .get_or_insert_with(|| artifact.device_policy.clone());
            validate_rl_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            let bounds = Self::bounds_from_artifact(&artifact)?;
            if artifact.fallback_weights.is_none() || artifact.fallback_bias.is_none() {
                bail!("RL artifact does not contain fallback Q-parameters");
            }
            let training_report =
                artifact
                    .training_report
                    .clone()
                    .unwrap_or_else(|| TradingRlTrainingReport {
                        train_rows: artifact.train_rows,
                        episode_count: 0,
                        state_dim: artifact.state_dim,
                        reward_horizon: artifact.reward_horizon,
                        episode_len: artifact.episode_len,
                        backend: artifact_effective_backend(&artifact),
                        device_policy: artifact_effective_device_policy(&artifact),
                        average_hold_reward: 0.0,
                        average_buy_reward: 0.0,
                        average_sell_reward: 0.0,
                        used_network_snapshot: false,
                        used_fallback_q: true,
                    });
            Ok(Self {
                inference_network: None,
                hidden_dims: artifact.hidden_dims.clone(),
                state_encoding: artifact.state_encoding,
                state_bins: artifact.state_bins,
                buffer_capacity: artifact.buffer_capacity,
                bounds: Some(bounds),
                fallback_weights: artifact.fallback_weights.clone(),
                fallback_bias: artifact.fallback_bias.clone(),
                feature_columns: artifact.feature_columns.clone(),
                training_report: Some(training_report),
                train_args: artifact,
            })
        }

        #[cfg(feature = "reinforcement-learning")]
        {
            let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
            let mut artifact: TradingRlArtifact = read_json(&path.join("rl_config.json"))?;
            Self::validate_artifact(&artifact)?;
            if artifact.feature_columns.is_empty() {
                artifact.feature_columns = default_rl_feature_columns(artifact.state_dim);
            }
            validate_rl_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            let bounds = Self::bounds_from_artifact(&artifact)?;
            let requested_backend = artifact_requested_backend(&artifact);
            let requested_device_policy = artifact_requested_device_policy(&artifact);
            let (device, effective_policy, effective_backend) =
                resolve_rl_inference_device(&requested_device_policy);
            let network_path = path.join("q_network.safetensors");
            let inference_network = if network_path.exists() {
                Some(
                    NeuralNetwork::load(
                        network_path.to_string_lossy().as_ref(),
                        artifact.state_dim,
                        &artifact.hidden_dims,
                        3,
                        &device,
                    )
                    .map_err(|err| anyhow::anyhow!("load RL network: {err}"))?,
                )
            } else {
                None
            };
            if inference_network.is_none()
                && (artifact.fallback_weights.is_none() || artifact.fallback_bias.is_none())
            {
                bail!("RL artifact does not contain a network snapshot or fallback Q-parameters");
            }
            artifact.requested_backend = Some(requested_backend.clone());
            artifact.requested_device_policy = Some(requested_device_policy.clone());
            artifact.effective_backend = Some(effective_backend.clone());
            artifact.effective_device_policy = Some(effective_policy.clone());
            artifact.backend = requested_backend;
            artifact.device_policy = requested_device_policy;
            let used_network_snapshot = inference_network.is_some();
            let training_report =
                artifact
                    .training_report
                    .clone()
                    .unwrap_or_else(|| TradingRlTrainingReport {
                        train_rows: artifact.train_rows,
                        episode_count: 0,
                        state_dim: artifact.state_dim,
                        reward_horizon: artifact.reward_horizon,
                        episode_len: artifact.episode_len,
                        backend: artifact_effective_backend(&artifact),
                        device_policy: artifact_effective_device_policy(&artifact),
                        average_hold_reward: 0.0,
                        average_buy_reward: 0.0,
                        average_sell_reward: 0.0,
                        used_network_snapshot,
                        used_fallback_q: artifact.fallback_weights.is_some()
                            && artifact.fallback_bias.is_some(),
                    });

            Ok(Self {
                inference_network,
                hidden_dims: artifact.hidden_dims.clone(),
                state_encoding: artifact.state_encoding,
                state_bins: artifact.state_bins,
                buffer_capacity: artifact.buffer_capacity,
                bounds: Some(bounds),
                fallback_weights: artifact.fallback_weights.clone(),
                fallback_bias: artifact.fallback_bias.clone(),
                feature_columns: artifact.feature_columns.clone(),
                training_report: Some(training_report),
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
    fn explicit_gpu_policy_fails_without_cuda_support() {
        let err =
            resolve_rl_training_device("cuda:0").expect_err("gpu policy should fail without cuda");
        let msg = format!("{err:#}");
        assert!(msg.contains("does not include reinforcement-learning-cuda support"));
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
            requested_backend: None,
            requested_device_policy: None,
            effective_backend: None,
            effective_device_policy: None,
            backend: "native".to_string(),
            device_policy: "cpu".to_string(),
            parallel_envs: 1,
            eval_episodes: 8,
            rllib_num_workers: 0,
            ray_tune_max_concurrency: 1,
            reward_horizon: 0,
            episode_len: 0,
            training_report: None,
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
        assert_eq!(
            degraded_reason.as_deref(),
            Some("rl_network_unavailable")
        );
    }
}
