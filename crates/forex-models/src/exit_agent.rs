// Exit Agent - Pure Rust Burn RL-Based Trade Exit Expert
// Ported from src/forex_bot/models/exit_agent.py
//
// Lightweight RL Network for Trade Exit Decisions.
// Learns to balance Greed (Holding for TP) vs Fear (Cutting Loss/Stall).

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::nn;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings, Recorder};

use polars::prelude::{DataFrame, DataType, Series};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::base::{
    build_runtime_artifact_metadata, build_runtime_prediction_with_details,
    canonical_three_class_label_mapping, dataframe_to_float32_array,
    feature_columns_from_dataframe, three_class_runtime_confidence,
};
use crate::burn_models::{resolve_train_device, TrainBackend};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{read_json, write_json, METADATA_FILE_NAME};
// ============================================================================
// BURN Q-NETWORK
// ============================================================================

#[derive(Module, Debug)]
pub struct ExitAgentNet<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct ExitAgentNetConfig {
    #[config(default = 6)]
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

impl ExitAgentNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ExitAgentNet<B> {
        ExitAgentNet {
            fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, 2).init(device),
        }
    }
}

impl<B: Backend> ExitAgentNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = burn::tensor::activation::relu(self.fc1.forward(x));
        let x = burn::tensor::activation::relu(self.fc2.forward(x));
        self.output.forward(x)
    }
}

// ============================================================================
// AGENT STATE & EXPERIENCE
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: i64,
    pub reward: f32,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingRegret {
    pub state: Vec<f32>,
    pub action: i64,
    pub exit_price: f64,
    pub time: i64,
    pub direction: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExitAgentArtifact {
    pub input_dim: usize,
    pub hidden_dim: usize,
    #[serde(default)]
    pub feature_columns: Vec<String>,
    pub gamma: f32,
    pub epsilon: f32,
    pub epsilon_min: f32,
    pub epsilon_decay: f32,
    pub memory_capacity: usize,
    pub reward_horizon: usize,
    pub warmup_steps: usize,
    pub train_rows: usize,
    pub trained_memory_size: usize,
    pub average_reward: f32,
    #[serde(default)]
    pub replay_memory: Vec<Experience>,
    #[serde(default)]
    pub pending_regret: HashMap<i32, PendingRegret>,
    #[serde(default)]
    pub training_report: Option<ExitAgentTrainingReport>,
    #[serde(default)]
    pub requested_device_policy: Option<String>,
    #[serde(default)]
    pub effective_device_policy: Option<String>,
    #[serde(default)]
    pub execution_backend: Option<String>,
}

impl Default for ExitAgentArtifact {
    fn default() -> Self {
        Self {
            input_dim: 6,
            hidden_dim: 64,
            feature_columns: Vec::new(),
            gamma: 0.99,
            epsilon: 0.2,
            epsilon_min: 0.05,
            epsilon_decay: 0.999,
            memory_capacity: 10_000,
            reward_horizon: 0,
            warmup_steps: 0,
            train_rows: 0,
            trained_memory_size: 0,
            average_reward: 0.0,
            replay_memory: Vec::new(),
            pending_regret: HashMap::new(),
            training_report: None,
            requested_device_policy: None,
            effective_device_policy: None,
            execution_backend: None,
        }
    }
}

fn exit_runtime_metadata(
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> RuntimeArtifactMetadata {
    build_runtime_artifact_metadata(
        "exit_agent",
        ModelFamily::Exit,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn validate_exit_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if metadata.model_name != "exit_agent" {
        anyhow::bail!(
            "exit-agent metadata model mismatch: expected exit_agent, got {}",
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Exit {
        anyhow::bail!(
            "exit-agent metadata family mismatch: expected {:?}, got {:?}",
            ModelFamily::Exit,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        anyhow::bail!(
            "exit-agent metadata state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        anyhow::bail!("exit-agent metadata label mapping mismatch");
    }
    if expected_feature_columns.is_empty() {
        anyhow::bail!("exit-agent metadata validation requires non-empty feature columns");
    }
    if metadata.feature_columns != expected_feature_columns {
        anyhow::bail!(
            "exit-agent metadata feature-column mismatch: expected {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        anyhow::bail!(
            "exit-agent metadata dataset-row mismatch: expected {}, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        anyhow::bail!(
            "exit-agent metadata rows are inconsistent: train_rows {} + val_rows {} != dataset_rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }
    Ok(())
}

fn validate_exit_artifact(artifact: &ExitAgentArtifact) -> Result<()> {
    if artifact.input_dim == 0 {
        anyhow::bail!("exit-agent artifact input_dim must be positive");
    }
    if artifact.hidden_dim == 0 {
        anyhow::bail!("exit-agent artifact hidden_dim must be positive");
    }
    if artifact.feature_columns.is_empty() {
        anyhow::bail!("exit-agent artifact must contain feature columns");
    }
    if artifact.feature_columns.len() != artifact.input_dim {
        anyhow::bail!(
            "exit-agent artifact feature-column mismatch: input_dim {} vs {} feature columns",
            artifact.input_dim,
            artifact.feature_columns.len()
        );
    }
    if artifact.train_rows == 0 {
        anyhow::bail!("exit-agent artifact must record at least one training row");
    }
    if !artifact.gamma.is_finite() || !(0.0..1.0).contains(&artifact.gamma) {
        anyhow::bail!("exit-agent artifact gamma must be finite and inside (0, 1)");
    }
    if !artifact.epsilon.is_finite() || !(0.0..=1.0).contains(&artifact.epsilon) {
        anyhow::bail!("exit-agent artifact epsilon must be finite and inside [0, 1]");
    }
    if !artifact.epsilon_min.is_finite() || !(0.0..=1.0).contains(&artifact.epsilon_min) {
        anyhow::bail!("exit-agent artifact epsilon_min must be finite and inside [0, 1]");
    }
    if artifact.epsilon < artifact.epsilon_min {
        anyhow::bail!(
            "exit-agent artifact epsilon {} must be >= epsilon_min {}",
            artifact.epsilon,
            artifact.epsilon_min
        );
    }
    if !artifact.epsilon_decay.is_finite() || !(0.90..=0.99999).contains(&artifact.epsilon_decay) {
        anyhow::bail!(
            "exit-agent artifact epsilon_decay must be finite and inside [0.90, 0.99999]"
        );
    }
    if artifact.memory_capacity < 1_024 {
        anyhow::bail!(
            "exit-agent artifact memory_capacity {} is below the supported minimum 1024",
            artifact.memory_capacity
        );
    }
    if !artifact.average_reward.is_finite() {
        anyhow::bail!("exit-agent artifact average_reward must be finite");
    }
    if let Some(report) = artifact.training_report.as_ref() {
        if report.train_rows != artifact.train_rows {
            anyhow::bail!(
                "exit-agent training report rows {} do not match artifact train_rows {}",
                report.train_rows,
                artifact.train_rows
            );
        }
        if report.memory_size != artifact.trained_memory_size {
            anyhow::bail!(
                "exit-agent training report memory_size {} does not match trained_memory_size {}",
                report.memory_size,
                artifact.trained_memory_size
            );
        }
        if !report.average_reward.is_finite() {
            anyhow::bail!("exit-agent training report average_reward must be finite");
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExitAgentTrainingReport {
    pub train_rows: usize,
    pub memory_size: usize,
    pub warmup_steps: usize,
    pub average_reward: f32,
}

/// Pure-Rust Burn ExitAgent
pub struct ExitAgent {
    model: ExitAgentNet<TrainBackend>,
    optim: OptimizerAdaptor<burn::optim::AdamW, ExitAgentNet<TrainBackend>, TrainBackend>,
    memory: VecDeque<Experience>,
    pending_regret: HashMap<i32, PendingRegret>,
    memory_capacity: usize,
    gamma: f32,
    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    input_dim: usize,
    hidden_dim: usize,
    feature_columns: Vec<String>,
    reward_horizon: usize,
    warmup_steps: usize,
    train_rows: usize,
    trained_memory_size: usize,
    average_reward: f32,
    training_report: Option<ExitAgentTrainingReport>,
    device: <TrainBackend as Backend>::Device,
    requested_device_policy: String,
    effective_device_policy: String,
    execution_backend: String,
}

impl ExitAgent {
    pub fn new(input_dim: usize) -> Self {
        Self::with_hidden_dim(input_dim, 64)
    }

    pub fn with_hidden_dim(input_dim: usize, hidden_dim: usize) -> Self {
        let (device, selection) = resolve_train_device("auto");
        let model = ExitAgentNetConfig::new()
            .with_input_dim(input_dim)
            .with_hidden_dim(hidden_dim)
            .init(&device);
        let optim = AdamWConfig::new().with_weight_decay(1e-4).init();

        Self {
            model,
            optim,
            memory: VecDeque::with_capacity(10000),
            pending_regret: HashMap::new(),
            memory_capacity: 10_000,
            gamma: 0.99,
            epsilon: 0.2,
            epsilon_min: 0.05,
            epsilon_decay: 0.999,
            input_dim,
            hidden_dim,
            feature_columns: Vec::new(),
            reward_horizon: 0,
            warmup_steps: 0,
            train_rows: 0,
            trained_memory_size: 0,
            average_reward: 0.0,
            training_report: None,
            device,
            requested_device_policy: selection.requested_policy,
            effective_device_policy: selection.effective_policy,
            execution_backend: selection.execution_backend,
        }
    }

    pub fn with_device_policy(mut self, policy: impl Into<String>) -> Self {
        let requested = policy.into();
        let (device, selection) = resolve_train_device(&requested);
        self.device = device;
        self.model = ExitAgentNetConfig::new()
            .with_input_dim(self.input_dim)
            .with_hidden_dim(self.hidden_dim)
            .init(&device);
        self.optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        self.requested_device_policy = selection.requested_policy;
        self.effective_device_policy = selection.effective_policy;
        self.execution_backend = selection.execution_backend;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        if gamma.is_finite() {
            self.gamma = gamma.clamp(0.01, 0.9999);
        }
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        if epsilon.is_finite() {
            self.epsilon = epsilon.clamp(0.0, 1.0);
        }
        self
    }

    pub fn with_exploration_schedule(mut self, epsilon_min: f32, epsilon_decay: f32) -> Self {
        if epsilon_min.is_finite() {
            self.epsilon_min = epsilon_min.clamp(0.0, 1.0);
        }
        if epsilon_decay.is_finite() {
            self.epsilon_decay = epsilon_decay.clamp(0.90, 0.99999);
        }
        if self.epsilon < self.epsilon_min {
            self.epsilon = self.epsilon_min;
        }
        self
    }

    pub fn with_memory_capacity(mut self, memory_capacity: usize) -> Self {
        self.memory_capacity = memory_capacity.max(1_024);
        self.memory = VecDeque::with_capacity(self.memory_capacity);
        self
    }

    pub fn with_reward_horizon(mut self, reward_horizon: usize) -> Self {
        self.reward_horizon = reward_horizon;
        self
    }

    pub fn with_warmup_steps(mut self, warmup_steps: usize) -> Self {
        self.warmup_steps = warmup_steps;
        self
    }

    fn normalize_direction(direction: i32) -> i32 {
        if direction < 0 {
            -1
        } else {
            1
        }
    }

    fn reward_from_trace(
        direction: i32,
        exit_price: f64,
        future_price_trace: &[f64],
        action: i64,
    ) -> f32 {
        if future_price_trace.is_empty() {
            return 0.0;
        }

        let mut min_future_price = f64::MAX;
        let mut max_future_price = f64::MIN;
        let direction = Self::normalize_direction(direction);
        let denom = exit_price.abs().max(1.0);
        let mut decayed_directional_edge = 0.0_f32;
        let mut reversal_pressure = 0.0_f32;
        for (step_idx, &p) in future_price_trace.iter().enumerate() {
            if p < min_future_price {
                min_future_price = p;
            }
            if p > max_future_price {
                max_future_price = p;
            }
            let signed_move = ((p - exit_price) * direction as f64 / denom).clamp(-2.0, 2.0) as f32;
            let decay = 1.0 / ((step_idx + 1) as f32).sqrt();
            decayed_directional_edge += signed_move * decay;
            reversal_pressure += (-signed_move).max(0.0) * decay;
        }

        let (favorable_move, adverse_move) = if direction > 0 {
            (max_future_price - exit_price, exit_price - min_future_price)
        } else {
            (exit_price - min_future_price, max_future_price - exit_price)
        };
        let favorable_ratio = (favorable_move.max(0.0) / denom) as f32;
        let adverse_ratio = (adverse_move.max(0.0) / denom) as f32;
        let hold_advantage = (favorable_ratio * 0.70 + decayed_directional_edge.max(0.0) * 0.30
            - adverse_ratio * 0.90
            - reversal_pressure * 0.20)
            .clamp(-1.0, 1.0);
        let close_advantage = (adverse_ratio * 0.85 + reversal_pressure * 0.35
            - favorable_ratio * 0.45
            - decayed_directional_edge.max(0.0) * 0.20)
            .clamp(-1.0, 1.0);

        let mut reward = if action == 1 {
            close_advantage
        } else {
            hold_advantage
        };
        if reward.abs() < 0.05 {
            reward *= 0.5;
        }
        reward.clamp(-1.0, 1.0)
    }

    /// Returns 0 (Hold) or 1 (Close).
    pub fn get_action(&self, state: &[f32], eval_mode: bool) -> i32 {
        let mut rng = rand::rng();
        if !eval_mode && rng.random::<f32>() < self.epsilon {
            return rng.random_range(0..=1);
        }

        // Forward pass
        let state_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(state.to_vec(), [state.len()]),
            &self.device,
        )
        .unsqueeze::<2>();

        let logits = self.model.forward(state_tensor);
        let action = logits
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap_or(vec![0])[0];
        action as i32
    }

    pub fn observe_exit(
        &mut self,
        ticket: i32,
        state: &[f32],
        action: i32,
        direction: i32,
        current_price: f64,
        timestamp: i64,
    ) {
        self.pending_regret.insert(
            ticket,
            PendingRegret {
                state: state.to_vec(),
                action: action as i64,
                exit_price: current_price,
                time: timestamp,
                direction: Self::normalize_direction(direction),
            },
        );
    }

    pub fn process_regret(&mut self, ticket: i32, future_price_trace: &[f64]) {
        if future_price_trace.is_empty() {
            return;
        }

        if let Some(data) = self.pending_regret.remove(&ticket) {
            let reward = Self::reward_from_trace(
                data.direction,
                data.exit_price,
                future_price_trace,
                data.action,
            );

            self.push_experience(Experience {
                state: data.state,
                action: data.action,
                reward,
                done: true,
            });
        }
    }

    pub fn train_step(&mut self) {
        if self.memory.len() < 32 {
            return;
        }

        let mut rng = rand::rng();
        let mut batch_indices: Vec<usize> = (0..self.memory.len()).collect();
        // Naive shuffling strategy for sampling
        use rand::seq::SliceRandom;
        batch_indices.shuffle(&mut rng);
        let batch_size = 32usize.min(self.memory.len());
        let batch_indices = &batch_indices[0..batch_size];

        let mut states_flat = Vec::with_capacity(batch_size * self.input_dim);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            let exp = &self.memory[idx];
            if exp.state.len() != self.input_dim {
                continue;
            }
            states_flat.extend_from_slice(&exp.state);
            actions.push(exp.action);
            rewards.push(exp.reward);
        }

        if actions.len() < 8 {
            return;
        }

        let effective_batch = actions.len();

        let states_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(states_flat, [effective_batch, self.input_dim]),
            &self.device,
        );
        let actions_tensor: Tensor<TrainBackend, 1, Int> =
            Tensor::from_data(TensorData::new(actions, [effective_batch]), &self.device);
        let rewards_tensor: Tensor<TrainBackend, 1> =
            Tensor::from_data(TensorData::new(rewards, [effective_batch]), &self.device);

        let q_values = self.model.forward(states_tensor);
        let q_value = q_values
            .gather(1, actions_tensor.unsqueeze_dim(1))
            .squeeze::<1>();

        let loss = burn::nn::loss::MseLoss::new().forward(
            q_value,
            rewards_tensor,
            burn::nn::loss::Reduction::Mean,
        );

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(1e-4, self.model.clone(), grads_params);

        self.epsilon = self.epsilon_min.max(self.epsilon * self.epsilon_decay);
    }

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }
    pub fn set_epsilon(&mut self, e: f32) {
        self.epsilon = e;
    }
    pub fn memory_size(&self) -> usize {
        self.memory.len()
    }

    fn artifact(&self) -> ExitAgentArtifact {
        ExitAgentArtifact {
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            feature_columns: self.feature_columns.clone(),
            gamma: self.gamma,
            epsilon: self.epsilon,
            epsilon_min: self.epsilon_min,
            epsilon_decay: self.epsilon_decay,
            memory_capacity: self.memory_capacity,
            reward_horizon: self.reward_horizon,
            warmup_steps: self.warmup_steps,
            train_rows: self.train_rows,
            trained_memory_size: self.trained_memory_size,
            average_reward: self.average_reward,
            replay_memory: self.memory.iter().cloned().collect(),
            pending_regret: self.pending_regret.clone(),
            training_report: self.training_report.clone(),
            requested_device_policy: Some(self.requested_device_policy.clone()),
            effective_device_policy: Some(self.effective_device_policy.clone()),
            execution_backend: Some(self.execution_backend.clone()),
        }
    }

    fn runtime_degraded_reason(&self) -> Option<String> {
        if self.requested_device_policy == self.effective_device_policy {
            None
        } else {
            Some(format!(
                "requested Burn device `{}` resolved to `{}` on the current build/runtime",
                self.requested_device_policy, self.effective_device_policy
            ))
        }
    }

    fn runtime_probabilities(hold_probability: f32, close_probability: f32) -> [f32; 3] {
        let total = (hold_probability + close_probability).max(f32::EPSILON);
        let hold_probability = (hold_probability / total).clamp(0.0, 1.0);
        let close_probability = (close_probability / total).clamp(0.0, 1.0);
        let disagreement = 1.0 - (hold_probability - close_probability).abs();
        let directional_confidence = hold_probability.max(close_probability);
        let neutral_probability = (disagreement * (1.0 - directional_confidence)).clamp(0.0, 0.35);
        let action_mass = 1.0 - neutral_probability;
        [
            hold_probability * action_mass,
            neutral_probability,
            close_probability * action_mass,
        ]
    }

    fn validated_runtime_probabilities(probabilities: &[f32]) -> Result<[f32; 2]> {
        if probabilities.len() != 2 {
            anyhow::bail!(
                "exit-agent runtime prediction expected 2 probabilities, received {}",
                probabilities.len()
            );
        }

        let hold_probability = probabilities[0];
        let close_probability = probabilities[1];
        if !hold_probability.is_finite() || !close_probability.is_finite() {
            anyhow::bail!("exit-agent runtime prediction returned non-finite probabilities");
        }
        if hold_probability < 0.0 || close_probability < 0.0 {
            anyhow::bail!("exit-agent runtime prediction returned negative probabilities");
        }

        let total = hold_probability + close_probability;
        if !total.is_finite() || total <= 0.0 {
            anyhow::bail!("exit-agent runtime prediction returned degenerate probability mass");
        }

        Ok([hold_probability / total, close_probability / total])
    }

    fn push_experience(&mut self, experience: Experience) {
        if self.memory.len() >= self.memory_capacity {
            self.memory.pop_front();
        }
        self.memory.push_back(experience);
    }

    pub fn fit_from_frame(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        self.fit_from_frame_with_report(x, y).map(|_| ())
    }

    pub fn fit_from_frame_with_report(
        &mut self,
        x: &DataFrame,
        y: &Series,
    ) -> Result<ExitAgentTrainingReport> {
        let features = dataframe_to_float32_array(x)?;
        let feature_columns = feature_columns_from_dataframe(x);
        let labels = y
            .cast(&DataType::Int32)
            .context("cast exit-agent labels to Int32")?;
        let labels = labels
            .i32()
            .context("access exit-agent labels as Int32")?
            .into_iter()
            .map(|value| value.context("exit-agent labels may not contain nulls"))
            .collect::<Result<Vec<_>, _>>()
            .context("collect exit-agent labels")?;

        if features.nrows() != labels.len() {
            anyhow::bail!(
                "exit-agent row mismatch: {} feature rows vs {} labels",
                features.nrows(),
                labels.len()
            );
        }
        if features.ncols() != self.input_dim {
            anyhow::bail!(
                "exit-agent feature mismatch: configured input_dim {} vs dataframe {}",
                self.input_dim,
                features.ncols()
            );
        }
        if feature_columns.is_empty() {
            anyhow::bail!("exit-agent requires at least one feature column");
        }
        if features.nrows() < 48 {
            anyhow::bail!(
                "exit-agent requires at least 48 rows, received {}",
                features.nrows()
            );
        }

        self.memory.clear();
        let horizon = if self.reward_horizon == 0 {
            (features.nrows() / 32).clamp(6, 24)
        } else {
            self.reward_horizon.clamp(2, 128)
        };
        for row_idx in 0..features.nrows().saturating_sub(horizon + 1) {
            let state = features.row(row_idx).iter().copied().collect::<Vec<_>>();
            if labels[row_idx] == 0 {
                continue;
            }
            let current_direction = Self::normalize_direction(labels[row_idx]);
            let mut bullish_score = 0.0_f32;
            let mut bearish_score = 0.0_f32;
            let mut neutral_score = 0.0_f32;
            let mut reversal_pressure = 0.0_f32;
            let mut previous_label = labels[row_idx];

            for step in 1..=horizon {
                let weight = 1.0 / step as f32;
                let next_label = labels[row_idx + step];
                match next_label {
                    1 => bullish_score += weight,
                    -1 => bearish_score += weight,
                    _ => neutral_score += weight,
                }
                if next_label != 0
                    && previous_label != 0
                    && next_label.signum() != previous_label.signum()
                {
                    reversal_pressure += weight;
                }
                previous_label = next_label;
            }

            let aligned_score = if current_direction > 0 {
                bullish_score
            } else {
                bearish_score
            };
            let opposing_score = if current_direction > 0 {
                bearish_score
            } else {
                bullish_score
            };
            let directional_edge = aligned_score - opposing_score;
            let close_reward =
                (opposing_score * 0.95 + reversal_pressure * 0.45 + neutral_score * 0.05
                    - aligned_score * 0.35)
                    .clamp(-1.0, 2.0);
            let hold_reward = (aligned_score * 0.95 + directional_edge.max(0.0) * 0.25
                - opposing_score * 0.55
                - reversal_pressure * 0.30
                - neutral_score * 0.08)
                .clamp(-1.0, 2.0);

            let action = if close_reward >= hold_reward { 1 } else { 0 };
            let reward = if action == 1 {
                close_reward
            } else {
                hold_reward
            };

            self.push_experience(Experience {
                state,
                action,
                reward,
                done: true,
            });
        }

        let warmup_steps = if self.warmup_steps == 0 {
            (self.memory.len() / 16).clamp(32, 256)
        } else {
            self.warmup_steps.max(8)
        };
        for _ in 0..warmup_steps {
            self.train_step();
        }

        let average_reward = if self.memory.is_empty() {
            0.0
        } else {
            self.memory.iter().map(|exp| exp.reward).sum::<f32>() / self.memory.len() as f32
        };
        self.train_rows = features.nrows();
        self.trained_memory_size = self.memory.len();
        self.average_reward = average_reward;
        self.feature_columns = feature_columns;
        let training_report = ExitAgentTrainingReport {
            train_rows: self.train_rows,
            memory_size: self.trained_memory_size,
            warmup_steps,
            average_reward,
        };
        self.training_report = Some(training_report.clone());

        info!(
            "trained exit agent from offline sequence dataset (rows={}, memory={})",
            features.nrows(),
            self.memory.len()
        );
        Ok(training_report)
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        let features = dataframe_to_float32_array(x)?;
        let actual_columns = feature_columns_from_dataframe(x);
        if features.ncols() != self.input_dim {
            anyhow::bail!(
                "exit-agent prediction feature mismatch: configured input_dim {} vs dataframe {}",
                self.input_dim,
                features.ncols()
            );
        }
        if !self.feature_columns.is_empty() && self.feature_columns != actual_columns {
            anyhow::bail!(
                "exit-agent prediction feature-column mismatch: expected {:?}, got {:?}",
                self.feature_columns,
                actual_columns
            );
        }

        let mut predictions = Vec::with_capacity(features.nrows());
        for row_idx in 0..features.nrows() {
            let state = features.row(row_idx).iter().copied().collect::<Vec<_>>();
            let state_tensor = Tensor::<TrainBackend, 1>::from_data(
                TensorData::new(state, [self.input_dim]),
                &self.device,
            )
            .unsqueeze::<2>();
            let probabilities =
                burn::tensor::activation::softmax(self.model.forward(state_tensor), 1)
                    .into_data()
                    .to_vec::<f32>()
                    .context("extract exit-agent runtime probabilities")?;

            let [hold_probability, close_probability] =
                Self::validated_runtime_probabilities(&probabilities)?;
            let runtime_probabilities =
                Self::runtime_probabilities(hold_probability, close_probability);
            let (confidence, abstain_recommended) =
                three_class_runtime_confidence(runtime_probabilities)?;
            predictions.push(build_runtime_prediction_with_details(
                "exit_agent",
                ModelFamily::Exit,
                CapabilityState::Implemented,
                runtime_probabilities,
                Some(confidence),
                Some(abstain_recommended),
                Some(self.execution_backend.clone()),
                self.runtime_degraded_reason(),
            )?);
        }

        Ok(predictions)
    }

    fn record_base_path(path: &Path) -> std::path::PathBuf {
        path.join("weights")
    }

    fn artifact_path(path: &Path) -> std::path::PathBuf {
        path.join("config.json")
    }

    fn optimizer_record_base_path(path: &Path) -> std::path::PathBuf {
        path.join("optimizer")
    }

    fn optimizer_record_file_path(path: &Path) -> std::path::PathBuf {
        let mut record_path = Self::optimizer_record_base_path(path);
        record_path.set_extension("mpk");
        record_path
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("create exit-agent directory {}", path.display()))?;
        let artifact = self.artifact();
        validate_exit_artifact(&artifact)?;

        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .valid()
            .save_file(Self::record_base_path(path), &recorder)
            .with_context(|| format!("persist exit-agent record to {}", path.display()))?;
        recorder
            .record(
                self.optim.to_record(),
                Self::optimizer_record_base_path(path),
            )
            .with_context(|| format!("persist exit-agent optimizer to {}", path.display()))?;

        let runtime_metadata =
            exit_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows);
        validate_exit_metadata(
            &runtime_metadata,
            &artifact.feature_columns,
            artifact.train_rows,
        )?;
        write_json(&path.join(METADATA_FILE_NAME), &runtime_metadata)?;
        write_json(&Self::artifact_path(path), &artifact)
            .with_context(|| format!("write exit-agent config to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let metadata: RuntimeArtifactMetadata = read_json(&path.join(METADATA_FILE_NAME))?;
        let artifact: ExitAgentArtifact = read_json(&Self::artifact_path(path))
            .with_context(|| format!("read exit-agent config from {}", path.display()))?;
        validate_exit_artifact(&artifact)?;
        validate_exit_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
        if artifact.replay_memory.len() > artifact.memory_capacity.max(1_024) {
            anyhow::bail!(
                "exit-agent artifact replay memory {} exceeds configured capacity {}",
                artifact.replay_memory.len(),
                artifact.memory_capacity.max(1_024)
            );
        }
        if artifact.train_rows != 0 && artifact.train_rows < artifact.replay_memory.len() {
            anyhow::bail!(
                "exit-agent artifact train_rows {} is smaller than persisted replay memory {}",
                artifact.train_rows,
                artifact.replay_memory.len()
            );
        }
        if artifact.trained_memory_size != 0
            && artifact.trained_memory_size < artifact.replay_memory.len()
        {
            anyhow::bail!(
                "exit-agent artifact trained_memory_size {} is smaller than persisted replay memory {}",
                artifact.trained_memory_size,
                artifact.replay_memory.len()
            );
        }

        let requested_device_policy = artifact
            .requested_device_policy
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let (device, selection) = resolve_train_device(&requested_device_policy);
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let model = ExitAgentNetConfig::new()
            .with_input_dim(artifact.input_dim)
            .with_hidden_dim(artifact.hidden_dim)
            .init(&device)
            .load_file(Self::record_base_path(path), &recorder, &device)
            .with_context(|| format!("load exit-agent record from {}", path.display()))?;

        let optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        let optim = if Self::optimizer_record_file_path(path).exists() {
            let optimizer_record = recorder
                .load(Self::optimizer_record_base_path(path), &device)
                .with_context(|| format!("load exit-agent optimizer from {}", path.display()))?;
            optim.load_record(optimizer_record)
        } else {
            warn!(
                "exit-agent optimizer checkpoint missing at {}; continuing with a fresh optimizer state",
                Self::optimizer_record_file_path(path).display()
            );
            optim
        };
        let replay_memory_len = artifact.replay_memory.len();

        Ok(Self {
            model,
            optim,
            memory: {
                let mut memory = VecDeque::with_capacity(artifact.memory_capacity.max(1_024));
                for experience in artifact
                    .replay_memory
                    .into_iter()
                    .take(artifact.memory_capacity.max(1_024))
                {
                    memory.push_back(experience);
                }
                memory
            },
            pending_regret: artifact.pending_regret,
            memory_capacity: artifact.memory_capacity.max(1_024),
            gamma: artifact.gamma,
            epsilon: artifact.epsilon,
            epsilon_min: artifact.epsilon_min.clamp(0.0, 1.0),
            epsilon_decay: artifact.epsilon_decay.clamp(0.90, 0.99999),
            input_dim: artifact.input_dim,
            hidden_dim: artifact.hidden_dim,
            feature_columns: artifact.feature_columns,
            reward_horizon: artifact.reward_horizon,
            warmup_steps: artifact.warmup_steps,
            train_rows: artifact.train_rows,
            trained_memory_size: artifact.trained_memory_size.max(replay_memory_len),
            average_reward: artifact.average_reward,
            training_report: artifact.training_report,
            device,
            requested_device_policy: selection.requested_policy,
            effective_device_policy: selection.effective_policy,
            execution_backend: selection.execution_backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{exit_runtime_metadata, ExitAgent, ExitAgentArtifact, Experience};
    use crate::base::three_class_runtime_confidence;
    use crate::statistical::common::{write_json, METADATA_FILE_NAME};
    use anyhow::Result;
    use burn::module::Param;
    use burn::tensor::Tensor;
    use polars::prelude::{DataFrame, NamedFrom, Series};
    use std::path::PathBuf;

    #[test]
    fn observe_exit_uses_explicit_direction() {
        let mut agent = ExitAgent::with_hidden_dim(6, 16);
        let state = vec![-10.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        agent.observe_exit(7, &state, 0, 1, 1.2345, 42);

        let pending = agent
            .pending_regret
            .get(&7)
            .expect("pending regret should be stored");
        assert_eq!(pending.direction, 1);
    }

    #[test]
    fn process_regret_keeps_pending_when_future_trace_is_empty() {
        let mut agent = ExitAgent::with_hidden_dim(6, 16);
        let state = vec![1.0, 0.2, 0.3, 0.4, 0.5, 0.6];

        agent.observe_exit(11, &state, 1, -1, 1.2000, 100);
        agent.process_regret(11, &[]);

        assert!(
            agent.pending_regret.contains_key(&11),
            "empty future trace should not consume the pending regret"
        );
    }

    #[test]
    fn reward_from_trace_prefers_hold_when_favorable_move_dominates() {
        let reward = ExitAgent::reward_from_trace(1, 1.2000, &[1.2050, 1.2040, 1.1980], 0);
        assert!(
            reward > 0.0,
            "hold should be rewarded when upside dominates"
        );

        let close_reward = ExitAgent::reward_from_trace(1, 1.2000, &[1.2050, 1.2040, 1.1980], 1);
        assert!(
            close_reward < 0.0,
            "closing should be penalized when upside dominates"
        );
    }

    #[test]
    fn runtime_probabilities_keep_exit_agent_mapping_truthful() {
        let mapped = ExitAgent::runtime_probabilities(0.7, 0.2);
        assert!(mapped[0] > mapped[2]);
        assert!(mapped[1] > 0.0);
        let total = mapped.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn runtime_probabilities_assign_more_neutral_mass_when_indecisive() {
        let decisive = ExitAgent::runtime_probabilities(0.8, 0.2);
        let indecisive = ExitAgent::runtime_probabilities(0.51, 0.49);
        assert!(
            indecisive[1] > decisive[1],
            "neutral mass should grow when hold and close are nearly tied"
        );
    }

    #[test]
    fn validated_runtime_probabilities_rejects_degenerate_rows() {
        let err = ExitAgent::validated_runtime_probabilities(&[0.0, 0.0])
            .expect_err("zero-mass probabilities should be rejected");
        assert!(
            err.to_string().contains("degenerate probability mass"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_rejects_replay_memory_longer_than_recorded_training_rows() {
        let path = unique_temp_dir("exit-agent-train-rows");
        let artifact = ExitAgentArtifact {
            input_dim: 6,
            hidden_dim: 16,
            feature_columns: vec![
                "f1".to_string(),
                "f2".to_string(),
                "f3".to_string(),
                "f4".to_string(),
                "f5".to_string(),
                "f6".to_string(),
            ],
            gamma: 0.99,
            epsilon: 0.2,
            epsilon_min: 0.05,
            epsilon_decay: 0.999,
            memory_capacity: 1_024,
            reward_horizon: 0,
            warmup_steps: 0,
            train_rows: 1,
            trained_memory_size: 2,
            average_reward: 0.0,
            training_report: None,
            requested_device_policy: None,
            effective_device_policy: None,
            execution_backend: None,
            replay_memory: vec![
                Experience {
                    state: vec![0.0; 6],
                    action: 0,
                    reward: 0.0,
                    done: true,
                },
                Experience {
                    state: vec![0.0; 6],
                    action: 1,
                    reward: 1.0,
                    done: true,
                },
            ],
            pending_regret: Default::default(),
        };
        std::fs::write(
            ExitAgent::artifact_path(&path),
            serde_json::to_vec_pretty(&artifact).expect("serialize artifact"),
        )
        .expect("write config");
        write_json(
            &path.join(METADATA_FILE_NAME),
            &exit_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows),
        )
        .expect("write metadata");

        let err = ExitAgent::load(&path)
            .err()
            .expect("inconsistent train rows should fail");
        assert!(
            err.to_string().contains("train_rows"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

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
    fn process_regret_respects_configured_memory_capacity() {
        let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
        let state = vec![1.0, 0.2, 0.3, 0.4, 0.5, 0.6];

        for ticket in 0..1_025 {
            agent.observe_exit(
                ticket,
                &state,
                0,
                1,
                1.2 + ticket as f64 * 0.001,
                ticket as i64,
            );
            agent.process_regret(ticket, &[1.2050, 1.2040, 1.1980]);
        }

        assert_eq!(agent.memory_size(), 1_024);
    }

    #[test]
    fn save_and_load_preserve_memory_capacity() {
        let mut agent = ExitAgent::with_hidden_dim(6, 16).with_memory_capacity(1_024);
        agent.feature_columns = vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
            "f5".to_string(),
            "f6".to_string(),
        ];
        agent.train_rows = 64;
        agent.training_report = Some(super::ExitAgentTrainingReport {
            train_rows: 64,
            memory_size: 0,
            warmup_steps: 0,
            average_reward: 0.0,
        });
        let path = unique_temp_dir("exit-agent-capacity");

        agent.save(&path).expect("save should succeed");
        let loaded = ExitAgent::load(&path).expect("load should succeed");

        assert_eq!(loaded.artifact().memory_capacity, 1_024);
        assert!(path.join("optimizer.mpk").exists());
        assert!(
            loaded.memory.capacity() >= 1_024,
            "loaded memory should honor the configured minimum capacity"
        );
        assert_eq!(
            loaded
                .training_report
                .as_ref()
                .expect("training report should round-trip")
                .train_rows,
            64
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn predict_runtime_uses_shared_three_class_confidence_gate() -> Result<()> {
        let mut agent = ExitAgent::with_hidden_dim(6, 16);
        let device = agent.device;
        agent.feature_columns = vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
            "f5".to_string(),
            "f6".to_string(),
        ];

        agent.model.fc1.weight = Param::from_tensor(Tensor::from_data([[0.0_f32; 16]; 6], &device));
        agent.model.fc1.bias = Some(Param::from_tensor(Tensor::from_data(
            [0.0_f32; 16],
            &device,
        )));
        agent.model.fc2.weight =
            Param::from_tensor(Tensor::from_data([[0.0_f32; 16]; 16], &device));
        agent.model.fc2.bias = Some(Param::from_tensor(Tensor::from_data(
            [0.0_f32; 16],
            &device,
        )));
        agent.model.output.weight =
            Param::from_tensor(Tensor::from_data([[0.0_f32; 2]; 16], &device));
        agent.model.output.bias = Some(Param::from_tensor(Tensor::from_data(
            [0.84729785_f32, 0.0],
            &device,
        )));

        let df = DataFrame::new(vec![
            Series::new("f1".into(), vec![0.0_f64]).into(),
            Series::new("f2".into(), vec![0.0_f64]).into(),
            Series::new("f3".into(), vec![0.0_f64]).into(),
            Series::new("f4".into(), vec![0.0_f64]).into(),
            Series::new("f5".into(), vec![0.0_f64]).into(),
            Series::new("f6".into(), vec![0.0_f64]).into(),
        ])?;

        let predictions = agent.predict_runtime(&df)?;
        let prediction = predictions
            .first()
            .expect("one runtime prediction should be produced");
        let expected_row = ExitAgent::runtime_probabilities(0.7, 0.3);
        let (expected_confidence, expected_abstain) = three_class_runtime_confidence(expected_row)?;

        assert!((prediction.confidence().expect("confidence") - expected_confidence).abs() < 1e-6);
        assert_eq!(prediction.abstain_recommended(), Some(expected_abstain));
        assert!(prediction.metadata().execution_backend.is_some());
        Ok(())
    }

    #[test]
    fn artifact_carries_device_policy_fields() {
        let agent = ExitAgent::with_hidden_dim(6, 16).with_device_policy("cpu");
        let artifact = agent.artifact();
        assert_eq!(artifact.requested_device_policy.as_deref(), Some("cpu"));
        assert_eq!(artifact.effective_device_policy.as_deref(), Some("cpu"));
        assert!(artifact.execution_backend.is_some());
    }
}
