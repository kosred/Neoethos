// Exit Agent - Pure Rust Burn RL-Based Trade Exit Expert
// Derived from src/forex_bot/models/exit_agent.py
//
// Lightweight RL Network for Trade Exit Decisions.
// Learns to balance Greed (Holding for TP) vs Fear (Cutting Loss/Stall).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

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
    build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    dataframe_to_float32_array, feature_columns_from_dataframe, three_class_runtime_confidence,
    try_build_runtime_artifact_metadata,
};
use crate::burn_models::{TrainBackend, resolve_train_device};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{METADATA_FILE_NAME, read_json, write_json};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_state: Option<Vec<f32>>,
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
    #[serde(default)]
    pub runtime_metadata: Option<RuntimeArtifactMetadata>,
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
            runtime_metadata: None,
        }
    }
}

fn exit_runtime_metadata(
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
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

fn validate_exit_metadata_consistency(
    sidecar: &RuntimeArtifactMetadata,
    embedded: &RuntimeArtifactMetadata,
) -> Result<()> {
    if sidecar.model_name != embedded.model_name
        || sidecar.family != embedded.family
        || sidecar.state != embedded.state
    {
        anyhow::bail!("exit-agent metadata identity mismatch between sidecar and embedded payload");
    }
    if sidecar.feature_columns != embedded.feature_columns {
        anyhow::bail!("exit-agent metadata feature columns drift between sidecar and embedded");
    }
    if sidecar.label_mapping != embedded.label_mapping {
        anyhow::bail!("exit-agent metadata label mapping drift between sidecar and embedded");
    }
    if sidecar.training_summary != embedded.training_summary {
        anyhow::bail!("exit-agent metadata training summary drift between sidecar and embedded");
    }
    Ok(())
}

fn resolve_exit_runtime_metadata(
    path: &Path,
    artifact: &ExitAgentArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_exit_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            if let Some(embedded) = artifact.runtime_metadata.as_ref() {
                validate_exit_metadata(embedded, &artifact.feature_columns, artifact.train_rows)?;
                validate_exit_metadata_consistency(&metadata, embedded)?;
            }
            Ok(metadata)
        }
        Err(error) => {
            let embedded = artifact.runtime_metadata.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "exit-agent metadata sidecar missing or unreadable at {} and no embedded runtime metadata in artifact: {}",
                    metadata_path.display(),
                    error
                )
            })?;
            warn!(
                "exit-agent metadata sidecar unavailable at {} ({}); falling back to embedded artifact metadata",
                metadata_path.display(),
                error
            );
            validate_exit_metadata(embedded, &artifact.feature_columns, artifact.train_rows)?;
            Ok(embedded.clone())
        }
    }
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
    if artifact.replay_memory.len() > artifact.memory_capacity {
        anyhow::bail!(
            "exit-agent artifact replay memory {} exceeds memory_capacity {}",
            artifact.replay_memory.len(),
            artifact.memory_capacity
        );
    }
    if !artifact.average_reward.is_finite() {
        anyhow::bail!("exit-agent artifact average_reward must be finite");
    }
    for (idx, experience) in artifact.replay_memory.iter().enumerate() {
        if experience.state.len() != artifact.input_dim {
            anyhow::bail!(
                "exit-agent replay experience {} has state width {} but input_dim is {}",
                idx,
                experience.state.len(),
                artifact.input_dim
            );
        }
        if !matches!(experience.action, 0 | 1) {
            anyhow::bail!(
                "exit-agent replay experience {} has unsupported action {}; expected 0 or 1",
                idx,
                experience.action
            );
        }
        if !experience.reward.is_finite() {
            anyhow::bail!("exit-agent replay experience {idx} has non-finite reward");
        }
        if experience.state.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("exit-agent replay experience {idx} has non-finite state values");
        }
        if let Some(next_state) = experience.next_state.as_ref() {
            if next_state.len() != artifact.input_dim {
                anyhow::bail!(
                    "exit-agent replay experience {} has next_state width {} but input_dim is {}",
                    idx,
                    next_state.len(),
                    artifact.input_dim
                );
            }
            if next_state.iter().any(|value| !value.is_finite()) {
                anyhow::bail!(
                    "exit-agent replay experience {idx} has non-finite next_state values"
                );
            }
        }
    }
    for (ticket, regret) in &artifact.pending_regret {
        if regret.state.len() != artifact.input_dim {
            anyhow::bail!(
                "exit-agent pending regret {} has state width {} but input_dim is {}",
                ticket,
                regret.state.len(),
                artifact.input_dim
            );
        }
        if !matches!(regret.action, 0 | 1) {
            anyhow::bail!(
                "exit-agent pending regret {} has unsupported action {}; expected 0 or 1",
                ticket,
                regret.action
            );
        }
        if !regret.exit_price.is_finite() {
            anyhow::bail!("exit-agent pending regret {ticket} has non-finite exit_price");
        }
        if regret.state.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("exit-agent pending regret {ticket} has non-finite state values");
        }
        if !matches!(regret.direction, -1 | 1) {
            anyhow::bail!(
                "exit-agent pending regret {} has invalid direction {}; expected -1 or 1",
                ticket,
                regret.direction
            );
        }
    }
    let runtime_fields = [
        artifact.requested_device_policy.as_deref(),
        artifact.effective_device_policy.as_deref(),
        artifact.execution_backend.as_deref(),
    ];
    let runtime_fields_present = runtime_fields
        .iter()
        .filter(|value| value.is_some())
        .count();
    if runtime_fields_present != 0 && runtime_fields_present != runtime_fields.len() {
        anyhow::bail!(
            "exit-agent artifact must persist requested_device_policy, effective_device_policy, and execution_backend together"
        );
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
        if (report.average_reward - artifact.average_reward).abs() > 1e-6 {
            anyhow::bail!(
                "exit-agent training report average_reward {} does not match artifact average_reward {}",
                report.average_reward,
                artifact.average_reward
            );
        }
        if report.warmup_steps != artifact.warmup_steps {
            anyhow::bail!(
                "exit-agent training report warmup_steps {} does not match artifact warmup_steps {}",
                report.warmup_steps,
                artifact.warmup_steps
            );
        }
        if report.reward_horizon != artifact.reward_horizon {
            anyhow::bail!(
                "exit-agent training report reward_horizon {} does not match artifact reward_horizon {}",
                report.reward_horizon,
                artifact.reward_horizon
            );
        }
        if report.feature_count != artifact.input_dim {
            anyhow::bail!(
                "exit-agent training report feature_count {} does not match artifact input_dim {}",
                report.feature_count,
                artifact.input_dim
            );
        }
        if report.requested_device_policy.trim().is_empty()
            || report.effective_device_policy.trim().is_empty()
            || report.execution_backend.trim().is_empty()
        {
            anyhow::bail!("exit-agent training report must persist runtime identity");
        }
        if artifact.requested_device_policy.as_deref()
            != Some(report.requested_device_policy.as_str())
            || artifact.effective_device_policy.as_deref()
                != Some(report.effective_device_policy.as_str())
            || artifact.execution_backend.as_deref() != Some(report.execution_backend.as_str())
        {
            anyhow::bail!(
                "exit-agent training report runtime identity does not match artifact runtime identity"
            );
        }
    }
    if (artifact.trained_memory_size > 0 || !artifact.replay_memory.is_empty())
        && artifact.training_report.is_none()
    {
        anyhow::bail!(
            "exit-agent trained artifacts must persist a training_report alongside replay state"
        );
    }
    if (artifact.trained_memory_size > 0 || !artifact.replay_memory.is_empty())
        && runtime_fields_present != runtime_fields.len()
    {
        anyhow::bail!("exit-agent trained artifacts must persist complete runtime identity fields");
    }
    if !artifact.replay_memory.is_empty()
        && artifact.trained_memory_size != artifact.replay_memory.len()
    {
        anyhow::bail!(
            "exit-agent artifact trained_memory_size {} must match persisted replay memory {}",
            artifact.trained_memory_size,
            artifact.replay_memory.len()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExitAgentTrainingReport {
    pub train_rows: usize,
    pub memory_size: usize,
    pub warmup_steps: usize,
    pub average_reward: f32,
    pub reward_horizon: usize,
    pub feature_count: usize,
    pub requested_device_policy: String,
    pub effective_device_policy: String,
    pub execution_backend: String,
}

/// Pure-Rust Burn ExitAgent
pub struct ExitAgent {
    model: ExitAgentNet<TrainBackend>,
    target_model: ExitAgentNet<TrainBackend>,
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
    trained_checkpoint_ready: bool,
    train_step_count: usize,
    device: <TrainBackend as Backend>::Device,
    requested_device_policy: String,
    effective_device_policy: String,
    execution_backend: String,
    persisted_requested_device_policy: Option<String>,
    persisted_effective_device_policy: Option<String>,
    persisted_execution_backend: Option<String>,
}

impl ExitAgent {
    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    fn invalidate_trained_runtime_state(&mut self) {
        self.memory.clear();
        self.pending_regret.clear();
        self.train_rows = 0;
        self.trained_memory_size = 0;
        self.average_reward = 0.0;
        self.training_report = None;
        self.trained_checkpoint_ready = false;
        self.train_step_count = 0;
        self.target_model = self.model.clone();
        self.persisted_requested_device_policy = None;
        self.persisted_effective_device_policy = None;
        self.persisted_execution_backend = None;
    }

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
            target_model: model.clone(),
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
            trained_checkpoint_ready: false,
            train_step_count: 0,
            device,
            requested_device_policy: selection.requested_policy,
            effective_device_policy: selection.effective_policy,
            execution_backend: selection.execution_backend,
            persisted_requested_device_policy: None,
            persisted_effective_device_policy: None,
            persisted_execution_backend: None,
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
        self.target_model = self.model.clone();
        self.optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        self.invalidate_trained_runtime_state();
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
        if direction < 0 { -1 } else { 1 }
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

    fn target_q_max(&self, state: &[f32]) -> Option<f32> {
        if state.len() != self.input_dim || state.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let state_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(state.to_vec(), [self.input_dim]),
            &self.device,
        )
        .unsqueeze::<2>();
        self.target_model
            .forward(state_tensor)
            .into_data()
            .to_vec::<f32>()
            .ok()
            .and_then(|values| {
                values
                    .into_iter()
                    .filter(|value| value.is_finite())
                    .max_by(|left, right| left.total_cmp(right))
            })
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
                next_state: None,
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
        let mut targets = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            let exp = &self.memory[idx];
            if exp.state.len() != self.input_dim {
                continue;
            }
            states_flat.extend_from_slice(&exp.state);
            actions.push(exp.action);
            let bootstrap = if exp.done {
                0.0
            } else {
                exp.next_state
                    .as_deref()
                    .and_then(|next_state| self.target_q_max(next_state))
                    .map(|max_next_q| self.gamma * max_next_q)
                    .unwrap_or(0.0)
            };
            targets.push((exp.reward + bootstrap).clamp(-2.0, 2.0));
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
        let target_tensor: Tensor<TrainBackend, 1> =
            Tensor::from_data(TensorData::new(targets, [effective_batch]), &self.device);

        let q_values = self.model.forward(states_tensor);
        let q_value = q_values
            .gather(1, actions_tensor.unsqueeze_dim(1))
            .squeeze::<1>();

        let loss = burn::nn::loss::MseLoss::new().forward(
            q_value,
            target_tensor,
            burn::nn::loss::Reduction::Mean,
        );

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(1e-4, self.model.clone(), grads_params);
        self.train_step_count = self.train_step_count.saturating_add(1);
        if self.train_step_count.is_multiple_of(32) {
            self.target_model = self.model.clone();
        }

        self.epsilon = self.epsilon_min.max(self.epsilon * self.epsilon_decay);
    }

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }
    pub fn set_epsilon(&mut self, e: f32) {
        if e.is_finite() {
            self.epsilon = e.clamp(self.epsilon_min, 1.0);
        }
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
            runtime_metadata: None,
        }
    }

    fn has_runtime_feature_schema(&self) -> bool {
        !self.feature_columns.is_empty() && self.feature_columns.len() == self.input_dim
    }

    fn has_trained_runtime_state(&self) -> bool {
        if self.train_rows == 0
            || !self.has_runtime_feature_schema()
            || !self.trained_checkpoint_ready
        {
            return false;
        }

        self.trained_memory_size > 0
            || self
                .training_report
                .as_ref()
                .map(|report| report.memory_size > 0)
                .unwrap_or(false)
            || !self.memory.is_empty()
    }

    fn ensure_persistable_runtime_state(&self) -> Result<()> {
        if !self.has_runtime_feature_schema() {
            anyhow::bail!("exit-agent cannot persist without a runtime feature schema");
        }
        if !self.has_trained_runtime_state() {
            anyhow::bail!("exit-agent cannot persist from an untrained runtime state");
        }
        let report = self
            .training_report
            .as_ref()
            .context("exit-agent cannot persist without a training report")?;
        if report.memory_size == 0 {
            anyhow::bail!("exit-agent cannot persist with zero training-report memory_size");
        }
        if self.memory.is_empty() {
            anyhow::bail!("exit-agent cannot persist with empty replay memory");
        }
        Ok(())
    }

    fn runtime_degraded_reason(&self) -> Option<String> {
        let mut reasons = Vec::new();
        if let Some(persisted_requested_policy) = self.persisted_requested_device_policy.as_ref()
            && persisted_requested_policy != &self.requested_device_policy
        {
            reasons.push(format!(
                    "persisted requested device `{persisted_requested_policy}` differs from current runtime `{}`",
                    self.requested_device_policy
                ));
        }
        if self.requested_device_policy != self.effective_device_policy {
            reasons.push(format!(
                "requested Burn device `{}` resolved to `{}` on the current build/runtime",
                self.requested_device_policy, self.effective_device_policy
            ));
        }
        if let Some(persisted_policy) = self.persisted_effective_device_policy.as_ref()
            && persisted_policy != &self.effective_device_policy
        {
            reasons.push(format!(
                "persisted effective device `{persisted_policy}` differs from current runtime `{}`",
                self.effective_device_policy
            ));
        }
        if let Some(persisted_backend) = self.persisted_execution_backend.as_ref()
            && persisted_backend != &self.execution_backend
        {
            reasons.push(format!(
                    "persisted execution backend `{persisted_backend}` differs from current runtime `{}`",
                    self.execution_backend
                ));
        }
        if !self.has_runtime_feature_schema() {
            reasons.push("exit_agent is missing persisted feature schema".to_string());
        }
        if self.training_report.is_none() {
            if self.train_rows > 0 || self.trained_memory_size > 0 {
                reasons.push(
                    "exit_agent is running without a persisted training report for a trained artifact"
                        .to_string(),
                );
            } else {
                reasons.push("exit_agent has no persisted training report".to_string());
            }
        }
        if !self.trained_checkpoint_ready {
            reasons.push(
                "exit_agent has no verified trained checkpoint loaded in runtime".to_string(),
            );
        }
        if self.trained_memory_size == 0 {
            reasons.push("exit_agent has zero persisted replay memory".to_string());
        }
        if !self.has_trained_runtime_state() {
            reasons
                .push("exit_agent runtime state is not trained enough for inference".to_string());
        }

        if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        }
    }

    fn runtime_probabilities(hold_probability: f32, close_probability: f32) -> [f32; 3] {
        let total = (hold_probability + close_probability).max(f32::EPSILON);
        let hold_probability = (hold_probability / total).clamp(0.0, 1.0);
        let close_probability = (close_probability / total).clamp(0.0, 1.0);
        let indecision = (1.0 - (hold_probability - close_probability).abs()).clamp(0.0, 1.0);
        let neutral_probability =
            (indecision * hold_probability.min(close_probability)).clamp(0.0, 0.5);
        let directional_scale = (1.0 - neutral_probability).max(f32::EPSILON);
        [
            hold_probability * directional_scale,
            neutral_probability,
            close_probability * directional_scale,
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
            let next_state = features
                .row((row_idx + 1).min(features.nrows() - 1))
                .iter()
                .copied()
                .collect::<Vec<_>>();
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
                next_state: Some(next_state),
                action,
                reward,
                done: action == 1 || row_idx + horizon + 2 >= features.nrows(),
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
        self.target_model = self.model.clone();
        let training_report = ExitAgentTrainingReport {
            train_rows: self.train_rows,
            memory_size: self.trained_memory_size,
            warmup_steps,
            average_reward,
            reward_horizon: horizon,
            feature_count: self.input_dim,
            requested_device_policy: self.requested_device_policy.clone(),
            effective_device_policy: self.effective_device_policy.clone(),
            execution_backend: self.execution_backend.clone(),
        };
        self.training_report = Some(training_report.clone());
        self.trained_checkpoint_ready = true;

        info!(
            "trained exit agent from offline sequence dataset (rows={}, memory={})",
            features.nrows(),
            self.memory.len()
        );
        Ok(training_report)
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_inference_runtime_ready()?;
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

    fn ensure_inference_runtime_ready(&self) -> Result<()> {
        if !self.has_trained_runtime_state() {
            anyhow::bail!("exit-agent cannot run inference from an untrained runtime state");
        }
        if !self.has_runtime_feature_schema() {
            anyhow::bail!("exit-agent cannot run inference without persisted feature schema");
        }
        if self.training_report.is_none() {
            anyhow::bail!("exit-agent cannot run inference without a persisted training report");
        }
        if self.requested_device_policy.trim().is_empty()
            || self.effective_device_policy.trim().is_empty()
            || self.execution_backend.trim().is_empty()
        {
            anyhow::bail!("exit-agent cannot run inference without complete runtime identity");
        }
        if !self.trained_checkpoint_ready {
            anyhow::bail!("exit-agent cannot run inference without a verified trained checkpoint");
        }
        Ok(())
    }

    fn staged_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("tmp_artifact")
    }

    fn backup_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("bak_artifact")
    }

    fn cleanup_artifact_dir(path: &Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("remove staged exit-agent artifact {}", path.display()))?;
        }
        Ok(())
    }

    fn replace_artifact_dir(staged_path: &Path, target_path: &Path) -> Result<()> {
        let backup_path = Self::backup_artifact_dir(target_path);
        Self::cleanup_artifact_dir(&backup_path)?;
        if target_path.exists() {
            std::fs::rename(target_path, &backup_path).with_context(|| {
                format!(
                    "move previous exit-agent artifact into backup {}",
                    backup_path.display()
                )
            })?;
        }
        if let Err(error) = std::fs::rename(staged_path, target_path) {
            if backup_path.exists() {
                if let Err(restore_err) = std::fs::rename(&backup_path, target_path) {
                    tracing::error!(
                        target: "forex_models::artifact",
                        backup = %backup_path.display(),
                        target = %target_path.display(),
                        error = %restore_err,
                        "failed to restore backup after staged-rename failure;                      artifact directory may be in an inconsistent state"
                    );
                }
            }
            anyhow::bail!(
                "rename staged exit-agent artifact into {} failed: {}",
                target_path.display(),
                error
            );
        }
        Self::cleanup_artifact_dir(&backup_path)?;
        Ok(())
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
        self.ensure_persistable_runtime_state()?;
        let artifact = self.artifact();
        validate_exit_artifact(&artifact)?;
        let staged_path = Self::staged_artifact_dir(path);
        Self::cleanup_artifact_dir(&staged_path)?;
        std::fs::create_dir_all(&staged_path).with_context(|| {
            format!(
                "create staged exit-agent directory {}",
                staged_path.display()
            )
        })?;
        if let Err(error) = (|| -> Result<()> {
            let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            self.model
                .clone()
                .valid()
                .save_file(Self::record_base_path(&staged_path), &recorder)
                .with_context(|| {
                    format!("persist exit-agent record to {}", staged_path.display())
                })?;
            recorder
                .record(
                    self.optim.to_record(),
                    Self::optimizer_record_base_path(&staged_path),
                )
                .with_context(|| {
                    format!("persist exit-agent optimizer to {}", staged_path.display())
                })?;

            let runtime_metadata =
                exit_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
            validate_exit_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            let mut artifact_to_write = artifact.clone();
            artifact_to_write.runtime_metadata = Some(runtime_metadata.clone());
            write_json(&staged_path.join(METADATA_FILE_NAME), &runtime_metadata)?;
            write_json(&Self::artifact_path(&staged_path), &artifact_to_write)
                .with_context(|| format!("write exit-agent config to {}", staged_path.display()))?;
            Ok(())
        })() {
            let _ = Self::cleanup_artifact_dir(&staged_path);
            return Err(error);
        }
        Self::replace_artifact_dir(&staged_path, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let artifact: ExitAgentArtifact = read_json(&Self::artifact_path(path))
            .with_context(|| format!("read exit-agent config from {}", path.display()))?;
        validate_exit_artifact(&artifact)?;
        let _metadata = resolve_exit_runtime_metadata(path, &artifact)?;
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
        let requested_device_policy = artifact
            .requested_device_policy
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let persisted_requested_device_policy = artifact.requested_device_policy.clone();
        let persisted_effective_device_policy = artifact.effective_device_policy.clone();
        let persisted_execution_backend = artifact.execution_backend.clone();
        let (device, selection) = resolve_train_device(&requested_device_policy);
        if let Some(persisted_policy) = persisted_effective_device_policy.as_deref()
            && persisted_policy != selection.effective_policy
        {
            warn!(
                "exit-agent persisted effective device policy `{}` differs from current runtime `{}` while loading {}",
                persisted_policy,
                selection.effective_policy,
                path.display()
            );
        }
        if let Some(persisted_backend) = persisted_execution_backend.as_deref()
            && persisted_backend != selection.execution_backend
        {
            warn!(
                "exit-agent persisted execution backend `{}` differs from current runtime `{}` while loading {}",
                persisted_backend,
                selection.execution_backend,
                path.display()
            );
        }
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
        Ok(Self {
            target_model: model.clone(),
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
            trained_memory_size: artifact.trained_memory_size,
            average_reward: artifact.average_reward,
            training_report: artifact.training_report,
            trained_checkpoint_ready: true,
            train_step_count: 0,
            device,
            requested_device_policy: selection.requested_policy,
            effective_device_policy: selection.effective_policy,
            execution_backend: selection.execution_backend,
            persisted_requested_device_policy,
            persisted_effective_device_policy,
            persisted_execution_backend,
        })
    }
}

#[cfg(test)]
#[path = "exit_agent_tests.rs"]
mod tests;
