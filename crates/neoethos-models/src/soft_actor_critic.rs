//! Soft Actor-Critic for **discrete** action settings (SAC-Discrete).
//!
//! Faithful Rust/Burn implementation of Christodoulou (2019),
//! "Soft Actor-Critic for Discrete Action Settings"
//! (<https://arxiv.org/abs/1910.07207>). This replaces the prior
//! silent no-op where the `models.use_sac_agent` config flag merely
//! aliased to the DQN [`crate::exit_agent::ExitAgent`].
//!
//! SAC is an **entry / direction** policy (Hold / Buy / Sell). Its
//! 3-class policy probabilities participate in the soft-voting
//! ensemble exactly like the DQN entry voter
//! ([`crate::ensemble_inference::DqnAdapter`]) — they are NOT filtered
//! out like the exit agent's `ExitDecision3` outputs.
//!
//! ## Why the full-information critic regression
//!
//! The RL training data ([`crate::rl::TradingTransition`]) records the
//! reward for ALL THREE actions at every state (`rewards: [f32; 3]`),
//! not just the action that was actually taken. This is the
//! fully-observed-reward setting. Standard SAC samples a single action
//! `a ~ π(·|s)` and regresses `Q(s,a)` against the soft-Bellman target.
//! Here, because `r(s,a)` is observed for every `a`, we instead regress
//! BOTH critics over ALL three actions (`MSE` over the length-3 Q
//! vector) against the per-action soft target `y(s,a)`. This is a
//! deliberate, well-justified adaptation that uses all the available
//! reward signal each step — it strictly dominates the
//! single-sampled-action regression when the full reward vector is
//! known, and removes the sampling variance from the critic update.
//!
//! ## Discrete closed-form losses (exact, no RNG sampling)
//!
//! Let `p = softmax(actor_logits(s))`, `logp = log_softmax(...)`,
//! `Qmin(s,a) = min(Q1(s,a), Q2(s,a))`, `α = exp(log_alpha)`,
//! `γ` the discount, `H = target_entropy`. For the next state `s'` use
//! the actor's policy `p'`, `logp'` and the TARGET critics `Q1ₜ, Q2ₜ`:
//!
//! - Soft state value of `s'`:
//!   `V(s') = Σ_a p'(a) · [ min(Q1ₜ,Q2ₜ)(s',a) − α·logp'(a) ]`
//! - Critic target (all actions, full information):
//!   `y(s,a) = r(s,a) + γ·(1−done)·V(s')`
//! - Critic loss: `MSE(Q1(s,·), y(s,·)) + MSE(Q2(s,·), y(s,·))`
//! - Actor loss (exact expectation over the 3 actions):
//!   `L_π = Σ_a p(a) · [ α·logp(a) − Qmin(s,a) ]`
//! - Temperature loss (automatic entropy tuning):
//!   `L_α = − Σ_a p(a) · log_alpha · ( logp(a).detach + H )`
//!
//! All target / stop-gradient terms use [`Tensor::detach`].
//!
//! ## Conventions mirrored from [`crate::exit_agent`]
//!
//! `TrainBackend`/`InferBackend` aliases, AdamW via `OptimizerAdaptor`,
//! `GradientsParams` update pattern, deterministic seeded init, atomic
//! staged artifact save + `load`, fail-loud `bail!`/`Context`, runtime
//! metadata via `try_build_runtime_artifact_metadata`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::module::AutodiffModule;
use burn::nn;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::BackendTypes;

use polars::prelude::{DataFrame, Series};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::base::{
    build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::burn_models::{TrainBackend, resolve_train_device};
use crate::rl::{TradingEpisode, build_training_episodes_public};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;
use crate::statistical::common::{
    FeatureScaler, METADATA_FILE_NAME, feature_matrix_from_dataframe, read_json,
    remap_three_class_labels, write_json,
};

/// Number of discrete trading actions (Hold / Buy / Sell).
const NUM_ACTIONS: usize = 3;

// ============================================================================
// NETWORKS
// ============================================================================

/// Actor π(·|s): emits 3 logits → categorical policy over actions.
#[derive(Module, Debug)]
pub struct SacActorNet<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    head: nn::Linear<B>,
}

/// Critic Q(s): emits a length-3 vector of action-values.
#[derive(Module, Debug)]
pub struct SacCriticNet<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    head: nn::Linear<B>,
}

/// Learnable log-temperature `log_alpha` (single scalar parameter).
///
/// Wrapped in a [`Module`] so AdamW can optimize it through the same
/// `GradientsParams` pattern used for the networks.
#[derive(Module, Debug)]
pub struct SacTemperature<B: Backend> {
    log_alpha: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct SacNetConfig {
    #[config(default = 8)]
    pub input_dim: usize,
    #[config(default = 256)]
    pub hidden_dim: usize,
}

impl SacNetConfig {
    pub fn init_actor<B: Backend>(&self, device: &B::Device) -> SacActorNet<B> {
        SacActorNet {
            fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            head: nn::LinearConfig::new(self.hidden_dim, NUM_ACTIONS).init(device),
        }
    }

    pub fn init_critic<B: Backend>(&self, device: &B::Device) -> SacCriticNet<B> {
        SacCriticNet {
            fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            head: nn::LinearConfig::new(self.hidden_dim, NUM_ACTIONS).init(device),
        }
    }
}

impl<B: Backend> SacActorNet<B> {
    /// Raw actor logits, shape `[batch, 3]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.fc1.forward(x));
        let h = burn::tensor::activation::relu(self.fc2.forward(h));
        self.head.forward(h)
    }

    /// Policy probabilities `softmax(logits)`, shape `[batch, 3]`.
    pub fn policy(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(self.forward(x), 1)
    }
}

impl<B: Backend> SacCriticNet<B> {
    /// Action-value vector `Q(s, ·)`, shape `[batch, 3]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.fc1.forward(x));
        let h = burn::tensor::activation::relu(self.fc2.forward(h));
        self.head.forward(h)
    }
}

impl<B: Backend> SacTemperature<B> {
    fn init(device: &B::Device, init_log_alpha: f32) -> Self {
        // A 1->1 Linear with zero weight is a trainable scalar bias.
        // Seed the bias to the requested initial log_alpha and freeze
        // the weight at zero so the only learnable degree of freedom is
        // the scalar bias (== log_alpha). The weight still receives
        // gradients (input is the constant 1.0) which is fine — it is a
        // valid extra parameter, but we read log_alpha from the network
        // output at the constant input 1.0 so the scalar is well-defined.
        let mut linear = nn::LinearConfig::new(1, 1).with_bias(true).init(device);
        let weight_dims = linear.weight.val().dims();
        linear.weight = burn::module::Param::from_tensor(Tensor::zeros(weight_dims, device));
        if let Some(bias) = linear.bias.take() {
            let bias_dims = bias.val().dims();
            linear.bias = Some(burn::module::Param::from_tensor(
                Tensor::ones(bias_dims, device) * init_log_alpha,
            ));
        }
        Self { log_alpha: linear }
    }

    /// `log_alpha` as a `[1, 1]` tensor evaluated at the constant input
    /// `1.0` (weight is zero so this is exactly the trainable bias).
    fn log_alpha(&self, device: &B::Device) -> Tensor<B, 2> {
        let one = Tensor::<B, 2>::ones([1, 1], device);
        self.log_alpha.forward(one)
    }

    /// Scalar `alpha = exp(log_alpha)` value (read off the graph).
    fn alpha_value(&self, device: &B::Device) -> Result<f32> {
        let log_alpha = self
            .log_alpha(device)
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("read sac log_alpha tensor: {err:?}"))?;
        let log_alpha = log_alpha
            .first()
            .copied()
            .context("sac log_alpha tensor is empty")?;
        if !log_alpha.is_finite() {
            bail!("sac log_alpha is non-finite");
        }
        Ok(log_alpha.exp())
    }
}

// ============================================================================
// TRAINING TUPLES
// ============================================================================

/// One SAC training tuple built from a [`TradingTransition`].
///
/// Holds the per-action reward vector (full information) so the critic
/// target can be computed for every action without sampling.
#[derive(Clone, Debug)]
struct SacTuple {
    state: Vec<f32>,
    next_state: Vec<f32>,
    rewards: [f32; NUM_ACTIONS],
    done: bool,
}

fn tuples_from_episodes(episodes: &[TradingEpisode], state_dim: usize) -> Result<Vec<SacTuple>> {
    let mut tuples = Vec::new();
    for episode in episodes {
        for transition in &episode.transitions {
            if transition.state.len() != state_dim || transition.next_state.len() != state_dim {
                bail!(
                    "sac transition state dimension mismatch: expected {}, got {} / {}",
                    state_dim,
                    transition.state.len(),
                    transition.next_state.len()
                );
            }
            if transition
                .state
                .iter()
                .chain(transition.next_state.iter())
                .any(|value| !value.is_finite())
            {
                bail!("sac transition contains non-finite state values");
            }
            if transition.rewards.iter().any(|value| !value.is_finite()) {
                bail!("sac transition contains non-finite rewards");
            }
            tuples.push(SacTuple {
                state: transition.state.clone(),
                next_state: transition.next_state.clone(),
                rewards: transition.rewards,
                done: transition.done,
            });
        }
    }
    if tuples.is_empty() {
        bail!("sac training built no tuples from the provided episodes");
    }
    Ok(tuples)
}

// ============================================================================
// ARTIFACT
// ============================================================================

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SacTrainingReport {
    pub train_rows: usize,
    pub tuple_count: usize,
    pub state_dim: usize,
    pub epochs: usize,
    pub batches: usize,
    pub reward_horizon: usize,
    pub episode_len: usize,
    pub final_alpha: f32,
    pub final_critic_loss: f32,
    pub final_actor_loss: f32,
    pub average_hold_reward: f32,
    pub average_buy_reward: f32,
    pub average_sell_reward: f32,
    pub requested_device_policy: String,
    pub effective_device_policy: String,
    pub execution_backend: String,
    pub used_feature_scaler: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftActorCriticArtifact {
    pub state_dim: usize,
    pub hidden_dim: usize,
    #[serde(default)]
    pub feature_columns: Vec<String>,
    pub gamma: f32,
    pub tau: f32,
    pub learning_rate: f64,
    pub target_entropy_scale: f32,
    pub target_entropy: f32,
    pub init_log_alpha: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub reward_horizon: usize,
    pub episode_len: usize,
    pub train_rows: usize,
    pub tuple_count: usize,
    pub final_alpha: f32,
    #[serde(default)]
    pub feature_scaler: Option<FeatureScaler>,
    #[serde(default)]
    pub training_report: Option<SacTrainingReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_device_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_device_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_backend: Option<String>,
    #[serde(default)]
    pub runtime_metadata: Option<RuntimeArtifactMetadata>,
}

impl Default for SoftActorCriticArtifact {
    fn default() -> Self {
        Self {
            state_dim: 0,
            hidden_dim: 256,
            feature_columns: Vec::new(),
            gamma: 0.99,
            tau: 0.01,
            learning_rate: 3e-4,
            target_entropy_scale: 0.98,
            target_entropy: 0.0,
            init_log_alpha: 0.0,
            epochs: 32,
            batch_size: 64,
            reward_horizon: 0,
            episode_len: 0,
            train_rows: 0,
            tuple_count: 0,
            final_alpha: 1.0,
            feature_scaler: None,
            training_report: None,
            requested_device_policy: None,
            effective_device_policy: None,
            execution_backend: None,
            runtime_metadata: None,
        }
    }
}

/// Default discrete-SAC target entropy: `scale * log(num_actions)`
/// (Christodoulou §5 uses ~0.98 of the maximum entropy `log|A|`).
fn default_target_entropy(target_entropy_scale: f32) -> f32 {
    target_entropy_scale * (NUM_ACTIONS as f32).ln()
}

fn sac_runtime_metadata(
    feature_columns: Vec<String>,
    dataset_rows: usize,
) -> Result<RuntimeArtifactMetadata> {
    try_build_runtime_artifact_metadata(
        "sac",
        ModelFamily::Rl,
        CapabilityState::Implemented,
        feature_columns,
        canonical_three_class_label_mapping(),
        TrainingSummaryMetadata::new(dataset_rows, dataset_rows, 0),
    )
}

fn validate_sac_metadata(
    metadata: &RuntimeArtifactMetadata,
    expected_feature_columns: &[String],
    expected_dataset_rows: usize,
) -> Result<()> {
    if metadata.model_name != "sac" {
        bail!(
            "sac metadata model mismatch: expected sac, got {}",
            metadata.model_name
        );
    }
    if metadata.family != ModelFamily::Rl {
        bail!(
            "sac metadata family mismatch: expected {:?}, got {:?}",
            ModelFamily::Rl,
            metadata.family
        );
    }
    if metadata.state != CapabilityState::Implemented {
        bail!(
            "sac metadata state mismatch: expected {:?}, got {:?}",
            CapabilityState::Implemented,
            metadata.state
        );
    }
    if metadata.label_mapping != canonical_three_class_label_mapping() {
        bail!("sac metadata label mapping mismatch");
    }
    if expected_feature_columns.is_empty() {
        bail!("sac metadata validation requires non-empty feature columns");
    }
    if metadata.feature_columns != expected_feature_columns {
        bail!(
            "sac metadata feature-column mismatch: expected {:?}, got {:?}",
            expected_feature_columns,
            metadata.feature_columns
        );
    }
    if metadata.training_summary.dataset_rows != expected_dataset_rows {
        bail!(
            "sac metadata dataset-row mismatch: expected {}, got {}",
            expected_dataset_rows,
            metadata.training_summary.dataset_rows
        );
    }
    if metadata.training_summary.train_rows + metadata.training_summary.val_rows
        != metadata.training_summary.dataset_rows
    {
        bail!(
            "sac metadata rows are inconsistent: train_rows {} + val_rows {} != dataset_rows {}",
            metadata.training_summary.train_rows,
            metadata.training_summary.val_rows,
            metadata.training_summary.dataset_rows
        );
    }
    Ok(())
}

fn validate_sac_artifact(artifact: &SoftActorCriticArtifact) -> Result<()> {
    if artifact.state_dim == 0 {
        bail!("sac artifact state_dim must be positive");
    }
    if artifact.hidden_dim == 0 {
        bail!("sac artifact hidden_dim must be positive");
    }
    if artifact.feature_columns.is_empty() {
        bail!("sac artifact must contain feature columns");
    }
    if artifact.feature_columns.len() != artifact.state_dim {
        bail!(
            "sac artifact feature-column mismatch: state_dim {} vs {} feature columns",
            artifact.state_dim,
            artifact.feature_columns.len()
        );
    }
    if artifact.train_rows == 0 {
        bail!("sac artifact must record at least one training row");
    }
    if artifact.tuple_count == 0 {
        bail!("sac artifact must record at least one training tuple");
    }
    if !artifact.gamma.is_finite() || !(0.0..1.0).contains(&artifact.gamma) {
        bail!("sac artifact gamma must be finite and inside (0, 1)");
    }
    if !artifact.tau.is_finite() || !(0.0..=1.0).contains(&artifact.tau) {
        bail!("sac artifact tau must be finite and inside (0, 1]");
    }
    if !artifact.learning_rate.is_finite() || artifact.learning_rate <= 0.0 {
        bail!("sac artifact learning_rate must be finite and positive");
    }
    if !artifact.target_entropy_scale.is_finite() || artifact.target_entropy_scale <= 0.0 {
        bail!("sac artifact target_entropy_scale must be finite and positive");
    }
    if !artifact.target_entropy.is_finite() {
        bail!("sac artifact target_entropy must be finite");
    }
    if !artifact.init_log_alpha.is_finite() {
        bail!("sac artifact init_log_alpha must be finite");
    }
    if artifact.epochs == 0 {
        bail!("sac artifact epochs must be positive");
    }
    if artifact.batch_size == 0 {
        bail!("sac artifact batch_size must be positive");
    }
    if !artifact.final_alpha.is_finite() || artifact.final_alpha < 0.0 {
        bail!("sac artifact final_alpha must be finite and non-negative");
    }
    if let Some(scaler) = artifact.feature_scaler.as_ref() {
        if scaler.means.len() != artifact.state_dim || scaler.stds.len() != artifact.state_dim {
            bail!(
                "sac artifact feature_scaler mismatch: expected {} dims, got means {} / stds {}",
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
            bail!("sac artifact feature_scaler contains non-finite values");
        }
        if scaler.stds.iter().any(|value| *value <= f32::EPSILON) {
            bail!("sac artifact feature_scaler contains non-positive standard deviations");
        }
    }
    let runtime_fields = [
        artifact.requested_device_policy.as_deref(),
        artifact.effective_device_policy.as_deref(),
        artifact.execution_backend.as_deref(),
    ];
    let present = runtime_fields.iter().filter(|v| v.is_some()).count();
    if present != 0 && present != runtime_fields.len() {
        bail!(
            "sac artifact must persist requested_device_policy, effective_device_policy and execution_backend together"
        );
    }
    if let Some(report) = artifact.training_report.as_ref() {
        if report.train_rows != artifact.train_rows {
            bail!(
                "sac training report rows {} do not match artifact train_rows {}",
                report.train_rows,
                artifact.train_rows
            );
        }
        if report.tuple_count != artifact.tuple_count {
            bail!(
                "sac training report tuple_count {} does not match artifact tuple_count {}",
                report.tuple_count,
                artifact.tuple_count
            );
        }
        if report.state_dim != artifact.state_dim {
            bail!(
                "sac training report state_dim {} does not match artifact state_dim {}",
                report.state_dim,
                artifact.state_dim
            );
        }
        if (report.final_alpha - artifact.final_alpha).abs() > 1e-5 {
            bail!(
                "sac training report final_alpha {} does not match artifact final_alpha {}",
                report.final_alpha,
                artifact.final_alpha
            );
        }
        for value in [
            report.final_alpha,
            report.final_actor_loss,
            report.final_critic_loss,
            report.average_hold_reward,
            report.average_buy_reward,
            report.average_sell_reward,
        ] {
            if !value.is_finite() {
                bail!("sac training report contains non-finite statistics");
            }
        }
        if report.requested_device_policy.trim().is_empty()
            || report.effective_device_policy.trim().is_empty()
            || report.execution_backend.trim().is_empty()
        {
            bail!("sac training report must persist runtime identity");
        }
        if report.used_feature_scaler != artifact.feature_scaler.is_some() {
            bail!(
                "sac training report feature_scaler flag {} does not match artifact scaler presence {}",
                report.used_feature_scaler,
                artifact.feature_scaler.is_some()
            );
        }
    } else {
        bail!("sac trained artifacts must persist a training_report");
    }
    if present != runtime_fields.len() {
        bail!("sac trained artifacts must persist complete runtime identity fields");
    }
    Ok(())
}

fn resolve_sac_runtime_metadata(
    path: &Path,
    artifact: &SoftActorCriticArtifact,
) -> Result<RuntimeArtifactMetadata> {
    let metadata_path = path.join(METADATA_FILE_NAME);
    let reconstructed =
        sac_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
    validate_sac_metadata(
        &reconstructed,
        &artifact.feature_columns,
        artifact.train_rows,
    )?;
    match read_json::<RuntimeArtifactMetadata>(&metadata_path) {
        Ok(metadata) => {
            validate_sac_metadata(&metadata, &artifact.feature_columns, artifact.train_rows)?;
            if metadata.model_name != reconstructed.model_name
                || metadata.family != reconstructed.family
                || metadata.state != reconstructed.state
                || metadata.feature_columns != reconstructed.feature_columns
                || metadata.label_mapping != reconstructed.label_mapping
                || metadata.training_summary != reconstructed.training_summary
            {
                bail!(
                    "sac metadata sidecar mismatch with reconstructed runtime metadata at {}",
                    metadata_path.display()
                );
            }
            Ok(metadata)
        }
        Err(file_err) => {
            warn!(
                path = %metadata_path.display(),
                error = %file_err,
                "sac metadata sidecar missing/unreadable; using reconstructed runtime metadata from artifact"
            );
            Ok(reconstructed)
        }
    }
}

// ============================================================================
// SOFT ACTOR-CRITIC
// ============================================================================

type SacOptim<M> = OptimizerAdaptor<burn::optim::AdamW, M, TrainBackend>;

/// Pure-Rust Burn discrete Soft Actor-Critic agent.
pub struct SoftActorCritic {
    actor: SacActorNet<TrainBackend>,
    critic1: SacCriticNet<TrainBackend>,
    critic2: SacCriticNet<TrainBackend>,
    target_critic1: SacCriticNet<TrainBackend>,
    target_critic2: SacCriticNet<TrainBackend>,
    temperature: SacTemperature<TrainBackend>,
    actor_optim: SacOptim<SacActorNet<TrainBackend>>,
    critic1_optim: SacOptim<SacCriticNet<TrainBackend>>,
    critic2_optim: SacOptim<SacCriticNet<TrainBackend>>,
    alpha_optim: SacOptim<SacTemperature<TrainBackend>>,

    state_dim: usize,
    hidden_dim: usize,
    feature_columns: Vec<String>,
    feature_scaler: Option<FeatureScaler>,

    gamma: f32,
    tau: f32,
    learning_rate: f64,
    target_entropy_scale: f32,
    target_entropy: f32,
    init_log_alpha: f32,
    epochs: usize,
    batch_size: usize,
    reward_horizon: usize,
    episode_len: usize,

    train_rows: usize,
    tuple_count: usize,
    final_alpha: f32,
    training_report: Option<SacTrainingReport>,
    trained_checkpoint_ready: bool,

    device: <TrainBackend as BackendTypes>::Device,
    requested_device_policy: String,
    effective_device_policy: String,
    execution_backend: String,
    persisted_requested_device_policy: Option<String>,
    persisted_effective_device_policy: Option<String>,
    persisted_execution_backend: Option<String>,
}

impl SoftActorCritic {
    pub fn new(state_dim: usize) -> Self {
        Self::with_hidden_dim(state_dim, 256)
    }

    pub fn with_hidden_dim(state_dim: usize, hidden_dim: usize) -> Self {
        let state_dim = state_dim.max(1);
        let hidden_dim = hidden_dim.max(8);
        let (device, selection) = resolve_train_device("auto");
        let cfg = SacNetConfig::new()
            .with_input_dim(state_dim)
            .with_hidden_dim(hidden_dim);
        let actor = cfg.init_actor(&device);
        let critic1 = cfg.init_critic(&device);
        let critic2 = cfg.init_critic(&device);
        let temperature = SacTemperature::init(&device, 0.0);

        Self {
            target_critic1: critic1.clone(),
            target_critic2: critic2.clone(),
            actor_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            critic1_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            critic2_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            alpha_optim: AdamWConfig::new().with_weight_decay(0.0).init(),
            actor,
            critic1,
            critic2,
            temperature,
            state_dim,
            hidden_dim,
            feature_columns: Vec::new(),
            feature_scaler: None,
            gamma: 0.99,
            tau: 0.01,
            learning_rate: 3e-4,
            target_entropy_scale: 0.98,
            target_entropy: default_target_entropy(0.98),
            init_log_alpha: 0.0,
            epochs: 32,
            batch_size: 64,
            reward_horizon: 0,
            episode_len: 0,
            train_rows: 0,
            tuple_count: 0,
            final_alpha: 1.0,
            training_report: None,
            trained_checkpoint_ready: false,
            device,
            requested_device_policy: selection.requested_policy,
            effective_device_policy: selection.effective_policy,
            execution_backend: selection.execution_backend,
            persisted_requested_device_policy: None,
            persisted_effective_device_policy: None,
            persisted_execution_backend: None,
        }
    }

    /// Read-only view of the trained feature column names + ordering.
    /// Required by the [`crate::ensemble_inference::ExpertModel`] adapter.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        if gamma.is_finite() {
            self.gamma = gamma.clamp(0.01, 0.9999);
        }
        self
    }

    pub fn with_tau(mut self, tau: f32) -> Self {
        if tau.is_finite() {
            self.tau = tau.clamp(1e-4, 1.0);
        }
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        if learning_rate.is_finite() && learning_rate > 0.0 {
            self.learning_rate = learning_rate;
        }
        self
    }

    pub fn with_target_entropy_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 {
            self.target_entropy_scale = scale.clamp(0.01, 1.0);
            self.target_entropy = default_target_entropy(self.target_entropy_scale);
        }
        self
    }

    pub fn with_train_schedule(mut self, epochs: usize, batch_size: usize) -> Self {
        self.epochs = epochs.max(1);
        self.batch_size = batch_size.max(8);
        self
    }

    pub fn with_episode_layout(mut self, reward_horizon: usize, episode_len: usize) -> Self {
        self.reward_horizon = reward_horizon;
        self.episode_len = episode_len;
        self
    }

    fn preprocess_state(&self, state: &[f32]) -> Result<Vec<f32>> {
        if let Some(scaler) = self.feature_scaler.as_ref() {
            if state.len() != scaler.means.len() {
                bail!(
                    "sac runtime scaler dimension mismatch: expected {}, got {}",
                    scaler.means.len(),
                    state.len()
                );
            }
            let features = ndarray::Array2::from_shape_vec((1, state.len()), state.to_vec())
                .context("shape sac runtime state for scaling")?;
            let scaled = scaler.transform(&features)?;
            Ok(scaled.row(0).iter().copied().collect())
        } else {
            Ok(state.to_vec())
        }
    }

    /// Soft-update the target critics toward the live critics by `tau`
    /// (Polyak averaging). `θ_tgt ← τ·θ + (1−τ)·θ_tgt`.
    fn soft_update_targets(&mut self) {
        self.target_critic1 =
            polyak_update(self.target_critic1.clone(), &self.critic1, self.tau);
        self.target_critic2 =
            polyak_update(self.target_critic2.clone(), &self.critic2, self.tau);
    }

    /// Forward the full training batch and apply ONE SAC update step
    /// (critics + actor + temperature). Returns `(critic_loss,
    /// actor_loss, alpha)` scalars for diagnostics.
    fn update_on_batch(&mut self, batch: &[SacTuple]) -> Result<(f32, f32, f32)> {
        let batch_size = batch.len();
        if batch_size == 0 {
            bail!("sac update received an empty batch");
        }
        let device = self.device.clone();

        let mut states_flat = Vec::with_capacity(batch_size * self.state_dim);
        let mut next_states_flat = Vec::with_capacity(batch_size * self.state_dim);
        let mut rewards_flat = Vec::with_capacity(batch_size * NUM_ACTIONS);
        let mut not_done_flat = Vec::with_capacity(batch_size * NUM_ACTIONS);
        for tuple in batch {
            states_flat.extend_from_slice(&tuple.state);
            next_states_flat.extend_from_slice(&tuple.next_state);
            let mask = if tuple.done { 0.0_f32 } else { 1.0_f32 };
            for action in 0..NUM_ACTIONS {
                rewards_flat.push(tuple.rewards[action]);
                not_done_flat.push(mask);
            }
        }

        let states: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(states_flat, [batch_size, self.state_dim]),
            &device,
        );
        let next_states: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(next_states_flat, [batch_size, self.state_dim]),
            &device,
        );
        let rewards: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(rewards_flat, [batch_size, NUM_ACTIONS]),
            &device,
        );
        let not_done: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(not_done_flat, [batch_size, NUM_ACTIONS]),
            &device,
        );

        let alpha = self.temperature.alpha_value(&device)?;

        // ---- Soft state value of next state V(s') ----
        // V(s') = Σ_a p'(a)·[ min(Q1ₜ,Q2ₜ)(s',a) − α·logp'(a) ]
        let next_logits = self.actor.forward(next_states.clone());
        let next_probs = burn::tensor::activation::softmax(next_logits.clone(), 1);
        let next_log_probs = burn::tensor::activation::log_softmax(next_logits, 1);
        let next_q1 = self.target_critic1.forward(next_states.clone());
        let next_q2 = self.target_critic2.forward(next_states);
        let next_qmin = tensor_min(next_q1, next_q2);
        // soft per-action value of s': Qmin − α·logp'
        let soft_next = next_qmin - next_log_probs.clone().mul_scalar(alpha);
        // V(s') broadcast back to all actions: Σ_a p'(a)·soft_next, shape [batch,1] → [batch,3]
        let next_v = (next_probs * soft_next)
            .sum_dim(1)
            .reshape([batch_size, 1]);

        // ---- Critic targets (all actions, full information) ----
        // y(s,a) = r(s,a) + γ·(1−done)·V(s')   — stop-gradient
        let gamma = self.gamma;
        let next_v_broadcast = next_v.repeat_dim(1, NUM_ACTIONS);
        let targets = (rewards + not_done * next_v_broadcast.mul_scalar(gamma)).detach();

        // ---- Critic loss & update ----
        let q1_pred = self.critic1.forward(states.clone());
        let critic1_loss = burn::nn::loss::MseLoss::new().forward(
            q1_pred,
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );
        let critic1_loss_value =
            scalar_from_tensor(critic1_loss.clone(), "sac critic1 loss")?;
        let grads = critic1_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.critic1);
        self.critic1 = self
            .critic1_optim
            .step(self.learning_rate, self.critic1.clone(), grads_params);

        let q2_pred = self.critic2.forward(states.clone());
        let critic2_loss = burn::nn::loss::MseLoss::new().forward(
            q2_pred,
            targets,
            burn::nn::loss::Reduction::Mean,
        );
        let critic2_loss_value =
            scalar_from_tensor(critic2_loss.clone(), "sac critic2 loss")?;
        let grads = critic2_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.critic2);
        self.critic2 = self
            .critic2_optim
            .step(self.learning_rate, self.critic2.clone(), grads_params);

        // ---- Actor loss & update ----
        // L_π = Σ_a p(a)·[ α·logp(a) − min(Q1,Q2)(s,a) ]  (Q detached)
        let logits = self.actor.forward(states.clone());
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);
        let q1 = self.critic1.forward(states.clone()).detach();
        let q2 = self.critic2.forward(states).detach();
        let qmin = tensor_min(q1, q2);
        let actor_objective = log_probs.clone().mul_scalar(alpha) - qmin;
        // mean over batch of Σ_a p(a)·objective(a)
        let actor_loss =
            (probs.clone() * actor_objective).sum() / (batch_size as f32);
        let actor_loss_value = scalar_from_tensor(actor_loss.clone(), "sac actor loss")?;
        let grads = actor_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.actor);
        self.actor = self
            .actor_optim
            .step(self.learning_rate, self.actor.clone(), grads_params);

        // ---- Temperature (entropy) loss & update ----
        // L_α = − Σ_a p(a)·log_alpha·( logp(a).detach + H )
        // policy terms are detached (only log_alpha is optimized here).
        let probs_d = probs.detach();
        let log_probs_d = log_probs.detach();
        let entropy_gap = (log_probs_d + self.target_entropy) * probs_d;
        // Σ_a entropy_gap(a) per row → [batch, 1]
        let entropy_gap_sum = entropy_gap.sum_dim(1).reshape([batch_size, 1]);
        let log_alpha = self.temperature.log_alpha(&device); // [1,1], differentiable
        let log_alpha_broadcast = log_alpha.repeat_dim(0, batch_size);
        // mean over batch of  −log_alpha·entropy_gap_sum
        let alpha_loss =
            ((log_alpha_broadcast * entropy_gap_sum).sum() / (batch_size as f32)).neg();
        let grads = alpha_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.temperature);
        self.temperature =
            self.alpha_optim
                .step(self.learning_rate, self.temperature.clone(), grads_params);

        // ---- Polyak target update ----
        self.soft_update_targets();

        let critic_loss_value = critic1_loss_value + critic2_loss_value;
        Ok((critic_loss_value, actor_loss_value, alpha))
    }

    /// Train from a feature DataFrame + label Series (the standard
    /// orchestrator entry point). Builds episodes via the shared RL
    /// builder, scales features, and runs `epochs` passes of
    /// mini-batched SAC updates.
    pub fn train_on_frame(&mut self, x: &DataFrame, y: &Series) -> Result<SacTrainingReport> {
        let (features, feature_columns) = feature_matrix_from_dataframe(x)?;
        if features.ncols() != self.state_dim {
            bail!(
                "sac feature mismatch: configured state_dim {} vs dataframe {}",
                self.state_dim,
                features.ncols()
            );
        }
        if feature_columns.is_empty() {
            bail!("sac requires at least one feature column");
        }
        if features.nrows() < 48 {
            bail!("sac requires at least 48 rows, received {}", features.nrows());
        }
        let scaler = FeatureScaler::fit(&features)?;
        let scaled = scaler.transform(&features)?;
        let labels = remap_three_class_labels(y)?;
        if scaled.nrows() != labels.len() {
            bail!(
                "sac row mismatch: {} feature rows vs {} labels",
                scaled.nrows(),
                labels.len()
            );
        }

        let horizon = if self.reward_horizon > 0 {
            self.reward_horizon.clamp(2, 128)
        } else {
            (scaled.nrows() / 24).clamp(6, 32)
        };
        let episode_len = if self.episode_len > 0 {
            self.episode_len.clamp(horizon.max(8), 512)
        } else {
            (scaled.nrows() / 12).clamp(24, 128)
        };
        let episodes = build_training_episodes_public(&scaled, &labels, episode_len, horizon)?;
        let tuples = tuples_from_episodes(&episodes, self.state_dim)?;

        self.feature_columns = feature_columns;
        self.feature_scaler = Some(scaler);
        self.reward_horizon = horizon;
        self.episode_len = episode_len;
        self.train_rows = scaled.nrows();
        self.tuple_count = tuples.len();
        self.target_entropy = default_target_entropy(self.target_entropy_scale);

        // Reset temperature/targets to a clean trainable state so a
        // re-train always starts from the configured init_log_alpha.
        self.temperature = SacTemperature::init(&self.device, self.init_log_alpha);
        self.alpha_optim = AdamWConfig::new().with_weight_decay(0.0).init();
        self.target_critic1 = self.critic1.clone();
        self.target_critic2 = self.critic2.clone();

        let batch_size = self.batch_size.min(tuples.len()).max(1);
        let mut final_critic_loss = 0.0_f32;
        let mut final_actor_loss = 0.0_f32;
        let mut final_alpha = 1.0_f32;
        let mut batches = 0usize;

        for epoch in 0..self.epochs {
            // Deterministic, dependency-free chunking (no RNG): rotate
            // the start offset each epoch so batches differ but the
            // schedule stays reproducible.
            let offset = (epoch * batch_size / 2) % tuples.len().max(1);
            let mut idx = offset;
            let mut seen = 0usize;
            while seen < tuples.len() {
                let mut batch = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    batch.push(tuples[idx % tuples.len()].clone());
                    idx += 1;
                    seen += 1;
                    if seen >= tuples.len() {
                        break;
                    }
                }
                let (critic_loss, actor_loss, alpha) = self.update_on_batch(&batch)?;
                if !critic_loss.is_finite() || !actor_loss.is_finite() || !alpha.is_finite() {
                    bail!(
                        "sac training diverged (non-finite loss): critic={critic_loss}, actor={actor_loss}, alpha={alpha}"
                    );
                }
                final_critic_loss = critic_loss;
                final_actor_loss = actor_loss;
                final_alpha = alpha;
                batches += 1;
            }
        }

        let (hold_avg, buy_avg, sell_avg) = average_rewards(&tuples);
        self.final_alpha = final_alpha;
        self.trained_checkpoint_ready = true;

        let report = SacTrainingReport {
            train_rows: self.train_rows,
            tuple_count: self.tuple_count,
            state_dim: self.state_dim,
            epochs: self.epochs,
            batches,
            reward_horizon: self.reward_horizon,
            episode_len: self.episode_len,
            final_alpha,
            final_critic_loss,
            final_actor_loss,
            average_hold_reward: hold_avg,
            average_buy_reward: buy_avg,
            average_sell_reward: sell_avg,
            requested_device_policy: self.requested_device_policy.clone(),
            effective_device_policy: self.effective_device_policy.clone(),
            execution_backend: self.execution_backend.clone(),
            used_feature_scaler: self.feature_scaler.is_some(),
        };
        self.training_report = Some(report.clone());

        info!(
            "trained SAC-discrete agent (rows={}, tuples={}, batches={}, final_alpha={:.4}, critic_loss={:.5}, actor_loss={:.5})",
            self.train_rows, self.tuple_count, batches, final_alpha, final_critic_loss, final_actor_loss
        );
        Ok(report)
    }

    fn ensure_runtime_ready(&self) -> Result<()> {
        if !self.trained_checkpoint_ready {
            bail!("sac cannot run inference from an untrained runtime state");
        }
        if self.feature_columns.is_empty() || self.feature_columns.len() != self.state_dim {
            bail!("sac cannot run inference without a persisted feature schema");
        }
        if self.training_report.is_none() {
            bail!("sac cannot run inference without a persisted training report");
        }
        if self.requested_device_policy.trim().is_empty()
            || self.effective_device_policy.trim().is_empty()
            || self.execution_backend.trim().is_empty()
        {
            bail!("sac cannot run inference without complete runtime identity");
        }
        Ok(())
    }

    fn runtime_degraded_reason(&self) -> Option<String> {
        let mut reasons = Vec::new();
        if let Some(persisted) = self.persisted_requested_device_policy.as_ref()
            && persisted != &self.requested_device_policy
        {
            reasons.push(format!(
                "persisted requested device `{persisted}` differs from current runtime `{}`",
                self.requested_device_policy
            ));
        }
        if let Some(persisted) = self.persisted_effective_device_policy.as_ref()
            && persisted != &self.effective_device_policy
        {
            reasons.push(format!(
                "persisted effective device `{persisted}` differs from current runtime `{}`",
                self.effective_device_policy
            ));
        }
        if let Some(persisted) = self.persisted_execution_backend.as_ref()
            && persisted != &self.execution_backend
        {
            reasons.push(format!(
                "persisted execution backend `{persisted}` differs from current runtime `{}`",
                self.execution_backend
            ));
        }
        if self.requested_device_policy != self.effective_device_policy {
            reasons.push(format!(
                "requested Burn device `{}` resolved to `{}` on the current build/runtime",
                self.requested_device_policy, self.effective_device_policy
            ));
        }
        if reasons.is_empty() {
            None
        } else {
            Some(reasons.join("; "))
        }
    }

    /// Policy probabilities `[p_hold, p_buy, p_sell]` for a single state.
    ///
    /// Maps directly to the canonical 3-class `[neutral, buy, sell]`
    /// ordering (Hold == neutral), matching the DQN entry voter.
    pub fn policy_probabilities(&self, state: &[f32]) -> Result<[f32; NUM_ACTIONS]> {
        if state.len() != self.state_dim {
            bail!(
                "sac policy state dimension mismatch: expected {}, got {}",
                self.state_dim,
                state.len()
            );
        }
        let scaled = self.preprocess_state(state)?;
        let state_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(scaled, [self.state_dim]),
            &self.device,
        )
        .unsqueeze::<2>();
        let probs = self
            .actor
            .policy(state_tensor)
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("extract sac policy probabilities: {err:?}"))?;
        if probs.len() != NUM_ACTIONS {
            bail!(
                "sac policy returned {} probabilities, expected {NUM_ACTIONS}",
                probs.len()
            );
        }
        let mut out = [0.0_f32; NUM_ACTIONS];
        let mut total = 0.0_f32;
        for (idx, value) in probs.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                bail!("sac policy returned an invalid probability {value}");
            }
            out[idx] = value;
            total += value;
        }
        if !total.is_finite() || total <= f32::EPSILON {
            bail!("sac policy returned degenerate probability mass");
        }
        for value in &mut out {
            *value /= total;
        }
        Ok(out)
    }

    pub fn predict_runtime(&self, x: &DataFrame) -> Result<Vec<RuntimePrediction>> {
        self.ensure_runtime_ready()?;
        let (features, columns) = feature_matrix_from_dataframe(x)?;
        if features.ncols() != self.state_dim {
            bail!(
                "sac prediction feature mismatch: configured state_dim {} vs dataframe {}",
                self.state_dim,
                features.ncols()
            );
        }
        if !self.feature_columns.is_empty() && self.feature_columns != columns {
            bail!(
                "sac prediction feature-column mismatch: expected {:?}, got {:?}",
                self.feature_columns,
                columns
            );
        }

        let degraded_reason = self.runtime_degraded_reason();
        let mut predictions = Vec::with_capacity(features.nrows());
        for row in features.outer_iter() {
            let state = row.iter().copied().collect::<Vec<_>>();
            let probabilities = self.policy_probabilities(&state)?;
            let (confidence, abstain) = three_class_runtime_confidence(probabilities)?;
            predictions.push(build_runtime_prediction_with_details(
                "sac",
                ModelFamily::Rl,
                CapabilityState::Implemented,
                probabilities,
                Some(confidence),
                Some(abstain),
                Some(self.execution_backend.clone()),
                degraded_reason.clone(),
            )?);
        }
        Ok(predictions)
    }

    fn artifact(&self) -> SoftActorCriticArtifact {
        SoftActorCriticArtifact {
            state_dim: self.state_dim,
            hidden_dim: self.hidden_dim,
            feature_columns: self.feature_columns.clone(),
            gamma: self.gamma,
            tau: self.tau,
            learning_rate: self.learning_rate,
            target_entropy_scale: self.target_entropy_scale,
            target_entropy: self.target_entropy,
            init_log_alpha: self.init_log_alpha,
            epochs: self.epochs,
            batch_size: self.batch_size,
            reward_horizon: self.reward_horizon,
            episode_len: self.episode_len,
            train_rows: self.train_rows,
            tuple_count: self.tuple_count,
            final_alpha: self.final_alpha,
            feature_scaler: self.feature_scaler.clone(),
            training_report: self.training_report.clone(),
            requested_device_policy: Some(self.requested_device_policy.clone()),
            effective_device_policy: Some(self.effective_device_policy.clone()),
            execution_backend: Some(self.execution_backend.clone()),
            runtime_metadata: None,
        }
    }

    // ---- atomic staged artifact save / load ----

    fn staged_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("tmp_artifact")
    }

    fn backup_artifact_dir(path: &Path) -> PathBuf {
        path.with_extension("bak_artifact")
    }

    fn cleanup_artifact_dir(path: &Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("remove staged sac artifact {}", path.display()))?;
        }
        Ok(())
    }

    fn replace_artifact_dir(staged_path: &Path, target_path: &Path) -> Result<()> {
        let backup_path = Self::backup_artifact_dir(target_path);
        Self::cleanup_artifact_dir(&backup_path)?;
        if target_path.exists() {
            std::fs::rename(target_path, &backup_path).with_context(|| {
                format!(
                    "move previous sac artifact into backup {}",
                    backup_path.display()
                )
            })?;
        }
        if let Err(error) = std::fs::rename(staged_path, target_path) {
            if backup_path.exists() {
                if let Err(restore_err) = std::fs::rename(&backup_path, target_path) {
                    tracing::error!(
                        target: "neoethos_models::artifact",
                        backup = %backup_path.display(),
                        target = %target_path.display(),
                        error = %restore_err,
                        "failed to restore backup after staged-rename failure; sac artifact directory may be in an inconsistent state"
                    );
                }
            }
            bail!(
                "rename staged sac artifact into {} failed: {}",
                target_path.display(),
                error
            );
        }
        Self::cleanup_artifact_dir(&backup_path)?;
        Ok(())
    }

    fn actor_record_base(path: &Path) -> PathBuf {
        path.join("actor")
    }

    fn critic1_record_base(path: &Path) -> PathBuf {
        path.join("critic1")
    }

    fn critic2_record_base(path: &Path) -> PathBuf {
        path.join("critic2")
    }

    fn temperature_record_base(path: &Path) -> PathBuf {
        path.join("temperature")
    }

    fn artifact_path(path: &Path) -> PathBuf {
        path.join("config.json")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.trained_checkpoint_ready {
            bail!("sac cannot persist from an untrained runtime state");
        }
        let artifact = self.artifact();
        validate_sac_artifact(&artifact)?;
        let staged_path = Self::staged_artifact_dir(path);
        Self::cleanup_artifact_dir(&staged_path)?;
        std::fs::create_dir_all(&staged_path)
            .with_context(|| format!("create staged sac directory {}", staged_path.display()))?;

        if let Err(error) = (|| -> Result<()> {
            let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            self.actor
                .clone()
                .valid()
                .save_file(Self::actor_record_base(&staged_path), &recorder)
                .with_context(|| format!("persist sac actor to {}", staged_path.display()))?;
            self.critic1
                .clone()
                .valid()
                .save_file(Self::critic1_record_base(&staged_path), &recorder)
                .with_context(|| format!("persist sac critic1 to {}", staged_path.display()))?;
            self.critic2
                .clone()
                .valid()
                .save_file(Self::critic2_record_base(&staged_path), &recorder)
                .with_context(|| format!("persist sac critic2 to {}", staged_path.display()))?;
            self.temperature
                .clone()
                .valid()
                .save_file(Self::temperature_record_base(&staged_path), &recorder)
                .with_context(|| {
                    format!("persist sac temperature to {}", staged_path.display())
                })?;

            let runtime_metadata =
                sac_runtime_metadata(artifact.feature_columns.clone(), artifact.train_rows)?;
            validate_sac_metadata(
                &runtime_metadata,
                &artifact.feature_columns,
                artifact.train_rows,
            )?;
            let mut artifact_to_write = artifact.clone();
            artifact_to_write.runtime_metadata = Some(runtime_metadata.clone());
            write_json(&staged_path.join(METADATA_FILE_NAME), &runtime_metadata)?;
            write_json(&Self::artifact_path(&staged_path), &artifact_to_write)
                .with_context(|| format!("write sac config to {}", staged_path.display()))?;
            Ok(())
        })() {
            let _ = Self::cleanup_artifact_dir(&staged_path);
            return Err(error);
        }
        Self::replace_artifact_dir(&staged_path, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let artifact: SoftActorCriticArtifact = read_json(&Self::artifact_path(path))
            .with_context(|| format!("read sac config from {}", path.display()))?;
        validate_sac_artifact(&artifact)?;
        let _metadata = resolve_sac_runtime_metadata(path, &artifact)?;

        let requested_device_policy = artifact
            .requested_device_policy
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let persisted_requested_device_policy = artifact.requested_device_policy.clone();
        let persisted_effective_device_policy = artifact.effective_device_policy.clone();
        let persisted_execution_backend = artifact.execution_backend.clone();
        let (device, selection) = resolve_train_device(&requested_device_policy);
        if let Some(persisted) = persisted_effective_device_policy.as_deref()
            && persisted != selection.effective_policy
        {
            warn!(
                "sac persisted effective device policy `{}` differs from current runtime `{}` while loading {}",
                persisted,
                selection.effective_policy,
                path.display()
            );
        }

        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        let cfg = SacNetConfig::new()
            .with_input_dim(artifact.state_dim)
            .with_hidden_dim(artifact.hidden_dim);
        let actor = cfg
            .init_actor(&device)
            .load_file(Self::actor_record_base(path), &recorder, &device)
            .with_context(|| format!("load sac actor from {}", path.display()))?;
        let critic1 = cfg
            .init_critic(&device)
            .load_file(Self::critic1_record_base(path), &recorder, &device)
            .with_context(|| format!("load sac critic1 from {}", path.display()))?;
        let critic2 = cfg
            .init_critic(&device)
            .load_file(Self::critic2_record_base(path), &recorder, &device)
            .with_context(|| format!("load sac critic2 from {}", path.display()))?;
        let temperature = SacTemperature::init(&device, artifact.init_log_alpha)
            .load_file(Self::temperature_record_base(path), &recorder, &device)
            .with_context(|| format!("load sac temperature from {}", path.display()))?;

        Ok(Self {
            target_critic1: critic1.clone(),
            target_critic2: critic2.clone(),
            actor_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            critic1_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            critic2_optim: AdamWConfig::new().with_weight_decay(1e-4).init(),
            alpha_optim: AdamWConfig::new().with_weight_decay(0.0).init(),
            actor,
            critic1,
            critic2,
            temperature,
            state_dim: artifact.state_dim,
            hidden_dim: artifact.hidden_dim,
            feature_columns: artifact.feature_columns,
            feature_scaler: artifact.feature_scaler,
            gamma: artifact.gamma,
            tau: artifact.tau,
            learning_rate: artifact.learning_rate,
            target_entropy_scale: artifact.target_entropy_scale,
            target_entropy: artifact.target_entropy,
            init_log_alpha: artifact.init_log_alpha,
            epochs: artifact.epochs,
            batch_size: artifact.batch_size,
            reward_horizon: artifact.reward_horizon,
            episode_len: artifact.episode_len,
            train_rows: artifact.train_rows,
            tuple_count: artifact.tuple_count,
            final_alpha: artifact.final_alpha,
            training_report: artifact.training_report,
            trained_checkpoint_ready: true,
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

// ============================================================================
// HELPERS
// ============================================================================

/// Elementwise `min(a, b)` for two same-shape tensors.
fn tensor_min<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    // min(a,b) = a − relu(a − b)
    let diff = a.clone() - b;
    a - burn::tensor::activation::relu(diff)
}

/// Polyak (soft) update: `dst ← τ·src + (1−τ)·dst`, applied to every
/// float parameter tensor of the critic module.
fn polyak_update<B: Backend>(
    dst: SacCriticNet<B>,
    src: &SacCriticNet<B>,
    tau: f32,
) -> SacCriticNet<B> {
    let SacCriticNet { fc1, fc2, head } = dst;
    SacCriticNet {
        fc1: blend_linear(fc1, &src.fc1, tau),
        fc2: blend_linear(fc2, &src.fc2, tau),
        head: blend_linear(head, &src.head, tau),
    }
}

fn blend_linear<B: Backend>(dst: nn::Linear<B>, src: &nn::Linear<B>, tau: f32) -> nn::Linear<B> {
    let mut out = dst;
    // detach so the target network is a non-differentiable copy.
    let new_weight =
        out.weight.val().mul_scalar(1.0 - tau) + src.weight.val().mul_scalar(tau);
    out.weight = burn::module::Param::from_tensor(new_weight.detach());
    if let (Some(dst_bias), Some(src_bias)) = (out.bias.take(), src.bias.as_ref()) {
        let new_bias = dst_bias.val().mul_scalar(1.0 - tau) + src_bias.val().mul_scalar(tau);
        out.bias = Some(burn::module::Param::from_tensor(new_bias.detach()));
    }
    out
}

fn scalar_from_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    context: &str,
) -> Result<f32> {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("{context}: read scalar tensor: {err:?}"))?;
    let value = values
        .first()
        .copied()
        .with_context(|| format!("{context}: scalar tensor is empty"))?;
    if !value.is_finite() {
        bail!("{context}: scalar tensor contained non-finite value {value}");
    }
    Ok(value)
}

fn average_rewards(tuples: &[SacTuple]) -> (f32, f32, f32) {
    if tuples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sums = [0.0_f32; NUM_ACTIONS];
    for tuple in tuples {
        for action in 0..NUM_ACTIONS {
            sums[action] += tuple.rewards[action];
        }
    }
    let denom = tuples.len() as f32;
    (sums[0] / denom, sums[1] / denom, sums[2] / denom)
}

#[cfg(test)]
#[path = "soft_actor_critic_tests.rs"]
mod tests;
