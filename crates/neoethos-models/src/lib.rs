// **2026-05-25 — gpu-vulkan build fix**: burn-wgpu + burn-cubecl pull
// in `wgpu_hal::dynamic::DynShaderModule` and `naga::ir::ImageClass`
// through deeply nested generics. The default 128 trait-resolution
// recursion limit overflows when verifying `Sync` bounds on the
// ensemble-inference adapters. 512 covers the deepest chains we have
// without measurably slowing the type-checker. Standard fix per the
// rustc error E0275 documentation.
#![recursion_limit = "512"]

// Base classes and utilities (derived from models/base.py)
pub mod base;
pub mod common;
pub mod runtime;

// Machine learning models
pub mod deep_models;
pub mod ensemble;
pub mod ensemble_inference;
pub mod parallel_trainer;
pub mod training_orchestrator;
pub mod tree_models;

pub use ensemble_inference::{
    DEFAULT_BOOTSTRAP_EXPERT_NAMES, EnsemblePredictor, ExpertLoadError, ExpertLoadOutcome,
    ExpertLoader, ExpertModel, ExpertOutputKind, ExpertPrediction, ExpertRegistry,
    SoftVotingEnsemble, SoftVotingEnsembleConfig, build_default_registry,
    build_ensemble_for_symbol, build_ensemble_for_symbol_with_config, load_experts_for_symbol,
};

pub use deep_models::{
    BurnDeepExpert, KANExpert, MLPExpert, NBeatsExpert, NBeatsxNfExpert, PatchTSTExpert,
    TabNetExpert, TiDEExpert, TiDENfExpert, TimesNetExpert, TransformerExpert,
};
pub use ensemble::{
    CalibrationMethod, ConformalGate, ConformalPredictionExpert, MetaBlender, MetaDecisionStack,
    ProbabilityCalibrationExpert, ProbabilityCalibrator,
};
pub use parallel_trainer::{ModelTrainingFailure, ModelTrainingProgress, ParallelTrainingSummary};
pub use training_orchestrator::{TrainingOrchestrator, TrainingRunSummary, set_training_cancel};

// Hardware detection (derived from models/device.py)
pub mod hardware;

// Evaluation helpers (simple backtest, signal conversion)
pub mod evaluation_helpers;

// Model registry (model discovery and validation)
pub mod registry;

// Genetic strategy expert (evolutionary algorithms over the feature stack)
pub mod genetic;

// Exit agent (RL-based trade exit decisions)
pub mod exit_agent;

// Soft Actor-Critic (discrete) — RL entry/direction policy
pub mod soft_actor_critic;

// Pure Rust ML Modular Modules
pub mod anomaly;
pub mod evolution;
pub mod forecasting;
pub mod rl;
pub mod statistical;
pub mod streaming;

pub use anomaly::IsolationForestExpert;
pub use evolution::{NeatExpert, NeuroEvoExpert, NeuroEvoOptimizer};
pub use forecasting::{
    SwarmEnsembleStrategy, SwarmForecastConfig, SwarmForecastResult, SwarmForecaster,
};
pub use genetic::GeneticStrategyExpert;
pub use rl::{
    TradingAction, TradingEpisode, TradingReinforcementLearner, TradingStateEncoding,
    TradingTransition,
};
pub use soft_actor_critic::{SoftActorCritic, SoftActorCriticArtifact};
pub use statistical::{BayesianLogitExpert, ElasticNetExpert, LogisticExpert};
pub use streaming::{
    AdaptiveGradientBooster, OnlineHoeffdingExpert, OnlinePassiveAggressiveExpert,
};

// Pure-Rust neural networks via Burn framework (no legacy, no GIL)
pub mod burn_models;

#[cfg(feature = "onnx")]
pub use runtime::onnx::ONNXInferenceEngine;
