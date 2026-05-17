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
    EnsemblePredictor, ExpertLoadError, ExpertLoadOutcome, ExpertLoader, ExpertModel,
    ExpertOutputKind, ExpertPrediction, ExpertRegistry, SoftVotingEnsemble,
    SoftVotingEnsembleConfig,
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
pub use training_orchestrator::{TrainingOrchestrator, TrainingRunSummary};

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
pub use statistical::{BayesianLogitExpert, ElasticNetExpert, LogisticExpert};
pub use streaming::{
    AdaptiveGradientBooster, OnlineHoeffdingExpert, OnlinePassiveAggressiveExpert,
};

// Pure-Rust neural networks via Burn framework (no legacy, no GIL)
pub mod burn_models;

#[cfg(feature = "onnx")]
pub use runtime::onnx::ONNXInferenceEngine;
