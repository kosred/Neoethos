// Base classes and utilities (ported from models/base.py)
pub mod base;
pub mod runtime;

// Machine learning models
pub mod deep_models;
pub mod ensemble;
pub mod parallel_trainer;
pub mod training_orchestrator;
pub mod tree_models;

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

// Hardware detection (ported from models/device.py)
pub mod hardware;

// ONNX export for ultra-fast inference
pub mod onnx_exporter;

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

// Pure-Rust neural networks via Burn framework (no Python, no GIL)
pub mod burn_models;

#[cfg(feature = "onnx")]
use anyhow::{Context, Result};
#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{Session, Value, inputs};
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use tracing::{info, warn};

#[cfg(feature = "onnx")]
pub struct ONNXInferenceEngine {
    sessions: HashMap<String, Session>,
    model_outputs: HashMap<String, String>,
}

#[cfg(feature = "onnx")]
impl ONNXInferenceEngine {
    pub fn new() -> Result<Self> {
        // ort 2.x init
        if let Err(e) = ort::init()
            .with_name("forex_models_ort")
            .with_execution_providers([
                ort::ExecutionProvider::CUDA(Default::default()),
                ort::ExecutionProvider::CPU(Default::default()),
            ])
            .commit()
        {
            warn!("ORT Init (or check): {}", e);
        }

        Ok(Self {
            sessions: HashMap::new(),
            model_outputs: HashMap::new(),
        })
    }

    pub fn load_models(&mut self, models_dir: impl AsRef<Path>) -> Result<()> {
        let models_dir = models_dir.as_ref();
        if !models_dir.exists() {
            warn!("Models directory not found: {:?}", models_dir);
            return Ok(());
        }

        let onnx_dir = models_dir.join("onnx");
        if !onnx_dir.exists() {
            warn!("ONNX directory not found: {:?}", onnx_dir);
            return Ok(());
        }

        for entry in std::fs::read_dir(onnx_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "onnx") {
                let name = path.file_stem().unwrap().to_string_lossy().to_string();
                if let Err(e) = self.load_model(&name, &path) {
                    warn!("Failed to load model {}: {}", name, e);
                }
            }
        }

        Ok(())
    }

    pub fn load_model(&mut self, name: &str, path: &Path) -> Result<()> {
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_inter_threads(4)?
            .commit_from_file(path)
            .context(format!("Failed to load model {}", name))?;

        let outputs = &session.outputs;
        let mut proba_output_name = String::new();

        for out in outputs {
            if out.name.to_lowercase().contains("prob") {
                proba_output_name = out.name.clone();
                break;
            }
        }
        if proba_output_name.is_empty() && !outputs.is_empty() {
            proba_output_name = outputs.last().unwrap().name.clone();
        }

        self.sessions.insert(name.to_string(), session);
        self.model_outputs
            .insert(name.to_string(), proba_output_name);
        info!("Loaded ONNX model: {}", name);
        Ok(())
    }

    pub fn predict_proba(&self, model_name: &str, features: &Array2<f32>) -> Result<Array2<f32>> {
        let session = self
            .sessions
            .get(model_name)
            .context(format!("Model {} not loaded", model_name))?;

        let input_tensor = Value::from_array(features.clone())?;
        let outputs = session.run(ort::inputs![input_tensor]?)?;

        let output_name = self
            .model_outputs
            .get(model_name)
            .context("Output name not found")?;
        let output_value = outputs.get(output_name).context("Output tensor missing")?;
        let output_tensor = output_value.try_extract_tensor::<f32>()?;
        let output_array = output_tensor.into_owned();

        let shape = output_array.shape();
        if shape.len() == 1 {
            Ok(output_array.into_shape((shape[0], 1))?)
        } else {
            Ok(output_array.into_dimensionality()?)
        }
    }
}
