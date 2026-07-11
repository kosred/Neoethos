// Parallel Model Trainer - multi-core training for Rust-native workloads.
// The runtime now treats every active family as a native or self-contained path.
use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::{Column, DataFrame, NamedFrom, Series};
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tracing::info;

use crate::base::dataframe_to_float32_array;
use crate::runtime::capabilities::{CapabilityState, ModelFamily};

fn rust_threads_hint() -> usize {
    // Shares the single config-driven CPU budget with tree-model training
    // (core hardware knob -> RAYON_NUM_THREADS -> cores-1).
    crate::tree_models::config::cpu_threads_hint()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTrainingFailure {
    pub name: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelTrainingProgress {
    Started {
        model: String,
        total_models: usize,
    },
    Succeeded {
        model: String,
        completed_models: usize,
        failed_models: usize,
        total_models: usize,
    },
    Failed {
        model: String,
        error: String,
        completed_models: usize,
        failed_models: usize,
        total_models: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelTrainingSummary {
    pub total_models: usize,
    pub successful_models: Vec<String>,
    pub failed_models: Vec<ModelTrainingFailure>,
}

#[derive(Debug, Clone)]
pub struct TrainingPayload {
    pub frame: Arc<DataFrame>,
    pub dense_features: Arc<Array2<f32>>,
    pub labels: Arc<Vec<i32>>,
}

impl TrainingPayload {
    pub fn from_frame(frame: DataFrame, labels: Vec<i32>) -> Result<Self> {
        if frame.height() != labels.len() {
            anyhow::bail!(
                "training payload row/label mismatch: {} rows vs {} labels",
                frame.height(),
                labels.len()
            );
        }

        let dense_features = dataframe_to_float32_array(&frame)
            .context("build dense feature matrix from training dataframe")?;

        Ok(Self {
            frame: Arc::new(frame),
            dense_features: Arc::new(dense_features),
            labels: Arc::new(labels),
        })
    }

    pub fn from_dense(dense_features: Array2<f32>, labels: Vec<i32>) -> Result<Self> {
        let feature_names = (0..dense_features.ncols())
            .map(|column_idx| format!("feature_{column_idx}"))
            .collect::<Vec<_>>();

        Self::from_named_dense(dense_features, labels, feature_names)
    }

    pub fn from_named_dense(
        dense_features: Array2<f32>,
        labels: Vec<i32>,
        feature_names: Vec<String>,
    ) -> Result<Self> {
        if dense_features.nrows() != labels.len() {
            anyhow::bail!(
                "training payload row/label mismatch: {} rows vs {} labels",
                dense_features.nrows(),
                labels.len()
            );
        }

        if dense_features.ncols() != feature_names.len() {
            anyhow::bail!(
                "training payload feature-name mismatch: {} cols vs {} names",
                dense_features.ncols(),
                feature_names.len()
            );
        }

        let columns = feature_names
            .into_iter()
            .enumerate()
            .map(|(column_idx, column_name)| {
                let values = dense_features
                    .column(column_idx)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                Column::from(Series::new(column_name.into(), values))
            })
            .collect::<Vec<_>>();

        let frame = DataFrame::new(columns).context("build dataframe from dense features")?;

        Ok(Self {
            frame: Arc::new(frame),
            dense_features: Arc::new(dense_features),
            labels: Arc::new(labels),
        })
    }
}

/// Train multiple models in parallel using a bounded Rayon thread pool.
pub fn train_models_parallel<F>(
    model_configs: Vec<ModelConfig>,
    payload: Arc<TrainingPayload>,
    train_fn: F,
) -> Result<Vec<String>>
where
    F: Fn(&ModelConfig, &TrainingPayload) -> Result<()> + Send + Sync + Clone + 'static,
{
    let summary = train_models_parallel_with_progress(model_configs, payload, |_| {}, train_fn)?;

    if !summary.failed_models.is_empty() {
        anyhow::bail!(
            "Parallel training failed for [{}]; successful models: [{}]",
            summary
                .failed_models
                .iter()
                .map(|failure| failure.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            summary.successful_models.join(", ")
        );
    }

    Ok(summary.successful_models)
}

pub fn train_models_parallel_with_progress<F, R>(
    model_configs: Vec<ModelConfig>,
    payload: Arc<TrainingPayload>,
    progress_fn: R,
    train_fn: F,
) -> Result<ParallelTrainingSummary>
where
    F: Fn(&ModelConfig, &TrainingPayload) -> Result<()> + Send + Sync + Clone + 'static,
    R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
{
    if model_configs.is_empty() {
        anyhow::bail!("No model configs provided for parallel training");
    }

    let total_models = model_configs.len();
    let threads = rust_threads_hint();
    let completed_counter = Arc::new(AtomicUsize::new(0));
    let failed_counter = Arc::new(AtomicUsize::new(0));

    info!(
        "Starting parallel training for {} models (threads={})",
        total_models, threads
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build Rayon thread pool")?;

    let results: Vec<Result<String, ModelTrainingFailure>> = pool.install(|| {
        model_configs
            .into_par_iter()
            .map(|config| {
                let payload = Arc::clone(&payload);
                let train_fn = train_fn.clone();
                let progress_fn = progress_fn.clone();
                let completed_counter = Arc::clone(&completed_counter);
                let failed_counter = Arc::clone(&failed_counter);

                progress_fn(ModelTrainingProgress::Started {
                    model: config.name.clone(),
                    total_models,
                });
                info!(
                    "Thread {:?}: Training {}",
                    std::thread::current().id(),
                    config.name
                );

                let result = train_fn(&config, &payload);

                match result {
                    Ok(_) => {
                        let completed_models = completed_counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let failed_models = failed_counter.load(Ordering::SeqCst);
                        progress_fn(ModelTrainingProgress::Succeeded {
                            model: config.name.clone(),
                            completed_models,
                            failed_models,
                            total_models,
                        });
                        info!(
                            "Thread {:?}: Completed {}",
                            std::thread::current().id(),
                            config.name
                        );
                        Ok(config.name)
                    }
                    Err(err) => {
                        let error = err.to_string();
                        let failed_models = failed_counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let completed_models = completed_counter.load(Ordering::SeqCst);
                        progress_fn(ModelTrainingProgress::Failed {
                            model: config.name.clone(),
                            error: error.clone(),
                            completed_models,
                            failed_models,
                            total_models,
                        });
                        info!(
                            "Thread {:?}: Failed {} - {}",
                            std::thread::current().id(),
                            config.name,
                            error
                        );
                        Err(ModelTrainingFailure {
                            name: config.name,
                            error,
                        })
                    }
                }
            })
            .collect()
    });

    let mut successes = Vec::new();
    let mut failures = Vec::new();
    for result in results {
        match result {
            Ok(name) => successes.push(name),
            Err(failure) => {
                info!("Model {} failed: {}", failure.name, failure.error);
                failures.push(failure);
            }
        }
    }

    info!(
        "Parallel training complete: {} succeeded, {} failed",
        successes.len(),
        failures.len()
    );

    Ok(ParallelTrainingSummary {
        total_models,
        successful_models: successes,
        failed_models: failures,
    })
}

/// Model configuration for parallel training
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub capability_family: ModelFamily,
    pub capability_state: CapabilityState,
    pub params: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    LightGBM,
    XGBoost,
    CatBoost,
    SklearsTree,
    MLP,
    NBeats,
    NBeatsxNf,
    TiDE,
    TiDENf,
    TabNet,
    KAN,
    Transformer,
    PatchTST,
    TimesNet,
    ElasticNet,
    Logistic,
    BayesianLogit,
    MetaBlender,
    ProbabilityCalibrator,
    ConformalGate,
    MetaStack,
    ExitAgent,
    SacAgent,
    OnlinePassiveAggressive,
    OnlineHoeffding,
    IsolationForest,
    Dqn,
    SwarmForecaster,
    Genetic,
    NeuroEvo,
    Neat,
    /// 3-state Hidden Markov regime model (the "34th model", 2026-05-25).
    /// Loader + adapter shipped then, but training was never wired — every
    /// install reported it "missing" forever. Wired 2026-07-11.
    HmmRegime,
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_parallel_training() {
        // Create sample data
        let n_samples = 1000;
        let n_features = 20;

        let payload = Arc::new(
            TrainingPayload::from_dense(
                Array2::<f32>::zeros((n_samples, n_features)),
                vec![0i32; n_samples],
            )
            .expect("build dense payload"),
        );

        // Create model configs
        let configs = vec![
            ModelConfig {
                name: "model_1".to_string(),
                model_type: ModelType::LightGBM,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
            ModelConfig {
                name: "model_2".to_string(),
                model_type: ModelType::XGBoost,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
            ModelConfig {
                name: "model_3".to_string(),
                model_type: ModelType::MLP,
                capability_family: ModelFamily::Deep,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
        ];

        // Test helper training function
        let train_fn = |config: &ModelConfig, _payload: &TrainingPayload| {
            println!("Training {}", config.name);
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok(())
        };

        // Train in parallel
        let results = train_models_parallel(configs, payload, train_fn).unwrap();

        assert_eq!(results.len(), 3);
        println!("Successfully trained: {:?}", results);
    }

    #[test]
    fn test_parallel_training_returns_error_when_any_model_fails() {
        let payload = Arc::new(
            TrainingPayload::from_dense(Array2::<f32>::zeros((16, 4)), vec![0i32; 16])
                .expect("build dense payload"),
        );
        let configs = vec![
            ModelConfig {
                name: "ok_model".to_string(),
                model_type: ModelType::LightGBM,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
            ModelConfig {
                name: "bad_model".to_string(),
                model_type: ModelType::XGBoost,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
        ];

        let train_fn = |config: &ModelConfig, _payload: &TrainingPayload| -> Result<()> {
            if config.name == "bad_model" {
                anyhow::bail!("synthetic failure");
            }
            Ok(())
        };

        let err = train_models_parallel(configs, payload, train_fn)
            .expect_err("expected aggregated failure");
        let msg = err.to_string();
        assert!(msg.contains("bad_model"), "unexpected error: {msg}");
        assert!(msg.contains("ok_model"), "unexpected error: {msg}");
    }

    #[test]
    fn test_parallel_training_rejects_empty_model_set() {
        let payload = Arc::new(
            TrainingPayload::from_dense(Array2::<f32>::zeros((8, 2)), vec![0i32; 8])
                .expect("build dense payload"),
        );
        let configs: Vec<ModelConfig> = Vec::new();

        let train_fn = |_config: &ModelConfig, _payload: &TrainingPayload| -> Result<()> { Ok(()) };

        let err = train_models_parallel(configs, payload, train_fn)
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("No model configs"));
    }

    #[test]
    fn test_parallel_training_summary_reports_live_model_events() {
        let payload = Arc::new(
            TrainingPayload::from_dense(Array2::<f32>::zeros((12, 3)), vec![0i32; 12])
                .expect("build dense payload"),
        );
        let configs = vec![
            ModelConfig {
                name: "ok_model".to_string(),
                model_type: ModelType::LightGBM,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
            ModelConfig {
                name: "bad_model".to_string(),
                model_type: ModelType::XGBoost,
                capability_family: ModelFamily::Tree,
                capability_state: CapabilityState::Implemented,
                params: Default::default(),
            },
        ];
        let seen_events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = Arc::clone(&seen_events);

        let train_fn = |config: &ModelConfig, _payload: &TrainingPayload| -> Result<()> {
            if config.name == "bad_model" {
                anyhow::bail!("synthetic failure");
            }
            Ok(())
        };

        let summary = train_models_parallel_with_progress(
            configs,
            payload,
            move |event| {
                event_sink
                    .lock()
                    .expect("event sink mutex poisoned")
                    .push(event);
            },
            train_fn,
        )
        .expect("parallel summary should be produced");

        assert_eq!(summary.total_models, 2);
        assert_eq!(summary.successful_models, vec!["ok_model".to_string()]);
        assert_eq!(summary.failed_models.len(), 1);
        assert_eq!(summary.failed_models[0].name, "bad_model");
        assert!(summary.failed_models[0].error.contains("synthetic failure"));

        let events = seen_events.lock().expect("event sink mutex poisoned");
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTrainingProgress::Started { model, total_models } if model == "ok_model" && *total_models == 2
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTrainingProgress::Succeeded { model, completed_models, failed_models, total_models }
                if model == "ok_model"
                    && *completed_models >= 1
                    && *failed_models <= 1
                    && *total_models == 2
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTrainingProgress::Failed { model, error, completed_models, failed_models, total_models }
                if model == "bad_model"
                    && error.contains("synthetic failure")
                    && *completed_models <= 1
                    && *failed_models >= 1
                    && *total_models == 2
        )));
    }
}
