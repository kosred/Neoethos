// Parallel Model Trainer - multi-core training for Rust-native workloads.
// The runtime now treats every active family as a native or self-contained path.
use anyhow::{Context, Result};
use neoethos_data::FeatureFrame;
use neoethos_execution_budget::CpuLease;
use rayon::prelude::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tracing::info;

use crate::runtime::capabilities::{CapabilityState, ModelFamily};

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
    /// The single physical f64+validity feature backing shared by every model.
    /// Backend-local f32 conversion, when intrinsically required, happens only
    /// inside that backend's named adapter after validity selection.
    pub frame: Arc<FeatureFrame>,
    pub labels: Arc<Vec<i32>>,
    /// Stable mapping back to the canonical feature rows. Orchestration may
    /// select eligible rows, but it must never silently lose their identity.
    pub source_row_indices: Arc<Vec<usize>>,
}

impl TrainingPayload {
    pub fn from_frame(frame: FeatureFrame, labels: Vec<i32>) -> Result<Self> {
        let source_row_indices = (0..frame.n_samples()).collect();
        Self::from_frame_with_source_rows(frame, labels, source_row_indices)
    }

    pub fn from_frame_with_source_rows(
        frame: FeatureFrame,
        labels: Vec<i32>,
        source_row_indices: Vec<usize>,
    ) -> Result<Self> {
        if frame.n_samples() != labels.len() {
            anyhow::bail!(
                "training payload row/label mismatch: {} rows vs {} labels",
                frame.n_samples(),
                labels.len()
            );
        }
        if source_row_indices.len() != labels.len() {
            anyhow::bail!(
                "training payload source-row mismatch: {} source rows vs {} labels",
                source_row_indices.len(),
                labels.len(),
            );
        }
        for pair in source_row_indices.windows(2) {
            anyhow::ensure!(
                pair[0] < pair[1],
                "training payload source-row indices must be strictly increasing; found {} then {}",
                pair[0],
                pair[1]
            );
        }

        Ok(Self {
            frame: Arc::new(frame),
            labels: Arc::new(labels),
            source_row_indices: Arc::new(source_row_indices),
        })
    }
}

/// Train multiple models in parallel using a bounded Rayon thread pool.
pub fn train_models_parallel<F>(
    model_configs: Vec<ModelConfig>,
    payload: Arc<TrainingPayload>,
    lease: &CpuLease,
    train_fn: F,
) -> Result<Vec<String>>
where
    F: Fn(&ModelConfig, &TrainingPayload, &CpuLease) -> Result<()> + Send + Sync + Clone,
{
    let summary =
        train_models_parallel_with_progress(model_configs, payload, lease, |_| {}, train_fn)?;

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
    lease: &CpuLease,
    progress_fn: R,
    train_fn: F,
) -> Result<ParallelTrainingSummary>
where
    F: Fn(&ModelConfig, &TrainingPayload, &CpuLease) -> Result<()> + Send + Sync + Clone,
    R: Fn(ModelTrainingProgress) + Send + Sync + Clone,
{
    if model_configs.is_empty() {
        anyhow::bail!("No model configs provided for parallel training");
    }

    let total_models = model_configs.len();
    let threads = lease.width().get();
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

    // Thread-oversubscription fix (2026-07-13): up to `threads` models train
    // concurrently, and each tree model reads `cpu_threads_hint_for` for its
    // OWN internal pool — so tell that hint the concurrency, and it hands
    // each model `budget / concurrency` threads instead of the full budget
    // (previously budget² threads thrashed on `budget` cores). The guard
    // restores the single-model default on drop, even on panic.
    let _concurrency_guard =
        crate::tree_models::config::TrainingConcurrencyGuard::new(threads.min(total_models));

    let results: Vec<Result<String, ModelTrainingFailure>> = pool.install(|| {
        lease.scope(|| {
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

                    let result = train_fn(&config, &payload, lease);

                    match result {
                        Ok(_) => {
                            let completed_models =
                                completed_counter.fetch_add(1, Ordering::SeqCst) + 1;
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
        })
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
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    use neoethos_execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
    use std::sync::{Arc, Mutex};

    fn sample_payload(rows: usize, features: usize) -> Arc<TrainingPayload> {
        let columns = (0..features)
            .map(|feature| {
                FeatureColumnF64::new(
                    format!("feature_{feature}"),
                    (0..rows)
                        .map(|row| row as f64 + feature as f64 / 10.0)
                        .collect(),
                    vec![FeatureCellValidity::Valid; rows],
                )
                .expect("valid test feature column")
            })
            .collect();
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
        .expect("valid canonical test feature frame");
        Arc::new(
            TrainingPayload::from_frame(frame, vec![0; rows]).expect("build f64 training payload"),
        )
    }

    fn lease(width: usize) -> CpuLease {
        let width = WorkerLimit::new(width).expect("positive worker width");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("isolated test lease")
    }

    #[test]
    fn test_parallel_training() {
        // Create sample data
        let n_samples = 1000;
        let n_features = 20;

        let payload = sample_payload(n_samples, n_features);
        let cpu_lease = lease(3);

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
        let train_fn = |config: &ModelConfig, _payload: &TrainingPayload, lease: &CpuLease| {
            assert_eq!(lease.width().get(), 3);
            println!("Training {}", config.name);
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok(())
        };

        // Train in parallel
        let results = train_models_parallel(configs, payload, &cpu_lease, train_fn).unwrap();

        assert_eq!(results.len(), 3);
        println!("Successfully trained: {:?}", results);
    }

    #[test]
    fn test_parallel_training_returns_error_when_any_model_fails() {
        let payload = sample_payload(16, 4);
        let cpu_lease = lease(2);
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

        let train_fn =
            |config: &ModelConfig, _payload: &TrainingPayload, _lease: &CpuLease| -> Result<()> {
                if config.name == "bad_model" {
                    anyhow::bail!("synthetic failure");
                }
                Ok(())
            };

        let err = train_models_parallel(configs, payload, &cpu_lease, train_fn)
            .expect_err("expected aggregated failure");
        let msg = err.to_string();
        assert!(msg.contains("bad_model"), "unexpected error: {msg}");
        assert!(msg.contains("ok_model"), "unexpected error: {msg}");
    }

    #[test]
    fn test_parallel_training_rejects_empty_model_set() {
        let payload = sample_payload(8, 2);
        let cpu_lease = lease(1);
        let configs: Vec<ModelConfig> = Vec::new();

        let train_fn = |_config: &ModelConfig,
                        _payload: &TrainingPayload,
                        _lease: &CpuLease|
         -> Result<()> { Ok(()) };

        let err = train_models_parallel(configs, payload, &cpu_lease, train_fn)
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("No model configs"));
    }

    #[test]
    fn test_parallel_training_summary_reports_live_model_events() {
        let payload = sample_payload(12, 3);
        let cpu_lease = lease(2);
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

        let train_fn =
            |config: &ModelConfig, _payload: &TrainingPayload, _lease: &CpuLease| -> Result<()> {
                if config.name == "bad_model" {
                    anyhow::bail!("synthetic failure");
                }
                Ok(())
            };

        let summary = train_models_parallel_with_progress(
            configs,
            payload,
            &cpu_lease,
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
