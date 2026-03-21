// Parallel Model Trainer - multi-core training for Rust-native workloads
// Note: Python-backed training still obeys the GIL unless it releases it or runs in separate processes.
use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::env;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::info;

fn read_threads_env(keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Ok(val) = env::var(key) {
            if let Ok(parsed) = val.trim().parse::<usize>() {
                if parsed > 0 {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

fn rust_threads_hint() -> usize {
    read_threads_env(&[
        "FOREX_BOT_RUST_THREADS",
        "FOREX_BOT_CPU_THREADS",
        "FOREX_BOT_CPU_BUDGET",
        "RAYON_NUM_THREADS",
    ])
    .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1))
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

/// Train multiple models in parallel using a bounded Rayon thread pool.
/// Rust-native training will run in parallel; Python-backed training may still serialize under GIL.
pub fn train_models_parallel<F>(
    model_configs: Vec<ModelConfig>,
    x: Arc<Array2<f32>>,
    y: Arc<Vec<i32>>,
    train_fn: F,
) -> Result<Vec<String>>
where
    F: Fn(&str, &Array2<f32>, &[i32]) -> Result<()> + Send + Sync + Clone + 'static,
{
    let summary = train_models_parallel_with_progress(model_configs, x, y, |_| {}, train_fn)?;

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
    x: Arc<Array2<f32>>,
    y: Arc<Vec<i32>>,
    progress_fn: R,
    train_fn: F,
) -> Result<ParallelTrainingSummary>
where
    F: Fn(&str, &Array2<f32>, &[i32]) -> Result<()> + Send + Sync + Clone + 'static,
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
                let x = Arc::clone(&x);
                let y = Arc::clone(&y);
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

                let result = train_fn(&config.name, &x, &y);

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
#[derive(Clone)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub params: std::collections::HashMap<String, String>,
}

#[derive(Clone)]
pub enum ModelType {
    LightGBM,
    XGBoost,
    CatBoost,
    MLP,
    NBeats,
    TiDE,
    TabNet,
    KAN,
    Genetic,
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

        let x = Arc::new(Array2::<f32>::zeros((n_samples, n_features)));
        let y = Arc::new(vec![0i32; n_samples]);

        // Create model configs
        let configs = vec![
            ModelConfig {
                name: "model_1".to_string(),
                model_type: ModelType::LightGBM,
                params: Default::default(),
            },
            ModelConfig {
                name: "model_2".to_string(),
                model_type: ModelType::XGBoost,
                params: Default::default(),
            },
            ModelConfig {
                name: "model_3".to_string(),
                model_type: ModelType::MLP,
                params: Default::default(),
            },
        ];

        // Training function (mock)
        let train_fn = |name: &str, _x: &Array2<f32>, _y: &[i32]| {
            println!("Training {}", name);
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok(())
        };

        // Train in parallel
        let results = train_models_parallel(configs, x, y, train_fn).unwrap();

        assert_eq!(results.len(), 3);
        println!("Successfully trained: {:?}", results);
    }

    #[test]
    fn test_parallel_training_returns_error_when_any_model_fails() {
        let x = Arc::new(Array2::<f32>::zeros((16, 4)));
        let y = Arc::new(vec![0i32; 16]);
        let configs = vec![
            ModelConfig {
                name: "ok_model".to_string(),
                model_type: ModelType::LightGBM,
                params: Default::default(),
            },
            ModelConfig {
                name: "bad_model".to_string(),
                model_type: ModelType::XGBoost,
                params: Default::default(),
            },
        ];

        let train_fn = |name: &str, _x: &Array2<f32>, _y: &[i32]| -> Result<()> {
            if name == "bad_model" {
                anyhow::bail!("synthetic failure");
            }
            Ok(())
        };

        let err = train_models_parallel(configs, x, y, train_fn)
            .expect_err("expected aggregated failure");
        let msg = err.to_string();
        assert!(msg.contains("bad_model"), "unexpected error: {msg}");
        assert!(msg.contains("ok_model"), "unexpected error: {msg}");
    }

    #[test]
    fn test_parallel_training_rejects_empty_model_set() {
        let x = Arc::new(Array2::<f32>::zeros((8, 2)));
        let y = Arc::new(vec![0i32; 8]);
        let configs: Vec<ModelConfig> = Vec::new();

        let train_fn = |_name: &str, _x: &Array2<f32>, _y: &[i32]| -> Result<()> { Ok(()) };

        let err = train_models_parallel(configs, x, y, train_fn)
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("No model configs"));
    }

    #[test]
    fn test_parallel_training_summary_reports_live_model_events() {
        let x = Arc::new(Array2::<f32>::zeros((12, 3)));
        let y = Arc::new(vec![0i32; 12]);
        let configs = vec![
            ModelConfig {
                name: "ok_model".to_string(),
                model_type: ModelType::LightGBM,
                params: Default::default(),
            },
            ModelConfig {
                name: "bad_model".to_string(),
                model_type: ModelType::XGBoost,
                params: Default::default(),
            },
        ];
        let seen_events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = Arc::clone(&seen_events);

        let train_fn = |name: &str, _x: &Array2<f32>, _y: &[i32]| -> Result<()> {
            if name == "bad_model" {
                anyhow::bail!("synthetic failure");
            }
            Ok(())
        };

        let summary = train_models_parallel_with_progress(
            configs,
            x,
            y,
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
                if model == "ok_model" && *completed_models == 1 && *failed_models == 0 && *total_models == 2
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTrainingProgress::Failed { model, error, completed_models, failed_models, total_models }
                if model == "bad_model"
                    && error.contains("synthetic failure")
                    && *completed_models == 1
                    && *failed_models == 1
                    && *total_models == 2
        )));
    }
}
