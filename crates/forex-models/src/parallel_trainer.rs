// Parallel Model Trainer - multi-core training for Rust-native workloads
// Note: Python-backed training still obeys the GIL unless it releases it or runs in separate processes.
use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::env;
use std::sync::Arc;
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
    let threads = rust_threads_hint();
    info!(
        "Starting parallel training for {} models (threads={})",
        model_configs.len(),
        threads
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build Rayon thread pool")?;

    let results: Vec<(String, Result<()>)> = pool.install(|| {
        model_configs
            .into_par_iter()
            .map(|config| {
                let x = Arc::clone(&x);
                let y = Arc::clone(&y);
                let train_fn = train_fn.clone();

                info!(
                    "Thread {:?}: Training {}",
                    std::thread::current().id(),
                    config.name
                );

                let result = train_fn(&config.name, &x, &y);

                match &result {
                    Ok(_) => info!(
                        "Thread {:?}: Completed {}",
                        std::thread::current().id(),
                        config.name
                    ),
                    Err(e) => info!(
                        "Thread {:?}: Failed {} - {}",
                        std::thread::current().id(),
                        config.name,
                        e
                    ),
                }

                (config.name, result)
            })
            .collect()
    });

    // Collect results
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for (name, result) in results {
        match result {
            Ok(_) => successes.push(name),
            Err(e) => {
                info!("Model {} failed: {}", name, e);
                failures.push(name);
            }
        }
    }

    info!(
        "Parallel training complete: {} succeeded, {} failed",
        successes.len(),
        failures.len()
    );

    Ok(successes)
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
}
