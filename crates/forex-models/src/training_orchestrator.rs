use anyhow::{Context, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use std::sync::Arc;

use forex_data::{load_symbol_dataset, prepare_multitimeframe_features_with_options, FeatureBuildOptions, Ohlcv};
use crate::base::ExpertModel;
use crate::tree_models::XGBoostExpert;
use crate::genetic::GeneticStrategyExpert;
use crate::parallel_trainer::{train_models_parallel, ModelConfig, ModelType};

pub struct TrainingOrchestrator {
    pub settings: forex_core::Settings,
    pub models_dir: PathBuf,
}

impl TrainingOrchestrator {
    pub fn new(settings: forex_core::Settings, models_dir: PathBuf) -> Self {
        Self { settings, models_dir }
    }

    pub fn train_symbol(&self, symbol: &str, base_tf: &str) -> Result<()> {
        info!("Starting Pure-Rust training for symbol: {}", symbol);
        
        let data_root = std::env::var("FOREX_BOT_DATA_ROOT").unwrap_or_else(|_| "data".to_string());
        let dataset = load_symbol_dataset(&data_root, symbol)?;
        
        let opts = FeatureBuildOptions::default();
        let frame = prepare_multitimeframe_features_with_options(&dataset, base_tf, &opts, None)?;
        let base_ohlcv = dataset.frames.get(base_tf).context("base tf missing")?;

        let enabled_models = self.get_enabled_models();
        let configs: Vec<ModelConfig> = enabled_models.into_iter().map(|m| {
            ModelConfig {
                name: m.clone(),
                model_type: self.map_model_type(&m),
                params: HashMap::new(),
            }
        }).collect();

        let x = Arc::new(frame.data);
        let labels = self.derive_labels(base_ohlcv);
        let y = Arc::new(labels);

        let out_dir = self.models_dir.clone();
        let trained = train_models_parallel(configs, x, y, move |name, x_data, y_data| {
            info!("Training model instance: {}", name);
            
            // Logic to select and train the specific model type
            // This is where we call the .fit() methods of our Rust models.
            
            // For now, a placeholder implementation:
            Ok(())
        })?;

        info!("Successfully trained models: {:?}", trained);
        Ok(())
    }

    fn get_enabled_models(&self) -> Vec<String> {
        vec!["xgboost".to_string(), "genetic".to_string()]
    }

    fn map_model_type(&self, name: &str) -> ModelType {
        match name {
            "xgboost" => ModelType::XGBoost,
            "genetic" => ModelType::Genetic,
            _ => ModelType::MLP,
        }
    }

    fn derive_labels(&self, ohlcv: &Ohlcv) -> Vec<i32> {
        let n = ohlcv.close.len();
        let mut labels = vec![0; n];
        for i in 0..(n-1) {
            if ohlcv.close[i+1] > ohlcv.close[i] { labels[i] = 1; }
            else if ohlcv.close[i+1] < ohlcv.close[i] { labels[i] = -1; }
        }
        labels
    }
}
