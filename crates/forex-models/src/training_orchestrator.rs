use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::parallel_trainer::{
    train_models_parallel_with_progress, ModelConfig, ModelTrainingFailure, ModelTrainingProgress,
    ModelType,
};
use forex_data::{
    load_symbol_dataset, prepare_multitimeframe_features_with_options, FeatureBuildOptions, Ohlcv,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRunSummary {
    pub planned_models: Vec<String>,
    pub completed_models: Vec<String>,
    pub failed_models: Vec<ModelTrainingFailure>,
}

pub struct TrainingOrchestrator {
    pub settings: forex_core::Settings,
    pub models_dir: PathBuf,
}

impl TrainingOrchestrator {
    pub fn new(settings: forex_core::Settings, models_dir: PathBuf) -> Self {
        Self {
            settings,
            models_dir,
        }
    }

    pub fn train_symbol(&self, symbol: &str, base_tf: &str) -> Result<()> {
        let summary = self.train_symbol_with_progress(symbol, base_tf, |_| {})?;
        if !summary.failed_models.is_empty() {
            anyhow::bail!(
                "Training failed for [{}]; successful models: [{}]",
                summary
                    .failed_models
                    .iter()
                    .map(|failure| failure.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                summary.completed_models.join(", ")
            );
        }

        Ok(())
    }

    pub fn train_symbol_with_progress<R>(
        &self,
        symbol: &str,
        base_tf: &str,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        info!("Starting Pure-Rust training for symbol: {}", symbol);

        let data_root = std::env::var("FOREX_BOT_DATA_ROOT").unwrap_or_else(|_| "data".to_string());
        let dataset = load_symbol_dataset(&data_root, symbol)?;

        let opts = FeatureBuildOptions::default();
        let frame = prepare_multitimeframe_features_with_options(&dataset, base_tf, &opts, None)?;
        let base_ohlcv = dataset.frames.get(base_tf).context("base tf missing")?;

        let enabled_models = self.get_enabled_models()?;
        let planned_models = enabled_models.clone();
        let configs: Vec<ModelConfig> = enabled_models
            .into_iter()
            .map(|m| ModelConfig {
                name: m.clone(),
                model_type: self.map_model_type(&m),
                params: HashMap::new(),
            })
            .collect();

        let x = Arc::new(frame.data);
        let labels = self.derive_labels(base_ohlcv);
        let y = Arc::new(labels);

        let trained = train_models_parallel_with_progress(
            configs,
            x,
            y,
            progress_fn,
            move |name, _x_data, _y_data| {
                info!("Training model instance: {}", name);
                anyhow::bail!(
                    "Training dispatch for model `{}` is not implemented in the Rust orchestrator",
                    name
                )
            },
        )?;

        info!(
            "Successfully trained models: {:?}",
            trained.successful_models
        );
        Ok(TrainingRunSummary {
            planned_models,
            completed_models: trained.successful_models,
            failed_models: trained.failed_models,
        })
    }

    fn get_enabled_models(&self) -> Result<Vec<String>> {
        let mut models: Vec<String> = self
            .settings
            .models
            .ml_models
            .iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .collect();

        models.sort();
        models.dedup();

        if models.is_empty() {
            anyhow::bail!("No model configs enabled in settings.models.ml_models");
        }

        let unsupported: Vec<String> = models
            .iter()
            .filter_map(|name| self.training_support_error(name))
            .collect();
        if !unsupported.is_empty() {
            anyhow::bail!(
                "Pure Rust training is not production-ready for the configured models: {}",
                unsupported.join("; ")
            );
        }

        Ok(models)
    }

    fn map_model_type(&self, name: &str) -> ModelType {
        match name {
            "xgboost" => ModelType::XGBoost,
            "genetic" => ModelType::Genetic,
            _ => ModelType::MLP,
        }
    }

    fn training_support_error(&self, name: &str) -> Option<String> {
        match name {
            "xgboost" => Some(format!("{name} (fit/save path is still a placeholder)")),
            "xgboost_rf" => Some(format!("{name} (fit/save path is not wired in the Rust orchestrator)")),
            "xgboost_dart" => Some(format!("{name} (fit/save path is not wired in the Rust orchestrator)")),
            "lightgbm" => Some(format!("{name} (fit/save path is not wired in the Rust orchestrator)")),
            "catboost" => Some(format!("{name} (fit/save path is not wired in the Rust orchestrator)")),
            "catboost_alt" => Some(format!("{name} (fit/save path is not wired in the Rust orchestrator)")),
            "mlp" | "nbeats" | "tide" | "tabnet" | "kan" => Some(format!(
                "{name} (Rust orchestrator has no active deep-model trainer dispatch)"
            )),
            "genetic" => Some(format!(
                "{name} (requires forex_bot.models.genetic, which is not present in the tracked repo/runtime)"
            )),
            other => Some(format!("{other} (unknown or unsupported training model)")),
        }
    }

    fn derive_labels(&self, ohlcv: &Ohlcv) -> Vec<i32> {
        let n = ohlcv.close.len();
        let mut labels = vec![0; n];
        for (i, slot) in labels.iter_mut().enumerate() {
            if ohlcv.close[i] > ohlcv.open[i] {
                *slot = 1;
            } else if ohlcv.close[i] < ohlcv.open[i] {
                *slot = -1;
            }
        }
        labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orchestrator_with_models(models: &[&str]) -> TrainingOrchestrator {
        let mut settings = forex_core::Settings::default();
        settings.models.ml_models = models.iter().map(|name| (*name).to_string()).collect();
        TrainingOrchestrator::new(settings, PathBuf::from("models"))
    }

    #[test]
    fn get_enabled_models_reads_from_settings_and_fails_for_unwired_models() {
        let orchestrator = orchestrator_with_models(&["xgboost", "mlp"]);
        let err = orchestrator
            .get_enabled_models()
            .expect_err("expected explicit unsupported-model error");
        let msg = err.to_string();
        assert!(msg.contains("xgboost"), "unexpected error: {msg}");
        assert!(msg.contains("mlp"), "unexpected error: {msg}");
    }

    #[test]
    fn get_enabled_models_rejects_empty_model_config() {
        let orchestrator = orchestrator_with_models(&[]);
        let err = orchestrator
            .get_enabled_models()
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("settings.models.ml_models"));
    }
}
