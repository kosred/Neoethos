use anyhow::{Result, Context};
use std::path::Path;
use forex_data::{load_symbol_dataset, ensure_timeframes_with_resample, prepare_multitimeframe_features, MANDATORY_TFS};
use crate::discovery::{run_discovery_cycle, DiscoveryConfig, save_portfolio_json};
use tracing::info;

pub struct DiscoveryOrchestrator {
    pub data_root: String,
    pub output_dir: String,
    pub config: DiscoveryConfig,
}

impl DiscoveryOrchestrator {
    pub fn new(data_root: &str, output_dir: &str, config: DiscoveryConfig) -> Self {
        Self {
            data_root: data_root.to_string(),
            output_dir: output_dir.to_string(),
            config,
        }
    }

    pub fn run_batch(&self, symbols: &[String], timeframes: &[String]) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)?;
        
        for symbol in symbols {
            info!("Processing symbol: {}", symbol);
            let ds = match load_symbol_dataset(&self.data_root, symbol) {
                Ok(d) => d,
                Err(e) => {
                    info!("Skipping symbol {}: {}", symbol, e);
                    continue;
                }
            };

            for tf in timeframes {
                info!("  Timeframe: {}", tf);
                let ds_ready = match ensure_timeframes_with_resample(&ds, tf, MANDATORY_TFS) {
                    Ok(d) => d,
                    Err(e) => {
                        info!("    Skipping tf {}: {}", tf, e);
                        continue;
                    }
                };

                let features = match prepare_multitimeframe_features(&ds_ready, tf, &[], None) {
                    Ok(f) => f,
                    Err(e) => {
                        info!("    Feature prep failed: {}", e);
                        continue;
                    }
                };

                let base_ohlcv = ds_ready.frames.get(tf).context("base tf missing")?;
                let result = run_discovery_cycle(&features, base_ohlcv, &self.config)?;
                
                info!("    Found {} strategies", result.portfolio.len());
                
                let out_path = Path::new(&self.output_dir).join(format!("{}_{}.json", symbol, tf));
                save_portfolio_json(out_path, &result.portfolio, &features.names)?;
            }
        }
        Ok(())
    }
}
