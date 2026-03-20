use anyhow::{Result, Context};
use std::path::Path;
use forex_data::{load_symbol_dataset, ensure_timeframes_with_resample, prepare_multitimeframe_features, MANDATORY_TFS};
use crate::discovery::{ensure_non_empty_portfolio, run_discovery_cycle, DiscoveryConfig, save_portfolio_json};
use tracing::info;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchDiscoverySummary {
    pub symbols_seen: usize,
    pub work_units_seen: usize,
    pub portfolios_saved: usize,
    pub skipped_symbols: usize,
    pub skipped_timeframes: usize,
    pub feature_failures: usize,
    pub empty_portfolios: usize,
}

impl BatchDiscoverySummary {
    fn finalize(self) -> Result<Self> {
        if self.portfolios_saved == 0 {
            anyhow::bail!(
                "Batch discovery produced no usable portfolios (symbols={}, work_units={}, skipped_symbols={}, skipped_timeframes={}, feature_failures={}, empty_portfolios={})",
                self.symbols_seen,
                self.work_units_seen,
                self.skipped_symbols,
                self.skipped_timeframes,
                self.feature_failures,
                self.empty_portfolios
            );
        }
        Ok(self)
    }
}

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

    pub fn run_batch(&self, symbols: &[String], timeframes: &[String]) -> Result<BatchDiscoverySummary> {
        std::fs::create_dir_all(&self.output_dir)?;
        let mut summary = BatchDiscoverySummary::default();
        
        for symbol in symbols {
            summary.symbols_seen += 1;
            info!("Processing symbol: {}", symbol);
            let ds = match load_symbol_dataset(&self.data_root, symbol) {
                Ok(d) => d,
                Err(e) => {
                    summary.skipped_symbols += 1;
                    info!("Skipping symbol {}: {}", symbol, e);
                    continue;
                }
            };

            for tf in timeframes {
                summary.work_units_seen += 1;
                info!("  Timeframe: {}", tf);
                let ds_ready = match ensure_timeframes_with_resample(&ds, tf, MANDATORY_TFS) {
                    Ok(d) => d,
                    Err(e) => {
                        summary.skipped_timeframes += 1;
                        info!("    Skipping tf {}: {}", tf, e);
                        continue;
                    }
                };

                let features = match prepare_multitimeframe_features(&ds_ready, tf, &[], None) {
                    Ok(f) => f,
                    Err(e) => {
                        summary.feature_failures += 1;
                        info!("    Feature prep failed: {}", e);
                        continue;
                    }
                };

                let base_ohlcv = ds_ready.frames.get(tf).context("base tf missing")?;
                let result = run_discovery_cycle(&features, base_ohlcv, &self.config)?;
                if let Err(err) = ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, tf)) {
                    summary.empty_portfolios += 1;
                    info!("    {}", err);
                    continue;
                }
                
                info!("    Found {} strategies", result.portfolio.len());
                
                let out_path = Path::new(&self.output_dir).join(format!("{}_{}.json", symbol, tf));
                save_portfolio_json(out_path, &result.portfolio, &features.names)?;
                summary.portfolios_saved += 1;
            }
        }
        summary.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_summary_rejects_zero_saved_portfolios() {
        let summary = BatchDiscoverySummary {
            symbols_seen: 1,
            work_units_seen: 2,
            skipped_symbols: 0,
            skipped_timeframes: 1,
            feature_failures: 1,
            empty_portfolios: 0,
            portfolios_saved: 0,
        };

        let err = summary.finalize().expect_err("expected zero-save batch to fail");
        let msg = err.to_string();
        assert!(msg.contains("no usable portfolios"), "unexpected error: {msg}");
        assert!(msg.contains("work_units=2"), "unexpected error: {msg}");
    }

    #[test]
    fn batch_summary_accepts_at_least_one_saved_portfolio() {
        let summary = BatchDiscoverySummary {
            portfolios_saved: 1,
            work_units_seen: 3,
            ..Default::default()
        };

        let finalized = summary.finalize().expect("expected non-empty batch success");
        assert_eq!(finalized.portfolios_saved, 1);
    }
}
