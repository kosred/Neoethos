use crate::discovery::{
    DiscoveryConfig, ensure_non_empty_portfolio, run_discovery_cycle, save_discovery_profile_json,
    save_portfolio_json, save_quality_report_json, save_trade_log_json,
};
use anyhow::Result;
use forex_data::{
    MANDATORY_TFS, ensure_timeframes_with_resample, load_symbol_dataset,
    prepare_multitimeframe_features,
};
use std::path::Path;
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
    pub discovery_failures: usize,
}

impl BatchDiscoverySummary {
    fn finalize(self) -> Result<Self> {
        if self.portfolios_saved == 0 {
            anyhow::bail!(
                "Batch discovery produced no usable portfolios (symbols={}, work_units={}, skipped_symbols={}, skipped_timeframes={}, feature_failures={}, empty_portfolios={}, discovery_failures={})",
                self.symbols_seen,
                self.work_units_seen,
                self.skipped_symbols,
                self.skipped_timeframes,
                self.feature_failures,
                self.empty_portfolios,
                self.discovery_failures
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

    pub fn run_batch(
        &self,
        symbols: &[String],
        timeframes: &[String],
    ) -> Result<BatchDiscoverySummary> {
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

                let htfs: Vec<&str> = self
                    .config
                    .higher_timeframes
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let features = match prepare_multitimeframe_features(&ds_ready, tf, &htfs, None) {
                    Ok(f) => f,
                    Err(e) => {
                        summary.feature_failures += 1;
                        info!("    Feature prep failed: {}", e);
                        continue;
                    }
                };

                let base_ohlcv = match ds_ready.frames.get(tf) {
                    Some(o) => o,
                    None => {
                        summary.feature_failures += 1;
                        info!("    Skipping tf {}: base ohlcv missing", tf);
                        continue;
                    }
                };
                let mut runtime_config = self.config.clone();
                runtime_config.timeframe_label = tf.clone();
                // Previously this used `?` and aborted the whole batch on a
                // single discovery failure, while every other error in the
                // loop counted toward `summary.skipped_*` and continued.
                let result = match run_discovery_cycle(&features, base_ohlcv, &runtime_config) {
                    Ok(r) => r,
                    Err(e) => {
                        summary.discovery_failures += 1;
                        info!("    Discovery failed for {} {}: {}", symbol, tf, e);
                        continue;
                    }
                };
                if let Err(err) = ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, tf))
                {
                    summary.empty_portfolios += 1;
                    info!("    {}", err);
                    continue;
                }

                info!("    Found {} strategies", result.portfolio.len());

                let out_path = Path::new(&self.output_dir).join(format!("{}_{}.json", symbol, tf));
                save_portfolio_json(&out_path, &result)?;
                let profile_path =
                    Path::new(&self.output_dir).join(format!("{}_{}_profile.json", symbol, tf));
                save_discovery_profile_json(profile_path, &runtime_config, &result)?;
                if !result.quality_metrics.is_empty() {
                    let quality_path =
                        Path::new(&self.output_dir).join(format!("{}_{}_quality.json", symbol, tf));
                    save_quality_report_json(quality_path, &result)?;
                }
                if !result.logged_trades.is_empty() {
                    let trade_log_path = Path::new(&self.output_dir)
                        .join(format!("{}_{}_trade_logs.json", symbol, tf));
                    save_trade_log_json(trade_log_path, &result)?;
                }
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
            ..Default::default()
        };

        let err = summary
            .finalize()
            .expect_err("expected zero-save batch to fail");
        let msg = err.to_string();
        assert!(
            msg.contains("no usable portfolios"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("work_units=2"), "unexpected error: {msg}");
    }

    #[test]
    fn batch_summary_accepts_at_least_one_saved_portfolio() {
        let summary = BatchDiscoverySummary {
            portfolios_saved: 1,
            work_units_seen: 3,
            ..Default::default()
        };

        let finalized = summary
            .finalize()
            .expect("expected non-empty batch success");
        assert_eq!(finalized.portfolios_saved, 1);
    }
}
