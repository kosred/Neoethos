use crate::eval::BacktestSettings;
use crate::genetic::{Gene, month_day_indices, signals_for_gene};
use forex_data::{FeatureFrame, Ohlcv};

#[derive(Debug, Clone)]
pub struct GauntletConfig {
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub max_daily_dd: f64,
    pub warn_only: bool,
    pub backtest: BacktestSettings,
}

impl Default for GauntletConfig {
    fn default() -> Self {
        Self {
            min_win_rate: 0.55,
            min_profit_factor: 1.2,
            max_drawdown_pct: 0.07,
            max_daily_dd: 0.04,
            warn_only: false,
            backtest: BacktestSettings::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyGauntlet {
    pub config: GauntletConfig,
}

impl StrategyGauntlet {
    pub fn new(config: Option<GauntletConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    pub fn run(&self, features: &FeatureFrame, ohlcv: &Ohlcv, gene: &Gene) -> bool {
        if features.data.nrows() == 0 {
            return false;
        }
        if ohlcv.close.len() != features.data.nrows() {
            return false;
        }

        let signals = signals_for_gene(features, gene);
        if signals.is_empty() || signals.len() != ohlcv.close.len() {
            return false;
        }
        let (months, days) = month_day_indices(&features.timestamps);

        let mut settings = self.config.backtest.clone();
        settings.sl_pips = gene.sl_pips;
        settings.tp_pips = gene.tp_pips;

        let metrics = crate::eval::fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            &signals,
            &months,
            &days,
            &features.timestamps,
            &settings,
        );

        let net_profit = metrics[0];
        let max_dd = metrics[3];
        let win_rate = metrics[4];
        let profit_factor = metrics[5];
        let max_daily_dd = metrics[10];

        if win_rate < self.config.min_win_rate {
            return self.config.warn_only;
        }
        if max_dd > self.config.max_drawdown_pct {
            return self.config.warn_only;
        }
        if max_daily_dd > self.config.max_daily_dd {
            return self.config.warn_only;
        }
        if profit_factor <= self.config.min_profit_factor {
            return self.config.warn_only;
        }
        if net_profit <= 0.0 {
            return self.config.warn_only;
        }
        true
    }
}
