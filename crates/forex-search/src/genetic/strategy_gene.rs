use serde::{Deserialize, Serialize};
use rand::Rng;
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gene {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub long_threshold: f32,
    pub short_threshold: f32,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trades_count: usize,
    pub generation: usize,
    pub strategy_id: String,
    pub use_ob: bool,
    pub use_fvg: bool,
    pub use_liq_sweep: bool,
    pub mtf_confirmation: bool,
    pub use_premium_discount: bool,
    pub use_inducement: bool,
    #[serde(default)]
    pub use_bos: bool,
    #[serde(default)]
    pub use_choch: bool,
    #[serde(default)]
    pub use_eqh: bool,
    #[serde(default)]
    pub use_eql: bool,
    #[serde(default)]
    pub use_displacement: bool,
    pub tp_pips: f64,
    pub sl_pips: f64,
    pub slice_pass_rate: f64,
    pub consistency: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FilteringConfig {
    pub max_dd: f64,
    pub min_profit: f64,
    pub min_trades: f64,
    pub min_sharpe: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub anomaly_guard: bool,
    pub elite_mode: bool,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            max_dd: 0.15,
            min_profit: 10.0,
            min_trades: 10.0,
            min_sharpe: 0.3,
            min_win_rate: 0.50,
            min_profit_factor: 1.05,
            anomaly_guard: true,
            elite_mode: false,
        }
    }
}

impl Gene {
    pub fn is_anomalous(&self) -> bool {
        let trades = self.trades_count as f64;
        let win_rate = self.win_rate;
        let profit_factor = self.profit_factor;
        let profit = self.fitness; // Using fitness as profit proxy
        let dd = self.max_drawdown;
        let ppt = if trades > 0.0 { profit / trades } else { 0.0 };

        // Thresholds from evo_prop.py
        let min_trades = 120.0;
        let max_dd = 0.0025;
        let min_win_rate = 0.92;
        let min_pf = 12.0;
        let min_profit = 200_000.0;
        let max_ppt = 2_000.0;

        let suspicious_combo = trades >= min_trades
            && dd <= max_dd
            && win_rate >= min_win_rate
            && profit_factor >= min_pf
            && profit >= min_profit;

        let suspicious_ppt = trades >= 40.0 && dd <= 0.01 && ppt >= max_ppt;

        let suspicious_ultra = trades >= 50.0 && dd <= 0.001 && profit >= 150_000.0 && ppt >= 1_000.0;

        let suspicious_low_dd = trades >= 80.0 && dd <= 0.001 && profit >= 50_000.0;

        suspicious_combo || suspicious_ppt || suspicious_ultra || suspicious_low_dd
    }

    pub fn passes_filter(&self, cfg: &FilteringConfig) -> bool {
        if self.fitness <= cfg.min_profit {
            return false;
        }
        if self.max_drawdown > cfg.max_dd {
            return false;
        }
        if (self.trades_count as f64) < cfg.min_trades {
            return false;
        }
        if self.sharpe_ratio < cfg.min_sharpe {
            return false;
        }
        if self.win_rate < cfg.min_win_rate {
            return false;
        }
        if self.profit_factor < cfg.min_profit_factor {
            return false;
        }
        if cfg.anomaly_guard && self.is_anomalous() {
            return false;
        }
        true
    }

    pub fn normalize(&mut self, n_indicators: usize, min_indicators: usize) {
        if self.indices.is_empty() {
            self.indices.push(0);
        }
        if self.weights.len() != self.indices.len() {
            self.weights = vec![1.0; self.indices.len()];
        }
        let n_indicators = n_indicators.max(1);
        for idx in &mut self.indices {
            if *idx >= n_indicators {
                *idx %= n_indicators;
            }
        }
        let min_indicators = min_indicators.clamp(1, n_indicators);
        if self.indices.len() < min_indicators {
            let mut rng = rand::rng();
            let mut seen = HashSet::new();
            for idx in &self.indices {
                seen.insert(*idx);
            }
            while self.indices.len() < min_indicators {
                let idx = rng.random_range(0..n_indicators);
                if seen.insert(idx) {
                    self.indices.push(idx);
                    self.weights.push(rng.random_range(0.1..1.0));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub smc_gate_threshold: f32,
    pub smc_weight_ob: f32,
    pub smc_weight_fvg: f32,
    pub smc_weight_liq: f32,
    pub smc_weight_mtf: f32,
    pub smc_weight_premium: f32,
    pub smc_weight_inducement: f32,
    pub smc_weight_bos: f32,
    pub smc_weight_choch: f32,
    pub smc_weight_eqh: f32,
    pub smc_weight_eql: f32,
    pub smc_weight_displacement: f32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        fn env_f64(name: &str, default: f64) -> f64 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(default)
        }
        fn env_f32(name: &str, default: f32) -> f32 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(default)
        }

        Self {
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: env_f64("FOREX_BOT_PROP_PIP_VALUE", 0.0001),
            spread_pips: env_f64("FOREX_BOT_PROP_SPREAD_PIPS", 1.5),
            commission_per_trade: env_f64("FOREX_BOT_PROP_COMMISSION", 0.0),
            pip_value_per_lot: env_f64("FOREX_BOT_PROP_PIP_VALUE_PER_LOT", 10.0),
            smc_gate_threshold: env_f32("FOREX_BOT_PROP_SMC_GATE", 0.75),
            smc_weight_ob: env_f32("FOREX_BOT_PROP_SMC_W_OB", 1.0),
            smc_weight_fvg: env_f32("FOREX_BOT_PROP_SMC_W_FVG", 1.0),
            smc_weight_liq: env_f32("FOREX_BOT_PROP_SMC_W_LIQ", 1.0),
            smc_weight_mtf: env_f32("FOREX_BOT_PROP_SMC_W_MTF", 1.0),
            smc_weight_premium: env_f32("FOREX_BOT_PROP_SMC_W_PREMIUM", 1.0),
            smc_weight_inducement: env_f32("FOREX_BOT_PROP_SMC_W_INDUCEMENT", 1.0),
            smc_weight_bos: env_f32("FOREX_BOT_PROP_SMC_W_BOS", 1.0),
            smc_weight_choch: env_f32("FOREX_BOT_PROP_SMC_W_CHOCH", 1.0),
            smc_weight_eqh: env_f32("FOREX_BOT_PROP_SMC_W_EQH", 1.0),
            smc_weight_eql: env_f32("FOREX_BOT_PROP_SMC_W_EQL", 1.0),
            smc_weight_displacement: env_f32("FOREX_BOT_PROP_SMC_W_DISPLACEMENT", 1.0),
        }
    }
}
