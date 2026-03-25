use tracing::{debug, info, warn};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct PropMetaState {
    pub daily_dd_pct: f64,
    pub daily_profit_pct: f64,
    pub volatility_regime: String,
    pub recent_win_rate: f64,
    pub consecutive_losses: i32,
    pub model_confidence: f64,
    pub hour_of_day: i32,
    pub market_regime: String,
}

impl Default for PropMetaState {
    fn default() -> Self {
        Self {
            daily_dd_pct: 0.0,
            daily_profit_pct: 0.0,
            volatility_regime: "normal".to_string(),
            recent_win_rate: 0.0,
            consecutive_losses: 0,
            model_confidence: 0.0,
            hour_of_day: 0,
            market_regime: "Normal".to_string(),
        }
    }
}

pub struct MetaController {
    pub max_daily_dd: f64,
    pub safety_buffer: f64,
    pub base_risk: f64,
    pub base_confidence: f64,
    pub silent: bool,
    pub k_steepness: f64,
    last_log_time: u64,
}

impl MetaController {
    pub fn new(
        max_daily_dd: Option<f64>,
        safety_buffer: Option<f64>,
        base_risk_per_trade: Option<f64>,
        base_confidence: Option<f64>,
        silent: Option<bool>,
        k_steepness: Option<f64>,
    ) -> Self {
        let silent = silent.unwrap_or(false);
        let k_steepness = k_steepness.unwrap_or(200.0);
        let base_confidence = base_confidence.unwrap_or(0.55);
        if !silent {
            info!("MetaController init: k={:.1}, base_conf={:.2}", k_steepness, base_confidence);
        }

        Self {
            max_daily_dd: max_daily_dd.unwrap_or(0.045),
            safety_buffer: safety_buffer.unwrap_or(0.025),
            base_risk: base_risk_per_trade.unwrap_or(0.015),
            base_confidence,
            silent,
            k_steepness,
            last_log_time: 0,
        }
    }

    pub fn get_risk_parameters(&mut self, state: &PropMetaState) -> (f64, f64, bool) {
        let dd_delta = state.daily_dd_pct - self.safety_buffer;
        
        // Clip exponent to prevent overflow/underflow
        let exponent = (self.k_steepness * dd_delta).clamp(-20.0, 20.0);
        let survival_multiplier = 1.0 / (1.0 + exponent.exp());

        // 1. Base Volatility Scaling
        let vol_multiplier = match state.volatility_regime.to_lowercase().as_str() {
            "low" => 1.1,
            "normal" => 1.0,
            "high" => 0.7,
            _ => 1.0,
        };

        // 2. Advanced Regime Scaling
        let mut regime_scale = 1.0;
        if state.market_regime.contains("Volatile") {
            regime_scale = 0.5;
        } else if state.market_regime.contains("Quiet") {
            regime_scale = 1.2;
        }

        if state.market_regime.contains("Bear") || state.market_regime.contains("Bull") {
            regime_scale *= 1.0;
        }

        let mut perf_multiplier: f64 = 1.0;
        if state.recent_win_rate < 0.4 {
            perf_multiplier = perf_multiplier.min(0.8);
        }
        if state.consecutive_losses >= 2 {
            perf_multiplier = perf_multiplier.min(0.8);
        }
        if state.consecutive_losses >= 4 {
            perf_multiplier = perf_multiplier.min(0.5);
        }

        
        if state.daily_profit_pct >= 0.035 {
            perf_multiplier = perf_multiplier.min(0.01);
            let now_capper = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            if !self.silent && (now_capper - self.last_log_time > 10) {
                warn!("Meta-Controller: Consistency Capper active. Daily Profit >= 3.5% ({:.2}%). Risk drastically scaled down.", state.daily_profit_pct * 100.0);
            }
        }
        let mut final_risk_multiplier = survival_multiplier * vol_multiplier * regime_scale * perf_multiplier;


        let confidence_adjustment = (1.0 - survival_multiplier) * 0.2;
        let mut final_confidence_threshold = self.base_confidence + confidence_adjustment;
        final_confidence_threshold = final_confidence_threshold.min(0.85);

        let mut allow_trading = true;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        if state.daily_dd_pct >= (self.max_daily_dd - 0.002) {
            allow_trading = false;
            final_risk_multiplier = 0.0;
            let now_capper = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            if !self.silent && (now_capper - self.last_log_time > 10) {
                warn!("Meta-Controller: Hard Stop Triggered! DD={:.2}% >= {:.2}%", 
                    state.daily_dd_pct * 100.0, 
                    (self.max_daily_dd - 0.002) * 100.0
                );
                self.last_log_time = now;
            }
        }

        if !self.silent && state.daily_dd_pct > 0.01 {
            if (now - self.last_log_time > 60) && survival_multiplier < 0.5 {
                info!("Meta-Controller: DD={:.2}% | Regime={} | RiskMult={:.2} | ReqConf={:.2}",
                    state.daily_dd_pct * 100.0,
                    state.market_regime,
                    final_risk_multiplier,
                    final_confidence_threshold
                );
                self.last_log_time = now;
            } else if now - self.last_log_time > 60 {
                debug!("Meta-Controller: DD={:.2}% | Regime={} | RiskMult={:.2} | ReqConf={:.2}",
                    state.daily_dd_pct * 100.0,
                    state.market_regime,
                    final_risk_multiplier,
                    final_confidence_threshold
                );
            }
        }

        (final_risk_multiplier, final_confidence_threshold, allow_trading)
    }
}
