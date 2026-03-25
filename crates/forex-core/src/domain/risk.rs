use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum ChallengePhase {
    #[default]
    Phase1,
    Phase2,
    Funded,
}

impl From<&str> for ChallengePhase {
    fn from(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "phase2" | "phase_2" | "verification" | "verify" => Self::Phase2,
            "funded" | "live" => Self::Funded,
            _ => Self::Phase1,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropFirmRules {
    pub max_daily_loss_pct: f64,
    pub max_total_loss_pct: f64,
    pub profit_target_pct: f64,
    pub min_trading_days: usize,
    pub max_trading_days: usize,
    pub max_lot_size: f64,
    pub news_trading_allowed: bool,
    pub weekend_holding: bool,
    pub scaling_enabled: bool,
    pub daily_dd_warning_pct: f64,
    pub daily_dd_stop_trading_pct: f64,
    pub daily_profit_lock_pct: f64,
    pub max_trades_per_day: usize,
}

impl Default for PropFirmRules {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: 0.045,
            max_total_loss_pct: 0.10,
            profit_target_pct: 0.10,
            min_trading_days: 5,
            max_trading_days: 60,
            max_lot_size: 10.0,
            news_trading_allowed: false,
            weekend_holding: false,
            scaling_enabled: true,
            daily_dd_warning_pct: 0.035,
            daily_dd_stop_trading_pct: 0.040,
            daily_profit_lock_pct: 0.03,
            max_trades_per_day: 15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChallengeRiskPreset {
    pub phase: String,
    pub risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub min_confidence_threshold: f64,
    pub max_trades_per_day: usize,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub daily_profit_lock_pct: f64,
    pub monthly_profit_target_pct: f64,
    pub challenge_target_return_pct: f64,
    pub challenge_target_trading_days: usize,
}

pub fn resolve_challenge_risk_preset(phase: &str) -> ChallengeRiskPreset {
    let phase_enum = ChallengePhase::from(phase);
    match phase_enum {
        ChallengePhase::Phase2 => ChallengeRiskPreset {
            phase: "phase_2".to_string(),
            risk_per_trade: 0.0025,
            max_risk_per_trade: 0.0040,
            min_confidence_threshold: 0.68,
            max_trades_per_day: 3,
            daily_drawdown_limit: 0.045,
            total_drawdown_limit: 0.10,
            daily_profit_lock_pct: 0.012,
            monthly_profit_target_pct: 0.05,
            challenge_target_return_pct: 0.05,
            challenge_target_trading_days: 22,
        },
        ChallengePhase::Funded => ChallengeRiskPreset {
            phase: "funded".to_string(),
            risk_per_trade: 0.0030,
            max_risk_per_trade: 0.0050,
            min_confidence_threshold: 0.65,
            max_trades_per_day: 4,
            daily_drawdown_limit: 0.045,
            total_drawdown_limit: 0.10,
            daily_profit_lock_pct: 0.0,
            monthly_profit_target_pct: 0.06,
            challenge_target_return_pct: 0.06,
            challenge_target_trading_days: 22,
        },
        ChallengePhase::Phase1 => ChallengeRiskPreset {
            phase: "phase_1".to_string(),
            risk_per_trade: 0.0030,
            max_risk_per_trade: 0.0050,
            min_confidence_threshold: 0.66,
            max_trades_per_day: 3,
            daily_drawdown_limit: 0.045,
            total_drawdown_limit: 0.10,
            daily_profit_lock_pct: 0.015,
            monthly_profit_target_pct: 0.10,
            challenge_target_return_pct: 0.10,
            challenge_target_trading_days: 22,
        },
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TradeRecord {
    pub entry_time_sec: u64,
    pub exit_time_sec: u64,
    pub pnl: f64,
    pub was_stopped: bool,
    pub duration_minutes: f64,
    pub size: f64,
    pub direction: Option<i32>,
}

#[derive(Debug, Clone, Copy)]
pub struct TradeGateInput {
    pub equity: f64,
    pub confidence: f64,
    pub current_time_sec: u64,
    pub current_hour: u32,
    pub current_min: u32,
    pub weekday: u32,
    pub market_volatility: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PositionSizingInput {
    pub equity: f64,
    pub base_risk_pct: f64,
    pub max_risk_cap: f64,
    pub confidence: f64,
    pub uncertainty: f64,
    pub market_volatility: f64,
    pub target_volatility: f64,
    pub is_volatile_regime: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RevengeTradeDetector {
    pub recent_trades: Vec<TradeRecord>,
    pub max_trades_tracked: usize,
}

impl Default for RevengeTradeDetector {
    fn default() -> Self {
        Self {
            recent_trades: Vec::new(),
            max_trades_tracked: 10,
        }
    }
}

impl RevengeTradeDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_trade(
        &mut self,
        entry_time_sec: u64,
        exit_time_sec: u64,
        pnl: f64,
        was_stopped: bool,
        size: f64,
        direction: Option<i32>,
    ) {
        let duration_minutes = (exit_time_sec.saturating_sub(entry_time_sec)) as f64 / 60.0;
        self.recent_trades.push(TradeRecord {
            entry_time_sec,
            exit_time_sec,
            pnl,
            was_stopped,
            duration_minutes,
            size,
            direction,
        });

        if self.recent_trades.len() > self.max_trades_tracked {
            self.recent_trades.remove(0);
        }
    }

    pub fn is_revenge_trading(&self, current_time_sec: u64, current_hour: u32) -> bool {
        if self.recent_trades.len() < 2 {
            return false;
        }

        let last_trade = self.recent_trades.last().unwrap();
        let time_since_last_min = (current_time_sec.saturating_sub(last_trade.exit_time_sec)) as f64 / 60.0;

        if time_since_last_min < 15.0 && last_trade.pnl < 0.0 {
            return true;
        }

        let mut consecutive_losses = 0;
        for trade in self.recent_trades.iter().rev().take(5) {
            if trade.pnl < 0.0 {
                consecutive_losses += 1;
            } else {
                break;
            }
        }
        
        if consecutive_losses >= 3 {
            let optimal_times = (7..9).contains(&current_hour) || (13..15).contains(&current_hour);
            if !optimal_times {
                return true;
            }
        }

        if self.recent_trades.len() >= 3 {
            let recent_idx = self.recent_trades.len() - 3;
            let recent = &self.recent_trades[recent_idx..];
            let mut sum_prev_sizes = 0.0;
            let mut count_prev = 0;
            for t in &recent[..recent.len() - 1] {
                sum_prev_sizes += t.size;
                count_prev += 1;
            }
            if count_prev > 0 {
                let mean_prev = sum_prev_sizes / (count_prev as f64);
                let last_size = recent.last().unwrap().size;
                let prev_pnl = recent[recent.len() - 2].pnl;
                if mean_prev > 0.0 && last_size > 1.5 * mean_prev && prev_pnl < 0.0 {
                    return true;
                }
            }
        }

        if self.recent_trades.len() >= 3 {
            let n = self.recent_trades.len();
            let t1 = &self.recent_trades[n - 3];
            let t2 = &self.recent_trades[n - 2];
            let t3 = &self.recent_trades[n - 1];

            if t1.direction.is_some() 
                && t1.direction == t2.direction 
                && t2.direction == t3.direction 
                && t3.pnl < 0.0 
                && t2.pnl < 0.0 
            {
                return true;
            }
        }

        if self.recent_trades.len() >= 2 {
            let n = self.recent_trades.len();
            let prev = &self.recent_trades[n - 2];
            let last = &self.recent_trades[n - 1];
            let gap_min = (last.entry_time_sec.saturating_sub(prev.exit_time_sec)) as f64 / 60.0;

            if gap_min < 30.0
                && last.pnl < 0.0
                && prev.pnl < 0.0
                && last.direction.is_some()
                && last.direction == prev.direction
            {
                return true;
            }
        }

        false
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskManager {
    pub prop_rules: PropFirmRules,
    pub challenge_mode: bool,

    pub total_peak_equity: f64,
    pub day_start_equity: f64,
    pub day_peak_equity: f64,
    pub month_start_equity: f64,
    pub challenge_start_equity: f64,

    pub daily_loss: f64,
    pub daily_profit: f64,
    pub session_trades: usize,
    pub session_trade_counts: HashMap<String, usize>,
    pub consecutive_losses: usize,

    pub circuit_breaker_triggered: bool,
    pub recovery_mode: bool,
    pub reflection_mode: bool,
    pub monthly_target_hit: bool,
    pub challenge_target_hit: bool,

    pub monthly_profit_target_pct: f64,
    pub challenge_target_return_pct: f64,

    pub monthly_return_pct: f64,
    pub challenge_return_pct: f64,

    pub last_session_date_id: Option<u32>,
    pub month_start_date_id: Option<u32>,
    pub challenge_start_date_id: Option<u32>,

    pub kill_window_until_sec: Option<u64>,
    pub session_start_hour: u32,
    pub session_start_min: u32,
    pub session_end_hour: u32,
    pub session_end_min: u32,
    pub block_night_session: bool,
    pub night_block_start_hour: u32,
    pub night_block_end_hour: u32,
    pub night_min_volatility: f64,

    pub revenge_detector: RevengeTradeDetector,
    pub min_confidence_threshold: f64,
}

impl RiskManager {
    pub fn new(prop_rules: PropFirmRules, challenge_mode: bool, initial_balance: f64) -> Self {
        Self {
            prop_rules,
            challenge_mode,
            total_peak_equity: initial_balance,
            day_start_equity: initial_balance,
            day_peak_equity: initial_balance,
            month_start_equity: initial_balance,
            challenge_start_equity: if challenge_mode { initial_balance } else { 0.0 },
            daily_loss: 0.0,
            daily_profit: 0.0,
            session_trades: 0,
            session_trade_counts: HashMap::new(),
            consecutive_losses: 0,
            circuit_breaker_triggered: false,
            recovery_mode: false,
            reflection_mode: false,
            monthly_target_hit: false,
            challenge_target_hit: false,
            monthly_profit_target_pct: 0.04,
            challenge_target_return_pct: 0.10,
            monthly_return_pct: 0.0,
            challenge_return_pct: 0.0,
            last_session_date_id: None,
            month_start_date_id: None,
            challenge_start_date_id: None,
            kill_window_until_sec: None,
            session_start_hour: 0,
            session_start_min: 0,
            session_end_hour: 23,
            session_end_min: 59,
            block_night_session: true,
            night_block_start_hour: 0,
            night_block_end_hour: 6,
            night_min_volatility: 0.0008,
            revenge_detector: RevengeTradeDetector::new(),
            min_confidence_threshold: 0.55,
        }
    }

    pub fn set_session_times(&mut self, start_h: u32, start_m: u32, end_h: u32, end_m: u32) {
        self.session_start_hour = start_h;
        self.session_start_min = start_m;
        self.session_end_hour = end_h;
        self.session_end_min = end_m;
    }

    pub fn set_night_block(&mut self, enabled: bool, start_h: u32, end_h: u32, min_vol: f64) {
        self.block_night_session = enabled;
        self.night_block_start_hour = start_h;
        self.night_block_end_hour = end_h;
        self.night_min_volatility = min_vol;
    }

    pub fn update_kill_window(&mut self, until_sec: u64) {
        self.kill_window_until_sec = Some(until_sec);
    }

    pub fn drawdown_state(&self, equity: f64) -> (f64, f64, f64, f64, f64) {
        let daily_dd_pct = if self.day_start_equity > 0.0 {
            (self.day_start_equity - equity) / self.day_start_equity
        } else {
            0.0
        };
        let intraday_dd_pct = if self.day_peak_equity > 0.0 {
            (self.day_peak_equity - equity) / self.day_peak_equity
        } else {
            0.0
        };
        let total_dd_pct = if self.challenge_start_equity > 0.0 {
            (self.challenge_start_equity - equity) / self.challenge_start_equity
        } else if self.total_peak_equity > 0.0 {
            (self.total_peak_equity - equity) / self.total_peak_equity
        } else {
            0.0
        };
        let dd_used = daily_dd_pct.max(intraday_dd_pct).max(0.0);
        let dd_limit = self.prop_rules.daily_dd_stop_trading_pct.max(1e-9);
        (daily_dd_pct, intraday_dd_pct, dd_used, dd_limit, total_dd_pct)
    }

    pub fn update_recovery_state(&mut self, equity: f64) {
        if self.day_start_equity <= 0.0 {
            return;
        }
        let daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity;

        if daily_dd_pct >= self.prop_rules.daily_dd_warning_pct {
            if !self.recovery_mode {
                self.recovery_mode = true;
            }
        } else if self.recovery_mode {
            let recovery_threshold_pct = 0.005;
            let half_warning = self.prop_rules.daily_dd_warning_pct / 2.0;
            if equity >= (self.day_start_equity * (1.0 - recovery_threshold_pct))
                || daily_dd_pct <= half_warning
            {
                self.recovery_mode = false;
            }
        }
    }

    pub fn is_trading_session(&self, current_hour: u32, current_min: u32, weekday: u32) -> bool {
        if weekday >= 5 {
            return false;
        }
        let cur_total_min = current_hour * 60 + current_min;
        let start_total_min = self.session_start_hour * 60 + self.session_start_min;
        let end_total_min = self.session_end_hour * 60 + self.session_end_min;

        if end_total_min < start_total_min {
            cur_total_min >= start_total_min || cur_total_min <= end_total_min
        } else {
            cur_total_min >= start_total_min && cur_total_min <= end_total_min
        }
    }

    pub fn check_trade_allowed(&mut self, input: TradeGateInput) -> (bool, String) {
        if self.challenge_mode && self.challenge_target_hit {
            return (false, "Challenge target reached".to_string());
        }
        if !self.challenge_mode && self.monthly_target_hit {
            return (false, "Monthly profit target reached".to_string());
        }

        if self.circuit_breaker_triggered {
            return (false, "Circuit breaker active".to_string());
        }

        if !self.is_trading_session(input.current_hour, input.current_min, input.weekday) {
            return (false, "Outside trading session".to_string());
        }

        if self.block_night_session {
            let s = self.night_block_start_hour;
            let e = self.night_block_end_hour;
            let in_window = if e < s {
                input.current_hour >= s || input.current_hour < e
            } else {
                input.current_hour >= s && input.current_hour < e
            };
            if in_window && input.market_volatility < self.night_min_volatility {
                return (false, format!("Night session blocked (vol={:.5}<{:.5})", input.market_volatility, self.night_min_volatility));
            }
        }

        if let Some(kill_until) = self.kill_window_until_sec {
            if input.current_time_sec < kill_until {
                return (false, "News kill window active".to_string());
            }
        }

        if self.revenge_detector.is_revenge_trading(input.current_time_sec, input.current_hour) {
            return (false, "Revenge trading detected".to_string());
        }

        let (daily_dd, intraday_dd, _dd_used, _dd_limit, total_dd) = self.drawdown_state(input.equity);
        if total_dd >= self.prop_rules.max_total_loss_pct {
            self.circuit_breaker_triggered = true;
            return (false, format!("Total drawdown limit ({:.2}%)", total_dd * 100.0));
        }
        if daily_dd >= self.prop_rules.daily_dd_stop_trading_pct {
            self.circuit_breaker_triggered = true;
            return (false, format!("Daily drawdown limit ({:.2}%)", daily_dd * 100.0));
        }
        if intraday_dd >= self.prop_rules.daily_dd_stop_trading_pct {
            self.circuit_breaker_triggered = true;
            return (false, format!("Intraday trailing limit ({:.2}%)", intraday_dd * 100.0));
        }

        if self.session_trades >= self.prop_rules.max_trades_per_day {
            return (false, "Max trades per day reached".to_string());
        }

        if input.confidence < self.min_confidence_threshold {
             return (false, format!("Confidence {:.2} too low", input.confidence));
        }

        (true, "OK".to_string())
    }

    pub fn on_trade_opened(&mut self) {
        self.session_trades += 1;
    }

    pub fn on_trade_closed(&mut self, pnl: f64, equity: f64) {
        if pnl > 0.0 {
            self.daily_profit += pnl;
            self.consecutive_losses = 0;
        } else {
            self.daily_loss += pnl.abs();
            self.consecutive_losses += 1;
        }
        if equity > self.total_peak_equity { self.total_peak_equity = equity; }
        if equity > self.day_peak_equity { self.day_peak_equity = equity; }
        self.update_recovery_state(equity);
    }

    pub fn calculate_position_size(&mut self, input: PositionSizingInput) -> f64 {
        let signal_multiplier = if input.confidence >= 0.80 {
            1.00
        } else if input.confidence >= 0.60 {
            0.50 + (input.confidence - 0.60) * 2.5
        } else {
            0.30
        };

        let uncertainty_penalty = 1.0 - (input.uncertainty * 0.5);
        let mut risk_pct = input.base_risk_pct * signal_multiplier * uncertainty_penalty;

        let mut current_cap = input.max_risk_cap;
        if self.recovery_mode {
            current_cap = input.max_risk_cap * 0.5;
        }
        risk_pct = risk_pct.min(current_cap);

        if input.target_volatility > 0.0 && input.market_volatility > 0.0 {
            let mut vol_scale = input.target_volatility / input.market_volatility;
            vol_scale = vol_scale.clamp(0.35, 1.30);
            risk_pct *= vol_scale;
        }

        if input.is_volatile_regime {
            risk_pct *= 0.5;
        }

        let (_, _, dd_used, dd_limit, total_dd_pct) = self.drawdown_state(input.equity);
        let dd_frac = dd_used / dd_limit.max(1e-9);

        if dd_frac >= 0.75 {
            risk_pct *= 0.35;
        } else if dd_frac >= 0.50 {
            risk_pct *= 0.60;
        }

        let max_total_loss = self.prop_rules.max_total_loss_pct.max(1e-6);
        if total_dd_pct >= max_total_loss {
            return 0.0;
        } else if total_dd_pct > 0.0 {
            let scale = 1.0 - (total_dd_pct / max_total_loss);
            risk_pct *= scale.max(0.3);
        }

        risk_pct.clamp(0.0, input.max_risk_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::{PositionSizingInput, PropFirmRules, RiskManager, TradeGateInput};

    fn test_rules() -> PropFirmRules {
        PropFirmRules {
            max_total_loss_pct: 0.10,
            daily_dd_stop_trading_pct: 0.04,
            ..PropFirmRules::default()
        }
    }

    #[test]
    fn drawdown_state_reports_total_drawdown_from_challenge_start_equity() {
        let manager = RiskManager::new(test_rules(), true, 10_000.0);

        let (_, _, _, _, total_dd) = manager.drawdown_state(9_250.0);

        assert!((total_dd - 0.075).abs() < 1e-9);
    }

    #[test]
    fn check_trade_allowed_triggers_circuit_breaker_on_total_drawdown_limit() {
        let mut manager = RiskManager::new(test_rules(), true, 10_000.0);

        let (allowed, reason) = manager.check_trade_allowed(TradeGateInput {
            equity: 9_000.0,
            confidence: 0.90,
            current_time_sec: 1_700_000_000,
            current_hour: 10,
            current_min: 0,
            weekday: 2,
            market_volatility: 0.0015,
        });

        assert!(!allowed);
        assert!(manager.circuit_breaker_triggered);
        assert!(reason.contains("Total drawdown limit"));
    }

    #[test]
    fn calculate_position_size_returns_zero_once_total_drawdown_limit_is_hit() {
        let mut manager = RiskManager::new(test_rules(), true, 10_000.0);

        let position_size_risk = manager.calculate_position_size(PositionSizingInput {
            equity: 9_000.0,
            base_risk_pct: 0.01,
            max_risk_cap: 0.02,
            confidence: 0.90,
            uncertainty: 0.0,
            market_volatility: 0.0010,
            target_volatility: 0.0010,
            is_volatile_regime: false,
        });

        assert_eq!(position_size_risk, 0.0);
    }
}
