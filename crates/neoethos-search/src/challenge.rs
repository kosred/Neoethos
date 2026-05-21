use neoethos_core::domain::prop_firm::{PropFirmChallengeDefaults, PropFirmConstraints};

#[derive(Debug, Clone, Copy)]
pub struct ChallengeTarget {
    pub total_profit_target: f64,
    pub daily_target: f64,
    pub max_daily_dd: f64,
    pub max_total_dd: f64,
    pub min_trading_days: i32,
    pub max_trading_days: i32,
}

impl Default for ChallengeTarget {
    fn default() -> Self {
        // Prop-firm constraint values sourced from the canonical
        // `PropFirmConstraints` struct per operator directive 2026-05-14.
        let ftmo = PropFirmConstraints::FTMO_STANDARD;
        let challenge_defaults = PropFirmChallengeDefaults::FTMO_STANDARD;
        let total_profit_target = ftmo.challenge_profit_target_pct as f64;
        Self {
            total_profit_target,
            daily_target: total_profit_target / challenge_defaults.daily_target_trading_days as f64,
            max_daily_dd: ftmo.max_daily_loss_pct as f64,
            max_total_dd: ftmo.max_overall_drawdown_pct as f64,
            min_trading_days: challenge_defaults.relaxed_min_trading_days as i32,
            max_trading_days: challenge_defaults.max_trading_days as i32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChallengeOptimizer {
    pub target: ChallengeTarget,
}

#[derive(Debug, Clone, Copy)]
pub struct RiskAllocationInput {
    pub current_profit: f64,
    pub days_left: i32,
    pub current_drawdown: f64,
    pub win_rate: f64,
    pub avg_risk_reward: f64,
    pub daily_loss_pct: f64,
    pub realized_trades_per_day: f64,
}

impl ChallengeOptimizer {
    pub fn new(target: Option<ChallengeTarget>) -> Self {
        Self {
            target: target.unwrap_or_default(),
        }
    }

    pub fn optimize_risk(&self, current_profit: f64, days_left: i32) -> f64 {
        self.optimize_risk_allocation(RiskAllocationInput {
            current_profit,
            days_left,
            current_drawdown: 0.0,
            win_rate: 0.55,
            avg_risk_reward: 2.0,
            daily_loss_pct: 0.0,
            realized_trades_per_day: 2.0,
        })
    }

    pub fn optimize_risk_allocation(&self, input: RiskAllocationInput) -> f64 {
        let remaining_target = self.target.total_profit_target - input.current_profit;
        if remaining_target <= 0.0 {
            return 0.0025;
        }

        let est_trades = (input.days_left.max(1) as f64 * input.realized_trades_per_day).max(1.0);
        let expectancy = (input.win_rate * input.avg_risk_reward) - (1.0 - input.win_rate);

        let required_risk = if expectancy <= 0.1 {
            0.01
        } else {
            remaining_target / (est_trades * expectancy.max(1e-6))
        };

        let kelly = self.kelly_criterion(input.win_rate, input.avg_risk_reward);
        let kelly_limit = kelly * 0.25;

        let daily_room = (self.target.max_daily_dd - input.daily_loss_pct).max(0.0);
        let total_room = (self.target.max_total_dd - input.current_drawdown).max(0.0);
        let safety_limit = daily_room.min(total_room).max(0.0);
        let daily_utilization = if self.target.max_daily_dd > 0.0 {
            (input.daily_loss_pct / self.target.max_daily_dd).clamp(0.0, 1.5)
        } else {
            1.0
        };
        let total_utilization = if self.target.max_total_dd > 0.0 {
            (input.current_drawdown / self.target.max_total_dd).clamp(0.0, 1.5)
        } else {
            1.0
        };
        let time_pressure = if self.target.max_trading_days > 0 {
            1.0 - (input.days_left.max(0) as f64 / self.target.max_trading_days as f64)
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let pace_factor = (1.0 - 0.45 * time_pressure).clamp(0.40, 1.0);
        let drawdown_factor =
            (1.0 - 0.55 * daily_utilization.max(total_utilization)).clamp(0.20, 1.0);
        let quality_factor = if expectancy <= 0.0 {
            0.25
        } else {
            ((input.win_rate.clamp(0.0, 1.0) + (input.avg_risk_reward / 3.0).clamp(0.0, 1.0)) * 0.5)
                .clamp(0.35, 1.0)
        };
        let safety_cap = (safety_limit * 0.5).max(0.0);

        let mut optimal_risk = required_risk.min(kelly_limit).min(safety_cap)
            * pace_factor
            * drawdown_factor
            * quality_factor;

        if input.daily_loss_pct >= self.target.max_daily_dd * 0.9
            || input.current_drawdown >= self.target.max_total_dd * 0.9
        {
            optimal_risk = optimal_risk.min(0.0025);
        }

        optimal_risk.clamp(0.001, 0.015)
    }

    fn kelly_criterion(&self, win_rate: f64, rw_ratio: f64) -> f64 {
        if rw_ratio <= 0.0 {
            return 0.0;
        }
        let kelly = win_rate - ((1.0 - win_rate) / rw_ratio);
        kelly.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use neoethos_core::domain::prop_firm::{PropFirmChallengeDefaults, PropFirmConstraints};

    use super::ChallengeTarget;

    #[test]
    fn challenge_target_uses_shared_ftmo_window_defaults() {
        let target = ChallengeTarget::default();
        let constraints = PropFirmConstraints::FTMO_STANDARD;
        let defaults = PropFirmChallengeDefaults::FTMO_STANDARD;

        assert_eq!(
            target.daily_target,
            constraints.challenge_profit_target_pct as f64
                / defaults.daily_target_trading_days as f64
        );
        assert_eq!(
            target.min_trading_days,
            defaults.relaxed_min_trading_days as i32
        );
        assert_eq!(target.max_trading_days, defaults.max_trading_days as i32);
    }
}
