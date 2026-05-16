//! Prop firm hard constraints and local challenge/risk defaults.
//!
//! Source of truth: FTMO Trader Challenge rules (https://ftmo.com/en/trading-objectives/).
//! Other prop firms (MyForexFunds, The5%ers, FundedNext) use similar
//! limits ±0.5% — if we need per-firm customization later, this struct
//! becomes a runtime config but the defaults stay.
//! External prop-firm numbers belong in [`PropFirmConstraints`]. Local
//! forex-ai policy defaults live beside it so search, validation, and live
//! risk code do not carry duplicate literals.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropFirmConstraints {
    /// Maximum daily loss as fraction of account equity (FTMO: 5%).
    /// A loss exceeding this in a single trading day fails the challenge.
    pub max_daily_loss_pct: f32,
    /// Maximum overall drawdown as fraction of account equity (FTMO: 10%).
    /// Trailing or static depending on firm; FTMO uses static-from-initial-balance.
    pub max_overall_drawdown_pct: f32,
    /// Profit target as fraction of account equity to clear the challenge
    /// (FTMO Phase 1: 10%, Phase 2: 5%).
    pub challenge_profit_target_pct: f32,
    /// Operator-mandated minimum monthly net profit target (4% per
    /// directive 2026-05-14). Live strategies that drop below this
    /// monthly should be flagged for review.
    pub min_monthly_net_profit_pct: f32,
    /// Minimum trading days per challenge cycle (FTMO: 4 trading days
    /// for the Aggressive variant, 10 for Standard).
    pub min_trading_days: u32,
}

impl PropFirmConstraints {
    /// Canonical FTMO Trader Challenge values plus operator's 4%
    /// monthly profit floor.
    pub const FTMO_STANDARD: Self = Self {
        max_daily_loss_pct: 0.05,
        max_overall_drawdown_pct: 0.10,
        challenge_profit_target_pct: 0.10,
        min_monthly_net_profit_pct: 0.04, // operator directive
        min_trading_days: 10,
    };
}

/// Local operating defaults for challenge-cycle planning.
///
/// These are not external prop-firm rules. They are forex-ai runtime
/// defaults that need one canonical owner because the search optimizer,
/// validation artifacts, and live risk presets all reason about the same
/// challenge window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropFirmChallengeDefaults {
    /// Denominator used to turn a full challenge profit target into a daily
    /// pacing target.
    pub daily_target_trading_days: u32,
    /// Short-cycle fixture/window minimum used by local validation flows.
    pub relaxed_min_trading_days: u32,
    /// Planning horizon used by phase-specific risk presets.
    pub target_trading_days: u32,
    /// Upper bound used for challenge-cycle pacing.
    pub max_trading_days: u32,
}

impl PropFirmChallengeDefaults {
    pub const FTMO_STANDARD: Self = Self {
        daily_target_trading_days: 20,
        relaxed_min_trading_days: 5,
        target_trading_days: 22,
        max_trading_days: 60,
    };
}

/// Local runtime defaults layered under the hard prop-firm constraints.
///
/// These numbers are guard-rail policy, not FTMO facts. Keeping them here
/// prevents duplicated risk bands and trade caps from drifting across crates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropFirmRuntimeDefaults {
    pub max_lot_size: f64,
    pub daily_dd_warning_pct: f64,
    pub daily_dd_stop_trading_pct: f64,
    pub daily_profit_lock_pct: f64,
    pub max_trades_per_day: usize,
    pub recovery_halt_drawdown_pct: f64,
    pub recovery_top_strategy_drawdown_pct: f64,
    pub recovery_min_sharpe_drawdown_pct: f64,
    pub recovery_top_three_drawdown_pct: f64,
    pub recovery_top_strategy_rank: usize,
    pub recovery_caution_strategy_rank: usize,
    pub recovery_max_trades_per_day: usize,
    pub recovery_min_strategy_sharpe: f64,
    pub recovery_mode_risk_multiplier: f64,
    pub defensive_mode_risk_multiplier: f64,
    pub caution_mode_risk_multiplier: f64,
}

impl PropFirmRuntimeDefaults {
    pub const FTMO_STANDARD: Self = Self {
        max_lot_size: 10.0,
        daily_dd_warning_pct: 0.035,
        daily_dd_stop_trading_pct: 0.040,
        daily_profit_lock_pct: 0.03,
        max_trades_per_day: 15,
        recovery_halt_drawdown_pct: 0.05,
        recovery_top_strategy_drawdown_pct: 0.04,
        recovery_min_sharpe_drawdown_pct: 0.03,
        recovery_top_three_drawdown_pct: 0.02,
        recovery_top_strategy_rank: 1,
        recovery_caution_strategy_rank: 3,
        recovery_max_trades_per_day: 2,
        recovery_min_strategy_sharpe: 1.0,
        recovery_mode_risk_multiplier: 0.25,
        defensive_mode_risk_multiplier: 0.50,
        caution_mode_risk_multiplier: 0.75,
    };
}

#[cfg(test)]
mod tests {
    use super::{PropFirmChallengeDefaults, PropFirmConstraints, PropFirmRuntimeDefaults};

    #[test]
    fn ftmo_runtime_defaults_stay_inside_hard_constraints() {
        let constraints = PropFirmConstraints::FTMO_STANDARD;
        let challenge = PropFirmChallengeDefaults::FTMO_STANDARD;
        let runtime = PropFirmRuntimeDefaults::FTMO_STANDARD;

        assert!(challenge.relaxed_min_trading_days < constraints.min_trading_days);
        assert!(challenge.max_trading_days > constraints.min_trading_days);
        assert!(challenge.daily_target_trading_days <= challenge.target_trading_days);
        assert!(runtime.daily_dd_warning_pct < runtime.daily_dd_stop_trading_pct);
        assert!(runtime.daily_dd_stop_trading_pct <= constraints.max_daily_loss_pct as f64);
        assert!(
            runtime.recovery_top_three_drawdown_pct < runtime.recovery_min_sharpe_drawdown_pct
        );
        assert!(
            runtime.recovery_min_sharpe_drawdown_pct
                < runtime.recovery_top_strategy_drawdown_pct
        );
        assert!(runtime.recovery_top_strategy_drawdown_pct < runtime.recovery_halt_drawdown_pct);
        assert!(runtime.recovery_halt_drawdown_pct <= constraints.max_overall_drawdown_pct as f64);
        assert!(runtime.recovery_mode_risk_multiplier < runtime.defensive_mode_risk_multiplier);
        assert!(runtime.defensive_mode_risk_multiplier < runtime.caution_mode_risk_multiplier);
        assert!(runtime.caution_mode_risk_multiplier <= 1.0);
    }
}
