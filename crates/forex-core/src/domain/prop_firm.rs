//! Prop firm constraint constants — the ONLY hardcoded numeric values
//! allowed in production code per operator directive 2026-05-14.
//!
//! Source of truth: FTMO Trader Challenge rules (https://ftmo.com/en/trading-objectives/).
//! Other prop firms (MyForexFunds, The5%ers, FundedNext) use similar
//! limits ±0.5% — if we need per-firm customization later, this struct
//! becomes a runtime config but the defaults stay.

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
