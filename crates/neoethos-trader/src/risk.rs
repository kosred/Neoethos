//! Phase-1 `RiskGate` stubs.
//!
//! The intended Phase-5 gate routes every intent through
//! `RiskyModeManager::check_trade_allowed` (kill switches, daily/monthly caps,
//! the equity-floor of design §8). Phase 1 ships a permissive gate (proves the
//! allow path) plus a tiny max-concurrent-positions cap so the reject path is
//! exercised end-to-end.
//!
//! **UPDATED 2026-08-10 (audit #137).** The other real gate is
//! `neoethos_core::domain::risk::RiskManager::check_trade_allowed` (the
//! prop-firm daily-loss / total-drawdown / drawdown-recovery /
//! revenge-trade tiers). On 2026-08-09 this header recorded that the type had
//! **no production constructor anywhere in the workspace**; the operator then
//! decided "WIRE IT", so it now has exactly one —
//! `RiskManager::from_settings(&Settings, live_equity)` — and
//! `neoethos-app/src/app_services/live_trading.rs` calls it for every engine
//! whose `system.trading_mode` is not `risky`.
//!
//! What has NOT changed is this crate: nothing here constructs either manager.
//! The gates in this file are still the Phase-1 stubs, and a replay run is
//! still risk-free by construction.
//!
//! [`PermissiveRiskGate`] is what every `data_replay::replay_*` path actually
//! uses, which is why every replay reports it in
//! [`crate::engine::EngineStats::fidelity_warnings`]: no trade in a replay run
//! was ever refused for risk.

use crate::contracts::{AccountSnapshot, KillSwitchTier, RiskGate, TradeIntent};

/// Allows every intent. The Phase-1 default while the real risk wiring is
/// stubbed — the loop's allow path is what we're validating here.
#[derive(Debug, Clone, Copy, Default)]
pub struct PermissiveRiskGate;

impl RiskGate for PermissiveRiskGate {
    fn check(&self, _intent: &TradeIntent, _account: &AccountSnapshot) -> Result<(), KillSwitchTier> {
        Ok(())
    }
}

/// Rejects new `Open`s once `max_open` positions are already on the book; always
/// allows `Close`/`Amend`/`Cancel` (de-risking must never be blocked). A minimal
/// stand-in for the real exposure caps so the reject path has coverage.
#[derive(Debug, Clone, Copy)]
pub struct MaxOpenPositionsGate {
    pub max_open: usize,
}

impl MaxOpenPositionsGate {
    pub fn new(max_open: usize) -> Self {
        Self { max_open }
    }
}

impl RiskGate for MaxOpenPositionsGate {
    fn check(&self, intent: &TradeIntent, account: &AccountSnapshot) -> Result<(), KillSwitchTier> {
        match intent {
            TradeIntent::Open { .. } if account.open_positions >= self.max_open => {
                Err(KillSwitchTier::ExposureCap)
            }
            _ => Ok(()),
        }
    }
}
