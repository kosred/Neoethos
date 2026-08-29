//! Phase-1 `RiskGate` stubs.
//!
//! The intended Phase-5 gate routes every intent through
//! `RiskyModeManager::check_trade_allowed` (kill switches, daily/monthly caps,
//! the equity-floor of design §8). Phase 1 ships a permissive gate (proves the
//! allow path) plus a tiny max-concurrent-positions cap so the reject path is
//! exercised end-to-end.
//!
//! **CORRECTION 2026-08-09 (audit #137).** This header also named
//! `neoethos_core::domain::risk::RiskManager::check_trade_allowed` (the
//! PropFirm daily-loss / max-DD / drawdown-recovery tiers) as the other real
//! gate. That type has **no production constructor anywhere in the workspace** —
//! all three `RiskManager::new` call sites are inside its own `#[cfg(test)]`
//! module — so it is not something this crate can be "wired to" today. Whether
//! it gets wired or deleted is an open operator decision recorded on the struct
//! itself. Until then, do not read this file as "the real thing exists, we just
//! haven't plugged it in".
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
    fn check(
        &self,
        _intent: &TradeIntent,
        _account: &AccountSnapshot,
    ) -> Result<(), KillSwitchTier> {
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
