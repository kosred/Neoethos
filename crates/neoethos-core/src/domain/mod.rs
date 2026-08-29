//! Domain rules that decide what the system is allowed to do.
//!
//! 2026-08-09 (D3 purge): `consistency`, `drift_monitor`, `events`,
//! `meta_controller`, `news_filter`, `order_execution` and `portfolio` were
//! deleted. None was re-exported here and none had a single reference outside
//! its own file — verified across `crates/`, `desktop/src`,
//! `desktop/src-tauri/src`, `mesh/`, `mcp/`, `scripts/` and `.github/`. They
//! were parallel, never-constructed implementations of gates the live engine
//! already performs elsewhere (news blackout, portfolio correlation, order
//! sizing), which is exactly what makes them dangerous: a reader looking for
//! "the news gate" found a plausible one that never ran.

pub mod daily_entry_cap;
pub mod demo_gate;
pub mod errors;
pub mod kelly;
pub mod promotion_gate;
pub mod prop_firm;
pub mod risk;
pub mod risky_mode;

pub use demo_gate::{DemoForwardDecision, DemoForwardGateConfig, evaluate_demo_forward_gate};
pub use kelly::{risk_constrained_kelly, risk_constrained_kelly_empirical};
pub use promotion_gate::{
    CriterionResult, PromotionDecision, PromotionGateConfig, PromotionMetrics, aggregate_portfolio,
    evaluate_promotion,
};
pub use prop_firm::{
    PropFirmChallengeDefaults, PropFirmConstraints, PropFirmPhaseRiskDefaults,
    PropFirmRuntimeDefaults,
};
pub use risky_mode::{
    DEFAULT_RISKY_TRADES_PER_DAY, KillSwitchTier, MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY,
    RiskyModeConfig, RiskyModeManager, RiskyStage, build_logarithmic_stages,
};
