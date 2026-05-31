pub mod consistency;
pub mod errors;
pub mod events;
pub mod meta_controller;

pub mod drift_monitor;
pub mod news_filter;
pub mod order_execution;
pub mod portfolio;
pub mod promotion_gate;
pub mod prop_firm;
pub mod risk;
pub mod risky_mode;

pub use promotion_gate::{
    CriterionResult, PromotionDecision, PromotionGateConfig, PromotionMetrics,
    aggregate_portfolio, evaluate_promotion,
};
pub use prop_firm::{
    PropFirmChallengeDefaults, PropFirmConstraints, PropFirmPhaseRiskDefaults,
    PropFirmRuntimeDefaults,
};
pub use risky_mode::{
    DEFAULT_RISKY_TRADES_PER_DAY, KillSwitchTier, MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY,
    RiskyModeConfig, RiskyModeManager, RiskyStage, build_logarithmic_stages,
};
