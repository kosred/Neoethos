pub mod consistency;
pub mod errors;
pub mod events;
pub mod meta_controller;

pub mod drift_monitor;
pub mod news_filter;
pub mod order_execution;
pub mod portfolio;
pub mod prop_firm;
pub mod risk;
pub mod risky_mode;

pub use prop_firm::{
    PropFirmChallengeDefaults, PropFirmConstraints, PropFirmPhaseRiskDefaults,
    PropFirmRuntimeDefaults,
};
pub use risky_mode::{
    KillSwitchTier, RiskyModeConfig, RiskyModeManager, RiskyStage, build_logarithmic_stages,
};
