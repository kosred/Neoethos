//! Persistent main-chrome widgets — Demo/Paper/Live status pill and
//! the red HALT panic button. Implements giants-pattern gaps #1 and
//! #3 from
//! `docs/audits/research/wizard_onboarding_competitive_analysis.md`
//! §10 (ThinkOrSwim paperMoney pill, MT4/5 AutoTrading panic button,
//! TradingView gray-vs-red Trading Panel).
//!
//! Both widgets are rendered once per frame from the main top-bar
//! draw loop in `main.rs`. They live in this dedicated `chrome` module
//! so the existing top-bar / status-bar inline code stays focused on
//! the engine + connection ribbon and the new operator-safety
//! controls are grouped together for a future audit.

pub mod halt_button;
pub mod status_pill;
