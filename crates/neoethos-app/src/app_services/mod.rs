pub mod backoff;
pub mod bootstrap_writer;
// **F-CORE3 cluster consolidation (2026-05-25)**: canonical registry of
// every `NEOETHOS_BOT_CTRADER_*` / `NEOETHOS_BOT_PNL_*` / `NEOETHOS_*` env
// override the app crate honours. Mirror of `neoethos_core::env_overrides`.
// Call-sites elsewhere consult `env_overrides::*` typed getters instead
// of reading `std::env::var` directly.
pub mod env_overrides;
pub mod broker_config;
pub mod broker_persistence;
pub mod ctrader_account;
pub mod ctrader_auth;
pub mod ctrader_bootstrap;
pub mod ctrader_data;
pub mod ctrader_errors;
pub mod ctrader_execution;
pub mod ctrader_history;
#[cfg(test)]
mod ctrader_integration_tests;
pub mod ctrader_live_auth;
pub mod ctrader_messages;
pub mod ctrader_money;
pub mod ctrader_openapi;
pub mod ctrader_tls;
pub mod discovery;
pub mod broker_api;
// Phase C (2026-05-28) — one-shot tool that captures real
// `ProtoOASymbolByIdRes` payloads from the configured cTrader
// account and writes them as fixtures under tests/fixtures/. Used
// to verify Phase A.1 schema assumptions against actual broker
// bytes (see `--capture-symbols` CLI flag in main.rs).
pub mod capture_symbols;
pub mod challenge_sim;
pub mod embedded_credentials;
pub mod jobs;
pub mod journal_reconcile;
pub mod journal_store;
pub mod journal_analytics;
pub mod journal_stats;
pub mod live_journal;
pub mod live_gate;
pub mod pnl;
pub mod reauth;
pub mod live_spots;
pub mod live_spots_streamer;
pub mod experience_store;
pub mod experience_train;
pub mod federation;
pub mod live_parity;
pub mod live_trading;
pub mod news_calendar;
pub mod news_research;
pub mod pending_actions;
pub mod rediscovery;
pub mod risky_mode_persistence;
pub mod secure_store;
pub mod spread_stats;
pub mod strategy_blacklist;
pub mod supervisor;
pub mod tail_risk;
pub mod trading_types;
pub mod training;
pub mod validation;

use crate::app_services::jobs::JobSnapshot;

// 2026-08-08 dead-code purge: the CTraderConnectUpdated / BootstrapUpdated /
// ConnectOutcome / ChartDataUpdated / BackgroundTaskPanic variants (all
// #[allow(dead_code)] leftovers of the removed legacy TradingSession event
// bus) had zero construct and zero match sites repo-wide and were deleted.
#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
}
