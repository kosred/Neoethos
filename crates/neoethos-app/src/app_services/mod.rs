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
pub mod ctrader_proto_messages;
pub mod ctrader_state_machine;
pub mod ctrader_tls;
pub mod discovery;
// `dxtrade` — REMOVED 2026-07-11 (operator directive, superseding the
// 2026-05-21 "keep DXtrade" restore). The D3 adapter never shipped past
// auth stubs, had ZERO dispatch call sites, and the broker-agnostic
// direction is now MCP bridges (see mcp/ — e.g. the MT5 MCP servers in
// mcp_servers.example.json). Git history keeps the 2755-line module if
// a native DXtrade adapter is ever wanted again.
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

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
    // Was sent by the now-removed legacy TradingSession connect path; retained
    // as the event-bus shape pending Flutter live-status wiring.
    #[allow(dead_code)]
    CTraderConnectUpdated(crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot),
    // Was sent by the now-removed legacy TradingSession bootstrap path; retained
    // as the event-bus shape for the future bootstrap-progress UI.
    #[allow(dead_code)]
    BootstrapUpdated(JobSnapshot),
    // Was sent by the now-removed legacy TradingSession connect path; retained
    // as the event-bus shape until the Flutter UI subscribes for live status.
    #[allow(dead_code)]
    ConnectOutcome(Result<String, String>),
    /// Background chart-data fetch completed. The UI should refresh the
    /// chart panel without blocking the render thread on WebSocket I/O.
    // Was constructed by the now-removed legacy TradingSession chart fetcher;
    // retained as the event-bus shape until the Flutter chart panel reads it.
    #[allow(dead_code)]
    ChartDataUpdated(Box<crate::app_services::trading_types::MarketChartSnapshot>),
    /// A background OS thread panicked, caught inside the worker so the
    /// process is not killed — but the operator MUST see it rather than have
    /// the join handle silently discarded (which once left the UI showing
    /// "Running…" forever). Retained as the event-bus shape from the
    /// now-removed legacy background-task spawner.
    #[allow(dead_code)]
    BackgroundTaskPanic {
        task: String,
        message: String,
    },
}
