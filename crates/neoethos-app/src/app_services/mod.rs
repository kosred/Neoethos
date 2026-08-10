pub mod backoff;
pub mod bootstrap_writer;
// **F-CORE3 cluster consolidation (2026-05-25)**: canonical registry of
// every `NEOETHOS_BOT_CTRADER_*` / `NEOETHOS_BOT_PNL_*` / `NEOETHOS_*` env
// override the app crate honours. Mirror of `neoethos_core::env_overrides`.
// Call-sites elsewhere consult `env_overrides::*` typed getters instead
// of reading `std::env::var` directly.
pub mod env_overrides;
// 2026-08-10 config consolidation: the tombstone list. Every env var this
// crate USED to read is named here, and a name still exported at startup
// produces an ERROR saying it was ignored and which config key replaced it.
// Non-negotiable #4: a deleted lever must announce itself, not fall silent.
pub mod retired_env;
pub mod broker_config;
pub mod broker_persistence;
pub mod ctrader_account;
pub mod ctrader_auth;
pub mod ctrader_bootstrap;
pub mod ctrader_data;
pub mod ctrader_errors;
pub mod ctrader_execution;
// 2026-08-09 dead-code purge (batch D2): `ctrader_history` (1,164 lines) was
// deleted. Every one of its public items had zero callers; the production
// history path is `broker_api::download_history_blocking` (POST /data/fetch,
// `server/data_control.rs:254`) and `broker_api::fetch_broker_order_history_blocking`
// (`data_control.rs:441`), both of which chunk the request — which
// `ctrader_history` never did.
#[cfg(test)]
mod ctrader_integration_tests;
pub mod ctrader_live_auth;
pub mod ctrader_messages;
pub mod ctrader_money;
// 2026-08-09 dead-code purge (batch D2): `ctrader_openapi` (the four
// protoc-generated `include!` blocks) was deleted along with the
// `crates/neoethos-app/proto/` directory, the codegen in `build.rs` and the
// `protobuf` / `protoc-bin-vendored` / `cc` build dependencies. Not one
// generated type was ever referenced — the cTrader wire format is hand-rolled
// JSON in `ctrader_messages.rs`.
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
// 2026-08-09 (#238, audit §2.11): the reader the broker's margin-call feed
// never had. `ctrader_messages::build_margin_call_list_request` had zero
// callers outside its own test, so an UNREALISED margin emergency reached the
// operator only through cTrader's own platform. This module polls the account's
// margin level, halts every order-OPENING route at
// `broker_api::prepare_new_order`, and starts the persisted 24h kill-switch
// cooldown. Spawned from `main.rs` alongside the other background services.
pub mod margin_call;
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
