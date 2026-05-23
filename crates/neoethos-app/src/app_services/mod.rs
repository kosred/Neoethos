pub mod api_test;
pub mod backoff;
pub mod bootstrap_writer;
pub mod broker_config;
// `broker_control` — RESTORED 2026-05-21. Same wrongful-delete fix
// as `dxtrade` above. The global OnceLock crossbeam channel is the
// designed-but-not-yet-installed bridge between the streaming worker
// and the UI loop for HardwareKill / ConnectionRestored signals.
// Streaming Task #3 today routes via `ServiceEvent` instead, but the
// broker_control path stays available for cases where a Send-Sync
// global is preferable to a tokio mpsc (e.g. a panic-safe HALT path
// from a non-async worker).
pub mod broker_control;
pub mod broker_persistence;
pub mod ctrader_account;
pub mod ctrader_auth;
pub mod ctrader_bootstrap;
pub mod ctrader_data;
pub mod ctrader_execution;
pub mod ctrader_history;
#[cfg(test)]
mod ctrader_integration_tests;
pub mod ctrader_live_auth;
pub mod ctrader_messages;
pub mod ctrader_money;
pub mod ctrader_openapi;
pub mod ctrader_proto_messages;
pub mod ctrader_session;
pub mod ctrader_state_machine;
pub mod ctrader_streaming;
pub mod ctrader_tls;
pub mod discovery;
// `dxtrade` — RESTORED 2026-05-21 after a wrongful deletion the same
// day. The operator directive is explicit: DXtrade is planned to
// become a fully-wired adapter alongside cTrader. The module owns the
// REST + WebSocket client, OAuth-style session token handshake, order
// REST API, and streaming Push API (Phases D3.1–D3.3 in the module
// docs). Audit 2026-05-20 noted zero UI callers TODAY, but that's a
// wiring-pending state, not an abandonment. When TradingSession
// starts dispatching DxTrade through the runtime adapter trait, the
// allow below comes off.
pub mod broker_api;
pub mod dxtrade;
pub mod embedded_credentials;
pub mod jobs;
pub mod live_journal;
pub mod pnl;
pub mod reauth;
pub mod risky_mode_persistence;
pub mod secure_store;
pub mod trading;
pub mod training;

use crate::app_services::jobs::JobSnapshot;

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
    LlmNewsUpdated(String),
    Heartbeat,
    CTraderConnectUpdated(crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot),
    BootstrapUpdated(JobSnapshot),
    ConnectOutcome(Result<String, String>),
    /// Background chart-data fetch completed. The UI should refresh the
    /// chart panel without blocking the render thread on WebSocket I/O.
    ChartDataUpdated(Box<crate::app_services::trading::MarketChartSnapshot>),
    /// A background OS thread spawned via
    /// `app_services::trading::background::spawn_background_task` panicked.
    /// The panic was caught inside the worker so the process is not killed,
    /// but the operator MUST see it — previously a panic in (e.g.) the chart
    /// fetcher left the UI showing "Running…" forever because the join
    /// handle was simply discarded.
    BackgroundTaskPanic {
        task: String,
        message: String,
    },
}
