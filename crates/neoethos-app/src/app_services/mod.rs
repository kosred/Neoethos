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
// Only compiled when the gemma-backend feature is on — the entire
// module is consumed by the feature-gated `chat_impl` in
// `server/gemma.rs`. Without the feature the LLM endpoints return
// 503 before any tool dispatch happens, so the registry, parser,
// and tools themselves are unreachable. Gating the module avoids
// 30+ false-positive dead-code warnings in the default build.
#[cfg(feature = "gemma-backend")]
pub mod gemma_memory;
#[cfg(feature = "gemma-backend")]
pub mod gemma_news_watcher;
#[cfg(feature = "gemma-backend")]
pub mod gemma_tools;
// news_sources is feature-gated alongside the gemma stack today.
// When a non-LLM consumer lands (planned dashboard newsfeed widget,
// headless CLI `--scan-news` mode), open this up by dropping the
// `cfg` and the only adjustment needed is removing dead-code
// allows that the gating currently obviates. Until then, gating
// keeps the default build free of unused-module warnings.
#[cfg(feature = "gemma-backend")]
pub mod news_sources;
pub mod risky_mode_persistence;
pub mod secure_store;
pub mod signal_journal;
pub mod trading;
pub mod training;

use crate::app_services::jobs::JobSnapshot;

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    DiscoveryUpdated(JobSnapshot),
    TrainingUpdated(JobSnapshot),
    // Sent by trading::session::start_connect (allow-listed legacy TradingSession
    // method) and inspected only via Debug logging on the event bus.
    #[allow(dead_code)]
    CTraderConnectUpdated(crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot),
    // Sent by trading::session at 3 sites for the bootstrap progress stream;
    // the inner snapshot is currently only surfaced via Debug logging on the
    // event bus. Field read is provided for the future bootstrap-progress UI.
    #[allow(dead_code)]
    BootstrapUpdated(JobSnapshot),
    // Sent by trading::session at start_connect failure paths; inspected only
    // via Debug logging until the Flutter UI subscribes for live status text.
    #[allow(dead_code)]
    ConnectOutcome(Result<String, String>),
    /// Background chart-data fetch completed. The UI should refresh the
    /// chart panel without blocking the render thread on WebSocket I/O.
    // Constructed in trading::session::1150; inner snapshot consumed via
    // Debug logging on the event bus until the chart panel reads it directly.
    #[allow(dead_code)]
    ChartDataUpdated(Box<crate::app_services::trading::MarketChartSnapshot>),
    /// A background OS thread spawned via
    /// `app_services::trading::background::spawn_background_task` panicked.
    /// The panic was caught inside the worker so the process is not killed,
    /// but the operator MUST see it — previously a panic in (e.g.) the chart
    /// fetcher left the UI showing "Running…" forever because the join
    /// handle was simply discarded.
    // Fields are read by trading::background tests (panic_with_string_payload_is_surfaced,
    // panic_with_static_str_payload_is_surfaced); production reads via Debug logging.
    #[allow(dead_code)]
    BackgroundTaskPanic {
        task: String,
        message: String,
    },
}
