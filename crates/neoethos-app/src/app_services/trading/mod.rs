// Imports re-exported via `pub(super) use` so the `session`, `orders`, and
// `market_data` sibling modules can pull common types out of `super::*`
// without each duplicating the full `crate::app_services::...` path list.
// `pub(super)` keeps these aliases visible only inside `trading::*`, so the
// external (`neoethos-app`) surface is unchanged.
pub(super) use crate::app_services::ServiceEvent;
pub(super) use crate::app_services::broker_config::{
    AdapterReadinessSnapshot, BrokerSessionState, BrokerSettingsReadiness, BrokerSettingsState,
    CTraderBrokerEnvironment,
};
pub(super) use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeBackend, CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot,
    ProductionCTraderAccountRuntimeBackend,
};
pub(super) use crate::app_services::ctrader_auth::{
    CTraderAuthSession, CTraderAuthSnapshot, CTraderTokenBundle,
};
pub(super) use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderSymbolLookupRequest, HistoricalBar, load_chart_history,
    resolve_symbol,
};
pub(super) use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, CTraderExecutionStatus, ProductionCTraderExecutionBackend,
};
pub(super) use crate::app_services::ctrader_history::{
    CTraderPositionOrderHistoryBackend, ProductionCTraderPositionOrderHistoryBackend,
};
pub(super) use crate::app_services::ctrader_live_auth::{
    CTraderAccountDiscoveryBackend, CTraderAccountDiscoveryRequest, CTraderEnvironment,
    CTraderLiveAuthBackend, CTraderLiveAuthResult, CTraderTokenRefreshRequest,
    ProductionCTraderLiveAuthBackend,
};
pub(super) use crate::app_services::ctrader_messages::{
    CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderCancelOrderRequest, CTraderClosePositionRequest,
    CTraderNewOrderRequest, CTraderOrderType, CTraderTimeInForce, CTraderTradeSide,
};
pub(super) use crate::app_services::ctrader_streaming::{
    CTraderLiveChartUpdate, CTraderLiveChartUpdateRequest, CTraderLiveStreamingBackend,
    ProductionCTraderLiveStreamingBackend, merge_live_spot_update_into_bars,
};
pub(super) use crate::app_services::jobs::{JobKind, JobSnapshot, JobState};
pub(super) use neoethos_core::{KillSwitchTier, RiskyModeConfig, RiskyModeManager};
// Batch 14 authoritative PnL path. Re-exported into `trading::*` so
// `orders.rs` can reach the helpers via `super::*` without a long
// fully-qualified path on every call site. Only the symbols `orders.rs`
// actually references are listed here — adding the rest would trigger
// `unused_imports` because the parser/scaler types stay encapsulated
// inside `pnl::` (callers reach them transitively via
// `fetch_unrealized_pnl_for_all_positions`).
pub(super) use crate::app_services::pnl::{
    PnLDriftCircuitBreaker, evaluate_pnl_drift_circuit_breaker,
    fetch_unrealized_pnl_for_all_positions,
};
#[cfg(test)]
pub(super) use crate::app_services::secure_store::{
    CTRADER_TOKEN_STORE_SERVICE, CTRADER_TOKEN_STORE_USER, CTraderSecureStore,
    KeyringSecretStoreBackend, LEGACY_CTRADER_TOKEN_STORE_SERVICE, LEGACY_CTRADER_TOKEN_STORE_USER,
};
pub(super) use crate::app_services::secure_store::{
    CTraderTokenStore, production_ctrader_token_store,
};
pub(super) use crate::app_state::{AppState, DataSource};
pub(super) use anyhow::Context;
pub(super) use neoethos_data::{discover_timeframes, load_symbol_timeframe};
pub(super) use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

pub mod auto_trade;
pub mod auto_trade_producer;
mod background;
mod client_order;
mod diagnostics;
pub mod ensemble_predictor_adapter;
mod market_data;
mod orders;
mod risk_gate;
mod session;
mod snapshots;

// Sub-module re-exports kept as plain (private) `use` so the source
// items can stay `pub(super)` in their carved-out submodules. The
// `session`, `orders`, `market_data`, `snapshots`, `risk_gate`, and
// `diagnostics` siblings reach these names via `use super::...`;
// private items in a parent module are visible to direct submodules,
// so no `pub(...)` modifier is required here. The trading-public
// surface is unchanged.
use client_order::{
    CTRADER_TOKEN_REFRESH_WINDOW_SECS, current_unix_seconds, next_client_order_seq,
};
use diagnostics::{
    extract_client_order_id_from_request, find_existing_client_order_id,
    format_ctrader_connect_error, format_ctrader_terminal_info, format_execution_journal_line,
    format_execution_outcome_status, non_empty_option, record_app_event,
    synthesize_idempotent_retry_outcome,
};
use risk_gate::{
    ctrader_protocol_volume_from_units, prop_firm_pre_trade_check,
    validate_and_convert_lot_size_to_ctrader_volume,
};
use snapshots::{
    MAX_CHART_CANDLES, chart_history_window_ms, preferred_chart_timeframe,
    run_ctrader_bootstrap_batch_with_context, supported_ctrader_chart_timeframes,
    sync_ctrader_discovered_accounts_into_targets, sync_discovered_accounts_with_targets,
};

// Public re-exports so the trading module surface is unchanged.
pub use snapshots::{
    build_execution_surface_snapshot_with_runtime,
    build_market_chart_snapshot_from_historical_bars, build_market_chart_snapshot_from_ohlcv,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAdapterKind {
    CTrader,
    DxTrade,
}

impl TradingAdapterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CTrader => "cTrader",
            Self::DxTrade => "DXtrade",
        }
    }

    pub fn integration_mode(self) -> &'static str {
        match self {
            Self::CTrader => "Remote Open API",
            Self::DxTrade => "Remote broker API",
        }
    }

    pub fn requires_local_terminal(self) -> bool {
        false
    }

    pub fn supports_market_data(self) -> bool {
        true
    }

    pub fn supports_live_orders(self) -> bool {
        true
    }

    /// True if the broker adapter implements `Cancel Pending Order` /
    /// `Close Open Position` round-trips. Note — the UI
    /// used to do `snapshot.adapter_name == "cTrader"` for these
    /// buttons, which permanently locked DXtrade (or any future
    /// adapter) out even if its execution backend handled the
    /// operations. Capability flags are the right abstraction.
    ///
    /// Today only cTrader has the wired backends (`CancelOrder` /
    /// `ClosePosition` via the cTrader Open API); DXtrade's draft
    /// backend does not yet implement them. Flip the DXtrade arm to
    /// `true` once `dxtrade.rs::cancel_order`/`close_position` ship.
    // Capability flags used by the Flutter UI's order/position screens
    // (planned wiring — broker_control HTTP endpoints will read these
    // before allowing Cancel/Close buttons to enable).
    #[allow(dead_code)]
    pub fn supports_order_cancellation(self) -> bool {
        match self {
            Self::CTrader => true,
            Self::DxTrade => false,
        }
    }

    /// True if the broker adapter implements `Close Open Position`.
    /// Separated from `supports_order_cancellation` because the two
    /// capabilities CAN diverge per broker (e.g. an adapter that
    /// supports cancelling resting orders but only flattens positions
    /// via a counter-trade rather than a dedicated Close call).
    #[allow(dead_code)]
    pub fn supports_position_close(self) -> bool {
        match self {
            Self::CTrader => true,
            Self::DxTrade => false,
        }
    }
}

pub const SUPPORTED_TRADING_ADAPTERS: [TradingAdapterKind; 2] =
    [TradingAdapterKind::CTrader, TradingAdapterKind::DxTrade];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingPanelMode {
    LocalOnly,
    Disconnected,
    Connected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionSnapshot {
    pub adapter_name: String,
    pub integration_mode: String,
    pub requires_local_terminal: bool,
    pub supports_market_data: bool,
    pub supports_live_orders: bool,
    pub mode: TradingPanelMode,
    pub connected: bool,
    pub status_text: String,
    pub terminal_info: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChartCandle {
    pub timestamp: Option<i64>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChartOverlay {
    pub label: String,
    pub candle_index: usize,
    pub price: f64,
}

/// A single decision emitted by the bot (manual or AI-driven). Held
/// in [`TradingSession`] as a bounded ring buffer and exposed to the
/// chart panel as [`ChartOverlay`] markers — closing audit gap #11
/// "bot decision overlays on chart" (`Vec::new()` → real decisions).
///
/// Producer sites:
/// - Manual orders set by `execute_buy_order` / `execute_sell_order`
///   (`source = Manual`) so the operator can see their own fills on
///   the chart immediately.
/// - Future auto-trade pipeline (D1) will push `source = Ai` entries
///   with confidence + ensemble metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct BotDecisionEntry {
    /// Trading symbol the decision targets (e.g. `"EURUSD"`).
    pub symbol: String,
    /// Long / short / flat side.
    pub side: BotDecisionSide,
    /// Quoted price at the time of the decision (mid, or the fill
    /// price when known).
    pub price: f64,
    /// Unix-ms timestamp of the decision — used to map the entry to
    /// the nearest candle when building [`ChartOverlay`]s.
    pub timestamp_ms: i64,
    /// Short human label rendered on the chart (e.g. `"BUY 0.74"`).
    pub label: String,
    /// Where the decision came from. `Manual` for operator clicks,
    /// `Ai` for the auto-trade pipeline.
    pub source: BotDecisionSource,
    /// Optional confidence in `[0.0, 1.0]` for AI decisions. f64 per
    /// operator directive §7.2 — matches `AutoTradeSignal::confidence`
    /// at the boundary.
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotDecisionSide {
    Buy,
    Sell,
    Flat,
}

#[allow(dead_code)] // `Manual` variant scaffolded for the operator-
// origin path; only `Ai` is currently emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotDecisionSource {
    /// Operator clicked BUY/SELL in the UI.
    Manual,
    /// AI auto-trade pipeline emitted the signal.
    Ai,
}

/// Lightweight summary of which experts loaded vs missed vs
/// degraded — returned by [`TradingSession::start_auto_trade_with_ensemble`]
/// so the chrome can render a "Running ensemble: X/33 experts
/// active — Y missing, Z degraded" banner without holding a
/// reference to the live ensemble (which moves into the
/// predictor at start time). The 33 is `DEFAULT_BOOTSTRAP_EXPERT_NAMES.len()`
/// — 34 KNOWN_MODEL_NAMES minus the deferred `swarm_forecaster`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnsembleLoadSummary {
    /// Canonical names of experts that loaded healthy and are
    /// participating in inference.
    pub loaded_names: Vec<String>,
    /// Canonical names of experts whose artifact directory was
    /// absent on disk (training never ran for them).
    pub missing: Vec<String>,
    /// Canonical names of experts whose artifact existed but
    /// failed to load cleanly (corruption / version skew / backend
    /// init failure). The reason strings live in the
    /// EnsemblePredictor's `load_outcome().degraded` — chrome
    /// banners can choose to render names only.
    pub degraded_names: Vec<String>,
}

#[allow(dead_code)] // chrome banner helpers ("X/Y experts active")
// — wired into the AUTO ON status_msg as a
// string today; will be replaced with these
// accessors when the banner gains its own
// dedicated widget.
impl EnsembleLoadSummary {
    /// Count of experts actively participating in inference.
    pub fn loaded_count(&self) -> usize {
        self.loaded_names.len()
    }
    /// `true` when at least one expert loaded successfully — the
    /// producer can emit signals.
    pub fn has_any_loaded(&self) -> bool {
        !self.loaded_names.is_empty()
    }
}

/// Identifies which call site is asking `execute_ctrader_order` to
/// send a new order. The Risky Mode autonomous-only contract
/// (research §7.1 / [`neoethos_core::RiskyModeConfig::autonomous_only_contract_accepted`])
/// rejects every [`Self::Manual`] order when armed; AI signals from
/// the auto-trade dispatcher carry [`Self::Ai`] and bypass that
/// gate. Both still go through the rest of the Risky Mode tier
/// hierarchy + the prop-firm gate.
///
/// This is the v0.4.5 finish on the §7.1 autonomous-only invariant:
/// the operator's wizard acknowledgement isn't merely a config flag
/// — it actively blocks manual BUY/SELL while Risky Mode is armed.
// OrderSource::Manual is constructed by execute_buy_order / execute_sell_order
// (still maintained as test-only entry points for the order pipeline); the AI /
// AiApproved variants are constructed by auto_trade_producer in production.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSource {
    /// UI button click (`execute_buy_order` / `execute_sell_order`)
    /// or any human-driven order entry. Rejected by Risky Mode when
    /// [`neoethos_core::RiskyModeManager::rejects_manual_orders`] is
    /// true (autonomous-only contract signed).
    Manual,
    /// AI auto-trade dispatcher emit. Passes the manual-order
    /// rejection gate; still subject to every other Risky Mode tier
    /// (per-trade, per-day, per-stage, per-month, manual-halt,
    /// hardware-halt, pre-send sanity) and the prop-firm gate.
    Ai,
    /// AI-suggested trade that the user explicitly **Approved**
    /// via a conversational helper. Distinct from [`Self::Ai`] so
    /// the audit log can slice "user-approved suggestions" cleanly
    /// from autonomous-ensemble trades. Passes the same gates as
    /// `Ai` — every Risky Mode tier + the prop-firm gate still
    /// applies. NEVER set without an explicit user-click Approve.
    #[allow(dead_code)]
    AiSuggested,
}

#[allow(dead_code)] // `is_ai_originated` is used by the Risky Mode
// manual-rejection gate, which fires only when
// Risky Mode is armed (off by default). Kept
// public for the gate tests in trading_tests.rs.
impl OrderSource {
    /// `true` when the source is AI-originated (autonomous OR
    /// user-approved suggestion). Risky Mode treats both the
    /// same way for the manual-rejection gate.
    pub fn is_ai_originated(self) -> bool {
        matches!(self, Self::Ai | Self::AiSuggested)
    }
}

/// Hard cap on the decision ring buffer. 512 entries × ~120 bytes ≈
/// 60 KB — negligible memory, but enough to keep weeks of decisions
/// in scope at typical fill rates. Older entries get dropped FIFO.
pub const BOT_DECISION_BUFFER_CAPACITY: usize = 512;

#[derive(Debug, Clone, PartialEq)]
pub struct MarketChartSnapshot {
    pub symbol: String,
    pub timeframe: String,
    pub available_timeframes: Vec<String>,
    pub candles: Vec<ChartCandle>,
    pub overlays: Vec<ChartOverlay>,
    pub price_min: f64,
    pub price_max: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub price_change_pct: Option<f64>,
    pub headline: String,
    pub overlay_status: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionAction {
    pub label: String,
    pub enabled: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSelectionOption {
    pub id: i64,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionTicketSnapshot {
    pub lot_size: f64,
    pub slippage_in_points: i32,
    pub comment: String,
    pub label: String,
    pub max_lot_size: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionSurfaceSnapshot {
    pub symbol: String,
    pub adapter_name: String,
    pub integration_mode: String,
    pub connection_status: String,
    pub supported_adapters: Vec<String>,
    pub primary_actions: Vec<ExecutionAction>,
    pub warnings: Vec<String>,
    pub diagnostics: Vec<String>,
    pub positions: Vec<String>,
    pub pending_orders: Vec<String>,
    pub bot_timeline: Vec<String>,
    pub history_rows: Vec<String>,
    pub journal_rows: Vec<String>,
    pub selected_position_id: Option<i64>,
    pub selected_order_id: Option<i64>,
    pub position_choices: Vec<ExecutionSelectionOption>,
    pub pending_order_choices: Vec<ExecutionSelectionOption>,
    pub ticket: ExecutionTicketSnapshot,
}

/// Trading-environment classifier used by operator status surfaces and
/// consulted by the HALT flow to label its audit-log lines.
///
/// Maps verbatim to the four autonomy stages defined in
/// `docs/audits/research/wizard_onboarding_competitive_analysis.md`
/// §10.2 (Discovery -> Paper -> LiveSmall -> LiveFull). The wizard
/// promotion gates (§10.3) own the transitions; the chrome only
/// observes which mode the session currently sits in.
///
/// `Paper`/`LiveSmall`/`LiveFull` are scaffolded for the §10.3
/// promotion ladder (Discovery → Paper → LiveSmall → LiveFull) but
/// the chrome only emits `Demo` / `Live` today; the additional
/// variants will be set by the auto-promotion code that's queued
/// behind Task #10 (auto-trigger after wizard Apply).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingEnvironment {
    /// Historical-replay only, no broker connection. Pill: gray.
    /// Matches §10.2 "Discovery"/Stage 1 — sweeping templates against
    /// the historical cache with no live data.
    Demo,
    /// Live broker data, simulated execution. Pill: amber.
    /// Matches §10.2 "Paper trade"/Stage 2 — forward test on live
    /// streaming data with simulated fills (ThinkOrSwim paperMoney
    /// pattern, competitive analysis §1.4).
    #[allow(dead_code)] // §10.3 promotion ladder not yet wired
    Paper,
    /// Real money, capped per-trade size by the promotion gate. Pill: red.
    /// Matches §10.2 "Live small"/Stage 3 — uses real capital with the
    /// `min_trading_days=10` + `min_monthly_net_profit_pct=0.04` gate.
    #[allow(dead_code)]
    LiveSmall,
    /// Real money, no extra cap. Pill: red, bold.
    /// Matches §10.2 "Live full"/Stage 4 — uncapped after 30 days at
    /// Stage 3 with all promotion gates passed.
    #[allow(dead_code)]
    LiveFull,
}

#[allow(dead_code)] // `is_live_money` is consumed by the audit-log
// severity classifier, which is wired but only
// fires on environments that don't exist yet.
impl TradingEnvironment {
    /// Operator-facing label rendered inside the status pill.
    pub fn pill_label(self) -> &'static str {
        match self {
            Self::Demo => "DEMO",
            Self::Paper => "PAPER",
            Self::LiveSmall => "LIVE SMALL",
            Self::LiveFull => "LIVE",
        }
    }

    /// Whether this environment ever places real orders. Used by
    /// audit-log lines so a HALT during Demo is recorded with the
    /// correct severity.
    pub fn is_live_money(self) -> bool {
        matches!(self, Self::LiveSmall | Self::LiveFull)
    }
}

/// Per-session HALT state — the T-Manual layer in the kill-switch
/// hierarchy from `wizard_onboarding_competitive_analysis.md` §10.4.
///
/// HALT sits ABOVE T1 (per-trade) and T2 (per-day) which live in
/// `risk_gate.rs`. Once tripped, every new order is rejected at the
/// pre-trade gate regardless of other thresholds; the only way out is
/// the operator clearing the sentinel file via the "Clear HALT" banner.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HaltState {
    /// `true` once `trip_manual_halt` has been called and the
    /// `Clear HALT` banner has not yet flipped it back.
    pub halted: bool,
    /// Absolute path to the sentinel file written by
    /// `trip_manual_halt` (`<data-dir>/HALTED_<unix-secs>.flag`).
    /// `None` until a HALT has been tripped.
    pub sentinel_path: Option<PathBuf>,
    /// Stats from the most recent trip — for banner display and tests.
    pub last_trip: Option<HaltTripSummary>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HaltTripSummary {
    pub timestamp_unix_secs: u64,
    pub positions_closed: usize,
    pub orders_cancelled: usize,
    pub account_id: String,
    pub environment_label: String,
}

/// Read-only UI snapshot of the live Risky Mode manager. Built by
/// [`TradingSession::risky_mode_state`] when Risky Mode is active; the
/// chrome stage-progress bar / kill-switch banner consume this via the
/// status pill — they never touch [`RiskyModeManager`] directly.
///
/// Mirrors the fields required by research §7.2 (stage progress),
/// §7.4 (kill-switch banner), and §7.5 (retreat indicator). All
/// numeric fields are `f64` per operator directive §7.2 — f64 carries
/// ~15-16 decimal digits, which keeps cents accurate at the $50,000-
/// target scale. The earlier f32 build is retired with this snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskyModeState {
    /// Zero-based stage index (`0 = S1`). Matches
    /// [`neoethos_core::RiskyStage::stage_idx`].
    pub current_stage_idx: u8,
    /// Total number of stages in the configured taper. Used by the UI
    /// to render "Stage 3 / 16".
    pub total_stages: u8,
    /// Live bankroll in USD, updated by `record_trade_outcome` at
    /// every close.
    pub current_bankroll_usd: f64,
    /// Lower edge of the current stage's bankroll window — feeds the
    /// stage-progress bar.
    pub stage_bankroll_lower_usd: f64,
    /// Upper edge of the current stage's bankroll window.
    pub stage_bankroll_upper_usd: f64,
    /// Cumulative daily loss in USD (positive number = loss).
    pub daily_loss_accumulated_usd: f64,
    /// Cumulative monthly loss in USD (positive number = loss).
    pub monthly_loss_accumulated_usd: f64,
    /// Last kill-switch trip — `None` when no halt has fired since
    /// construction or the last `clear_halt`.
    pub last_kill_switch_trip: Option<(KillSwitchTier, chrono::DateTime<chrono::Utc>)>,
    /// Heuristic ruin-probability estimate at the current stage
    /// (research §9.3 Brownian-motion formula).
    pub ruin_probability_estimate: f64,
}

// TradingSession is the legacy egui session struct. Many of its fields are
// initialised but never read in production because the HTTP server has its own
// pipelines. Kept whole as the test fixture for trading_tests.rs (391 tests
// rely on this). When the test surface is migrated to direct ctrader_* helper
// calls, this struct can be slimmed down.
#[allow(dead_code)]
pub struct TradingSession {
    configured_adapter: TradingAdapterKind,
    broker_settings: BrokerSettingsState,
    ctrader_auth: Option<CTraderAuthSession>,
    ctrader_live_auth_backend: Arc<dyn CTraderLiveAuthBackend>,
    ctrader_account_discovery_backend: Arc<dyn CTraderAccountDiscoveryBackend>,
    ctrader_account_runtime_backend: Arc<dyn CTraderAccountRuntimeBackend>,
    ctrader_execution_backend: Arc<dyn CTraderExecutionBackend>,
    ctrader_position_order_history_backend: Arc<dyn CTraderPositionOrderHistoryBackend>,
    ctrader_live_streaming_backend: Arc<dyn CTraderLiveStreamingBackend>,
    ctrader_token_store: Arc<dyn CTraderTokenStore>,
    ctrader_live_auth_rx: Option<Receiver<Result<CTraderLiveAuthResult, String>>>,
    adapter: Option<TradingAdapter>,
    connected: bool,
    terminal_info: String,
    market_chart_cache: Option<CachedMarketSnapshot>,
    execution_surface_cache: Option<CachedExecutionSnapshot>,
    ctrader_live_spot_cache: Option<CachedCTraderLiveSpotUpdate>,
    trade_journal: Vec<String>,
    initial_equity: Option<f64>,
    day_start_equity: Option<f64>,
    /// Broker-time day id (`unix_ms / 86_400_000`). When the periodic refresh
    /// observes a new day id we reset `day_start_equity` via
    /// `handle_day_boundary`; otherwise the daily-DD reference would be
    /// frozen at session start (D6 in the prop-firm safety audit).
    last_observed_day_id: Option<i64>,
    ctrader_runtime_refreshed_at: Option<Instant>,
    connect_handle: Option<std::thread::JoinHandle<()>>,
    /// Note — atomic re-entrancy guard for `start_connect`.
    /// The previous design relied on `background_task_running(Connect)`,
    /// which races against the very brief window between
    /// `std::thread::spawn` and the OS scheduling the worker thread: two
    /// rapid clicks could both see `is_finished() == false` (because no
    /// thread had been scheduled yet) and BOTH spawn. The atomic flag
    /// closes the race — `compare_exchange(false, true)` only succeeds
    /// for one caller. Reset by the spawn closure's RAII guard.
    connect_in_flight: Arc<AtomicBool>,
    bootstrap_handle: Option<std::thread::JoinHandle<()>>,
    /// Background handle for the cTrader chart-data fetch. A live
    /// connection opens a fresh WSS socket per call, so this is run
    /// off the request path to avoid blocking callers.
    chart_fetch_handle: Option<std::thread::JoinHandle<()>>,
    /// Completed chart snapshot waiting to be promoted into the regular
    /// `market_chart_cache`. Set by `apply_chart_data_update` when
    /// `ServiceEvent::ChartDataUpdated` arrives; consumed by the next
    /// `market_chart_snapshot` call.
    pending_ctrader_chart: Option<MarketChartSnapshot>,
    /// Trading environment for the session — used by the status pill
    /// and HALT button. Defaults to Demo; the wizard / autonomy
    /// controller (§10.3 promotion gates) is responsible for advancing
    /// this to Paper / LiveSmall / LiveFull.
    trading_environment: TradingEnvironment,
    /// T-Manual kill switch (HALT). See `HaltState` docs for details.
    halt_state: HaltState,
    /// Risky Mode state machine (research §4–§5). `None` when the
    /// session is in Standard mode; populated by
    /// [`TradingSession::enable_risky_mode`] when the wizard's
    /// Step 3 risk slider reaches `10` (or the operator activates
    /// Risky Mode at runtime). When `Some(_)`, the
    /// [`KillSwitchTier`] gate runs BEFORE
    /// [`risk_gate::prop_firm_pre_trade_check`] inside
    /// `execute_ctrader_order` — Risky Mode is the strictly tighter
    /// outer layer (research §11.3, operator directive 2026-05-15).
    risky_mode_manager: Option<RiskyModeManager>,
    /// Ring buffer of recent bot decisions (manual fills + AI signals).
    /// Mapped to [`ChartOverlay`]s by `bot_decisions_to_overlays` and
    /// consumed by `market_data::load_*_market_chart_snapshot`. Capped
    /// at [`BOT_DECISION_BUFFER_CAPACITY`] entries; older entries are
    /// dropped FIFO. Closes audit gap #11.
    bot_decisions: Vec<BotDecisionEntry>,
    /// Live auto-trade producer handle. `Some(_)` when the producer
    /// thread is running on this session — the operator started it
    /// via `start_auto_trade_producer` and the session owns the
    /// cancel flag + signal receiver until `stop_auto_trade_producer`
    /// tears it down. The main app loop calls `drain_auto_trade_signals`
    /// on every tick to forward emitted signals through the §7.1
    /// dispatcher gate chain.
    auto_trade_producer: Option<auto_trade_producer::AutoTradeProducerHandle>,
}

enum TradingAdapter {
    CTrader(CTraderAccountRuntimeSnapshot),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MarketChartCacheKey {
    data_root: PathBuf,
    data_source: DataSource,
    adapter_kind: TradingAdapterKind,
    symbol: String,
    timeframe: String,
    ctrader_environment: Option<CTraderEnvironment>,
    ctrader_account_id: Option<String>,
}

#[derive(Debug, Clone)]
struct CachedMarketSnapshot {
    key: MarketChartCacheKey,
    refreshed_at: Instant,
    snapshot: MarketChartSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecutionSnapshotCacheKey {
    data_source: DataSource,
    symbol: String,
    adapter_kind: TradingAdapterKind,
    connected: bool,
}

#[derive(Debug, Clone)]
struct CachedExecutionSnapshot {
    key: ExecutionSnapshotCacheKey,
    refreshed_at: Instant,
    snapshot: ExecutionSurfaceSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CTraderLiveSpotCacheKey {
    environment: CTraderEnvironment,
    account_id: String,
    symbol_id: i64,
    timeframe: String,
}

#[derive(Debug, Clone)]
struct CachedCTraderLiveSpotUpdate {
    key: CTraderLiveSpotCacheKey,
    refreshed_at: Instant,
    update: CTraderLiveChartUpdate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CTraderBootstrapContext {
    client_id: String,
    client_secret: String,
    access_token: String,
    environment: CTraderEnvironment,
    account_id: String,
}

enum ExecutionFeedHandle<'a> {
    CTrader(&'a CTraderAccountRuntimeSnapshot),
    Unavailable { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppExecutionRuntimeSnapshot {
    CTrader(CTraderAccountRuntimeSnapshot),
}

// `#[allow(dead_code)]` covers the items the audit (2026-05-20)
// flagged as "never used" on this big public surface:
//   - `new` (callers use `new_with_persisted_credentials` after first
//     launch, but tests still construct via `new` so the symbol stays)
//   - environment / halt-state / risky-mode accessor + mutator family
//     (auto-armed at boot, no UI toggle wired)
//   - bot-decision recording + auto-trade-producer lifecycle helpers
//     (the wired AUTO ON toggle hits the `_for_state` convenience
//     above; the lower-level methods stay public for tests)
//   - `refresh_runtime` (sync variant; the heartbeat uses the async
//     `start_runtime_refresh_in_background` instead)
// Annotated at the impl level instead of deleting because every item
// has live test coverage AND constitutes the documented public API of
// TradingSession.
#[allow(dead_code)]
impl TradingSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a session with broker settings pre-loaded from the
    /// per-user credentials TOML file (see
    /// [`crate::app_services::broker_persistence::load_broker_settings`])
    /// AND Risky Mode auto-armed from
    /// [`crate::app_services::risky_mode_persistence::load_risky_mode_state`]
    /// when the wizard previously persisted an armed state. The
    /// auto-arm uses `RiskyModeConfig::default()` (research §4.1
    /// $20 → $50_000 logarithmic stage table) with `starting_capital_usd`
    /// from the persisted file falling back to the config default.
    /// Closes `TODO(risky-mode-boot-wire)` — the wizard's
    /// `risky_mode_armed` flag is now load-bearing across restarts.
    ///
    /// Used by `main.rs` so the production app starts with credentials the
    /// user has already saved. Tests should keep using [`Self::new`] /
    /// [`Self::with_configured_adapter_for_test`] which start with empty
    /// defaults and are unaffected by whatever is on the developer's disk.
    pub fn new_with_persisted_credentials() -> Self {
        let mut session = Self::default();
        session.broker_settings = crate::app_services::broker_persistence::load_broker_settings();
        session.auto_arm_risky_mode_from_persisted_state();
        session
    }

    /// Internal helper used by [`Self::new_with_persisted_credentials`]
    /// to apply the wizard's persisted Risky Mode arm decision at
    /// session boot. Idempotent: a re-call when no file is on disk is
    /// a no-op. Safe: when the persisted config fails
    /// `RiskyModeConfig::validate` (e.g. autonomous-only contract not
    /// accepted), the call logs an error and leaves Risky Mode
    /// disabled rather than panicking — a half-armed session is
    /// always worse than a disabled one.
    fn auto_arm_risky_mode_from_persisted_state(&mut self) {
        use crate::app_services::risky_mode_persistence::load_risky_mode_state;
        let state = match load_risky_mode_state() {
            Ok(Some(s)) => s,
            Ok(None) => return,
            Err(err) => {
                tracing::error!(
                    target: "neoethos_app::risky_mode",
                    error = %err,
                    "failed to load risky_mode_state.json at session boot; \
                     Risky Mode will stay disabled until the wizard re-runs"
                );
                return;
            }
        };

        if !state.armed {
            tracing::debug!(
                target: "neoethos_app::risky_mode",
                "persisted risky_mode_state.json present but disarmed; \
                 skipping auto-arm"
            );
            return;
        }

        let mut config = RiskyModeConfig::default();
        config.autonomous_only_contract_accepted = state.autonomous_only_contract_accepted;
        if let Some(ack) = state.ruin_ceiling_acknowledged {
            config.acknowledged_ruin_probability_ceiling = ack;
        }
        let starting_bankroll = state
            .starting_capital_usd
            .unwrap_or(config.starting_capital_usd);

        match self.enable_risky_mode(config, starting_bankroll) {
            Ok(()) => {
                tracing::info!(
                    target: "neoethos_app::risky_mode",
                    starting_bankroll,
                    "Risky Mode auto-armed from persisted state"
                );
            }
            Err(err) => {
                tracing::error!(
                    target: "neoethos_app::risky_mode",
                    error = %err,
                    "persisted risky_mode_state.json was armed but \
                     enable_risky_mode rejected the config; \
                     Risky Mode stays disabled. Re-run the wizard's \
                     AutonomyRisk step to fix the config."
                );
            }
        }
    }

    #[cfg(test)]
    pub fn with_configured_adapter_for_test(kind: TradingAdapterKind) -> Self {
        Self {
            configured_adapter: kind,
            broker_settings: BrokerSettingsState::default(),
            ctrader_auth: None,
            ctrader_live_auth_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_discovery_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_runtime_backend: Arc::new(ProductionCTraderAccountRuntimeBackend),
            ctrader_execution_backend: Arc::new(ProductionCTraderExecutionBackend),
            ctrader_position_order_history_backend: Arc::new(
                ProductionCTraderPositionOrderHistoryBackend,
            ),
            ctrader_live_streaming_backend: Arc::new(ProductionCTraderLiveStreamingBackend),
            ctrader_token_store: Arc::new(CTraderSecureStore::new(
                LEGACY_CTRADER_TOKEN_STORE_SERVICE,
                LEGACY_CTRADER_TOKEN_STORE_USER,
                KeyringSecretStoreBackend,
            )),
            ctrader_live_auth_rx: None,
            adapter: None,
            connected: false,
            terminal_info: String::new(),
            market_chart_cache: None,
            execution_surface_cache: None,
            ctrader_live_spot_cache: None,
            trade_journal: Vec::new(),
            initial_equity: None,
            day_start_equity: None,
            last_observed_day_id: None,
            ctrader_runtime_refreshed_at: None,
            connect_handle: None,
            connect_in_flight: Arc::new(AtomicBool::new(false)),
            bootstrap_handle: None,
            chart_fetch_handle: None,
            pending_ctrader_chart: None,
            trading_environment: TradingEnvironment::Demo,
            halt_state: HaltState::default(),
            risky_mode_manager: None,
            bot_decisions: Vec::new(),
            auto_trade_producer: None,
        }
    }

    #[cfg(test)]
    pub fn set_ctrader_store_for_test(
        &mut self,
        backend: crate::app_services::secure_store::MemorySecretStoreBackend,
    ) {
        self.ctrader_token_store = Arc::new(CTraderSecureStore::new(
            CTRADER_TOKEN_STORE_SERVICE,
            CTRADER_TOKEN_STORE_USER,
            backend,
        ));
    }

    #[cfg(test)]
    pub fn seed_ctrader_token_bundle_for_test(
        &self,
        bundle: crate::app_services::ctrader_auth::CTraderTokenBundle,
    ) -> anyhow::Result<()> {
        self.ctrader_token_store.save_token_bundle(&bundle)
    }

    #[cfg(test)]
    pub fn set_ctrader_live_auth_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend,
    ) {
        self.ctrader_live_auth_backend = Arc::new(backend);
    }

    #[cfg(test)]
    pub fn set_ctrader_account_discovery_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend,
    ) {
        self.ctrader_account_discovery_backend = Arc::new(backend);
    }

    #[cfg(test)]
    pub fn set_ctrader_account_runtime_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend,
    ) {
        self.ctrader_account_runtime_backend = Arc::new(backend);
    }

    #[cfg(test)]
    pub fn set_ctrader_execution_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_execution::StubCTraderExecutionBackend,
    ) {
        self.ctrader_execution_backend = Arc::new(backend);
    }

    #[cfg(test)]
    pub fn set_ctrader_position_order_history_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_history::StubCTraderPositionOrderHistoryBackend,
    ) {
        self.ctrader_position_order_history_backend = Arc::new(backend);
    }

    #[cfg(test)]
    pub fn set_ctrader_live_streaming_backend_for_test(
        &mut self,
        backend: crate::app_services::ctrader_streaming::StubCTraderLiveStreamingBackend,
    ) {
        self.ctrader_live_streaming_backend = Arc::new(backend);
        self.ctrader_live_spot_cache = None;
    }

    // Session-lifecycle / auth methods moved to `session.rs` (Batch 6):
    //   is_connected, configured_adapter, broker_settings_mut,
    //   adapter_readiness, can_attempt_connect, ctrader_auth_snapshot,
    //   start_ctrader_bootstrap_batch, start_ctrader_auth,
    //   receive_ctrader_authorization_code,
    //   build_ctrader_token_exchange_request, start_ctrader_live_auth,
    //   poll_ctrader_live_auth, restore_ctrader_session,
    //   clear_ctrader_saved_session, discover_ctrader_accounts.

    /// Current trading environment classifier consumed by the
    /// persistent status pill in the main chrome.
    pub fn trading_environment(&self) -> TradingEnvironment {
        self.trading_environment
    }

    /// Adjust the session's trading environment. Owned by the wizard
    /// promotion gates (`wizard_onboarding_competitive_analysis.md`
    /// §10.3); the chrome only renders what is already set.
    pub fn set_trading_environment(&mut self, env: TradingEnvironment) {
        self.trading_environment = env;
    }

    /// Read-only view of the current HALT state. Used by the chrome
    /// to decide whether to render the "TRADING HALTED" banner.
    pub fn halt_state(&self) -> &HaltState {
        &self.halt_state
    }

    /// `true` once `trip_manual_halt` has set the flag and the
    /// operator has not yet cleared it. Consulted by every order
    /// path (T-Manual sits above T1 / T2 in §10.4 of the
    /// competitive analysis).
    pub fn is_halted(&self) -> bool {
        self.halt_state.halted
    }

    /// T-Manual kill switch — the red HALT button in the chrome.
    ///
    /// Sequence (mirrors `wizard_onboarding_competitive_analysis.md`
    /// §10.4 T4):
    ///   1. Sets the `halted` flag so every subsequent order is
    ///      rejected at the pre-trade gate.
    ///   2. Iterates open positions and calls the existing close path
    ///      (`close_selected_position`) for each — preserves the
    ///      audit-logging layer; does NOT introduce a bypass.
    ///   3. Iterates pending orders and calls the existing cancel
    ///      path (`cancel_selected_order`) for each.
    ///   4. Calls `risky_mode_manager.trip_manual_halt()` if Risky
    ///      Mode is active in the session — keeps the Risky Mode
    ///      kill-switch tier coherent with T-Manual (research §5.5).
    ///   5. Emits a `tracing::error!(target: "neoethos_app::halt", ...)`
    ///      line carrying operator, account_id, positions_closed,
    ///      orders_cancelled, environment.
    ///   6. Writes a sentinel file `<data-dir>/HALTED_<unix-secs>.flag`
    ///      that the operator must remove (or the "Clear HALT"
    ///      banner button removes for them) before trading can resume.
    ///
    /// Returns the `HaltTripSummary` so callers (and tests) can
    /// inspect what was closed.
    pub fn trip_manual_halt(&mut self, state: &mut AppState) -> HaltTripSummary {
        // (1) flip the gate FIRST so any concurrent order submission
        // landing during the iteration below is rejected at the gate.
        self.halt_state.halted = true;

        // (2) snapshot the positions / orders BEFORE iterating so that
        // close/cancel calls (which mutate `state.order_ticket`) do not
        // perturb the list we are walking.
        let (position_ids, order_ids) = match self.connected_ctrader_runtime() {
            Some(runtime) => {
                let positions: Vec<i64> = runtime
                    .reconcile
                    .positions
                    .iter()
                    .map(|p| p.position_id)
                    .collect();
                let orders: Vec<i64> = runtime
                    .reconcile
                    .pending_orders
                    .iter()
                    .map(|o| o.order_id)
                    .collect();
                (positions, orders)
            }
            None => (Vec::new(), Vec::new()),
        };

        let mut positions_closed = 0usize;
        for position_id in &position_ids {
            state.order_ticket.selected_position_id = Some(*position_id);
            // Existing close path: hard-fails on bad account ids and
            // routes through `execute_ctrader_request`, which is the
            // audit-logging entry point. Re-use it verbatim per the
            // operator constraint "HALT must use the existing
            // close/cancel paths".
            self.close_selected_position(state);
            positions_closed += 1;
        }

        let mut orders_cancelled = 0usize;
        for order_id in &order_ids {
            state.order_ticket.selected_order_id = Some(*order_id);
            self.cancel_selected_order(state);
            orders_cancelled += 1;
        }

        // (4) Risky Mode kill-switch composition. When Risky Mode is
        // active the manual halt also trips the Risky Mode sticky
        // halt — research §5.5 explicitly couples the operator UI
        // panic-flatten to the per-mode kill-switch tier so a later
        // `execute_ctrader_order` cannot slip past Risky Mode's
        // sanity gate even if the operator clears `halt_state.halted`
        // without re-enabling the Risky Mode side. See
        // `neoethos-core/src/domain/risky_mode.rs::trip_manual_halt`.
        if let Some(rm) = self.risky_mode_manager.as_mut() {
            rm.trip_manual_halt();
        }

        let account_id = self
            .selected_ctrader_execution_account_id()
            .unwrap_or_default();
        let env_label = self.trading_environment.pill_label().to_string();
        let timestamp_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // (5) structured error-level event for the operator log.
        // Includes everything an auditor needs to reconstruct the
        // HALT after the fact.
        tracing::error!(
            target: "neoethos_app::halt",
            account_id = %account_id,
            environment = %env_label,
            positions_closed,
            orders_cancelled,
            "T-Manual HALT tripped"
        );

        // (6) sentinel file. Written under the app's data directory
        // so the operator can `ls` it from the shell and confirm the
        // halt is still in force. We do NOT fail the halt if the
        // sentinel write fails — the in-memory `halted` flag is
        // authoritative; the file is a durable record.
        let sentinel_path = state
            .runtime
            .data_dir
            .join(format!("HALTED_{timestamp_unix_secs}.flag"));
        if let Some(parent) = sentinel_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let sentinel_body = format!(
            "T-Manual HALT tripped\n\
             timestamp_unix_secs={timestamp_unix_secs}\n\
             account_id={account_id}\n\
             environment={env_label}\n\
             positions_closed={positions_closed}\n\
             orders_cancelled={orders_cancelled}\n"
        );
        match std::fs::write(&sentinel_path, sentinel_body) {
            Ok(()) => {
                self.halt_state.sentinel_path = Some(sentinel_path.clone());
            }
            Err(err) => {
                tracing::error!(
                    target: "neoethos_app::halt",
                    error = %err,
                    path = %sentinel_path.display(),
                    "HALT sentinel write failed; halt remains in force via in-memory flag"
                );
            }
        }

        let summary = HaltTripSummary {
            timestamp_unix_secs,
            positions_closed,
            orders_cancelled,
            account_id,
            environment_label: env_label,
        };
        self.halt_state.last_trip = Some(summary.clone());
        summary
    }

    /// Public-surface read-through of the broker session's
    /// "currently selected execution account id". Used by the chrome
    /// status pill so it can render `<env> · <account_id>` without
    /// reaching into the private `broker_settings` field. Returns
    /// `None` when no broker accounts have been discovered or the
    /// operator has not yet flagged one for execution.
    pub fn selected_ctrader_execution_account_id_public(&self) -> Option<String> {
        self.selected_ctrader_execution_account_id()
    }

    /// Clear the HALT and allow new orders to flow again. Wired to
    /// the "Clear HALT" button in the chrome banner. Removes the
    /// sentinel file from disk so a fresh `ls <data-dir>` shows the
    /// halt is no longer active.
    pub fn clear_halt(&mut self) {
        self.halt_state.halted = false;
        if let Some(path) = self.halt_state.sentinel_path.take() {
            // Best-effort: a missing sentinel (e.g. operator removed
            // it manually) is fine; surface IO errors only as warn.
            if let Err(err) = std::fs::remove_file(&path) {
                if err.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        target: "neoethos_app::halt",
                        error = %err,
                        path = %path.display(),
                        "HALT sentinel remove failed during clear_halt"
                    );
                }
            }
        }
        tracing::info!(
            target: "neoethos_app::halt",
            "T-Manual HALT cleared by operator"
        );
    }

    // ── Bot decision overlays (audit gap #11) ───────────────────────────
    //
    // The chart panel paints `MarketChartSnapshot.overlays`. Before this
    // patch every snapshot was built with `Vec::new()` — the type existed,
    // the paint code existed, but no producer connected the two.
    // `record_bot_decision` is the producer; `bot_decisions_to_overlays`
    // converts the buffered entries into renderable markers by mapping each
    // decision's timestamp to the nearest candle index in the visible
    // window.

    /// Push a new bot decision onto the ring buffer. Older entries are
    /// dropped FIFO once the buffer hits
    /// [`BOT_DECISION_BUFFER_CAPACITY`].
    ///
    /// Called from manual order paths (`execute_buy_order` /
    /// `execute_sell_order` → `record_decision_for_fill`) and — when D1
    /// lands — from the AI auto-trade pipeline. Idempotency: nothing
    /// filters duplicates; multiple consecutive fills at the same price
    /// produce multiple markers, which matches operator expectations.
    pub fn record_bot_decision(&mut self, entry: BotDecisionEntry) {
        self.bot_decisions.push(entry);
        // Cap the buffer. `Vec::drain(..delta)` is O(N) but only
        // executes when we cross the cap; at typical fill rates this
        // happens days apart.
        let len = self.bot_decisions.len();
        if len > BOT_DECISION_BUFFER_CAPACITY {
            let delta = len - BOT_DECISION_BUFFER_CAPACITY;
            self.bot_decisions.drain(..delta);
        }
    }

    /// All recorded decisions for `symbol`, oldest first. Returns
    /// `&[]` if none — never `Option<_>` to keep the call sites flat.
    pub fn bot_decisions_for(&self, symbol: &str) -> Vec<&BotDecisionEntry> {
        self.bot_decisions
            .iter()
            .filter(|d| d.symbol == symbol)
            .collect()
    }

    /// Convert recorded decisions for `symbol` into `ChartOverlay`
    /// markers that align with `candles`. A decision is mapped to the
    /// candle whose `timestamp` is closest (and not after — we never
    /// paint a marker on a future candle that doesn't exist yet).
    /// Decisions older than the first candle or newer than the last
    /// are dropped silently.
    pub fn bot_decisions_to_overlays(
        &self,
        symbol: &str,
        candles: &[ChartCandle],
    ) -> Vec<ChartOverlay> {
        if candles.is_empty() {
            return Vec::new();
        }
        let mut overlays = Vec::new();
        for d in self.bot_decisions_for(symbol) {
            if let Some(idx) = nearest_candle_index(candles, d.timestamp_ms) {
                overlays.push(ChartOverlay {
                    label: d.label.clone(),
                    candle_index: idx,
                    price: d.price,
                });
            }
        }
        overlays
    }

    #[cfg(test)]
    pub fn bot_decision_buffer_len(&self) -> usize {
        self.bot_decisions.len()
    }

    // ── Risky Mode integration (research §4–§5). The composition
    // model is: the session holds at most one `RiskyModeManager`; when
    // it is `Some(_)`, the `execute_ctrader_order` pipeline runs the
    // Risky Mode `check_trade_allowed` gate BEFORE the prop-firm
    // `prop_firm_pre_trade_check` (§11.3 — Risky Mode is the tighter
    // outer layer, NOT a replacement for FTMO). The HALT button
    // composes both kill switches (research §5.5). Stage advancement
    // is driven by `record_trade_outcome` calls from the close path.

    /// Activate Risky Mode for this session. Constructs a
    /// [`RiskyModeManager`] from the supplied config + starting
    /// bankroll and stores it on the session. Idempotent in the
    /// sense that calling it twice replaces the existing manager
    /// with a freshly-validated one (the operator wants to re-anchor
    /// Risky Mode to a new bankroll).
    ///
    /// Returns `Err` when the supplied config fails its own
    /// [`RiskyModeConfig::validate`] (research §10.5 hard floors) or
    /// the bankroll is non-finite / non-positive. The session is
    /// left unchanged on error so a wizard-side validation failure
    /// never leaves Risky Mode in a half-armed state.
    pub fn enable_risky_mode(
        &mut self,
        config: RiskyModeConfig,
        starting_bankroll_usd: f64,
    ) -> anyhow::Result<()> {
        let manager = RiskyModeManager::new(config, starting_bankroll_usd)?;
        self.risky_mode_manager = Some(manager);
        tracing::info!(
            target: "neoethos_app::risky_mode",
            starting_bankroll_usd,
            "Risky Mode enabled on TradingSession"
        );
        Ok(())
    }

    /// Disarm Risky Mode for this session — falls back to Standard
    /// mode (the prop-firm gate continues to run as it always has).
    /// Used by the wizard / operator UI when the risk slider is
    /// dialed back below `10`.
    pub fn disable_risky_mode(&mut self) {
        if self.risky_mode_manager.is_some() {
            tracing::info!(
                target: "neoethos_app::risky_mode",
                "Risky Mode disabled on TradingSession"
            );
        }
        self.risky_mode_manager = None;
    }

    /// `true` when Risky Mode is currently armed. Consulted by the
    /// chrome status pill so it can render the Risky Mode banner
    /// (research §7.1).
    pub fn risky_mode_active(&self) -> bool {
        self.risky_mode_manager.is_some()
    }

    /// Read-only handle to the Risky Mode manager. `pub(super)` so
    /// `orders.rs` can compose `check_trade_allowed` into the order
    /// pipeline without re-importing the type.
    pub(super) fn risky_mode_manager(&self) -> Option<&RiskyModeManager> {
        self.risky_mode_manager.as_ref()
    }

    /// Mutable handle to the Risky Mode manager. `pub(super)` so the
    /// close-path in `orders.rs` can feed realised PnL via
    /// `record_trade_outcome`.
    pub(super) fn risky_mode_manager_mut(&mut self) -> Option<&mut RiskyModeManager> {
        self.risky_mode_manager.as_mut()
    }

    /// Read-only UI snapshot of Risky Mode state. Returns `None`
    /// when Risky Mode is not active. The chrome stage-progress bar
    /// (research §7.2), kill-switch banner (§7.4), and retreat
    /// indicator (§7.5) consume this; the manager itself stays
    /// behind the `pub(super)` accessor.
    pub fn risky_mode_state(&self) -> Option<RiskyModeState> {
        let manager = self.risky_mode_manager.as_ref()?;
        let stage = manager.current_stage();
        let total_stages = manager.config().stages.len();
        Some(RiskyModeState {
            current_stage_idx: stage.stage_idx,
            total_stages: total_stages.min(u8::MAX as usize) as u8,
            current_bankroll_usd: manager.current_bankroll_usd(),
            stage_bankroll_lower_usd: stage.bankroll_lower_usd,
            stage_bankroll_upper_usd: stage.bankroll_upper_usd,
            daily_loss_accumulated_usd: manager.daily_loss_accumulated_usd(),
            monthly_loss_accumulated_usd: manager.monthly_loss_accumulated_usd(),
            last_kill_switch_trip: manager.last_kill_switch_trip(),
            ruin_probability_estimate: manager.current_ruin_probability_estimate(),
        })
    }

    // ── Auto-trade producer lifecycle ─────────────────────────────────────
    //
    // The producer is the second half of the auto-trade pipeline (the
    // first half — the dispatcher gate chain — lives in `auto_trade.rs`).
    // The session owns the producer's cancel flag + signal receiver +
    // thread handle so a `disconnect` / `disable_auto_trade` flow can
    // tear it down without leaking the thread.

    /// Start the live-inference producer thread. The session takes
    /// ownership of the cancel flag + signal receiver; subsequent
    /// `drain_auto_trade_signals` calls drain the receiver into
    /// `dispatch_auto_trade_signal`. Returns `Err` if a producer is
    /// already running on this session (the operator must
    /// `stop_auto_trade_producer` first to swap configs).
    pub fn start_auto_trade_producer(
        &mut self,
        config: auto_trade_producer::LiveInferenceProducerConfig,
        bar_source: Arc<dyn auto_trade_producer::LiveBarSource>,
        predictor: Arc<dyn auto_trade_producer::ModelPredictor>,
    ) -> anyhow::Result<()> {
        if self.auto_trade_producer.is_some() {
            anyhow::bail!("auto-trade producer is already running on this session — stop it first");
        }
        let symbol = config.symbol.clone();
        let (tx, rx) = std::sync::mpsc::channel::<auto_trade::AutoTradeSignal>();
        let producer =
            auto_trade_producer::LiveInferenceProducer::new(config, bar_source, predictor, tx)?;
        let cancel = producer.cancel_flag();
        let handle = producer.spawn()?;
        self.auto_trade_producer = Some(auto_trade_producer::AutoTradeProducerHandle {
            cancel,
            handle: Some(handle),
            signal_rx: rx,
            symbol,
        });
        tracing::info!(
            target: "neoethos_app::auto_trade::producer",
            symbol = self
                .auto_trade_producer
                .as_ref()
                .map(|h| h.symbol())
                .unwrap_or(""),
            "auto-trade producer started"
        );
        Ok(())
    }

    /// `true` when an auto-trade producer is currently running on
    /// this session. Consulted by the chrome status pill and by the
    /// Settings panel toggle.
    pub fn auto_trade_producer_running(&self) -> bool {
        self.auto_trade_producer.is_some()
    }

    /// Symbol the running producer is bound to, if any.
    pub fn auto_trade_producer_symbol(&self) -> Option<&str> {
        self.auto_trade_producer.as_ref().map(|h| h.symbol())
    }

    /// Stop the auto-trade producer. Flips the cancel flag and joins
    /// the thread. Returns the [`auto_trade_producer::ProducerOutcome`]
    /// the loop terminated with, or `None` if no producer was running.
    pub fn stop_auto_trade_producer(&mut self) -> Option<auto_trade_producer::ProducerOutcome> {
        let mut handle = self.auto_trade_producer.take()?;
        handle
            .cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let outcome = handle.handle.take().and_then(|jh| jh.join().ok());
        tracing::info!(
            target: "neoethos_app::auto_trade::producer",
            outcome = ?outcome,
            "auto-trade producer stopped"
        );
        outcome
    }

    /// One-call end-to-end auto-trade activation.
    ///
    /// Convenience entry point that builds the full inference
    /// pipeline against the operator's saved trained-model
    /// directory and starts the producer thread:
    ///
    /// 1. Calls [`neoethos_models::build_ensemble_for_symbol`] to
    ///    scan `<models_dir>/<symbol>/<timeframe>/`, load every
    ///    trained expert it finds, and construct a
    ///    SoftVotingEnsemble (genetic + neuro_evo excluded by
    ///    default — strategy discoverers).
    /// 2. Wraps the ensemble in an
    ///    [`ensemble_predictor_adapter::EnsembleModelPredictor`]
    ///    so it conforms to the producer's `ModelPredictor`
    ///    trait.
    /// 3. Wraps the supplied bar source + the bridge predictor
    ///    in a [`auto_trade_producer::LiveInferenceProducerConfig`]
    ///    for the requested symbol and starts the producer
    ///    thread.
    ///
    /// Returns a load-summary so the chrome can render "Running
    /// ensemble: X/33 experts active — Y missing, Z degraded"
    /// (33 = `DEFAULT_BOOTSTRAP_EXPERT_NAMES.len()`, which is
    /// `KNOWN_MODEL_NAMES.len() - 1` for the deferred swarm).
    /// The full `ExpertLoadOutcome` (with the live Box<dyn
    /// ExpertModel> handles) is owned by the running predictor;
    /// callers that need richer introspection should query the
    /// predictor directly.
    ///
    /// Errors when:
    /// - No experts loaded from disk (operator hasn't trained
    ///   yet, or `models_dir` is wrong).
    /// - A producer is already running on this session.
    /// - The bar source backend fails to initialise.
    pub fn start_auto_trade_with_ensemble(
        &mut self,
        models_dir: &std::path::Path,
        symbol: &str,
        timeframe: &str,
        bar_source: Arc<dyn auto_trade_producer::LiveBarSource>,
    ) -> anyhow::Result<EnsembleLoadSummary> {
        if self.auto_trade_producer.is_some() {
            anyhow::bail!("auto-trade producer is already running on this session — stop it first");
        }
        // Build ensemble from saved artifacts. SoftVotingEnsemble::new
        // already rejects the no-voters case for us.
        let ensemble = neoethos_models::build_ensemble_for_symbol(models_dir, symbol, timeframe)
            .with_context(|| {
                format!(
                    "build_ensemble_for_symbol({}, {}, {}) failed",
                    models_dir.display(),
                    symbol,
                    timeframe
                )
            })?;
        // Snapshot the outcome metadata BEFORE the ensemble moves
        // into the predictor (Box<dyn ExpertModel> isn't Clone).
        // `load_outcome` is a trait method on EnsemblePredictor;
        // bring the trait into scope so method-call syntax works.
        use neoethos_models::EnsemblePredictor as _;
        let outcome = ensemble.load_outcome();
        let summary = EnsembleLoadSummary {
            loaded_names: outcome
                .loaded_names()
                .into_iter()
                .map(String::from)
                .collect(),
            missing: outcome.missing.clone(),
            degraded_names: outcome
                .degraded
                .iter()
                .map(|e| e.name().to_string())
                .collect(),
        };
        // Wrap in the producer's ModelPredictor adapter.
        let predictor: Arc<dyn auto_trade_producer::ModelPredictor> = Arc::new(
            ensemble_predictor_adapter::EnsembleModelPredictor::new(Arc::new(ensemble)),
        );
        // Producer config with the operator's symbol.
        let cfg = auto_trade_producer::LiveInferenceProducerConfig::for_symbol(symbol);
        // Standard start_auto_trade_producer path.
        self.start_auto_trade_producer(cfg, bar_source, predictor)?;
        tracing::info!(
            target: "neoethos_app::auto_trade::producer",
            symbol,
            timeframe,
            loaded = summary.loaded_names.len(),
            missing = summary.missing.len(),
            degraded = summary.degraded_names.len(),
            "auto-trade producer started with ensemble from disk"
        );
        Ok(summary)
    }

    /// Top-of-chrome AUTO ON entry point — builds the live-inference
    /// pipeline against the currently-selected symbol/timeframe and
    /// kicks off the producer.
    ///
    /// What it bundles (previously the chrome toggle was a cosmetic
    /// flag — see audit 2026-05-20, task #7):
    /// 1. Resolves `state.selected_pair` against the broker to get
    ///    `(symbol_id, digits)` so the live spot subscribe request is
    ///    well-formed. This is a synchronous network call — the
    ///    operator is clicking a button, so the brief block is OK.
    /// 2. Mints a `CTraderLiveChartUpdateRequest` with fresh credentials
    ///    + access token + the discovered account.
    /// 3. Wraps the session's shared `ctrader_live_streaming_backend` in
    ///    a [`auto_trade_producer::CTraderLiveBarSource`] adapter so the
    ///    producer can poll it for closed trendbars.
    /// 4. Delegates to [`Self::start_auto_trade_with_ensemble`] which
    ///    scans `<data_dir>/models/<symbol>/<timeframe>/`, loads every
    ///    trained expert it finds, wraps the ensemble in a
    ///    `ModelPredictor` adapter, and spawns the inference thread.
    ///
    /// Errors propagate verbatim so the chrome handler can surface
    /// "AUTO ON failed: <reason>" to the operator without flipping the
    /// flag. Typical failures: no trained ensemble on disk yet, the
    /// selected symbol can't be resolved against the broker, no
    /// discovered account, missing token bundle.
    pub fn start_auto_trade_for_state(
        &mut self,
        state: &AppState,
    ) -> anyhow::Result<EnsembleLoadSummary> {
        use crate::app_services::ctrader_streaming::CTraderLiveChartUpdateRequest;
        let symbol = state.selected_pair.clone();
        let timeframe = state.chart_timeframe.clone();
        if symbol.trim().is_empty() {
            anyhow::bail!("AUTO ON requires a selected symbol — none is active");
        }
        if timeframe.trim().is_empty() {
            anyhow::bail!("AUTO ON requires a selected chart timeframe — none is active");
        }

        // 1. Resolve symbol against the broker so we have symbol_id + digits.
        let resolved = self
            .resolve_selected_ctrader_symbol(&symbol)
            .with_context(|| {
                format!("AUTO ON failed to resolve cTrader symbol {symbol} against the broker")
            })?;

        // 2. Build the live-spot subscribe request.
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            anyhow::bail!("AUTO ON requires configured cTrader client_id and client_secret");
        }
        let access_token = self
            .ensure_fresh_ctrader_token_bundle("AUTO ON requires a stored cTrader token bundle")?
            .access_token;
        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "AUTO ON requires a discovered cTrader account to bind execution to"
                )
            })?;

        let live_request = CTraderLiveChartUpdateRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            symbol_id: resolved.symbol.symbol_id,
            digits: resolved.symbol.digits,
            timeframe: timeframe.clone(),
            subscribe_to_spot_timestamp: false,
        };

        // 3. Build the bar source against the shared streaming backend.
        let bar_source: std::sync::Arc<dyn auto_trade_producer::LiveBarSource> =
            std::sync::Arc::new(auto_trade_producer::CTraderLiveBarSource::new(
                std::sync::Arc::clone(&self.ctrader_live_streaming_backend),
                live_request,
            ));

        // 4. Delegate to the canonical "load ensemble + spawn producer" path.
        let models_dir = state.runtime.data_dir.join("models");
        self.start_auto_trade_with_ensemble(&models_dir, &symbol, &timeframe, bar_source)
    }

    /// Drain all pending auto-trade signals from the producer's
    /// outbound channel and push each one through the §7.1
    /// dispatcher gate chain. Returns the number of signals
    /// dispatched (regardless of whether the dispatcher accepted
    /// them or rejected them at a gate). Called by the main app
    /// loop on every tick — non-blocking.
    pub fn drain_auto_trade_signals(&mut self, state: &mut AppState) -> usize {
        let signals: Vec<auto_trade::AutoTradeSignal> = {
            let Some(handle) = self.auto_trade_producer.as_ref() else {
                return 0;
            };
            handle.signal_rx.try_iter().collect()
        };
        let count = signals.len();
        for sig in signals {
            let decision = self.dispatch_auto_trade_signal(state, sig);
            tracing::debug!(
                target: "neoethos_app::auto_trade::producer",
                decision = ?decision,
                "auto-trade producer signal dispatched"
            );
        }
        count
    }

    pub fn snapshot(&self, state: &AppState) -> ConnectionSnapshot {
        let mode = panel_mode(state.data_source, self.connected);
        let adapter_kind = self
            .adapter
            .as_ref()
            .map(TradingAdapter::kind)
            .unwrap_or(self.configured_adapter);
        let adapter_name = adapter_kind.as_str().to_string();
        let status_text = match mode {
            TradingPanelMode::LocalOnly => "Local Mode".to_string(),
            TradingPanelMode::Disconnected => "Offline".to_string(),
            TradingPanelMode::Connected => {
                if state.status_msg.trim().is_empty()
                    || state.status_msg == "cTrader Ready"
                    || state.status_msg == "Local Mode"
                {
                    "Connected".to_string()
                } else {
                    state.status_msg.clone()
                }
            }
        };

        ConnectionSnapshot {
            adapter_name,
            integration_mode: adapter_kind.integration_mode().to_string(),
            requires_local_terminal: adapter_kind.requires_local_terminal(),
            supports_market_data: adapter_kind.supports_market_data(),
            supports_live_orders: adapter_kind.supports_live_orders(),
            mode,
            connected: self.connected,
            status_text,
            terminal_info: self.terminal_info.clone(),
        }
    }

    // `market_chart_snapshot` and the cTrader chart-history helpers moved
    // to `market_data.rs` (Batch 6).

    pub fn execution_surface_snapshot(&mut self, state: &AppState) -> ExecutionSurfaceSnapshot {
        let connection = self.snapshot(state);
        let adapter_kind = self.active_adapter_kind();
        let cache_key = ExecutionSnapshotCacheKey {
            data_source: state.data_source,
            symbol: state.selected_pair.clone(),
            adapter_kind,
            connected: self.connected,
        };

        if let Some(cache) = &self.execution_surface_cache
            && cache.key == cache_key
            && cache.refreshed_at.elapsed() < Duration::from_secs(1)
        {
            return cache.snapshot.clone();
        }

        let mut runtime_warnings = Vec::new();
        let runtime = match self
            .execution_feed_handle(state)
            .load_runtime_snapshot(&state.selected_pair, 24)
        {
            Ok(snapshot) => Some(snapshot),
            Err(err) => {
                runtime_warnings.push(err.to_string());
                None
            }
        };

        let mut snapshot = build_execution_surface_snapshot_with_runtime(
            state,
            &connection,
            runtime.as_ref(),
            runtime_warnings,
        );
        snapshot.journal_rows = self.trade_journal.clone();
        self.execution_surface_cache = Some(CachedExecutionSnapshot {
            key: cache_key,
            refreshed_at: Instant::now(),
            snapshot: snapshot.clone(),
        });
        snapshot
    }

    // Connect / disconnect (`start_connect`, `handle_ctrader_connect_result`,
    // `connect`, `disconnect`) moved to `session.rs` (Batch 6). The new-/
    // cancel-/close-order entry points (`execute_buy_order`,
    // `execute_sell_order`, `cancel_selected_order`,
    // `close_selected_position`) moved to `orders.rs`.

    // `select_adapter` moved to `session.rs` (Batch 6).

    pub(super) fn overlay_status(&self, state: &AppState) -> String {
        match state.data_source {
            DataSource::Local => {
                "Trade overlays unavailable in Local mode until execution events are wired.".to_string()
            }
            DataSource::CTrader => match self.active_adapter_kind() {
                TradingAdapterKind::CTrader if !self.connected => {
                    "Trade overlays unavailable while the cTrader runtime is disconnected."
                        .to_string()
                }
                TradingAdapterKind::CTrader => {
                    "Trade overlays will appear here once cTrader positions, fills, and bot execution events are wired.".to_string()
                }
                TradingAdapterKind::DxTrade => {
                    "Trade overlays will appear here once DXtrade execution events are wired.".to_string()
                }
            },
        }
    }

    /// Currently-active broker adapter kind. Note promoted
    /// this from `pub(super)` to `pub` so UI code (`execution_panel`) can
    /// gate capability-sensitive controls on the actual adapter rather
    /// than a string-equality check on `adapter_name`.
    pub fn active_adapter_kind(&self) -> TradingAdapterKind {
        self.adapter
            .as_ref()
            .map(TradingAdapter::kind)
            .unwrap_or(self.configured_adapter)
    }

    fn execution_feed_handle(&self, state: &AppState) -> ExecutionFeedHandle<'_> {
        match state.data_source {
            DataSource::Local => ExecutionFeedHandle::Unavailable {
                reason: "Execution feed is unavailable in Local mode.".to_string(),
            },
            DataSource::CTrader => match &self.adapter {
                Some(TradingAdapter::CTrader(runtime)) if self.connected => {
                    ExecutionFeedHandle::CTrader(runtime)
                }
                _ => ExecutionFeedHandle::Unavailable {
                    reason: self
                        .active_adapter_kind()
                        .execution_feed_unavailable_reason(self.connected),
                },
            },
        }
    }

    // `reset_connection_state` moved to `session.rs`.
    // Order-execution pipeline (`execute_ctrader_order`,
    // `execute_ctrader_request`, `build_ctrader_execution_runtime_request`,
    // `calculate_smart_atr_in_points`, `build_ctrader_order_request`,
    // `resolve_selected_ctrader_symbol`, `ctrader_account_equity`,
    // `ctrader_symbol_pip_position`) moved to `orders.rs`.

    /// Reset the per-day risk-tracking counters when the broker calendar
    /// day advances. Called from the periodic runtime refresh path; until
    /// this fires the daily-DD check would otherwise treat the entire
    /// session as a single "day" — D6 from the audit.
    pub fn handle_day_boundary(&mut self, broker_now_unix_ms: i64) {
        let day_id = broker_now_unix_ms / 86_400_000;
        if self.last_observed_day_id == Some(day_id) {
            return;
        }
        // Snapshot the live equity in a separate scope so the immutable borrow
        // on `self.connected_ctrader_runtime()` is released before we assign
        // back into `self.day_start_equity`.
        let live_equity: Option<f64> = self
            .connected_ctrader_runtime()
            .map(|r| r.trader.balance + r.trader.unrealized_pnl);
        if let Some(equity) = live_equity {
            self.day_start_equity = Some(equity);
            tracing::info!(
                target: "neoethos_app::risk",
                day_id,
                day_start_equity = equity,
                "day boundary crossed; daily-DD reference reset"
            );
        }
        self.last_observed_day_id = Some(day_id);
    }

    /// Roll the prop-firm phase forward (Challenge → Verification → Funded).
    /// Each phase has its own starting balance, so `initial_equity` and
    /// `day_start_equity` must be re-anchored when the operator marks the
    /// previous phase as complete — D7 from the audit.
    pub fn handle_phase_rollover(&mut self, new_phase_starting_equity: f64) {
        if !new_phase_starting_equity.is_finite() || new_phase_starting_equity <= 0.0 {
            tracing::warn!(
                target: "neoethos_app::risk",
                value = new_phase_starting_equity,
                "phase rollover rejected: starting equity must be finite and positive"
            );
            return;
        }
        self.initial_equity = Some(new_phase_starting_equity);
        self.day_start_equity = Some(new_phase_starting_equity);
        self.last_observed_day_id = None;
        tracing::info!(
            target: "neoethos_app::risk",
            new_phase_starting_equity,
            "prop-firm phase rolled over; total-DD and daily-DD anchors reset"
        );
    }

    pub fn refresh_runtime(&mut self, state: &mut AppState) -> anyhow::Result<()> {
        if !self.connected {
            return Ok(());
        }
        match &self.adapter {
            Some(TradingAdapter::CTrader(_)) => {
                if self
                    .ctrader_runtime_refreshed_at
                    .is_some_and(|refreshed_at| refreshed_at.elapsed() < Duration::from_secs(30))
                {
                    return Ok(());
                }

                let runtime = self.load_ctrader_account_runtime()?;
                self.terminal_info = format_ctrader_terminal_info(
                    &runtime.trader,
                    self.selected_ctrader_environment(),
                );
                state.account_balance = runtime.trader.balance;
                state.account_equity = self.calculate_equity_from_runtime(&runtime);
                self.adapter = Some(TradingAdapter::CTrader(runtime));
                self.ctrader_runtime_refreshed_at = Some(Instant::now());
                self.execution_surface_cache = None;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Non-blocking variant of [`Self::refresh_runtime`] for the
    /// 5-second heartbeat loop. Note — the synchronous
    /// `refresh_runtime` calls `load_ctrader_account_runtime` which
    /// opens a fresh WSS socket each time (~200-500 ms on a healthy
    /// network, longer when the broker is slow). Running it on the
    /// render thread visibly stalled the UI every 30 s.
    ///
    /// This variant:
    ///   - Respects the same 30 s cooldown (no point hammering the
    ///     broker — `ctrader_runtime_refreshed_at` is the single source
    ///     of truth for "when did we last refresh?").
    ///   - Returns `true` if it spawned a fetch, `false` if it
    ///     short-circuited on the cooldown.
    ///   - Routes the worker through `background::spawn_background_task`
    ///     so panics surface via `ServiceEvent::BackgroundTaskPanic`
    ///     (Task #2) and re-uses the same `ServiceEvent::CTraderConnectUpdated`
    ///     handler the connect flow already wires.
    pub fn start_runtime_refresh_in_background(
        &mut self,
        tx: tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
    ) -> bool {
        if !self.connected {
            return false;
        }
        if !matches!(self.adapter, Some(TradingAdapter::CTrader(_))) {
            return false;
        }
        if self
            .ctrader_runtime_refreshed_at
            .is_some_and(|refreshed_at| refreshed_at.elapsed() < Duration::from_secs(30))
        {
            return false;
        }
        // Build the request on the main thread (it needs &self for
        // broker_settings + adapter state). Note: this also calls
        // `refresh_ctrader_token_bundle` synchronously if the token is
        // about to expire — that's fast (<10ms) and avoids racing the
        // worker against a token rotation. The expensive bit (the
        // ProtoOAReconcileReq round-trip) is what we offload.
        let Ok(request) = self.build_ctrader_account_runtime_request() else {
            return false;
        };
        // Reserve the cooldown window NOW so a slow worker doesn't get
        // double-scheduled by the next heartbeat.
        self.ctrader_runtime_refreshed_at = Some(Instant::now());
        let backend = std::sync::Arc::clone(&self.ctrader_account_runtime_backend);
        background::spawn_background_task("runtime_refresh", tx.clone(), move || {
            match backend.load_account_runtime(&request) {
                Ok(runtime) => {
                    let _ = tx.blocking_send(
                        crate::app_services::ServiceEvent::CTraderConnectUpdated(runtime),
                    );
                }
                Err(err) => {
                    // Refresh failures during the heartbeat are
                    // expected during transient network blips — log
                    // and surface to the status strip rather than
                    // tearing down the session.
                    tracing::warn!(
                        target: "neoethos_app::heartbeat",
                        error = %err,
                        "background runtime refresh failed; will retry on next heartbeat"
                    );
                    let _ = tx.blocking_send(crate::app_services::ServiceEvent::ConnectOutcome(
                        Err(format!("Heartbeat refresh: {err}")),
                    ));
                }
            }
        });
        true
    }

    fn calculate_equity_from_runtime(&self, runtime: &CTraderAccountRuntimeSnapshot) -> f64 {
        let accrued: f64 = runtime
            .reconcile
            .positions
            .iter()
            .map(|pos| pos.swap.unwrap_or(0.0) + pos.commission.unwrap_or(0.0))
            .sum();
        runtime.trader.balance + accrued
    }
}

/// Find the index of the candle whose `timestamp` is the largest value
/// <= `target_ts`. Returns `None` if all candles are newer than the
/// target (the decision happened before the visible window) or if the
/// candle slice is empty or has no timestamps set.
///
/// O(N) linear scan — typical chart windows are <500 candles so this
/// is well below 1µs even on a worst case. A future bsearch can replace
/// this if the chart ever holds 10k+ bars and overlay count climbs.
pub(crate) fn nearest_candle_index(candles: &[ChartCandle], target_ts: i64) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, c) in candles.iter().enumerate() {
        let Some(ts) = c.timestamp else { continue };
        if ts > target_ts {
            break;
        }
        best = Some(i);
    }
    best
}

impl Default for TradingSession {
    fn default() -> Self {
        Self {
            configured_adapter: TradingAdapterKind::CTrader,
            broker_settings: BrokerSettingsState::default(),
            ctrader_auth: None,
            ctrader_live_auth_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_discovery_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_runtime_backend: Arc::new(ProductionCTraderAccountRuntimeBackend),
            ctrader_execution_backend: Arc::new(ProductionCTraderExecutionBackend),
            ctrader_position_order_history_backend: Arc::new(
                ProductionCTraderPositionOrderHistoryBackend,
            ),
            ctrader_live_streaming_backend: Arc::new(ProductionCTraderLiveStreamingBackend),
            ctrader_token_store: Arc::new(production_ctrader_token_store()),
            ctrader_live_auth_rx: None,
            adapter: None,
            connected: false,
            terminal_info: String::new(),
            market_chart_cache: None,
            execution_surface_cache: None,
            ctrader_live_spot_cache: None,
            trade_journal: Vec::new(),
            initial_equity: None,
            day_start_equity: None,
            last_observed_day_id: None,
            ctrader_runtime_refreshed_at: None,
            connect_handle: None,
            connect_in_flight: Arc::new(AtomicBool::new(false)),
            bootstrap_handle: None,
            chart_fetch_handle: None,
            pending_ctrader_chart: None,
            trading_environment: TradingEnvironment::Demo,
            halt_state: HaltState::default(),
            risky_mode_manager: None,
            bot_decisions: Vec::new(),
            auto_trade_producer: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    #[allow(dead_code)] // re-entrancy guard taxonomy: `Connect` is
    // checked by `background_task_running` but the
    // current connect path uses an explicit
    // AtomicBool (`connect_in_flight`) instead.
    // Kept on the enum because the taxonomy is
    // exposed to the trace target.
    Connect,
    Bootstrap,
}

impl TradingAdapter {
    fn kind(&self) -> TradingAdapterKind {
        match self {
            Self::CTrader(_) => TradingAdapterKind::CTrader,
        }
    }
}

impl TradingAdapterKind {
    fn execution_feed_unavailable_reason(self, connected: bool) -> String {
        match self {
            Self::CTrader if !connected => {
                "cTrader execution feed is unavailable until the remote account session connects."
                    .to_string()
            }
            Self::CTrader => "cTrader execution feed is currently unavailable.".to_string(),
            Self::DxTrade => "DXtrade execution feed is not wired yet.".to_string(),
        }
    }
}

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::CTrader, false) => TradingPanelMode::Disconnected,
        (DataSource::CTrader, true) => TradingPanelMode::Connected,
    }
}

impl MarketChartSnapshot {
    pub(super) fn with_overlay_status(mut self, overlay_status: String) -> Self {
        self.overlay_status = overlay_status;
        self
    }

    pub fn empty_for(
        symbol: &str,
        timeframe: &str,
        available_timeframes: Vec<String>,
        headline: String,
        overlay_status: String,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            available_timeframes,
            candles: Vec::new(),
            overlays: Vec::new(),
            price_min: 0.0,
            price_max: 0.0,
            bid: None,
            ask: None,
            price_change_pct: None,
            headline,
            overlay_status,
            warnings,
        }
    }
}

impl ExecutionFeedHandle<'_> {
    fn load_runtime_snapshot(
        &self,
        _symbol: &str,
        _lookback_hours: i64,
    ) -> anyhow::Result<AppExecutionRuntimeSnapshot> {
        match self {
            Self::CTrader(runtime) => Ok(AppExecutionRuntimeSnapshot::CTrader((*runtime).clone())),
            Self::Unavailable { reason } => Err(anyhow::anyhow!(reason.clone())),
        }
    }
}

#[cfg(test)]
#[path = "../trading_tests.rs"]
mod tests;
