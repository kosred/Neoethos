// Imports re-exported via `pub(super) use` so the `session`, `orders`, and
// `market_data` sibling modules can pull common types out of `super::*`
// without each duplicating the full `crate::app_services::...` path list.
// `pub(super)` keeps these aliases visible only inside `trading::*`, so the
// external (`forex-app`) surface is unchanged.
pub(super) use crate::app_record;
pub(super) use crate::app_services::ServiceEvent;
pub(super) use crate::app_services::broker_config::{
    AdapterReadinessSnapshot, BrokerAccountTarget, BrokerSessionState, BrokerSettingsState,
    CTraderBrokerEnvironment,
};
pub(super) use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeBackend, CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot,
    CTraderDealSnapshot, CTraderPendingOrderSnapshot, CTraderPositionSnapshot,
    ProductionCTraderAccountRuntimeBackend,
};
pub(super) use crate::app_services::ctrader_auth::{
    CTraderAccountSummary, CTraderAuthSession, CTraderAuthSnapshot, CTraderDiscoveredAccount,
    CTraderTokenBundle, CTraderTokenExchangeRequest,
};
pub(super) use crate::app_services::ctrader_bootstrap::{
    bootstrap_from_ctrader_history, plan_bootstrap_chunks,
};
pub(super) use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderSymbolInfo, CTraderSymbolLookupRequest, HistoricalBar,
    load_chart_history, resolve_symbol,
};
pub(super) use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, CTraderExecutionStatus, ProductionCTraderExecutionBackend,
};
pub(super) use crate::app_services::ctrader_live_auth::{
    CTRADER_DEFAULT_SCOPE, CTraderAccountDiscoveryBackend, CTraderAccountDiscoveryRequest,
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderLiveAuthRequest, CTraderLiveAuthResult,
    CTraderTokenRefreshRequest, ProductionCTraderLiveAuthBackend, build_default_loopback_config,
};
pub(super) use crate::app_services::ctrader_messages::{
    CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderAmendOrderRequest, CTraderCancelOrderRequest,
    CTraderClosePositionRequest, CTraderNewOrderRequest, CTraderOrderTriggerMethod,
    CTraderOrderType, CTraderTimeInForce, CTraderTradeSide,
    SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS, SUPPORTED_CTRADER_ORDER_TYPES,
    SUPPORTED_CTRADER_TIME_IN_FORCE, SUPPORTED_CTRADER_TRADE_SIDES, build_amend_order_request,
    build_cancel_order_request, build_close_position_request, build_new_order_request,
};
pub(super) use crate::app_services::ctrader_streaming::{
    CTraderLiveChartUpdate, CTraderLiveChartUpdateRequest, CTraderLiveStreamingBackend,
    ProductionCTraderLiveStreamingBackend, merge_live_spot_update_into_bars,
};
pub(super) use crate::app_services::jobs::{
    JobEventLevel, JobKind, JobSnapshot, JobState, push_recent_event,
};
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
pub(super) use crate::app_services::secure_store::{
    CTraderSecureStore, CTraderTokenStore, KeyringSecretStoreBackend,
};
pub(super) use crate::app_state::{AppState, DataSource, OrderTicketState};
pub(super) use anyhow::Context;
pub(super) use forex_core::logging::write_subsystem_record;
pub(super) use forex_core::sectioned_log::SubsystemSection;
pub(super) use forex_data::{Ohlcv, discover_timeframes, load_symbol_timeframe};
pub(super) use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};
pub(super) use tracing::error;

mod client_order;
mod diagnostics;
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
    append_ctrader_order_builder_diagnostics, extract_client_order_id_from_request,
    find_existing_client_order_id, format_ctrader_connect_error, format_ctrader_terminal_info,
    format_execution_journal_line, format_execution_outcome_status, non_empty_option,
    record_app_event, synthesize_idempotent_retry_outcome,
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

/// Trading-environment classifier used by the persistent status pill
/// in the main chrome (`ui::chrome::status_pill`) and consulted by the
/// HALT button to label its audit-log lines.
///
/// Maps verbatim to the four autonomy stages defined in
/// `docs/audits/research/wizard_onboarding_competitive_analysis.md`
/// §10.2 (Discovery -> Paper -> LiveSmall -> LiveFull). The wizard
/// promotion gates (§10.3) own the transitions; the chrome only
/// observes which mode the session currently sits in.
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
    Paper,
    /// Real money, capped per-trade size by the promotion gate. Pill: red.
    /// Matches §10.2 "Live small"/Stage 3 — uses real capital with the
    /// `min_trading_days=10` + `min_monthly_net_profit_pct=0.04` gate.
    LiveSmall,
    /// Real money, no extra cap. Pill: red, bold.
    /// Matches §10.2 "Live full"/Stage 4 — uncapped after 30 days at
    /// Stage 3 with all promotion gates passed.
    LiveFull,
}

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

pub struct TradingSession {
    configured_adapter: TradingAdapterKind,
    broker_settings: BrokerSettingsState,
    ctrader_auth: Option<CTraderAuthSession>,
    ctrader_live_auth_backend: Arc<dyn CTraderLiveAuthBackend>,
    ctrader_account_discovery_backend: Arc<dyn CTraderAccountDiscoveryBackend>,
    ctrader_account_runtime_backend: Arc<dyn CTraderAccountRuntimeBackend>,
    ctrader_execution_backend: Arc<dyn CTraderExecutionBackend>,
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
    bootstrap_handle: Option<std::thread::JoinHandle<()>>,
    /// Trading environment for the session — used by the status pill
    /// and HALT button. Defaults to Demo; the wizard / autonomy
    /// controller (§10.3 promotion gates) is responsible for advancing
    /// this to Paper / LiveSmall / LiveFull.
    trading_environment: TradingEnvironment,
    /// T-Manual kill switch (HALT). See `HaltState` docs for details.
    halt_state: HaltState,
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

impl TradingSession {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a session with broker settings pre-loaded from the
    /// per-user credentials TOML file (see
    /// [`crate::app_services::broker_persistence::load_broker_settings`]).
    ///
    /// Used by `main.rs` so the production app starts with credentials the
    /// user has already saved. Tests should keep using [`Self::new`] /
    /// [`Self::with_configured_adapter_for_test`] which start with empty
    /// defaults and are unaffected by whatever is on the developer's disk.
    pub fn new_with_persisted_credentials() -> Self {
        let mut session = Self::default();
        session.broker_settings = crate::app_services::broker_persistence::load_broker_settings();
        session
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
            ctrader_live_streaming_backend: Arc::new(ProductionCTraderLiveStreamingBackend),
            ctrader_token_store: Arc::new(CTraderSecureStore::new(
                "forex-ai.test",
                "ctrader.account",
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
            bootstrap_handle: None,
            trading_environment: TradingEnvironment::Demo,
            halt_state: HaltState::default(),
        }
    }

    #[cfg(test)]
    pub fn set_ctrader_store_for_test(
        &mut self,
        backend: crate::app_services::secure_store::MemorySecretStoreBackend,
    ) {
        self.ctrader_token_store = Arc::new(CTraderSecureStore::new(
            "forex-ai.test",
            "ctrader.account",
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
    ///   4. (TODO once `risky_mode.rs` lands) Calls
    ///      `risky_mode_manager.trip_manual_halt()` if Risky Mode
    ///      is active in the session.
    ///   5. Emits a `tracing::error!(target: "forex_app::halt", ...)`
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

        // TODO(risky-mode-halt): wire after risky_mode.rs lands —
        // call `risky_mode_manager.trip_manual_halt()` here if the
        // session has Risky Mode active. The Risky Mode agent owns
        // that module; this site is the single integration point.

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
            target: "forex_app::halt",
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
                    target: "forex_app::halt",
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
                        target: "forex_app::halt",
                        error = %err,
                        path = %path.display(),
                        "HALT sentinel remove failed during clear_halt"
                    );
                }
            }
        }
        tracing::info!(
            target: "forex_app::halt",
            "T-Manual HALT cleared by operator"
        );
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
    // cancel-/close-order entry points (`execute_buy_market`,
    // `execute_sell_market`, `cancel_selected_order`,
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

    pub(super) fn active_adapter_kind(&self) -> TradingAdapterKind {
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
                target: "forex_app::risk",
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
                target: "forex_app::risk",
                value = new_phase_starting_equity,
                "phase rollover rejected: starting equity must be finite and positive"
            );
            return;
        }
        self.initial_equity = Some(new_phase_starting_equity);
        self.day_start_equity = Some(new_phase_starting_equity);
        self.last_observed_day_id = None;
        tracing::info!(
            target: "forex_app::risk",
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
            ctrader_live_streaming_backend: Arc::new(ProductionCTraderLiveStreamingBackend),
            ctrader_token_store: Arc::new(CTraderSecureStore::new(
                "forex-ai",
                "ctrader.default",
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
            bootstrap_handle: None,
            trading_environment: TradingEnvironment::Demo,
            halt_state: HaltState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
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
