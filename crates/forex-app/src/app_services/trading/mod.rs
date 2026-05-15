use crate::app_record;
use crate::app_services::ServiceEvent;
use crate::app_services::broker_config::{
    AdapterReadinessSnapshot, BrokerAccountTarget, BrokerSessionState, BrokerSettingsState,
    CTraderBrokerEnvironment,
};
use crate::app_services::ctrader_account::{
    CTraderAccountRuntimeBackend, CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot,
    CTraderDealSnapshot, CTraderPendingOrderSnapshot, CTraderPositionSnapshot,
    ProductionCTraderAccountRuntimeBackend,
};
use crate::app_services::ctrader_auth::{
    CTraderAccountSummary, CTraderAuthSession, CTraderAuthSnapshot, CTraderDiscoveredAccount,
    CTraderTokenBundle, CTraderTokenExchangeRequest,
};
use crate::app_services::ctrader_bootstrap::{
    bootstrap_from_ctrader_history, plan_bootstrap_chunks,
};
use crate::app_services::ctrader_data::{
    CTraderChartHistoryRequest, CTraderSymbolInfo, CTraderSymbolLookupRequest, HistoricalBar,
    load_chart_history, resolve_symbol,
};
use crate::app_services::ctrader_execution::{
    CTraderExecutionBackend, CTraderExecutionOutcome, CTraderExecutionRequest,
    CTraderExecutionRuntimeRequest, CTraderExecutionStatus, ProductionCTraderExecutionBackend,
};
use crate::app_services::ctrader_live_auth::{
    CTRADER_DEFAULT_SCOPE, CTraderAccountDiscoveryBackend, CTraderAccountDiscoveryRequest,
    CTraderEnvironment, CTraderLiveAuthBackend, CTraderLiveAuthRequest, CTraderLiveAuthResult,
    CTraderTokenRefreshRequest, ProductionCTraderLiveAuthBackend, build_default_loopback_config,
};
use crate::app_services::ctrader_messages::{
    CTRADER_TOKEN_EXPIRED_SENTINEL, CTraderAmendOrderRequest, CTraderCancelOrderRequest,
    CTraderClosePositionRequest, CTraderNewOrderRequest, CTraderOrderTriggerMethod,
    CTraderOrderType, CTraderTimeInForce, CTraderTradeSide,
    SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS, SUPPORTED_CTRADER_ORDER_TYPES,
    SUPPORTED_CTRADER_TIME_IN_FORCE, SUPPORTED_CTRADER_TRADE_SIDES, build_amend_order_request,
    build_cancel_order_request, build_close_position_request, build_new_order_request,
};
use crate::app_services::ctrader_streaming::{
    CTraderLiveChartUpdate, CTraderLiveChartUpdateRequest, CTraderLiveStreamingBackend,
    ProductionCTraderLiveStreamingBackend, merge_live_spot_update_into_bars,
};
use crate::app_services::jobs::{JobEventLevel, JobKind, JobSnapshot, JobState, push_recent_event};
use crate::app_services::secure_store::{
    CTraderSecureStore, CTraderTokenStore, KeyringSecretStoreBackend,
};
use crate::app_state::{AppState, DataSource, OrderTicketState};
use anyhow::Context;
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use forex_data::{Ohlcv, discover_timeframes, load_symbol_timeframe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::error;

mod client_order;
mod diagnostics;
mod risk_gate;
mod snapshots;

use client_order::{CTRADER_TOKEN_REFRESH_WINDOW_SECS, current_unix_seconds, next_client_order_seq};
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

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn configured_adapter(&self) -> TradingAdapterKind {
        self.configured_adapter
    }

    pub fn broker_settings_mut(&mut self) -> &mut BrokerSettingsState {
        &mut self.broker_settings
    }

    pub fn adapter_readiness(&self) -> AdapterReadinessSnapshot {
        let mut readiness = self.broker_settings.readiness(self.configured_adapter);
        if self.connected {
            readiness.session_state = BrokerSessionState::Authenticated;
            readiness.status_line = format!("{} connected.", self.active_adapter_kind().as_str());
            readiness.can_attempt_connect = false;
        }
        readiness
    }

    pub fn can_attempt_connect(&self) -> bool {
        self.adapter_readiness().can_attempt_connect
    }

    pub fn ctrader_auth_snapshot(&self) -> Option<CTraderAuthSnapshot> {
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                if let Some(session) = &self.ctrader_auth {
                    Some(session.snapshot())
                } else {
                    let client_id = self.broker_settings.ctrader.client_id.trim();
                    let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim();
                    if client_id.is_empty() || redirect_uri.is_empty() {
                        None
                    } else {
                        Some(CTraderAuthSession::new(client_id, redirect_uri).snapshot())
                    }
                }
            }
            _ => None,
        }
    }

    pub fn start_ctrader_bootstrap_batch(
        &mut self,
        data_root: std::path::PathBuf,
        symbols: Vec<String>,
        timeframes: Vec<String>,
        years: u32,
        tx: tokio::sync::mpsc::Sender<ServiceEvent>,
    ) -> anyhow::Result<()> {
        self.reap_finished_background_tasks();
        if symbols.is_empty() || timeframes.is_empty() {
            return Err(anyhow::anyhow!(
                "bootstrap requires at least one symbol and one timeframe"
            ));
        }
        if self.background_task_running(TaskKind::Bootstrap) {
            return Err(anyhow::anyhow!("cTrader data bootstrap is already running"));
        }
        let context = self.resolve_ctrader_bootstrap_context()?;
        let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
        snapshot.state = JobState::Running;
        snapshot.progress.stage = "bootstrap_queued".to_string();
        snapshot.progress.message = format!(
            "Queued {} symbols across {} timeframes",
            symbols.len(),
            timeframes.len()
        );

        let tx_clone = tx.clone();
        let _ = tx.blocking_send(ServiceEvent::BootstrapUpdated(snapshot.clone()));

        let handle = std::thread::spawn(move || {
            let mut running_snapshot = snapshot;
            running_snapshot.state = JobState::Running;
            running_snapshot.progress.stage = "bootstrap_running".to_string();
            running_snapshot.progress.message =
                "Fetching and normalizing historical data".to_string();
            let _ =
                tx_clone.blocking_send(ServiceEvent::BootstrapUpdated(running_snapshot.clone()));

            let final_snapshot = match run_ctrader_bootstrap_batch_with_context(
                &context,
                &data_root,
                &symbols,
                &timeframes,
                years,
            ) {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    let mut failed = running_snapshot;
                    failed.state = JobState::Failed;
                    failed.progress.stage = "bootstrap_failed".to_string();
                    failed.progress.message = err.to_string();
                    failed.report.summary = format!("Bootstrap failed: {err}");
                    failed.report.errors.push(err.to_string());
                    failed
                }
            };
            let _ = tx_clone.blocking_send(ServiceEvent::BootstrapUpdated(final_snapshot));
        });
        self.bootstrap_handle = Some(handle);

        Ok(())
    }

    pub fn start_ctrader_auth(&mut self) -> anyhow::Result<CTraderAuthSnapshot> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || redirect_uri.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader auth requires client_id and redirect_uri"
            ));
        }
        let mut session = CTraderAuthSession::new(client_id, redirect_uri);
        session.start_authorization("trading");
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);
        Ok(snapshot)
    }

    pub fn receive_ctrader_authorization_code(
        &mut self,
        code: impl Into<String>,
    ) -> CTraderAuthSnapshot {
        let session = self.ctrader_auth.get_or_insert_with(|| {
            CTraderAuthSession::new(
                self.broker_settings.ctrader.client_id.clone(),
                self.broker_settings.ctrader.redirect_uri.clone(),
            )
        });
        session.receive_authorization_code(code);
        session.snapshot()
    }

    pub fn build_ctrader_token_exchange_request(
        &mut self,
    ) -> anyhow::Result<CTraderTokenExchangeRequest> {
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_secret.is_empty() {
            if let Some(session) = self.ctrader_auth.as_mut() {
                session.mark_failed("cTrader token exchange requires client_secret");
            }
            return Err(anyhow::anyhow!(
                "cTrader token exchange requires client_secret"
            ));
        }
        let session = self
            .ctrader_auth
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("cTrader auth has not started"))?;
        let request = session.build_token_exchange_request(client_secret);
        if !self.broker_settings.ctrader.accounts.is_empty() {
            session.set_accounts(
                self.broker_settings
                    .ctrader
                    .accounts
                    .iter()
                    .map(|account| CTraderAccountSummary {
                        account_id: account.account_id.clone(),
                        broker_title: account.label.clone(),
                        enabled_for_execution: account.enabled_for_execution,
                    })
                    .collect(),
            );
        }
        Ok(request)
    }

    pub fn start_ctrader_live_auth(
        &mut self,
        _tx: tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
    ) -> anyhow::Result<CTraderAuthSnapshot> {
        self.reap_finished_background_tasks();
        if self.ctrader_live_auth_rx.is_some() {
            return Err(anyhow::anyhow!("cTrader live auth is already in progress"));
        }
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || client_secret.is_empty() || redirect_uri.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader live auth requires client_id, client_secret, and redirect_uri"
            ));
        }
        let loopback = build_default_loopback_config(&redirect_uri)?;
        let callback_port = *loopback
            .allowed_ports()
            .first()
            .ok_or_else(|| anyhow::anyhow!("cTrader live auth has no callback ports configured"))?;
        let mut session = CTraderAuthSession::new(client_id.clone(), redirect_uri.clone());
        session.start_authorization(CTRADER_DEFAULT_SCOPE);
        session.mark_listening_for_callback(callback_port);
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);

        let request = CTraderLiveAuthRequest {
            client_id,
            client_secret,
            redirect_uri,
            scope: CTRADER_DEFAULT_SCOPE.to_string(),
            loopback,
        };
        let backend = Arc::clone(&self.ctrader_live_auth_backend);
        let (local_tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = backend.run(request).map_err(|err| err.to_string());
            let _ = local_tx.send(result.clone());
            if let Ok(_auth_result) = result {
                // Background update via ServiceEvent if needed
                // For now just prevent the unused variable warning
            }
        });
        self.ctrader_live_auth_rx = Some(rx);
        Ok(snapshot)
    }

    pub fn poll_ctrader_live_auth(&mut self) -> Option<CTraderAuthSnapshot> {
        let receiver = self.ctrader_live_auth_rx.as_ref()?;
        let outcome = match receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => {
                if let Some(session) = self.ctrader_auth.as_mut() {
                    session.mark_failed(
                        "cTrader live auth worker disconnected before returning a result",
                    );
                    let snapshot = session.snapshot();
                    self.ctrader_live_auth_rx = None;
                    return Some(snapshot);
                }
                self.ctrader_live_auth_rx = None;
                return None;
            }
        };
        self.ctrader_live_auth_rx = None;

        match outcome {
            Ok(result) => {
                {
                    let session = self.ctrader_auth.get_or_insert_with(|| {
                        CTraderAuthSession::new(
                            self.broker_settings.ctrader.client_id.clone(),
                            self.broker_settings.ctrader.redirect_uri.clone(),
                        )
                    });
                    session.mark_listening_for_callback(result.callback_port);
                    session.receive_authorization_code(result.authorization_code);
                    if let Err(err) = self
                        .ctrader_token_store
                        .save_token_bundle(&result.token_bundle)
                    {
                        session.mark_failed(format!("failed to save cTrader token bundle: {err}"));
                        return Some(session.snapshot());
                    }
                    session.restore_from_storage(result.token_bundle);
                }
                match self.discover_ctrader_accounts() {
                    Ok(Some(snapshot)) => Some(snapshot),
                    Ok(None) => self.ctrader_auth.as_ref().map(|session| session.snapshot()),
                    Err(err) => {
                        let session = self.ctrader_auth.as_mut()?;
                        session.mark_failed(format!("cTrader account discovery failed: {err}"));
                        Some(session.snapshot())
                    }
                }
            }
            Err(message) => {
                let session = self.ctrader_auth.get_or_insert_with(|| {
                    CTraderAuthSession::new(
                        self.broker_settings.ctrader.client_id.clone(),
                        self.broker_settings.ctrader.redirect_uri.clone(),
                    )
                });
                session.mark_failed(message);
                Some(session.snapshot())
            }
        }
    }

    pub fn restore_ctrader_session(&mut self) -> anyhow::Result<Option<CTraderAuthSnapshot>> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let redirect_uri = self.broker_settings.ctrader.redirect_uri.trim().to_string();
        if client_id.is_empty() || redirect_uri.is_empty() {
            return Ok(None);
        }

        let Some(bundle) = self.ctrader_token_store.load_token_bundle()? else {
            self.ctrader_auth = None;
            return Ok(None);
        };

        let mut session = CTraderAuthSession::new(client_id, redirect_uri);
        session.restore_from_storage(bundle);
        let snapshot = session.snapshot();
        self.ctrader_auth = Some(session);
        Ok(Some(snapshot))
    }

    pub fn clear_ctrader_saved_session(&mut self) -> anyhow::Result<()> {
        self.ctrader_token_store.clear_token_bundle()?;
        self.reset_connection_state();
        self.ctrader_auth = None;
        self.ctrader_live_auth_rx = None;
        Ok(())
    }

    pub fn discover_ctrader_accounts(&mut self) -> anyhow::Result<Option<CTraderAuthSnapshot>> {
        let auth_state = self.ctrader_auth.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cTrader account discovery requires a restored token session")
        })?;
        if !matches!(
            auth_state.snapshot().state,
            crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
                | crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
        ) {
            return Err(anyhow::anyhow!(
                "cTrader account discovery requires a restored token session"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader account discovery requires a stored token bundle",
            )?
            .access_token;
        let request = CTraderAccountDiscoveryRequest {
            client_id: self.broker_settings.ctrader.client_id.clone(),
            client_secret: self.broker_settings.ctrader.client_secret.clone(),
            access_token,
            environment: match self.broker_settings.ctrader.environment {
                CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
                CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
            },
        };
        let result = self
            .ctrader_account_discovery_backend
            .discover_accounts(&request)?;

        let synced_accounts = sync_ctrader_discovered_accounts_into_targets(
            &self.broker_settings.ctrader.accounts,
            &result.accounts,
        );
        self.broker_settings.ctrader.accounts = synced_accounts.clone();
        let session = self.ctrader_auth.as_mut().ok_or_else(|| {
            anyhow::anyhow!("cTrader account discovery requires a restored token session")
        })?;
        session.set_discovered_accounts(sync_discovered_accounts_with_targets(
            &result.accounts,
            &self.broker_settings.ctrader.accounts,
        ));
        Ok(Some(session.snapshot()))
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

    pub fn market_chart_snapshot(&mut self, state: &AppState) -> MarketChartSnapshot {
        let adapter_kind = self.active_adapter_kind();
        let available_timeframes = if matches!(state.data_source, DataSource::Local) {
            discover_timeframes(&state.runtime.data_dir, &state.selected_pair).unwrap_or_default()
        } else if matches!(adapter_kind, TradingAdapterKind::CTrader) {
            supported_ctrader_chart_timeframes()
        } else {
            discover_timeframes(&state.runtime.data_dir, &state.selected_pair).unwrap_or_default()
        };
        let timeframe =
            preferred_chart_timeframe(&available_timeframes, state.chart_timeframe.as_str());
        let ctrader_environment = matches!(state.data_source, DataSource::CTrader)
            .then_some(adapter_kind)
            .filter(|kind| matches!(kind, TradingAdapterKind::CTrader))
            .map(|_| self.selected_ctrader_environment());
        let ctrader_account_id = matches!(state.data_source, DataSource::CTrader)
            .then_some(adapter_kind)
            .filter(|kind| matches!(kind, TradingAdapterKind::CTrader))
            .and_then(|_| self.selected_ctrader_chart_account_id());
        let cache_key = MarketChartCacheKey {
            data_root: state.runtime.data_dir.clone(),
            data_source: state.data_source,
            adapter_kind,
            symbol: state.selected_pair.clone(),
            timeframe: timeframe.clone(),
            ctrader_environment,
            ctrader_account_id,
        };

        if let Some(cache) = &self.market_chart_cache {
            let is_live_ctrader_chart = matches!(state.data_source, DataSource::CTrader)
                && matches!(adapter_kind, TradingAdapterKind::CTrader)
                && self.connected;
            if cache.key == cache_key
                && (!is_live_ctrader_chart || cache.refreshed_at.elapsed() < Duration::from_secs(1))
            {
                return cache.snapshot.clone();
            }
        }

        let overlay_status = self.overlay_status(state);
        let resolved_timeframes = if available_timeframes.is_empty() {
            vec![timeframe.clone()]
        } else {
            available_timeframes.clone()
        };
        let snapshot = match (state.data_source, adapter_kind) {
            (DataSource::CTrader, TradingAdapterKind::CTrader) => self
                .load_ctrader_market_chart_snapshot(
                    &state.selected_pair,
                    &timeframe,
                    resolved_timeframes,
                    overlay_status,
                ),
            _ => match load_symbol_timeframe(
                &state.runtime.data_dir,
                &state.selected_pair,
                &timeframe,
            ) {
                Ok(ohlcv) => build_market_chart_snapshot_from_ohlcv(
                    &state.selected_pair,
                    &timeframe,
                    resolved_timeframes,
                    &ohlcv,
                    Vec::new(),
                    Vec::new(),
                )
                .with_overlay_status(overlay_status),
                Err(err) => MarketChartSnapshot::empty_for(
                    &state.selected_pair,
                    &timeframe,
                    if available_timeframes.is_empty() {
                        vec![timeframe.clone()]
                    } else {
                        available_timeframes.clone()
                    },
                    format!(
                        "No market data loaded for {} {}",
                        state.selected_pair, timeframe
                    ),
                    overlay_status,
                    vec![format!(
                        "Failed to load {} market data for {}: {}",
                        timeframe, state.selected_pair, err
                    )],
                ),
            },
        };

        self.market_chart_cache = Some(CachedMarketSnapshot {
            key: cache_key,
            refreshed_at: Instant::now(),
            snapshot: snapshot.clone(),
        });
        snapshot
    }

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

    pub fn start_connect(
        &mut self,
        tx: tokio::sync::mpsc::Sender<ServiceEvent>,
    ) -> anyhow::Result<()> {
        self.reap_finished_background_tasks();
        if self.background_task_running(TaskKind::Connect) {
            return Err(anyhow::anyhow!("connection attempt already in progress"));
        }
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                let request = self.build_ctrader_account_runtime_request()?;
                let backend = Arc::clone(&self.ctrader_account_runtime_backend);
                let handle =
                    std::thread::spawn(move || match backend.load_account_runtime(&request) {
                        Ok(runtime) => {
                            let _ = tx.blocking_send(ServiceEvent::CTraderConnectUpdated(runtime));
                        }
                        Err(err) => {
                            let _ = tx
                                .blocking_send(ServiceEvent::ConnectOutcome(Err(err.to_string())));
                        }
                    });
                self.connect_handle = Some(handle);
            }
            _ => {
                let _ = tx.blocking_send(ServiceEvent::ConnectOutcome(Err(format!(
                    "Adapter {:?} not implemented",
                    self.configured_adapter
                ))));
            }
        }
        Ok(())
    }

    pub fn handle_ctrader_connect_result(
        &mut self,
        state: &mut AppState,
        runtime: CTraderAccountRuntimeSnapshot,
    ) {
        self.connected = true;
        self.ctrader_runtime_refreshed_at = Some(Instant::now());
        self.terminal_info =
            format_ctrader_terminal_info(&runtime.trader, self.selected_ctrader_environment());
        if self.initial_equity.is_none() {
            self.initial_equity = Some(runtime.trader.balance);
        }
        self.day_start_equity = Some(self.day_start_equity.unwrap_or(runtime.trader.balance));
        self.adapter = Some(TradingAdapter::CTrader(runtime));
        self.market_chart_cache = None;
        self.execution_surface_cache = None;
        state.status_msg = "cTrader connected".to_string();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn connect(&mut self, state: &mut AppState) {
        let readiness = self.adapter_readiness();
        match self.configured_adapter {
            TradingAdapterKind::CTrader => {
                self.reset_connection_state();
                if !readiness.can_attempt_connect {
                    let mut failed_readiness = readiness;
                    failed_readiness.session_state = BrokerSessionState::Failed;
                    state.status_msg = failed_readiness.status_line.clone();
                    record_app_event(
                        "ui_adapter_connect",
                        "FAILED",
                        format!(
                            "{} connect blocked: {}",
                            self.configured_adapter.as_str(),
                            failed_readiness.status_line
                        ),
                    );
                    return;
                }

                match self.load_ctrader_account_runtime() {
                    Ok(runtime) => {
                        self.connected = true;
                        self.terminal_info = format_ctrader_terminal_info(
                            &runtime.trader,
                            self.selected_ctrader_environment(),
                        );
                        if self.initial_equity.is_none() {
                            self.initial_equity = Some(runtime.trader.balance); // Fixed: Initial eq is balance + 0
                        }
                        self.day_start_equity =
                            Some(self.day_start_equity.unwrap_or(runtime.trader.balance));
                        self.adapter = Some(TradingAdapter::CTrader(runtime));
                        self.market_chart_cache = None;
                        self.execution_surface_cache = None;
                        state.status_msg = "cTrader connected".to_string();
                        record_app_event(
                            "ui_adapter_connect",
                            "SUCCESS",
                            "cTrader account runtime connected",
                        );
                    }
                    Err(err) => {
                        self.reset_connection_state();
                        state.status_msg = format_ctrader_connect_error(&err);
                        record_app_event(
                            "ui_adapter_connect",
                            "FAILED",
                            format!("cTrader connect failed: {err}"),
                        );
                    }
                }
            }
            TradingAdapterKind::DxTrade => {
                self.reset_connection_state();
                if !readiness.can_attempt_connect {
                    let mut failed_readiness = readiness;
                    failed_readiness.session_state = BrokerSessionState::Failed;
                    state.status_msg = failed_readiness.status_line.clone();
                    record_app_event(
                        "ui_adapter_connect",
                        "FAILED",
                        format!(
                            "{} connect blocked: {}",
                            self.configured_adapter.as_str(),
                            failed_readiness.status_line
                        ),
                    );
                } else {
                    state.status_msg = format!(
                        "{} credentials ready · live auth is not wired yet",
                        self.configured_adapter.as_str()
                    );
                    record_app_event(
                        "ui_adapter_connect",
                        "DEGRADED",
                        format!(
                            "{} credentials ready but live auth is not wired yet",
                            self.configured_adapter.as_str()
                        ),
                    );
                }
            }
        }
    }

    pub fn disconnect(&mut self, state: &mut AppState) {
        self.reset_connection_state();
        state.status_msg = "Offline".to_string();
        record_app_event("ui_disconnect", "SUCCESS", "UI broker connection closed");
    }

    pub fn execute_buy_market(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Buy);
    }

    pub fn execute_sell_market(&mut self, state: &mut AppState) {
        self.execute_ctrader_order(state, CTraderTradeSide::Sell);
    }

    pub fn cancel_selected_order(&mut self, state: &mut AppState) {
        let Some(order_id) = state.order_ticket.selected_order_id.or_else(|| {
            self.connected_ctrader_runtime().and_then(|runtime| {
                runtime
                    .reconcile
                    .pending_orders
                    .first()
                    .map(|order| order.order_id)
            })
        }) else {
            let message = "No pending cTrader order is selected for cancellation.".to_string();
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_cancel_order", "FAILED", message);
            return;
        };

        // HARD FAIL: silently defaulting account_id to 0 here would target
        // whichever account the broker resolves "0" to. Refuse the request
        // instead so the operator sees a real error.
        let account_id = match self
            .selected_ctrader_execution_account_id()
            .and_then(|id| id.parse::<i64>().ok())
        {
            Some(id) => id,
            None => {
                let message =
                    "cTrader order cancel rejected: no execution account selected/parseable"
                        .to_string();
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_cancel_order", "FAILED", message);
                return;
            }
        };
        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::CancelOrder(CTraderCancelOrderRequest {
                account_id,
                order_id,
            }),
            format!("Cancel order #{order_id}"),
        ) {
            Ok(outcome) => {
                state.status_msg = format_execution_outcome_status("Cancelled order", &outcome);
                state.order_ticket.selected_order_id = Some(order_id);
            }
            Err(err) => {
                let message = format!("cTrader order cancel failed: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_cancel_order", "FAILED", message);
            }
        }
    }

    pub fn close_selected_position(&mut self, state: &mut AppState) {
        let Some(position_id) = state.order_ticket.selected_position_id.or_else(|| {
            self.connected_ctrader_runtime().and_then(|runtime| {
                runtime
                    .reconcile
                    .positions
                    .first()
                    .map(|position| position.position_id)
            })
        }) else {
            let message = "No open cTrader position is selected for closing.".to_string();
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_close_position", "FAILED", message);
            return;
        };

        let Some(volume) = self
            .connected_ctrader_runtime()
            .and_then(|runtime| {
                runtime
                    .reconcile
                    .positions
                    .iter()
                    .find(|position| position.position_id == position_id)
            })
            .map(|position| position.volume)
        else {
            let message =
                format!("Selected cTrader position #{position_id} is no longer available.");
            state.status_msg = message.clone();
            self.append_trade_journal(message.clone());
            record_app_event("ctrader_close_position", "FAILED", message);
            return;
        };

        // audit-fix F5: surface overflow at the caller rather than letting
        // the silent cast through.
        let protocol_volume = match ctrader_protocol_volume_from_units(volume) {
            Ok(v) => v,
            Err(err) => {
                let message = format!("cTrader close-position rejected: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
                return;
            }
        };
        // HARD FAIL: same reasoning as cancel_order — refusing to send a
        // close-position request without a parseable account id is safer
        // than letting the broker resolve account_id=0.
        let account_id = match self
            .selected_ctrader_execution_account_id()
            .and_then(|id| id.parse::<i64>().ok())
        {
            Some(id) => id,
            None => {
                let message =
                    "cTrader position close rejected: no execution account selected/parseable"
                        .to_string();
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
                return;
            }
        };
        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::ClosePosition(CTraderClosePositionRequest {
                account_id,
                position_id,
                volume: protocol_volume,
            }),
            format!("Close position #{position_id}"),
        ) {
            Ok(outcome) => {
                state.status_msg = format_execution_outcome_status("Closed position", &outcome);
                state.order_ticket.selected_position_id = Some(position_id);
            }
            Err(err) => {
                let message = format!("cTrader position close failed: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_close_position", "FAILED", message);
            }
        }
    }

    pub fn select_adapter(&mut self, state: &mut AppState, kind: TradingAdapterKind) {
        let previous = self.active_adapter_kind();
        self.reset_connection_state();
        self.configured_adapter = kind;
        state.status_msg = match state.data_source {
            DataSource::Local => "Local Mode".to_string(),
            DataSource::CTrader => format!("{} selected · disconnected", kind.as_str()),
        };
        if matches!(kind, TradingAdapterKind::CTrader)
            && let Ok(Some(_)) = self.restore_ctrader_session()
        {
            state.status_msg = "cTrader selected · session restored".to_string();
        }
        record_app_event(
            "ui_adapter_select",
            "SUCCESS",
            format!(
                "selected trading adapter {} (previous {})",
                kind.as_str(),
                previous.as_str()
            ),
        );
    }

    fn overlay_status(&self, state: &AppState) -> String {
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

    fn active_adapter_kind(&self) -> TradingAdapterKind {
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

    fn reset_connection_state(&mut self) {
        self.adapter = None;
        self.connected = false;
        self.terminal_info.clear();
        self.market_chart_cache = None;
        self.execution_surface_cache = None;
        self.ctrader_live_spot_cache = None;
        self.ctrader_runtime_refreshed_at = None;
    }

    fn execute_ctrader_order(&mut self, state: &mut AppState, side: CTraderTradeSide) {
        match self.build_ctrader_order_request(state, side) {
            Ok(order_request) => {
                let account_equity = self.ctrader_account_equity();
                let pip_position = self
                    .ctrader_symbol_pip_position(&state.selected_pair)
                    .unwrap_or(4);
                if let Err(err) = prop_firm_pre_trade_check(
                    &state.risk,
                    &order_request,
                    account_equity,
                    self.initial_equity.unwrap_or(account_equity),
                    self.day_start_equity.unwrap_or(account_equity),
                    pip_position,
                    &state.selected_pair,
                ) {
                    let message = format!("Prop-firm risk gate blocked: {err}");
                    state.status_msg = message.clone();
                    self.append_trade_journal(message.clone());
                    record_app_event("prop_firm_risk_gate", "BLOCKED", message);
                    return;
                }
                match self.execute_ctrader_request(
                    state,
                    CTraderExecutionRequest::NewOrder(Box::new(order_request)),
                    format!("{} {}", side.label(), state.selected_pair),
                ) {
                    Ok(outcome) => {
                        state.status_msg = format_execution_outcome_status(
                            &format!("{} {}", side.label(), state.selected_pair),
                            &outcome,
                        );
                    }
                    Err(err) => {
                        let message = format!("cTrader order failed: {err}");
                        state.status_msg = message.clone();
                        self.append_trade_journal(message.clone());
                        record_app_event("ctrader_order", "FAILED", message);
                    }
                }
            }
            Err(err) => {
                let message = format!("cTrader order ticket invalid: {err}");
                state.status_msg = message.clone();
                self.append_trade_journal(message.clone());
                record_app_event("ctrader_market_order", "FAILED", message);
            }
        }
    }

    fn execute_ctrader_request(
        &mut self,
        state: &mut AppState,
        request: CTraderExecutionRequest,
        operator_action: String,
    ) -> anyhow::Result<CTraderExecutionOutcome> {
        if self.configured_adapter != TradingAdapterKind::CTrader {
            return Err(anyhow::anyhow!(
                "cTrader execution is only available when the cTrader adapter is selected"
            ));
        }
        if !self.connected {
            return Err(anyhow::anyhow!("cTrader runtime is not connected"));
        }

        let runtime_request = self.build_ctrader_execution_runtime_request(request.clone())?;
        let outcome = match self.ctrader_execution_backend.execute(&runtime_request) {
            Ok(outcome) => outcome,
            Err(err) => {
                // D11: cTrader signalled an OAuth-token failure. Force-
                // refresh the bundle (bypassing the time-window check) and
                // retry once. If refresh or retry also fails, surface the
                // original error so the operator sees the broker message.
                if !err.to_string().contains(CTRADER_TOKEN_EXPIRED_SENTINEL) {
                    return Err(err);
                }
                let warn = format!(
                    "cTrader token rejected by broker — forcing OAuth refresh and retrying: {err}"
                );
                self.append_trade_journal(warn.clone());
                state.status_msg = warn.clone();
                record_app_event("ctrader_token_refresh", "FORCED", warn);
                if let Err(refresh_err) = self.force_refresh_ctrader_token_bundle() {
                    return Err(refresh_err.context(err));
                }

                // SECURITY (audit-fix F3): before resubmitting the order
                // under the refreshed token, ask the broker whether this
                // `client_order_id` is already present. The original
                // attempt may have been accepted by the broker before the
                // network connection died — in which case retrying would
                // double the position. If reconcile fails, we do NOT
                // retry: surface the error so the operator can decide.
                if let Some(client_order_id) =
                    extract_client_order_id_from_request(&request)
                {
                    let reconcile = self.load_ctrader_account_runtime().map_err(|reconcile_err| {
                        anyhow::anyhow!(
                            "cTrader retry aborted: reconcile-before-retry failed and we cannot prove the previous \
                             attempt was not already accepted by the broker (client_order_id={client_order_id}). \
                             Original error: {err}. Reconcile error: {reconcile_err}"
                        )
                    })?;
                    if let Some(existing) =
                        find_existing_client_order_id(&reconcile.reconcile, &client_order_id)
                    {
                        let message = format!(
                            "cTrader retry skipped: broker already has client_order_id={client_order_id} ({existing}); \
                             treating as success to avoid duplicate order"
                        );
                        self.append_trade_journal(message.clone());
                        state.status_msg = message.clone();
                        record_app_event("ctrader_retry_duplicate_skipped", "SUCCESS", message);
                        return Ok(synthesize_idempotent_retry_outcome(
                            &reconcile.reconcile,
                            &client_order_id,
                        ));
                    }
                }

                let retry_request =
                    self.build_ctrader_execution_runtime_request(request.clone())?;
                self.ctrader_execution_backend.execute(&retry_request)?
            }
        };
        let journal_line = format_execution_journal_line(&operator_action, &outcome);
        self.append_trade_journal(journal_line.clone());
        record_app_event(
            "ctrader_order_execution",
            match outcome.status {
                CTraderExecutionStatus::Failed => "FAILED",
                CTraderExecutionStatus::Cancelled => "SUCCESS",
                CTraderExecutionStatus::Accepted
                | CTraderExecutionStatus::Filled
                | CTraderExecutionStatus::Replaced
                | CTraderExecutionStatus::PartialFill => "SUCCESS",
            },
            journal_line,
        );
        if let Err(err) = self.refresh_ctrader_runtime_after_execution() {
            let message =
                format!("cTrader execution succeeded but runtime refresh degraded: {err}");
            self.append_trade_journal(message.clone());
            state.status_msg = message.clone();
            record_app_event("ctrader_order_execution_refresh", "DEGRADED", message);
        }
        self.execution_surface_cache = None;
        self.market_chart_cache = None;
        Ok(outcome)
    }

    fn build_ctrader_execution_runtime_request(
        &mut self,
        request: CTraderExecutionRequest,
    ) -> anyhow::Result<CTraderExecutionRuntimeRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader execution requires configured client_id and client_secret"
            ));
        }
        let access_token = self
            .ensure_fresh_ctrader_token_bundle("cTrader execution requires a stored token bundle")?
            .access_token;
        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader execution requires a selected discovered account")
            })?;
        Ok(CTraderExecutionRuntimeRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            request,
        })
    }

    fn calculate_smart_atr_in_points(&self, _state: &AppState, symbol_name: &str) -> Option<i64> {
        let cache_entry = self.market_chart_cache.as_ref()?;
        let chart = &cache_entry.snapshot;
        if chart.candles.len() < 14 {
            return None;
        }
        let candles = &chart.candles[chart.candles.len() - 14..];
        let mut tr_sum = 0.0;
        for i in 1..candles.len() {
            let current = &candles[i];
            let prev = &candles[i - 1];
            let hl = current.high - current.low;
            let hc = (current.high - prev.close).abs();
            let lc = (current.low - prev.close).abs();
            let tr = hl.max(hc).max(lc);
            tr_sum += tr;
        }
        let atr = tr_sum / 13.0; // simple average of the 13 computed TRs

        // Convert ATR price delta into points (pipettes)
        let pip_position = self.ctrader_symbol_pip_position(symbol_name).unwrap_or(4);
        let point_multiplier = 10f64.powi(pip_position + 1);

        let atr_points = atr * point_multiplier;
        Some(atr_points as i64)
    }

    fn build_ctrader_order_request(
        &mut self,
        state: &AppState,
        side: CTraderTradeSide,
    ) -> anyhow::Result<CTraderNewOrderRequest> {
        let resolved = self.resolve_selected_ctrader_symbol(&state.selected_pair)?;
        let protocol_volume = validate_and_convert_lot_size_to_ctrader_volume(
            &state.order_ticket,
            state.risk.max_lot_size,
            &resolved.symbol,
        )?;

        let mut relative_stop_loss = None;
        let mut relative_take_profit = None;

        if state.order_ticket.smart_sl_enabled {
            if let Some(atr_points) =
                self.calculate_smart_atr_in_points(state, &state.selected_pair)
            {
                // Calculate based on dynamic volatility
                let sl_mult = 1.5;
                let tp_mult = sl_mult * state.order_ticket.smart_rr_ratio; // standard RR 2.0 -> SL=1.5x, TP=3.0x

                relative_stop_loss = Some((atr_points as f64 * sl_mult) as i64);
                relative_take_profit = Some((atr_points as f64 * tp_mult) as i64);

                tracing::info!(
                    "Smart SL applied: ATR={}pts, SL={:?}, TP={:?} (RR={})",
                    atr_points,
                    relative_stop_loss,
                    relative_take_profit,
                    state.order_ticket.smart_rr_ratio
                );
            } else {
                tracing::warn!(
                    "Smart SL requested but not enough trailing candles for ATR. Sending order without SL/TP bounds or falling back to defaults."
                );
            }
        }

        let order_type = match state.order_ticket.order_type {
            crate::app_state::OrderType::Market => CTraderOrderType::Market,
            crate::app_state::OrderType::Limit => CTraderOrderType::Limit,
            crate::app_state::OrderType::Stop => CTraderOrderType::Stop,
        };

        let (limit_price, stop_price) = match order_type {
            CTraderOrderType::Market => (None, None),
            CTraderOrderType::Limit => (Some(state.order_ticket.target_price), None),
            CTraderOrderType::Stop => (None, Some(state.order_ticket.target_price)),
            _ => (None, None),
        };

        Ok(CTraderNewOrderRequest {
            account_id: resolved.account_id,
            symbol_id: resolved.light_symbol.symbol_id,
            order_type,
            trade_side: side,
            volume: protocol_volume,
            limit_price,
            stop_price,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: None, // We use relative points below
            take_profit: None,
            comment: non_empty_option(&state.order_ticket.comment),
            base_slippage_price: None,
            slippage_in_points: Some(state.order_ticket.slippage_in_points),
            label: non_empty_option(&state.order_ticket.label),
            position_id: None,
            client_order_id: Some(format!(
                "{}-{}-{}-{:x}",
                side.label().to_ascii_lowercase(),
                state.selected_pair.to_ascii_lowercase(),
                // DOCUMENTED-DEFAULT: timestamp is decorative; `next_client_order_seq`
                // is the actual uniqueness guarantee. A clock-before-epoch failure
                // would just yield "0-<seq>" which is still unique.
                current_unix_seconds().unwrap_or_default(),
                next_client_order_seq()
            )),
            relative_stop_loss,
            relative_take_profit,
            guaranteed_stop_loss: None,
            trailing_stop_loss: state.order_ticket.trailing_stop.then_some(true),
            stop_trigger_method: None,
        })
    }

    fn resolve_selected_ctrader_symbol(
        &mut self,
        symbol_name: &str,
    ) -> anyhow::Result<crate::app_services::ctrader_data::CTraderResolvedSymbol> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader symbol resolution requires configured client_id and client_secret"
            ));
        }
        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader symbol resolution requires a stored token bundle",
            )?
            .access_token;
        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader symbol resolution requires a selected discovered account")
            })?;
        resolve_symbol(&CTraderSymbolLookupRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            symbol_name: symbol_name.to_string(),
        })
    }

    /// Live account equity = balance + sum of mark-to-market unrealized PnL.
    ///
    /// Critical for prop-firm rules: every published challenge measures
    /// drawdown by EQUITY, not balance, so an open losing position MUST be
    /// counted before the gate fires. `unrealized_pnl` is fed by the
    /// streaming subsystem (set to 0.0 until that wire is in); when 0.0
    /// while positions are open we surface a one-shot warning so the
    /// operator notices the missing live update.
    fn ctrader_account_equity(&self) -> f64 {
        let runtime = match self.connected_ctrader_runtime() {
            Some(r) => r,
            None => return 0.0,
        };
        let balance = runtime.trader.balance;
        let unrealized = runtime.trader.unrealized_pnl;
        if !runtime.reconcile.positions.is_empty() && unrealized == 0.0 {
            tracing::warn!(
                target: "forex_app::risk",
                positions = runtime.reconcile.positions.len(),
                "ctrader equity computed without unrealized PnL; daily-DD check is balance-only \
                 until the streaming subsystem populates trader.unrealized_pnl"
            );
        }
        balance + unrealized
    }

    /// Pip position (decimal places of one pip) for a forex symbol.
    ///
    /// The bot is FX-only — JPY pairs use 2 decimal pip notation, every
    /// other major/minor uses 4. We deliberately do NOT branch on metals or
    /// crypto here because the bot doesn't trade them; if an unknown symbol
    /// shape arrives, log a structured warn and default to 4 so operators
    /// can spot the mis-routed instrument instead of silently mispricing it.
    fn ctrader_symbol_pip_position(&self, symbol: &str) -> Option<i32> {
        let normalized = symbol.to_ascii_uppercase();
        if normalized.contains("JPY") {
            return Some(2);
        }
        // Heuristic: real FX symbols are exactly 6 alphabetic characters
        // (EURUSD, GBPCHF, ...). Anything else is suspicious in a forex-only
        // bot — log a warn but still return a sane default so we don't crash.
        let looks_like_fx_pair =
            normalized.len() == 6 && normalized.chars().all(|c| c.is_ascii_alphabetic());
        if !looks_like_fx_pair {
            tracing::warn!(
                target: "forex_app::risk",
                symbol,
                "symbol does not look like a 6-letter FX pair; defaulting pip_position=4"
            );
        }
        Some(4)
    }

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

    fn refresh_ctrader_runtime_after_execution(&mut self) -> anyhow::Result<()> {
        let runtime = self.load_ctrader_account_runtime()?;
        self.terminal_info =
            format_ctrader_terminal_info(&runtime.trader, self.selected_ctrader_environment());
        self.adapter = Some(TradingAdapter::CTrader(runtime));
        self.connected = true;
        self.ctrader_runtime_refreshed_at = Some(Instant::now());
        self.execution_surface_cache = None;
        Ok(())
    }

    fn connected_ctrader_runtime(&self) -> Option<&CTraderAccountRuntimeSnapshot> {
        match &self.adapter {
            Some(TradingAdapter::CTrader(runtime)) if self.connected => Some(runtime),
            _ => None,
        }
    }

    fn append_trade_journal(&mut self, line: String) {
        self.trade_journal.push(line);
        if self.trade_journal.len() > 16 {
            let overflow = self.trade_journal.len() - 16;
            self.trade_journal.drain(0..overflow);
        }
        self.execution_surface_cache = None;
    }

    fn load_ctrader_market_chart_snapshot(
        &mut self,
        symbol: &str,
        timeframe: &str,
        available_timeframes: Vec<String>,
        overlay_status: String,
    ) -> MarketChartSnapshot {
        match self.build_ctrader_chart_history_request(symbol, timeframe) {
            Ok(request) => match load_chart_history(&request) {
                Ok(history) => {
                    let mut warnings = Vec::new();
                    let live_update = if self.connected {
                        match self.build_ctrader_live_chart_update_request(
                            &request,
                            history.symbol.symbol_id,
                            history.symbol.digits,
                        ) {
                            Ok(live_request) => {
                                match self.load_ctrader_live_chart_update_cached(&live_request) {
                                    Ok(update) => Some(update),
                                    Err(err) => {
                                        warnings.push(format!(
                                            "Failed to load cTrader live {} update for {}: {}",
                                            timeframe, symbol, err
                                        ));
                                        None
                                    }
                                }
                            }
                            Err(err) => {
                                warnings.push(format!(
                                    "cTrader live {} update is unavailable for {}: {}",
                                    timeframe, symbol, err
                                ));
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let bars =
                        merge_live_spot_update_into_bars(&history.bars, live_update.as_ref());

                    let mut snapshot = build_market_chart_snapshot_from_historical_bars(
                        &history.symbol.symbol_name,
                        timeframe,
                        available_timeframes,
                        &bars,
                        Vec::new(),
                        warnings,
                    )
                    .with_overlay_status(overlay_status);
                    if let Some(update) = live_update {
                        snapshot.bid = update.bid;
                        snapshot.ask = update.ask;
                        let quote_line = match (update.bid, update.ask) {
                            (Some(bid), Some(ask)) => {
                                format!(" · bid {:.5} ask {:.5}", bid, ask)
                            }
                            (Some(bid), None) => format!(" · bid {:.5}", bid),
                            (None, Some(ask)) => format!(" · ask {:.5}", ask),
                            (None, None) => String::new(),
                        };
                        if !quote_line.is_empty() {
                            snapshot.headline.push_str(&quote_line);
                        }
                    }
                    snapshot
                }
                Err(err) => MarketChartSnapshot::empty_for(
                    symbol,
                    timeframe,
                    available_timeframes,
                    format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                    overlay_status,
                    vec![format!(
                        "Failed to load cTrader {} market data for {}: {}",
                        timeframe, symbol, err
                    )],
                ),
            },
            Err(err) => MarketChartSnapshot::empty_for(
                symbol,
                timeframe,
                available_timeframes,
                format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                overlay_status,
                vec![format!(
                    "cTrader chart history is unavailable for {} {}: {}",
                    symbol, timeframe, err
                )],
            ),
        }
    }

    fn load_ctrader_live_chart_update_cached(
        &mut self,
        request: &CTraderLiveChartUpdateRequest,
    ) -> anyhow::Result<CTraderLiveChartUpdate> {
        let cache_key = CTraderLiveSpotCacheKey {
            environment: request.environment,
            account_id: request.account_id.clone(),
            symbol_id: request.symbol_id,
            timeframe: request.timeframe.clone(),
        };

        if let Some(cache) = &self.ctrader_live_spot_cache
            && cache.key == cache_key
            && cache.refreshed_at.elapsed() < Duration::from_secs(1)
        {
            return Ok(cache.update.clone());
        }

        let update = self
            .ctrader_live_streaming_backend
            .load_live_chart_update(request)?;
        self.ctrader_live_spot_cache = Some(CachedCTraderLiveSpotUpdate {
            key: cache_key,
            refreshed_at: Instant::now(),
            update: update.clone(),
        });
        Ok(update)
    }

    fn build_ctrader_chart_history_request(
        &mut self,
        symbol: &str,
        timeframe: &str,
    ) -> anyhow::Result<CTraderChartHistoryRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader chart history requires configured client_id and client_secret"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle(
                "cTrader chart history requires a stored token bundle",
            )?
            .access_token;

        let account_id = self.selected_ctrader_chart_account_id().ok_or_else(|| {
            anyhow::anyhow!("cTrader chart history requires at least one discovered account")
        })?;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
            .as_millis() as i64;
        let window_ms = chart_history_window_ms(timeframe)
            .ok_or_else(|| anyhow::anyhow!("unsupported cTrader chart timeframe {}", timeframe))?;

        Ok(CTraderChartHistoryRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            symbol_name: symbol.to_string(),
            timeframe: timeframe.to_string(),
            from_timestamp_ms: now_ms.saturating_sub(window_ms),
            to_timestamp_ms: now_ms,
            count: Some((MAX_CHART_CANDLES + 24) as u32),
        })
    }

    fn build_ctrader_live_chart_update_request(
        &self,
        history_request: &CTraderChartHistoryRequest,
        symbol_id: i64,
        digits: i32,
    ) -> anyhow::Result<CTraderLiveChartUpdateRequest> {
        if history_request.account_id.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader live chart update requires a discovered account"
            ));
        }

        Ok(CTraderLiveChartUpdateRequest {
            client_id: history_request.client_id.clone(),
            client_secret: history_request.client_secret.clone(),
            access_token: history_request.access_token.clone(),
            environment: history_request.environment,
            account_id: history_request.account_id.clone(),
            symbol_id,
            digits,
            timeframe: history_request.timeframe.clone(),
            subscribe_to_spot_timestamp: true,
        })
    }

    fn selected_ctrader_chart_account_id(&self) -> Option<String> {
        self.broker_settings
            .ctrader
            .accounts
            .iter()
            .find(|account| account.enabled_for_execution)
            .or_else(|| self.broker_settings.ctrader.accounts.first())
            .map(|account| account.account_id.clone())
    }

    fn selected_ctrader_environment(&self) -> CTraderEnvironment {
        match self.broker_settings.ctrader.environment {
            CTraderBrokerEnvironment::Live => CTraderEnvironment::Live,
            CTraderBrokerEnvironment::Demo => CTraderEnvironment::Demo,
        }
    }

    fn load_ctrader_account_runtime(&mut self) -> anyhow::Result<CTraderAccountRuntimeSnapshot> {
        let request = self.build_ctrader_account_runtime_request()?;
        self.ctrader_account_runtime_backend
            .load_account_runtime(&request)
    }

    fn build_ctrader_account_runtime_request(
        &mut self,
    ) -> anyhow::Result<CTraderAccountRuntimeRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader account runtime requires configured client_id and client_secret"
            ));
        }

        if self.ctrader_auth.is_none() {
            self.restore_ctrader_session()?;
        }

        let auth_state = self
            .ctrader_auth
            .as_ref()
            .map(|session| session.snapshot().state);
        if !matches!(
            auth_state,
            Some(crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage)
                | Some(crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable)
        ) {
            return Err(anyhow::anyhow!(
                "cTrader connect requires a restored token session"
            ));
        }

        let access_token = self
            .ensure_fresh_ctrader_token_bundle("cTrader connect requires a stored token bundle")?
            .access_token;

        let account_id = self
            .selected_ctrader_execution_account_id()
            .ok_or_else(|| {
                anyhow::anyhow!("cTrader account runtime requires at least one discovered account")
            })?;

        Ok(CTraderAccountRuntimeRequest {
            client_id,
            client_secret,
            access_token,
            environment: self.selected_ctrader_environment(),
            account_id,
            return_protection_orders: true,
        })
    }

    fn resolve_ctrader_bootstrap_context(&mut self) -> anyhow::Result<CTraderBootstrapContext> {
        let request = self.build_ctrader_account_runtime_request()?;
        Ok(CTraderBootstrapContext {
            client_id: request.client_id,
            client_secret: request.client_secret,
            access_token: request.access_token,
            environment: request.environment,
            account_id: request.account_id,
        })
    }

    fn background_task_running(&self, kind: TaskKind) -> bool {
        match kind {
            TaskKind::Connect => self
                .connect_handle
                .as_ref()
                .is_some_and(|handle| !handle.is_finished()),
            TaskKind::Bootstrap => self
                .bootstrap_handle
                .as_ref()
                .is_some_and(|handle| !handle.is_finished()),
        }
    }

    fn reap_finished_background_tasks(&mut self) {
        if self
            .connect_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = self.connect_handle.take()
        {
            let _ = handle.join();
        }
        if self
            .bootstrap_handle
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = self.bootstrap_handle.take()
        {
            let _ = handle.join();
        }
    }

    fn selected_ctrader_execution_account_id(&self) -> Option<String> {
        self.broker_settings
            .ctrader
            .accounts
            .iter()
            .find(|account| account.enabled_for_execution)
            .or_else(|| self.broker_settings.ctrader.accounts.first())
            .map(|account| account.account_id.clone())
    }

    fn ensure_fresh_ctrader_token_bundle(
        &mut self,
        missing_bundle_message: &str,
    ) -> anyhow::Result<CTraderTokenBundle> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader token refresh requires configured client_id and client_secret"
            ));
        }

        let bundle = self
            .ctrader_token_store
            .load_token_bundle()?
            .ok_or_else(|| anyhow::anyhow!(missing_bundle_message.to_string()))?;
        let now_unix = current_unix_seconds()?;
        if !bundle.needs_refresh_at(now_unix, CTRADER_TOKEN_REFRESH_WINDOW_SECS) {
            return Ok(bundle);
        }

        self.refresh_ctrader_token_bundle(&client_id, &client_secret, &bundle)
    }

    /// D11: force a token refresh regardless of the local expiry timer.
    /// Used when the broker rejects the current `access_token` mid-session
    /// (e.g. server-side revocation, clock skew). Returns the new bundle so
    /// the next execution request rebuilds with a valid token.
    fn force_refresh_ctrader_token_bundle(&mut self) -> anyhow::Result<CTraderTokenBundle> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self
            .broker_settings
            .ctrader
            .client_secret
            .trim()
            .to_string();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader token refresh requires configured client_id and client_secret"
            ));
        }
        let bundle = self
            .ctrader_token_store
            .load_token_bundle()?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cTrader force-refresh requires a stored token bundle with a refresh token"
                )
            })?;
        self.refresh_ctrader_token_bundle(&client_id, &client_secret, &bundle)
    }

    fn refresh_ctrader_token_bundle(
        &mut self,
        client_id: &str,
        client_secret: &str,
        bundle: &CTraderTokenBundle,
    ) -> anyhow::Result<CTraderTokenBundle> {
        let refreshed_bundle =
            self.ctrader_live_auth_backend
                .refresh_token_bundle(&CTraderTokenRefreshRequest {
                    client_id: client_id.to_string(),
                    client_secret: client_secret.to_string(),
                    refresh_token: bundle.refresh_token.clone(),
                    scope: bundle.scope.clone(),
                })?;
        self.ctrader_token_store
            .save_token_bundle(&refreshed_bundle)
            .map_err(|err| {
                anyhow::anyhow!("failed to persist refreshed cTrader token bundle: {err}")
            })?;
        if let Some(session) = self.ctrader_auth.as_mut() {
            session.replace_persisted_token_bundle(refreshed_bundle.clone());
        }
        Ok(refreshed_bundle)
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
    fn with_overlay_status(mut self, overlay_status: String) -> Self {
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
