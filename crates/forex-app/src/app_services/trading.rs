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
    CTraderAmendOrderRequest, CTraderCancelOrderRequest, CTraderClosePositionRequest,
    CTraderNewOrderRequest, CTraderOrderTriggerMethod, CTraderOrderType, CTraderTimeInForce,
    CTraderTradeSide, SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS, SUPPORTED_CTRADER_ORDER_TYPES,
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

const CTRADER_TOKEN_REFRESH_WINDOW_SECS: i64 = 300;

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
        session.broker_settings =
            crate::app_services::broker_persistence::load_broker_settings();
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

    #[allow(dead_code)]
    fn run_ctrader_bootstrap_batch_legacy(
        &mut self,
        data_root: &std::path::Path,
        symbols: &[String],
        timeframes: &[String],
        years: u32,
    ) -> anyhow::Result<JobSnapshot> {
        if symbols.is_empty() || timeframes.is_empty() {
            return Err(anyhow::anyhow!(
                "bootstrap requires at least one symbol and one timeframe"
            ));
        }

        let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
        snapshot.state = JobState::Running;
        snapshot.progress.stage = "bootstrap_planning".to_string();
        snapshot.progress.message = format!(
            "Preparing {} symbols across {} timeframes",
            symbols.len(),
            timeframes.len()
        );
        snapshot
            .report
            .counters
            .push(("requested_symbols".to_string(), symbols.len() as u64));
        snapshot
            .report
            .counters
            .push(("requested_timeframes".to_string(), timeframes.len() as u64));
        snapshot
            .report
            .counters
            .push(("requested_years".to_string(), years as u64));

        let total_requests = (symbols.len() * timeframes.len()) as u64;
        let mut completed = 0_u64;
        let mut successes = 0_u64;
        let mut degraded = 0_u64;
        let mut failures = 0_u64;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
            .as_millis() as i64;

        for symbol in symbols {
            for timeframe in timeframes {
                let planned_chunks =
                    plan_bootstrap_chunks(now_ms, timeframe, years).with_context(|| {
                        format!(
                            "failed to plan cTrader bootstrap chunks for {} {} over {} years",
                            symbol, timeframe, years
                        )
                    })?;
                snapshot.progress.stage = "bootstrap_fetch".to_string();
                snapshot.progress.message = format!("Bootstrapping {symbol} {timeframe}");
                snapshot.report.events = push_recent_event(
                    &snapshot.report.events,
                    JobEventLevel::Info,
                    format!(
                        "bootstrap started for {symbol} {timeframe} with {} planned chunks",
                        planned_chunks.len()
                    ),
                );

                let mut request = self.build_ctrader_chart_history_request(symbol, timeframe)?;
                request.count = None;
                let outcome = bootstrap_from_ctrader_history(data_root, &request, now_ms, years);
                completed += 1;
                snapshot.progress.percent = Some(completed as f32 / total_requests as f32);

                match outcome {
                    Ok(outcome) => {
                        let missing_segments = outcome.coverage.missing_segments.len();
                        if outcome.coverage.fully_covered {
                            successes += 1;
                        } else {
                            degraded += 1;
                            snapshot.report.warnings.push(format!(
                                "{} {} has {} uncovered segments after bootstrap",
                                symbol, timeframe, missing_segments
                            ));
                        }
                        snapshot.report.entries.push(format!(
                            "{} {} | planned_chunks={} | bars_written={} | covered_segments={} | missing_segments={}",
                            symbol,
                            timeframe,
                            planned_chunks.len(),
                            outcome.bars_written,
                            outcome.coverage.covered_segments.len(),
                            missing_segments
                        ));
                        snapshot.report.events = push_recent_event(
                            &snapshot.report.events,
                            if outcome.coverage.fully_covered {
                                JobEventLevel::Info
                            } else {
                                JobEventLevel::Warning
                            },
                            format!(
                                "bootstrap finished for {symbol} {timeframe} with {} covered segments",
                                outcome.coverage.covered_segments.len()
                            ),
                        );
                    }
                    Err(err) => {
                        failures += 1;
                        snapshot
                            .report
                            .errors
                            .push(format!("{symbol} {timeframe}: {err}"));
                        snapshot.report.events = push_recent_event(
                            &snapshot.report.events,
                            JobEventLevel::Error,
                            format!("bootstrap failed for {symbol} {timeframe}: {err}"),
                        );
                    }
                }
            }
        }

        snapshot
            .report
            .counters
            .push(("completed_requests".to_string(), completed));
        snapshot
            .report
            .counters
            .push(("succeeded_requests".to_string(), successes));
        snapshot
            .report
            .counters
            .push(("degraded_requests".to_string(), degraded));
        snapshot
            .report
            .counters
            .push(("failed_requests".to_string(), failures));
        snapshot.report.highlights.push((
            "requests".to_string(),
            format!("{}/{} completed", completed, total_requests),
        ));
        snapshot.report.summary = format!(
            "Bootstrap finished: {} succeeded, {} degraded, {} failed",
            successes, degraded, failures
        );
        snapshot.report.log_path = Some("logs/forex-ai.log".to_string());
        snapshot.state = if failures == total_requests {
            JobState::Failed
        } else if failures > 0 || degraded > 0 {
            JobState::Degraded
        } else {
            JobState::Succeeded
        };
        snapshot.progress.stage = "bootstrap_complete".to_string();
        snapshot.progress.message = snapshot.report.summary.clone();
        Ok(snapshot)
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

        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::CancelOrder(CTraderCancelOrderRequest {
                account_id: self
                    .selected_ctrader_execution_account_id()
                    .and_then(|id| id.parse::<i64>().ok())
                    .unwrap_or_default(),
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

        match self.execute_ctrader_request(
            state,
            CTraderExecutionRequest::ClosePosition(CTraderClosePositionRequest {
                account_id: self
                    .selected_ctrader_execution_account_id()
                    .and_then(|id| id.parse::<i64>().ok())
                    .unwrap_or_default(),
                position_id,
                volume: ctrader_protocol_volume_from_units(volume),
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
        let outcome = self.ctrader_execution_backend.execute(&runtime_request)?;
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
                "{}-{}-{}",
                side.label().to_ascii_lowercase(),
                state.selected_pair.to_ascii_lowercase(),
                current_unix_seconds().unwrap_or_default()
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
        let looks_like_fx_pair = normalized.len() == 6 && normalized.chars().all(|c| c.is_ascii_alphabetic());
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
        if let Some(runtime) = self.connected_ctrader_runtime() {
            let live_equity = runtime.trader.balance + runtime.trader.unrealized_pnl;
            self.day_start_equity = Some(live_equity);
            tracing::info!(
                target: "forex_app::risk",
                day_id,
                day_start_equity = live_equity,
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

        let refreshed_bundle =
            self.ctrader_live_auth_backend
                .refresh_token_bundle(&CTraderTokenRefreshRequest {
                    client_id,
                    client_secret,
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

fn run_ctrader_bootstrap_batch_with_context(
    context: &CTraderBootstrapContext,
    data_root: &std::path::Path,
    symbols: &[String],
    timeframes: &[String],
    years: u32,
) -> anyhow::Result<JobSnapshot> {
    if symbols.is_empty() || timeframes.is_empty() {
        return Err(anyhow::anyhow!(
            "bootstrap requires at least one symbol and one timeframe"
        ));
    }

    let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
    snapshot.state = JobState::Running;
    snapshot.progress.stage = "bootstrap_planning".to_string();
    snapshot.progress.message = format!(
        "Preparing {} symbols across {} timeframes",
        symbols.len(),
        timeframes.len()
    );
    snapshot
        .report
        .counters
        .push(("requested_symbols".to_string(), symbols.len() as u64));
    snapshot
        .report
        .counters
        .push(("requested_timeframes".to_string(), timeframes.len() as u64));
    snapshot
        .report
        .counters
        .push(("requested_years".to_string(), years as u64));

    let total_requests = (symbols.len() * timeframes.len()) as u64;
    let mut completed = 0_u64;
    let mut successes = 0_u64;
    let mut degraded = 0_u64;
    let mut failures = 0_u64;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
        .as_millis() as i64;

    for symbol in symbols {
        for timeframe in timeframes {
            let planned_chunks =
                plan_bootstrap_chunks(now_ms, timeframe, years).with_context(|| {
                    format!(
                        "failed to plan cTrader bootstrap chunks for {} {} over {} years",
                        symbol, timeframe, years
                    )
                })?;
            snapshot.progress.stage = "bootstrap_fetch".to_string();
            snapshot.progress.message = format!("Bootstrapping {symbol} {timeframe}");
            snapshot.report.events = push_recent_event(
                &snapshot.report.events,
                JobEventLevel::Info,
                format!(
                    "bootstrap started for {symbol} {timeframe} with {} planned chunks",
                    planned_chunks.len()
                ),
            );

            let request = CTraderChartHistoryRequest {
                client_id: context.client_id.clone(),
                client_secret: context.client_secret.clone(),
                access_token: context.access_token.clone(),
                environment: context.environment,
                account_id: context.account_id.clone(),
                symbol_name: symbol.clone(),
                timeframe: timeframe.clone(),
                from_timestamp_ms: now_ms,
                to_timestamp_ms: now_ms,
                count: None,
            };
            let outcome = bootstrap_from_ctrader_history(data_root, &request, now_ms, years);
            completed += 1;
            snapshot.progress.percent = Some(completed as f32 / total_requests as f32);

            match outcome {
                Ok(outcome) => {
                    let missing_segments = outcome.coverage.missing_segments.len();
                    if outcome.coverage.fully_covered {
                        successes += 1;
                    } else {
                        degraded += 1;
                        snapshot.report.warnings.push(format!(
                            "{} {} has {} uncovered segments after bootstrap",
                            symbol, timeframe, missing_segments
                        ));
                    }
                    snapshot.report.entries.push(format!(
                        "{} {} | planned_chunks={} | bars_written={} | covered_segments={} | missing_segments={}",
                        symbol,
                        timeframe,
                        planned_chunks.len(),
                        outcome.bars_written,
                        outcome.coverage.covered_segments.len(),
                        missing_segments
                    ));
                    snapshot.report.events = push_recent_event(
                        &snapshot.report.events,
                        if outcome.coverage.fully_covered {
                            JobEventLevel::Info
                        } else {
                            JobEventLevel::Warning
                        },
                        format!(
                            "bootstrap finished for {symbol} {timeframe} with {} covered segments",
                            outcome.coverage.covered_segments.len()
                        ),
                    );
                }
                Err(err) => {
                    failures += 1;
                    snapshot
                        .report
                        .errors
                        .push(format!("{symbol} {timeframe}: {err}"));
                    snapshot.report.events = push_recent_event(
                        &snapshot.report.events,
                        JobEventLevel::Error,
                        format!("bootstrap failed for {symbol} {timeframe}: {err}"),
                    );
                }
            }
        }
    }

    snapshot
        .report
        .counters
        .push(("completed_requests".to_string(), completed));
    snapshot
        .report
        .counters
        .push(("succeeded_requests".to_string(), successes));
    snapshot
        .report
        .counters
        .push(("degraded_requests".to_string(), degraded));
    snapshot
        .report
        .counters
        .push(("failed_requests".to_string(), failures));
    snapshot.report.highlights.push((
        "requests".to_string(),
        format!("{}/{} completed", completed, total_requests),
    ));
    snapshot.report.summary = format!(
        "Bootstrap finished: {} succeeded, {} degraded, {} failed",
        successes, degraded, failures
    );
    snapshot.report.log_path = Some("logs/forex-ai.log".to_string());
    snapshot.state = if failures == total_requests {
        JobState::Failed
    } else if failures > 0 || degraded > 0 {
        JobState::Degraded
    } else {
        JobState::Succeeded
    };
    snapshot.progress.stage = "bootstrap_complete".to_string();
    snapshot.progress.message = snapshot.report.summary.clone();
    Ok(snapshot)
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

fn sync_ctrader_discovered_accounts_into_targets(
    existing_targets: &[BrokerAccountTarget],
    discovered_accounts: &[CTraderDiscoveredAccount],
) -> Vec<BrokerAccountTarget> {
    discovered_accounts
        .iter()
        .map(|account| {
            if let Some(existing) = existing_targets
                .iter()
                .find(|target| target.account_id == account.account_id)
            {
                BrokerAccountTarget {
                    account_id: existing.account_id.clone(),
                    label: existing.label.clone(),
                    enabled_for_execution: existing.enabled_for_execution,
                }
            } else {
                BrokerAccountTarget {
                    account_id: account.account_id.clone(),
                    label: if !account.account_name.trim().is_empty() {
                        account.account_name.clone()
                    } else if !account.broker_title.trim().is_empty() {
                        account.broker_title.clone()
                    } else {
                        format!("cTrader Account {}", account.account_id)
                    },
                    enabled_for_execution: false,
                }
            }
        })
        .collect()
}

fn sync_discovered_accounts_with_targets(
    discovered_accounts: &[CTraderDiscoveredAccount],
    targets: &[BrokerAccountTarget],
) -> Vec<CTraderDiscoveredAccount> {
    discovered_accounts
        .iter()
        .map(|account| {
            let enabled = targets
                .iter()
                .find(|target| target.account_id == account.account_id)
                .map(|target| target.enabled_for_execution)
                .unwrap_or(false);
            let mut synced = account.clone();
            synced.enabled_for_execution = enabled;
            synced
        })
        .collect()
}

fn current_unix_seconds() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
        .as_secs() as i64)
}

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::CTrader, false) => TradingPanelMode::Disconnected,
        (DataSource::CTrader, true) => TradingPanelMode::Connected,
    }
}

const MAX_CHART_CANDLES: usize = 96;

fn supported_ctrader_chart_timeframes() -> Vec<String> {
    [
        "M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn chart_history_window_ms(timeframe: &str) -> Option<i64> {
    let minutes = match timeframe.trim().to_ascii_uppercase().as_str() {
        "M1" => 1,
        "M2" => 2,
        "M3" => 3,
        "M4" => 4,
        "M5" => 5,
        "M10" => 10,
        "M15" => 15,
        "M30" => 30,
        "H1" => 60,
        "H4" => 240,
        "H12" => 720,
        "D1" => 1_440,
        "W1" => 10_080,
        "MN1" => 43_200,
        _ => return None,
    };
    Some(minutes * 60_000 * (MAX_CHART_CANDLES as i64 + 24))
}

fn ohlcv_from_historical_bars(bars: &[HistoricalBar]) -> Ohlcv {
    Ohlcv {
        timestamp: Some(bars.iter().map(|bar| bar.timestamp_ms).collect()),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: Some(
            bars.iter()
                .map(|bar| bar.volume.unwrap_or_default() as f64)
                .collect(),
        ),
    }
}

pub fn build_market_chart_snapshot_from_historical_bars(
    symbol: &str,
    timeframe: &str,
    available_timeframes: Vec<String>,
    bars: &[HistoricalBar],
    overlays: Vec<ChartOverlay>,
    warnings: Vec<String>,
) -> MarketChartSnapshot {
    let ohlcv = ohlcv_from_historical_bars(bars);
    build_market_chart_snapshot_from_ohlcv(
        symbol,
        timeframe,
        available_timeframes,
        &ohlcv,
        overlays,
        warnings,
    )
}

pub fn build_market_chart_snapshot_from_ohlcv(
    symbol: &str,
    timeframe: &str,
    available_timeframes: Vec<String>,
    ohlcv: &Ohlcv,
    overlays: Vec<ChartOverlay>,
    warnings: Vec<String>,
) -> MarketChartSnapshot {
    let start = ohlcv.len().saturating_sub(MAX_CHART_CANDLES);
    let timestamps = ohlcv.timestamp.as_deref();
    let volumes = ohlcv.volume.as_deref();
    let candles: Vec<ChartCandle> = (start..ohlcv.len())
        .map(|idx| ChartCandle {
            timestamp: timestamps.and_then(|ts| ts.get(idx)).copied(),
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
            volume: volumes.and_then(|v| v.get(idx)).copied().unwrap_or(0.0),
        })
        .collect();

    let (price_min, price_max) = if candles.is_empty() {
        (0.0, 0.0)
    } else {
        candles
            .iter()
            .fold((f64::MAX, f64::MIN), |(min_v, max_v), candle| {
                (min_v.min(candle.low), max_v.max(candle.high))
            })
    };

    let latest_close = candles
        .last()
        .map(|candle| candle.close)
        .unwrap_or_default();
    let headline = if candles.is_empty() {
        format!("No candles loaded for {symbol} {timeframe}")
    } else {
        format!(
            "{} candles · latest close {:.5} · range {:.5}-{:.5}",
            candles.len(),
            latest_close,
            price_min,
            price_max
        )
    };

    let price_change_pct = if candles.len() >= 2 {
        let first_open = candles.first().map(|c| c.open).unwrap_or(0.0);
        let last_close = candles.last().map(|c| c.close).unwrap_or(0.0);
        if first_open > 0.0 {
            Some((last_close - first_open) / first_open * 100.0)
        } else {
            None
        }
    } else {
        None
    };

    MarketChartSnapshot {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        available_timeframes,
        candles,
        overlays,
        price_min,
        price_max,
        bid: None,
        ask: None,
        price_change_pct,
        headline,
        overlay_status: "Trade overlays will appear here once execution events are available."
            .to_string(),
        warnings,
    }
}

pub fn build_execution_surface_snapshot_with_runtime(
    state: &AppState,
    connection: &ConnectionSnapshot,
    runtime: Option<&AppExecutionRuntimeSnapshot>,
    mut runtime_warnings: Vec<String>,
) -> ExecutionSurfaceSnapshot {
    let action_reason = match connection.mode {
        TradingPanelMode::LocalOnly => {
            Some("Local mode disables live order submission.".to_string())
        }
        TradingPanelMode::Disconnected => {
            Some("Connect a broker adapter before sending live orders.".to_string())
        }
        TradingPanelMode::Connected => None,
    };
    let action_enabled = action_reason.is_none() && connection.supports_live_orders;
    let mut warnings: Vec<String> = action_reason
        .clone()
        .into_iter()
        .chain((!connection.connected && connection.requires_local_terminal).then(|| {
            "The configured adapter requires a local terminal runtime that is not currently connected.".to_string()
        }))
        .collect();
    warnings.append(&mut runtime_warnings);
    let mut diagnostics = vec![
        format!("Adapter: {}", connection.adapter_name),
        format!("Integration: {}", connection.integration_mode),
        format!(
            "Market data capability: {}",
            if connection.supports_market_data {
                "available"
            } else {
                "unavailable"
            }
        ),
        format!(
            "Live order capability: {}",
            if connection.supports_live_orders {
                "available when connected"
            } else {
                "unavailable"
            }
        ),
    ];
    if !connection.terminal_info.trim().is_empty() {
        diagnostics.push(format!("Terminal: {}", connection.terminal_info));
    }
    if connection.adapter_name == "cTrader" {
        diagnostics.push(format!(
            "Supported trade sides: {}",
            SUPPORTED_CTRADER_TRADE_SIDES
                .iter()
                .map(|side| side.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported order types: {}",
            SUPPORTED_CTRADER_ORDER_TYPES
                .iter()
                .map(|order_type| order_type.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported time-in-force: {}",
            SUPPORTED_CTRADER_TIME_IN_FORCE
                .iter()
                .map(|tif| tif.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        diagnostics.push(format!(
            "Supported trigger methods: {}",
            SUPPORTED_CTRADER_ORDER_TRIGGER_METHODS
                .iter()
                .map(|trigger| trigger.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let (
        positions,
        pending_orders,
        bot_timeline,
        history_rows,
        position_choices,
        pending_order_choices,
    ) = if let Some(runtime) = runtime {
        match runtime {
            AppExecutionRuntimeSnapshot::CTrader(runtime) => {
                diagnostics.push(format!("Trader balance: {:.2}", runtime.trader.balance));
                diagnostics.push(format!("Trader account id: {}", runtime.trader.account_id));
                if let Some(leverage) = runtime.trader.leverage {
                    diagnostics.push(format!("Leverage: {:.2}x", leverage));
                }
                if let Some(account_type) = &runtime.trader.account_type {
                    diagnostics.push(format!("Account type: {account_type}"));
                }
                if let Some(broker_name) = &runtime.trader.broker_name {
                    diagnostics.push(format!("Broker: {broker_name}"));
                }
                diagnostics.push(format!(
                    "Open positions: {}",
                    runtime.reconcile.positions.len()
                ));
                diagnostics.push(format!(
                    "Pending orders: {}",
                    runtime.reconcile.pending_orders.len()
                ));
                diagnostics.push(format!("Recent fills: {}", runtime.recent_deals.len()));
                append_ctrader_order_builder_diagnostics(&mut diagnostics, runtime);
                (
                    runtime
                        .reconcile
                        .positions
                        .iter()
                        .map(format_ctrader_position_line)
                        .collect(),
                    runtime
                        .reconcile
                        .pending_orders
                        .iter()
                        .map(format_ctrader_pending_order_line)
                        .collect(),
                    runtime
                        .recent_deals
                        .iter()
                        .map(format_ctrader_deal_line)
                        .collect(),
                    runtime
                        .recent_deals
                        .iter()
                        .map(format_ctrader_history_row)
                        .collect(),
                    runtime
                        .reconcile
                        .positions
                        .iter()
                        .map(|position| ExecutionSelectionOption {
                            id: position.position_id,
                            label: format_ctrader_position_line(position),
                        })
                        .collect(),
                    runtime
                        .reconcile
                        .pending_orders
                        .iter()
                        .map(|order| ExecutionSelectionOption {
                            id: order.order_id,
                            label: format_ctrader_pending_order_line(order),
                        })
                        .collect(),
                )
            }
        }
    } else {
        diagnostics.push("Live execution runtime info is currently being managed via the central broker background loop.".to_string());
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };

    ExecutionSurfaceSnapshot {
        symbol: state.selected_pair.clone(),
        adapter_name: connection.adapter_name.clone(),
        integration_mode: connection.integration_mode.clone(),
        connection_status: connection.status_text.clone(),
        supported_adapters: SUPPORTED_TRADING_ADAPTERS
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        primary_actions: vec![
            ExecutionAction {
                label: "Buy".to_string(),
                enabled: action_enabled,
                reason: action_reason.clone(),
            },
            ExecutionAction {
                label: "Sell".to_string(),
                enabled: action_enabled,
                reason: action_reason,
            },
        ],
        warnings,
        diagnostics,
        positions,
        pending_orders,
        bot_timeline,
        history_rows,
        journal_rows: Vec::new(),
        selected_position_id: state.order_ticket.selected_position_id,
        selected_order_id: state.order_ticket.selected_order_id,
        position_choices,
        pending_order_choices,
        ticket: ExecutionTicketSnapshot {
            lot_size: state.order_ticket.lot_size,
            slippage_in_points: state.order_ticket.slippage_in_points,
            comment: state.order_ticket.comment.clone(),
            label: state.order_ticket.label.clone(),
            max_lot_size: state.risk.max_lot_size,
        },
    }
}

fn preferred_chart_timeframe(available_timeframes: &[String], requested: &str) -> String {
    if available_timeframes.iter().any(|tf| tf == requested) {
        return requested.to_string();
    }

    for preferred in ["M1", "M5", "M15", "H1"] {
        if available_timeframes.iter().any(|tf| tf == preferred) {
            return preferred.to_string();
        }
    }

    available_timeframes
        .first()
        .cloned()
        .unwrap_or_else(|| requested.to_string())
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

fn format_ctrader_terminal_info(
    trader: &crate::app_services::ctrader_account::CTraderTraderSnapshot,
    environment: CTraderEnvironment,
) -> String {
    let broker = trader.broker_name.as_deref().unwrap_or("cTrader Open API");
    format!(
        "{} · {} · account {} · balance {:.2}",
        broker,
        match environment {
            CTraderEnvironment::Live => "Live",
            CTraderEnvironment::Demo => "Demo",
        },
        trader.account_id,
        trader.balance
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn format_ctrader_connect_error(err: &anyhow::Error) -> String {
    let message = err.to_string();
    if message.contains("restored token session") || message.contains("stored token bundle") {
        return "cTrader login required · restore or start auth first".to_string();
    }
    if message.contains("at least one discovered account") {
        return "cTrader account discovery required before connecting".to_string();
    }
    format!("cTrader connect failed: {message}")
}

fn format_ctrader_position_line(position: &CTraderPositionSnapshot) -> String {
    let mut line = format!(
        "#{} · symbol {} {} {:.2}",
        position.position_id, position.symbol_id, position.trade_side, position.volume
    );
    if let Some(price) = position.price {
        line.push_str(&format!(" · open {:.5}", price));
    }
    if let Some(stop_loss) = position.stop_loss {
        line.push_str(&format!(" · sl {:.5}", stop_loss));
    }
    if let Some(take_profit) = position.take_profit {
        line.push_str(&format!(" · tp {:.5}", take_profit));
    }
    if let Some(swap) = position.swap {
        line.push_str(&format!(" · swap {:+.2}", swap));
    }
    if let Some(commission) = position.commission {
        line.push_str(&format!(" · fee {:+.2}", commission));
    }
    line.push_str(" · unrealized pnl unavailable");
    line
}

fn format_ctrader_pending_order_line(order: &CTraderPendingOrderSnapshot) -> String {
    let mut line = format!(
        "#{} · symbol {} {} {} {:.2}",
        order.order_id, order.symbol_id, order.trade_side, order.order_type, order.volume
    );
    if let Some(limit_price) = order.limit_price {
        line.push_str(&format!(" @ {:.5}", limit_price));
    } else if let Some(stop_price) = order.stop_price {
        line.push_str(&format!(" @ {:.5}", stop_price));
    }
    if let Some(stop_loss) = order.stop_loss {
        line.push_str(&format!(" · sl {:.5}", stop_loss));
    }
    if let Some(take_profit) = order.take_profit {
        line.push_str(&format!(" · tp {:.5}", take_profit));
    }
    line
}

fn format_ctrader_deal_line(deal: &CTraderDealSnapshot) -> String {
    let mut line = format!(
        "#{} · {} {} {:.2}",
        deal.deal_id, deal.deal_status, deal.trade_side, deal.filled_volume
    );
    if let Some(execution_price) = deal.execution_price {
        line.push_str(&format!(" @ {:.5}", execution_price));
    }
    if let Some(gross_profit) = deal.gross_profit {
        line.push_str(&format!(" · pnl {:+.2}", gross_profit));
    }
    if let Some(fee) = deal.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    }
    if let Some(net_profit) = deal.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    line
}

fn format_ctrader_history_row(deal: &CTraderDealSnapshot) -> String {
    let mut line = format!(
        "{} · deal #{} · pos #{} · symbol {} {} {:.2}",
        format_timestamp_ms(deal.execution_timestamp_ms),
        deal.deal_id,
        deal.position_id,
        deal.symbol_id,
        deal.trade_side,
        deal.filled_volume
    );
    if let Some(entry_price) = deal.entry_price {
        line.push_str(&format!(" · entry {:.5}", entry_price));
    }
    if let Some(execution_price) = deal.execution_price {
        line.push_str(&format!(" · exit {:.5}", execution_price));
    }
    if let Some(gross_profit) = deal.gross_profit {
        line.push_str(&format!(" · gross {:+.2}", gross_profit));
    } else {
        line.push_str(" · gross n/a");
    }
    if let Some(fee) = deal.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    } else {
        line.push_str(" · fee n/a");
    }
    if let Some(net_profit) = deal.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    } else {
        line.push_str(" · net n/a");
    }
    line
}

fn append_ctrader_order_builder_diagnostics(
    diagnostics: &mut Vec<String>,
    runtime: &CTraderAccountRuntimeSnapshot,
) {
    let account_id = runtime.trader.account_id;
    let symbol_id = runtime
        .reconcile
        .positions
        .first()
        .map(|position| position.symbol_id)
        .or_else(|| {
            runtime
                .reconcile
                .pending_orders
                .first()
                .map(|order| order.symbol_id)
        });

    if let Some(symbol_id) = symbol_id {
        let seed_volume = runtime
            .reconcile
            .positions
            .first()
            .map(|position| position.volume)
            .or_else(|| {
                runtime
                    .reconcile
                    .pending_orders
                    .first()
                    .map(|order| order.volume)
            })
            .unwrap_or(1.0);
        let request = build_new_order_request(
            &CTraderNewOrderRequest {
                account_id,
                symbol_id,
                order_type: CTraderOrderType::Market,
                trade_side: CTraderTradeSide::Buy,
                volume: units_to_ctrader_protocol_volume(seed_volume),
                limit_price: None,
                stop_price: None,
                time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
                expiration_timestamp_ms: None,
                stop_loss: None,
                take_profit: None,
                comment: Some("preview".to_string()),
                base_slippage_price: None,
                slippage_in_points: Some(10),
                label: Some("preview".to_string()),
                position_id: None,
                client_order_id: Some("preview-new".to_string()),
                relative_stop_loss: None,
                relative_take_profit: None,
                guaranteed_stop_loss: Some(false),
                trailing_stop_loss: Some(false),
                stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
            },
            "preview-new-order",
        );
        diagnostics.push(format!(
            "New order builder ready: payload {}",
            request.payload_type
        ));
    }

    if let Some(order) = runtime.reconcile.pending_orders.first() {
        let cancel_request = build_cancel_order_request(
            &CTraderCancelOrderRequest {
                account_id,
                order_id: order.order_id,
            },
            "preview-cancel-order",
        );
        let amend_request = build_amend_order_request(
            &CTraderAmendOrderRequest {
                account_id,
                order_id: order.order_id,
                volume: Some(units_to_ctrader_protocol_volume(order.volume)),
                limit_price: order.limit_price,
                stop_price: order.stop_price,
                expiration_timestamp_ms: None,
                stop_loss: order.stop_loss,
                take_profit: order.take_profit,
                slippage_in_points: Some(10),
                relative_stop_loss: None,
                relative_take_profit: None,
                guaranteed_stop_loss: Some(false),
                trailing_stop_loss: Some(false),
                stop_trigger_method: Some(CTraderOrderTriggerMethod::Trade),
            },
            "preview-amend-order",
        );
        diagnostics.push(format!(
            "Pending-order builders ready: cancel payload {} · amend payload {}",
            cancel_request.payload_type, amend_request.payload_type
        ));
    }

    if let Some(position) = runtime.reconcile.positions.first() {
        let close_request = build_close_position_request(
            &CTraderClosePositionRequest {
                account_id,
                position_id: position.position_id,
                volume: units_to_ctrader_protocol_volume(position.volume),
            },
            "preview-close-position",
        );
        diagnostics.push(format!(
            "Close-position builder ready: payload {}",
            close_request.payload_type
        ));
    }
}

fn units_to_ctrader_protocol_volume(volume: f64) -> i64 {
    (volume * 100.0).round() as i64
}

fn ctrader_protocol_volume_from_units(volume: f64) -> i64 {
    units_to_ctrader_protocol_volume(volume)
}

fn ctrader_protocol_volume_from_lots(lots: f64, symbol: &CTraderSymbolInfo) -> anyhow::Result<i64> {
    let lot_size = symbol
        .lot_size
        .ok_or_else(|| anyhow::anyhow!("cTrader symbol metadata is missing lotSize"))?;
    Ok((lots * lot_size as f64).round() as i64)
}

fn validate_and_convert_lot_size_to_ctrader_volume(
    ticket: &OrderTicketState,
    max_lot_size: f64,
    symbol: &CTraderSymbolInfo,
) -> anyhow::Result<i64> {
    if ticket.lot_size <= 0.0 {
        return Err(anyhow::anyhow!("lot size must be greater than zero"));
    }
    if ticket.lot_size > max_lot_size {
        return Err(anyhow::anyhow!(
            "lot size {:.2} exceeds app risk limit {:.2}",
            ticket.lot_size,
            max_lot_size
        ));
    }
    let protocol_volume = ctrader_protocol_volume_from_lots(ticket.lot_size, symbol)?;
    if let Some(min_volume) = symbol.min_volume
        && protocol_volume < min_volume
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} is below broker minimum {:.2}",
            ticket.lot_size,
            min_volume as f64 / symbol.lot_size.unwrap_or(1) as f64
        ));
    }
    if let Some(max_volume) = symbol.max_volume
        && protocol_volume > max_volume
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} exceeds broker maximum {:.2}",
            ticket.lot_size,
            max_volume as f64 / symbol.lot_size.unwrap_or(1) as f64
        ));
    }
    if let Some(step_volume) = symbol.step_volume
        && step_volume > 0
        && protocol_volume % step_volume != 0
    {
        return Err(anyhow::anyhow!(
            "lot size {:.2} does not align with broker step volume",
            ticket.lot_size
        ));
    }
    Ok(protocol_volume)
}

fn format_execution_journal_line(action: &str, outcome: &CTraderExecutionOutcome) -> String {
    let timestamp = outcome
        .timestamp_ms
        .map(format_timestamp_ms)
        .unwrap_or_else(|| "event-time-unavailable".to_string());
    let mut line = format!(
        "{} · {} · status {}",
        timestamp,
        action,
        match outcome.status {
            CTraderExecutionStatus::Accepted => "ACCEPTED",
            CTraderExecutionStatus::Filled => "FILLED",
            CTraderExecutionStatus::Replaced => "REPLACED",
            CTraderExecutionStatus::Cancelled => "CANCELLED",
            CTraderExecutionStatus::PartialFill => "PARTIAL_FILL",
            CTraderExecutionStatus::Failed => "FAILED",
        }
    );
    if let Some(symbol_id) = outcome.symbol_id {
        line.push_str(&format!(" · symbol {}", symbol_id));
    }
    if let Some(trade_side) = &outcome.trade_side {
        line.push_str(&format!(" · side {}", trade_side));
    }
    if let Some(lot_size) = outcome.lot_size {
        line.push_str(&format!(" · size {:.2}", lot_size));
    }
    if let Some(order_id) = outcome.order_id {
        line.push_str(&format!(" · order {}", order_id));
    }
    if let Some(position_id) = outcome.position_id {
        line.push_str(&format!(" · position {}", position_id));
    }
    if let Some(execution_price) = outcome.execution_price {
        line.push_str(&format!(" · price {:.5}", execution_price));
    }
    if let Some(gross_profit) = outcome.gross_profit {
        line.push_str(&format!(" · gross {:+.2}", gross_profit));
    }
    if let Some(fee) = outcome.fee {
        line.push_str(&format!(" · fee {:+.2}", fee));
    }
    if let Some(net_profit) = outcome.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    if let Some(error_code) = &outcome.error_code {
        line.push_str(&format!(" · error {}", error_code));
    }
    if let Some(description) = &outcome.description {
        line.push_str(&format!(" · {}", description));
    }
    line
}

fn format_execution_outcome_status(prefix: &str, outcome: &CTraderExecutionOutcome) -> String {
    let mut line = format!(
        "{} {}",
        prefix,
        match outcome.status {
            CTraderExecutionStatus::Accepted => "accepted",
            CTraderExecutionStatus::Filled => "filled",
            CTraderExecutionStatus::Replaced => "replaced",
            CTraderExecutionStatus::Cancelled => "cancelled",
            CTraderExecutionStatus::PartialFill => "partially filled",
            CTraderExecutionStatus::Failed => "failed",
        }
    );
    if let Some(net_profit) = outcome.net_profit {
        line.push_str(&format!(" · net {:+.2}", net_profit));
    }
    if let Some(error_code) = &outcome.error_code {
        line.push_str(&format!(" · error {}", error_code));
    }
    line
}

fn non_empty_option(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn prop_firm_pre_trade_check(
    risk: &forex_core::config::RiskConfig,
    order: &CTraderNewOrderRequest,
    account_equity: f64,
    initial_equity: f64,
    day_start_equity: f64,
    pip_position: i32,
) -> anyhow::Result<()> {
    if risk.require_stop_loss && order.stop_loss.is_none() {
        return Err(anyhow::anyhow!(
            "Mandatory stop-loss rule violated: order missing stop_loss"
        ));
    }

    if day_start_equity > 0.0 {
        let daily_drawdown_ratio =
            ((day_start_equity - account_equity) / day_start_equity).max(0.0);
        if daily_drawdown_ratio >= risk.daily_drawdown_limit {
            return Err(anyhow::anyhow!(
                "Daily drawdown limit reached: current {:.2}% >= max {:.2}% (measured via Equity)",
                daily_drawdown_ratio * 100.0,
                risk.daily_drawdown_limit * 100.0
            ));
        }
    }

    if initial_equity > 0.0 {
        let total_drawdown_ratio = ((initial_equity - account_equity) / initial_equity).max(0.0);
        if total_drawdown_ratio >= risk.total_drawdown_limit {
            return Err(anyhow::anyhow!(
                "Total drawdown limit reached: current {:.2}% >= max {:.2}% (measured via Equity)",
                total_drawdown_ratio * 100.0,
                risk.total_drawdown_limit * 100.0
            ));
        }
    }

    // HARD risk-per-trade gate (D4+D5). Previously this was a `tracing::warn`
    // that did not block the order, and the loss estimate used `* 10.0` —
    // i.e. it assumed every pip is worth $10/std-lot. Wrong for non-USD
    // accounts and for any quote currency != account currency. We now
    // compute the real per-pip account-currency value via the forex-search
    // cost model and reject the order if it would exceed the configured
    // `risk_per_trade` percentage. Override the live FX rate the model
    // needs for cross pairs via `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`.
    if let (Some(sl), Some(entry_estimate)) =
        (order.stop_loss, order.limit_price.or(order.stop_price))
    {
        let pip_multiplier = 10.0_f64.powi(pip_position);
        let pip_distance = (entry_estimate - sl).abs() * pip_multiplier;

        // The order references the cTrader symbol by numeric id; without a
        // local id→string cache we let `infer_market_cost_profile` fall back
        // to the `FOREX_BOT_PROP_SYMBOL` env override (default EURUSD) and
        // the corresponding pip-value heuristics. Operators can also bypass
        // the heuristic by setting `FOREX_BOT_PROP_PIP_VALUE_PER_LOT`.
        let cost = forex_search::genetic::strategy_gene::infer_market_cost_profile(
            "",
            "",
            Some(entry_estimate),
            None,
            None,
        );
        // `pip_value_per_lot` is account-currency-units per pip per standard
        // (1.0) lot. cTrader volume is in cents of a standard lot, so divide.
        let estimated_loss =
            pip_distance * (order.volume as f64 / 100.0) * cost.pip_value_per_lot;
        let max_loss = risk.risk_per_trade * account_equity;
        if estimated_loss > max_loss {
            return Err(anyhow::anyhow!(
                "Risk-per-trade exceeded: estimated loss {:.2} > max allowed {:.2} ({:.2}%) at {:.1} pips",
                estimated_loss,
                max_loss,
                risk.risk_per_trade * 100.0,
                pip_distance
            ));
        }
    }

    Ok(())
}

fn format_timestamp_ms(timestamp_ms: i64) -> String {
    timestamp_ms.to_string()
}

fn record_app_event(operation: &str, status: &str, message: impl Into<String>) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::App,
        app_record(operation, status, message),
    ) {
        error!("Failed to write APP section log: {}", err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::AppRuntimeConfig;
    use forex_core::config::RiskConfig;
    use forex_data::Ohlcv;
    use std::path::PathBuf;

    fn sample_state(source: DataSource, status_msg: &str) -> AppState {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: matches!(source, DataSource::Local),
            auto_discovery: false,
            auto_training: false,
        };
        let mut state = AppState::new(
            runtime,
            &forex_core::Settings::default(),
            vec!["EURUSD".to_string()],
        );
        state.data_source = source;
        state.status_msg = status_msg.to_string();
        state
    }

    fn fresh_ctrader_token_bundle(
        access_token: &str,
        refresh_token: &str,
    ) -> crate::app_services::ctrader_auth::CTraderTokenBundle {
        crate::app_services::ctrader_auth::CTraderTokenBundle {
            access_token: access_token.to_string(),
            refresh_token: refresh_token.to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: current_unix_seconds().expect("current unix time"),
        }
    }

    #[test]
    fn connection_snapshot_reports_local_mode_without_live_runtime() {
        let state = sample_state(DataSource::Local, "Local Mode");
        let session = TradingSession::new();

        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.mode, TradingPanelMode::LocalOnly);
        assert!(!snapshot.connected);
        assert_eq!(snapshot.status_text, "Local Mode");
        assert_eq!(snapshot.terminal_info, "");
    }

    #[test]
    fn panel_mode_uses_data_source_and_connection_state() {
        assert_eq!(
            panel_mode(DataSource::Local, false),
            TradingPanelMode::LocalOnly
        );
        assert_eq!(
            panel_mode(DataSource::CTrader, false),
            TradingPanelMode::Disconnected
        );
        assert_eq!(
            panel_mode(DataSource::CTrader, true),
            TradingPanelMode::Connected
        );
    }

    #[test]
    fn connection_snapshot_reports_remote_api_metadata_for_stubbed_ctrader() {
        let state = sample_state(DataSource::CTrader, "Offline");
        let session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.adapter_name, "cTrader");
        assert_eq!(snapshot.integration_mode, "Remote Open API");
        assert!(!snapshot.requires_local_terminal);
        assert!(snapshot.supports_market_data);
        assert!(snapshot.supports_live_orders);
        assert_eq!(snapshot.mode, TradingPanelMode::Disconnected);
    }

    #[test]
    fn connection_snapshot_reports_remote_api_metadata_for_stubbed_dxtrade() {
        let state = sample_state(DataSource::CTrader, "Offline");
        let session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::DxTrade);

        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.adapter_name, "DXtrade");
        assert_eq!(snapshot.integration_mode, "Remote broker API");
        assert!(!snapshot.requires_local_terminal);
        assert!(snapshot.supports_market_data);
        assert!(snapshot.supports_live_orders);
        assert_eq!(snapshot.mode, TradingPanelMode::Disconnected);
    }

    #[test]
    fn market_chart_snapshot_uses_recent_real_candles_and_preserves_price_bounds() {
        let ohlcv = Ohlcv {
            timestamp: Some((1_i64..=140).collect()),
            open: (0..140).map(|idx| 1.1000 + idx as f64 * 0.0001).collect(),
            high: (0..140).map(|idx| 1.1010 + idx as f64 * 0.0001).collect(),
            low: (0..140).map(|idx| 1.0990 + idx as f64 * 0.0001).collect(),
            close: (0..140).map(|idx| 1.1005 + idx as f64 * 0.0001).collect(),
            volume: None,
        };

        let snapshot = build_market_chart_snapshot_from_ohlcv(
            "EURUSD",
            "M5",
            vec!["M1".to_string(), "M5".to_string(), "H1".to_string()],
            &ohlcv,
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(snapshot.symbol, "EURUSD");
        assert_eq!(snapshot.timeframe, "M5");
        assert_eq!(snapshot.available_timeframes, vec!["M1", "M5", "H1"]);
        assert_eq!(snapshot.candles.len(), 96);
        assert_eq!(snapshot.candles.first().and_then(|c| c.timestamp), Some(45));
        assert_eq!(snapshot.candles.last().and_then(|c| c.timestamp), Some(140));
        assert!(snapshot.price_min < snapshot.price_max);
        assert!(snapshot.headline.contains("96 candles"));
    }

    #[test]
    fn build_market_chart_snapshot_from_historical_bars_preserves_recent_candles() {
        let bars: Vec<HistoricalBar> = (0..140)
            .map(|idx| HistoricalBar {
                timestamp_ms: 1_700_000_000_000 + idx as i64 * 60_000,
                open: 1.1000 + idx as f64 * 0.0001,
                high: 1.1010 + idx as f64 * 0.0001,
                low: 1.0990 + idx as f64 * 0.0001,
                close: 1.1005 + idx as f64 * 0.0001,
                volume: Some(10 + idx as i64),
            })
            .collect();

        let snapshot = build_market_chart_snapshot_from_historical_bars(
            "EURUSD",
            "M5",
            vec!["M1".to_string(), "M5".to_string()],
            &bars,
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(snapshot.candles.len(), 96);
        assert_eq!(
            snapshot.candles.first().and_then(|candle| candle.timestamp),
            Some(1_700_002_640_000)
        );
        assert_eq!(snapshot.available_timeframes, vec!["M1", "M5"]);
    }

    #[test]
    fn ctrader_chart_history_request_uses_enabled_target_and_selected_environment() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:8877/callback".to_string();
        session.broker_settings_mut().ctrader.environment = CTraderBrokerEnvironment::Demo;
        session.broker_settings_mut().ctrader.accounts = vec![
            BrokerAccountTarget {
                account_id: "1001".to_string(),
                label: "standby".to_string(),
                enabled_for_execution: false,
            },
            BrokerAccountTarget {
                account_id: "2002".to_string(),
                label: "primary".to_string(),
                enabled_for_execution: true,
            },
        ];
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed token bundle");

        let request = session
            .build_ctrader_chart_history_request("EURUSD", "M15")
            .expect("request should build");

        assert_eq!(request.environment, CTraderEnvironment::Demo);
        assert_eq!(request.account_id, "2002");
        assert_eq!(request.symbol_name, "EURUSD");
        assert_eq!(request.timeframe, "M15");
        assert_eq!(request.count, Some((MAX_CHART_CANDLES + 24) as u32));
        assert!(request.to_timestamp_ms >= request.from_timestamp_ms);
    }

    #[test]
    fn market_chart_snapshot_reports_ctrader_requirements_instead_of_fake_fallback() {
        let state = sample_state(DataSource::CTrader, "Offline");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();

        let snapshot = session.market_chart_snapshot(&state);

        assert!(snapshot.candles.is_empty());
        assert_eq!(snapshot.timeframe, "M1");
        assert_eq!(
            snapshot.available_timeframes.first().map(String::as_str),
            Some("M1")
        );
        assert!(
            snapshot
                .warnings
                .iter()
                .any(|warning| warning.contains("stored token bundle"))
        );
        assert!(snapshot.headline.contains("No cTrader market data loaded"));
    }

    #[test]
    fn execution_surface_snapshot_disables_live_actions_and_surfaces_unwired_gaps() {
        let state = sample_state(DataSource::Local, "Local Mode");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = session.execution_surface_snapshot(&state);

        assert_eq!(snapshot.symbol, "EURUSD");
        assert_eq!(snapshot.adapter_name, "cTrader");
        assert_eq!(snapshot.primary_actions.len(), 2);
        assert!(
            snapshot
                .primary_actions
                .iter()
                .all(|action| !action.enabled)
        );
        assert!(
            snapshot
                .warnings
                .iter()
                .any(|warning| warning.contains("Local mode"))
        );
        assert!(
            snapshot
                .diagnostics
                .iter()
                .any(|line| line.contains("central broker background loop"))
        );
    }

    #[test]
    fn execution_surface_snapshot_surfaces_adapter_specific_unwired_feed_reason() {
        let state = sample_state(DataSource::CTrader, "Offline");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = session.execution_surface_snapshot(&state);

        assert!(snapshot.warnings.iter().any(|warning| warning.contains(
            "cTrader execution feed is unavailable until the remote account session connects"
        )));
    }

    #[test]
    fn selecting_adapter_updates_configured_runtime_and_status_message() {
        let mut state = sample_state(DataSource::CTrader, "Offline");
        let mut session = TradingSession::new();

        session.select_adapter(&mut state, TradingAdapterKind::CTrader);
        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.adapter_name, "cTrader");
        assert_eq!(snapshot.integration_mode, "Remote Open API");
        assert!(!session.is_connected());
        assert_eq!(state.status_msg, "cTrader selected · disconnected");
    }

    #[test]
    fn connect_sets_missing_credentials_status_for_unready_remote_adapter() {
        let mut state = sample_state(DataSource::CTrader, "Offline");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        session.connect(&mut state);

        assert_eq!(
            state.status_msg,
            "cTrader configuration incomplete: missing client_id, client_secret, redirect_uri"
        );
        assert!(!session.is_connected());
    }

    #[test]
    fn connect_requires_restored_ctrader_session_before_runtime_probe() {
        let mut state = sample_state(DataSource::CTrader, "Offline");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();

        session.connect(&mut state);

        assert_eq!(
            state.status_msg,
            "cTrader login required · restore or start auth first"
        );
        assert!(!session.is_connected());
    }

    #[test]
    fn connect_uses_ctrader_account_runtime_probe_when_session_is_restored() {
        let mut state = sample_state(DataSource::CTrader, "Offline");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
            account_id: "712345".to_string(),
            label: "primary".to_string(),
            enabled_for_execution: true,
        }];
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed token bundle");
        session.set_ctrader_account_runtime_backend_for_test(
            crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
                crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                    trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                        account_id: 712345,
                        balance: 1000.0,
                        leverage: Some(50.0),
                        trader_login: Some(998877),
                        account_type: Some("NETTED".to_string()),
                        broker_name: Some("Demo Broker".to_string()),
                        money_digits: 2,
                        unrealized_pnl: 0.0,
                    },
                    reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                        account_id: 712345,
                        positions: Vec::new(),
                        pending_orders: Vec::new(),
                    },
                    recent_deals: Vec::new(),
                },
            ),
        );

        session.connect(&mut state);

        assert!(session.is_connected());
        assert_eq!(state.status_msg, "cTrader connected");
    }

    #[test]
    fn ctrader_live_spot_cache_reuses_backend_update_within_ttl() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let backend =
            crate::app_services::ctrader_streaming::StubCTraderLiveStreamingBackend::success(
                crate::app_services::ctrader_streaming::CTraderLiveChartUpdate {
                    symbol_id: 14,
                    bid: Some(1.09995),
                    ask: Some(1.10015),
                    timestamp_ms: Some(1_710_000_200_000),
                    latest_trendbar: None,
                },
            );
        session.set_ctrader_live_streaming_backend_for_test(backend.clone());

        let request = crate::app_services::ctrader_streaming::CTraderLiveChartUpdateRequest {
            client_id: "client".to_string(),
            client_secret: "secret".to_string(),
            access_token: "access".to_string(),
            environment: crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo,
            account_id: "712345".to_string(),
            symbol_id: 14,
            digits: 5,
            timeframe: "M1".to_string(),
            subscribe_to_spot_timestamp: true,
        };

        let first = session
            .load_ctrader_live_chart_update_cached(&request)
            .expect("first live update");
        let second = session
            .load_ctrader_live_chart_update_cached(&request)
            .expect("second live update");

        assert_eq!(first, second);
        assert_eq!(backend.call_count(), 1);
    }

    #[test]
    fn execution_surface_snapshot_uses_ctrader_reconcile_runtime_when_connected() {
        let mut state = sample_state(DataSource::CTrader, "Connected");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
            account_id: "712345".to_string(),
            label: "primary".to_string(),
            enabled_for_execution: true,
        }];
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed token bundle");
        session.set_ctrader_account_runtime_backend_for_test(
            crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
                crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                    trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                        account_id: 712345,
                        balance: 1000.0,
                        leverage: Some(50.0),
                        trader_login: Some(998877),
                        account_type: Some("NETTED".to_string()),
                        broker_name: Some("Demo Broker".to_string()),
                        money_digits: 2,
                        unrealized_pnl: 0.0,
                    },
                    reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                        account_id: 712345,
                        positions: vec![
                            crate::app_services::ctrader_account::CTraderPositionSnapshot {
                                position_id: 9001,
                                symbol_id: 14,
                                trade_side: "BUY".to_string(),
                                volume: 25.0,
                                open_timestamp_ms: Some(1710000000000),
                                price: Some(1.10123),
                                stop_loss: Some(1.095),
                                take_profit: Some(1.11),
                                swap: None,
                                commission: None,
                                label: Some("trend".to_string()),
                                comment: Some("bot".to_string()),
                            },
                        ],
                        pending_orders: vec![
                            crate::app_services::ctrader_account::CTraderPendingOrderSnapshot {
                                order_id: 8001,
                                symbol_id: 14,
                                trade_side: "SELL".to_string(),
                                order_type: "LIMIT".to_string(),
                                volume: 15.0,
                                open_timestamp_ms: Some(1710000100000),
                                limit_price: Some(1.099),
                                stop_price: None,
                                stop_loss: Some(1.105),
                                take_profit: Some(1.09),
                                label: Some("breakout".to_string()),
                                comment: Some("pending".to_string()),
                            },
                        ],
                    },
                    recent_deals: vec![crate::app_services::ctrader_account::CTraderDealSnapshot {
                        deal_id: 3001,
                        order_id: 8001,
                        position_id: 9001,
                        symbol_id: 14,
                        trade_side: "BUY".to_string(),
                        deal_status: "FILLED".to_string(),
                        volume: 15.0,
                        filled_volume: 15.0,
                        execution_timestamp_ms: 1710000201000,
                        execution_price: Some(1.0990),
                        entry_price: Some(1.0980),
                        gross_profit: Some(12.5),
                        fee: Some(-0.4),
                        swap: Some(0.0),
                        pnl_conversion_fee: Some(0.0),
                        net_profit: Some(12.1),
                    }],
                },
            ),
        );

        session.connect(&mut state);
        let snapshot = session.execution_surface_snapshot(&state);

        assert!(snapshot.positions.iter().any(|line| line.contains("#9001")));
        assert!(
            snapshot
                .pending_orders
                .iter()
                .any(|line| line.contains("#8001"))
        );
        assert!(
            snapshot
                .bot_timeline
                .iter()
                .any(|line| line.contains("#3001"))
        );
        assert!(
            snapshot
                .diagnostics
                .iter()
                .any(|line| line.contains("Recent fills: 1"))
        );
        assert!(
            snapshot
                .diagnostics
                .iter()
                .any(|line| line.contains("Trader balance"))
        );
        assert!(snapshot.warnings.is_empty());
    }

    #[test]
    fn cancel_selected_order_records_ctrader_journal_and_updates_status() {
        let mut state = sample_state(DataSource::CTrader, "Connected");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
            account_id: "712345".to_string(),
            label: "primary".to_string(),
            enabled_for_execution: true,
        }];
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed token bundle");
        session.set_ctrader_account_runtime_backend_for_test(
            crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
                crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                    trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                        account_id: 712345,
                        balance: 1000.0,
                        leverage: Some(50.0),
                        trader_login: Some(998877),
                        account_type: Some("NETTED".to_string()),
                        broker_name: Some("Demo Broker".to_string()),
                        money_digits: 2,
                        unrealized_pnl: 0.0,
                    },
                    reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                        account_id: 712345,
                        positions: vec![
                            crate::app_services::ctrader_account::CTraderPositionSnapshot {
                                position_id: 9001,
                                symbol_id: 14,
                                trade_side: "BUY".to_string(),
                                volume: 25.0,
                                open_timestamp_ms: Some(1710000000000),
                                price: Some(1.10123),
                                stop_loss: Some(1.095),
                                take_profit: Some(1.11),
                                swap: None,
                                commission: None,
                                label: Some("trend".to_string()),
                                comment: Some("bot".to_string()),
                            },
                        ],
                        pending_orders: vec![
                            crate::app_services::ctrader_account::CTraderPendingOrderSnapshot {
                                order_id: 8001,
                                symbol_id: 14,
                                trade_side: "SELL".to_string(),
                                order_type: "LIMIT".to_string(),
                                volume: 15.0,
                                open_timestamp_ms: Some(1710000100000),
                                limit_price: Some(1.099),
                                stop_price: None,
                                stop_loss: Some(1.105),
                                take_profit: Some(1.09),
                                label: Some("breakout".to_string()),
                                comment: Some("pending".to_string()),
                            },
                        ],
                    },
                    recent_deals: Vec::new(),
                },
            ),
        );
        session.connect(&mut state);
        state.order_ticket.selected_order_id = Some(8001);
        session.set_ctrader_execution_backend_for_test(
            crate::app_services::ctrader_execution::StubCTraderExecutionBackend::succeed(
                crate::app_services::ctrader_execution::CTraderExecutionOutcome {
                    status:
                        crate::app_services::ctrader_execution::CTraderExecutionStatus::Cancelled,
                    account_id: 712345,
                    symbol_id: Some(14),
                    order_id: Some(8001),
                    position_id: None,
                    deal_id: None,
                    trade_side: Some("SELL".to_string()),
                    order_type: Some("LIMIT".to_string()),
                    lot_size: Some(15.0),
                    execution_price: None,
                    gross_profit: None,
                    fee: None,
                    swap: None,
                    net_profit: None,
                    timestamp_ms: Some(1710000300000),
                    error_code: None,
                    description: None,
                },
            ),
        );

        session.cancel_selected_order(&mut state);
        let snapshot = session.execution_surface_snapshot(&state);

        assert!(state.status_msg.contains("Cancelled order"));
        assert!(
            snapshot
                .journal_rows
                .iter()
                .any(|line| line.contains("Cancel order #8001"))
        );
    }

    #[test]
    fn close_selected_position_surfaces_ctrader_execution_failure() {
        let mut state = sample_state(DataSource::CTrader, "Connected");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
            account_id: "712345".to_string(),
            label: "primary".to_string(),
            enabled_for_execution: true,
        }];
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed token bundle");
        session.set_ctrader_account_runtime_backend_for_test(
            crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
                crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                    trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                        account_id: 712345,
                        balance: 1000.0,
                        leverage: Some(50.0),
                        trader_login: Some(998877),
                        account_type: Some("NETTED".to_string()),
                        broker_name: Some("Demo Broker".to_string()),
                        money_digits: 2,
                        unrealized_pnl: 0.0,
                    },
                    reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                        account_id: 712345,
                        positions: vec![
                            crate::app_services::ctrader_account::CTraderPositionSnapshot {
                                position_id: 9001,
                                symbol_id: 14,
                                trade_side: "BUY".to_string(),
                                volume: 25.0,
                                open_timestamp_ms: Some(1710000000000),
                                price: Some(1.10123),
                                stop_loss: Some(1.095),
                                take_profit: Some(1.11),
                                swap: None,
                                commission: None,
                                label: Some("trend".to_string()),
                                comment: Some("bot".to_string()),
                            },
                        ],
                        pending_orders: Vec::new(),
                    },
                    recent_deals: Vec::new(),
                },
            ),
        );
        session.connect(&mut state);
        state.order_ticket.selected_position_id = Some(9001);
        session.set_ctrader_execution_backend_for_test(
            crate::app_services::ctrader_execution::StubCTraderExecutionBackend::fail(
                "BROKER_REJECTED",
            ),
        );

        session.close_selected_position(&mut state);

        assert!(state.status_msg.contains("failed"));
        assert!(
            session
                .trade_journal
                .iter()
                .any(|line| line.contains("BROKER_REJECTED"))
        );
    }

    #[test]
    fn start_ctrader_auth_exposes_authorize_url_when_ready() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();

        let snapshot = session.start_ctrader_auth().expect("auth snapshot");

        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AwaitingAuthorizationCode
        );
        assert!(
            snapshot
                .authorize_url
                .as_deref()
                .unwrap_or_default()
                .contains("client_id=client")
        );
    }

    #[test]
    fn receive_ctrader_authorization_code_updates_auth_snapshot() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.start_ctrader_auth().expect("auth snapshot");

        let snapshot = session.receive_ctrader_authorization_code("code-123");

        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AuthorizationCodeReceived
        );
        assert!(snapshot.authorization_code_present);
    }

    #[test]
    fn build_ctrader_token_exchange_request_uses_configured_secret() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.start_ctrader_auth().expect("auth snapshot");
        session.receive_ctrader_authorization_code("code-123");

        let request = session
            .build_ctrader_token_exchange_request()
            .expect("token request");

        assert_eq!(request.code, "code-123");
        assert_eq!(request.client_secret, "secret");
        assert_eq!(request.redirect_uri, "http://localhost:3000/callback");
    }

    #[test]
    fn build_ctrader_token_exchange_request_keeps_staged_targets_pending_until_discovery() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.broker_settings_mut().ctrader.accounts.push(
            crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "acct-1".to_string(),
                label: "Primary".to_string(),
                enabled_for_execution: true,
            },
        );
        session.start_ctrader_auth().expect("auth snapshot");
        session.receive_ctrader_authorization_code("code-123");

        let _ = session
            .build_ctrader_token_exchange_request()
            .expect("token request");
        let auth = session.ctrader_auth_snapshot().expect("auth snapshot");

        assert_eq!(
            auth.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AccessTokenReady
        );
        assert_eq!(auth.account_count, 0);
        assert_eq!(auth.enabled_target_count, 0);
    }

    #[test]
    fn restore_ctrader_session_loads_saved_bundle_into_auth_snapshot() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed should succeed");

        let snapshot = session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");

        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
        );
        assert!(snapshot.token_persisted);
    }

    #[test]
    fn start_ctrader_live_auth_persists_tokens_and_updates_snapshot() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session.set_ctrader_live_auth_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::success(
                crate::app_services::ctrader_live_auth::CTraderLiveAuthResult {
                    callback_port: 43001,
                    authorization_code: "code-123".to_string(),
                    token_bundle: crate::app_services::ctrader_auth::CTraderTokenBundle {
                        access_token: "access".to_string(),
                        refresh_token: "refresh".to_string(),
                        token_type: "bearer".to_string(),
                        expires_in: 3600,
                        scope: "trading".to_string(),
                        created_at_unix: current_unix_seconds().expect("current unix time"),
                    },
                },
            ),
        );
        session.set_ctrader_account_discovery_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
                crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                    access_token: "access".to_string(),
                    permission_scope: "SCOPE_TRADE".to_string(),
                    accounts: vec![
                        crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                            account_id: "101".to_string(),
                            broker_title: "Broker A".to_string(),
                            account_name: "Primary Demo".to_string(),
                            trader_login: Some(500101),
                            is_live: Some(false),
                            enabled_for_execution: false,
                        },
                    ],
                },
            ),
        );

        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let snapshot = session
            .start_ctrader_live_auth(tx)
            .expect("live auth should start");
        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::ListeningForCallback
        );

        let mut completed = None;
        for _ in 0..20 {
            if let Some(snapshot) = session.poll_ctrader_live_auth() {
                completed = Some(snapshot);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let completed = completed.expect("poll should return completion");

        assert_eq!(
            completed.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
        );
        assert!(completed.token_persisted);
        assert_eq!(completed.account_count, 1);
        assert_eq!(
            session.broker_settings_mut().ctrader.accounts[0].account_id,
            "101"
        );
    }

    #[test]
    fn failed_ctrader_live_auth_reports_backend_error() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session.set_ctrader_live_auth_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::failure(
                "INVALID_CLIENT: cTrader rejected the OAuth application",
            ),
        );

        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        session
            .start_ctrader_live_auth(tx)
            .expect("live auth should start");

        let mut completed = None;
        for _ in 0..20 {
            if let Some(snapshot) = session.poll_ctrader_live_auth() {
                completed = Some(snapshot);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let completed = completed.expect("poll should return failure");

        assert_eq!(
            completed.state,
            crate::app_services::ctrader_auth::CTraderAuthState::Failed
        );
        assert!(completed.status_line.contains("INVALID_CLIENT"));
    }

    #[test]
    fn live_auth_completion_reports_account_discovery_failure() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session.set_ctrader_live_auth_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::success(
                crate::app_services::ctrader_live_auth::CTraderLiveAuthResult {
                    callback_port: 43001,
                    authorization_code: "code-123".to_string(),
                    token_bundle: fresh_ctrader_token_bundle("access", "refresh"),
                },
            ),
        );
        session.set_ctrader_account_discovery_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::failure(
                "INVALID_REQUEST: account list failed",
            ),
        );

        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        session
            .start_ctrader_live_auth(tx)
            .expect("live auth should start");

        let mut completed = None;
        for _ in 0..20 {
            if let Some(snapshot) = session.poll_ctrader_live_auth() {
                completed = Some(snapshot);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let completed = completed.expect("poll should return discovery failure");

        assert_eq!(
            completed.state,
            crate::app_services::ctrader_auth::CTraderAuthState::Failed
        );
        assert!(completed.token_persisted);
        assert!(
            completed
                .status_line
                .contains("INVALID_REQUEST: account list failed")
        );
    }

    #[test]
    fn clear_ctrader_saved_session_clears_restored_state() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed");

        session
            .clear_ctrader_saved_session()
            .expect("clear should succeed");

        let restored = session
            .restore_ctrader_session()
            .expect("restore should succeed");
        assert!(restored.is_none());
    }

    #[test]
    fn discover_ctrader_accounts_syncs_discovered_catalog_into_targets() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.accounts.push(
            crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "101".to_string(),
                label: "Operator Primary".to_string(),
                enabled_for_execution: true,
            },
        );
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");
        session.set_ctrader_account_discovery_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
                crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                    access_token: "access".to_string(),
                    permission_scope: "SCOPE_TRADE".to_string(),
                    accounts: vec![
                        crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                            account_id: "101".to_string(),
                            broker_title: "Broker A".to_string(),
                            account_name: "Primary Live".to_string(),
                            trader_login: Some(500101),
                            is_live: Some(true),
                            enabled_for_execution: false,
                        },
                        crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                            account_id: "202".to_string(),
                            broker_title: "Broker B".to_string(),
                            account_name: "Secondary Demo".to_string(),
                            trader_login: Some(500202),
                            is_live: Some(false),
                            enabled_for_execution: false,
                        },
                    ],
                },
            ),
        );

        let snapshot = session
            .discover_ctrader_accounts()
            .expect("discovery should succeed")
            .expect("snapshot should exist");

        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
        );
        assert_eq!(snapshot.account_count, 2);
        assert_eq!(snapshot.enabled_target_count, 1);
        assert_eq!(session.broker_settings_mut().ctrader.accounts.len(), 2);
        assert!(
            session
                .broker_settings_mut()
                .ctrader
                .accounts
                .iter()
                .any(|account| account.account_id == "101"
                    && account.label == "Operator Primary"
                    && account.enabled_for_execution)
        );
        assert!(
            session
                .broker_settings_mut()
                .ctrader
                .accounts
                .iter()
                .any(|account| account.account_id == "202" && !account.enabled_for_execution)
        );
    }

    #[test]
    fn discover_ctrader_accounts_uses_configured_demo_environment() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.environment = CTraderBrokerEnvironment::Demo;
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");

        let backend =
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
                crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                    access_token: "access".to_string(),
                    permission_scope: "SCOPE_TRADE".to_string(),
                    accounts: Vec::new(),
                },
            );
        session.set_ctrader_account_discovery_backend_for_test(backend.clone());

        session
            .discover_ctrader_accounts()
            .expect("discovery should succeed");

        let request = backend
            .last_request()
            .expect("stub backend should capture discovery request");
        assert_eq!(request.environment, CTraderEnvironment::Demo);
    }

    #[test]
    fn discover_ctrader_accounts_requires_restored_token_session() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();

        let err = session
            .discover_ctrader_accounts()
            .expect_err("discovery should fail without a restored token");

        assert!(
            err.to_string()
                .contains("cTrader account discovery requires a restored token session")
        );
    }

    #[test]
    fn discover_ctrader_accounts_requires_persisted_token_bundle() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session.set_ctrader_account_discovery_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
                crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                    access_token: "access".to_string(),
                    permission_scope: "SCOPE_TRADE".to_string(),
                    accounts: Vec::new(),
                },
            ),
        );
        let mut auth = crate::app_services::ctrader_auth::CTraderAuthSession::new(
            "client",
            "http://127.0.0.1:43001/callback",
        );
        auth.restore_from_storage(crate::app_services::ctrader_auth::CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: current_unix_seconds().expect("current unix time"),
        });
        session.ctrader_auth = Some(auth);

        let err = session
            .discover_ctrader_accounts()
            .expect_err("discovery should fail without persisted token bundle");

        assert!(
            err.to_string()
                .contains("cTrader account discovery requires a stored token bundle")
        );
    }

    #[test]
    fn discover_ctrader_accounts_uses_selected_demo_environment() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.environment =
            crate::app_services::broker_config::CTraderBrokerEnvironment::Demo;
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");
        let backend =
            crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
                crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                    access_token: "access".to_string(),
                    permission_scope: "SCOPE_TRADE".to_string(),
                    accounts: Vec::new(),
                },
            );
        session.set_ctrader_account_discovery_backend_for_test(backend.clone());

        session
            .discover_ctrader_accounts()
            .expect("discovery should succeed")
            .expect("snapshot should exist");

        let request = backend
            .last_request()
            .expect("stub should capture the discovery request");
        assert_eq!(request.environment, CTraderEnvironment::Demo);
    }

    #[test]
    fn ctrader_runtime_request_refreshes_expired_token_bundle_before_use() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.accounts.push(
            crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "101".to_string(),
                label: "Primary".to_string(),
                enabled_for_execution: true,
            },
        );
        let store = crate::app_services::secure_store::MemorySecretStoreBackend::default();
        session.set_ctrader_store_for_test(store.clone());
        session
            .seed_ctrader_token_bundle_for_test(
                crate::app_services::ctrader_auth::CTraderTokenBundle {
                    access_token: "expired-access".to_string(),
                    refresh_token: "expired-refresh".to_string(),
                    token_type: "bearer".to_string(),
                    expires_in: 60,
                    scope: "trading".to_string(),
                    created_at_unix: 1,
                },
            )
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");
        session.set_ctrader_live_auth_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::with_refresh_success(
                crate::app_services::ctrader_auth::CTraderTokenBundle {
                    access_token: "fresh-access".to_string(),
                    refresh_token: "fresh-refresh".to_string(),
                    token_type: "bearer".to_string(),
                    expires_in: 2628000,
                    scope: "trading".to_string(),
                    created_at_unix: 1_900_000_000,
                },
            ),
        );

        let request = session
            .build_ctrader_account_runtime_request()
            .expect("runtime request should refresh");

        assert_eq!(request.access_token, "fresh-access");
        let stored = session
            .ctrader_token_store
            .load_token_bundle()
            .expect("load should succeed")
            .expect("bundle should exist");
        assert_eq!(stored.access_token, "fresh-access");
        assert_eq!(stored.refresh_token, "fresh-refresh");
    }

    #[test]
    fn ctrader_runtime_request_fails_closed_when_refresh_fails() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.accounts.push(
            crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "101".to_string(),
                label: "Primary".to_string(),
                enabled_for_execution: true,
            },
        );
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(
                crate::app_services::ctrader_auth::CTraderTokenBundle {
                    access_token: "expired-access".to_string(),
                    refresh_token: "expired-refresh".to_string(),
                    token_type: "bearer".to_string(),
                    expires_in: 60,
                    scope: "trading".to_string(),
                    created_at_unix: 1,
                },
            )
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed")
            .expect("snapshot should exist");
        session.set_ctrader_live_auth_backend_for_test(
            crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::with_refresh_failure(
                "token refresh failed",
            ),
        );

        let err = session
            .build_ctrader_account_runtime_request()
            .expect_err("runtime request should fail when refresh fails");

        assert!(err.to_string().contains("token refresh failed"));
    }

    #[test]
    fn refresh_runtime_skips_ctrader_probe_within_refresh_window() {
        let mut state = sample_state(DataSource::CTrader, "Connected");
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let backend =
            crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
                crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                    trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                        account_id: 712345,
                        balance: 1000.0,
                        leverage: Some(50.0),
                        trader_login: Some(998877),
                        account_type: Some("NETTED".to_string()),
                        broker_name: Some("Demo Broker".to_string()),
                        money_digits: 2,
                        unrealized_pnl: 0.0,
                    },
                    reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                        account_id: 712345,
                        positions: Vec::new(),
                        pending_orders: Vec::new(),
                    },
                    recent_deals: Vec::new(),
                },
            );
        session.set_ctrader_account_runtime_backend_for_test(backend.clone());
        session.handle_ctrader_connect_result(
            &mut state,
            crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                    account_id: 712345,
                    balance: 1000.0,
                    leverage: Some(50.0),
                    trader_login: Some(998877),
                    account_type: Some("NETTED".to_string()),
                    broker_name: Some("Demo Broker".to_string()),
                    money_digits: 2,
                    unrealized_pnl: 0.0,
                },
                reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                    account_id: 712345,
                    positions: Vec::new(),
                    pending_orders: Vec::new(),
                },
                recent_deals: Vec::new(),
            },
        );

        session
            .refresh_runtime(&mut state)
            .expect("refresh within throttle window should succeed");

        assert!(backend.last_request().is_none());
    }

    #[test]
    fn start_ctrader_bootstrap_batch_rejects_concurrent_request() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.bootstrap_handle = Some(std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }));
        let (tx, _rx) = tokio::sync::mpsc::channel(4);

        let err = session
            .start_ctrader_bootstrap_batch(
                std::path::PathBuf::from("data"),
                vec!["EURUSD".to_string()],
                vec!["M15".to_string()],
                1,
                tx,
            )
            .expect_err("concurrent bootstrap should be rejected");

        assert!(err.to_string().contains("already running"));
        if let Some(handle) = session.bootstrap_handle.take() {
            let _ = handle.join();
        }
    }

    fn sample_prop_firm_order() -> CTraderNewOrderRequest {
        CTraderNewOrderRequest {
            account_id: 101,
            symbol_id: 1,
            order_type: CTraderOrderType::Market,
            trade_side: CTraderTradeSide::Buy,
            volume: 100000,
            limit_price: None,
            stop_price: None,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: Some(1.05000),
            take_profit: None,
            comment: None,
            base_slippage_price: None,
            slippage_in_points: None,
            label: None,
            position_id: None,
            client_order_id: None,
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_trigger_method: None,
        }
    }

    #[test]
    fn prop_firm_gate_blocks_order_without_stop_loss() {
        let mut order = sample_prop_firm_order();
        order.stop_loss = None;
        let risk = RiskConfig::default();
        let err =
            prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4).unwrap_err();
        assert!(err.to_string().contains("missing stop_loss"));
    }

    #[test]
    fn prop_firm_gate_blocks_when_daily_drawdown_breached() {
        let order = sample_prop_firm_order();
        let risk = RiskConfig::default();
        let err =
            prop_firm_pre_trade_check(&risk, &order, 9500.0, 10000.0, 10000.0, 4).unwrap_err();
        assert!(err.to_string().contains("Daily drawdown limit reached"));
        assert!(err.to_string().contains("current 5.00% >= max 4.00%"));
    }

    #[test]
    fn prop_firm_gate_respects_jpy_pip_precision() {
        let mut order = sample_prop_firm_order();
        order.limit_price = Some(150.00);
        order.stop_loss = Some(149.50); // 50 pips in JPY (2 digits)
        let risk = RiskConfig::default();
        // This should pass if 2-digit precision is used (50 pips).
        // If 4-digit was used, it would think it's 5000 pips.
        assert!(prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 2).is_ok());
    }

    #[test]
    fn prop_firm_gate_blocks_when_total_drawdown_breached() {
        let order = sample_prop_firm_order();
        let risk = RiskConfig::default();
        // Set day_start_equity equal to account_equity so daily DD is 0%, forcing it to hit total DD rule
        let err = prop_firm_pre_trade_check(&risk, &order, 8900.0, 10000.0, 8900.0, 4).unwrap_err();
        assert!(err.to_string().contains("Total drawdown limit reached"));
        assert!(err.to_string().contains("current 11.00% >= max 7.00%"));
    }

    #[test]
    fn prop_firm_gate_passes_valid_order_within_limits() {
        let order = sample_prop_firm_order();
        let risk = RiskConfig::default();
        assert!(prop_firm_pre_trade_check(&risk, &order, 10100.0, 10000.0, 10000.0, 4).is_ok());
    }

    #[test]
    fn prop_firm_gate_respects_disabled_stop_loss_requirement() {
        let mut order = sample_prop_firm_order();
        order.stop_loss = None;
        let risk = RiskConfig {
            require_stop_loss: false,
            ..RiskConfig::default()
        };
        assert!(prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4).is_ok());
    }
}
