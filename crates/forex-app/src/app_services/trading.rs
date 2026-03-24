use crate::app_record;
use crate::app_services::broker_config::{
    AdapterReadinessSnapshot, BrokerAccountTarget, BrokerSessionState, BrokerSettingsState,
    CTraderBrokerEnvironment,
};
use crate::app_services::ctrader_data::{
    load_chart_history, CTraderChartHistoryRequest, HistoricalBar,
};
use crate::app_services::ctrader_auth::{
    CTraderAccountSummary, CTraderAuthSession, CTraderAuthSnapshot, CTraderDiscoveredAccount,
    CTraderTokenExchangeRequest,
};
use crate::app_services::ctrader_live_auth::{
    build_default_loopback_config, CTraderAccountDiscoveryBackend,
    CTraderAccountDiscoveryRequest, CTraderEnvironment, CTraderLiveAuthBackend,
    CTraderLiveAuthRequest, CTraderLiveAuthResult, ProductionCTraderLiveAuthBackend,
    CTRADER_DEFAULT_SCOPE,
};
use crate::app_services::secure_store::{
    CTraderSecureStore, CTraderTokenStore, KeyringSecretStoreBackend,
};
use crate::app_state::{AppState, DataSource};
use forex_data::{discover_timeframes, load_symbol_timeframe, Ohlcv};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::{DealInfo, MT5Engine, PendingOrderInfo, PositionInfo};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAdapterKind {
    Mt5,
    CTrader,
    DxTrade,
}

impl TradingAdapterKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mt5 => "MT5",
            Self::CTrader => "cTrader",
            Self::DxTrade => "DXtrade",
        }
    }

    pub fn integration_mode(self) -> &'static str {
        match self {
            Self::Mt5 => "Local terminal bridge",
            Self::CTrader => "Remote Open API",
            Self::DxTrade => "Remote broker API",
        }
    }

    pub fn requires_local_terminal(self) -> bool {
        matches!(self, Self::Mt5)
    }

    pub fn supports_market_data(self) -> bool {
        true
    }

    pub fn supports_live_orders(self) -> bool {
        true
    }
}

pub const SUPPORTED_TRADING_ADAPTERS: [TradingAdapterKind; 3] = [
    TradingAdapterKind::Mt5,
    TradingAdapterKind::CTrader,
    TradingAdapterKind::DxTrade,
];

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct BrokerExecutionSnapshot {
    pub positions: Vec<PositionInfo>,
    pub pending_orders: Vec<PendingOrderInfo>,
    pub recent_deals: Vec<DealInfo>,
}

pub struct TradingSession {
    configured_adapter: TradingAdapterKind,
    broker_settings: BrokerSettingsState,
    ctrader_auth: Option<CTraderAuthSession>,
    ctrader_live_auth_backend: Arc<dyn CTraderLiveAuthBackend>,
    ctrader_account_discovery_backend: Arc<dyn CTraderAccountDiscoveryBackend>,
    ctrader_token_store: Arc<dyn CTraderTokenStore>,
    ctrader_live_auth_rx: Option<Receiver<Result<CTraderLiveAuthResult, String>>>,
    adapter: Option<TradingAdapter>,
    connected: bool,
    terminal_info: String,
    market_chart_cache: Option<CachedMarketSnapshot>,
    execution_surface_cache: Option<CachedExecutionSnapshot>,
}

enum TradingAdapter {
    Mt5(MT5Engine),
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

enum ExecutionFeedHandle<'a> {
    Mt5(&'a MT5Engine),
    Unavailable { reason: String },
}

impl TradingSession {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn from_connected_terminal_for_test(terminal_info: impl Into<String>) -> Self {
        Self {
            configured_adapter: TradingAdapterKind::Mt5,
            broker_settings: BrokerSettingsState::default(),
            ctrader_auth: None,
            ctrader_live_auth_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_discovery_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_token_store: Arc::new(CTraderSecureStore::new(
                "forex-ai",
                "ctrader.default",
                KeyringSecretStoreBackend,
            )),
            ctrader_live_auth_rx: None,
            adapter: None,
            connected: true,
            terminal_info: terminal_info.into(),
            market_chart_cache: None,
            execution_surface_cache: None,
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
        let session = self
            .ctrader_auth
            .get_or_insert_with(|| {
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
        let client_secret = self.broker_settings.ctrader.client_secret.trim().to_string();
        if client_secret.is_empty() {
            if let Some(session) = self.ctrader_auth.as_mut() {
                session.mark_failed();
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

    pub fn start_ctrader_live_auth(&mut self) -> anyhow::Result<CTraderAuthSnapshot> {
        let client_id = self.broker_settings.ctrader.client_id.trim().to_string();
        let client_secret = self.broker_settings.ctrader.client_secret.trim().to_string();
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
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = backend.run(request).map_err(|err| err.to_string());
            let _ = tx.send(result);
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
                    session.mark_failed();
                    let snapshot = session.snapshot();
                    self.ctrader_live_auth_rx = None;
                    return Some(snapshot);
                }
                self.ctrader_live_auth_rx = None;
                return None;
            }
        };
        self.ctrader_live_auth_rx = None;

        let session = self.ctrader_auth.get_or_insert_with(|| {
            CTraderAuthSession::new(
                self.broker_settings.ctrader.client_id.clone(),
                self.broker_settings.ctrader.redirect_uri.clone(),
            )
        });

        match outcome {
            Ok(result) => {
                session.mark_listening_for_callback(result.callback_port);
                session.receive_authorization_code(result.authorization_code);
                if self
                    .ctrader_token_store
                    .save_token_bundle(&result.token_bundle)
                    .is_err()
                {
                    session.mark_failed();
                    return Some(session.snapshot());
                }
                session.restore_from_storage(result.token_bundle);
                Some(session.snapshot())
            }
            Err(_) => {
                session.mark_failed();
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
        self.ctrader_auth = None;
        self.ctrader_live_auth_rx = None;
        Ok(())
    }

    pub fn discover_ctrader_accounts(&mut self) -> anyhow::Result<Option<CTraderAuthSnapshot>> {
        let session = self
            .ctrader_auth
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("cTrader account discovery requires a restored token session"))?;
        if !matches!(
            session.snapshot().state,
            crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
                | crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
        ) {
            return Err(anyhow::anyhow!(
                "cTrader account discovery requires a restored token session"
            ));
        }

        let access_token = self
            .ctrader_token_store
            .load_token_bundle()?
            .map(|bundle| bundle.access_token)
            .ok_or_else(|| anyhow::anyhow!("cTrader account discovery requires a stored token bundle"))?;
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
            TradingPanelMode::Disconnected => state.status_msg.clone(),
            TradingPanelMode::Connected => {
                if state.status_msg.trim().is_empty() {
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
        let ctrader_environment = matches!(state.data_source, DataSource::MT5)
            .then_some(adapter_kind)
            .filter(|kind| matches!(kind, TradingAdapterKind::CTrader))
            .map(|_| self.selected_ctrader_environment());
        let ctrader_account_id = matches!(state.data_source, DataSource::MT5)
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
            if cache.key == cache_key {
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
            (DataSource::MT5, TradingAdapterKind::CTrader) => self
                .load_ctrader_market_chart_snapshot(
                    &state.selected_pair,
                    &timeframe,
                    resolved_timeframes,
                    overlay_status,
                ),
            _ => match load_symbol_timeframe(&state.runtime.data_dir, &state.selected_pair, &timeframe) {
                Ok(ohlcv) => build_market_chart_snapshot_from_ohlcv(
                    &state.selected_pair,
                    &timeframe,
                    resolved_timeframes,
                    &ohlcv,
                    Vec::new(),
                    Vec::new(),
                )
                .with_overlay_status(overlay_status),
                Err(err) => MarketChartSnapshot {
                    symbol: state.selected_pair.clone(),
                    timeframe: timeframe.clone(),
                    available_timeframes: if available_timeframes.is_empty() {
                        vec![timeframe.clone()]
                    } else {
                        available_timeframes.clone()
                    },
                    candles: Vec::new(),
                    overlays: Vec::new(),
                    price_min: 0.0,
                    price_max: 0.0,
                    headline: format!("No market data loaded for {} {}", state.selected_pair, timeframe),
                    overlay_status,
                    warnings: vec![format!(
                        "Failed to load {} market data for {}: {}",
                        timeframe, state.selected_pair, err
                    )],
                },
            },
        };

        self.market_chart_cache = Some(CachedMarketSnapshot {
            key: cache_key,
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

        if let Some(cache) = &self.execution_surface_cache {
            if cache.key == cache_key && cache.refreshed_at.elapsed() < Duration::from_secs(1) {
                return cache.snapshot.clone();
            }
        }

        let mut runtime_warnings = Vec::new();
        let runtime = match self.execution_feed_handle(state).load_runtime_snapshot(&state.selected_pair, 24) {
            Ok(snapshot) => Some(snapshot),
            Err(err) => {
                runtime_warnings.push(err.to_string());
                None
            }
        };

        let snapshot = build_execution_surface_snapshot_with_runtime(
            state,
            &connection,
            runtime.as_ref(),
            runtime_warnings,
        );
        self.execution_surface_cache = Some(CachedExecutionSnapshot {
            key: cache_key,
            refreshed_at: Instant::now(),
            snapshot: snapshot.clone(),
        });
        snapshot
    }

    pub fn connect(&mut self, state: &mut AppState) {
        let readiness = self.adapter_readiness();
        match self.configured_adapter {
            TradingAdapterKind::Mt5 => match MT5Engine::new() {
                Ok(mut engine) => match engine.initialize() {
                    Ok(true) => {
                        state.status_msg = "Connected".to_string();
                        self.connected = true;
                        self.adapter = Some(TradingAdapter::Mt5(engine));
                        self.terminal_info = self
                            .adapter
                            .as_ref()
                            .map(TradingAdapter::terminal_info)
                            .unwrap_or_default();
                        self.market_chart_cache = None;
                        self.execution_surface_cache = None;
                        record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
                    }
                    _ => {
                        self.reset_runtime_state();
                        state.status_msg =
                            "Connection Failed (module missing or terminal closed)".to_string();
                        record_app_event(
                            "ui_mt5_connect",
                            "DEGRADED",
                            "UI MT5 connection failed (module missing or terminal closed)",
                        );
                    }
                },
                Err(err) => {
                    self.reset_runtime_state();
                    state.status_msg = format!("Error: {:?}", err);
                    record_app_event(
                        "ui_mt5_connect",
                        "FAILED",
                        format!("UI MT5 bridge error: {err}"),
                    );
                }
            },
            TradingAdapterKind::CTrader | TradingAdapterKind::DxTrade => {
                self.reset_runtime_state();
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
        self.reset_runtime_state();
        state.status_msg = "Offline".to_string();
        record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
    }

    pub fn select_adapter(&mut self, state: &mut AppState, kind: TradingAdapterKind) {
        let previous = self.active_adapter_kind();
        self.reset_runtime_state();
        self.configured_adapter = kind;
        state.status_msg = match state.data_source {
            DataSource::Local => "Local Mode".to_string(),
            DataSource::MT5 => format!("{} selected · disconnected", kind.as_str()),
        };
        if matches!(kind, TradingAdapterKind::CTrader) {
            if let Ok(Some(_)) = self.restore_ctrader_session() {
                state.status_msg = "cTrader selected · session restored".to_string();
            }
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
            DataSource::MT5 => match self.active_adapter_kind() {
                TradingAdapterKind::Mt5 if !self.connected => {
                    "Trade overlays unavailable while the trading runtime is disconnected.".to_string()
                }
                TradingAdapterKind::Mt5 => {
                    "Trade overlays will appear here once broker-backed fills and bot execution events are wired.".to_string()
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
            DataSource::MT5 => match &self.adapter {
                Some(TradingAdapter::Mt5(engine)) if self.connected => ExecutionFeedHandle::Mt5(engine),
                _ => ExecutionFeedHandle::Unavailable {
                    reason: self.active_adapter_kind().execution_feed_unavailable_reason(self.connected),
                },
            },
        }
    }

    fn reset_runtime_state(&mut self) {
        self.adapter = None;
        self.connected = false;
        self.terminal_info.clear();
        self.market_chart_cache = None;
        self.execution_surface_cache = None;
        self.ctrader_auth = None;
        self.ctrader_live_auth_rx = None;
    }

    fn load_ctrader_market_chart_snapshot(
        &self,
        symbol: &str,
        timeframe: &str,
        available_timeframes: Vec<String>,
        overlay_status: String,
    ) -> MarketChartSnapshot {
        match self.build_ctrader_chart_history_request(symbol, timeframe) {
            Ok(request) => match load_chart_history(&request) {
                Ok(history) => build_market_chart_snapshot_from_historical_bars(
                    &history.symbol.symbol_name,
                    timeframe,
                    available_timeframes,
                    &history.bars,
                    Vec::new(),
                    Vec::new(),
                )
                .with_overlay_status(overlay_status),
                Err(err) => MarketChartSnapshot {
                    symbol: symbol.to_string(),
                    timeframe: timeframe.to_string(),
                    available_timeframes,
                    candles: Vec::new(),
                    overlays: Vec::new(),
                    price_min: 0.0,
                    price_max: 0.0,
                    headline: format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                    overlay_status,
                    warnings: vec![format!(
                        "Failed to load cTrader {} market data for {}: {}",
                        timeframe, symbol, err
                    )],
                },
            },
            Err(err) => MarketChartSnapshot {
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                available_timeframes,
                candles: Vec::new(),
                overlays: Vec::new(),
                price_min: 0.0,
                price_max: 0.0,
                headline: format!("No cTrader market data loaded for {} {}", symbol, timeframe),
                overlay_status,
                warnings: vec![format!(
                    "cTrader chart history is unavailable for {} {}: {}",
                    symbol, timeframe, err
                )],
            },
        }
    }

    fn build_ctrader_chart_history_request(
        &self,
        symbol: &str,
        timeframe: &str,
    ) -> anyhow::Result<CTraderChartHistoryRequest> {
        let client_id = self.broker_settings.ctrader.client_id.trim();
        let client_secret = self.broker_settings.ctrader.client_secret.trim();
        if client_id.is_empty() || client_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "cTrader chart history requires configured client_id and client_secret"
            ));
        }

        let access_token = self
            .ctrader_token_store
            .load_token_bundle()?
            .map(|bundle| bundle.access_token)
            .ok_or_else(|| anyhow::anyhow!("cTrader chart history requires a stored token bundle"))?;

        let account_id = self
            .selected_ctrader_chart_account_id()
            .ok_or_else(|| anyhow::anyhow!("cTrader chart history requires at least one discovered account"))?;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
            .as_millis() as i64;
        let window_ms = chart_history_window_ms(timeframe)
            .ok_or_else(|| anyhow::anyhow!("unsupported cTrader chart timeframe {}", timeframe))?;

        Ok(CTraderChartHistoryRequest {
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
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
}

impl Default for TradingSession {
    fn default() -> Self {
        Self {
            configured_adapter: TradingAdapterKind::Mt5,
            broker_settings: BrokerSettingsState::default(),
            ctrader_auth: None,
            ctrader_live_auth_backend: Arc::new(ProductionCTraderLiveAuthBackend),
            ctrader_account_discovery_backend: Arc::new(ProductionCTraderLiveAuthBackend),
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
        }
    }
}

impl TradingAdapter {
    fn kind(&self) -> TradingAdapterKind {
        match self {
            Self::Mt5(_) => TradingAdapterKind::Mt5,
        }
    }

    fn terminal_info(&self) -> String {
        match self {
            Self::Mt5(engine) => engine.terminal_info().unwrap_or_default(),
        }
    }
}

impl TradingAdapterKind {
    fn execution_feed_unavailable_reason(self, connected: bool) -> String {
        match self {
            Self::Mt5 if !connected => {
                "MT5 execution feed is unavailable until the local terminal connects.".to_string()
            }
            Self::Mt5 => "MT5 execution feed is currently unavailable.".to_string(),
            Self::CTrader => "cTrader execution feed is not wired yet.".to_string(),
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

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::MT5, false) => TradingPanelMode::Disconnected,
        (DataSource::MT5, true) => TradingPanelMode::Connected,
    }
}

const MAX_CHART_CANDLES: usize = 96;

fn supported_ctrader_chart_timeframes() -> Vec<String> {
    ["M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1"]
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
    let candles: Vec<ChartCandle> = (start..ohlcv.len())
        .map(|idx| ChartCandle {
            timestamp: timestamps.and_then(|ts| ts.get(idx)).copied(),
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
        })
        .collect();

    let (price_min, price_max) = if candles.is_empty() {
        (0.0, 0.0)
    } else {
        candles.iter().fold((f64::MAX, f64::MIN), |(min_v, max_v), candle| {
            (min_v.min(candle.low), max_v.max(candle.high))
        })
    };

    let latest_close = candles.last().map(|candle| candle.close).unwrap_or_default();
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

    MarketChartSnapshot {
        symbol: symbol.to_string(),
        timeframe: timeframe.to_string(),
        available_timeframes,
        candles,
        overlays,
        price_min,
        price_max,
        headline,
        overlay_status:
            "Trade overlays will appear here once execution events are available.".to_string(),
        warnings,
    }
}

pub fn build_execution_surface_snapshot_with_runtime(
    state: &AppState,
    connection: &ConnectionSnapshot,
    runtime: Option<&BrokerExecutionSnapshot>,
    mut runtime_warnings: Vec<String>,
) -> ExecutionSurfaceSnapshot {
    let action_reason = match connection.mode {
        TradingPanelMode::LocalOnly => Some("Local mode disables live order submission.".to_string()),
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

    let (positions, pending_orders, bot_timeline) = if let Some(runtime) = runtime {
        diagnostics.push(format!("Open positions: {}", runtime.positions.len()));
        diagnostics.push(format!("Pending orders: {}", runtime.pending_orders.len()));
        diagnostics.push(format!("Recent fills: {}", runtime.recent_deals.len()));
        (
            runtime.positions.iter().map(format_position_line).collect(),
            runtime
                .pending_orders
                .iter()
                .map(format_pending_order_line)
                .collect(),
            runtime.recent_deals.iter().map(format_deal_line).collect(),
        )
    } else {
        diagnostics.push("Broker positions/orders feed is not wired yet for the app execution surface.".to_string());
        diagnostics.push("Bot execution timeline is not wired yet for the app execution surface.".to_string());
        (Vec::new(), Vec::new(), Vec::new())
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
}

fn broker_execution_snapshot_from_mt5(
    engine: &MT5Engine,
    symbol: Option<&str>,
    lookback_hours: i64,
) -> anyhow::Result<BrokerExecutionSnapshot> {
    Ok(BrokerExecutionSnapshot {
        positions: engine.positions(symbol)?,
        pending_orders: engine.orders(symbol)?,
        recent_deals: engine.recent_deals(symbol, lookback_hours)?,
    })
}

impl ExecutionFeedHandle<'_> {
    fn load_runtime_snapshot(
        &self,
        symbol: &str,
        lookback_hours: i64,
    ) -> anyhow::Result<BrokerExecutionSnapshot> {
        match self {
            Self::Mt5(engine) => broker_execution_snapshot_from_mt5(engine, Some(symbol), lookback_hours),
            Self::Unavailable { reason } => Err(anyhow::anyhow!(reason.clone())),
        }
    }
}

fn format_position_line(position: &PositionInfo) -> String {
    format!(
        "#{} · {} {} {:.2} · open {:.5} · current {:.5} · pnl {:+.2}",
        position.ticket,
        position.symbol,
        position.order_side,
        position.volume,
        position.price_open,
        position.price_current,
        position.profit
    )
}

fn format_pending_order_line(order: &PendingOrderInfo) -> String {
    format!(
        "#{} · {} {} {:.2} @ {:.5} · sl {:.5} · tp {:.5}",
        order.ticket,
        order.symbol,
        order.order_kind,
        order.volume_initial,
        order.price_open,
        order.stop_loss,
        order.take_profit
    )
}

fn format_deal_line(deal: &DealInfo) -> String {
    format!(
        "#{} · {} {} {:.2} @ {:.5} · pnl {:+.2} · fee {:+.2}",
        deal.ticket,
        deal.entry_kind,
        deal.order_side,
        deal.volume,
        deal.price,
        deal.profit,
        deal.fee
    )
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
    use crate::app_state::{AppRuntimeConfig, HardwareState, RiskState};
    use forex_data::Ohlcv;
    use mt5_bridge::{DealInfo, PendingOrderInfo, PositionInfo};
    use std::path::PathBuf;

    fn sample_state(source: DataSource, status_msg: &str) -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: matches!(source, DataSource::Local),
            },
            data_source: source,
            status_msg: status_msg.to_string(),
            selected_pair: "EURUSD".to_string(),
            chart_timeframe: "M1".to_string(),
            available_symbols: vec!["EURUSD".to_string()],
            discovery_job: None,
            training_job: None,
            canonical_log_path: PathBuf::from("logs").join("forex-ai.log"),
            hardware: HardwareState::default(),
            risk: RiskState::default(),
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
    fn connection_snapshot_reports_connected_mt5_session_details() {
        let state = sample_state(DataSource::MT5, "Connected");
        let session = TradingSession::from_connected_terminal_for_test(
            "TerminalInfo(community_account=False, connected=True)",
        );

        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.adapter_name, "MT5");
        assert_eq!(snapshot.mode, TradingPanelMode::Connected);
        assert!(snapshot.connected);
        assert_eq!(snapshot.status_text, "Connected");
        assert!(snapshot
            .terminal_info
            .contains("TerminalInfo(community_account=False"));
    }

    #[test]
    fn panel_mode_uses_data_source_and_connection_state() {
        assert_eq!(panel_mode(DataSource::Local, false), TradingPanelMode::LocalOnly);
        assert_eq!(panel_mode(DataSource::MT5, false), TradingPanelMode::Disconnected);
        assert_eq!(panel_mode(DataSource::MT5, true), TradingPanelMode::Connected);
    }

    #[test]
    fn connection_snapshot_reports_adapter_kind_for_connected_runtime() {
        let state = sample_state(DataSource::MT5, "Connected");
        let session = TradingSession::from_connected_terminal_for_test(
            "TerminalInfo(community_account=False, connected=True)",
        );

        let snapshot = session.snapshot(&state);

        assert_eq!(snapshot.adapter_name, "MT5");
        assert_eq!(snapshot.mode, TradingPanelMode::Connected);
    }

    #[test]
    fn connection_snapshot_reports_remote_api_metadata_for_stubbed_ctrader() {
        let state = sample_state(DataSource::MT5, "Offline");
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
        let state = sample_state(DataSource::MT5, "Offline");
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "Bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_710_000_000,
            })
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
        let state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();

        let snapshot = session.market_chart_snapshot(&state);

        assert!(snapshot.candles.is_empty());
        assert_eq!(snapshot.timeframe, "M1");
        assert_eq!(snapshot.available_timeframes.first().map(String::as_str), Some("M1"));
        assert!(snapshot
            .warnings
            .iter()
            .any(|warning| warning.contains("stored token bundle")));
        assert!(snapshot
            .headline
            .contains("No cTrader market data loaded"));
    }

    #[test]
    fn execution_surface_snapshot_disables_live_actions_and_surfaces_unwired_gaps() {
        let state = sample_state(DataSource::Local, "Local Mode");
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = session.execution_surface_snapshot(&state);

        assert_eq!(snapshot.symbol, "EURUSD");
        assert_eq!(snapshot.adapter_name, "cTrader");
        assert_eq!(snapshot.primary_actions.len(), 2);
        assert!(snapshot.primary_actions.iter().all(|action| !action.enabled));
        assert!(snapshot
            .warnings
            .iter()
            .any(|warning| warning.contains("Local mode")));
        assert!(snapshot
            .diagnostics
            .iter()
            .any(|line| line.contains("positions/orders feed is not wired yet")));
    }

    #[test]
    fn execution_surface_snapshot_formats_positions_orders_and_recent_fills() {
        let state = sample_state(DataSource::MT5, "Connected");
        let connection = TradingSession::from_connected_terminal_for_test(
            "TerminalInfo(connected=True)",
        )
        .snapshot(&state);
        let runtime = BrokerExecutionSnapshot {
            positions: vec![PositionInfo {
                ticket: 1001,
                symbol: "EURUSD".to_string(),
                order_side: "BUY".to_string(),
                volume: 0.20,
                price_open: 1.1000,
                price_current: 1.1025,
                profit: 50.0,
                stop_loss: 1.0950,
                take_profit: 1.1100,
                comment: "trend".to_string(),
                opened_at: 1710001000,
            }],
            pending_orders: vec![PendingOrderInfo {
                ticket: 2001,
                symbol: "EURUSD".to_string(),
                order_kind: "BUY_LIMIT".to_string(),
                volume_initial: 0.15,
                price_open: 1.0985,
                stop_loss: 1.0940,
                take_profit: 1.1070,
                comment: "breakout".to_string(),
                created_at: 1710002000,
            }],
            recent_deals: vec![DealInfo {
                ticket: 3001,
                order_ticket: 2001,
                position_id: 4001,
                symbol: "EURUSD".to_string(),
                entry_kind: "IN".to_string(),
                order_side: "BUY".to_string(),
                volume: 0.15,
                price: 1.0990,
                profit: 12.5,
                fee: -0.4,
                comment: "filled".to_string(),
                executed_at: 1710003000,
            }],
        };

        let snapshot =
            build_execution_surface_snapshot_with_runtime(&state, &connection, Some(&runtime), Vec::new());

        assert!(snapshot.positions.iter().any(|line| line.contains("BUY 0.20")));
        assert!(snapshot
            .pending_orders
            .iter()
            .any(|line| line.contains("BUY_LIMIT 0.15")));
        assert!(snapshot
            .bot_timeline
            .iter()
            .any(|line| line.contains("IN BUY 0.15")));
        assert!(snapshot.warnings.is_empty());
    }

    #[test]
    fn execution_surface_snapshot_surfaces_adapter_specific_unwired_feed_reason() {
        let state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = session.execution_surface_snapshot(&state);

        assert!(snapshot
            .warnings
            .iter()
            .any(|warning| warning.contains("cTrader execution feed is not wired yet")));
    }

    #[test]
    fn selecting_adapter_updates_configured_runtime_and_status_message() {
        let mut state = sample_state(DataSource::MT5, "Offline");
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
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        session.connect(&mut state);

        assert_eq!(
            state.status_msg,
            "cTrader configuration incomplete: missing client_id, client_secret, redirect_uri"
        );
        assert!(!session.is_connected());
    }

    #[test]
    fn connect_marks_ready_remote_adapter_as_auth_pending_until_live_wiring_exists() {
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();

        session.connect(&mut state);

        assert_eq!(
            state.status_msg,
            "cTrader credentials ready · live auth is not wired yet"
        );
        assert!(!session.is_connected());
    }

    #[test]
    fn start_ctrader_auth_exposes_authorize_url_when_ready() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();

        let snapshot = session.start_ctrader_auth().expect("auth snapshot");

        assert_eq!(
            snapshot.state,
            crate::app_services::ctrader_auth::CTraderAuthState::AwaitingAuthorizationCode
        );
        assert!(snapshot
            .authorize_url
            .as_deref()
            .unwrap_or_default()
            .contains("client_id=client"));
    }

    #[test]
    fn receive_ctrader_authorization_code_updates_auth_snapshot() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session
            .broker_settings_mut()
            .ctrader
            .accounts
            .push(crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "acct-1".to_string(),
                label: "Primary".to_string(),
                enabled_for_execution: true,
            });
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_774_147_200,
            })
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
                        created_at_unix: 1_774_147_200,
                    },
                },
            ),
        );

        let snapshot = session.start_ctrader_live_auth().expect("live auth should start");
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
            crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
        );
        assert!(completed.token_persisted);
    }

    #[test]
    fn clear_ctrader_saved_session_clears_restored_state() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_774_147_200,
            })
            .expect("seed should succeed");
        session
            .restore_ctrader_session()
            .expect("restore should succeed");

        session
            .clear_ctrader_saved_session()
            .expect("clear should succeed");

        let restored = session.restore_ctrader_session().expect("restore should succeed");
        assert!(restored.is_none());
    }

    #[test]
    fn discover_ctrader_accounts_syncs_discovered_catalog_into_targets() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session
            .broker_settings_mut()
            .ctrader
            .accounts
            .push(crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "101".to_string(),
                label: "Operator Primary".to_string(),
                enabled_for_execution: true,
            });
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_774_147_200,
            })
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
        assert!(session
            .broker_settings_mut()
            .ctrader
            .accounts
            .iter()
            .any(|account| account.account_id == "101"
                && account.label == "Operator Primary"
                && account.enabled_for_execution));
        assert!(session
            .broker_settings_mut()
            .ctrader
            .accounts
            .iter()
            .any(|account| account.account_id == "202" && !account.enabled_for_execution));
    }

    #[test]
    fn discover_ctrader_accounts_uses_configured_demo_environment() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();
        session.broker_settings_mut().ctrader.environment = CTraderBrokerEnvironment::Demo;
        session.set_ctrader_store_for_test(
            crate::app_services::secure_store::MemorySecretStoreBackend::default(),
        );
        session
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_774_147_200,
            })
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
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://127.0.0.1:43001/callback".to_string();

        let err = session
            .discover_ctrader_accounts()
            .expect_err("discovery should fail without a restored token");

        assert!(err
            .to_string()
            .contains("cTrader account discovery requires a restored token session"));
    }

    #[test]
    fn discover_ctrader_accounts_requires_persisted_token_bundle() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
            created_at_unix: 1_774_147_200,
        });
        session.ctrader_auth = Some(auth);

        let err = session
            .discover_ctrader_accounts()
            .expect_err("discovery should fail without persisted token bundle");

        assert!(err
            .to_string()
            .contains("cTrader account discovery requires a stored token bundle"));
    }

    #[test]
    fn discover_ctrader_accounts_uses_selected_demo_environment() {
        let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
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
            .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 3600,
                scope: "trading".to_string(),
                created_at_unix: 1_774_147_200,
            })
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
}
