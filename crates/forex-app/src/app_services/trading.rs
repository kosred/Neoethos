use crate::app_record;
use crate::app_state::{AppState, DataSource};
use forex_data::{discover_timeframes, load_symbol_timeframe, Ohlcv};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::{DealInfo, MT5Engine, PendingOrderInfo, PositionInfo};
use std::path::PathBuf;
use std::time::{Duration, Instant};
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
    symbol: String,
    timeframe: String,
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
            adapter: None,
            connected: false,
            terminal_info: String::new(),
            market_chart_cache: None,
            execution_surface_cache: None,
        }
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn configured_adapter(&self) -> TradingAdapterKind {
        self.configured_adapter
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
        let available_timeframes =
            discover_timeframes(&state.runtime.data_dir, &state.selected_pair).unwrap_or_default();
        let timeframe =
            preferred_chart_timeframe(&available_timeframes, state.chart_timeframe.as_str());
        let cache_key = MarketChartCacheKey {
            data_root: state.runtime.data_dir.clone(),
            data_source: state.data_source,
            symbol: state.selected_pair.clone(),
            timeframe: timeframe.clone(),
        };

        if let Some(cache) = &self.market_chart_cache {
            if cache.key == cache_key {
                return cache.snapshot.clone();
            }
        }

        let overlay_status = self.overlay_status(state);
        let snapshot = match load_symbol_timeframe(&state.runtime.data_dir, &state.selected_pair, &timeframe)
        {
            Ok(ohlcv) => build_market_chart_snapshot_from_ohlcv(
                &state.selected_pair,
                &timeframe,
                if available_timeframes.is_empty() {
                    vec![timeframe.clone()]
                } else {
                    available_timeframes.clone()
                },
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
                state.status_msg = format!(
                    "{} adapter is defined but not wired yet",
                    self.configured_adapter.as_str()
                );
                record_app_event(
                    "ui_adapter_connect",
                    "DEGRADED",
                    format!(
                        "{} adapter requested but not wired yet",
                        self.configured_adapter.as_str()
                    ),
                );
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
            DataSource::MT5 if !self.connected => {
                "Trade overlays unavailable while the trading runtime is disconnected.".to_string()
            }
            DataSource::MT5 => {
                "Trade overlays will appear here once broker-backed fills and bot execution events are wired.".to_string()
            }
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
    }
}

impl Default for TradingSession {
    fn default() -> Self {
        Self {
            configured_adapter: TradingAdapterKind::Mt5,
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

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::MT5, false) => TradingPanelMode::Disconnected,
        (DataSource::MT5, true) => TradingPanelMode::Connected,
    }
}

const MAX_CHART_CANDLES: usize = 96;

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
}
