use crate::app_record;
use crate::app_state::{AppState, DataSource};
use forex_data::{discover_timeframes, load_symbol_timeframe, Ohlcv};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::MT5Engine;
use std::path::PathBuf;
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

pub struct TradingSession {
    configured_adapter: TradingAdapterKind,
    adapter: Option<TradingAdapter>,
    connected: bool,
    terminal_info: String,
    market_chart_cache: Option<CachedMarketSnapshot>,
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
        }
    }

    pub fn is_connected(&self) -> bool {
        self.connected
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
                        record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
                    }
                    _ => {
                        self.connected = false;
                        self.adapter = None;
                        self.terminal_info.clear();
                        self.market_chart_cache = None;
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
                    self.connected = false;
                    self.adapter = None;
                    self.terminal_info.clear();
                    self.market_chart_cache = None;
                    state.status_msg = format!("Error: {:?}", err);
                    record_app_event(
                        "ui_mt5_connect",
                        "FAILED",
                        format!("UI MT5 bridge error: {err}"),
                    );
                }
            },
            TradingAdapterKind::CTrader | TradingAdapterKind::DxTrade => {
                self.connected = false;
                self.adapter = None;
                self.terminal_info.clear();
                self.market_chart_cache = None;
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
        self.adapter = None;
        self.connected = false;
        self.terminal_info.clear();
        self.market_chart_cache = None;
        state.status_msg = "Offline".to_string();
        record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
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
}

impl Default for TradingSession {
    fn default() -> Self {
        Self {
            configured_adapter: TradingAdapterKind::Mt5,
            adapter: None,
            connected: false,
            terminal_info: String::new(),
            market_chart_cache: None,
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

pub fn build_execution_surface_snapshot(
    state: &AppState,
    session: &TradingSession,
) -> ExecutionSurfaceSnapshot {
    let snapshot = session.snapshot(state);
    let action_reason = match snapshot.mode {
        TradingPanelMode::LocalOnly => Some("Local mode disables live order submission.".to_string()),
        TradingPanelMode::Disconnected => {
            Some("Connect a broker adapter before sending live orders.".to_string())
        }
        TradingPanelMode::Connected => None,
    };
    let action_enabled = action_reason.is_none() && snapshot.supports_live_orders;
    let warnings = action_reason
        .clone()
        .into_iter()
        .chain((!snapshot.connected && snapshot.requires_local_terminal).then(|| {
            "The configured adapter requires a local terminal runtime that is not currently connected.".to_string()
        }))
        .collect();
    let mut diagnostics = vec![
        format!("Adapter: {}", snapshot.adapter_name),
        format!("Integration: {}", snapshot.integration_mode),
        format!(
            "Market data capability: {}",
            if snapshot.supports_market_data {
                "available"
            } else {
                "unavailable"
            }
        ),
        format!(
            "Live order capability: {}",
            if snapshot.supports_live_orders {
                "available when connected"
            } else {
                "unavailable"
            }
        ),
        "Broker positions/orders feed is not wired yet for the app execution surface.".to_string(),
        "Bot execution timeline is not wired yet for the app execution surface.".to_string(),
    ];
    if !snapshot.terminal_info.trim().is_empty() {
        diagnostics.push(format!("Terminal: {}", snapshot.terminal_info));
    }

    ExecutionSurfaceSnapshot {
        symbol: state.selected_pair.clone(),
        adapter_name: snapshot.adapter_name,
        integration_mode: snapshot.integration_mode,
        connection_status: snapshot.status_text,
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
        positions: Vec::new(),
        pending_orders: Vec::new(),
        bot_timeline: Vec::new(),
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
        let session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

        let snapshot = build_execution_surface_snapshot(&state, &session);

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
}
