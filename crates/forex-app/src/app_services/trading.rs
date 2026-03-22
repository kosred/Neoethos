use crate::app_record;
use crate::app_state::{AppState, DataSource};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::MT5Engine;
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

pub struct TradingSession {
    configured_adapter: TradingAdapterKind,
    adapter: Option<TradingAdapter>,
    connected: bool,
    terminal_info: String,
}

enum TradingAdapter {
    Mt5(MT5Engine),
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
        }
    }

    #[cfg(test)]
    pub fn with_configured_adapter_for_test(kind: TradingAdapterKind) -> Self {
        Self {
            configured_adapter: kind,
            adapter: None,
            connected: false,
            terminal_info: String::new(),
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
                        record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
                    }
                    _ => {
                        self.connected = false;
                        self.adapter = None;
                        self.terminal_info.clear();
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
        state.status_msg = "Offline".to_string();
        record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
    }
}

impl Default for TradingSession {
    fn default() -> Self {
        Self {
            configured_adapter: TradingAdapterKind::Mt5,
            adapter: None,
            connected: false,
            terminal_info: String::new(),
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
}
