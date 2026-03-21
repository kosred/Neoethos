use crate::app_record;
use crate::app_state::{AppState, DataSource};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::MT5Engine;
use tracing::error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingPanelMode {
    LocalOnly,
    Disconnected,
    Connected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionSnapshot {
    pub mode: TradingPanelMode,
    pub connected: bool,
    pub status_text: String,
    pub terminal_info: String,
}

#[derive(Default)]
pub struct TradingSession {
    engine: Option<MT5Engine>,
    connected: bool,
    terminal_info: String,
}

impl TradingSession {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn from_connected_terminal_for_test(terminal_info: impl Into<String>) -> Self {
        Self {
            engine: None,
            connected: true,
            terminal_info: terminal_info.into(),
        }
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn snapshot(&self, state: &AppState) -> ConnectionSnapshot {
        let mode = panel_mode(state.data_source, self.connected);
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
            mode,
            connected: self.connected,
            status_text,
            terminal_info: self.terminal_info.clone(),
        }
    }

    pub fn connect(&mut self, state: &mut AppState) {
        match MT5Engine::new() {
            Ok(mut engine) => match engine.initialize() {
                Ok(true) => {
                    state.status_msg = "Connected".to_string();
                    self.terminal_info = engine.terminal_info().unwrap_or_default();
                    self.connected = true;
                    self.engine = Some(engine);
                    record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
                }
                _ => {
                    self.connected = false;
                    self.engine = None;
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
                self.engine = None;
                self.terminal_info.clear();
                state.status_msg = format!("Error: {:?}", err);
                record_app_event("ui_mt5_connect", "FAILED", format!("UI MT5 bridge error: {err}"));
            }
        }
    }

    pub fn disconnect(&mut self, state: &mut AppState) {
        self.engine = None;
        self.connected = false;
        self.terminal_info.clear();
        state.status_msg = "Offline".to_string();
        record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
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
    use crate::app_state::{AppRuntimeConfig, HardwareState, RiskState, Tab};
    use std::path::PathBuf;

    fn sample_state(source: DataSource, status_msg: &str) -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: matches!(source, DataSource::Local),
            },
            current_tab: Tab::Trading,
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
}
