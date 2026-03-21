use crate::app_record;
use crate::app_state::{AppState, DataSource};
use eframe::egui;
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

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::MT5, false) => TradingPanelMode::Disconnected,
        (DataSource::MT5, true) => TradingPanelMode::Connected,
    }
}

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    mt5: &mut Option<MT5Engine>,
    terminal_info: &mut String,
) {
    ui.heading("Live Trading Terminal");
    ui.separator();

    match panel_mode(state.data_source, mt5.is_some()) {
        TradingPanelMode::LocalOnly => {
            ui.label("Live trading is disabled in Local mode.");
            ui.label("Please switch to MT5 source if you are on Windows.");
        }
        TradingPanelMode::Disconnected => {
            if ui.button("🚀 Connect to MetaTrader 5").clicked() {
                connect_mt5(state, mt5, terminal_info);
            }
        }
        TradingPanelMode::Connected => {
            ui.group(|ui| {
                ui.label("Account Details:");
                ui.label(terminal_info.as_str());
            });
            if ui.button("🛑 Disconnect").clicked() {
                *mt5 = None;
                state.status_msg = "Offline".to_string();
                record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
            }
        }
    }
}

fn connect_mt5(state: &mut AppState, mt5: &mut Option<MT5Engine>, terminal_info: &mut String) {
    match MT5Engine::new() {
        Ok(mut engine) => match engine.initialize() {
            Ok(true) => {
                state.status_msg = "Connected".to_string();
                *terminal_info = engine.terminal_info().unwrap_or_default();
                *mt5 = Some(engine);
                record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
            }
            _ => {
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
            state.status_msg = format!("Error: {:?}", err);
            record_app_event("ui_mt5_connect", "FAILED", format!("UI MT5 bridge error: {err}"));
        }
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

