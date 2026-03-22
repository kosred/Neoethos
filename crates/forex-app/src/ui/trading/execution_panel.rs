use crate::app_services::trading::{
    TradingPanelMode, TradingSession, SUPPORTED_TRADING_ADAPTERS,
};
use crate::app_state::AppState;
use crate::ui::components::open_log;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPanel {
    pub connection_status: String,
    pub adapter_name: String,
    pub integration_mode: String,
    pub supported_adapters: Vec<String>,
    pub primary_actions: Vec<String>,
}

pub fn build_execution_panel(state: &AppState, session: &TradingSession) -> ExecutionPanel {
    let snapshot = session.snapshot(state);
    ExecutionPanel {
        connection_status: snapshot.status_text,
        adapter_name: snapshot.adapter_name,
        integration_mode: snapshot.integration_mode,
        supported_adapters: SUPPORTED_TRADING_ADAPTERS
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        primary_actions: vec!["Buy".to_string(), "Sell".to_string()],
    }
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &mut TradingSession) {
    let panel = build_execution_panel(state, session);
    let snapshot = session.snapshot(state);

    ui.strong("Execution");
    ui.add_space(8.0);
    ui.label(egui::RichText::new(format!("Adapter: {}", panel.adapter_name)).color(theme::TEXT_MUTED));
    ui.label(egui::RichText::new(format!("Integration: {}", panel.integration_mode)).color(theme::TEXT_MUTED));
    ui.label(egui::RichText::new(format!("Status: {}", panel.connection_status)).color(theme::TEXT_MUTED));
    ui.label(
        egui::RichText::new(format!(
            "Supported: {}",
            panel.supported_adapters.join(", ")
        ))
        .color(theme::TEXT_MUTED),
    );
    ui.add_space(8.0);

    ui.horizontal(|ui| {
        ui.add_sized(
            [ui.available_width() / 2.0 - 4.0, 34.0],
            egui::Button::new(egui::RichText::new("Buy").color(theme::TEXT_PRIMARY))
                .fill(theme::SUCCESS.linear_multiply(0.45)),
        );
        ui.add_sized(
            [ui.available_width(), 34.0],
            egui::Button::new(egui::RichText::new("Sell").color(theme::TEXT_PRIMARY))
                .fill(theme::DANGER.linear_multiply(0.45)),
        );
    });

    ui.add_space(8.0);
    match snapshot.mode {
        TradingPanelMode::Disconnected => {
            if ui.button("Connect Runtime").clicked() {
                session.connect(state);
            }
        }
        TradingPanelMode::Connected => {
            if ui.button("Disconnect Runtime").clicked() {
                session.disconnect(state);
            }
        }
        TradingPanelMode::LocalOnly => {
            ui.label(egui::RichText::new("Execution remains disabled in Local mode.").color(theme::WARNING));
        }
    }

    if ui.button("Open Log").clicked() {
        if let Err(err) = open_log(&state.canonical_log_path) {
            state.status_msg = format!("Log open failed: {}", err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::TradingSession;
    use crate::app_state::{AppRuntimeConfig, AppState, DataSource, HardwareState, RiskState};
    use std::path::PathBuf;

    fn sample_state() -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: false,
            },
            data_source: DataSource::MT5,
            status_msg: "Offline".to_string(),
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
    fn execution_panel_surfaces_primary_actions_and_runtime_summary() {
        let state = sample_state();
        let session = TradingSession::new();
        let panel = build_execution_panel(&state, &session);

        assert!(panel.primary_actions.contains(&"Buy".to_string()));
        assert!(panel.primary_actions.contains(&"Sell".to_string()));
        assert!(panel.supported_adapters.contains(&"cTrader".to_string()));
        assert_eq!(panel.connection_status, "Offline");
    }
}
