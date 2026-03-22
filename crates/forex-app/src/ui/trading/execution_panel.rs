use crate::app_services::trading::{ExecutionSurfaceSnapshot, TradingSession};
use crate::app_state::AppState;
use crate::ui::components::open_log;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPanel {
    pub symbol: String,
    pub connection_status: String,
    pub adapter_name: String,
    pub integration_mode: String,
    pub supported_adapters: Vec<String>,
    pub primary_actions: Vec<String>,
    pub warnings: Vec<String>,
    pub diagnostics: Vec<String>,
}

pub fn build_execution_panel(snapshot: &ExecutionSurfaceSnapshot) -> ExecutionPanel {
    ExecutionPanel {
        symbol: snapshot.symbol.clone(),
        connection_status: snapshot.connection_status.clone(),
        adapter_name: snapshot.adapter_name.clone(),
        integration_mode: snapshot.integration_mode.clone(),
        supported_adapters: snapshot.supported_adapters.clone(),
        primary_actions: snapshot
            .primary_actions
            .iter()
            .map(|action| action.label.clone())
            .collect(),
        warnings: snapshot.warnings.clone(),
        diagnostics: snapshot.diagnostics.clone(),
    }
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &mut TradingSession) {
    let snapshot = session.execution_surface_snapshot(state);
    let panel = build_execution_panel(&snapshot);

    ui.strong(format!("Execution · {}", panel.symbol));
    ui.add_space(8.0);
    ui.label(egui::RichText::new(format!("Adapter: {}", panel.adapter_name)).color(theme::TEXT_MUTED));
    ui.label(
        egui::RichText::new(format!("Integration: {}", panel.integration_mode)).color(theme::TEXT_MUTED),
    );
    ui.label(
        egui::RichText::new(format!("Status: {}", panel.connection_status)).color(theme::TEXT_MUTED),
    );
    ui.label(
        egui::RichText::new(format!("Supported: {}", panel.supported_adapters.join(", ")))
            .color(theme::TEXT_MUTED),
    );
    ui.add_space(8.0);

    ui.horizontal(|ui| {
        for (idx, action) in snapshot.primary_actions.iter().enumerate() {
            let width = if idx == 0 {
                ui.available_width() / 2.0 - 4.0
            } else {
                ui.available_width()
            };
            let fill = if action.label == "Buy" {
                theme::SUCCESS.linear_multiply(0.45)
            } else {
                theme::DANGER.linear_multiply(0.45)
            };
            let response = ui.add_enabled(
                action.enabled,
                egui::Button::new(egui::RichText::new(&action.label).color(theme::TEXT_PRIMARY))
                    .fill(fill)
                    .min_size(egui::vec2(width, 34.0)),
            );
            if response.hovered() {
                if let Some(reason) = &action.reason {
                    response.on_hover_text(reason);
                }
            }
        }
    });

    ui.add_space(8.0);
    if !snapshot.positions.is_empty() || !snapshot.pending_orders.is_empty() {
        ui.label(
            egui::RichText::new(format!(
                "Positions: {} · Pending Orders: {}",
                snapshot.positions.len(),
                snapshot.pending_orders.len()
            ))
            .color(theme::TEXT_MUTED),
        );
    }

    for warning in &panel.warnings {
        ui.label(egui::RichText::new(warning).color(theme::WARNING));
    }

    if !panel.diagnostics.is_empty() {
        ui.add_space(8.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Diagnostics");
            ui.add_space(6.0);
            for line in &panel.diagnostics {
                ui.label(egui::RichText::new(line).color(theme::TEXT_MUTED));
            }
        });
    }

    ui.add_space(8.0);
    if snapshot.connection_status == "Local Mode" {
        ui.label(egui::RichText::new("Execution remains disabled in Local mode.").color(theme::WARNING));
    }
    if snapshot.connection_status != "Local Mode" {
        if session.is_connected() {
            if ui.button("Disconnect Runtime").clicked() {
                session.disconnect(state);
            }
        } else if ui.button("Connect Runtime").clicked() {
            session.connect(state);
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
    fn execution_panel_surfaces_primary_actions_runtime_summary_and_warnings() {
        let state = sample_state();
        let mut session = TradingSession::new();
        let snapshot = session.execution_surface_snapshot(&state);
        let panel = build_execution_panel(&snapshot);

        assert!(panel.primary_actions.contains(&"Buy".to_string()));
        assert!(panel.primary_actions.contains(&"Sell".to_string()));
        assert!(panel.supported_adapters.contains(&"cTrader".to_string()));
        assert_eq!(panel.connection_status, "Offline");
        assert!(panel
            .diagnostics
            .iter()
            .any(|line| line.contains("positions/orders feed is not wired yet")));
    }
}
