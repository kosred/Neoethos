use crate::app_services::trading::TradingSession;
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchlistPanel {
    pub selected_symbol: String,
    pub symbols: Vec<String>,
    pub runtime_status: String,
    pub adapter_name: String,
}

pub fn build_watchlist_panel(state: &AppState, session: &TradingSession) -> WatchlistPanel {
    let snapshot = session.snapshot(state);
    WatchlistPanel {
        selected_symbol: state.selected_pair.clone(),
        symbols: state.available_symbols.clone(),
        runtime_status: snapshot.status_text,
        adapter_name: snapshot.adapter_name,
    }
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &TradingSession) {
    let panel = build_watchlist_panel(state, session);

    ui.strong("Watchlist");
    ui.add_space(8.0);
    ui.label(egui::RichText::new(format!("Adapter: {}", panel.adapter_name)).color(theme::TEXT_MUTED));
    ui.label(egui::RichText::new(format!("Runtime: {}", panel.runtime_status)).color(theme::TEXT_MUTED));
    ui.add_space(8.0);

    for symbol in &panel.symbols {
        let selected = *symbol == panel.selected_symbol;
        let text = if selected {
            egui::RichText::new(symbol).color(theme::ACCENT).strong()
        } else {
            egui::RichText::new(symbol).color(theme::TEXT_PRIMARY)
        };
        if ui
            .add_sized(
                [ui.available_width(), 30.0],
                egui::Button::new(text).selected(selected),
            )
            .clicked()
        {
            state.selected_pair = symbol.clone();
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
            available_symbols: vec!["EURUSD".to_string(), "GBPUSD".to_string(), "XAUUSD".to_string()],
            discovery_job: None,
            training_job: None,
            bootstrap_form: crate::app_state::BootstrapFormState::default_for_symbol("EURUSD"),
            bootstrap_job: None,
            canonical_log_path: PathBuf::from("logs").join("forex-ai.log"),
            hardware: HardwareState::default(),
            risk: RiskState::default(),
        }
    }

    #[test]
    fn watchlist_panel_marks_selected_symbol_and_runtime() {
        let state = sample_state();
        let session = TradingSession::new();
        let panel = build_watchlist_panel(&state, &session);

        assert_eq!(panel.selected_symbol, "EURUSD");
        assert!(panel.symbols.contains(&"XAUUSD".to_string()));
        assert_eq!(panel.runtime_status, "Offline");
    }
}
