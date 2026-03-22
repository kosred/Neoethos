use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BottomStripPanel {
    pub sections: Vec<String>,
}

pub fn build_bottom_strip(_state: &AppState) -> BottomStripPanel {
    BottomStripPanel {
        sections: vec![
            "Positions / Orders / PnL".to_string(),
            "Bot Decisions Timeline".to_string(),
            "Execution Diagnostics".to_string(),
            "Manual Notes".to_string(),
        ],
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState) {
    let panel = build_bottom_strip(state);

    ui.columns(panel.sections.len(), |columns| {
        for (idx, section) in panel.sections.iter().enumerate() {
            theme::section_frame(columns[idx].style()).show(&mut columns[idx], |ui| {
                ui.strong(section);
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Workspace shell placeholder").color(theme::TEXT_MUTED));
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn bottom_strip_groups_positions_orders_timeline_and_notes() {
        let panel = build_bottom_strip(&sample_state());

        assert_eq!(panel.sections.len(), 4);
        assert!(panel.sections.contains(&"Positions / Orders / PnL".to_string()));
        assert!(panel.sections.contains(&"Bot Decisions Timeline".to_string()));
        assert!(panel.sections.contains(&"Manual Notes".to_string()));
    }
}
