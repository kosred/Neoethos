use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewsPanel {
    pub items: Vec<String>,
}

pub fn build_news_panel(state: &AppState) -> NewsPanel {
    NewsPanel {
        items: vec![
            format!("ECB speaker in 12m · {}", state.selected_pair),
            "US yields rising into open".to_string(),
            "LLM summary · risk-on fading, stay selective".to_string(),
        ],
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState) {
    let panel = build_news_panel(state);
    ui.strong("News + Events");
    ui.add_space(8.0);
    for item in &panel.items {
        theme::card_frame(ui.style()).show(ui, |ui| {
            ui.label(egui::RichText::new(item).color(theme::TEXT_PRIMARY));
        });
        ui.add_space(6.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppRuntimeConfig, AppState};
    use std::path::PathBuf;

    fn sample_state() -> AppState {
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: PathBuf::from("data"),
            start_local: false,
        };
        AppState::new(runtime, vec!["EURUSD".to_string()])
    }

    #[test]
    fn news_panel_groups_market_brief_and_upcoming_events() {
        let panel = build_news_panel(&sample_state());

        assert!(panel.items.iter().any(|item: &String| item.contains("ECB")));
        assert!(panel.items.iter().any(|item: &String| item.contains("LLM summary")));
    }
}
