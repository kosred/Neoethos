use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewsPanel {
    pub items: Vec<String>,
}

pub fn build_news_panel(state: &AppState) -> NewsPanel {
    let mut items = vec![format!(
        "News filter status · {} · provider={}",
        state.llm_news_filter.current_status, state.llm_news_filter.llm_provider
    )];
    if state.llm_news_filter.recent_events.is_empty() {
        items.push(format!(
            "No live news events connected for {} yet.",
            state.selected_pair
        ));
    } else {
        for event in state.llm_news_filter.recent_events.iter().take(6) {
            items.push(format!(
                "{} {} @ {}",
                event.currency, event.impact, event.timestamp_ms
            ));
        }
    }
    NewsPanel { items }
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
        AppState::new(
            runtime,
            &forex_core::Settings::default(),
            vec!["EURUSD".to_string()],
        )
    }

    #[test]
    fn news_panel_groups_market_brief_and_upcoming_events() {
        let panel = build_news_panel(&sample_state());

        assert!(
            panel
                .items
                .iter()
                .any(|item: &String| item.contains("News filter status"))
        );
        assert!(
            panel
                .items
                .iter()
                .any(|item: &String| item.contains("No live news events connected"))
        );
    }
}
