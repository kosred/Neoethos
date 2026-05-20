use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewsItem {
    pub currency: String,
    pub impact: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewsPanel {
    pub filter_status: String,
    pub provider: String,
    pub items: Vec<NewsItem>,
    pub blackout_active: bool,
}

pub fn build_news_panel(state: &AppState) -> NewsPanel {
    let filter = &state.llm_news_filter;
    let blackout_active = filter.current_status.to_uppercase().contains("BLACKOUT");

    let items = filter
        .recent_events
        .iter()
        .take(20)
        .map(|ev| NewsItem {
            currency: ev.currency.clone(),
            impact: ev.impact.clone(),
            timestamp: ev.timestamp_ms.to_string(),
        })
        .collect();

    NewsPanel {
        filter_status: filter.current_status.clone(),
        provider: filter.llm_provider.clone(),
        items,
        blackout_active,
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState) {
    let panel = build_news_panel(state);

    egui::ScrollArea::vertical()
        .id_salt("news_scroll")
        .show(ui, |ui| {
            // ── Header ───────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.strong(
                    egui::RichText::new("Market News & Events")
                        .size(13.0)
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if panel.blackout_active {
                        theme::status_badge(ui, "BLACKOUT", theme::DANGER);
                    } else {
                        theme::status_badge(ui, "CLEAR", theme::SUCCESS);
                    }
                });
            });

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Filter: {}", panel.filter_status))
                        .size(11.0)
                        .color(theme::TEXT_MUTED),
                );
                ui.label(
                    egui::RichText::new(format!("· via {}", panel.provider))
                        .size(11.0)
                        .color(theme::TEXT_MUTED),
                );
            });

            if panel.blackout_active {
                ui.add_space(4.0);
                let mut frame = theme::section_frame(ui.style());
                frame.fill = theme::DANGER.linear_multiply(0.08);
                frame.stroke = egui::Stroke::new(1.0, theme::DANGER.linear_multiply(0.5));
                frame.show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "⬛ Trading blackout active — news event in progress. Execution may be restricted.",
                        )
                        .size(12.0)
                        .color(theme::DANGER),
                    );
                });
            }

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);

            // ── News items ────────────────────────────────────────────────
            if panel.items.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(
                            "No news events loaded.\nConnect to broker and enable LLM news filter in Settings.",
                        )
                        .size(12.0)
                        .color(theme::TEXT_MUTED),
                    );
                });
            } else {
                for item in &panel.items {
                    let impact_color = match item.impact.to_lowercase().as_str() {
                        "high" => theme::DANGER,
                        "medium" => theme::WARNING,
                        _ => theme::TEXT_MUTED,
                    };

                    theme::card_frame(ui.style()).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Impact badge
                            let badge_frame = egui::Frame::new()
                                .fill(impact_color.linear_multiply(0.15))
                                .inner_margin(egui::Margin::symmetric(6, 2))
                                .corner_radius(egui::CornerRadius::same(4));
                            badge_frame.show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(item.impact.to_uppercase())
                                        .size(10.0)
                                        .color(impact_color)
                                        .strong(),
                                );
                            });
                            ui.label(
                                egui::RichText::new(&item.currency)
                                    .size(12.0)
                                    .color(theme::ACCENT)
                                    .strong(),
                            );
                            if !item.timestamp.is_empty() {
                                ui.label(
                                    egui::RichText::new(&item.timestamp)
                                        .size(10.0)
                                        .color(theme::TEXT_MUTED),
                                );
                            }
                        });
                    });
                    ui.add_space(2.0);
                }
            }

            // ── Help hint at bottom ───────────────────────────────────────
            ui.add_space(8.0);
            ui.separator();
            ui.label(
                egui::RichText::new(
                    "News events are fetched via LLM filter. Configure provider in Settings → Intelligence.",
                )
                .size(10.0)
                .color(theme::TEXT_MUTED),
            );
        });
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
            auto_discovery: false,
            auto_training: false,
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

        // Without live news events, the items list MUST be empty (the panel
        // should never invent placeholder items). The provider label MUST be
        // populated so the operator can see which backend is wired in.
        // Blackout is only triggered when `current_status` contains "BLACKOUT"
        // (see the dedicated `news_panel_blackout_detection` test).
        assert!(
            panel.items.is_empty(),
            "items must be empty when no live news events are loaded; got {} items",
            panel.items.len()
        );
        assert!(
            !panel.provider.is_empty(),
            "provider label must be populated for the operator to identify the news backend"
        );
        assert!(
            !panel.blackout_active,
            "blackout should not trigger on a default state with empty filter_status"
        );
    }

    #[test]
    fn news_panel_blackout_detection() {
        let mut state = sample_state();
        state.llm_news_filter.current_status = "BLACKOUT - NFP release".to_string();
        let panel = build_news_panel(&state);
        assert!(panel.blackout_active);
    }
}
