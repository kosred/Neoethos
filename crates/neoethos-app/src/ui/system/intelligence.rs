use crate::app_state::AppState;
use crate::ui::components::{
    DashboardCard, DashboardSection, render_dashboard_sections, render_summary_cards,
    render_view_header,
};
use crate::ui::system::shared::sync_news_now;
use crate::ui::theme;
use eframe::egui;

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        render_view_header(
            ui,
            "Intelligence",
            "Keep AI gating, news filters, providers, and decision-support signals together on their own control surface.",
        );
        ui.separator();

        let summary_cards = vec![
            DashboardCard {
                label: "AI Auto-Trade".to_string(),
                value: if state.auto_trade_enabled {
                    "Enabled".to_string()
                } else {
                    "Disabled".to_string()
                },
            },
            DashboardCard {
                label: "LLM Provider".to_string(),
                value: state.llm_news_filter.llm_provider.clone(),
            },
            DashboardCard {
                label: "News Gate".to_string(),
                value: if state.llm_news_filter.enabled {
                    state.llm_news_filter.current_status.clone()
                } else {
                    "Disabled".to_string()
                },
            },
            DashboardCard {
                label: "Recent Events".to_string(),
                value: state.llm_news_filter.recent_events.len().to_string(),
            },
        ];
        render_summary_cards(ui, "Intelligence Snapshot", &summary_cards);

        let sections = vec![
            DashboardSection {
                title: "Decision Controls".to_string(),
                rows: vec![
                    ("Selected Pair".to_string(), state.selected_pair.clone()),
                    (
                        "AI Auto-Trade".to_string(),
                        if state.auto_trade_enabled {
                            "Armed".to_string()
                        } else {
                            "Manual-only".to_string()
                        },
                    ),
                    (
                        "News Provider".to_string(),
                        state.llm_news_filter.llm_provider.clone(),
                    ),
                    (
                        "News Status".to_string(),
                        state.llm_news_filter.current_status.clone(),
                    ),
                ],
            },
            DashboardSection {
                title: "Feed Health".to_string(),
                rows: vec![
                    (
                        "Recent News Events".to_string(),
                        state.llm_news_filter.recent_events.len().to_string(),
                    ),
                    (
                        "API Key".to_string(),
                        if state.llm_news_filter.api_key.is_some() {
                            "Configured".to_string()
                        } else {
                            "Missing".to_string()
                        },
                    ),
                ],
            },
        ];
        render_dashboard_sections(ui, "intelligence_section", &sections);

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Intelligence Settings");
            ui.add_space(6.0);
            ui.checkbox(&mut state.auto_trade_enabled, "Enable AI Auto-Trade");

            ui.horizontal(|ui| {
                ui.label("LLM Provider");
                ui.add_sized(
                    [ui.available_width().max(200.0), 24.0],
                    egui::TextEdit::singleline(&mut state.llm_news_filter.llm_provider),
                );
            });

            // audit-fix F8: surface the secret only at the egui text-edit
            // boundary via secrecy::ExposeSecret, then re-wrap on assign
            // so the SecretString destructor zeroes the bytes on drop.
            use secrecy::ExposeSecret;
            let mut api_key: String = state
                .llm_news_filter
                .api_key
                .as_ref()
                .map(|s| s.expose_secret().to_string())
                .unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("API Key");
                ui.add_sized(
                    [ui.available_width().max(200.0), 24.0],
                    egui::TextEdit::singleline(&mut api_key).password(true),
                );
            });
            if api_key.trim().is_empty() {
                state.llm_news_filter.api_key = None;
            } else {
                state.llm_news_filter.api_key =
                    Some(secrecy::SecretString::from(api_key));
            }

            ui.checkbox(
                &mut state.llm_news_filter.enabled,
                "Enable LLM News Blackout Kill-Switch",
            );

            ui.horizontal(|ui| {
                ui.label(format!(
                    "Current Status: {}",
                    state.llm_news_filter.current_status
                ));
                if ui.button("Sync News Now").clicked() {
                    sync_news_now(state, tx);
                }
            });
        });

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            state.ai_insights_panel.news_blackout_active = if state.llm_news_filter.enabled {
                Some(
                    state
                        .llm_news_filter
                        .current_status
                        .eq_ignore_ascii_case("BLACKOUT"),
                )
            } else {
                Some(false)
            };
            state.ai_insights_panel.show(ui);
        });
    });
}
