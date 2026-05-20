use crate::app_services::trading::TradingSession;
use crate::app_services::ServiceEvent;
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;
use tokio::sync::mpsc;

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

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &mpsc::Sender<ServiceEvent>,
) {
    let panel = build_watchlist_panel(state, session);
    let chart_snapshot = session.market_chart_snapshot(state, Some(tx));
    let exec_snapshot = session.execution_surface_snapshot(state);
    let is_online = panel.runtime_status != "Offline" && !panel.runtime_status.is_empty();

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(
                egui::RichText::new("Markets")
                    .size(13.0)
                    .color(theme::TEXT_PRIMARY),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (badge_text, col) = if is_online {
                    ("LIVE", theme::SUCCESS)
                } else {
                    ("OFFLINE", theme::DANGER)
                };
                theme::status_badge(ui, badge_text, col);
            });
        });
        ui.label(
            egui::RichText::new(&panel.adapter_name)
                .color(theme::TEXT_MUTED)
                .size(11.0),
        );
        ui.add_space(4.0);

        egui::Grid::new("watchlist_header")
            .num_columns(4)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                for header in ["SYMBOL", "BID", "ASK", "CHG"] {
                    ui.label(
                        egui::RichText::new(header)
                            .size(9.5)
                            .color(theme::TEXT_MUTED)
                            .strong(),
                    );
                }
                ui.end_row();
            });
        ui.add_space(2.0);

        for symbol in &panel.symbols {
            let selected = *symbol == panel.selected_symbol;
            let (bid, ask, change) = if selected {
                (
                    format_quote(chart_snapshot.bid),
                    format_quote(chart_snapshot.ask),
                    format_change(chart_snapshot.price_change_pct),
                )
            } else {
                ("--".to_string(), "--".to_string(), "--".to_string())
            };
            let label = format!("{symbol:<7} {bid:>8} {ask:>8} {change:>7}");
            let color = if selected {
                theme::ACCENT
            } else {
                theme::TEXT_PRIMARY
            };
            if ui
                .add_sized(
                    [ui.available_width(), 26.0],
                    egui::Button::new(
                        egui::RichText::new(label)
                            .monospace()
                            .size(11.0)
                            .color(color),
                    )
                    .selected(selected),
                )
                .clicked()
            {
                state.selected_pair = symbol.clone();
            }
        }

        ui.add_space(8.0);

        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong(
                egui::RichText::new(format!("AI Signal - {}", panel.selected_symbol))
                    .color(theme::TEXT_PRIMARY)
                    .small(),
            );
            ui.add_space(4.0);

            let ai = &state.ai_insights_panel;
            if let (Some(buy), Some(sell), Some(neutral)) =
                (ai.prob_buy, ai.prob_sell, ai.prob_neutral)
            {
                compact_signal_bar(ui, "Buy ", buy, theme::SUCCESS);
                compact_signal_bar(ui, "Sell", sell, theme::DANGER);
                compact_signal_bar(ui, "Hold", neutral, theme::TEXT_MUTED);
            } else {
                ui.label(
                    egui::RichText::new("No model signal active")
                        .color(theme::TEXT_MUTED)
                        .small(),
                );
            }
        });

        ui.add_space(6.0);

        theme::section_frame(ui.style()).show(ui, |ui| {
            let pos_count = exec_snapshot.positions.len();
            let ord_count = exec_snapshot.pending_orders.len();
            ui.strong(
                egui::RichText::new("Exposure")
                    .color(theme::TEXT_PRIMARY)
                    .small(),
            );
            ui.add_space(4.0);

            if pos_count == 0 && ord_count == 0 {
                ui.label(
                    egui::RichText::new("No open positions or orders")
                        .color(theme::TEXT_MUTED)
                        .small(),
                );
            } else {
                for line in &exec_snapshot.positions {
                    ui.label(egui::RichText::new(line).color(theme::SUCCESS).small());
                }
                for line in &exec_snapshot.pending_orders {
                    ui.label(egui::RichText::new(line).color(theme::WARNING).small());
                }
            }
        });

        ui.add_space(6.0);

        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong(
                egui::RichText::new("Automation Tape")
                    .color(theme::TEXT_PRIMARY)
                    .small(),
            );
            ui.add_space(4.0);

            let timeline = &exec_snapshot.bot_timeline;
            if timeline.is_empty() {
                ui.label(
                    egui::RichText::new("No bot decisions yet")
                        .color(theme::TEXT_MUTED)
                        .small(),
                );
            } else {
                for entry in timeline.iter().rev().take(5) {
                    ui.label(egui::RichText::new(entry).color(theme::ACCENT).small());
                }
            }
        });
    });
}

fn format_quote(value: Option<f64>) -> String {
    value
        .map(|price| format!("{price:.5}"))
        .unwrap_or_else(|| "--".to_string())
}

fn format_change(value: Option<f64>) -> String {
    value
        .map(|pct| {
            let sign = if pct >= 0.0 { "+" } else { "" };
            format!("{sign}{pct:.2}%")
        })
        .unwrap_or_else(|| "--".to_string())
}

fn compact_signal_bar(ui: &mut egui::Ui, label: &str, value: f32, color: egui::Color32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(theme::TEXT_MUTED).small());
        ui.add(
            egui::ProgressBar::new(value)
                .text(format!("{:.0}%", value * 100.0))
                .fill(color),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::TradingSession;
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
        let mut state = AppState::new(
            runtime,
            &forex_core::Settings::default(),
            vec![
                "EURUSD".to_string(),
                "GBPUSD".to_string(),
                "XAUUSD".to_string(),
            ],
        );
        state.selected_pair = "EURUSD".to_string();
        state
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
