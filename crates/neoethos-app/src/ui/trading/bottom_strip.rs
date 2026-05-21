use crate::app_services::trading::{ExecutionSurfaceSnapshot, TradingSession};
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BottomStripSection {
    pub title: String,
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BottomStripPanel {
    pub sections: Vec<BottomStripSection>,
}

pub fn build_bottom_strip(snapshot: &ExecutionSurfaceSnapshot) -> BottomStripPanel {
    BottomStripPanel {
        sections: vec![
            BottomStripSection {
                title: "Positions".to_string(),
                lines: snapshot.positions.clone(),
            },
            BottomStripSection {
                title: "Orders".to_string(),
                lines: snapshot.pending_orders.clone(),
            },
            BottomStripSection {
                title: "Bot Log".to_string(),
                lines: snapshot.bot_timeline.clone(),
            },
            BottomStripSection {
                title: "Diagnostics".to_string(),
                lines: snapshot.diagnostics.clone(),
            },
            BottomStripSection {
                title: "Journal".to_string(),
                lines: snapshot
                    .history_rows
                    .iter()
                    .cloned()
                    .chain(snapshot.journal_rows.iter().cloned())
                    .collect(),
            },
        ],
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState, session: &mut TradingSession) {
    let snapshot = session.execution_surface_snapshot(state);
    let panel = build_bottom_strip(&snapshot);

    let tab_id = egui::Id::new("bottom_strip_selected_tab");
    let mut selected: usize = ui.data(|d| d.get_temp(tab_id).unwrap_or(0usize));
    // Clamp in case panel has fewer sections than the stored index.
    if selected >= panel.sections.len() {
        selected = 0;
    }

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (idx, section) in panel.sections.iter().enumerate() {
            let count = section.lines.len();
            let label = if count > 0 {
                format!("{}  ({})", section.title, count)
            } else {
                section.title.clone()
            };

            let is_selected = idx == selected;
            let text = egui::RichText::new(&label)
                .size(13.0)
                .color(if is_selected {
                    theme::ACCENT
                } else {
                    theme::TEXT_MUTED
                });

            let btn =
                egui::Button::new(text)
                    .selected(is_selected)
                    .corner_radius(egui::CornerRadius {
                        nw: 6,
                        ne: 6,
                        sw: 0,
                        se: 0,
                    });
            if ui.add(btn).clicked() {
                selected = idx;
            }
        }
    });

    ui.add(egui::Separator::default().horizontal().spacing(0.0));
    ui.add_space(4.0);

    egui::ScrollArea::vertical()
        .id_salt("bottom_strip_scroll")
        .show(ui, |ui| {
            if let Some(section) = panel.sections.get(selected) {
                if section.lines.is_empty() {
                    let empty_msg = match selected {
                        0 => "No open positions.",
                        1 => "No pending orders.",
                        2 => "No bot decisions recorded yet.",
                        3 => "No diagnostics available.",
                        _ => "No entries.",
                    };
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(empty_msg)
                                .color(theme::TEXT_MUTED)
                                .size(13.0),
                        );
                    });
                } else {
                    render_trade_watch_header(ui, section.title.as_str());
                    ui.add_space(2.0);
                    for (idx, line) in section.lines.iter().enumerate() {
                        let color = if selected == 1 {
                            theme::WARNING
                        } else if line.starts_with("ERROR") || line.starts_with("FAIL") {
                            theme::DANGER
                        } else if selected == 0 {
                            theme::SUCCESS
                        } else {
                            theme::TEXT_MUTED
                        };
                        let fill = if idx % 2 == 0 {
                            theme::SURFACE_BG
                        } else {
                            theme::PANEL_BG
                        };
                        egui::Frame::new()
                            .fill(fill)
                            .inner_margin(egui::Margin::symmetric(6, 3))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(
                                    egui::RichText::new(line)
                                        .monospace()
                                        .color(color)
                                        .size(12.0),
                                );
                            });
                    }
                }
            }
        });

    // Persist selected tab across frames.
    ui.data_mut(|d| d.insert_temp(tab_id, selected));
}

fn render_trade_watch_header(ui: &mut egui::Ui, title: &str) {
    let label = match title {
        "Positions" => "POSITION / SYMBOL / SIDE / SIZE / PNL",
        "Orders" => "ORDER / SYMBOL / TYPE / SIZE / PRICE",
        "Bot Log" => "AUTOMATION EVENT",
        "Diagnostics" => "RUNTIME DIAGNOSTIC",
        "Journal" => "JOURNAL ENTRY",
        _ => "ENTRY",
    };
    egui::Frame::new()
        .fill(theme::SURFACE_ALT)
        .inner_margin(egui::Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(label)
                    .monospace()
                    .size(10.0)
                    .strong()
                    .color(theme::TEXT_MUTED),
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_strip_uses_execution_snapshot_for_trade_watch_sections() {
        let snapshot = ExecutionSurfaceSnapshot {
            symbol: "EURUSD".to_string(),
            adapter_name: "cTrader".to_string(),
            integration_mode: "Remote Open API".to_string(),
            connection_status: "Connected".to_string(),
            supported_adapters: vec!["cTrader".to_string(), "DXtrade".to_string()],
            primary_actions: Vec::new(),
            warnings: Vec::new(),
            diagnostics: vec![
                "Adapter: cTrader".to_string(),
                "Market data capability: available".to_string(),
            ],
            positions: vec!["Open Position · EURUSD long · +24.5 pips".to_string()],
            pending_orders: vec!["EURUSD buy stop @ 1.10250".to_string()],
            bot_timeline: vec!["Bot entry approved · confidence 0.74".to_string()],
            history_rows: vec!["History row".to_string()],
            journal_rows: vec!["Journal row".to_string()],
            selected_position_id: Some(1),
            selected_order_id: Some(2),
            position_choices: vec![crate::app_services::trading::ExecutionSelectionOption {
                id: 1,
                label: "Position #1".to_string(),
            }],
            pending_order_choices: vec![crate::app_services::trading::ExecutionSelectionOption {
                id: 2,
                label: "Order #2".to_string(),
            }],
            ticket: crate::app_services::trading::ExecutionTicketSnapshot {
                lot_size: 0.10,
                slippage_in_points: 10,
                comment: String::new(),
                label: "manual".to_string(),
                max_lot_size: 10.0,
            },
        };

        let panel = build_bottom_strip(&snapshot);

        assert_eq!(panel.sections.len(), 5);
        assert_eq!(panel.sections[0].title, "Positions");
        assert!(
            panel.sections[0]
                .lines
                .iter()
                .any(|line| line.contains("Open Position"))
        );
        assert_eq!(panel.sections[1].title, "Orders");
        assert!(
            panel.sections[1]
                .lines
                .iter()
                .any(|line| line.contains("EURUSD buy stop"))
        );
        assert_eq!(panel.sections[2].title, "Bot Log");
        assert_eq!(
            panel.sections[2].lines,
            vec!["Bot entry approved · confidence 0.74".to_string()]
        );
        assert_eq!(panel.sections[3].title, "Diagnostics");
        assert!(
            panel.sections[3]
                .lines
                .iter()
                .any(|line| line.contains("Market data capability"))
        );
        assert_eq!(panel.sections[4].title, "Journal");
        assert!(
            panel.sections[4]
                .lines
                .iter()
                .any(|line| line.contains("History row"))
        );
    }
}
