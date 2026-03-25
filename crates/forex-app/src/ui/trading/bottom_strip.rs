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
                title: "Positions / Orders / PnL".to_string(),
                lines: snapshot
                    .positions
                    .iter()
                    .cloned()
                    .chain(
                        snapshot
                            .pending_orders
                            .iter()
                            .map(|order| format!("Pending · {order}")),
                    )
                    .collect(),
            },
            BottomStripSection {
                title: "Bot Decisions Timeline".to_string(),
                lines: snapshot.bot_timeline.clone(),
            },
            BottomStripSection {
                title: "Execution Diagnostics".to_string(),
                lines: snapshot.diagnostics.clone(),
            },
            BottomStripSection {
                title: "Manual Notes".to_string(),
                lines: vec![
                    format!("Symbol focus: {}", snapshot.symbol),
                    "Operator notes are local-only until workspace persistence lands.".to_string(),
                ],
            },
        ],
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState, session: &mut TradingSession) {
    let snapshot = session.execution_surface_snapshot(state);
    let panel = build_bottom_strip(&snapshot);

    ui.columns(panel.sections.len(), |columns| {
        for (idx, section) in panel.sections.iter().enumerate() {
            theme::section_frame(columns[idx].style()).show(&mut columns[idx], |ui| {
                ui.strong(&section.title);
                ui.add_space(8.0);
                if section.lines.is_empty() {
                    ui.label(
                        egui::RichText::new(match section.title.as_str() {
                            "Positions / Orders / PnL" => {
                                "No broker positions or pending orders are available yet."
                            }
                            "Bot Decisions Timeline" => {
                                "No bot execution decisions are available yet."
                            }
                            _ => "No entries available.",
                        })
                        .color(theme::TEXT_MUTED),
                    );
                } else {
                    for line in &section.lines {
                        ui.label(egui::RichText::new(line).color(theme::TEXT_MUTED));
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_strip_uses_execution_snapshot_for_sections_and_notes() {
        let snapshot = ExecutionSurfaceSnapshot {
            symbol: "EURUSD".to_string(),
            adapter_name: "MT5".to_string(),
            integration_mode: "Local terminal bridge".to_string(),
            connection_status: "Connected".to_string(),
            supported_adapters: vec!["MT5".to_string(), "cTrader".to_string()],
            primary_actions: Vec::new(),
            warnings: Vec::new(),
            diagnostics: vec![
                "Adapter: MT5".to_string(),
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

        assert_eq!(panel.sections.len(), 4);
        assert_eq!(panel.sections[0].title, "Positions / Orders / PnL");
        assert!(panel.sections[0]
            .lines
            .iter()
            .any(|line| line.contains("Open Position")));
        assert!(panel.sections[0]
            .lines
            .iter()
            .any(|line| line.contains("Pending")));
        assert_eq!(panel.sections[1].title, "Bot Decisions Timeline");
        assert_eq!(
            panel.sections[1].lines,
            vec!["Bot entry approved · confidence 0.74".to_string()]
        );
        assert_eq!(panel.sections[2].title, "Execution Diagnostics");
        assert!(panel.sections[2]
            .lines
            .iter()
            .any(|line| line.contains("Market data capability")));
        assert_eq!(panel.sections[3].title, "Manual Notes");
        assert!(panel.sections[3]
            .lines
            .iter()
            .any(|line| line.contains("Symbol focus: EURUSD")));
    }
}
