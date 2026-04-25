use crate::app_services::trading::{ExecutionSurfaceSnapshot, TradingSession};
use crate::app_state::{AppState, OrderType};
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
    pub history_rows: Vec<String>,
    pub journal_rows: Vec<String>,
    pub connect_enabled: bool,
    pub connect_reason: Option<String>,
}

pub fn build_execution_panel(
    snapshot: &ExecutionSurfaceSnapshot,
    connect_enabled: bool,
    connect_reason: Option<String>,
) -> ExecutionPanel {
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
        history_rows: snapshot.history_rows.clone(),
        journal_rows: snapshot.journal_rows.clone(),
        connect_enabled,
        connect_reason,
    }
}

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    let snapshot = session.execution_surface_snapshot(state);
    let readiness = session.adapter_readiness();
    let connect_reason = (!session.is_connected()).then(|| readiness.status_line.clone());
    let panel = build_execution_panel(&snapshot, session.can_attempt_connect(), connect_reason);

    ui.strong(format!("Execution · {}", panel.symbol));
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new(format!("Adapter: {}", panel.adapter_name)).color(theme::TEXT_MUTED),
    );
    ui.label(
        egui::RichText::new(format!("Integration: {}", panel.integration_mode))
            .color(theme::TEXT_MUTED),
    );
    ui.label(
        egui::RichText::new(format!("Status: {}", panel.connection_status))
            .color(theme::TEXT_MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "Supported: {}",
            panel.supported_adapters.join(", ")
        ))
        .color(theme::TEXT_MUTED),
    );
    ui.add_space(8.0);

    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.strong("Order Ticket");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut state.order_ticket.order_type,
                OrderType::Market,
                "Market",
            );
            ui.selectable_value(
                &mut state.order_ticket.order_type,
                OrderType::Limit,
                "Limit",
            );
            ui.selectable_value(&mut state.order_ticket.order_type, OrderType::Stop, "Stop");
        });
        ui.add_space(4.0);

        if state.order_ticket.order_type != OrderType::Market {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Target Price").color(theme::TEXT_MUTED));
                ui.add(egui::DragValue::new(&mut state.order_ticket.target_price).speed(0.0001));
            });
        }

        ui.horizontal(|ui| {
            ui.checkbox(
                &mut state.order_ticket.auto_lot_sizing,
                "Auto Sizing (Risk %)",
            );
        });
        if state.order_ticket.auto_lot_sizing {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Risk %").color(theme::TEXT_MUTED));
                ui.add(
                    egui::DragValue::new(&mut state.order_ticket.auto_risk_pct)
                        .speed(0.1)
                        .range(0.1..=10.0),
                );
                ui.label(egui::RichText::new("SL Pips").color(theme::TEXT_MUTED));
                ui.add(
                    egui::DragValue::new(&mut state.order_ticket.stop_loss_pips)
                        .speed(1.0)
                        .range(1.0..=500.0),
                );
            });
            let equity = if state.account_equity > 0.0 {
                state.account_equity
            } else {
                state.account_balance
            };
            let calculated_lots = if equity > 0.0 && state.order_ticket.stop_loss_pips > 0.0 {
                let risk_amount = equity * (state.order_ticket.auto_risk_pct / 100.0);
                (risk_amount / (state.order_ticket.stop_loss_pips * 10.0))
                    .clamp(0.01, state.risk.max_lot_size)
            } else {
                0.01
            };
            ui.label(
                egui::RichText::new(format!("Calculated Size: {:.2} Lots", calculated_lots))
                    .color(theme::WARNING),
            );
            state.order_ticket.lot_size = calculated_lots;
        } else {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Lot Size").color(theme::TEXT_MUTED));
                ui.add(
                    egui::DragValue::new(&mut state.order_ticket.lot_size)
                        .range(0.01..=state.risk.max_lot_size)
                        .speed(0.01),
                );
                ui.label(
                    egui::RichText::new(format!("Max {:.2}", snapshot.ticket.max_lot_size))
                        .color(theme::TEXT_MUTED),
                );
            });
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Slippage").color(theme::TEXT_MUTED));
            ui.add(egui::DragValue::new(&mut state.order_ticket.slippage_in_points).range(0..=500));
            ui.label(egui::RichText::new("points").color(theme::TEXT_MUTED));
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Label").color(theme::TEXT_MUTED));
            ui.text_edit_singleline(&mut state.order_ticket.label);
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Comment").color(theme::TEXT_MUTED));
            ui.text_edit_singleline(&mut state.order_ticket.comment);
        });
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut state.order_ticket.smart_sl_enabled,
                egui::RichText::new("Smart SL/TP (ATR)").color(theme::TEXT_MUTED),
            );
            if state.order_ticket.smart_sl_enabled {
                ui.label(egui::RichText::new("RR:").color(theme::TEXT_MUTED));
                ui.add(
                    egui::DragValue::new(&mut state.order_ticket.smart_rr_ratio)
                        .speed(0.1)
                        .range(0.5..=10.0),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.order_ticket.trailing_stop, "Trailing Stop");
        });
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

    if !snapshot.position_choices.is_empty() {
        egui::ComboBox::from_label("Selected Position")
            .selected_text(
                state
                    .order_ticket
                    .selected_position_id
                    .map(|id| format!("#{id}"))
                    .unwrap_or_else(|| "Choose position".to_string()),
            )
            .show_ui(ui, |ui| {
                for option in &snapshot.position_choices {
                    ui.selectable_value(
                        &mut state.order_ticket.selected_position_id,
                        Some(option.id),
                        &option.label,
                    );
                }
            });
    }

    if !snapshot.pending_order_choices.is_empty() {
        egui::ComboBox::from_label("Selected Pending Order")
            .selected_text(
                state
                    .order_ticket
                    .selected_order_id
                    .map(|id| format!("#{id}"))
                    .unwrap_or_else(|| "Choose pending order".to_string()),
            )
            .show_ui(ui, |ui| {
                for option in &snapshot.pending_order_choices {
                    ui.selectable_value(
                        &mut state.order_ticket.selected_order_id,
                        Some(option.id),
                        &option.label,
                    );
                }
            });
    }

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        let action_reason = snapshot
            .primary_actions
            .first()
            .and_then(|action| (!action.enabled).then(|| action.reason.clone()))
            .flatten();
        let buy = ui.add_enabled(
            snapshot
                .primary_actions
                .first()
                .map(|action| action.enabled)
                .unwrap_or(false),
            egui::Button::new(
                egui::RichText::new(if state.order_ticket.order_type == OrderType::Market {
                    "Buy Market"
                } else {
                    "Place Buy Order"
                })
                .color(theme::TEXT_PRIMARY),
            )
            .fill(theme::SUCCESS.linear_multiply(0.45))
            .min_size(egui::vec2(ui.available_width() / 2.0 - 4.0, 34.0)),
        );
        if let Some(reason) = &action_reason {
            if buy.hovered() {
                buy.clone().on_hover_text(reason);
            }
        }
        if buy.clicked() {
            session.execute_buy_market(state);
        }

        let sell = ui.add_enabled(
            snapshot
                .primary_actions
                .get(1)
                .map(|action| action.enabled)
                .unwrap_or(false),
            egui::Button::new(
                egui::RichText::new(if state.order_ticket.order_type == OrderType::Market {
                    "Sell Market"
                } else {
                    "Place Sell Order"
                })
                .color(theme::TEXT_PRIMARY),
            )
            .fill(theme::DANGER.linear_multiply(0.45))
            .min_size(egui::vec2(ui.available_width(), 34.0)),
        );
        if let Some(reason) = &action_reason {
            if sell.hovered() {
                sell.clone().on_hover_text(reason);
            }
        }
        if sell.clicked() {
            session.execute_sell_market(state);
        }
    });

    ui.horizontal(|ui| {
        let cancel_enabled = session.is_connected()
            && !snapshot.pending_order_choices.is_empty()
            && snapshot.adapter_name == "cTrader";
        if ui
            .add_enabled(cancel_enabled, egui::Button::new("Cancel Selected"))
            .clicked()
        {
            session.cancel_selected_order(state);
        }
        let close_enabled = session.is_connected()
            && !snapshot.position_choices.is_empty()
            && snapshot.adapter_name == "cTrader";
        if ui
            .add_enabled(close_enabled, egui::Button::new("Close Selected"))
            .clicked()
        {
            session.close_selected_position(state);
        }
    });

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
        ui.label(
            egui::RichText::new("Execution remains disabled in Local mode.").color(theme::WARNING),
        );
    }
    if snapshot.connection_status != "Local Mode" {
        if session.is_connected() {
            if ui.button("Disconnect Runtime").clicked() {
                session.disconnect(state);
            }
        } else {
            let mut response =
                ui.add_enabled(panel.connect_enabled, egui::Button::new("Connect Runtime"));
            if let Some(reason) = &panel.connect_reason {
                if response.hovered() {
                    response = response.on_hover_text(reason);
                }
            }
            if response.clicked() {
                let _ = session.start_connect(tx.clone());
            }
        }
    }

    if ui.button("Open Log").clicked() {
        if let Err(err) = open_log(&state.canonical_log_path) {
            state.status_msg = format!("Log open failed: {}", err);
        }
    }

    if !snapshot.positions.is_empty() {
        ui.add_space(8.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Positions");
            ui.add_space(6.0);
            for row in &snapshot.positions {
                ui.label(egui::RichText::new(row).color(theme::TEXT_MUTED));
            }
        });
    }

    if !snapshot.pending_orders.is_empty() {
        ui.add_space(8.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Orders");
            ui.add_space(6.0);
            for row in &snapshot.pending_orders {
                ui.label(egui::RichText::new(row).color(theme::TEXT_MUTED));
            }
        });
    }

    if !panel.history_rows.is_empty() {
        ui.add_space(8.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("History");
            ui.add_space(6.0);
            for row in &panel.history_rows {
                ui.label(egui::RichText::new(row).color(theme::TEXT_MUTED));
            }
        });
    }

    if !panel.journal_rows.is_empty() {
        ui.add_space(8.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Journal");
            ui.add_space(6.0);
            for row in &panel.journal_rows {
                ui.label(egui::RichText::new(row).color(theme::TEXT_MUTED));
            }
        });
    }
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
        };
        AppState::new(
            runtime,
            &forex_core::Settings::default(),
            vec!["EURUSD".to_string()],
        )
    }

    #[test]
    fn execution_panel_surfaces_primary_actions_runtime_summary_and_warnings() {
        let state = sample_state();
        let mut session = TradingSession::new();
        let snapshot = session.execution_surface_snapshot(&state);
        let panel = build_execution_panel(&snapshot, true, None);

        assert!(panel.primary_actions.contains(&"Buy".to_string()));
        assert!(panel.primary_actions.contains(&"Sell".to_string()));
        assert!(panel.supported_adapters.contains(&"cTrader".to_string()));
        assert_eq!(panel.connection_status, "Offline");
        assert!(
            panel
                .diagnostics
                .iter()
                .any(|line| line.contains("central broker background loop"))
        );
        assert!(panel.history_rows.is_empty());
        assert!(panel.journal_rows.is_empty());
    }

    #[test]
    fn execution_panel_disables_connect_until_remote_credentials_are_ready() {
        let state = sample_state();
        let mut session =
            crate::app_services::trading::TradingSession::with_configured_adapter_for_test(
                crate::app_services::trading::TradingAdapterKind::CTrader,
            );
        let snapshot = session.execution_surface_snapshot(&state);
        let readiness = session.adapter_readiness();
        let panel = build_execution_panel(
            &snapshot,
            readiness.can_attempt_connect,
            Some(readiness.status_line.clone()),
        );

        assert!(!panel.connect_enabled);
        assert_eq!(
            panel.connect_reason.as_deref(),
            Some(
                "cTrader configuration incomplete: missing client_id, client_secret, redirect_uri"
            )
        );
    }
}
