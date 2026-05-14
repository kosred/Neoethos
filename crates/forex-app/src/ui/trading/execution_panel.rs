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

    render_execution_header(ui, &panel);
    ui.add_space(4.0);
    render_buy_sell(ui, state, session, &snapshot, &panel);

    ui.add_space(4.0);

    egui::ScrollArea::vertical()
        .id_salt("execution_scroll")
        .show(ui, |ui| {
            theme::section_frame(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(egui::RichText::new("Order Ticket").size(12.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("via {}", panel.integration_mode))
                                .size(10.5)
                                .color(theme::TEXT_MUTED),
                        );
                    });
                });
                ui.add_space(4.0);
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
                    ui.selectable_value(
                        &mut state.order_ticket.order_type,
                        OrderType::Stop,
                        "Stop",
                    );
                });
                ui.add_space(4.0);

                if state.order_ticket.order_type != OrderType::Market {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Price").color(theme::TEXT_MUTED));
                        ui.add(
                            egui::DragValue::new(&mut state.order_ticket.target_price)
                                .speed(0.0001),
                        );
                    });
                }

                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut state.order_ticket.auto_lot_sizing,
                        egui::RichText::new("Auto Lot (Risk %)").color(theme::TEXT_MUTED),
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
                        ui.label(egui::RichText::new("SL pips").color(theme::TEXT_MUTED));
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
                    let calculated_lots = if equity > 0.0 && state.order_ticket.stop_loss_pips > 0.0
                    {
                        let risk_amount = equity * (state.order_ticket.auto_risk_pct / 100.0);
                        (risk_amount / (state.order_ticket.stop_loss_pips * 10.0))
                            .clamp(0.01, state.risk.max_lot_size)
                    } else {
                        0.01
                    };
                    ui.label(
                        egui::RichText::new(format!("≈ {:.2} lots", calculated_lots))
                            .color(theme::WARNING),
                    );
                    state.order_ticket.lot_size = calculated_lots;
                } else {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Lots").color(theme::TEXT_MUTED));
                        ui.add(
                            egui::DragValue::new(&mut state.order_ticket.lot_size)
                                .range(0.01..=state.risk.max_lot_size)
                                .speed(0.01),
                        );
                        ui.label(
                            egui::RichText::new(format!("max {:.2}", snapshot.ticket.max_lot_size))
                                .size(11.0)
                                .color(theme::TEXT_MUTED),
                        );
                    });
                }
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Slip").color(theme::TEXT_MUTED));
                    ui.add(
                        egui::DragValue::new(&mut state.order_ticket.slippage_in_points)
                            .range(0..=500),
                    );
                    ui.label(egui::RichText::new("pts").color(theme::TEXT_MUTED));
                });
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Label").color(theme::TEXT_MUTED));
                    ui.text_edit_singleline(&mut state.order_ticket.label);
                });
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut state.order_ticket.smart_sl_enabled,
                        egui::RichText::new("ATR SL/TP").color(theme::TEXT_MUTED),
                    );
                    if state.order_ticket.smart_sl_enabled {
                        ui.label(egui::RichText::new("RR").color(theme::TEXT_MUTED));
                        ui.add(
                            egui::DragValue::new(&mut state.order_ticket.smart_rr_ratio)
                                .speed(0.1)
                                .range(0.5..=10.0),
                        );
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut state.order_ticket.trailing_stop, "Trail Stop");
                });
            });

            ui.add_space(4.0);

            // Position / order selectors
            if !snapshot.position_choices.is_empty() {
                egui::ComboBox::from_label("Position")
                    .selected_text(
                        state
                            .order_ticket
                            .selected_position_id
                            .map(|id| format!("#{id}"))
                            .unwrap_or_else(|| "Choose…".to_string()),
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
                egui::ComboBox::from_label("Pending Order")
                    .selected_text(
                        state
                            .order_ticket
                            .selected_order_id
                            .map(|id| format!("#{id}"))
                            .unwrap_or_else(|| "Choose…".to_string()),
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

            if !snapshot.position_choices.is_empty() || !snapshot.pending_order_choices.is_empty() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let cancel_enabled = session.is_connected()
                        && !snapshot.pending_order_choices.is_empty()
                        && snapshot.adapter_name == "cTrader";
                    if ui
                        .add_enabled(cancel_enabled, egui::Button::new("Cancel Order"))
                        .clicked()
                    {
                        session.cancel_selected_order(state);
                    }
                    let close_enabled = session.is_connected()
                        && !snapshot.position_choices.is_empty()
                        && snapshot.adapter_name == "cTrader";
                    if ui
                        .add_enabled(close_enabled, egui::Button::new("Close Position"))
                        .clicked()
                    {
                        session.close_selected_position(state);
                    }
                });
            }

            for warning in &panel.warnings {
                ui.label(
                    egui::RichText::new(warning)
                        .color(theme::WARNING)
                        .size(12.0),
                );
            }

            ui.add_space(4.0);
            if snapshot.connection_status == "Local Mode" {
                ui.label(
                    egui::RichText::new("Execution disabled in Local mode.")
                        .color(theme::WARNING)
                        .size(12.0),
                );
            } else if session.is_connected() {
                if ui.button("Disconnect").clicked() {
                    session.disconnect(state);
                }
            } else {
                let mut response = ui.add_enabled(
                    panel.connect_enabled,
                    egui::Button::new("Connect to Broker"),
                );
                if let Some(reason) = &panel.connect_reason
                    && response.hovered()
                {
                    response = response.on_hover_text(reason);
                }
                if response.clicked() {
                    if let Err(err) = session.start_connect(tx.clone()) {
                        tracing::warn!(
                            target: "forex_app::ui::trading::execution_panel",
                            error = %err,
                            "Connect to Broker click: session.start_connect failed"
                        );
                        state.status_msg = format!("Connect failed: {err}");
                    }
                }
            }

            if ui
                .button(
                    egui::RichText::new("Open Log")
                        .color(theme::TEXT_MUTED)
                        .size(11.0),
                )
                .clicked()
                && let Err(err) = open_log(&state.canonical_log_path)
            {
                state.status_msg = format!("Log open failed: {}", err);
            }

            if !snapshot.positions.is_empty() {
                ui.add_space(4.0);
                theme::section_frame(ui.style()).show(ui, |ui| {
                    ui.strong(egui::RichText::new("Positions").size(12.0));
                    ui.add_space(2.0);
                    for row in &snapshot.positions {
                        ui.label(egui::RichText::new(row).color(theme::SUCCESS).size(12.0));
                    }
                });
            }

            if !snapshot.pending_orders.is_empty() {
                ui.add_space(4.0);
                theme::section_frame(ui.style()).show(ui, |ui| {
                    ui.strong(egui::RichText::new("Orders").size(12.0));
                    ui.add_space(2.0);
                    for row in &snapshot.pending_orders {
                        ui.label(egui::RichText::new(row).color(theme::WARNING).size(12.0));
                    }
                });
            }

            if !panel.diagnostics.is_empty() {
                ui.add_space(4.0);
                theme::section_frame(ui.style()).show(ui, |ui| {
                    ui.strong(egui::RichText::new("Diagnostics").size(12.0));
                    ui.add_space(2.0);
                    for line in &panel.diagnostics {
                        ui.label(
                            egui::RichText::new(line)
                                .color(theme::TEXT_MUTED)
                                .size(11.0),
                        );
                    }
                });
            }
        });
}

fn render_execution_header(ui: &mut egui::Ui, panel: &ExecutionPanel) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.strong(
                egui::RichText::new(&panel.symbol)
                    .size(14.0)
                    .color(theme::TEXT_PRIMARY),
            );
            ui.label(
                egui::RichText::new(&panel.adapter_name)
                    .size(10.5)
                    .color(theme::TEXT_MUTED),
            );
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (badge_text, col) = if panel.connection_status.contains("Connected")
                || panel.connection_status.contains("Online")
            {
                ("LIVE", theme::SUCCESS)
            } else if panel.connection_status == "Local Mode" {
                ("LOCAL", theme::WARNING)
            } else {
                ("OFFLINE", theme::DANGER)
            };
            theme::status_badge(ui, badge_text, col);
        });
    });
}

fn render_buy_sell(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    snapshot: &ExecutionSurfaceSnapshot,
    panel: &ExecutionPanel,
) {
    let action_reason = snapshot
        .primary_actions
        .first()
        .and_then(|action| (!action.enabled).then(|| action.reason.clone()))
        .flatten();

    let button_h = 42.0;
    let half_w = (ui.available_width() - 4.0) / 2.0;

    ui.horizontal(|ui| {
        let sell_enabled = snapshot
            .primary_actions
            .get(1)
            .map(|a| a.enabled)
            .unwrap_or(false);
        let sell_label = if state.order_ticket.order_type == OrderType::Market {
            "SELL"
        } else {
            "SELL LIMIT"
        };
        let sell = ui.add_enabled(
            sell_enabled,
            egui::Button::new(
                egui::RichText::new(sell_label)
                    .strong()
                    .size(14.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(theme::DANGER.linear_multiply(if sell_enabled { 0.82 } else { 0.25 }))
            .min_size(egui::vec2(half_w, button_h)),
        );
        if let Some(reason) = &action_reason
            && sell.hovered()
        {
            sell.clone().on_hover_text(reason);
        }
        if sell.clicked() {
            session.execute_sell_market(state);
        }

        let buy_enabled = snapshot
            .primary_actions
            .first()
            .map(|a| a.enabled)
            .unwrap_or(false);
        let buy_label = if state.order_ticket.order_type == OrderType::Market {
            "BUY"
        } else {
            "BUY LIMIT"
        };
        let buy = ui.add_enabled(
            buy_enabled,
            egui::Button::new(
                egui::RichText::new(buy_label)
                    .strong()
                    .size(14.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(theme::SUCCESS.linear_multiply(if buy_enabled { 0.82 } else { 0.25 }))
            .min_size(egui::vec2(ui.available_width(), button_h)),
        );
        if let Some(reason) = &action_reason
            && buy.hovered()
        {
            buy.clone().on_hover_text(reason);
        }
        if buy.clicked() {
            session.execute_buy_market(state);
        }
    });

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!(
                "{:.2} lots  ·  slip {:.0}pts",
                state.order_ticket.lot_size, state.order_ticket.slippage_in_points
            ))
            .size(11.0)
            .color(theme::TEXT_MUTED),
        );
        if !panel.connection_status.contains("Local") && !session.is_connected() {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new("Not connected")
                        .size(11.0)
                        .color(theme::DANGER),
                );
            });
        }
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
