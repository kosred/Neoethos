use crate::app_record;
use crate::app_state::{AppState, DataSource};
use crate::ui::components::open_log;
use eframe::egui;
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::SubsystemSection;
use mt5_bridge::MT5Engine;
use tracing::error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingPanelMode {
    LocalOnly,
    Disconnected,
    Connected,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DashboardCard {
    label: String,
    value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DashboardSection {
    title: String,
    rows: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ConnectionDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

pub fn panel_mode(data_source: DataSource, connected: bool) -> TradingPanelMode {
    match (data_source, connected) {
        (DataSource::Local, _) => TradingPanelMode::LocalOnly,
        (DataSource::MT5, false) => TradingPanelMode::Disconnected,
        (DataSource::MT5, true) => TradingPanelMode::Connected,
    }
}

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    mt5: &mut Option<MT5Engine>,
    terminal_info: &mut String,
) {
    ui.heading("Live Trading Terminal");
    ui.separator();

    let mode = panel_mode(state.data_source, mt5.is_some());
    render_connection_dashboard(ui, state, mode, terminal_info);

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Open Log").clicked() {
            if let Err(err) = open_log(&state.canonical_log_path) {
                state.status_msg = format!(
                    "Log open failed: {}",
                    err
                );
            }
        }
    });

    ui.separator();
    match mode {
        TradingPanelMode::LocalOnly => {
            ui.label("Live trading is disabled in Local mode.");
            ui.label("Please switch to MT5 source if you are on Windows.");
        }
        TradingPanelMode::Disconnected => {
            if ui.button("🚀 Connect to MetaTrader 5").clicked() {
                connect_mt5(state, mt5, terminal_info);
            }
        }
        TradingPanelMode::Connected => {
            ui.group(|ui| {
                ui.label("Account Details:");
                ui.label(terminal_info.as_str());
            });
            if ui.button("🛑 Disconnect").clicked() {
                *mt5 = None;
                state.status_msg = "Offline".to_string();
                record_app_event("ui_mt5_disconnect", "SUCCESS", "UI MT5 connection closed");
            }
        }
    }
}

fn render_connection_dashboard(
    ui: &mut egui::Ui,
    state: &AppState,
    mode: TradingPanelMode,
    terminal_info: &str,
) {
    let dashboard = build_connection_dashboard(state, mode, terminal_info);

    if !dashboard.summary_cards.is_empty() {
        ui.strong("Connection Center");
        ui.horizontal_wrapped(|ui| {
            for card in &dashboard.summary_cards {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_min_size(egui::vec2(155.0, 54.0));
                    ui.small(&card.label);
                    ui.strong(&card.value);
                });
            }
        });
    }

    if !dashboard.sections.is_empty() {
        ui.add_space(8.0);
        ui.columns(2, |columns| {
            for (idx, section) in dashboard.sections.iter().enumerate() {
                columns[idx % 2].group(|ui| {
                    ui.set_min_width(260.0);
                    ui.strong(&section.title);
                    ui.add_space(6.0);
                    egui::Grid::new(format!("connection_dashboard_{}_{}", section.title, idx))
                        .num_columns(2)
                        .spacing([12.0, 6.0])
                        .show(ui, |ui| {
                            for (label, value) in &section.rows {
                                ui.label(label);
                                ui.strong(value);
                                ui.end_row();
                            }
                        });
                });
            }
        });
    }
}

fn build_connection_dashboard(
    state: &AppState,
    mode: TradingPanelMode,
    terminal_info: &str,
) -> ConnectionDashboard {
    let source = match state.data_source {
        DataSource::MT5 => "MT5",
        DataSource::Local => "Local",
    };
    let mode_label = match mode {
        TradingPanelMode::LocalOnly => "LocalOnly",
        TradingPanelMode::Disconnected => "Disconnected",
        TradingPanelMode::Connected => "Connected",
    };
    let capability = match mode {
        TradingPanelMode::LocalOnly => "Discovery/Training",
        TradingPanelMode::Disconnected => "Connect Required",
        TradingPanelMode::Connected => "Live Trading Ready",
    };

    let summary_cards = vec![
        DashboardCard {
            label: "Source".to_string(),
            value: source.to_string(),
        },
        DashboardCard {
            label: "Mode".to_string(),
            value: mode_label.to_string(),
        },
        DashboardCard {
            label: "Status".to_string(),
            value: state.status_msg.clone(),
        },
        DashboardCard {
            label: "Capabilities".to_string(),
            value: capability.to_string(),
        },
    ];

    let mut sections = Vec::new();
    sections.push(DashboardSection {
        title: "Runtime".to_string(),
        rows: vec![
            ("Data Source".to_string(), source.to_string()),
            (
                "Config".to_string(),
                state.runtime.config_path.clone(),
            ),
            (
                "Data Root".to_string(),
                state.runtime.data_dir.display().to_string(),
            ),
            ("Selected Pair".to_string(), state.selected_pair.clone()),
        ],
    });

    match mode {
        TradingPanelMode::LocalOnly => sections.push(DashboardSection {
            title: "Guidance".to_string(),
            rows: vec![
                (
                    "Mode".to_string(),
                    "Local runtime is active".to_string(),
                ),
                (
                    "Live Execution".to_string(),
                    "Disabled until MT5 source is selected".to_string(),
                ),
                (
                    "Available Operations".to_string(),
                    "Discovery, training, audit, local data prep".to_string(),
                ),
            ],
        }),
        TradingPanelMode::Disconnected => sections.push(DashboardSection {
            title: "Connection Guidance".to_string(),
            rows: vec![
                (
                    "Bridge".to_string(),
                    "MetaTrader5 Python bridge must initialize".to_string(),
                ),
                (
                    "Terminal".to_string(),
                    "Desktop terminal and account auth must be available".to_string(),
                ),
                (
                    "Next Action".to_string(),
                    "Use Connect to probe the MT5 runtime".to_string(),
                ),
            ],
        }),
        TradingPanelMode::Connected => {
            let terminal_summary = if terminal_info.trim().is_empty() {
                "Connected but terminal info is empty".to_string()
            } else {
                terminal_info
                    .lines()
                    .next()
                    .unwrap_or(terminal_info)
                    .trim()
                    .to_string()
            };
            sections.push(DashboardSection {
                title: "Terminal Snapshot".to_string(),
                rows: vec![
                    ("Session".to_string(), "Connected".to_string()),
                    ("Terminal".to_string(), terminal_summary),
                    (
                        "Available Operations".to_string(),
                        "Disconnect, monitor broker state, inspect logs".to_string(),
                    ),
                ],
            });
        }
    }

    ConnectionDashboard {
        summary_cards,
        sections,
    }
}

fn connect_mt5(state: &mut AppState, mt5: &mut Option<MT5Engine>, terminal_info: &mut String) {
    match MT5Engine::new() {
        Ok(mut engine) => match engine.initialize() {
            Ok(true) => {
                state.status_msg = "Connected".to_string();
                *terminal_info = engine.terminal_info().unwrap_or_default();
                *mt5 = Some(engine);
                record_app_event("ui_mt5_connect", "SUCCESS", "UI MT5 connection succeeded");
            }
            _ => {
                state.status_msg =
                    "Connection Failed (module missing or terminal closed)".to_string();
                record_app_event(
                    "ui_mt5_connect",
                    "DEGRADED",
                    "UI MT5 connection failed (module missing or terminal closed)",
                );
            }
        },
        Err(err) => {
            state.status_msg = format!("Error: {:?}", err);
            record_app_event("ui_mt5_connect", "FAILED", format!("UI MT5 bridge error: {err}"));
        }
    }
}

fn record_app_event(operation: &str, status: &str, message: impl Into<String>) {
    if let Err(err) = write_subsystem_record(
        SubsystemSection::App,
        app_record(operation, status, message),
    ) {
        error!("Failed to write APP section log: {}", err);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppRuntimeConfig, Tab};
    use std::path::PathBuf;

    fn sample_state(source: DataSource, status_msg: &str) -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: matches!(source, DataSource::Local),
            },
            current_tab: Tab::Trading,
            data_source: source,
            status_msg: status_msg.to_string(),
            selected_pair: "EURUSD".to_string(),
            available_symbols: vec!["EURUSD".to_string()],
            discovery_job: None,
            training_job: None,
            canonical_log_path: PathBuf::from("logs").join("forex-ai.log"),
            hardware: crate::app_state::HardwareState::default(),
            risk: crate::app_state::RiskState::default(),
        }
    }

    #[test]
    fn connection_dashboard_describes_local_mode_operations() {
        let state = sample_state(DataSource::Local, "Local Mode");

        let dashboard = build_connection_dashboard(&state, TradingPanelMode::LocalOnly, "");

        assert_eq!(dashboard.summary_cards[0].value, "Local");
        assert_eq!(dashboard.summary_cards[1].value, "LocalOnly");
        assert_eq!(dashboard.summary_cards[3].value, "Discovery/Training");
        assert_eq!(dashboard.sections[0].title, "Runtime");
        assert_eq!(dashboard.sections[1].title, "Guidance");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Live Execution"
                && value == "Disabled until MT5 source is selected"));
    }

    #[test]
    fn connection_dashboard_surfaces_connected_terminal_snapshot() {
        let state = sample_state(DataSource::MT5, "Connected");

        let dashboard = build_connection_dashboard(
            &state,
            TradingPanelMode::Connected,
            "TerminalInfo(community_account=False, connected=True)",
        );

        assert_eq!(dashboard.summary_cards[0].value, "MT5");
        assert_eq!(dashboard.summary_cards[1].value, "Connected");
        assert_eq!(dashboard.summary_cards[3].value, "Live Trading Ready");
        assert_eq!(dashboard.sections[1].title, "Terminal Snapshot");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Terminal"
                && value.contains("TerminalInfo")));
    }
}

