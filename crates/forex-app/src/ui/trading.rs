use crate::app_services::trading::{
    ConnectionSnapshot, TradingPanelMode, TradingSession, SUPPORTED_TRADING_ADAPTERS,
};
use crate::app_state::{AppState, DataSource};
use crate::ui::components::open_log;
use eframe::egui;

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

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
) {
    ui.heading("Live Trading Terminal");
    ui.separator();

    let snapshot = session.snapshot(state);
    render_connection_dashboard(ui, state, &snapshot);

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
    match snapshot.mode {
        TradingPanelMode::LocalOnly => {
            ui.label("Live trading is disabled in Local mode.");
            ui.label("Please switch to MT5 source if you are on Windows.");
        }
        TradingPanelMode::Disconnected => {
            if ui.button("🚀 Connect to MetaTrader 5").clicked() {
                session.connect(state);
            }
        }
        TradingPanelMode::Connected => {
            ui.group(|ui| {
                ui.label("Account Details:");
                ui.label(snapshot.terminal_info.as_str());
            });
            if ui.button("🛑 Disconnect").clicked() {
                session.disconnect(state);
            }
        }
    }
}

fn render_connection_dashboard(
    ui: &mut egui::Ui,
    state: &AppState,
    snapshot: &ConnectionSnapshot,
) {
    let dashboard = build_connection_dashboard(state, snapshot);

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
    snapshot: &ConnectionSnapshot,
) -> ConnectionDashboard {
    let supported_adapters = SUPPORTED_TRADING_ADAPTERS
        .iter()
        .map(|kind| kind.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let source = match state.data_source {
        DataSource::MT5 => "MT5",
        DataSource::Local => "Local",
    };
    let mode_label = match snapshot.mode {
        TradingPanelMode::LocalOnly => "LocalOnly",
        TradingPanelMode::Disconnected => "Disconnected",
        TradingPanelMode::Connected => "Connected",
    };
    let capability = match snapshot.mode {
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
            label: "Adapter".to_string(),
            value: snapshot.adapter_name.clone(),
        },
        DashboardCard {
            label: "Mode".to_string(),
            value: mode_label.to_string(),
        },
        DashboardCard {
            label: "Status".to_string(),
            value: snapshot.status_text.clone(),
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
            ("Adapter".to_string(), snapshot.adapter_name.clone()),
            (
                "Integration".to_string(),
                snapshot.integration_mode.clone(),
            ),
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

    match snapshot.mode {
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
                    snapshot.integration_mode.clone(),
                ),
                (
                    "Local Terminal".to_string(),
                    if snapshot.requires_local_terminal {
                        "Required".to_string()
                    } else {
                        "Not required".to_string()
                    },
                ),
                (
                    "Next Action".to_string(),
                    format!("Use Connect to probe the {} runtime", snapshot.adapter_name),
                ),
                (
                    "Available Adapters".to_string(),
                    supported_adapters,
                ),
            ],
        }),
        TradingPanelMode::Connected => {
            let terminal_summary = if snapshot.terminal_info.trim().is_empty() {
                "Connected but terminal info is empty".to_string()
            } else {
                snapshot
                    .terminal_info
                    .lines()
                    .next()
                    .unwrap_or(snapshot.terminal_info.as_str())
                    .trim()
                    .to_string()
            };
            sections.push(DashboardSection {
                title: "Terminal Snapshot".to_string(),
                rows: vec![
                    ("Session".to_string(), "Connected".to_string()),
                    ("Adapter".to_string(), snapshot.adapter_name.clone()),
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
        let snapshot = ConnectionSnapshot {
            adapter_name: "MT5".to_string(),
            integration_mode: "Local terminal bridge".to_string(),
            requires_local_terminal: true,
            supports_market_data: true,
            supports_live_orders: true,
            mode: TradingPanelMode::LocalOnly,
            connected: false,
            status_text: "Local Mode".to_string(),
            terminal_info: String::new(),
        };

        let dashboard = build_connection_dashboard(&state, &snapshot);

        assert_eq!(dashboard.summary_cards[0].value, "Local");
        assert_eq!(dashboard.summary_cards[1].value, "MT5");
        assert_eq!(dashboard.summary_cards[2].value, "LocalOnly");
        assert_eq!(dashboard.summary_cards[4].value, "Discovery/Training");
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
        let snapshot = ConnectionSnapshot {
            adapter_name: "MT5".to_string(),
            integration_mode: "Local terminal bridge".to_string(),
            requires_local_terminal: true,
            supports_market_data: true,
            supports_live_orders: true,
            mode: TradingPanelMode::Connected,
            connected: true,
            status_text: "Connected".to_string(),
            terminal_info: "TerminalInfo(community_account=False, connected=True)".to_string(),
        };

        let dashboard = build_connection_dashboard(&state, &snapshot);

        assert_eq!(dashboard.summary_cards[0].value, "MT5");
        assert_eq!(dashboard.summary_cards[1].value, "MT5");
        assert_eq!(dashboard.summary_cards[2].value, "Connected");
        assert_eq!(dashboard.summary_cards[4].value, "Live Trading Ready");
        assert_eq!(dashboard.sections[1].title, "Terminal Snapshot");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Terminal"
                && value.contains("TerminalInfo")));
    }

    #[test]
    fn connection_dashboard_lists_supported_adapters_for_disconnected_runtime() {
        let state = sample_state(DataSource::MT5, "Offline");
        let snapshot = ConnectionSnapshot {
            adapter_name: "MT5".to_string(),
            integration_mode: "Local terminal bridge".to_string(),
            requires_local_terminal: true,
            supports_market_data: true,
            supports_live_orders: true,
            mode: TradingPanelMode::Disconnected,
            connected: false,
            status_text: "Offline".to_string(),
            terminal_info: String::new(),
        };

        let dashboard = build_connection_dashboard(&state, &snapshot);

        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Available Adapters"
                && value.contains("MT5")
                && value.contains("cTrader")
                && value.contains("DXtrade")));
    }
}

