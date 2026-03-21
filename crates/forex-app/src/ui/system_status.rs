use crate::app_state::{AppState, DataSource};
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
struct SystemStatusDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, connected: bool) -> bool {
    let dashboard = build_system_status_dashboard(state, connected);
    let mut refresh_requested = false;

    ui.heading("System Status");
    ui.separator();

    ui.label("Data Source:");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.data_source, DataSource::MT5, "MT5");
        ui.selectable_value(&mut state.data_source, DataSource::Local, "Local");
    });

    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        for card in &dashboard.summary_cards {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_size(egui::vec2(132.0, 52.0));
                ui.small(&card.label);
                ui.strong(&card.value);
            });
        }
    });

    if !dashboard.sections.is_empty() {
        ui.add_space(8.0);
        for (idx, section) in dashboard.sections.iter().enumerate() {
            ui.group(|ui| {
                ui.strong(&section.title);
                ui.add_space(6.0);
                egui::Grid::new(format!("system_status_section_{}", idx))
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
            ui.add_space(6.0);
        }
    }

    if ui.button("🔄 Refresh Data").clicked() {
        refresh_requested = true;
    }

    refresh_requested
}

fn build_system_status_dashboard(state: &AppState, connected: bool) -> SystemStatusDashboard {
    let source = match state.data_source {
        DataSource::MT5 => "MT5",
        DataSource::Local => "Local",
    };
    let runtime_mode = if matches!(state.data_source, DataSource::Local) {
        "Local Runtime"
    } else {
        "Broker Runtime"
    };

    let summary_cards = vec![
        DashboardCard {
            label: "Source".to_string(),
            value: source.to_string(),
        },
        DashboardCard {
            label: "Mode".to_string(),
            value: runtime_mode.to_string(),
        },
        DashboardCard {
            label: "Status".to_string(),
            value: state.status_msg.clone(),
        },
        DashboardCard {
            label: "Symbols".to_string(),
            value: state.available_symbols.len().to_string(),
        },
    ];

    let mut sections = vec![DashboardSection {
        title: "Runtime".to_string(),
        rows: vec![
            (
                "Config".to_string(),
                state.runtime.config_path.clone(),
            ),
            (
                "Data Root".to_string(),
                state.runtime.data_dir.display().to_string(),
            ),
            ("Selected Pair".to_string(), state.selected_pair.clone()),
            (
                "CPU Cores".to_string(),
                state.hardware.cpu_cores.to_string(),
            ),
            (
                "GPU".to_string(),
                if state.hardware.gpu_enabled {
                    "Enabled".to_string()
                } else {
                    "Disabled".to_string()
                },
            ),
        ],
    }];

    match state.data_source {
        DataSource::Local => sections.push(DashboardSection {
            title: "Capabilities".to_string(),
            rows: vec![
                (
                    "Live Trading".to_string(),
                    "Disabled in Local mode".to_string(),
                ),
                (
                    "Primary Use".to_string(),
                    "Discovery, training, and local diagnostics".to_string(),
                ),
                (
                    "Broker Dependency".to_string(),
                    "None required".to_string(),
                ),
            ],
        }),
        DataSource::MT5 => sections.push(DashboardSection {
            title: "Broker Status".to_string(),
            rows: vec![
                (
                    "Connection".to_string(),
                    if connected {
                        "Online".to_string()
                    } else {
                        "Offline".to_string()
                    },
                ),
                (
                    "Bridge".to_string(),
                    "MetaTrader5 Python bridge".to_string(),
                ),
                (
                    "Guidance".to_string(),
                    if connected {
                        "Broker runtime is available".to_string()
                    } else {
                        "Use the Trading tab to connect and inspect terminal state".to_string()
                    },
                ),
            ],
        }),
    }

    SystemStatusDashboard {
        summary_cards,
        sections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppRuntimeConfig, HardwareState, RiskState, Tab};
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
            available_symbols: vec!["EURUSD".to_string(), "GBPUSD".to_string()],
            discovery_job: None,
            training_job: None,
            canonical_log_path: PathBuf::from("logs").join("forex-ai.log"),
            hardware: HardwareState::default(),
            risk: RiskState::default(),
        }
    }

    #[test]
    fn system_status_dashboard_describes_local_runtime_capabilities() {
        let state = sample_state(DataSource::Local, "Local Mode");

        let dashboard = build_system_status_dashboard(&state, false);

        assert_eq!(dashboard.summary_cards[0].value, "Local");
        assert_eq!(dashboard.summary_cards[1].value, "Local Runtime");
        assert_eq!(dashboard.summary_cards[2].value, "Local Mode");
        assert_eq!(dashboard.summary_cards[3].value, "2");
        assert_eq!(dashboard.sections[0].title, "Runtime");
        assert_eq!(dashboard.sections[1].title, "Capabilities");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Live Trading"
                && value == "Disabled in Local mode"));
    }

    #[test]
    fn system_status_dashboard_surfaces_mt5_connectivity_summary() {
        let state = sample_state(DataSource::MT5, "Connected");

        let dashboard = build_system_status_dashboard(&state, true);

        assert_eq!(dashboard.summary_cards[0].value, "MT5");
        assert_eq!(dashboard.summary_cards[1].value, "Broker Runtime");
        assert_eq!(dashboard.summary_cards[2].value, "Connected");
        assert_eq!(dashboard.sections[1].title, "Broker Status");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Connection" && value == "Online"));
    }
}
