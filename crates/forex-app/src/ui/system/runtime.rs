use crate::app_services::broker_config::AdapterReadinessSnapshot;
use crate::app_services::ctrader_auth::CTraderAuthSnapshot;
use crate::app_services::trading::{ConnectionSnapshot, TradingSession};
use crate::app_state::{AppState, DataSource};
use crate::ui::components::{
    render_dashboard_sections, render_summary_cards, render_view_header, DashboardCard,
    DashboardSection,
};
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RuntimeDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &mut TradingSession) -> bool {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let _ = session.poll_ctrader_live_auth();
        let snapshot = session.snapshot(state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();
        let dashboard = build_runtime_dashboard(state, &snapshot, &readiness, ctrader_auth.as_ref());
        let mut refresh_requested = false;

        render_view_header(
            ui,
            "Runtime",
            "Inspect the active runtime envelope, operator state, and authenticated broker session health.",
        );
        ui.separator();

        render_summary_cards(ui, "Runtime Snapshot", &dashboard.summary_cards);
        render_dashboard_sections(ui, "runtime_section", &dashboard.sections);

        ui.add_space(12.0);
        if ui.button("Refresh Runtime Data").clicked() {
            refresh_requested = true;
        }

        refresh_requested
    })
    .inner
}

fn build_runtime_dashboard(
    state: &AppState,
    connection: &ConnectionSnapshot,
    readiness: &AdapterReadinessSnapshot,
    ctrader_auth: Option<&CTraderAuthSnapshot>,
) -> RuntimeDashboard {
    let source = match state.data_source {
        DataSource::CTrader => "cTrader",
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
            label: "Adapter".to_string(),
            value: connection.adapter_name.clone(),
        },
        DashboardCard {
            label: "Readiness".to_string(),
            value: readiness.status_line.clone(),
        },
        DashboardCard {
            label: "Symbols".to_string(),
            value: state.available_symbols.len().to_string(),
        },
    ];

    let mut sections = vec![DashboardSection {
        title: "Runtime".to_string(),
        rows: vec![
            ("Config".to_string(), state.runtime.config_path.clone()),
            (
                "Data Root".to_string(),
                state.runtime.data_dir.display().to_string(),
            ),
            ("Selected Pair".to_string(), state.selected_pair.clone()),
            (
                "Selected Timeframe".to_string(),
                state.chart_timeframe.clone(),
            ),
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
            ("Adapter".to_string(), connection.adapter_name.clone()),
            (
                "Integration".to_string(),
                connection.integration_mode.clone(),
            ),
            ("Readiness".to_string(), readiness.status_line.clone()),
            (
                "Execution Targets".to_string(),
                format!("{} enabled", readiness.target_count),
            ),
            (
                "Account Balance".to_string(),
                if state.account_balance > 0.0 {
                    format!("${:.2}", state.account_balance)
                } else {
                    "Unavailable".to_string()
                },
            ),
            (
                "Account Equity".to_string(),
                if state.account_equity > 0.0 {
                    format!("${:.2}", state.account_equity)
                } else {
                    "Unavailable".to_string()
                },
            ),
        ],
    }];

    sections.push(match state.data_source {
        DataSource::Local => DashboardSection {
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
                ("Broker Dependency".to_string(), "None required".to_string()),
                (
                    "Armed Broker Adapter".to_string(),
                    connection.adapter_name.clone(),
                ),
            ],
        },
        DataSource::CTrader => DashboardSection {
            title: "Broker Status".to_string(),
            rows: vec![
                (
                    "Connection".to_string(),
                    if connection.connected {
                        "Online".to_string()
                    } else {
                        "Offline".to_string()
                    },
                ),
                ("Adapter".to_string(), connection.adapter_name.clone()),
                ("Bridge".to_string(), connection.integration_mode.clone()),
                (
                    "Guidance".to_string(),
                    if connection.connected {
                        "cTrader runtime is available".to_string()
                    } else {
                        "Use Broker Setup to restore cTrader auth and connect the selected account"
                            .to_string()
                    },
                ),
            ],
        },
        DataSource::MT5 => DashboardSection {
            title: "Broker Status".to_string(),
            rows: vec![
                (
                    "Connection".to_string(),
                    if connection.connected {
                        "Online".to_string()
                    } else {
                        "Offline".to_string()
                    },
                ),
                ("Adapter".to_string(), connection.adapter_name.clone()),
                ("Bridge".to_string(), connection.integration_mode.clone()),
                (
                    "Guidance".to_string(),
                    if connection.connected {
                        "Broker runtime is available".to_string()
                    } else if connection.requires_local_terminal {
                        "Use the broker page to connect and inspect terminal state".to_string()
                    } else {
                        "Remote adapter selected; runtime contract is staged or partially wired"
                            .to_string()
                    },
                ),
            ],
        },
    });

    if let Some(auth) = ctrader_auth {
        sections.push(DashboardSection {
            title: "cTrader Auth".to_string(),
            rows: vec![
                ("State".to_string(), format!("{:?}", auth.state)),
                ("Status".to_string(), auth.status_line.clone()),
                (
                    "Authorize URL".to_string(),
                    if auth.authorize_url.is_some() {
                        "Ready".to_string()
                    } else {
                        "Unavailable".to_string()
                    },
                ),
                (
                    "Authorization Code".to_string(),
                    if auth.authorization_code_present {
                        "Received".to_string()
                    } else {
                        "Missing".to_string()
                    },
                ),
                (
                    "Token Request".to_string(),
                    if auth.token_request_ready {
                        "Ready".to_string()
                    } else {
                        "Not ready".to_string()
                    },
                ),
                (
                    "Callback Port".to_string(),
                    auth.callback_port
                        .map(|port| port.to_string())
                        .unwrap_or_else(|| "Unassigned".to_string()),
                ),
                ("Persistence".to_string(), auth.persistence_status.clone()),
                ("Accounts".to_string(), auth.account_count.to_string()),
            ],
        });
    }

    RuntimeDashboard {
        summary_cards,
        sections,
    }
}
