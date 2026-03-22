use crate::app_state::{AppState, DataSource};
use crate::app_services::broker_config::{AdapterReadinessSnapshot, BrokerAccountTarget};
use crate::app_services::ctrader_auth::CTraderAuthSnapshot;
use crate::app_services::trading::{TradingSession, SUPPORTED_TRADING_ADAPTERS};
use crate::ui::components::{
    render_dashboard_sections, render_summary_cards, render_view_header, DashboardCard,
    DashboardSection,
};
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SystemStatusDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &mut TradingSession) -> bool {
    let snapshot = session.snapshot(state);
    let readiness = session.adapter_readiness();
    let ctrader_auth = session.ctrader_auth_snapshot();
    let dashboard = build_system_status_dashboard(state, &snapshot, &readiness, ctrader_auth.as_ref());
    let mut refresh_requested = false;

    render_view_header(
        ui,
        "System Status",
        "Control the active runtime source and inspect the local or broker-backed operating envelope.",
    );
    ui.separator();

    ui.label("Data Source:");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.data_source, DataSource::MT5, "MT5");
        ui.selectable_value(&mut state.data_source, DataSource::Local, "Local");
    });
    ui.add_space(8.0);
    ui.label("Broker Adapter:");
    ui.horizontal_wrapped(|ui| {
        for adapter in SUPPORTED_TRADING_ADAPTERS {
            let selected = session.configured_adapter() == adapter;
            if ui.selectable_label(selected, adapter.as_str()).clicked() {
                session.select_adapter(state, adapter);
            }
        }
    });
    ui.add_space(8.0);

    render_adapter_configuration(ui, session);
    ui.add_space(8.0);

    render_summary_cards(ui, "Runtime Snapshot", &dashboard.summary_cards);
    render_dashboard_sections(ui, "system_status_section", &dashboard.sections);

    if ui.button("🔄 Refresh Data").clicked() {
        refresh_requested = true;
    }

    refresh_requested
}

fn build_system_status_dashboard(
    state: &AppState,
    connection: &crate::app_services::trading::ConnectionSnapshot,
    readiness: &AdapterReadinessSnapshot,
    ctrader_auth: Option<&CTraderAuthSnapshot>,
) -> SystemStatusDashboard {
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
            (
                "Adapter".to_string(),
                connection.adapter_name.clone(),
            ),
            (
                "Integration".to_string(),
                connection.integration_mode.clone(),
            ),
            (
                "Readiness".to_string(),
                readiness.status_line.clone(),
            ),
            (
                "Execution Targets".to_string(),
                format!("{} enabled", readiness.target_count),
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
                (
                    "Armed Broker Adapter".to_string(),
                    connection.adapter_name.clone(),
                ),
            ],
        }),
        DataSource::MT5 => sections.push(DashboardSection {
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
                (
                    "Adapter".to_string(),
                    connection.adapter_name.clone(),
                ),
                (
                    "Bridge".to_string(),
                    connection.integration_mode.clone(),
                ),
                (
                    "Guidance".to_string(),
                    if connection.connected {
                        "Broker runtime is available".to_string()
                    } else if connection.requires_local_terminal {
                        "Use the Trading tab to connect and inspect terminal state".to_string()
                    } else {
                        "Remote adapter selected; runtime contract is staged but not wired yet".to_string()
                    },
                ),
            ],
        }),
    }

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
                ("Accounts".to_string(), auth.account_count.to_string()),
            ],
        });
    }

    SystemStatusDashboard {
        summary_cards,
        sections,
    }
}

fn render_adapter_configuration(ui: &mut egui::Ui, session: &mut TradingSession) {
    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.strong("Adapter Configuration");
        ui.add_space(6.0);

        match session.configured_adapter() {
            crate::app_services::trading::TradingAdapterKind::Mt5 => {
                let settings = &mut session.broker_settings_mut().mt5;
                labeled_text_edit(ui, "Terminal Path", &mut settings.terminal_path);
                labeled_text_edit(ui, "Server", &mut settings.server);
                labeled_text_edit(ui, "Login", &mut settings.login);
                render_account_targets(ui, &mut settings.accounts, "MT5 Account");
            }
            crate::app_services::trading::TradingAdapterKind::CTrader => {
                let mut start_auth = false;
                let mut accept_code = false;
                let mut prepare_token_request = false;
                let code_to_accept = {
                    let settings = &mut session.broker_settings_mut().ctrader;
                    labeled_text_edit(ui, "Client ID", &mut settings.client_id);
                    labeled_text_edit(ui, "Client Secret", &mut settings.client_secret);
                    labeled_text_edit(ui, "Redirect URI", &mut settings.redirect_uri);
                    labeled_text_edit(
                        ui,
                        "Authorization Code",
                        &mut settings.authorization_code_input,
                    );
                    let code = settings.authorization_code_input.clone();
                    ui.horizontal(|ui| {
                        if ui.button("Start cTrader Auth").clicked() {
                            start_auth = true;
                        }
                        if ui.button("Accept Code").clicked() && !code.trim().is_empty() {
                            accept_code = true;
                        }
                        if ui.button("Prepare Token Request").clicked() {
                            prepare_token_request = true;
                        }
                    });
                    render_account_targets(ui, &mut settings.accounts, "cTrader Account");
                    code
                };

                if start_auth {
                    let _ = session.start_ctrader_auth();
                }
                if accept_code {
                    session.receive_ctrader_authorization_code(code_to_accept);
                }
                if prepare_token_request {
                    let _ = session.build_ctrader_token_exchange_request();
                }
            }
            crate::app_services::trading::TradingAdapterKind::DxTrade => {
                let settings = &mut session.broker_settings_mut().dxtrade;
                labeled_text_edit(ui, "Platform URL", &mut settings.platform_url);
                labeled_text_edit(ui, "Username", &mut settings.username);
                labeled_text_edit(ui, "Password", &mut settings.password);
                render_account_targets(ui, &mut settings.accounts, "DXtrade Account");
            }
        }
    });
}

fn labeled_text_edit(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add_sized(
            [ui.available_width().max(200.0), 24.0],
            egui::TextEdit::singleline(value),
        );
    });
}

fn render_account_targets(
    ui: &mut egui::Ui,
    accounts: &mut Vec<BrokerAccountTarget>,
    default_prefix: &str,
) {
    ui.add_space(6.0);
    ui.strong("Execution Targets");
    for (idx, account) in accounts.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.checkbox(&mut account.enabled_for_execution, "");
            ui.label(format!("Target {}", idx + 1));
            ui.add_sized([120.0, 24.0], egui::TextEdit::singleline(&mut account.account_id));
            ui.add_sized([160.0, 24.0], egui::TextEdit::singleline(&mut account.label));
        });
    }
    if ui.button("+ Add Account Target").clicked() {
        let next = accounts.len() + 1;
        accounts.push(BrokerAccountTarget {
            account_id: String::new(),
            label: format!("{default_prefix} {next}"),
            enabled_for_execution: false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::{TradingAdapterKind, TradingSession};
    use crate::app_state::{AppRuntimeConfig, HardwareState, RiskState};
    use std::path::PathBuf;

    fn sample_state(source: DataSource, status_msg: &str) -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: matches!(source, DataSource::Local),
            },
            data_source: source,
            status_msg: status_msg.to_string(),
            selected_pair: "EURUSD".to_string(),
            chart_timeframe: "M1".to_string(),
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
        let session = TradingSession::new();
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert_eq!(dashboard.summary_cards[0].value, "Local");
        assert_eq!(dashboard.summary_cards[1].value, "Local Runtime");
        assert_eq!(dashboard.summary_cards[2].value, "Local Mode");
        assert_eq!(dashboard.summary_cards[3].value, "MT5");
        assert_eq!(dashboard.summary_cards[5].value, "2");
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
        let session = TradingSession::from_connected_terminal_for_test("TerminalInfo(connected=True)");
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert_eq!(dashboard.summary_cards[0].value, "MT5");
        assert_eq!(dashboard.summary_cards[1].value, "Broker Runtime");
        assert_eq!(dashboard.summary_cards[2].value, "Connected");
        assert_eq!(dashboard.summary_cards[3].value, "MT5");
        assert_eq!(dashboard.summary_cards[4].value, "MT5 connected.");
        assert_eq!(dashboard.sections[1].title, "Broker Status");
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Connection" && value == "Online"));
    }

    #[test]
    fn system_status_dashboard_surfaces_selected_remote_adapter_metadata() {
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::new();
        session.select_adapter(&mut state, TradingAdapterKind::CTrader);
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert_eq!(dashboard.summary_cards[3].value, "cTrader");
        assert!(dashboard.summary_cards[4]
            .value
            .contains("configuration incomplete"));
        assert!(dashboard.sections[0]
            .rows
            .iter()
            .any(|(label, value)| label == "Integration" && value == "Remote Open API"));
        assert!(dashboard.sections[1]
            .rows
            .iter()
            .any(|(label, value)| label == "Guidance"
                && value.contains("Remote adapter selected")));
    }

    #[test]
    fn system_status_dashboard_surfaces_remote_readiness_and_target_counts() {
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::new();
        session.select_adapter(&mut state, TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session
            .broker_settings_mut()
            .ctrader
            .accounts
            .push(crate::app_services::broker_config::BrokerAccountTarget {
                account_id: "acct-1".to_string(),
                label: "Primary".to_string(),
                enabled_for_execution: true,
            });
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert!(dashboard.sections.iter().any(|section| {
            section.rows.iter().any(|(label, value)| {
                label == "Readiness" && value.contains("OAuth app credentials ready")
            })
        }));
        assert!(dashboard.sections.iter().any(|section| {
            section
                .rows
                .iter()
                .any(|(label, value)| label == "Execution Targets" && value == "1 enabled")
        }));
    }

    #[test]
    fn system_status_dashboard_surfaces_ctrader_auth_state() {
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::new();
        session.select_adapter(&mut state, TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.start_ctrader_auth().expect("auth start");
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert!(dashboard.sections.iter().any(|section| {
            section.title == "cTrader Auth"
                && section.rows.iter().any(|(label, value)| {
                    label == "State" && value == "AwaitingAuthorizationCode"
                })
        }));
    }

    #[test]
    fn system_status_dashboard_surfaces_ctrader_received_code_and_accounts() {
        let mut state = sample_state(DataSource::MT5, "Offline");
        let mut session = TradingSession::new();
        session.select_adapter(&mut state, TradingAdapterKind::CTrader);
        session.broker_settings_mut().ctrader.client_id = "client".to_string();
        session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
        session.broker_settings_mut().ctrader.redirect_uri =
            "http://localhost:3000/callback".to_string();
        session.start_ctrader_auth().expect("auth start");
        session.receive_ctrader_authorization_code("code-123");
        let connection = session.snapshot(&state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        let dashboard =
            build_system_status_dashboard(&state, &connection, &readiness, ctrader_auth.as_ref());

        assert!(dashboard.sections.iter().any(|section| {
            section.title == "cTrader Auth"
                && section.rows.iter().any(|(label, value)| {
                    label == "Authorization Code" && value == "Received"
                })
        }));
    }
}
