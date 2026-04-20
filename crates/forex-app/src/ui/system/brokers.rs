use crate::app_services::broker_config::CTraderBrokerEnvironment;
use crate::app_services::trading::{
    TradingAdapterKind, TradingSession, SUPPORTED_TRADING_ADAPTERS,
};
use crate::app_state::{AppState, DataSource};
use crate::ui::components::{render_summary_cards, render_view_header, DashboardCard};
use crate::ui::system::shared::{labeled_text_edit, render_account_targets};
use crate::ui::theme;
use eframe::egui;

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let _ = session.poll_ctrader_live_auth();
        let connection = session.snapshot(state);
        let readiness = session.adapter_readiness();
        let ctrader_auth = session.ctrader_auth_snapshot();

        render_view_header(
            ui,
            "Broker Setup",
            "Keep broker wiring, account targets, credentials, and auth flows isolated from the rest of the operator workspace.",
        );
        ui.separator();

        let mut summary_cards = vec![
            DashboardCard {
                label: "Data Source".to_string(),
                value: match state.data_source {
                    DataSource::CTrader => "cTrader".to_string(),
                    DataSource::MT5 => "MT5".to_string(),
                    DataSource::Local => "Local".to_string(),
                },
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
                label: "Integration".to_string(),
                value: connection.integration_mode.clone(),
            },
            DashboardCard {
                label: "Targets".to_string(),
                value: readiness.target_count.to_string(),
            },
        ];
        if let Some(auth) = ctrader_auth.as_ref() {
            summary_cards.push(DashboardCard {
                label: "cTrader Auth".to_string(),
                value: auth.status_line.clone(),
            });
        }
        render_summary_cards(ui, "Broker Snapshot", &summary_cards);

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Runtime Source");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.data_source, DataSource::CTrader, "cTrader");
                #[cfg(feature = "legacy-mt5")]
                ui.selectable_value(&mut state.data_source, DataSource::MT5, "MT5 Legacy");
                ui.selectable_value(&mut state.data_source, DataSource::Local, "Local");
            });

            ui.add_space(8.0);
            ui.strong("Active Broker Adapter");
            ui.horizontal_wrapped(|ui| {
                for adapter in SUPPORTED_TRADING_ADAPTERS {
                    let selected = session.configured_adapter() == adapter;
                    if ui.selectable_label(selected, adapter.as_str()).clicked() {
                        session.select_adapter(state, adapter);
                    }
                }
            });
        });

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Adapter Configuration");
            ui.add_space(6.0);
            render_adapter_configuration(ui, session, tx);
        });
    });
}

fn render_adapter_configuration(
    ui: &mut egui::Ui,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    match session.configured_adapter() {
        TradingAdapterKind::Mt5 => {
            #[cfg(not(feature = "legacy-mt5"))]
            {
                ui.label("Legacy MT5 bridge is disabled in the default Rust/cTrader runtime.");
                return;
            }
            #[cfg(feature = "legacy-mt5")]
            {
                let settings = &mut session.broker_settings_mut().mt5;
                labeled_text_edit(ui, "Terminal Path", &mut settings.terminal_path);
                labeled_text_edit(ui, "Server", &mut settings.server);
                labeled_text_edit(ui, "Login", &mut settings.login);
                render_account_targets(ui, &mut settings.accounts, "MT5 Account");
            }
        }
        TradingAdapterKind::CTrader => {
            let mut start_live_auth = false;
            let mut discover_accounts = false;
            let mut restore_saved_session = false;
            let mut clear_saved_session = false;
            let mut start_auth = false;
            let mut accept_code = false;
            let mut prepare_token_request = false;
            let code_to_accept = {
                let settings = &mut session.broker_settings_mut().ctrader;
                labeled_text_edit(ui, "Client ID", &mut settings.client_id);
                labeled_text_edit(ui, "Client Secret", &mut settings.client_secret);
                labeled_text_edit(ui, "Redirect URI", &mut settings.redirect_uri);
                ui.horizontal(|ui| {
                    ui.label("Environment");
                    ui.selectable_value(
                        &mut settings.environment,
                        CTraderBrokerEnvironment::Live,
                        "Live",
                    );
                    ui.selectable_value(
                        &mut settings.environment,
                        CTraderBrokerEnvironment::Demo,
                        "Demo",
                    );
                });
                ui.label(format!(
                    "Current cTrader environment: {}",
                    settings.environment.as_str()
                ));

                ui.horizontal_wrapped(|ui| {
                    if ui.button("Start cTrader Login (Automatic)").clicked() {
                        start_live_auth = true;
                    }
                    if ui.button("Start cTrader Auth").clicked() {
                        start_auth = true;
                    }
                    if ui.button("Prepare Token Request").clicked() {
                        prepare_token_request = true;
                    }
                });

                ui.add_space(4.0);
                ui.label(
                    "If automatic transfer fails, copy the 'code' parameter from the browser URL and paste it below:",
                );
                labeled_text_edit(ui, "Manual Code", &mut settings.authorization_code_input);
                if ui.button("Accept Code").clicked()
                    && !settings.authorization_code_input.trim().is_empty()
                {
                    accept_code = true;
                }

                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Discover Accounts").clicked() {
                        discover_accounts = true;
                    }
                    if ui.button("Restore Saved Session").clicked() {
                        restore_saved_session = true;
                    }
                    if ui.button("Clear Saved Session").clicked() {
                        clear_saved_session = true;
                    }
                });

                ui.add_space(6.0);
                ui.label("Account Management");
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Create Demo Account").clicked() {
                        let _ = open::that("https://app.ctrader.com/accounts/create-demo");
                    }
                    if ui.button("Create Live Account").clicked() {
                        let _ = open::that("https://app.ctrader.com/accounts/create-live");
                    }
                });
                render_account_targets(ui, &mut settings.accounts, "cTrader Account");
                settings.authorization_code_input.clone()
            };

            if start_live_auth {
                let _ = session.start_ctrader_live_auth(tx.clone());
            }
            if discover_accounts {
                let _ = session.discover_ctrader_accounts();
            }
            if restore_saved_session {
                let _ = session.restore_ctrader_session();
            }
            if clear_saved_session {
                let _ = session.clear_ctrader_saved_session();
            }
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
        TradingAdapterKind::DxTrade => {
            let settings = &mut session.broker_settings_mut().dxtrade;
            labeled_text_edit(ui, "Platform URL", &mut settings.platform_url);
            labeled_text_edit(ui, "Username", &mut settings.username);
            labeled_text_edit(ui, "Password", &mut settings.password);
            render_account_targets(ui, &mut settings.accounts, "DXtrade Account");
        }
    }
}
