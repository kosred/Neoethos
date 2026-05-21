use crate::app_services::broker_config::CTraderBrokerEnvironment;
use crate::app_services::ctrader_auth::CTraderAuthSnapshot;
use crate::app_services::trading::{
    SUPPORTED_TRADING_ADAPTERS, TradingAdapterKind, TradingSession,
};
use crate::app_state::{AppState, DataSource};
use crate::ui::components::{DashboardCard, render_summary_cards, render_view_header};
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
        if let Some(snapshot) = session.poll_ctrader_live_auth() {
            state.status_msg = snapshot.status_line;
        }
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
            render_adapter_configuration(ui, state, session, tx);
        });
    });
}

fn ctrader_result_status<T>(
    result: anyhow::Result<T>,
    success_status: impl FnOnce(T) -> String,
    failure_prefix: &str,
) -> String {
    match result {
        Ok(value) => success_status(value),
        Err(err) => format!("{failure_prefix}: {err}"),
    }
}

fn ctrader_optional_snapshot_status(
    snapshot: Option<CTraderAuthSnapshot>,
    fallback: &str,
) -> String {
    snapshot
        .map(|snapshot| snapshot.status_line)
        .unwrap_or_else(|| fallback.to_string())
}

fn render_adapter_configuration(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    match session.configured_adapter() {
        TradingAdapterKind::CTrader => {
            let mut start_live_auth = false;
            let mut discover_accounts = false;
            let mut restore_saved_session = false;
            let mut clear_saved_session = false;
            let mut start_auth = false;
            let mut accept_code = false;
            let mut prepare_token_request = false;
            let mut create_demo_account = false;
            let mut create_live_account = false;
            let mut save_credentials = false;
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
                    if ui
                        .button("Save Credentials to Disk")
                        .on_hover_text(
                            "Persists Client ID / Secret / Redirect URI / Environment / accounts to the broker_credentials.toml file so they auto-load next launch. Transient fields (auth code, DxTrade password) are never saved.",
                        )
                        .clicked()
                    {
                        save_credentials = true;
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
                        create_demo_account = true;
                    }
                    if ui.button("Create Live Account").clicked() {
                        create_live_account = true;
                    }
                });
                render_account_targets(ui, &mut settings.accounts, "cTrader Account");
                settings.authorization_code_input.clone()
            };

            if start_live_auth {
                state.status_msg = ctrader_result_status(
                    session.start_ctrader_live_auth(tx.clone()),
                    |snapshot| snapshot.status_line,
                    "cTrader login failed",
                );
            }
            if discover_accounts {
                state.status_msg = ctrader_result_status(
                    session.discover_ctrader_accounts(),
                    |snapshot| {
                        ctrader_optional_snapshot_status(
                            snapshot,
                            "No cTrader account discovery snapshot returned.",
                        )
                    },
                    "cTrader account discovery failed",
                );
            }
            if restore_saved_session {
                state.status_msg = ctrader_result_status(
                    session.restore_ctrader_session(),
                    |snapshot| {
                        ctrader_optional_snapshot_status(
                            snapshot,
                            "No saved cTrader session found.",
                        )
                    },
                    "cTrader session restore failed",
                );
            }
            if clear_saved_session {
                state.status_msg = ctrader_result_status(
                    session.clear_ctrader_saved_session(),
                    |_| "cTrader saved session cleared.".to_string(),
                    "cTrader session clear failed",
                );
            }
            if start_auth {
                state.status_msg = ctrader_result_status(
                    session.start_ctrader_auth(),
                    |snapshot| snapshot.status_line,
                    "cTrader auth setup failed",
                );
            }
            if accept_code {
                let snapshot = session.receive_ctrader_authorization_code(code_to_accept);
                state.status_msg = snapshot.status_line;
            }
            if prepare_token_request {
                state.status_msg = ctrader_result_status(
                    session.build_ctrader_token_exchange_request(),
                    |_| {
                        session
                            .ctrader_auth_snapshot()
                            .map(|snapshot| snapshot.status_line)
                            .unwrap_or_else(|| "cTrader token request is ready.".to_string())
                    },
                    "cTrader token request failed",
                );
            }
            if create_demo_account {
                state.status_msg = match open::that(
                    crate::app_services::broker_config::CTRADER_CREATE_DEMO_ACCOUNT_URL,
                ) {
                    Ok(()) => "Opened cTrader demo account page.".to_string(),
                    Err(err) => format!("Failed to open cTrader demo account page: {err}"),
                };
            }
            if create_live_account {
                state.status_msg = match open::that(
                    crate::app_services::broker_config::CTRADER_CREATE_LIVE_ACCOUNT_URL,
                ) {
                    Ok(()) => "Opened cTrader live account page.".to_string(),
                    Err(err) => format!("Failed to open cTrader live account page: {err}"),
                };
            }
            if save_credentials {
                let settings_snapshot = session.broker_settings_mut().clone();
                state.status_msg =
                    match crate::app_services::broker_persistence::save_broker_settings(
                        &settings_snapshot,
                    ) {
                        Ok(()) => "Broker credentials saved to disk.".to_string(),
                        Err(err) => format!("Failed to save broker credentials: {err}"),
                    };
            }
        }
        TradingAdapterKind::DxTrade => {
            let settings = &mut session.broker_settings_mut().dxtrade;
            labeled_text_edit(ui, "Platform URL", &mut settings.platform_url);
            labeled_text_edit(ui, "Username", &mut settings.username);
            // Phase D3.1 (2026-05-18): the official DXtrade
            // `POST /dxsca-web/login` endpoint requires a `domain`
            // field alongside username + password. Brokers
            // typically configure this to a fixed string per
            // environment (often `default`); the wizard surfaces
            // it as an editable row so the operator can match
            // whatever their broker assigned.
            labeled_text_edit(ui, "Domain", &mut settings.domain);
            labeled_text_edit(ui, "Password", &mut settings.password);
            render_account_targets(ui, &mut settings.accounts, "DXtrade Account");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrader_result_status_includes_error_details() {
        let status = ctrader_result_status::<()>(
            Err(anyhow::anyhow!("INVALID_CLIENT")),
            |_| "ok".to_string(),
            "cTrader login failed",
        );

        assert_eq!(status, "cTrader login failed: INVALID_CLIENT");
    }
}
