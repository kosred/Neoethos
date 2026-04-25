mod app_services;
mod app_state;
mod ui;
mod workspace;

use crate::ui::components::render_ribbon_item;
use app_services::{
    discovery::DiscoveryJobHandle, trading::TradingSession, training::TrainingJobHandle,
    ServiceEvent,
};
use app_state::{AppRuntimeConfig, AppState};
use clap::Parser;
use eframe::egui;
use forex_core::logging::{setup_logging, write_subsystem_record};
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use forex_core::Settings;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{error, info};
use workspace::{render_workspace, WorkspaceState, WorkspaceViewer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    headless: bool,

    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long, default_value_t = false)]
    local: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    setup_logging(true)?;
    let settings = Settings::from_yaml(&args.config)?;
    let runtime = AppRuntimeConfig::from_settings(args.config.clone(), args.local, &settings);
    write_subsystem_record(
        SubsystemSection::App,
        app_record(
            "app_startup",
            "STARTED",
            format!(
                "starting app headless={} local={} config={}",
                args.headless, args.local, args.config
            ),
        ),
    )?;

    if args.headless {
        info!("Starting Forex AI in Headless Server Mode...");
        run_headless_loop(runtime).await;
        Ok(())
    } else {
        info!("Starting Forex AI in GUI Mode...");
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
            ..Default::default()
        };

        eframe::run_native(
            "Forex AI - Pure Rust Terminal",
            options,
            Box::new(|cc| {
                Ok(Box::new(ForexApp::new(
                    cc,
                    runtime.clone(),
                    settings.clone(),
                )))
            }),
        )?;
        Ok(())
    }
}

async fn run_headless_loop(runtime: AppRuntimeConfig) {
    info!("Loading configuration from: {}", runtime.config_path);

    if runtime.start_local {
        info!("Running in Pure Local Mode (Linux Server Discovery/Training).");
        let symbols = forex_data::discover_symbols(&runtime.data_dir).unwrap_or_default();
        info!(
            "Successfully mapped {} local symbols in '{}' directory.",
            symbols.len(),
            runtime.data_dir.display()
        );
        if let Err(err) = write_subsystem_record(
            SubsystemSection::App,
            app_record(
                "headless_local_start",
                "SUCCESS",
                format!(
                    "mapped {} symbols from {}",
                    symbols.len(),
                    runtime.data_dir.display()
                ),
            ),
        ) {
            error!("Failed to write APP section log: {}", err);
        }
        for sym in &symbols {
            info!("  - Symbol available: {}", sym);
        }

        info!("Headless engine is now active and monitoring background tasks.");
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            info!(
                "Headless keep-alive tick: System Health OK | Cores: {} | Mode: LOCAL",
                num_cpus::get()
            );
        }
    } else {
        info!("Running in cTrader-first Headless Broker Mode.");
        if let Err(err) = write_subsystem_record(
            SubsystemSection::App,
            app_record(
                "headless_ctrader_start",
                "READY",
                "cTrader is the canonical broker runtime; auth/connect are managed by app services",
            ),
        ) {
            error!("Failed to write APP section log: {}", err);
        }
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            info!(
                "Headless keep-alive tick: System Health OK | Cores: {} | Mode: CTRADER",
                num_cpus::get()
            );
        }
    }
}

struct ForexApp {
    trading_session: TradingSession,
    workspace: WorkspaceState,
    state: AppState,

    // Message Bus
    tx: mpsc::Sender<ServiceEvent>,
    rx: mpsc::Receiver<ServiceEvent>,
    discovery_handle: Option<DiscoveryJobHandle>,
    training_handle: Option<TrainingJobHandle>,
}

impl ForexApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        runtime: AppRuntimeConfig,
        settings: Settings,
    ) -> Self {
        ui::theme::apply_theme(&_cc.egui_ctx);
        let (tx, rx) = mpsc::channel(10000);
        let symbols = forex_data::discover_symbols(&runtime.data_dir).unwrap_or_default();
        let state = AppState::new(runtime.clone(), &settings, symbols);
        let _heartbeat_handle = spawn_account_heartbeat(tx.clone());

        Self {
            trading_session: TradingSession::new(),
            workspace: WorkspaceState::default(),
            state,
            tx: tx.clone(),
            rx,
            discovery_handle: None,
            training_handle: None,
        }
    }

    fn refresh_symbols(&mut self) {
        self.state.available_symbols =
            forex_data::discover_symbols(&self.state.runtime.data_dir).unwrap_or_default();
    }

    fn process_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                ServiceEvent::DiscoveryUpdated(snapshot) => {
                    let terminal = matches!(
                        snapshot.state,
                        app_services::jobs::JobState::Succeeded
                            | app_services::jobs::JobState::Degraded
                            | app_services::jobs::JobState::Failed
                            | app_services::jobs::JobState::Cancelled
                    );
                    self.state.discovery_job = Some(snapshot);
                    if terminal {
                        self.discovery_handle = None;
                    }
                }
                ServiceEvent::TrainingUpdated(snapshot) => {
                    let terminal = matches!(
                        snapshot.state,
                        app_services::jobs::JobState::Succeeded
                            | app_services::jobs::JobState::Degraded
                            | app_services::jobs::JobState::Failed
                            | app_services::jobs::JobState::Cancelled
                    );
                    self.state.training_job = Some(snapshot);
                    if terminal {
                        self.training_handle = None;
                    }
                }
                ServiceEvent::LlmNewsUpdated(status) => {
                    self.state.llm_news_filter.current_status = status;
                }
                ServiceEvent::Heartbeat => {
                    if self.trading_session.is_connected() {
                        let _ = self.trading_session.refresh_runtime(&mut self.state);
                    }
                }
                ServiceEvent::CTraderConnectUpdated(runtime) => {
                    self.trading_session
                        .handle_ctrader_connect_result(&mut self.state, runtime);
                }
                ServiceEvent::BootstrapUpdated(snapshot) => {
                    self.state.bootstrap_job = Some(snapshot);
                }
                ServiceEvent::ConnectOutcome(result) => match result {
                    Ok(msg) => self.state.status_msg = msg,
                    Err(err) => self.state.status_msg = format!("Connect Error: {}", err),
                },
            }
            ctx.request_repaint();
        }
    }
}

impl eframe::App for ForexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages(ctx);

        egui::TopBottomPanel::top("top_panel")
            .frame(ui::theme::top_panel_frame(ctx.style().as_ref()))
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    // Branding
                    ui.heading(" FOREX AI"); // Using a chart icon placeholder or just text
                    ui.add_space(20.0);

                    // Ribbon Data
                    ui.vertical_centered(|ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 24.0;

                            render_ribbon_item(
                                ui,
                                "SYMBOL",
                                &self.state.selected_pair,
                                ui::theme::ACCENT,
                            );
                            render_ribbon_item(
                                ui,
                                "SOURCE",
                                match self.state.data_source {
                                    app_state::DataSource::CTrader => "CTRADER",
                                    app_state::DataSource::Local => "LOCAL",
                                },
                                ui::theme::TEXT_MUTED,
                            );

                            let status_color = if self.state.status_msg.contains("Connected")
                                || self.state.status_msg.contains("Online")
                            {
                                ui::theme::SUCCESS
                            } else {
                                ui::theme::DANGER
                            };
                            render_ribbon_item(ui, "STATUS", &self.state.status_msg, status_color);
                        });
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        if ui.button("⚙").clicked() {
                            // Toggle settings
                        }
                        ui.label(format!("CPU Cores: {}", self.state.hardware.cpu_cores));
                    });
                });
                ui.add_space(4.0);
            });

        egui::CentralPanel::default()
            .frame(ui::theme::central_panel_frame(ctx.style().as_ref()))
            .show(ctx, |ui| {
                let mut viewer = WorkspaceViewer::new(
                    &mut self.state,
                    &mut self.trading_session,
                    &self.tx,
                    &mut self.discovery_handle,
                    &mut self.training_handle,
                );
                render_workspace(ui, &mut self.workspace, &mut viewer);
                if viewer.refresh_requested() {
                    self.refresh_symbols();
                }
            });

        let discovery_active = self
            .state
            .discovery_job
            .as_ref()
            .map(|snapshot| {
                matches!(
                    snapshot.state,
                    app_services::jobs::JobState::Queued | app_services::jobs::JobState::Running
                )
            })
            .unwrap_or(false);
        let training_active = self
            .state
            .training_job
            .as_ref()
            .map(|snapshot| {
                matches!(
                    snapshot.state,
                    app_services::jobs::JobState::Queued | app_services::jobs::JobState::Running
                )
            })
            .unwrap_or(false);

        if discovery_active || training_active {
            ctx.request_repaint();
        }
    }
}

pub(crate) fn app_record(
    operation: &str,
    status: &str,
    message: impl Into<String>,
) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!("app-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::App,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message: message.into(),
        body: String::new(),
    }
}

fn system_time_string() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_secs();
    format!("unix:{seconds}")
}

fn spawn_account_heartbeat(tx: mpsc::Sender<ServiceEvent>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            if tx.send(ServiceEvent::Heartbeat).await.is_err() {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{app_record, AppRuntimeConfig};
    use crate::app_state::DataSource;
    use forex_core::Settings;
    use std::path::PathBuf;

    #[test]
    fn app_runtime_config_uses_settings_data_dir() {
        let mut settings = Settings::default();
        settings.system.data_dir = PathBuf::from("custom-data-root");

        let runtime = AppRuntimeConfig::from_settings("config.yaml".to_string(), true, &settings);

        assert_eq!(runtime.data_dir, PathBuf::from("custom-data-root"));
        assert!(runtime.start_local);
    }

    #[test]
    fn app_record_targets_app_section() {
        let record = app_record("headless_start", "STARTED", "headless startup");

        assert_eq!(
            record.subsystem,
            forex_core::sectioned_log::SubsystemSection::App
        );
        assert_eq!(record.operation, "headless_start");
        assert_eq!(record.status, "STARTED");
        assert_eq!(record.message, "headless startup");
    }

    #[test]
    fn trading_panel_mode_disables_live_controls_in_local_mode() {
        let mode = crate::app_services::trading::panel_mode(DataSource::Local, false);

        assert_eq!(
            mode,
            crate::app_services::trading::TradingPanelMode::LocalOnly
        );
    }

    #[test]
    fn trading_panel_mode_switches_to_connected_when_ctrader_is_connected() {
        let mode = crate::app_services::trading::panel_mode(DataSource::CTrader, true);

        assert_eq!(
            mode,
            crate::app_services::trading::TradingPanelMode::Connected
        );
    }

    #[test]
    fn hardware_slider_bounds_preserve_existing_cpu_range() {
        let bounds = crate::ui::hardware::cpu_slider_bounds();

        assert_eq!(bounds.start(), &1);
        assert_eq!(bounds.end(), &252);
    }

    #[test]
    fn risk_slider_bounds_preserve_existing_guard_ranges() {
        let drawdown = crate::ui::risk::drawdown_slider_bounds();
        let lot_size = crate::ui::risk::lot_size_slider_bounds();

        assert_eq!(drawdown.start(), &0.01);
        assert_eq!(drawdown.end(), &0.20);
        assert_eq!(lot_size.start(), &0.01);
        assert_eq!(lot_size.end(), &50.0);
    }
}
