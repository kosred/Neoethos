mod app_services;
mod app_state;
mod ui;
mod workspace;

use app_services::{
    discovery::DiscoveryJobHandle,
    trading::TradingSession,
    training::TrainingJobHandle,
    ServiceEvent,
};
use app_state::{AppRuntimeConfig, AppState};
use eframe::egui;
use forex_core::logging::{setup_logging, write_subsystem_record};
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use forex_core::Settings;
use mt5_bridge::MT5Engine;
use clap::Parser;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use std::time::{SystemTime, UNIX_EPOCH};
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
    setup_logging(false)?;
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
            Box::new(|cc| Ok(Box::new(ForexApp::new(cc, runtime.clone())))),
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
            info!("Headless keep-alive tick: System Health OK | Cores: {} | Mode: LOCAL", num_cpus::get());
        }
    } else {
        match MT5Engine::new() {
            Ok(mut engine) => {
                if let Ok(true) = engine.initialize() {
                    info!("MT5 successfully connected. Ready for Live Trading.");
                    if let Err(err) = write_subsystem_record(
                        SubsystemSection::App,
                        app_record("headless_mt5_start", "SUCCESS", "MT5 connected for headless mode"),
                    ) {
                        error!("Failed to write APP section log: {}", err);
                    }
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
                    loop {
                        interval.tick().await;
                    }
                } else {
                    warn!("MT5 Connection failed or MetaTrader5 module missing. Headless trading disabled.");
                    if let Err(err) = write_subsystem_record(
                        SubsystemSection::App,
                        app_record(
                            "headless_mt5_start",
                            "DEGRADED",
                            "MT5 connection failed or MetaTrader5 module missing",
                        ),
                    ) {
                        error!("Failed to write APP section log: {}", err);
                    }
                }
            }
            Err(e) => {
                error!("Fatal Bridge Error: {:?}", e);
                if let Err(log_err) = write_subsystem_record(
                    SubsystemSection::App,
                    app_record("headless_mt5_start", "FAILED", format!("fatal bridge error: {e}")),
                ) {
                    error!("Failed to write APP section log: {}", log_err);
                }
            }
        }
    }
}

struct ForexApp {
    trading_session: TradingSession,
    workspace: WorkspaceState,
    state: AppState,
    
    // Message Bus
    tx: mpsc::UnboundedSender<ServiceEvent>,
    rx: mpsc::UnboundedReceiver<ServiceEvent>,
    discovery_handle: Option<DiscoveryJobHandle>,
    training_handle: Option<TrainingJobHandle>,
    
}

impl ForexApp {
    fn new(_cc: &eframe::CreationContext<'_>, runtime: AppRuntimeConfig) -> Self {
        ui::theme::apply_theme(&_cc.egui_ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let symbols = forex_data::discover_symbols(&runtime.data_dir).unwrap_or_default();
        let state = AppState::new(runtime.clone(), symbols);
        
        Self {
            trading_session: TradingSession::new(),
            workspace: WorkspaceState::default(),
            state,
            tx,
            rx,
            discovery_handle: None,
            training_handle: None,
        }
    }

    fn refresh_symbols(&mut self) {
        self.state.available_symbols =
            forex_data::discover_symbols(&self.state.runtime.data_dir).unwrap_or_default();
    }

    fn process_messages(&mut self) {
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
            }
        }
    }
}

impl eframe::App for ForexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages();

        egui::TopBottomPanel::top("top_panel")
            .frame(ui::theme::top_panel_frame(ctx.style().as_ref()))
            .show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let source = match self.state.data_source {
                        app_state::DataSource::MT5 => "MT5",
                        app_state::DataSource::Local => "Local",
                    };
                    ui.heading("Forex AI Terminal");
                    ui.separator();
                    ui.label(format!("Symbol: {}", self.state.selected_pair));
                    ui.separator();
                    ui.label(format!("Source: {source}"));
                    ui.separator();
                    ui.label(format!("Status: {}", self.state.status_msg));
                    ui.separator();
                    ui.label("Workspace: Dockable");
                });
            });
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

#[cfg(test)]
mod tests {
    use crate::app_state::DataSource;
    use forex_core::Settings;
    use super::{app_record, AppRuntimeConfig};
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

        assert_eq!(record.subsystem, forex_core::sectioned_log::SubsystemSection::App);
        assert_eq!(record.operation, "headless_start");
        assert_eq!(record.status, "STARTED");
        assert_eq!(record.message, "headless startup");
    }

    #[test]
    fn trading_panel_mode_disables_live_controls_in_local_mode() {
        let mode = crate::app_services::trading::panel_mode(DataSource::Local, false);

        assert_eq!(mode, crate::app_services::trading::TradingPanelMode::LocalOnly);
    }

    #[test]
    fn trading_panel_mode_switches_to_disconnect_when_mt5_is_connected() {
        let mode = crate::app_services::trading::panel_mode(DataSource::MT5, true);

        assert_eq!(mode, crate::app_services::trading::TradingPanelMode::Connected);
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

        assert_eq!(drawdown.start(), &0.1);
        assert_eq!(drawdown.end(), &10.0);
        assert_eq!(lot_size.start(), &0.01);
        assert_eq!(lot_size.end(), &50.0);
    }
}
