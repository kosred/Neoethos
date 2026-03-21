mod app_services;
mod app_state;
mod ui;

use app_services::{discovery::DiscoveryJobHandle, training::TrainingJobHandle, ServiceEvent};
use app_state::{AppRuntimeConfig, AppState, DataSource, Tab};
use eframe::egui;
use forex_core::logging::{setup_logging, write_subsystem_record};
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use forex_core::Settings;
use mt5_bridge::MT5Engine;
use clap::Parser;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use std::time::{SystemTime, UNIX_EPOCH};

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
    mt5: Option<MT5Engine>,
    state: AppState,
    
    // Message Bus
    tx: mpsc::UnboundedSender<ServiceEvent>,
    rx: mpsc::UnboundedReceiver<ServiceEvent>,
    discovery_handle: Option<DiscoveryJobHandle>,
    training_handle: Option<TrainingJobHandle>,
    
    // Trading State
    terminal_info: String,
}

impl ForexApp {
    fn new(_cc: &eframe::CreationContext<'_>, runtime: AppRuntimeConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let symbols = forex_data::discover_symbols(&runtime.data_dir).unwrap_or_default();
        let state = AppState::new(runtime.clone(), symbols);
        
        Self {
            mt5: None,
            state,
            tx,
            rx,
            discovery_handle: None,
            training_handle: None,
            terminal_info: String::new(),
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

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.selectable_value(&mut self.state.current_tab, Tab::Trading, "📊 Trading");
                    ui.selectable_value(&mut self.state.current_tab, Tab::Discovery, "🔍 Discovery");
                    ui.selectable_value(&mut self.state.current_tab, Tab::Training, "🧠 Training");
                    ui.selectable_value(&mut self.state.current_tab, Tab::Hardware, "⚙️ Hardware");
                    ui.selectable_value(&mut self.state.current_tab, Tab::Risk, "🛡️ Risk");
                });
            });
        });

        egui::SidePanel::left("left_status").show(ctx, |ui| {
            ui.heading("System Status");
            ui.separator();
            
            ui.label("Data Source:");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.state.data_source, DataSource::MT5, "MT5");
                ui.selectable_value(&mut self.state.data_source, DataSource::Local, "Local");
            });

            ui.separator();
            if self.state.data_source == DataSource::MT5 {
                ui.label(format!("MT5: {}", self.state.status_msg));
                if self.mt5.is_some() {
                    ui.colored_label(egui::Color32::GREEN, "● Online");
                } else {
                    ui.colored_label(egui::Color32::RED, "○ Offline");
                }
            } else {
                ui.colored_label(egui::Color32::BLUE, "🏠 Local Mode");
            }

            ui.separator();
            ui.label(format!("CPU Cores: {}", self.state.hardware.cpu_cores));
            ui.label(format!(
                "GPU: {}",
                if self.state.hardware.gpu_enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
            ));
            
            if ui.button("🔄 Refresh Data").clicked() {
                self.refresh_symbols();
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.state.current_tab {
                Tab::Trading => ui::trading::render(
                    ui,
                    &mut self.state,
                    &mut self.mt5,
                    &mut self.terminal_info,
                ),
                Tab::Discovery => ui::discovery::render(
                    ui,
                    &mut self.state,
                    &self.tx,
                    &mut self.discovery_handle,
                ),
                Tab::Training => ui::training::render(
                    ui,
                    &mut self.state,
                    &self.tx,
                    &mut self.training_handle,
                ),
                Tab::Hardware => ui::hardware::render(ui, &mut self.state.hardware),
                Tab::Risk => ui::risk::render(ui, &mut self.state.risk),
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
        let mode = crate::ui::trading::panel_mode(DataSource::Local, false);

        assert_eq!(mode, crate::ui::trading::TradingPanelMode::LocalOnly);
    }

    #[test]
    fn trading_panel_mode_switches_to_disconnect_when_mt5_is_connected() {
        let mode = crate::ui::trading::panel_mode(DataSource::MT5, true);

        assert_eq!(mode, crate::ui::trading::TradingPanelMode::Connected);
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
