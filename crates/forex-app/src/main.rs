mod app_services;
mod app_state;
mod ui;
mod workspace;

use crate::ui::components::render_ribbon_item;
use app_services::{
    ServiceEvent, discovery::DiscoveryJobHandle, trading::TradingSession,
    training::TrainingJobHandle,
};
use app_state::{AppRuntimeConfig, AppState};
use clap::Parser;
use eframe::egui;
use forex_core::Settings;
use forex_core::logging::{setup_logging, write_subsystem_record};
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{error, info};
use workspace::{WorkspaceState, WorkspaceTab, WorkspaceViewer, render_workspace};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    headless: bool,

    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long, default_value_t = false)]
    local: bool,

    /// Auto-start discovery on launch (headless VPS/WSL2 use-case)
    #[arg(long, default_value_t = false)]
    auto_discovery: bool,

    /// Auto-start training on launch (headless VPS/WSL2 use-case)
    #[arg(long, default_value_t = false)]
    auto_training: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    setup_logging(true)?;
    let settings = Settings::from_yaml(&args.config)?;
    let runtime = AppRuntimeConfig::from_settings(
        args.config.clone(),
        args.local,
        args.auto_discovery,
        args.auto_training,
        &settings,
    );
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
    use app_services::{
        discovery::{DiscoveryRequest, start_discovery_job},
        training::{TrainingRequest, start_training_job},
    };
    use std::path::PathBuf;

    info!("Loading configuration from: {}", runtime.config_path);

    let symbols = forex_data::discover_symbols(&runtime.data_dir).unwrap_or_default();
    info!(
        "Headless: mapped {} local symbols in '{}'",
        symbols.len(),
        runtime.data_dir.display()
    );

    let (tx, _rx) = mpsc::channel(1000);

    if runtime.auto_discovery {
        let symbol = symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());
        info!("Headless: auto-starting discovery for {}", symbol);
        let request = DiscoveryRequest {
            data_root: runtime.data_dir.clone(),
            symbol,
            base_tf: "M1".to_string(),
            higher_tfs: vec!["M5".to_string(), "M15".to_string(), "H1".to_string()],
            config: forex_search::DiscoveryConfig::default(),
        };
        match start_discovery_job(request, tx.clone()) {
            Ok(_handle) => info!("Headless: discovery job started"),
            Err(err) => error!("Headless: failed to start discovery: {}", err),
        }
    }

    if runtime.auto_training {
        let symbol = symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "EURUSD".to_string());
        info!("Headless: auto-starting training for {}", symbol);
        let request = TrainingRequest {
            config_path: runtime.config_path.clone(),
            models_dir: PathBuf::from("models"),
            symbol,
            base_tf: "M1".to_string(),
        };
        match start_training_job(request, tx.clone()) {
            Ok(_handle) => info!("Headless: training job started"),
            Err(err) => error!("Headless: failed to start training: {}", err),
        }
    }

    let mode = if runtime.start_local {
        "LOCAL"
    } else {
        "CTRADER"
    };
    if let Err(err) = write_subsystem_record(
        SubsystemSection::App,
        app_record(
            "headless_start",
            "READY",
            format!(
                "mode={} auto_discovery={} auto_training={}",
                mode, runtime.auto_discovery, runtime.auto_training
            ),
        ),
    ) {
        error!("Failed to write APP section log: {}", err);
    }

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
    loop {
        interval.tick().await;
        info!(
            "Headless keep-alive: Cores={} Mode={} Discovery={} Training={}",
            num_cpus::get(),
            mode,
            runtime.auto_discovery,
            runtime.auto_training,
        );
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
            trading_session: TradingSession::new_with_persisted_credentials(),
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

    fn trigger_start_discovery(&mut self) {
        use app_services::discovery::{DiscoveryRequest, failed_snapshot, start_discovery_job};
        let higher_tfs: Vec<String> = self
            .state
            .discovery_form
            .higher_tfs
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let request = DiscoveryRequest {
            data_root: self.state.runtime.data_dir.clone(),
            symbol: self.state.selected_pair.clone(),
            base_tf: self.state.discovery_form.base_tf.clone(),
            higher_tfs,
            config: forex_search::DiscoveryConfig {
                timeframe_label: self.state.discovery_form.base_tf.clone(),
                population: self.state.discovery_form.population as usize,
                generations: self.state.discovery_form.generations as usize,
                max_indicators: self.state.discovery_form.max_indicators as usize,
                candidate_count: self.state.discovery_form.target_candidates as usize,
                portfolio_size: self.state.discovery_form.portfolio_size as usize,
                corr_threshold: self.state.discovery_form.correlation_threshold as f64,
                min_trades_per_day: self.state.discovery_form.min_trades_per_day as f64,
                ..forex_search::DiscoveryConfig::default()
            },
        };
        match start_discovery_job(request, self.tx.clone()) {
            Ok(handle) => {
                self.state.discovery_job = Some(handle.snapshot.clone());
                self.discovery_handle = Some(handle);
            }
            Err(err) => {
                self.state.discovery_job =
                    Some(failed_snapshot(app_services::jobs::JobKind::Discovery, err));
            }
        }
    }

    fn trigger_start_training(&mut self) {
        use app_services::training::{TrainingRequest, failed_snapshot, start_training_job};
        use std::path::PathBuf;
        let request = TrainingRequest {
            config_path: self.state.runtime.config_path.clone(),
            models_dir: PathBuf::from("models"),
            symbol: self.state.selected_pair.clone(),
            base_tf: self.state.chart_timeframe.clone(),
        };
        match start_training_job(request, self.tx.clone()) {
            Ok(handle) => {
                self.state.training_job = Some(handle.snapshot.clone());
                self.training_handle = Some(handle);
            }
            Err(err) => {
                self.state.training_job = Some(failed_snapshot(err));
            }
        }
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

        // --- Snapshot display state before the panel borrows begin ---
        let discovery_running = self
            .state
            .discovery_job
            .as_ref()
            .map(|s| {
                matches!(
                    s.state,
                    app_services::jobs::JobState::Queued | app_services::jobs::JobState::Running
                )
            })
            .unwrap_or(false);
        let training_running = self
            .state
            .training_job
            .as_ref()
            .map(|s| {
                matches!(
                    s.state,
                    app_services::jobs::JobState::Queued | app_services::jobs::JobState::Running
                )
            })
            .unwrap_or(false);
        let discovery_dot = engine_dot_color(self.state.discovery_job.as_ref());
        let training_dot = engine_dot_color(self.state.training_job.as_ref());
        let discovery_label = engine_short_label(self.state.discovery_job.as_ref());
        let training_label = engine_short_label(self.state.training_job.as_ref());

        // Intent variables — set inside closures, acted on after
        let mut nav_target: Option<WorkspaceTab> = None;
        let mut start_discovery = false;
        let mut stop_discovery = false;
        let mut start_training = false;
        let mut stop_training = false;

        egui::TopBottomPanel::top("top_panel")
            .frame(ui::theme::top_panel_frame(ctx.style().as_ref()))
            .exact_height(50.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("FOREX AI")
                            .size(15.0)
                            .strong()
                            .color(ui::theme::TEXT_PRIMARY),
                    );
                    ui::theme::status_badge(ui, "PRO", ui::theme::ACCENT);
                    ui.add(egui::Separator::default().vertical().spacing(10.0));

                    // ── Navigation dropdown ──────────────────────────────
                    egui::menu::menu_button(ui, "  Navigate ", |ui| {
                        ui.set_min_width(160.0);

                        ui.label(
                            egui::RichText::new("TRADING")
                                .size(10.0)
                                .color(ui::theme::TEXT_MUTED)
                                .strong(),
                        );
                        for (tab, label, hint) in [
                            (WorkspaceTab::Dashboard, "Dashboard", "Overview & equity"),
                            (WorkspaceTab::Chart, "Chart", "Price chart"),
                            (WorkspaceTab::Watchlist, "Markets", "Watchlist"),
                            (WorkspaceTab::Execution, "Order Ticket", "Place orders"),
                            (WorkspaceTab::BottomStrip, "Trade Watch", "Open positions"),
                            (WorkspaceTab::News, "News", "Filtered news feed"),
                        ] {
                            if ui.button(label).on_hover_text(hint).clicked() {
                                nav_target = Some(tab);
                                ui.close_menu();
                            }
                        }

                        ui.separator();
                        ui.label(
                            egui::RichText::new("AI ENGINE")
                                .size(10.0)
                                .color(ui::theme::TEXT_MUTED)
                                .strong(),
                        );
                        for (tab, label, hint) in [
                            (
                                WorkspaceTab::Discovery,
                                "Discovery",
                                "Genetic strategy search",
                            ),
                            (WorkspaceTab::Training, "Training", "Swarm model training"),
                            (
                                WorkspaceTab::Intelligence,
                                "Intelligence",
                                "AI model status",
                            ),
                        ] {
                            if ui.button(label).on_hover_text(hint).clicked() {
                                nav_target = Some(tab);
                                ui.close_menu();
                            }
                        }

                        ui.separator();
                        ui.label(
                            egui::RichText::new("SYSTEM")
                                .size(10.0)
                                .color(ui::theme::TEXT_MUTED)
                                .strong(),
                        );
                        for (tab, label, hint) in [
                            (
                                WorkspaceTab::BrokerSetup,
                                "Broker Setup",
                                "cTrader auth & accounts",
                            ),
                            (WorkspaceTab::Runtime, "Runtime", "Connection & session"),
                            (
                                WorkspaceTab::DataBootstrap,
                                "Data Bootstrap",
                                "Download OHLCV history",
                            ),
                            (WorkspaceTab::Hardware, "Hardware", "CPU / GPU config"),
                            (WorkspaceTab::Risk, "Risk Settings", "Drawdown & lot rules"),
                            (WorkspaceTab::Settings, "Settings", "App configuration"),
                        ] {
                            if ui.button(label).on_hover_text(hint).clicked() {
                                nav_target = Some(tab);
                                ui.close_menu();
                            }
                        }
                    });

                    ui.add_space(2.0);

                    // ── Engine controls dropdown ─────────────────────────
                    egui::menu::menu_button(ui, "  Engine ", |ui| {
                        ui.set_min_width(220.0);

                        // Discovery row
                        ui.horizontal(|ui| {
                            ui::theme::status_dot(ui, discovery_dot, 8.0);
                            ui.label(
                                egui::RichText::new("Discovery")
                                    .size(12.0)
                                    .color(ui::theme::TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(&discovery_label)
                                    .size(10.0)
                                    .color(discovery_dot),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if discovery_running {
                                        if ui
                                            .small_button("Stop")
                                            .on_hover_text("Cancel active discovery run")
                                            .clicked()
                                        {
                                            stop_discovery = true;
                                            ui.close_menu();
                                        }
                                    } else {
                                        if ui
                                            .small_button("Start")
                                            .on_hover_text(
                                                "Start discovery with current form settings",
                                            )
                                            .clicked()
                                        {
                                            start_discovery = true;
                                            ui.close_menu();
                                        }
                                    }
                                },
                            );
                        });

                        ui.separator();

                        // Training row
                        ui.horizontal(|ui| {
                            ui::theme::status_dot(ui, training_dot, 8.0);
                            ui.label(
                                egui::RichText::new("Training")
                                    .size(12.0)
                                    .color(ui::theme::TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(&training_label)
                                    .size(10.0)
                                    .color(training_dot),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if training_running {
                                        if ui
                                            .small_button("Stop")
                                            .on_hover_text("Cancel active training run")
                                            .clicked()
                                        {
                                            stop_training = true;
                                            ui.close_menu();
                                        }
                                    } else {
                                        if ui
                                            .small_button("Start")
                                            .on_hover_text(
                                                "Start training with current symbol & TF",
                                            )
                                            .clicked()
                                        {
                                            start_training = true;
                                            ui.close_menu();
                                        }
                                    }
                                },
                            );
                        });

                        ui.separator();
                        ui.label(
                            egui::RichText::new("Configure via Navigate → AI Engine panels")
                                .size(10.0)
                                .color(ui::theme::TEXT_MUTED),
                        );
                    });

                    ui.add(egui::Separator::default().vertical().spacing(10.0));

                    // ── Status ribbon ────────────────────────────────────
                    ui.spacing_mut().item_spacing.x = 14.0;
                    render_ribbon_item(ui, "SYMBOL", &self.state.selected_pair, ui::theme::ACCENT);
                    render_ribbon_item(
                        ui,
                        "TF",
                        &self.state.chart_timeframe,
                        ui::theme::TEXT_PRIMARY,
                    );
                    render_ribbon_item(
                        ui,
                        "SRC",
                        match self.state.data_source {
                            app_state::DataSource::CTrader => "cTrader",
                            app_state::DataSource::Local => "Local",
                        },
                        ui::theme::TEXT_MUTED,
                    );
                    let equity = if self.state.account_equity > 0.0 {
                        self.state.account_equity
                    } else {
                        self.state.account_balance
                    };
                    render_ribbon_item(
                        ui,
                        "EQ",
                        &format!("{equity:.2}"),
                        if equity > 0.0 {
                            ui::theme::SUCCESS
                        } else {
                            ui::theme::TEXT_MUTED
                        },
                    );

                    let status_color = if self.state.status_msg.contains("Connected")
                        || self.state.status_msg.contains("Online")
                        || self.state.status_msg.contains("Ready")
                    {
                        ui::theme::SUCCESS
                    } else if self.state.status_msg.contains("Error")
                        || self.state.status_msg.contains("Fail")
                    {
                        ui::theme::DANGER
                    } else {
                        ui::theme::WARNING
                    };
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui::theme::status_dot(ui, status_color, 11.0);
                        ui.label(
                            egui::RichText::new(compact_status_text(&self.state.status_msg))
                                .color(status_color)
                                .strong()
                                .size(12.0),
                        );
                    });

                    // ── Right-aligned controls ───────────────────────────
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(6.0);
                        if ui
                            .add(egui::Button::new(
                                egui::RichText::new("Settings")
                                    .size(12.0)
                                    .color(ui::theme::TEXT_MUTED),
                            ))
                            .on_hover_text("Open Settings panel")
                            .clicked()
                        {
                            self.workspace.focus_tab(WorkspaceTab::Settings);
                        }

                        ui.add(egui::Separator::default().vertical().spacing(8.0));

                        let auto_label = if self.state.auto_trade_enabled {
                            egui::RichText::new("AUTO ON")
                                .color(ui::theme::SUCCESS)
                                .strong()
                                .size(12.0)
                        } else {
                            egui::RichText::new("AUTO OFF")
                                .color(ui::theme::TEXT_MUTED)
                                .size(12.0)
                        };
                        if ui
                            .add(egui::Button::new(auto_label))
                            .on_hover_text("Toggle automatic trade execution")
                            .clicked()
                        {
                            self.state.auto_trade_enabled = !self.state.auto_trade_enabled;
                        }

                        ui.add(egui::Separator::default().vertical().spacing(8.0));

                        ui.label(
                            egui::RichText::new(format!(
                                "CPU {}  GPU {}",
                                self.state.hardware.cpu_cores,
                                if self.state.hardware.gpu_enabled {
                                    "ON"
                                } else {
                                    "OFF"
                                }
                            ))
                            .size(11.0)
                            .color(ui::theme::TEXT_MUTED),
                        );
                    });
                });
            });

        // ── Handle navigation intent ─────────────────────────────────────
        if let Some(tab) = nav_target {
            self.workspace.focus_tab(tab);
        }

        // ── Handle engine start/stop intents ─────────────────────────────
        if start_discovery {
            self.trigger_start_discovery();
        }
        if stop_discovery {
            if let Some(handle) = &self.discovery_handle {
                handle.cancel.request();
            }
        }
        if start_training {
            self.trigger_start_training();
        }
        if stop_training {
            if let Some(handle) = &self.training_handle {
                handle.cancel.request();
            }
        }

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

        if discovery_running || training_running {
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

fn compact_status_text(status: &str) -> String {
    const MAX_CHARS: usize = 30;
    if status.chars().count() <= MAX_CHARS {
        return status.to_string();
    }
    let mut compact = status.chars().take(MAX_CHARS - 3).collect::<String>();
    compact.push_str("...");
    compact
}

fn engine_dot_color(job: Option<&app_services::jobs::JobSnapshot>) -> egui::Color32 {
    match job {
        None => ui::theme::TEXT_MUTED,
        Some(s) => match s.state {
            app_services::jobs::JobState::Queued | app_services::jobs::JobState::Running => {
                ui::theme::ACCENT
            }
            app_services::jobs::JobState::Succeeded => ui::theme::SUCCESS,
            app_services::jobs::JobState::Degraded => ui::theme::WARNING,
            app_services::jobs::JobState::Failed => ui::theme::DANGER,
            app_services::jobs::JobState::Cancelled => ui::theme::TEXT_MUTED,
        },
    }
}

fn engine_short_label(job: Option<&app_services::jobs::JobSnapshot>) -> String {
    match job {
        None => "Idle".to_string(),
        Some(s) => match s.state {
            app_services::jobs::JobState::Queued => "Queued".to_string(),
            app_services::jobs::JobState::Running => {
                if s.progress.stage.is_empty() {
                    "Running".to_string()
                } else {
                    s.progress.stage.clone()
                }
            }
            app_services::jobs::JobState::Succeeded => "Done".to_string(),
            app_services::jobs::JobState::Degraded => "Degraded".to_string(),
            app_services::jobs::JobState::Failed => "Failed".to_string(),
            app_services::jobs::JobState::Cancelled => "Cancelled".to_string(),
        },
    }
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
    use super::{AppRuntimeConfig, app_record};
    use crate::app_state::DataSource;
    use forex_core::Settings;
    use std::path::PathBuf;

    #[test]
    fn app_runtime_config_uses_settings_data_dir() {
        let mut settings = Settings::default();
        settings.system.data_dir = PathBuf::from("custom-data-root");

        let runtime = AppRuntimeConfig::from_settings(
            "config.yaml".to_string(),
            true,
            false,
            false,
            &settings,
        );

        assert_eq!(runtime.data_dir, PathBuf::from("custom-data-root"));
        assert!(runtime.start_local);
        assert!(!runtime.auto_discovery);
        assert!(!runtime.auto_training);
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
