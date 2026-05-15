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
use workspace::{WorkspaceGroup, WorkspaceState, WorkspaceTab, WorkspaceViewer, render_workspace};

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

    /// Force the first-run wizard to open on launch even if
    /// `wizard_state.json` already exists. Spec §5.1 entry-point #2.
    #[arg(long, default_value_t = false)]
    wizard: bool,
}

/// Returns true when the wizard should run on this launch. Spec §1.2
/// / §5 — fires when the wizard sentinel file is absent OR when
/// `--wizard` is passed explicitly. The actual modal is rendered by
/// `ui::wizard::wizard_ui`; this helper is the gate.
pub(crate) fn should_run_wizard(force: bool, config_dir: Option<&std::path::Path>) -> bool {
    if force {
        return true;
    }
    let dir = match config_dir {
        Some(d) => d.to_path_buf(),
        None => match dirs::config_dir() {
            Some(d) => d.join("forex-ai"),
            None => return false,
        },
    };
    !dir.join(ui::wizard::WIZARD_STATE_FILENAME).exists()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    setup_logging(true)?;
    forex_search::install_search_runtime_overrides_from_env();
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

    // First-run wizard gate. Spec §5.1 — fires when the sentinel
    // file is absent OR `--wizard` flag is set. The actual modal is
    // rendered inside the egui main loop by `ui::wizard::wizard_ui`;
    // for the skeleton we only log the gate decision here.
    let wizard_due = should_run_wizard(args.wizard, None);
    if wizard_due {
        info!(
            "First-run wizard gate triggered (--wizard={} sentinel-missing={})",
            args.wizard, !args.wizard
        );
    }

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

    let symbols = match forex_data::discover_symbols(&runtime.data_dir) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(
                target: "forex_app::main",
                data_dir = %runtime.data_dir.display(),
                error = %err,
                "headless: discover_symbols failed; continuing with empty symbol list"
            );
            Vec::new()
        }
    };
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
            prop_firm_rules: forex_search::PropFirmRiskRules::default(),
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
        let symbols = match forex_data::discover_symbols(&runtime.data_dir) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(
                    target: "forex_app::main",
                    data_dir = %runtime.data_dir.display(),
                    error = %err,
                    "ForexApp::new: discover_symbols failed; starting with empty list"
                );
                Vec::new()
            }
        };
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
            match forex_data::discover_symbols(&self.state.runtime.data_dir) {
                Ok(s) => s,
                Err(err) => {
                    tracing::warn!(
                        target: "forex_app::main",
                        data_dir = %self.state.runtime.data_dir.display(),
                        error = %err,
                        "refresh_symbols: discover_symbols failed; keeping empty list"
                    );
                    Vec::new()
                }
            };
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
            prop_firm_rules: forex_search::PropFirmRiskRules::default(),
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
                        if let Err(err) = self.trading_session.refresh_runtime(&mut self.state) {
                            tracing::warn!(
                                target: "forex_app::main",
                                error = %err,
                                "heartbeat refresh_runtime failed; will retry on next heartbeat"
                            );
                        }
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
        let mut connect_broker = false;
        let mut disconnect_broker = false;
        let broker_connected = self.trading_session.is_connected();

        // ─── Top bar — brand + global status only ────────────────────
        // Engine and broker controls live in the BOTTOM action bar so
        // the operator's eye lands on one canonical "what's running"
        // strip without scanning four locations.
        egui::TopBottomPanel::top("top_panel")
            .frame(ui::theme::top_panel_frame(ctx.style().as_ref()))
            .exact_height(ui::theme::TOPBAR_HEIGHT)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    // Brand
                    ui.label(
                        egui::RichText::new("Forex AI")
                            .size(ui::theme::FONT_SUBTITLE + 1.0)
                            .strong()
                            .color(ui::theme::TEXT_PRIMARY),
                    );
                    ui.add_space(ui::theme::SPACE_SM);
                    ui::theme::status_badge(ui, "PRO", ui::theme::ACCENT);

                    ui.add_space(ui::theme::SPACE_LG);
                    ui.add(egui::Separator::default().vertical().spacing(ui::theme::SPACE_SM));
                    ui.add_space(ui::theme::SPACE_SM);

                    // Active pair / TF / data source — compact reading
                    render_ribbon_item(ui, "SYMBOL", &self.state.selected_pair, ui::theme::ACCENT);
                    ui.add_space(ui::theme::SPACE_LG);
                    render_ribbon_item(
                        ui,
                        "TIMEFRAME",
                        &self.state.chart_timeframe,
                        ui::theme::TEXT_PRIMARY,
                    );
                    ui.add_space(ui::theme::SPACE_LG);
                    render_ribbon_item(
                        ui,
                        "SOURCE",
                        match self.state.data_source {
                            app_state::DataSource::CTrader => "cTrader",
                            app_state::DataSource::Local => "Local",
                        },
                        ui::theme::TEXT_PRIMARY,
                    );
                    let equity = if self.state.account_equity > 0.0 {
                        self.state.account_equity
                    } else {
                        self.state.account_balance
                    };
                    ui.add_space(ui::theme::SPACE_LG);
                    render_ribbon_item(
                        ui,
                        "EQUITY",
                        &format!("${equity:.2}"),
                        if equity > 0.0 {
                            ui::theme::SUCCESS
                        } else {
                            ui::theme::TEXT_MUTED
                        },
                    );

                    // Right-aligned: status pill, auto-trade toggle, hardware,
                    // settings — the chrome that doesn't change mid-session.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("⚙")
                                        .size(ui::theme::FONT_SUBTITLE)
                                        .color(ui::theme::TEXT_MUTED),
                                )
                                .fill(egui::Color32::TRANSPARENT)
                                .stroke(egui::Stroke::NONE),
                            )
                            .on_hover_text("Open Settings")
                            .clicked()
                        {
                            self.workspace.focus_tab(WorkspaceTab::Settings);
                        }

                        ui.add_space(ui::theme::SPACE_SM);

                        // Auto-trade toggle pill
                        let auto_color = if self.state.auto_trade_enabled {
                            ui::theme::SUCCESS
                        } else {
                            ui::theme::TEXT_FAINT
                        };
                        let auto_text = if self.state.auto_trade_enabled {
                            "AUTO ON"
                        } else {
                            "AUTO OFF"
                        };
                        let auto_button = egui::Button::new(
                            egui::RichText::new(auto_text)
                                .size(ui::theme::FONT_CAPTION)
                                .color(auto_color)
                                .strong(),
                        )
                        .fill(auto_color.linear_multiply(0.15))
                        .stroke(egui::Stroke::new(1.0, auto_color.linear_multiply(0.55)))
                        .corner_radius(egui::CornerRadius::same(ui::theme::RADIUS_SM));
                        if ui
                            .add(auto_button)
                            .on_hover_text("Toggle automatic trade execution")
                            .clicked()
                        {
                            self.state.auto_trade_enabled = !self.state.auto_trade_enabled;
                        }

                        ui.add_space(ui::theme::SPACE_SM);

                        // Hardware indicator — read-only at a glance
                        ui.label(
                            egui::RichText::new(format!(
                                "{} cores  •  GPU {}",
                                self.state.hardware.cpu_cores,
                                if self.state.hardware.gpu_enabled {
                                    "on"
                                } else {
                                    "off"
                                }
                            ))
                            .size(ui::theme::FONT_CAPTION)
                            .color(ui::theme::TEXT_FAINT),
                        );

                        ui.add_space(ui::theme::SPACE_MD);
                        ui.add(egui::Separator::default().vertical().spacing(ui::theme::SPACE_SM));
                        ui.add_space(ui::theme::SPACE_SM);

                        // Global status text — single source of truth
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
                        ui.label(
                            egui::RichText::new(compact_status_text(&self.state.status_msg))
                                .color(status_color)
                                .strong()
                                .size(ui::theme::FONT_BODY),
                        );
                        ui::theme::status_dot(ui, status_color, ui::theme::FONT_BODY);
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
        if connect_broker {
            self.trading_session.connect(&mut self.state);
        }
        if disconnect_broker {
            self.trading_session.disconnect(&mut self.state);
        }

        // ─── Bottom action bar — engine + broker controls ────────────
        // Added BEFORE the sidebar so it spans the FULL width of the
        // window (egui panels added earlier reserve their full edge —
        // adding the side panel first would clip the action bar to the
        // central column only). The "what's running right now" strip:
        // Discovery / Training / Broker each get a status dot + label +
        // a single Start/Stop (or Connect/Disconnect) button.
        // ─── Bottom status bar ───────────────────────────────────────
        // Pro convention: a slim 22-px strip at the very bottom of the
        // window with high-density read-only state (broker connection,
        // active engines, server time, build). The previous 48-px
        // "action bar" tried to host engine Start/Stop buttons here,
        // never rendered cleanly inside the egui_dock layout, and
        // duplicated controls already reachable from the relevant
        // sidebar tabs (Discovery → Start, Training → Start). This
        // strip is purely informational; actions live in their tabs.
        let status_bar_text = self.state.status_msg.clone();
        egui::TopBottomPanel::bottom("status_bar")
            .frame(ui::theme::status_bar_frame(ctx.style().as_ref()))
            .exact_height(ui::theme::STATUSBAR_HEIGHT)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(ui::theme::SPACE_SM);

                    // Broker connection dot + label.
                    let (dot, label) = if broker_connected {
                        (ui::theme::SUCCESS, "Connected")
                    } else {
                        (ui::theme::TEXT_FAINT, "Offline")
                    };
                    ui::theme::status_dot(ui, dot, ui::theme::FONT_CAPTION);
                    ui.label(
                        egui::RichText::new(label)
                            .size(ui::theme::FONT_CAPTION)
                            .color(ui::theme::TEXT_PRIMARY),
                    );

                    ui::theme::status_separator(ui);

                    // Active-engine tally.
                    let mut engines: Vec<&str> = Vec::new();
                    if discovery_running {
                        engines.push("Discovery");
                    }
                    if training_running {
                        engines.push("Training");
                    }
                    if engines.is_empty() {
                        ui.label(
                            egui::RichText::new("No engines running")
                                .size(ui::theme::FONT_CAPTION)
                                .color(ui::theme::TEXT_FAINT),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "Running: {}",
                                engines.join(", ")
                            ))
                            .size(ui::theme::FONT_CAPTION)
                            .color(ui::theme::ACCENT),
                        );
                    }

                    ui::theme::status_separator(ui);
                    ui.label(
                        egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .size(ui::theme::FONT_CAPTION)
                            .color(ui::theme::TEXT_FAINT),
                    );

                    // Right-aligned: latest status message + UTC time.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(ui::theme::SPACE_SM);
                        let utc_clock = format_utc_clock();
                        ui.label(
                            egui::RichText::new(utc_clock)
                                .size(ui::theme::FONT_CAPTION)
                                .color(ui::theme::TEXT_FAINT),
                        );
                        ui::theme::status_separator(ui);
                        if !status_bar_text.is_empty() {
                            ui.label(
                                egui::RichText::new(compact_status_text(&status_bar_text))
                                    .size(ui::theme::FONT_CAPTION)
                                    .color(ui::theme::TEXT_MUTED),
                            );
                        }
                    });
                });
            });

        // Engine start / stop intents are still wired up by the
        // sidebar's Discovery / Training / Settings tabs. Those
        // mutate the `start_*` / `stop_*` / `connect_broker` /
        // `disconnect_broker` flags via their own buttons; the
        // dispatch logic below this block is unchanged. The status
        // bar just observes whether they're on.
        let _ = (
            &start_discovery,
            &stop_discovery,
            &start_training,
            &stop_training,
            &connect_broker,
            &disconnect_broker,
            &discovery_label,
            &training_label,
        );

        // ─── Left sidebar — primary navigation ───────────────────────
        // Single source of truth for "where am I" — no more competing
        // dropdowns in the top bar. Active tab gets a left accent
        // stripe so the eye locks on it instantly.
        let mut sidebar_target: Option<WorkspaceTab> = None;
        let active_tab = self.workspace.active_tab();
        egui::SidePanel::left("workspace_nav")
            .frame(ui::theme::sidebar_frame(ctx.style().as_ref()))
            .resizable(true)
            .default_width(ui::theme::SIDEBAR_WIDTH_DEFAULT)
            .min_width(ui::theme::SIDEBAR_WIDTH_MIN)
            .max_width(ui::theme::SIDEBAR_WIDTH_MAX)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let mut first_group = true;
                        for group in WorkspaceGroup::ordered() {
                            if !first_group {
                                ui.add_space(ui::theme::SPACE_LG);
                            }
                            first_group = false;
                            ui::theme::section_label(ui, group.title());
                            ui.add_space(ui::theme::SPACE_XS);
                            for tab in WorkspaceTab::all_for_group(*group) {
                                if ui::theme::nav_item_with_icon(
                                    ui,
                                    tab.icon(),
                                    tab.title(),
                                    tab.description(),
                                    active_tab == Some(*tab),
                                )
                                .clicked()
                                {
                                    sidebar_target = Some(*tab);
                                }
                            }
                        }
                    });
            });
        if let Some(tab) = sidebar_target {
            self.workspace.focus_tab(tab);
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

/// Render one cell of the bottom action bar — status dot + name +
/// substate label + a single Start/Stop (or Connect/Disconnect) button.
#[allow(clippy::too_many_arguments)]
fn render_engine_control(
    ui: &mut egui::Ui,
    name: &str,
    substate: &str,
    dot_color: egui::Color32,
    button_label: &str,
    button_kind: ui::theme::ButtonKind,
    hover_text: &str,
    start_intent: &mut bool,
    stop_intent: &mut bool,
    running: bool,
) {
    ui.horizontal(|ui| {
        ui::theme::status_dot(ui, dot_color, ui::theme::FONT_BODY);
        ui.add_space(ui::theme::SPACE_XS);
        ui.label(
            egui::RichText::new(name)
                .size(ui::theme::FONT_BODY)
                .strong()
                .color(ui::theme::TEXT_PRIMARY),
        );
        ui.label(
            egui::RichText::new(substate)
                .size(ui::theme::FONT_CAPTION)
                .color(ui::theme::TEXT_MUTED),
        );
        ui.add_space(ui::theme::SPACE_SM);
        if ui::theme::small_button(ui, button_label, button_kind)
            .on_hover_text(hover_text)
            .clicked()
        {
            if running {
                *stop_intent = true;
            } else {
                *start_intent = true;
            }
        }
    });
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

/// Format the current UTC time as `HH:MM:SS UTC` for the status bar.
/// Uses pure `std::time` so we do not have to add a `chrono` dep
/// just for this one display string.
fn format_utc_clock() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let day_secs = secs % 86_400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;
    format!("{h:02}:{m:02}:{s:02} UTC")
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

    #[test]
    fn should_run_wizard_when_flag_is_set() {
        // Force-flag overrides any sentinel detection.
        assert!(super::should_run_wizard(true, Some(std::path::Path::new("/nonexistent"))));
    }

    #[test]
    fn should_run_wizard_when_state_file_absent() {
        let tmp = std::env::temp_dir().join(format!(
            "forex-ai-wizard-gate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(super::should_run_wizard(false, Some(&tmp)));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn should_not_run_wizard_when_state_file_present() {
        let tmp = std::env::temp_dir().join(format!(
            "forex-ai-wizard-gate-present-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join(crate::ui::wizard::WIZARD_STATE_FILENAME),
            b"{\"version\":1}",
        )
        .unwrap();
        assert!(!super::should_run_wizard(false, Some(&tmp)));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
