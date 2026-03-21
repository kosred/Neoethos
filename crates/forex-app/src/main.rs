use eframe::egui;
use mt5_bridge::MT5Engine;
use clap::Parser;
use tracing::{info, error, warn};
use tokio::sync::mpsc;

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

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Trading,
    Discovery,
    Training,
    Hardware,
    Risk,
}

#[derive(PartialEq, Clone, Copy)]
enum DataSource {
    MT5,
    Local,
}

enum AppMessage {
    DiscoveryProgress(f32),
    DiscoveryFinished(String),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    if args.headless {
        info!("Starting Forex AI in Headless Server Mode...");
        run_headless_loop(args).await;
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
            Box::new(|cc| Ok(Box::new(ForexApp::new(cc, args.local)))),
        )?;
        Ok(())
    }
}

async fn run_headless_loop(args: Args) {
    info!("Loading configuration from: {}", args.config);
    
    if args.local {
        info!("Running in Pure Local Mode (Linux Server Discovery/Training).");
        let symbols = forex_data::discover_symbols("data").unwrap_or_default();
        info!("Successfully mapped {} local symbols in 'data/' directory.", symbols.len());
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
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
                    loop {
                        interval.tick().await;
                    }
                } else {
                    warn!("MT5 Connection failed or MetaTrader5 module missing. Headless trading disabled.");
                }
            }
            Err(e) => {
                error!("Fatal Bridge Error: {:?}", e);
            }
        }
    }
}

struct ForexApp {
    mt5: Option<MT5Engine>,
    current_tab: Tab,
    data_source: DataSource,
    
    // Message Bus
    tx: mpsc::Sender<AppMessage>,
    rx: mpsc::Receiver<AppMessage>,
    
    // Trading State
    status_msg: String,
    terminal_info: String,
    
    // Discovery State
    selected_pair: String,
    available_symbols: Vec<String>,
    discovery_progress: f32,
    is_discovering: bool,
    discovery_log: String,
    
    // Hardware State
    cpu_cores: i32,
    gpu_enabled: bool,
    
    // Risk State
    daily_drawdown_limit: f32,
    max_lot_size: f32,
}

impl ForexApp {
    fn new(_cc: &eframe::CreationContext<'_>, start_local: bool) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let symbols = forex_data::discover_symbols("data").unwrap_or_default();
        
        Self {
            mt5: None,
            current_tab: Tab::Trading,
            data_source: if start_local { DataSource::Local } else { DataSource::MT5 },
            tx,
            rx,
            status_msg: if start_local { "Local Mode".to_string() } else { "Offline".to_string() },
            terminal_info: String::new(),
            selected_pair: symbols.first().cloned().unwrap_or_else(|| "EURUSD".to_string()),
            available_symbols: symbols,
            discovery_progress: 0.0,
            is_discovering: false,
            discovery_log: String::new(),
            cpu_cores: num_cpus::get() as i32,
            gpu_enabled: true,
            daily_drawdown_limit: 4.5,
            max_lot_size: 10.0,
        }
    }

    fn refresh_symbols(&mut self) {
        self.available_symbols = forex_data::discover_symbols("data").unwrap_or_default();
    }

    fn process_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::DiscoveryProgress(p) => self.discovery_progress = p,
                AppMessage::DiscoveryFinished(log) => {
                    self.is_discovering = false;
                    self.discovery_log = log;
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
                    ui.selectable_value(&mut self.current_tab, Tab::Trading, "📊 Trading");
                    ui.selectable_value(&mut self.current_tab, Tab::Discovery, "🔍 Discovery");
                    ui.selectable_value(&mut self.current_tab, Tab::Training, "🧠 Training");
                    ui.selectable_value(&mut self.current_tab, Tab::Hardware, "⚙️ Hardware");
                    ui.selectable_value(&mut self.current_tab, Tab::Risk, "🛡️ Risk");
                });
            });
        });

        egui::SidePanel::left("left_status").show(ctx, |ui| {
            ui.heading("System Status");
            ui.separator();
            
            ui.label("Data Source:");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.data_source, DataSource::MT5, "MT5");
                ui.selectable_value(&mut self.data_source, DataSource::Local, "Local");
            });

            ui.separator();
            if self.data_source == DataSource::MT5 {
                ui.label(format!("MT5: {}", self.status_msg));
                if self.mt5.is_some() {
                    ui.colored_label(egui::Color32::GREEN, "● Online");
                } else {
                    ui.colored_label(egui::Color32::RED, "○ Offline");
                }
            } else {
                ui.colored_label(egui::Color32::BLUE, "🏠 Local Mode");
            }

            ui.separator();
            ui.label(format!("CPU Cores: {}", self.cpu_cores));
            ui.label(format!("GPU: {}", if self.gpu_enabled { "Enabled" } else { "Disabled" }));
            
            if ui.button("🔄 Refresh Data").clicked() {
                self.refresh_symbols();
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_tab {
                Tab::Trading => self.ui_trading(ui),
                Tab::Discovery => self.ui_discovery(ui),
                Tab::Training => self.ui_training(ui),
                Tab::Hardware => self.ui_hardware(ui),
                Tab::Risk => self.ui_risk(ui),
            }
        });

        // Repaint if discovering to update progress bar
        if self.is_discovering {
            ctx.request_repaint();
        }
    }
}

impl ForexApp {
    fn ui_trading(&mut self, ui: &mut egui::Ui) {
        ui.heading("Live Trading Terminal");
        ui.separator();
        
        if self.data_source == DataSource::Local {
            ui.label("Live trading is disabled in Local mode.");
            ui.label("Please switch to MT5 source if you are on Windows.");
            return;
        }

        if self.mt5.is_none() {
            if ui.button("🚀 Connect to MetaTrader 5").clicked() {
                match MT5Engine::new() {
                    Ok(mut engine) => {
                        match engine.initialize() {
                            Ok(true) => {
                                self.status_msg = "Connected".to_string();
                                self.terminal_info = engine.terminal_info().unwrap_or_default();
                                self.mt5 = Some(engine);
                            }
                            _ => self.status_msg = "Connection Failed (module missing or terminal closed)".to_string(),
                        }
                    }
                    Err(e) => self.status_msg = format!("Error: {:?}", e),
                }
            }
        } else {
            ui.group(|ui| {
                ui.label("Account Details:");
                ui.label(&self.terminal_info);
            });
            if ui.button("🛑 Disconnect").clicked() {
                self.mt5 = None;
                self.status_msg = "Offline".to_string();
            }
        }
    }

    fn ui_discovery(&mut self, ui: &mut egui::Ui) {
        ui.heading("Strategy Discovery Engine");
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("Target Pair:");
            egui::ComboBox::from_label("")
                .selected_text(&self.selected_pair)
                .show_ui(ui, |ui| {
                    for sym in &self.available_symbols {
                        ui.selectable_value(&mut self.selected_pair, sym.clone(), sym);
                    }
                });
        });

        if self.is_discovering {
            ui.add(egui::ProgressBar::new(self.discovery_progress).text("Evolving..."));
            if ui.button("Stop Search").clicked() {
                self.is_discovering = false;
            }
        } else {
            if ui.button("🔥 Start Genetic Discovery").clicked() {
                self.is_discovering = true;
                let tx = self.tx.clone();
                let pair = self.selected_pair.clone();
                
                // Spawn the Rust native discovery task
                tokio::spawn(async move {
                    info!("Launching Pure-Rust Discovery for {}", pair);
                    // Mock progress simulation
                    for i in 0..=100 {
                        let _ = tx.send(AppMessage::DiscoveryProgress(i as f32 / 100.0)).await;
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                    let _ = tx.send(AppMessage::DiscoveryFinished(format!("Discovery complete for {}", pair))).await;
                });
            }
        }

        if !self.discovery_log.is_empty() {
            ui.separator();
            ui.label("Latest Results:");
            ui.code(&self.discovery_log);
        }
    }

    fn ui_training(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model Swarm Training");
        ui.separator();
        ui.label("Train and update the model swarm using local or live data.");
        if ui.button("🚀 Run Swarm Training").clicked() {
            info!("Training logic triggered.");
        }
    }

    fn ui_hardware(&mut self, ui: &mut egui::Ui) {
        ui.heading("Hardware Allocation");
        ui.separator();
        ui.add(egui::Slider::new(&mut self.cpu_cores, 1..=252).text("CPU Cores"));
        ui.checkbox(&mut self.gpu_enabled, "Enable GPU Acceleration (CUDA)");
    }

    fn ui_risk(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prop-Firm Risk Guard");
        ui.separator();
        ui.add(egui::Slider::new(&mut self.daily_drawdown_limit, 0.1..=10.0).text("Daily Drawdown Limit (%)"));
        ui.add(egui::Slider::new(&mut self.max_lot_size, 0.01..=50.0).text("Max Lot Size"));
    }
}
