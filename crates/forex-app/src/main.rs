use eframe::egui;
use mt5_bridge::MT5Engine;
use clap::Parser;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run the application without a graphical user interface.
    #[arg(short, long, default_value_t = false)]
    headless: bool,

    /// Optional path to the configuration file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse CLI arguments
    let args = Args::parse();

    if args.headless {
        info!("Starting Forex AI in Headless Server Mode...");
        run_headless_loop(args).await;
        Ok(())
    } else {
        info!("Starting Forex AI in GUI Mode...");
        // Start the native UI
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
            ..Default::default()
        };

        eframe::run_native(
            "Forex AI - Pure Rust Terminal",
            options,
            Box::new(|cc| Ok(Box::new(ForexApp::new(cc)))),
        )?;
        Ok(())
    }
}

/// The headless loop for linux servers and daemon execution
async fn run_headless_loop(args: Args) {
    info!("Loading configuration from: {}", args.config);
    // TODO: load config via forex_core::Settings::from_yaml(&args.config)
    
    // Initialize MT5 Engine in the background
    match MT5Engine::new() {
        Ok(mut engine) => {
            if let Ok(true) = engine.initialize() {
                info!("MT5 successfully connected in Headless Mode.");
                
                // Infinite polling loop mimicking forex-ai.py
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    // ... Engine tick ...
                }
            } else {
                error!("Failed to initialize MT5 Engine.");
            }
        }
        Err(e) => {
            error!("Fatal Bridge Error: {:?}", e);
        }
    }
}

/// The GUI application struct
struct ForexApp {
    mt5: Option<MT5Engine>,
    status_msg: String,
    terminal_info: String,
}

impl ForexApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            mt5: None,
            status_msg: "Offline".to_string(),
            terminal_info: String::new(),
        }
    }
}

impl eframe::App for ForexApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel")
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Control Panel");
                
                ui.separator();
                ui.label("MetaTrader 5 Bridge");
                if ui.button("Connect MT5").clicked() {
                    match MT5Engine::new() {
                        Ok(mut engine) => {
                            match engine.initialize() {
                                Ok(true) => {
                                    self.status_msg = "MT5 Connected".to_string();
                                    if let Ok(info) = engine.terminal_info() {
                                        self.terminal_info = info;
                                    }
                                    self.mt5 = Some(engine);
                                }
                                _ => {
                                    self.status_msg = "MT5 Connection Failed".to_string();
                                }
                            }
                        }
                        Err(e) => {
                            self.status_msg = format!("Bridge Error: {:?}", e);
                        }
                    }
                }
                
                if ui.button("Disconnect").clicked() {
                    self.mt5 = None;
                    self.status_msg = "Offline".to_string();
                    self.terminal_info = String::new();
                }

                ui.separator();
                ui.label(format!("Status: {}", self.status_msg));
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Dashboard");
            
            egui::ScrollArea::vertical().show(ui, |ui| {
                if !self.terminal_info.is_empty() {
                    ui.group(|ui| {
                        ui.label("MT5 Terminal Info:");
                        ui.label(&self.terminal_info);
                    });
                } else {
                    ui.label("Welcome to Forex AI (Pure Rust). Connect to MT5 to begin.");
                }
            });
        });
    }
}
