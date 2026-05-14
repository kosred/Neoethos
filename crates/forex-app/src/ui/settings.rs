use crate::app_services::ServiceEvent;
use crate::app_state::AppState;
use crate::ui::components::render_view_header;
use crate::ui::system::shared::sync_news_now;
use crate::ui::theme;
use eframe::egui;
use tokio::sync::mpsc;

pub fn render(ui: &mut egui::Ui, state: &mut AppState, tx: &mpsc::Sender<ServiceEvent>) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        render_view_header(
            ui,
            "Settings",
            "All system parameters in one place. Every section includes guidance on valid ranges and intent.",
        );
        ui.separator();

        egui::CollapsingHeader::new("Discovery Parameters")
            .default_open(true)
            .show(ui, |ui| {
                render_discovery_settings(ui, state);
            });

        egui::CollapsingHeader::new("Risk Guard")
            .default_open(true)
            .show(ui, |ui| {
                render_risk_settings(ui, state);
            });

        egui::CollapsingHeader::new("Hardware")
            .default_open(false)
            .show(ui, |ui| {
                render_hardware_settings(ui, state);
            });

        egui::CollapsingHeader::new("Intelligence & News")
            .default_open(false)
            .show(ui, |ui| {
                render_intelligence_settings(ui, state, tx);
            });
    });
}

fn param_label(ui: &mut egui::Ui, title: &str, description: &str) {
    ui.label(
        egui::RichText::new(title)
            .color(theme::TEXT_PRIMARY)
            .strong(),
    );
    ui.label(
        egui::RichText::new(description)
            .color(theme::TEXT_MUTED)
            .small(),
    );
}

fn render_discovery_settings(ui: &mut egui::Ui, state: &mut AppState) {
    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.label(
            egui::RichText::new(
                "Controls the genetic search engine that assembles tradable strategy portfolios \
                 from indicator combinations.",
            )
            .color(theme::TEXT_MUTED)
            .small(),
        );
        ui.add_space(8.0);

        egui::Grid::new("settings_discovery_grid")
            .num_columns(2)
            .spacing([16.0, 12.0])
            .show(ui, |ui| {
                // Base Timeframe
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Base Timeframe",
                        "Primary timeframe for strategy signal generation.\nTypically M1 or M5.",
                    );
                });
                ui.text_edit_singleline(&mut state.discovery_form.base_tf);
                ui.end_row();

                // Higher Timeframes
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Higher Timeframes",
                        "Comma-separated list for multi-timeframe confluence filters.\nExample: M5, M15, H1",
                    );
                });
                ui.text_edit_singleline(&mut state.discovery_form.higher_tfs);
                ui.end_row();

                // Population
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Population",
                        "Strategy candidates per evolutionary generation.\nHigher = more diversity, slower search.\nRecommended: 50–200.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.population, 10..=500)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Generations
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Generations",
                        "Evolutionary cycles. More cycles = better refinement at the cost of time.\nRecommended: 10–50.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.generations, 1..=100)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Max Indicators
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Max Indicators",
                        "Complexity cap per strategy. Lower = simpler, more robust strategies.\nRecommended: 5–15.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.max_indicators, 1..=30)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Target Candidates
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Target Candidates",
                        "Profitable strategies collected before portfolio assembly begins.\nRecommended: 100–500.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.target_candidates, 10..=1000)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Portfolio Size
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Portfolio Size",
                        "Maximum strategies assembled into the final portfolio.\nLarger portfolios spread risk across more independent strategies.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.portfolio_size, 1..=500)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Correlation Threshold
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Correlation Threshold",
                        "Maximum allowed correlation between portfolio strategies.\nLower = more independent strategies, better diversification.\nRecommended: 0.5–0.75.",
                    );
                });
                ui.add(
                    egui::Slider::new(
                        &mut state.discovery_form.correlation_threshold,
                        0.0..=1.0,
                    )
                    .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Min Trades / Day
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Min Trades / Day",
                        "Minimum average trade frequency required. Filters out strategies\nthat rarely trigger. Recommended: 0.5–3.0.",
                    );
                });
                ui.add(
                    egui::Slider::new(
                        &mut state.discovery_form.min_trades_per_day,
                        0.1..=10.0,
                    )
                    .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();
            });
    });
}

fn render_risk_settings(ui: &mut egui::Ui, state: &mut AppState) {
    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.label(
            egui::RichText::new(
                "Prop-firm challenge guard rails. These limits are enforced before every trade dispatch.",
            )
            .color(theme::TEXT_MUTED)
            .small(),
        );
        ui.add_space(8.0);

        egui::Grid::new("settings_risk_grid")
            .num_columns(2)
            .spacing([16.0, 12.0])
            .show(ui, |ui| {
                // Daily Drawdown Limit
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Daily Drawdown Limit",
                        "Maximum allowed equity loss in a single day (fraction).\n4% = 0.04. Most prop firms require ≤ 4–5%.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.risk.daily_drawdown_limit, 0.01..=0.20)
                        .custom_formatter(|v, _| format!("{:.1}%", v * 100.0))
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Total Drawdown Limit
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Total Drawdown Limit",
                        "Maximum allowed total equity drawdown from initial balance (fraction).\n7% = 0.07. Most prop firms require ≤ 8–10%.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.risk.total_drawdown_limit, 0.05..=0.50)
                        .custom_formatter(|v, _| format!("{:.1}%", v * 100.0))
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Risk Per Trade
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Risk Per Trade",
                        "Fraction of equity risked per single trade.\n1–2% is standard; higher = faster growth but more volatile.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.risk.risk_per_trade, 0.005..=0.10)
                        .custom_formatter(|v, _| format!("{:.1}%", v * 100.0))
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Max Lot Size
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Max Lot Size",
                        "Absolute lot size cap per individual trade.\nActs as a hard ceiling regardless of risk calculation.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.risk.max_lot_size, 0.01..=50.0)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // Require Stop Loss
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "Require Stop Loss",
                        "Reject any trade that does not have a stop loss attached.\nMandatory for most prop firm challenges.",
                    );
                });
                ui.checkbox(&mut state.risk.require_stop_loss, "Enforce stop loss on every trade");
                ui.end_row();
            });
    });
}

fn render_hardware_settings(ui: &mut egui::Ui, state: &mut AppState) {
    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.label(
            egui::RichText::new(
                "CPU and GPU allocation for discovery, training, and evaluation pipelines.",
            )
            .color(theme::TEXT_MUTED)
            .small(),
        );
        ui.add_space(8.0);

        egui::Grid::new("settings_hardware_grid")
            .num_columns(2)
            .spacing([16.0, 12.0])
            .show(ui, |ui| {
                // CPU Cores
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "CPU Cores",
                        "Thread budget for the discovery and training backend.\nDefault: all available cores. Reduce to leave headroom for the OS.",
                    );
                });
                ui.add(
                    egui::Slider::new(&mut state.hardware.cpu_cores, 1..=252)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.end_row();

                // GPU
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "GPU Acceleration (CUDA)",
                        "Enable NVIDIA CUDA backend for model training and fitness evaluation.\nRequires CUDA 11.x+ and a supported GPU. Falls back to CPU if unavailable.",
                    );
                });
                ui.checkbox(
                    &mut state.hardware.gpu_enabled,
                    "Enable GPU acceleration",
                );
                ui.end_row();
            });
    });
}

fn render_intelligence_settings(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &mpsc::Sender<ServiceEvent>,
) {
    theme::section_frame(ui.style()).show(ui, |ui| {
        ui.label(
            egui::RichText::new(
                "AI decision gating, LLM news filters, and auto-trade arming. \
                 Keep auto-trade disabled until the model stack is fully validated.",
            )
            .color(theme::TEXT_MUTED)
            .small(),
        );
        ui.add_space(8.0);

        egui::Grid::new("settings_intel_grid")
            .num_columns(2)
            .spacing([16.0, 12.0])
            .show(ui, |ui| {
                // AI Auto-Trade
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "AI Auto-Trade",
                        "Allows the ensemble to dispatch live trades automatically.\nKeep OFF until models are validated on a demo account.",
                    );
                });
                let mut auto = state.auto_trade_enabled;
                let label = if auto { "ARMED — live trades enabled" } else { "Manual-safe mode" };
                ui.checkbox(&mut auto, label);
                state.auto_trade_enabled = auto;
                ui.end_row();

                // LLM Provider
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "LLM Provider",
                        "Service used for news sentiment analysis.\nSupported: openai, perplexity.",
                    );
                });
                ui.add_sized(
                    [160.0, 24.0],
                    egui::TextEdit::singleline(&mut state.llm_news_filter.llm_provider),
                );
                ui.end_row();

                // API Key
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "LLM API Key",
                        "API key for the configured LLM provider.",
                    );
                });
                // audit-fix F8: dereference the Zeroizing wrapper for the
                // egui text-edit buffer; re-wrap on assign.
                let mut api_key: String = state
                    .llm_news_filter
                    .api_key
                    .as_deref()
                    .cloned()
                    .unwrap_or_default();
                ui.add_sized(
                    [160.0, 24.0],
                    egui::TextEdit::singleline(&mut api_key).password(true),
                );
                state.llm_news_filter.api_key = if api_key.trim().is_empty() {
                    None
                } else {
                    Some(zeroize::Zeroizing::new(api_key))
                };
                ui.end_row();

                // News Kill-Switch
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "News Blackout Kill-Switch",
                        "Block all trades during high-impact economic news events.\nStrongly recommended for prop firm challenges.",
                    );
                });
                ui.checkbox(
                    &mut state.llm_news_filter.enabled,
                    "Enable LLM news filter",
                );
                ui.end_row();

                // News Lookahead
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "News Blackout Before (min)",
                        "Minutes before a news event to enter blackout.\nTypically 5–30 min depending on risk tolerance.",
                    );
                });
                ui.label(
                    egui::RichText::new(format!(
                        "{} min",
                        state.llm_news_filter.blackout_minutes_before
                    ))
                    .color(theme::TEXT_PRIMARY),
                );
                ui.end_row();

                // News kill window
                ui.vertical(|ui| {
                    param_label(
                        ui,
                        "News Blackout After (min)",
                        "Minutes after a news event to remain in blackout.\nTypically 5–15 min.",
                    );
                });
                ui.label(
                    egui::RichText::new(format!(
                        "{} min",
                        state.llm_news_filter.blackout_minutes_after
                    ))
                    .color(theme::TEXT_PRIMARY),
                );
                ui.end_row();

                // Sync now
                ui.label(
                    egui::RichText::new("News Status")
                        .color(theme::TEXT_PRIMARY)
                        .strong(),
                );
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(&state.llm_news_filter.current_status)
                            .color(theme::TEXT_MUTED),
                    );
                    if ui.button("Sync Now").clicked() {
                        sync_news_now(state, tx);
                    }
                });
                ui.end_row();
            });
    });
}
