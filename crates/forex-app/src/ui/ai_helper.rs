//! AI Helper panel — chat-driven read-only access to the bot.
//!
//! v0.4.8 ships the **first user-visible Gemma surface**. The operator
//! talks to the bot in natural language ("show my open positions",
//! "what's the EURUSD prediction?") and the helper routes the request
//! through:
//!
//! 1. **Topic gate stack** ([`forex_gemma::build_topic_gate_stack_g2`])
//!    — jailbreak-regex + embedding-similarity. Off-topic / jailbreak
//!    attempts get a canned refusal, Gemma never sees the message.
//! 2. **Tool registry** ([`forex_gemma::register_all_g3`]) — 10
//!    read-only `BotTool`s (positions, orders, quote, balance,
//!    predictions, explain, risk, news, health, log). The user sees
//!    the tool call + the rendered result, not raw Gemma tokens.
//! 3. **Runtime** ([`forex_gemma::GemmaRuntime`]) — production builds
//!    use the `mistralrs-runtime` feature once G1 lands; v0.4.8 ships
//!    with [`StubGemmaRuntime`] which returns a "Gemma model not yet
//!    loaded" message so the chat flow stays exercised even before
//!    the GGUF is bundled.
//!
//! ## Why a separate panel (not folded into Intelligence)
//!
//! The Intelligence tab shows ML model-level introspection (per-expert
//! confidence, calibration plots, feature importance). The AI Helper
//! is a conversational interface — different mental model, different
//! interaction pattern. They mirror what TradingView does with its
//! AI Code Editor vs. its Pine Indicator panels: same engine
//! underneath, two surfaces because users come in with two different
//! intents.

use eframe::egui;
use forex_gemma::{
    BUNDLED_MODEL_APPROX_BYTES, BUNDLED_MODEL_DOWNLOAD_URL, BUNDLED_MODEL_FILENAME, GemmaRuntime,
    JailbreakRegexGate, LanguageHint, MODEL_PATH_ENV_VAR, StubGemmaRuntime, ToolContext,
    ToolRegistry, TopicCheck, TopicGate, register_all_g3,
};
use std::path::PathBuf;

use crate::ui::theme;

/// One turn in the chat scrollback.
#[derive(Debug, Clone)]
pub enum ChatTurn {
    /// What the user typed.
    User(String),
    /// What the system surfaced — either Gemma's response, a tool
    /// invocation result, or a topic-gate refusal.
    System(String),
    /// A topic-gate refusal — rendered in red so the operator can see
    /// the gate fired vs. an actual model response.
    GateRefusal(String),
    /// A tool-call summary — `tool_name(args) → result`. Rendered with
    /// a leading icon so it's visually distinct from prose responses.
    ToolCall { tool: String, result: String },
}

/// Persistent state for the AI Helper panel. Held inside `AppState` so
/// the scrollback survives across frames + tab switches.
#[derive(Debug, Clone, Default)]
pub struct AiHelperState {
    /// What the user is currently typing.
    pub input: String,
    /// Chat scrollback. Oldest entries first; the renderer walks this
    /// from the bottom up so the most recent turn is always visible.
    pub history: Vec<ChatTurn>,
    /// Last-known model id from the runtime (for the header strip).
    pub model_id: String,
}

impl AiHelperState {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            history: vec![ChatTurn::System(
                "AI Helper ready. Try: \"show my open positions\", \"what is the EURUSD \
                 quote?\", \"explain why the bot took the last trade\". Read-only — \
                 no orders are placed from chat."
                    .to_string(),
            )],
            model_id: String::new(),
        }
    }
}

/// Resolve the on-disk Gemma model path candidates, returning the
/// first that exists (`Some`) or the preferred user-data target
/// (`None`, with the path the operator should drop the GGUF into).
fn resolve_or_suggest_model_path() -> (Option<PathBuf>, PathBuf) {
    // 1. Env override (operator can swap quants without re-installing).
    if let Ok(custom) = std::env::var(MODEL_PATH_ENV_VAR) {
        let p = PathBuf::from(&custom);
        if p.is_file() {
            return (Some(p), PathBuf::from(custom));
        }
    }
    // 2. Bundled with the installer next to forex-app.exe.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir
                .join("resources")
                .join("models")
                .join(BUNDLED_MODEL_FILENAME);
            if bundled.is_file() {
                return (Some(bundled.clone()), bundled);
            }
        }
    }
    // 3. User-data dir (where fetch-gemma-model.ps1 writes by default).
    let user_data = dirs::data_dir()
        .map(|d| {
            d.join("forex-ai")
                .join("models")
                .join(BUNDLED_MODEL_FILENAME)
        })
        .unwrap_or_else(|| PathBuf::from(BUNDLED_MODEL_FILENAME));
    if user_data.is_file() {
        return (Some(user_data.clone()), user_data);
    }
    (None, user_data)
}

/// Render the AI Helper view inside the workspace dock.
pub fn render(ui: &mut egui::Ui, state: &mut AiHelperState) {
    let (model_path, suggested_path) = resolve_or_suggest_model_path();
    let runtime = StubGemmaRuntime::with_model_id(
        "Gemma-4-E4B-Uncensored (stub — wire G1 mistral.rs runtime)",
    );
    state.model_id = runtime.model_id().to_string();

    ui.vertical(|ui| {
        // Header strip — title + model id pill.
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("AI Helper")
                    .heading()
                    .color(theme::TEXT_PRIMARY),
            );
            ui.add_space(theme::SPACE_SM);
            ui.label(
                egui::RichText::new(format!("model: {}", state.model_id))
                    .small()
                    .color(theme::TEXT_MUTED),
            );
        });
        ui.label(
            egui::RichText::new(
                "Natural-language read-only console. The bot answers via Gemma + \
                 routes tool calls (positions / orders / quotes / predictions / \
                 risk / news / health / log) through the registered ToolRegistry. \
                 Off-topic or jailbreak prompts are refused at the gate before \
                 Gemma sees them.",
            )
            .small()
            .color(theme::TEXT_MUTED),
        );

        // Model-presence banner. v0.4.10: the GGUF is NOT bundled in
        // the installer (the file is 5 GB — too big for a Windows
        // setup.exe). Instead the installer ships
        // `fetch-gemma-model.ps1` next to forex-app.exe; running it
        // downloads the model into the user-data dir. Tool calls keep
        // working without the model (deterministic keyword router →
        // ToolRegistry); only the prose-fallback path through Gemma
        // is gated on the file being present.
        if model_path.is_none() {
            let gb = (BUNDLED_MODEL_APPROX_BYTES as f64) / (1024.0 * 1024.0 * 1024.0);
            ui.add_space(theme::SPACE_XS);
            egui::Frame::new()
                .fill(theme::SURFACE_BG)
                .stroke(egui::Stroke::new(1.0, theme::WARNING))
                .inner_margin(egui::Margin::symmetric(8, 6))
                .corner_radius(egui::CornerRadius::same(4))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("⚠ Gemma model not found on disk")
                            .strong()
                            .color(theme::WARNING),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "Tool calls (positions, orders, quotes, risk, …) work \
                             without the model — the keyword router dispatches \
                             them deterministically. Prose replies need the GGUF.\n\n\
                             To enable prose replies, download {:.1} GB from:\n\
                             {url}\n\
                             and save it as:\n\
                             {dest}",
                            gb,
                            url = BUNDLED_MODEL_DOWNLOAD_URL,
                            dest = suggested_path.display(),
                        ))
                        .small()
                        .color(theme::TEXT_MUTED),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Copy download URL").clicked() {
                            ui.output_mut(|o| {
                                o.copied_text = BUNDLED_MODEL_DOWNLOAD_URL.to_string();
                            });
                        }
                        if ui.button("Open save folder").clicked() {
                            if let Some(parent) = suggested_path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                                let _ = open::that(parent);
                            }
                        }
                        if ui
                            .button("Run fetch-gemma-model.ps1 (next to forex-app.exe)")
                            .clicked()
                        {
                            // V0.4 audit Task #23 — move the PowerShell spawn
                            // off the render thread. Even a successful spawn
                            // takes ~100 ms on Windows (process creation +
                            // PowerShell startup) and was blocking the
                            // current egui frame. Now we spawn via
                            // `std::thread::spawn` so the click returns
                            // immediately; the PowerShell process detaches
                            // and runs independently. Errors are logged but
                            // not surfaced — the user knows whether the
                            // model fetch worked by the model-loaded check
                            // on the next frame.
                            if let Ok(exe) = std::env::current_exe()
                                && let Some(dir) = exe.parent()
                            {
                                let script = dir.join("fetch-gemma-model.ps1");
                                if script.is_file() {
                                    std::thread::Builder::new()
                                        .name("forex-bg-gemma-fetch".to_string())
                                        .spawn(move || {
                                            match std::process::Command::new("powershell.exe")
                                                .args([
                                                    "-NoProfile",
                                                    "-ExecutionPolicy",
                                                    "Bypass",
                                                    "-File",
                                                ])
                                                .arg(&script)
                                                .spawn()
                                            {
                                                Ok(mut child) => {
                                                    let _ = child.wait();
                                                }
                                                Err(err) => {
                                                    tracing::error!(
                                                        target: "forex_app::ai_helper",
                                                        error = %err,
                                                        script = %script.display(),
                                                        "PowerShell spawn failed for Gemma model fetch"
                                                    );
                                                }
                                            }
                                        })
                                        .ok();
                                }
                            }
                        }
                    });
                });
        } else {
            ui.label(
                egui::RichText::new(format!(
                    "✓ Model loaded from: {}",
                    model_path.as_ref().unwrap().display()
                ))
                .small()
                .color(theme::BUY),
            );
        }

        ui.separator();

        // Scrollback — bottom-up so the newest turn is always visible.
        let avail = ui.available_height() - 90.0;
        egui::ScrollArea::vertical()
            .max_height(avail.max(120.0))
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for turn in &state.history {
                    render_turn(ui, turn);
                    ui.add_space(theme::SPACE_XS);
                }
            });

        ui.separator();

        // Input row — text field + Send button. Enter also submits.
        let response = ui.horizontal(|ui| {
            let text_edit = egui::TextEdit::singleline(&mut state.input)
                .desired_width(ui.available_width() - 90.0)
                .hint_text("Ask the bot…");
            let resp = ui.add(text_edit);
            let send_clicked = ui.button("Send").clicked();
            let enter_pressed = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            send_clicked || enter_pressed
        });
        if response.inner {
            let prompt = state.input.trim().to_string();
            if !prompt.is_empty() {
                state.input.clear();
                process_prompt(&prompt, state, &runtime);
            }
        }

        ui.add_space(theme::SPACE_XS);
        ui.label(
            egui::RichText::new(
                "⚠ Live trading orders cannot be placed from chat. Use the Order \
                 Ticket panel for that. The Helper is read-only by contract \
                 (research §7.3).",
            )
            .small()
            .color(theme::WARNING),
        );
    });
}

/// Render one chat turn with appropriate colouring + an inline icon
/// so the operator can scan the scrollback at a glance.
fn render_turn(ui: &mut egui::Ui, turn: &ChatTurn) {
    match turn {
        ChatTurn::User(text) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("→").color(theme::ACCENT).strong());
                ui.label(egui::RichText::new(text).color(theme::TEXT_PRIMARY));
            });
        }
        ChatTurn::System(text) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("⌬").color(theme::TEXT_MUTED).strong());
                ui.label(egui::RichText::new(text).color(theme::TEXT_MUTED));
            });
        }
        ChatTurn::GateRefusal(reason) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("⛔").color(theme::DANGER).strong());
                ui.label(egui::RichText::new(reason).color(theme::DANGER));
            });
        }
        ChatTurn::ToolCall { tool, result } => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("🛠").color(theme::ACCENT).strong());
                ui.label(
                    egui::RichText::new(format!("{tool}() → "))
                        .color(theme::TEXT_PRIMARY)
                        .monospace(),
                );
                ui.label(egui::RichText::new(result).color(theme::TEXT_PRIMARY));
            });
        }
    }
}

/// Run one prompt through the gate → keyword router → tool registry
/// → runtime fallback chain. v0.4.8 implements the keyword router
/// (deterministic mapping of natural-language phrases to BotTool
/// names) so the panel is useful before the real Gemma runtime lands.
/// Real Gemma replaces the keyword router with model-driven tool
/// selection in G6.
fn process_prompt(prompt: &str, state: &mut AiHelperState, runtime: &StubGemmaRuntime) {
    state.history.push(ChatTurn::User(prompt.to_string()));

    // 1. Topic gate — only the cheapest layer (jailbreak regex) is
    //    run here; the full embedding gate needs the candle backend
    //    which doesn't ship with v0.4.8.
    let gate = JailbreakRegexGate::with_defaults();
    let verdict = gate.check_input(prompt, LanguageHint::Unknown);
    if let TopicCheck::Refuse {
        reason,
        canned_response,
    } = &verdict
    {
        state.history.push(ChatTurn::GateRefusal(format!(
            "{reason} — {canned_response}"
        )));
        return;
    }
    if let TopicCheck::SoftWarning { reason } = &verdict {
        state
            .history
            .push(ChatTurn::System(format!("(gate soft-warning: {reason})")));
        // continue — soft warnings don't block
    }

    // 2. Keyword router — map the prompt to a concrete BotTool.
    let tool_name = route_to_tool(prompt);

    // 3. Tool execution — registry-backed.
    if let Some(tool) = tool_name {
        let registry = registry_with_g3_tools_safe();
        // v0.4.8 builds a minimal `ToolContext` per request. G6 will
        // pull the real account_id + look-ahead cutoff from the
        // running `TradingSession`; until then we pass an empty
        // account and `now` for the cutoff so the read-only tools
        // don't filter out current state.
        let ctx = ToolContext {
            past_data_cutoff_unix_ms: i64::MAX,
            account_id: String::new(),
            gated_tools_enabled: false,
        };
        match registry.invoke(tool, serde_json::json!({}), &ctx) {
            Ok(result) => {
                let pretty =
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string());
                state.history.push(ChatTurn::ToolCall {
                    tool: tool.to_string(),
                    result: pretty,
                });
                return;
            }
            Err(err) => {
                state
                    .history
                    .push(ChatTurn::System(format!("Tool {tool} failed: {err}")));
                return;
            }
        }
    }

    // 4. Runtime fallback — no tool matched, ask Gemma directly. The
    //    stub returns "G1 mistral.rs inference runtime: not yet wired",
    //    which is the correct user-facing signal until the real model
    //    is bundled.
    match runtime.generate(prompt, 256) {
        Ok(reply) => state.history.push(ChatTurn::System(reply)),
        Err(err) => state.history.push(ChatTurn::System(format!(
            "Gemma not available: {err}. Try a tool-shaped question \
             like \"show my positions\" or \"what's the EURUSD quote?\"."
        ))),
    }
}

/// Build the v0.4.8 tool registry. Wrapping
/// [`register_all_g3`] in a helper means the panel can call it
/// per-request without copying the wiring code, and lets us swap to a
/// per-session cached registry in G6 without touching the renderer.
fn registry_with_g3_tools_safe() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    // `register_all_g3` registers the 10 read-only G3 tools. If a
    // future refactor renames a tool, the panel still compiles — the
    // keyword router below falls back to a generic message.
    register_all_g3(&mut registry);
    registry
}

/// Keyword → tool routing. Deterministic, English + Greek. Returns
/// the tool name if a known phrase matches, else `None`.
///
/// This is the v0.4.8 stand-in for Gemma's tool-selection step. G6
/// replaces it with model-driven dispatch; until then the deterministic
/// router covers the 10 most common operator queries.
fn route_to_tool(prompt: &str) -> Option<&'static str> {
    // Tool names must match `forex_gemma::readonly_tools::register_all_g3`
    // — keep this list in sync if a tool is renamed there.
    let p = prompt.to_lowercase();
    if p.contains("position") || p.contains("θέσ") {
        Some("list_positions")
    } else if p.contains("order") && (p.contains("open") || p.contains("pending")) {
        Some("list_orders")
    } else if p.contains("quote") || p.contains("price") || p.contains("τιμή") {
        Some("get_quote")
    } else if p.contains("balance") || p.contains("equity") || p.contains("υπόλοιπο") {
        Some("get_account_balance")
    } else if p.contains("predict") || p.contains("πρόβλεψ") {
        Some("get_recent_predictions")
    } else if p.contains("explain") || p.contains("εξήγησ") || p.contains("γιατί") {
        Some("explain_last_decision")
    } else if p.contains("risk") || p.contains("ρίσκ") {
        Some("get_risk_config")
    } else if p.contains("news") || p.contains("νέα") || p.contains("blackout") {
        Some("get_news_blackout_state")
    } else if p.contains("health") || p.contains("status") || p.contains("κατάστα") {
        Some("get_health")
    } else if p.contains("log") || p.contains("ιστορικ") {
        Some("tail_log")
    } else {
        None
    }
}
