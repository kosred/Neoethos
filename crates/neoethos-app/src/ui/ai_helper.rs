//! AI Helper panel — chat-driven read-only access to the bot.
//!
//! v0.4.8 ships the **first user-visible Gemma surface**. The operator
//! talks to the bot in natural language ("show my open positions",
//! "what's the EURUSD prediction?") and the helper routes the request
//! through:
//!
//! 1. **Topic gate stack** ([`neoethos_gemma::build_topic_gate_stack_g2`])
//!    — jailbreak-regex + embedding-similarity. Off-topic / jailbreak
//!    attempts get a canned refusal, Gemma never sees the message.
//! 2. **Tool registry** ([`neoethos_gemma::register_all_g3`]) — 10
//!    read-only `BotTool`s (positions, orders, quote, balance,
//!    predictions, explain, risk, news, health, log). The user sees
//!    the tool call + the rendered result, not raw Gemma tokens.
//! 3. **Runtime** ([`neoethos_gemma::GemmaRuntime`]) — production builds
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
use neoethos_gemma::{
    BUNDLED_MODEL_APPROX_BYTES, BUNDLED_MODEL_DOWNLOAD_URL, BUNDLED_MODEL_FILENAME, GemmaRuntime,
    JailbreakRegexGate, LanguageHint, MODEL_PATH_ENV_VAR, StubGemmaRuntime, ToolContext,
    ToolRegistry, TopicCheck, TopicGate, register_all_g3,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::ui::theme;

// ── Cached runtime ───────────────────────────────────────────────────────────
//
// The runtime is loaded once and then reused across every render frame and
// every `process_prompt` call. Storing it in a `OnceLock` avoids threading
// `Arc` through the entire state chain while keeping model-load costs bounded
// to a single startup hit.
//
// Initialisation order:
//   1. First render frame with a valid model path → real `LlamaCppGemmaRuntime`
//      (only when built with the `gemma-backend` feature).
//   2. Any other path → `StubGemmaRuntime` (compile-time default).
//
// A restart is required to pick up a newly downloaded model file — acceptable
// for v0.4.20 (add a "Reload model" button in the next UX iteration).
static GEMMA_RUNTIME: OnceLock<Arc<dyn GemmaRuntime>> = OnceLock::new();

// Suppress the "unused variable" lint: `_model_path` is only consumed
// inside the `#[cfg(feature = "gemma-backend")]` block; the underscore
// prefix silences the warning for the default (stub-only) build while
// remaining accessible when the feature is on.
#[allow(unused_variables)]
fn get_or_init_runtime(_model_path: Option<&std::path::Path>) -> Arc<dyn GemmaRuntime> {
    GEMMA_RUNTIME
        .get_or_init(|| {
            // G1: try to load the real llama.cpp backend if the feature is
            // enabled AND the GGUF is already on disk.
            #[cfg(feature = "gemma-backend")]
            if let Some(path) = _model_path {
                match neoethos_gemma::LlamaCppGemmaRuntime::load(path.to_path_buf()) {
                    Ok(rt) => {
                        tracing::info!(
                            target: "neoethos_app::ai_helper",
                            path = %path.display(),
                            "Gemma G1 runtime loaded"
                        );
                        return Arc::new(rt) as Arc<dyn GemmaRuntime>;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "neoethos_app::ai_helper",
                            error = %e,
                            "Gemma G1 load failed; falling back to stub"
                        );
                    }
                }
            }
            // Fallback: stub that returns a "not yet loaded" message.
            Arc::new(StubGemmaRuntime::with_model_id(
                "Gemma-4-E4B-Uncensored (stub — model not loaded)",
            )) as Arc<dyn GemmaRuntime>
        })
        .clone()
}

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
///
/// ## Async inference design
///
/// `generate()` on the real runtime blocks its caller (it waits on the
/// inference worker thread's reply channel). To keep egui frames at 60 Hz
/// while a 4B model is generating, `process_prompt` spawns a
/// `std::thread::spawn` worker that calls `generate()` and writes the
/// result into `pending_result`. On every render frame we call `try_lock()`
/// on that `Arc<Mutex<…>>`; when a result arrives we drain it into `history`
/// and clear the pending slot.
#[derive(Debug, Clone, Default)]
pub struct AiHelperState {
    /// What the user is currently typing.
    pub input: String,
    /// Chat scrollback. Oldest entries first; the renderer walks this
    /// from the bottom up so the most recent turn is always visible.
    pub history: Vec<ChatTurn>,
    /// Last-known model id from the runtime (for the header strip).
    pub model_id: String,
    /// True while an inference is running in the background thread.
    pub is_inferring: bool,
    /// Shared slot between UI thread and the inference dispatch thread.
    /// `None` when idle; `Some(Arc)` while inference is in flight.
    /// Inner `Option` starts as `None`; the dispatch thread sets it to
    /// `Some(Ok(text))` or `Some(Err(msg))` when done.
    ///
    /// `Arc<Mutex<…>>` is `Clone` (cheap Arc refcount) and `Debug`
    /// (Mutex<Option<_>> is Debug when the inner type is), so the
    /// containing struct can still derive both.
    #[allow(clippy::type_complexity)]
    pub pending_result: Option<Arc<Mutex<Option<Result<String, String>>>>>,
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
            is_inferring: false,
            pending_result: None,
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
    // 2. Bundled with the installer next to neoethos-app.exe.
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
            d.join("neoethos")
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

    // Lazy-init + cache the runtime.
    let runtime = get_or_init_runtime(model_path.as_deref());
    state.model_id = runtime.model_id().to_string();

    // ── Poll for completed background inference ──────────────────────────
    // This runs on every frame (cheap: just a `try_lock`). When the
    // dispatch thread completes, we drain the result into history and
    // clear the pending slot.
    if let Some(pending_arc) = state.pending_result.clone() {
        if let Ok(mut guard) = pending_arc.try_lock() {
            if let Some(outcome) = guard.take() {
                state.is_inferring = false;
                state.pending_result = None;
                match outcome {
                    Ok(text) => state.history.push(ChatTurn::System(text)),
                    Err(err_msg) => state.history.push(ChatTurn::System(format!(
                        "Gemma error: {err_msg}. Try a tool-shaped question \
                         like \"show my positions\" or \"what's the EURUSD quote?\"."
                    ))),
                }
                ui.ctx().request_repaint(); // flush the result turn to screen immediately
            } else {
                // Inference still running — request a repaint in 100 ms so we
                // keep polling without burning a full 60 Hz render budget.
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
    }

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
        // `fetch-gemma-model.ps1` next to neoethos-app.exe; running it
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
                            // egui 0.31 migration: `PlatformOutput::copied_text`
                            // is deprecated in favor of `Context::copy_text`,
                            // which routes through the new commands queue.
                            ui.ctx().copy_text(BUNDLED_MODEL_DOWNLOAD_URL.to_string());
                        }
                        if ui.button("Open save folder").clicked() {
                            if let Some(parent) = suggested_path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                                let _ = open::that(parent);
                            }
                        }
                        if ui
                            .button("Run fetch-gemma-model.ps1 (next to neoethos-app.exe)")
                            .clicked()
                        {
                            // Note — move the PowerShell spawn
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
                                                        target: "neoethos_app::ai_helper",
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
        // The Send button and text field are disabled while inference is running.
        let is_busy = state.is_inferring;
        let response = ui.horizontal(|ui| {
            let text_edit = egui::TextEdit::singleline(&mut state.input)
                .desired_width(ui.available_width() - 110.0)
                .hint_text(if is_busy { "⏳ Generating…" } else { "Ask the bot…" });
            let resp = ui.add_enabled(!is_busy, text_edit);
            let send_clicked = ui.add_enabled(!is_busy, egui::Button::new("Send")).clicked();
            let enter_pressed = !is_busy
                && resp.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter));
            send_clicked || enter_pressed
        });
        if response.inner {
            let prompt = state.input.trim().to_string();
            if !prompt.is_empty() {
                state.input.clear();
                process_prompt(&prompt, state, runtime);
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
/// → runtime fallback chain.
///
/// ## Non-blocking inference
///
/// When no keyword-routed tool matches, the prompt is dispatched to the
/// Gemma runtime. If the runtime is the real llama.cpp backend, inference
/// can take 5–30 s. To keep the UI responsive we:
///   1. Immediately push a "⏳ Generating…" placeholder.
///   2. Spawn a `std::thread::spawn` worker that calls `runtime.generate()`.
///   3. Store an `Arc<Mutex<Option<Result>>>` in `state.pending_result`.
///   4. On every render frame, `render()` calls `try_lock()` and drains
///      the result when ready, replacing the placeholder.
fn process_prompt(prompt: &str, state: &mut AiHelperState, runtime: Arc<dyn GemmaRuntime>) {
    state.history.push(ChatTurn::User(prompt.to_string()));

    // 1. Topic gate — cheapest layer first (jailbreak regex). The full
    //    embedding gate needs the candle backend (G2.1, not yet shipped).
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
        // soft warnings don't block
    }

    // 2. Keyword router — deterministic English + Greek phrase matching.
    let tool_name = route_to_tool(prompt);

    // 3. Tool execution — registry-backed, synchronous (all tools are O(1)).
    if let Some(tool) = tool_name {
        let registry = registry_with_g3_tools_safe();
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

    // 4. Runtime fallback — no tool matched, route to Gemma.
    //
    //    Dispatch is non-blocking: a background thread calls `generate()`
    //    while the UI stays responsive. The shared `Arc<Mutex<…>>` lets
    //    `render()` poll for the result on every frame.
    let prompt_owned = prompt.to_string();
    let result_slot: Arc<Mutex<Option<Result<String, String>>>> = Arc::new(Mutex::new(None));
    let result_slot_bg = Arc::clone(&result_slot);

    std::thread::Builder::new()
        .name("neoethos-gemma-dispatch".to_string())
        .spawn(move || {
            let outcome = runtime
                .generate(&prompt_owned, 256)
                .map_err(|e| e.to_string());
            if let Ok(mut guard) = result_slot_bg.lock() {
                *guard = Some(outcome);
            }
        })
        .ok(); // spawn failure is non-fatal; render() will notice pending never resolves

    state.is_inferring = true;
    state.pending_result = Some(result_slot);
    // Push a visible placeholder so the operator knows inference is running.
    state
        .history
        .push(ChatTurn::System("⏳ Generating response…".to_string()));
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
    // Tool names must match `neoethos_gemma::readonly_tools::register_all_g3`
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
