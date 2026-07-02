//! Autonomous LLM supervisor — the operator's tireless co-pilot.
//!
//! Periodically (or on demand) gathers a compact snapshot of EVERYTHING that
//! matters — engines, live autopilot, journal stats, blacklist, account,
//! portfolios — hands it to the operator's ChatGPT subscription (the same
//! Codex OAuth the AI Desk uses; no extra key), and executes the actions the
//! model proposes.
//!
//! ## Authority tiers (the safety architecture)
//!
//! - **T1 observe** (autonomous): read every surface; `note` findings; fetch
//!   public URLs for research.
//! - **T2 reversible controls** (autonomous): start/stop discovery + training,
//!   start/stop live engines (the demo-forward gate still blocks ineligible
//!   strategies on REAL-money environments — that gate is not bypassable from
//!   here), and config changes THROUGH the same validated/clamped
//!   `POST /settings` applier the UI uses.
//! - **T3 money moves** (approval-gated): closing a position goes through the
//!   pending-actions queue (#136) — the human's click executes, never the LLM.
//!
//! Every tick and every action lands in `<data_dir>/supervisor_log.jsonl` so
//! the operator can always answer "what did it do and why".
//!
//! Defensive by contract: every step best-effort; an LLM outage, a malformed
//! reply, or a failed action logs and NEVER destabilises the app.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};

use crate::server::state::AppApiState;

// ── Persistent supervisor config ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SupervisorConfig {
    /// Master switch — the loop does nothing while false.
    pub enabled: bool,
    /// Minutes between autonomous ticks (clamped 5..=240).
    pub interval_minutes: u64,
    /// Hard cap on actions executed per tick (clamped 1..=5).
    pub max_actions_per_tick: usize,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            enabled: false, // explicit operator opt-in from the UI
            interval_minutes: 30,
            max_actions_per_tick: 3,
        }
    }
}

fn data_dir() -> Option<PathBuf> {
    neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
        .ok()
        .map(|s| s.system.data_dir)
}

fn config_path() -> Option<PathBuf> {
    data_dir().map(|d| d.join("supervisor.json"))
}

fn log_path() -> Option<PathBuf> {
    data_dir().map(|d| d.join("supervisor_log.jsonl"))
}

pub fn load_config() -> SupervisorConfig {
    config_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn save_config(cfg: &SupervisorConfig) -> Result<()> {
    let path = config_path().context("data dir unresolvable for supervisor.json")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, serde_json::to_string_pretty(cfg)?)
        .with_context(|| format!("write {}", path.display()))
}

// ── Journal (JSONL, append-only) ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupervisorLogEntry {
    pub ts_ms: i64,
    /// "tick" | "action" | "error" | "note"
    pub kind: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

fn log_entry(kind: &str, detail: impl Into<String>, action: Option<serde_json::Value>, result: Option<String>) {
    let entry = SupervisorLogEntry {
        ts_ms: chrono::Utc::now().timestamp_millis(),
        kind: kind.to_string(),
        detail: detail.into(),
        action,
        result,
    };
    let Some(path) = log_path() else { return };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(line) = serde_json::to_string(&entry) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Most-recent-first tail of the supervisor journal (for the UI + the next
/// tick's own memory).
pub fn recent_log(limit: usize) -> Vec<SupervisorLogEntry> {
    let Some(path) = log_path() else { return Vec::new() };
    let Ok(raw) = std::fs::read_to_string(&path) else { return Vec::new() };
    let mut out: Vec<SupervisorLogEntry> = raw
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    out.reverse();
    out.truncate(limit);
    out
}

// ── Action protocol (STRICT whitelist — serde-tagged, no free-form) ─────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum SupervisorAction {
    /// Record an observation/diagnosis for the operator. Always allowed.
    Note { text: String },
    /// Kick a discovery run (same validated body as the UI button).
    StartDiscovery { symbol: String, base_tf: String },
    StopDiscovery,
    StartTraining { symbol: String, base_tf: String },
    StopTraining,
    /// Start live engines for portfolio files. The demo-forward gate still
    /// blocks ineligible strategies on REAL-money environments.
    StartLive { portfolio_paths: Vec<String> },
    /// Stop ALL live engines (risk-reducing — always allowed).
    StopLive,
    /// Change settings THROUGH the same clamped/validated applier as the UI.
    /// Payload = the camelCase `POST /settings` body (subset).
    UpdateSettings { payload: serde_json::Value },
    /// T3: propose closing a position — lands in the Actions approval queue;
    /// the OPERATOR's click executes it, never this agent.
    ProposeClose { position_id: i64, reason: String },
    /// Fetch a public URL (research). The text excerpt lands in the log so the
    /// NEXT tick can read it.
    FetchUrl { url: String },
}

// ── State bundle ────────────────────────────────────────────────────────────

async fn gather_bundle(state: &AppApiState) -> serde_json::Value {
    // Engines (discovery/training) — same DTO the UI polls.
    let engines = serde_json::to_value(
        crate::server::system_status::engines(State(state.clone())).await.0,
    )
    .unwrap_or(serde_json::Value::Null);

    // Live autopilot overview.
    let live: Vec<serde_json::Value> = {
        match state.live_trading.lock() {
            Ok(handles) => handles
                .iter()
                .map(|h| serde_json::to_value(h.snapshot()).unwrap_or(serde_json::Value::Null))
                .collect(),
            Err(_) => Vec::new(),
        }
    };

    // Journal stats (last 7 days) + last closed trades.
    let (stats, last_trades) = match data_dir() {
        Some(dir) => {
            let now = chrono::Utc::now().timestamp_millis();
            let from = now - 7 * 24 * 3600 * 1000;
            let trades =
                crate::app_services::journal_store::query_closed_trades(&dir, Some(from), None);
            let equity = crate::app_services::journal_store::query_equity(&dir, Some(from), None);
            let stats = serde_json::to_value(
                crate::app_services::journal_stats::compute_stats(&trades, &equity),
            )
            .unwrap_or(serde_json::Value::Null);
            let tail: Vec<serde_json::Value> = trades
                .iter()
                .rev()
                .take(15)
                .map(|t| {
                    serde_json::json!({
                        "symbol": t.symbol, "side": t.side, "netProfit": t.net_profit,
                        "closedMs": t.exit_ts_ms,
                    })
                })
                .collect();
            (stats, serde_json::Value::Array(tail))
        }
        None => (serde_json::Value::Null, serde_json::Value::Null),
    };

    // Account snapshot (cached — no broker roundtrip on the tick path).
    let account = state
        .account()
        .await
        .map(|a| {
            serde_json::json!({
                "balance": a.balance, "equity": a.equity, "currency": a.currency,
                "openPositions": a.positions.len(),
            })
        })
        .unwrap_or(serde_json::Value::Null);

    // Discovered portfolios + permanent blacklist.
    let portfolios = serde_json::to_value(
        crate::server::portfolios::list(State(state.clone())).await.0,
    )
    .unwrap_or(serde_json::Value::Null);
    let blacklist =
        serde_json::to_value(crate::app_services::strategy_blacklist::load())
            .unwrap_or(serde_json::Value::Null);

    // The agent's own recent memory (notes, fetched research, action results).
    let memory = serde_json::to_value(recent_log(20)).unwrap_or(serde_json::Value::Null);

    serde_json::json!({
        "nowUtc": chrono::Utc::now().to_rfc3339(),
        "engines": engines,
        "liveEngines": live,
        "journalStats7d": stats,
        "recentClosedTrades": last_trades,
        "account": account,
        "portfolios": portfolios,
        "blacklist": blacklist,
        "supervisorMemory": memory,
    })
}

// ── Prompt ──────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT: &str = r#"You are the NeoEthos Supervisor — an autonomous operations co-pilot for a
pure-Rust forex trading application. You watch the whole system and keep it
healthy and honest. You are NOT a financial advisor and you never invent
numbers.

You receive a JSON state bundle: discovery/training engine status, live
autopilot engines (with loss streaks), 7-day journal stats, recent closed
trades, account, discovered portfolios (with blacklisted flags), the permanent
blacklist, and your own recent action log (your memory).

Reply with ONLY a JSON array (no prose, no markdown fences) of at most N
actions (N given per request). Available actions:

  {"action":"note","text":"..."}                                  — record a finding/diagnosis for the operator (use freely)
  {"action":"start_discovery","symbol":"EURUSD","base_tf":"M15"}  — kick a strategy search
  {"action":"stop_discovery"}
  {"action":"start_training","symbol":"EURUSD","base_tf":"M15"}
  {"action":"stop_training"}
  {"action":"start_live","portfolio_paths":["..."]}               — start live engines (never blacklisted paths)
  {"action":"stop_live"}                                          — stop ALL live engines (risk-reducing)
  {"action":"update_settings","payload":{...}}                    — camelCase POST /settings subset, e.g. {"riskPerTrade":0.005}
  {"action":"propose_close","position_id":123,"reason":"..."}     — queues for HUMAN approval, never executes itself
  {"action":"fetch_url","url":"https://..."}                      — research; excerpt appears in your memory next tick

Judgement guidelines:
- Prefer observation (note) over intervention. Act only on clear evidence.
- A strategy with a rising loss streak near its cull limit deserves a note, not
  a preemptive stop (auto-cull handles it).
- If NO discovery/training is running and the machine is idle, consider
  starting discovery for a pair with data but few/stale strategies.
- Never start a blacklisted portfolio. Never raise risk settings on a losing
  week. Keep any riskPerTrade suggestion ≤ 0.01 (1%).
- If everything is healthy, a single note saying so is a perfect reply.
Reply with [] if nothing is worth doing."#;

// ── Tick ────────────────────────────────────────────────────────────────────

static TICK_RUNNING: AtomicBool = AtomicBool::new(false);

/// One supervisor cycle: gather → ask the LLM → execute (whitelisted, capped).
/// Returns a short human-readable summary. Guarded against overlap.
pub async fn tick(state: AppApiState) -> Result<String> {
    if TICK_RUNNING.swap(true, Ordering::SeqCst) {
        anyhow::bail!("a supervisor tick is already running");
    }
    let result = tick_inner(state).await;
    TICK_RUNNING.store(false, Ordering::SeqCst);
    result
}

async fn tick_inner(state: AppApiState) -> Result<String> {
    let cfg = load_config();
    let max_actions = cfg.max_actions_per_tick.clamp(1, 5);
    let bundle = gather_bundle(&state).await;

    let user_prompt = format!(
        "State bundle:\n{}\n\nReply with a JSON array of at most {max_actions} actions.",
        serde_json::to_string_pretty(&bundle).unwrap_or_default()
    );

    // Same ChatGPT-subscription path the AI Desk + news briefing use.
    let store = neoethos_codex::AuthStore::at_default();
    let client = neoethos_codex::CodexClient::new(store);
    let mut request = neoethos_codex::ChatCompletionRequest::simple(&user_prompt);
    request.messages.insert(
        0,
        neoethos_codex::ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
    );

    let reply = client
        .chat(request)
        .await
        .context("Codex chat failed — is the AI Desk signed in?")?
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default();

    let actions = parse_actions(&reply);
    log_entry(
        "tick",
        format!("tick complete — {} action(s) proposed", actions.len()),
        None,
        None,
    );

    let mut executed = 0usize;
    let mut summary_parts: Vec<String> = Vec::new();
    for action in actions.into_iter().take(max_actions) {
        let label = action_label(&action);
        let action_json = serde_json::to_value(&action).unwrap_or(serde_json::Value::Null);
        match execute(&state, action).await {
            Ok(outcome) => {
                log_entry("action", label.clone(), Some(action_json), Some(outcome.clone()));
                summary_parts.push(format!("{label}: {outcome}"));
                executed += 1;
            }
            Err(e) => {
                log_entry("error", label.clone(), Some(action_json), Some(e.to_string()));
                summary_parts.push(format!("{label}: FAILED — {e}"));
            }
        }
    }

    let summary = if executed == 0 && summary_parts.is_empty() {
        "tick complete — no actions".to_string()
    } else {
        summary_parts.join(" | ")
    };
    Ok(summary)
}

/// Extract the first JSON array from the reply (models occasionally wrap the
/// array in prose or code fences despite instructions).
fn parse_actions(reply: &str) -> Vec<SupervisorAction> {
    let start = match reply.find('[') {
        Some(i) => i,
        None => return Vec::new(),
    };
    let end = match reply.rfind(']') {
        Some(i) if i > start => i,
        _ => return Vec::new(),
    };
    serde_json::from_str::<Vec<SupervisorAction>>(&reply[start..=end]).unwrap_or_default()
}

fn action_label(a: &SupervisorAction) -> String {
    match a {
        SupervisorAction::Note { text } => format!("note: {}", text.chars().take(160).collect::<String>()),
        SupervisorAction::StartDiscovery { symbol, base_tf } => format!("start_discovery {symbol} {base_tf}"),
        SupervisorAction::StopDiscovery => "stop_discovery".into(),
        SupervisorAction::StartTraining { symbol, base_tf } => format!("start_training {symbol} {base_tf}"),
        SupervisorAction::StopTraining => "stop_training".into(),
        SupervisorAction::StartLive { portfolio_paths } => format!("start_live ×{}", portfolio_paths.len()),
        SupervisorAction::StopLive => "stop_live (all)".into(),
        SupervisorAction::UpdateSettings { .. } => "update_settings".into(),
        SupervisorAction::ProposeClose { position_id, .. } => format!("propose_close #{position_id}"),
        SupervisorAction::FetchUrl { url } => format!("fetch_url {}", url.chars().take(120).collect::<String>()),
    }
}

/// Execute one whitelisted action by CALLING THE SAME HANDLERS the UI uses —
/// every server-side validation, clamp and gate applies to the agent too.
async fn execute(state: &AppApiState, action: SupervisorAction) -> Result<String> {
    use crate::server::{autonomous, engines_control, settings};
    match action {
        SupervisorAction::Note { text } => Ok(format!("noted: {text}")),

        SupervisorAction::StartDiscovery { symbol, base_tf } => {
            let body: engines_control::StartJobBody = serde_json::from_value(
                serde_json::json!({ "symbol": symbol, "base_tf": base_tf }),
            )?;
            let resp = engines_control::discovery_start(State(state.clone()), Some(Json(body))).await;
            Ok(format!("discovery start → {}", response_status(&resp)))
        }
        SupervisorAction::StopDiscovery => {
            let _ = engines_control::discovery_stop(State(state.clone())).await;
            Ok("discovery stop requested".into())
        }
        SupervisorAction::StartTraining { symbol, base_tf } => {
            let body: engines_control::StartJobBody = serde_json::from_value(
                serde_json::json!({ "symbol": symbol, "base_tf": base_tf }),
            )?;
            let resp = engines_control::training_start(State(state.clone()), Some(Json(body))).await;
            Ok(format!("training start → {}", response_status(&resp)))
        }
        SupervisorAction::StopTraining => {
            let _ = engines_control::training_stop(State(state.clone())).await;
            Ok("training stop requested".into())
        }

        SupervisorAction::StartLive { portfolio_paths } => {
            let body: autonomous::StartLiveBody = serde_json::from_value(
                serde_json::json!({ "portfolio_paths": portfolio_paths }),
            )?;
            let resp = autonomous::start_live(State(state.clone()), Json(body)).await;
            Ok(format!("live start → {}", response_status(&resp)))
        }
        SupervisorAction::StopLive => {
            let resp = autonomous::stop_live(State(state.clone())).await;
            Ok(format!("live stop-all → {}", response_status(&resp)))
        }

        SupervisorAction::UpdateSettings { payload } => {
            let dto: settings::SettingsUpdateDto = serde_json::from_value(payload)
                .context("payload is not a valid settings update")?;
            let resp = settings::update_settings(State(state.clone()), Json(dto)).await;
            Ok(format!("settings update → {}", response_status(&resp)))
        }

        SupervisorAction::ProposeClose { position_id, reason } => {
            let id = crate::app_services::pending_actions::propose(
                crate::app_services::pending_actions::ActionKind::ClosePosition {
                    position_id,
                    volume_units: 0, // 0 = entire position, resolved at execute
                    symbol_hint: None,
                },
                format!("[supervisor] {reason}"),
            )?;
            Ok(format!("queued for OPERATOR approval (action {id})"))
        }

        SupervisorAction::FetchUrl { url } => {
            if !url.starts_with("https://") {
                anyhow::bail!("only https URLs are allowed");
            }
            let text = tokio::task::spawn_blocking(move || -> Result<String> {
                let body = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(20))
                    .user_agent("neoethos-supervisor/1.0")
                    .build()?
                    .get(&url)
                    .send()?
                    .error_for_status()?
                    .text()?;
                // Crude tag strip → readable excerpt for the next tick's memory.
                let mut out = String::with_capacity(4096);
                let mut in_tag = false;
                for ch in body.chars() {
                    match ch {
                        '<' => in_tag = true,
                        '>' => in_tag = false,
                        c if !in_tag => out.push(c),
                        _ => {}
                    }
                    if out.len() >= 4000 {
                        break;
                    }
                }
                Ok(out.split_whitespace().collect::<Vec<_>>().join(" "))
            })
            .await??;
            Ok(format!("fetched {} chars: {}", text.len(), text.chars().take(1500).collect::<String>()))
        }
    }
}

fn response_status(resp: &axum::response::Response) -> String {
    let s = resp.status();
    if s.is_success() { format!("OK {s}") } else { format!("HTTP {s}") }
}

// ── Background loop ─────────────────────────────────────────────────────────

/// Spawn the supervisor heartbeat. Checks the persisted config every minute;
/// when enabled and the interval has elapsed, runs one tick. Failures log and
/// the loop lives on.
pub fn spawn(state: AppApiState) {
    tokio::spawn(async move {
        let mut last_tick: Option<std::time::Instant> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let cfg = load_config();
            if !cfg.enabled {
                continue;
            }
            let due = match last_tick {
                None => true,
                Some(t) => t.elapsed().as_secs() >= cfg.interval_minutes.clamp(5, 240) * 60,
            };
            if !due {
                continue;
            }
            last_tick = Some(std::time::Instant::now());
            match tick(state.clone()).await {
                Ok(summary) => tracing::info!(
                    target: "neoethos_app::supervisor",
                    %summary, "supervisor tick complete"
                ),
                Err(e) => {
                    tracing::warn!(
                        target: "neoethos_app::supervisor",
                        error = %e, "supervisor tick failed"
                    );
                    log_entry("error", format!("tick failed: {e}"), None, None);
                }
            }
        }
    });
}
