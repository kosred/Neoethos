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
    /// Standing operator DIRECTIVES — injected into every tick/chat prompt so
    /// the supervisor follows the human's strategy between conversations
    /// (e.g. "focus discovery on EURUSD+GBPUSD M15", "never start live
    /// engines without asking me first").
    pub directives: Vec<String>,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            enabled: false, // explicit operator opt-in from the UI
            interval_minutes: 30,
            max_actions_per_tick: 3,
            directives: Vec::new(),
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
    // Atomic write: the config carries operator-typed directives — a torn
    // write on crash would silently reset the supervisor (and lose them).
    neoethos_core::storage::json::write_json_atomic(&path, cfg)
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

fn log_entry(
    kind: &str,
    detail: impl Into<String>,
    action: Option<serde_json::Value>,
    result: Option<String>,
) {
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
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Most-recent-first tail of the supervisor journal (for the UI + the next
/// tick's own memory).
pub fn recent_log(limit: usize) -> Vec<SupervisorLogEntry> {
    let Some(path) = log_path() else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
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
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SupervisorAction {
    /// Record an observation/diagnosis for the operator. Always allowed.
    Note {
        text: String,
    },
    /// Kick a discovery run (same validated body as the UI button).
    StartDiscovery {
        dataset_selection: neoethos_data::SelectedDatasetGenerationV1,
    },
    StopDiscovery,
    StartTraining {
        training_handoff: String,
    },
    StopTraining,
    /// Start live engines for portfolio files. The demo-forward gate still
    /// blocks ineligible strategies on REAL-money environments.
    StartLive {
        portfolio_paths: Vec<String>,
    },
    /// Stop ALL live engines. This does not close broker positions and stops
    /// their local strategy supervision; it is not a substitute for an exit.
    StopLive,
    /// Change settings THROUGH the same clamped/validated applier as the UI.
    /// Payload = the camelCase `POST /settings` body (subset).
    UpdateSettings {
        payload: serde_json::Value,
    },
    /// T3: propose closing a position — lands in the Actions approval queue;
    /// the OPERATOR's click executes it, never this agent.
    ProposeClose {
        position_id: i64,
        reason: String,
    },
    /// Fetch a public URL (research). The text excerpt lands in the log so the
    /// NEXT tick can read it.
    FetchUrl {
        url: String,
    },
    /// List the MCP tools available via the local MCP sidecar (cTrader,
    /// filesystem, web, …). The result lands in the log for the next tick.
    McpTools,
    /// Invoke an MCP tool through the local MCP sidecar. `args` is the tool's
    /// JSON arguments object. Read-only/queryable tools are fine to call; the
    /// same T1-3 judgement guidelines apply as to every other action.
    McpCall {
        server: String,
        tool: String,
        #[serde(default)]
        args: serde_json::Value,
    },
}

// ── State bundle ────────────────────────────────────────────────────────────

/// The same cached observation is available to the UI and to every AI cycle.
/// It is monitoring context, never broker execution authority.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SupervisorObservation {
    observed_at_unix_ms: i64,
    account: Option<crate::server::account::AccountSnapshotDto>,
    account_failure: Option<crate::server::state::AccountRefreshFailure>,
    market: crate::server::live_spots::SpotsResponse,
    live_engines: Vec<crate::app_services::live_trading::LiveTradingStatus>,
    live_engine_error: Option<&'static str>,
}

pub(crate) async fn observation(state: &AppApiState) -> SupervisorObservation {
    observation_with_scope(
        state,
        crate::server::bridge::current_execution_account_scope,
    )
    .await
}

async fn observation_with_scope(
    state: &AppApiState,
    resolve_scope: impl FnOnce() -> Result<(
        i64,
        crate::app_services::ctrader_live_auth::CTraderEnvironment,
    )> + Send
    + 'static,
) -> SupervisorObservation {
    let (account, account_failure) =
        crate::server::account::observation_with_scope(state, resolve_scope).await;
    let live = state
        .live_trading
        .lock()
        .map_err(|_| "Live engine registry is unavailable")
        .and_then(|handles| {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .status
                        .lock()
                        .map(|status| status.clone())
                        .map_err(|_| "A live engine status is unavailable")
                })
                .collect::<std::result::Result<Vec<_>, _>>()
        });
    let (live_engines, live_engine_error) = match live {
        Ok(engines) => (engines, None),
        Err(error) => (Vec::new(), Some(error)),
    };
    SupervisorObservation {
        observed_at_unix_ms: chrono::Utc::now().timestamp_millis(),
        account: account.map(Into::into),
        account_failure,
        market: crate::server::live_spots::snapshot(),
        live_engines,
        live_engine_error,
    }
}

async fn gather_bundle(state: &AppApiState) -> serde_json::Value {
    let observation = observation(state).await;
    // Engines (discovery/training) — same DTO the UI polls.
    let engines = match crate::server::system_status::engines(State(state.clone())).await {
        Ok(Json(dto)) => serde_json::to_value(dto).unwrap_or(serde_json::Value::Null),
        Err(response) => inventory_context(response).await,
    };

    // Journal stats (last 7 days) + last closed trades.
    let (stats, last_trades) = match (data_dir(), observation.account.as_ref()) {
        (Some(dir), Some(account)) => {
            let now = chrono::Utc::now().timestamp_millis();
            let from = now - 7 * 24 * 3600 * 1000;
            let mut trades =
                crate::app_services::journal_store::query_closed_trades(&dir, Some(from), None);
            let mut equity =
                crate::app_services::journal_store::query_equity(&dir, Some(from), None);
            // Use the verified observation scope. No account means unknown
            // history, never all accounts; Demo/Live IDs are not interchangeable.
            trades.retain(|t| {
                journal_scope_matches(account, t.account_id.as_deref(), t.environment.as_deref())
            });
            equity.retain(|e| {
                journal_scope_matches(account, e.account_id.as_deref(), e.environment.as_deref())
            });
            let stats = serde_json::to_value(crate::app_services::journal_stats::compute_stats(
                &trades, &equity,
            ))
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
        _ => (serde_json::Value::Null, serde_json::Value::Null),
    };

    // Discovered portfolios + permanent blacklist.
    let portfolios = serde_json::to_value(
        crate::server::portfolios::list(State(state.clone()))
            .await
            .0,
    )
    .unwrap_or(serde_json::Value::Null);
    let blacklist = serde_json::to_value(crate::app_services::strategy_blacklist::load())
        .unwrap_or(serde_json::Value::Null);

    // The agent's own recent memory (notes, fetched research, action results).
    let memory = serde_json::to_value(recent_log(20)).unwrap_or(serde_json::Value::Null);

    // Reuse the UI inventories. The model must not invent a dataset generation
    // or reopen training by symbol/timeframe from old artifacts.
    let (data_inventory, intelligence) = tokio::join!(
        crate::server::system_status::data_bootstrap(State(state.clone())),
        crate::server::intelligence::intelligence(State(state.clone())),
    );
    let data_inventory = inventory_context(data_inventory).await;
    let intelligence = inventory_context(intelligence).await;

    serde_json::json!({
        "nowUtc": chrono::Utc::now().to_rfc3339(),
        "engines": engines,
        "dataInventory": data_inventory,
        "intelligence": intelligence,
        "observedAtUnixMs": observation.observed_at_unix_ms,
        "market": observation.market,
        "liveEngines": observation.live_engines,
        "liveEngineError": observation.live_engine_error,
        "journalStats7d": stats,
        "recentClosedTrades": last_trades,
        "account": observation.account,
        "accountFailure": observation.account_failure,
        "portfolios": portfolios,
        "blacklist": blacklist,
        "liveExperienceCount": crate::app_services::experience_store::count(),
        "supervisorMemory": memory,
    })
}

fn journal_scope_matches(
    account: &crate::server::account::AccountSnapshotDto,
    account_id: Option<&str>,
    environment: Option<&str>,
) -> bool {
    account_id == Some(account.source_account_id.as_str())
        && environment == Some(account.source_environment)
}

async fn inventory_context(response: axum::response::Response) -> serde_json::Value {
    let status = response.status().as_u16();
    match axum::body::to_bytes(response.into_body(), 1024 * 1024).await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::json!({"status": status, "error": "Inventory response is not valid JSON"})
        }),
        Err(_) => serde_json::json!({"status": status, "error": "Inventory unavailable or exceeds the supervisor context limit"}),
    }
}

// ── Prompt ──────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT: &str = r#"You are the NeoEthos Supervisor — an autonomous operations co-pilot for a
pure-Rust forex trading application. You watch the whole system and keep it
healthy and honest. You are NOT a financial advisor and you never invent
numbers.

You receive a JSON state bundle: discovery/training engine status, live
autopilot engines (with loss streaks), 7-day journal stats, recent closed
trades, the cached account with its full positions and fetchedAtUnixMs,
market.spots with bid/ask and received/broker timestamps, discovered portfolios
(with blacklisted flags), the permanent blacklist, and your own recent action
log (your memory), dataInventory.datasets and intelligence.trainingHandoffs.
Missing account, prices, timestamps or engine status mean
UNKNOWN, never zero exposure or a healthy connection. accountFailure records
the actual last refresh failure; any retained account snapshot is then stale.
Do not recommend re-authentication when the failure is missing financial-truth
capability rather than an authentication rejection. freshnessSeconds is the
age of the last received event, not a guarantee that both quote sides are fresh.
Inspect position IDs, stops, targets, account currency and engine protection
errors before proposing position changes. Account pnlUsd is a legacy field name:
its value is in account.currency, not necessarily USD.

Reply with ONLY a JSON array (no prose, no markdown fences) of at most N
actions (N given per request). Available actions:

  {"action":"note","text":"..."}                                  — record a finding/diagnosis for the operator (use freely)
  {"action":"start_discovery","dataset_selection":{"schema":"neoethos.selected-dataset-generation.v1","version":1,"dataset_identity":"<datasetIdentity>","generation_id":"<generation>","manifest_binding_sha256":"<manifestBindingSha256>"}} — copy one exact dataInventory.datasets entry, never invent values
  {"action":"stop_discovery"}
  {"action":"start_training","training_handoff":"<identity from intelligence.trainingHandoffs>"}
  {"action":"stop_training"}
  {"action":"start_live","portfolio_paths":["..."]}               — start live engines (never blacklisted paths)
  {"action":"stop_live"}                                          — stop ALL live engines, NOT their broker positions
  {"action":"update_settings","payload":{...}}                    — camelCase POST /settings subset, e.g. {"riskPerTrade":0.005}
  {"action":"propose_close","position_id":123,"reason":"..."}     — queues for HUMAN approval, never executes itself
  {"action":"fetch_url","url":"https://..."}                      — research; excerpt appears in your memory next tick
  {"action":"mcp_tools"}                                          — list MCP tools available (cTrader, filesystem, web) via the local MCP sidecar
  {"action":"mcp_call","server":"ctrader","tool":"...","args":{}} — invoke an MCP tool (same T1-3 judgement as any action)

Judgement guidelines:
- Prefer observation (note) over intervention. Act only on clear evidence.
- A strategy with a rising loss streak near its cull limit deserves a note, not
  a preemptive stop (auto-cull handles it).
- If NO discovery/training is running and the machine is idle, consider
  starting discovery for a pair with data but few/stale strategies.
- Discovery is ResearchOnly and does not auto-start training. Training must
  explicitly select one published handoff. If either inventory is unavailable,
  report its error; never replace exact selectors with symbol/timeframe guesses.
- Never start a blacklisted portfolio. Never raise risk settings on a losing
  week. Keep any riskPerTrade suggestion ≤ 0.01 (1%).
- Stopping engines also stops their local monitoring. It does not close open
  positions or add protection. Never describe stop_live as a position exit.
- The strategy engine and broker-confirmed protection manage entries/exits;
  this periodic AI cycle is not a tick-by-tick protective stop. Never claim a
  proposed close or a requested stop amendment has executed without a receipt.
- If everything is healthy, a single note saying so is a perfect reply.
Reply with [] if nothing is worth doing."#;

// ── Tick ────────────────────────────────────────────────────────────────────

static TICK_RUNNING: AtomicBool = AtomicBool::new(false);

pub(crate) fn cycle_running() -> bool {
    TICK_RUNNING.load(Ordering::SeqCst)
}

/// One supervisor cycle: gather → ask the LLM → execute (whitelisted, capped).
/// Returns a short human-readable summary. Guarded against overlap.
pub async fn tick(state: AppApiState) -> Result<String> {
    run_exclusive_cycle(&TICK_RUNNING, async move {
        run_cycle(state, None).await.map(|(_, summary)| summary)
    })
    .await
}

/// Operator ↔ supervisor CHAT: same state bundle, same whitelisted actions —
/// plus the operator's message steering the cycle. Returns
/// `(assistant_reply, actions_summary)`. The supervisor and the chat are ONE
/// brain: what you tell it here it acts on (within the same tiered authority).
pub async fn chat(state: AppApiState, message: String) -> Result<(String, String)> {
    run_exclusive_cycle(&TICK_RUNNING, async move {
        log_entry("chat", format!("operator: {message}"), None, None);
        run_cycle(state, Some(message)).await
    })
    .await
}

async fn run_exclusive_cycle<T>(
    running: &AtomicBool,
    cycle: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    if running.swap(true, Ordering::SeqCst) {
        anyhow::bail!("a supervisor cycle is already running — try again in a moment");
    }
    struct RunningGuard<'a>(&'a AtomicBool);
    impl Drop for RunningGuard<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _guard = RunningGuard(running);
    cycle.await
}

async fn run_cycle(
    state: AppApiState,
    operator_message: Option<String>,
) -> Result<(String, String)> {
    let cfg = load_config();
    let max_actions = cfg.max_actions_per_tick.clamp(1, 5);
    let bundle = gather_bundle(&state).await;

    let directives = if cfg.directives.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nSTANDING OPERATOR DIRECTIVES (follow these unless the operator's \
             live message overrides them):\n- {}",
            cfg.directives.join("\n- ")
        )
    };
    let operator = operator_message
        .as_deref()
        .map(|m| format!("\n\nOPERATOR MESSAGE (answer it via note actions, then act):\n{m}"))
        .unwrap_or_default();

    let user_prompt = format!(
        "State bundle:\n{}{directives}{operator}\n\nReply with a JSON array of at most \
         {max_actions} actions. Start with note action(s) addressed to the operator.",
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

    let actions = parse_actions(&reply)?;
    log_entry(
        "tick",
        format!("cycle complete — {} action(s) proposed", actions.len()),
        None,
        None,
    );

    let mut executed = 0usize;
    let mut summary_parts: Vec<String> = Vec::new();
    // Note texts double as the assistant's REPLY to the operator (chat mode).
    let mut reply_parts: Vec<String> = Vec::new();
    for action in actions.into_iter().take(max_actions) {
        if let SupervisorAction::Note { text } = &action {
            reply_parts.push(text.clone());
        }
        let label = action_label(&action);
        let action_json = serde_json::to_value(&action).unwrap_or(serde_json::Value::Null);
        match execute(&state, action).await {
            Ok(outcome) => {
                log_entry(
                    "action",
                    label.clone(),
                    Some(action_json),
                    Some(outcome.clone()),
                );
                summary_parts.push(format!("{label}: {outcome}"));
                executed += 1;
            }
            Err(e) => {
                log_entry(
                    "error",
                    label.clone(),
                    Some(action_json),
                    Some(e.to_string()),
                );
                summary_parts.push(format!("{label}: FAILED — {e}"));
            }
        }
    }

    let summary = if executed == 0 && summary_parts.is_empty() {
        "cycle complete — no actions".to_string()
    } else {
        summary_parts.join(" | ")
    };
    let assistant_reply = if reply_parts.is_empty() {
        summary.clone()
    } else {
        reply_parts.join("\n\n")
    };
    Ok((assistant_reply, summary))
}

/// Extract the first JSON array from the reply (models occasionally wrap the
/// array in prose or code fences despite instructions).
fn parse_actions(reply: &str) -> Result<Vec<SupervisorAction>> {
    let start = reply
        .find('[')
        .context("Supervisor returned no action array; no actions were executed")?;
    let end = reply
        .rfind(']')
        .filter(|end| *end > start)
        .context("Supervisor returned an incomplete action array; no actions were executed")?;
    serde_json::from_str::<Vec<SupervisorAction>>(&reply[start..=end])
        .context("Supervisor returned invalid or unsupported actions; no actions were executed")
}

/// Local MCP sidecar base URL.
///
/// 2026-08-10 config consolidation: this was a SECOND, independent
/// `NEOETHOS_MCP_URL` read. Two readers of one variable meant the Supervisor
/// and the `/mcp/status` card could dial different processes and neither
/// would say so. There is now one resolver, in the module that owns
/// `mcp_servers.json`, and it takes the port from that file — the same file
/// the sidecar itself reads. See `crate::server::mcp::sidecar_url`.
pub(crate) fn mcp_sidecar_url() -> String {
    crate::server::mcp::sidecar_url()
        .trim_end_matches('/')
        .to_string()
}

/// The message an operator sees when the MCP sidecar does not answer.
///
/// **W10b (2026-08-10).** Until today this said "is neoethos-mcp running?"
/// while `crates/neoethos-mcp` ALSO produced a binary called `neoethos-mcp` —
/// a different program with a different job. An operator following the
/// message could check the wrong process, find it running, and conclude the
/// message was lying. The crate has since been renamed (`[[bin]] name =
/// "neoethos-control-plane"`, `crates/neoethos-mcp/Cargo.toml:35`), so the
/// repo can no longer build two binaries with that name — but an installed
/// directory may still hold a stale executable from an earlier build, and the
/// two names are still one letter of context apart.
///
/// So the message names BOTH, and says which one is wanted:
///
/// * **`neoethos-mcp`** — the OUTBOUND sidecar, built from the separate
///   `mcp/` workspace, spawned next to the app, answers HTTP on the port in
///   `mcp_servers.json`. **This is the one that is not answering.**
/// * **`neoethos-control-plane`** — the INBOUND control plane, built from
///   `crates/neoethos-mcp`, which lets an external LLM drive this backend. It
///   is a different process and starting it will not fix this.
const MCP_SIDECAR_UNREACHABLE: &str = "MCP sidecar not reachable. The process that must be running is \
     `neoethos-mcp` (the OUTBOUND sidecar built from the `mcp/` workspace, \
     spawned next to the app, listening on the port in `mcp_servers.json`). \
     It is NOT `neoethos-control-plane` (the INBOUND control plane built from \
     `crates/neoethos-mcp`) — those are two different programs and starting \
     the wrong one will not fix this. Check Settings → MCP for the configured \
     port, and that `neoethos-mcp` sits in the app's own directory.";

/// Audit S02: conservative read-only allowlist for MCP tools the LLM may run
/// without operator approval. Matched on the tool name's leading verb; ANYTHING
/// not clearly read-only — including unknown tools — is treated as mutating and
/// must be confirmed. Fail-closed by design.
fn is_read_only_mcp_tool(tool: &str) -> bool {
    // Take the last path segment (drop any `server.`/`server/`/`ns:` prefix),
    // lowercase, and compare its leading token.
    let name = tool
        .rsplit(['.', '/', ':'])
        .next()
        .unwrap_or(tool)
        .to_ascii_lowercase();
    const READ_ONLY_VERBS: &[&str] = &[
        "list", "get", "read", "search", "fetch", "status", "describe", "find", "query", "show",
        "info", "count", "lookup", "view", "inspect", "resolve",
    ];
    READ_ONLY_VERBS.iter().any(|v| {
        name == *v || name.starts_with(&format!("{v}_")) || name.starts_with(&format!("{v}-"))
    })
}

fn action_label(a: &SupervisorAction) -> String {
    match a {
        SupervisorAction::Note { text } => {
            format!("note: {}", text.chars().take(160).collect::<String>())
        }
        SupervisorAction::McpTools => "mcp_tools".into(),
        SupervisorAction::McpCall { server, tool, .. } => format!("mcp_call {server}/{tool}"),
        SupervisorAction::StartDiscovery { dataset_selection } => {
            format!(
                "start_discovery {} {} {}",
                dataset_selection.identity().symbol_name(),
                dataset_selection.identity().timeframe().as_str(),
                dataset_selection.generation_id()
            )
        }
        SupervisorAction::StopDiscovery => "stop_discovery".into(),
        SupervisorAction::StartTraining { training_handoff } => {
            format!("start_training {training_handoff}")
        }
        SupervisorAction::StopTraining => "stop_training".into(),
        SupervisorAction::StartLive { portfolio_paths } => {
            format!("start_live ×{}", portfolio_paths.len())
        }
        SupervisorAction::StopLive => "stop_live (all)".into(),
        SupervisorAction::UpdateSettings { .. } => "update_settings".into(),
        SupervisorAction::ProposeClose { position_id, .. } => {
            format!("propose_close #{position_id}")
        }
        SupervisorAction::FetchUrl { url } => {
            format!("fetch_url {}", url.chars().take(120).collect::<String>())
        }
    }
}

/// Execute one whitelisted action by CALLING THE SAME HANDLERS the UI uses —
/// every server-side validation, clamp and gate applies to the agent too.
async fn execute(state: &AppApiState, action: SupervisorAction) -> Result<String> {
    use crate::server::{autonomous, engines_control, settings};
    match action {
        SupervisorAction::Note { text } => Ok(format!("noted: {text}")),

        SupervisorAction::StartDiscovery { dataset_selection } => {
            let response = engines_control::discovery_start(
                State(state.clone()),
                Some(Json(engines_control::StartJobBody {
                    dataset_selection: Some(dataset_selection),
                    ..Default::default()
                })),
            )
            .await;
            Ok(format!(
                "discovery start → {}",
                action_response(response).await?
            ))
        }
        SupervisorAction::StopDiscovery => {
            let _ = engines_control::discovery_stop(State(state.clone())).await;
            Ok("discovery stop requested".into())
        }
        SupervisorAction::StartTraining { training_handoff } => {
            let response = engines_control::training_start(
                State(state.clone()),
                Json(engines_control::TrainingStartBody {
                    training_handoff,
                    mode: crate::app_services::training::TrainingMode::TrainModels,
                }),
            )
            .await;
            Ok(format!(
                "training start → {}",
                action_response(response).await?
            ))
        }
        SupervisorAction::StopTraining => {
            let _ = engines_control::training_stop(State(state.clone())).await;
            Ok("training stop requested".into())
        }

        SupervisorAction::StartLive { portfolio_paths } => {
            let body: autonomous::StartLiveBody =
                serde_json::from_value(serde_json::json!({ "portfolio_paths": portfolio_paths }))?;
            let resp = autonomous::start_live(State(state.clone()), Json(body)).await;
            Ok(format!("live start → {}", action_response(resp).await?))
        }
        SupervisorAction::StopLive => {
            let resp = autonomous::stop_live(State(state.clone())).await;
            Ok(format!("live stop-all → {}", action_response(resp).await?))
        }

        SupervisorAction::UpdateSettings { payload } => {
            let dto: settings::SettingsUpdateDto = serde_json::from_value(payload)
                .context("payload is not a valid settings update")?;
            let resp = settings::update_settings(State(state.clone()), Json(dto)).await;
            Ok(format!(
                "settings update → {}",
                action_response(resp).await?
            ))
        }

        SupervisorAction::ProposeClose {
            position_id,
            reason,
        } => {
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
            Ok(format!(
                "fetched {} chars: {}",
                text.len(),
                text.chars().take(1500).collect::<String>()
            ))
        }

        SupervisorAction::McpTools => {
            let base = mcp_sidecar_url();
            let resp = reqwest::Client::new()
                .get(format!("{base}/tools"))
                .timeout(std::time::Duration::from_secs(15))
                .send()
                .await
                .context(MCP_SIDECAR_UNREACHABLE)?
                .error_for_status()?;
            let v: serde_json::Value = resp.json().await?;
            Ok(format!(
                "MCP tools: {}",
                serde_json::to_string(&v)
                    .unwrap_or_default()
                    .chars()
                    .take(2000)
                    .collect::<String>()
            ))
        }

        SupervisorAction::McpCall { server, tool, args } => {
            // Audit S02: the LLM may execute ONLY read-only MCP tools directly.
            // Anything mutating/unknown is queued as a pending action so the
            // OPERATOR confirms before it runs (place/cancel orders, write/
            // delete files, …) — the same protection money moves already get.
            if !is_read_only_mcp_tool(&tool) {
                let id = crate::app_services::pending_actions::propose(
                    crate::app_services::pending_actions::ActionKind::McpCall {
                        server: server.clone(),
                        tool: tool.clone(),
                        args: args.clone(),
                    },
                    format!("Supervisor requested MCP {server}/{tool} (not read-only)"),
                )?;
                return Ok(format!(
                    "MCP {server}/{tool} is not a read-only tool — queued for OPERATOR \
                     approval (action {id}); it will NOT run until the operator confirms."
                ));
            }
            let base = mcp_sidecar_url();
            let resp = reqwest::Client::new()
                .post(format!("{base}/call"))
                .timeout(std::time::Duration::from_secs(60))
                .json(&serde_json::json!({ "server": server, "tool": tool, "args": args }))
                .send()
                .await
                .context(MCP_SIDECAR_UNREACHABLE)?
                .error_for_status()?;
            let v: serde_json::Value = resp.json().await?;
            Ok(format!(
                "mcp_call {server}/{tool} → {}",
                serde_json::to_string(&v)
                    .unwrap_or_default()
                    .chars()
                    .take(2000)
                    .collect::<String>()
            ))
        }
    }
}

async fn action_response(resp: axum::response::Response) -> Result<String> {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .context("Could not read the backend action result; verify the actual engine state")?;
    let detail = String::from_utf8_lossy(&body);
    if !status.is_success() {
        anyhow::bail!(
            "HTTP {status}: {}",
            detail.chars().take(4000).collect::<String>()
        );
    }
    // Batch starts may return HTTP 200 after starting only SOME engines.
    // Preserve that distinction: a partial execution is not full success.
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body)
        && let Some(failed) = value.get("failed").and_then(serde_json::Value::as_array)
        && !failed.is_empty()
    {
        let started = value
            .get("started")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        anyhow::bail!(
            "Partial execution: {started} engines started, {} failed; inspect running engines. {}",
            failed.len(),
            serde_json::to_string(failed)?
                .chars()
                .take(4000)
                .collect::<String>()
        );
    }
    Ok(format!("OK {status}"))
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

#[cfg(test)]
mod s02_tests {
    use super::is_read_only_mcp_tool;

    #[test]
    fn read_only_tools_are_allowed_directly() {
        for t in [
            "list_positions",
            "get_account",
            "read_file",
            "search_web",
            "fetch",
            "status",
            "describe-tool",
            "find_symbol",
            "query",
            "ctrader.get_orders",
            "fs/read_file",
            "ns:list",
        ] {
            assert!(is_read_only_mcp_tool(t), "{t} should be read-only");
        }
    }

    #[test]
    fn mutating_or_unknown_tools_require_approval() {
        // Fail-closed: anything not clearly read-only must be gated.
        for t in [
            "place_order",
            "cancel_order",
            "write_file",
            "delete_file",
            "modify_position",
            "send_email",
            "transfer",
            "execute",
            "run",
            "ctrader.close_position",
            "fs/write",
            "do_something_weird",
        ] {
            assert!(!is_read_only_mcp_tool(t), "{t} must require approval");
        }
    }

    #[test]
    fn read_only_prefix_must_be_a_whole_token() {
        // "getaway"/"reader" must NOT count as read-only via substring.
        assert!(!is_read_only_mcp_tool("getaway_launch"));
        assert!(!is_read_only_mcp_tool("listen_and_trade"));
    }
}

#[cfg(test)]
mod operation_tests {
    use super::*;
    use crate::app_services::ctrader_live_auth::CTraderEnvironment;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn cancelling_a_cycle_releases_its_running_state() {
        let running = AtomicBool::new(false);
        let mut cycle = Box::pin(run_exclusive_cycle(
            &running,
            std::future::pending::<Result<()>>(),
        ));
        assert!(futures::poll!(cycle.as_mut()).is_pending());
        assert!(running.load(Ordering::SeqCst));
        drop(cycle);
        assert!(
            !running.load(Ordering::SeqCst),
            "a cancelled request must not permanently disable the supervisor"
        );
        assert!(
            run_exclusive_cycle(&running, async { Ok(()) })
                .await
                .is_ok()
        );
    }

    #[test]
    fn malformed_or_unknown_actions_are_not_reported_as_no_work() {
        for reply in [
            "The service is unavailable.",
            "[{\"action\":\"open_unchecked_trade\"}]",
            "[{\"action\":\"note\"}]",
            "[{\"action\":\"note\",\"text\":",
        ] {
            assert!(parse_actions(reply).is_err(), "silently accepted: {reply}");
        }
        assert!(parse_actions("[]").unwrap().is_empty());
        assert_eq!(
            parse_actions("```json\n[{\"action\":\"note\",\"text\":\"No change\"}]\n```")
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn research_actions_require_exact_current_selectors_and_reject_old_fields() {
        use neoethos_data::{
            BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
            SelectedDatasetGenerationV1,
        };
        let selection = SelectedDatasetGenerationV1::new(
            CanonicalDatasetIdentity::external(
                "test",
                "EURUSD",
                CanonicalTimeframe::M5,
                BarTimestampConvention::BarOpen,
            )
            .unwrap(),
            format!("g1-{}.vortex", "1".repeat(64)),
            "2".repeat(64),
        )
        .unwrap();
        let discovery =
            serde_json::json!({"action":"start_discovery", "dataset_selection": selection});
        let training =
            serde_json::json!({"action":"start_training", "training_handoff": "a".repeat(64)});
        for valid in [discovery, training] {
            assert!(serde_json::from_value::<SupervisorAction>(valid.clone()).is_ok());
            for field in ["symbol", "base_tf", "training_after_success"] {
                let mut ambiguous = valid.clone();
                ambiguous[field] = serde_json::json!("old-selector");
                assert!(serde_json::from_value::<SupervisorAction>(ambiguous).is_err());
            }
        }
        for action in ["start_discovery", "start_training"] {
            assert!(
                serde_json::from_value::<SupervisorAction>(serde_json::json!({
                    "action": action, "symbol":"EURUSD", "base_tf":"M5",
                }))
                .is_err()
            );
        }
        assert!(SYSTEM_PROMPT.contains("intelligence.trainingHandoffs"));
        assert!(SYSTEM_PROMPT.contains("dataInventory.datasets"));
    }

    #[tokio::test]
    async fn inventory_failures_remain_visible_instead_of_becoming_empty_success() {
        let failure = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error":"inventory denied"})),
        )
            .into_response();
        assert_eq!(
            inventory_context(failure).await["error"],
            "inventory denied"
        );
        let malformed = inventory_context("not JSON".into_response()).await;
        assert!(
            malformed["error"]
                .as_str()
                .unwrap()
                .contains("not valid JSON")
        );
        let oversized = inventory_context("x".repeat(1024 * 1024 + 1).into_response()).await;
        assert!(
            oversized["error"]
                .as_str()
                .unwrap()
                .contains("context limit")
        );
    }

    #[tokio::test]
    async fn backend_rejections_are_failed_actions_with_the_actual_reason() {
        let response = (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error":"position identity changed"})),
        )
            .into_response();
        let error = action_response(response)
            .await
            .expect_err("a rejected backend action is not a completed action");
        assert!(error.to_string().contains("position identity changed"));
    }

    #[tokio::test]
    async fn partial_batch_start_is_not_full_success() {
        let response = Json(serde_json::json!({
            "started": ["one-portfolio"],
            "failed": [{"portfolio":"another", "error":"missing execution authority"}],
        }))
        .into_response();
        let error = action_response(response).await.unwrap_err().to_string();
        assert!(error.contains("1 engines started, 1 failed"));
        assert!(error.contains("missing execution authority"));
        assert!(error.contains("inspect running engines"));

        let success = Json(serde_json::json!({"started":["one"],"failed":[]})).into_response();
        assert!(action_response(success).await.is_ok());
    }

    #[tokio::test]
    async fn an_overlapping_request_cannot_clear_the_active_cycle() {
        let running = AtomicBool::new(false);
        let mut first = Box::pin(run_exclusive_cycle(
            &running,
            std::future::pending::<Result<()>>(),
        ));
        assert!(futures::poll!(first.as_mut()).is_pending());
        assert!(
            run_exclusive_cycle(&running, async { Ok(()) })
                .await
                .is_err()
        );
        assert!(running.load(Ordering::SeqCst));
        drop(first);
        assert!(!running.load(Ordering::SeqCst));
        assert!(
            run_exclusive_cycle(&running, async {
                Err::<(), _>(anyhow::anyhow!("cycle failure"))
            })
            .await
            .is_err()
        );
        assert!(!running.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn observation_contains_positions_and_nullable_quote_evidence() {
        use crate::app_services::live_spots;
        use crate::server::state::{AccountSnapshotPayload, PositionPayload};

        struct ClearQuotes;
        impl Drop for ClearQuotes {
            fn drop(&mut self) {
                live_spots::clear();
            }
        }
        let _quotes = ClearQuotes;
        live_spots::clear();
        // Synthetic cached data only: no streamer, broker call or order.
        live_spots::update_tick(101, "EURUSD", Some(1.125), None, None);
        let state = AppApiState::new();
        state
            .set_account(AccountSnapshotPayload {
                source_account_id: 42,
                source_environment:
                    crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo,
                balance: 1000.0,
                equity: 1012.5,
                free_margin: 900.0,
                used_margin: 112.5,
                currency: "EUR".into(),
                fetched_at_unix_ms: 123456,
                positions: vec![PositionPayload {
                    position_id: 42,
                    volume_units: 100000,
                    symbol: "EURUSD".into(),
                    side: "BUY".into(),
                    volume: 1000.0,
                    open_timestamp_ms: Some(123000),
                    pnl_pips: None,
                    pnl_usd: 12.5,
                    entry_price: Some(1.1),
                    stop_loss: Some(1.09),
                    take_profit: Some(1.15),
                    volume_lots: Some(0.01),
                }],
            })
            .await;
        let value = serde_json::to_value(
            observation_with_scope(&state, || Ok((42, CTraderEnvironment::Demo))).await,
        )
        .unwrap();
        assert_eq!(value["account"]["sourceAccountId"], "42");
        assert_eq!(value["account"]["sourceEnvironment"], "Demo");
        assert_eq!(value["account"]["positions"][0]["positionId"], 42);
        assert_eq!(value["account"]["positions"][0]["stopLoss"], 1.09);
        assert_eq!(value["account"]["positions"][0]["takeProfit"], 1.15);
        assert_eq!(value["account"]["positions"][0]["pnlUsd"], 12.5);
        assert_eq!(value["account"]["currency"], "EUR");
        assert_eq!(value["account"]["fetchedAtUnixMs"], 123456);
        assert_eq!(value["market"]["spots"][0]["bid"], 1.125);
        assert!(value["market"]["spots"][0]["ask"].is_null());
        assert!(value["market"]["spots"][0]["brokerTimestampMs"].is_null());
        assert!(
            value["market"]["spots"][0]["receivedAtUnixMs"]
                .as_i64()
                .unwrap()
                > 0
        );
        assert!(value["liveEngineError"].is_null());
        assert!(value["accountFailure"].is_null());
        let account =
            crate::server::account::AccountSnapshotDto::from(state.account().await.unwrap());
        assert!(journal_scope_matches(&account, Some("42"), Some("Demo")));
        for (id, env) in [
            (Some("42"), Some("Live")),
            (Some("99"), Some("Demo")),
            (None, Some("Demo")),
            (Some("42"), None),
        ] {
            assert!(!journal_scope_matches(&account, id, env));
        }
        state
            .set_account_failure(&anyhow::anyhow!("broker snapshot rejected"))
            .await;
        let stale = serde_json::to_value(
            observation_with_scope(&state, || Ok((42, CTraderEnvironment::Demo))).await,
        )
        .unwrap();
        assert_eq!(stale["accountFailure"]["code"], "account_refresh_failed");
        assert_eq!(
            stale["accountFailure"]["detail"],
            "broker snapshot rejected"
        );
        assert_eq!(stale["account"]["positions"][0]["positionId"], 42);

        let switched = serde_json::to_value(
            observation_with_scope(&state, || Ok((42, CTraderEnvironment::Live))).await,
        )
        .unwrap();
        assert!(
            switched["account"].is_null(),
            "old Demo positions must not enter a Live observation"
        );
        let missing = serde_json::to_value(
            observation_with_scope(&AppApiState::new(), || {
                anyhow::bail!("No selected execution account")
            })
            .await,
        )
        .unwrap();
        assert!(
            missing["account"].is_null(),
            "no account must remain unknown, not zero exposure"
        );
        assert_eq!(
            missing["accountFailure"]["code"],
            "account_scope_unavailable"
        );
    }
}
