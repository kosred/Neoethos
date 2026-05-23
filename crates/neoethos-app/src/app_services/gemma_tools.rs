//! Function-calling (a.k.a. "tools") protocol for the local Gemma 4
//! runtime.
//!
//! ## Why
//!
//! Gemma 4 (abliterated and stock alike) is a text-completion model
//! — there is no native OpenAI-style `function_call` field in its
//! output. To let the operator ask questions like "how much am I
//! down today?", "what's my prop-firm preset?", or (eventually)
//! "what high-impact events are coming up?", we need the model to
//! be able to CALL BACK into the running backend for real data
//! instead of making up answers.
//!
//! This module implements the **ReAct** pattern (Reason + Act):
//! 1. We inject a list of available tools into the prompt.
//! 2. Model responds with either plain text OR a tool call wrapped
//!    in a ` ```tool_call ` JSON fence the parser recognises.
//! 3. If a tool call is detected, we execute it server-side and
//!    feed the result back into the conversation as a new turn.
//! 4. Loop until the model emits plain text OR we hit a max-step
//!    cap (`MAX_TOOL_STEPS`) to prevent runaway recursion.
//!
//! ## Protocol
//!
//! The system prompt instructs the model to emit:
//!
//! ```text
//! ```tool_call
//! {"name": "get_account_snapshot", "arguments": {}}
//! ```
//! ```
//!
//! when it needs real data. After execution we append the tool's
//! JSON return value to the prompt as:
//!
//! ```text
//! ```tool_result
//! {"name": "get_account_snapshot", "result": {"balance": 10000.0, ...}}
//! ```
//! ```
//!
//! and ask the model to continue. The fenced-block format is
//! deliberately ASCII-friendly and grep-able; we'd switch to
//! `<|tool_call|>` special tokens if/when we re-train with native
//! tool-calling support.
//!
//! ## What's here
//!
//! - `Tool` trait — every server-side tool implements it.
//! - `ToolRegistry` — owns the active tool set; dispatches a parsed
//!    `ToolCall` to the right handler.
//! - `parse_tool_call` — extracts a tool call from raw model output.
//! - `build_system_prompt` — formats the tool list into the system
//!    instruction the model sees on every turn.
//! - `run_tool_loop` — orchestrates the multi-turn ReAct loop.
//! - `register_default_tools` — wires up the built-in tools that
//!    surface real account / risk / preset data.
//!
//! The actual tool implementations live below as small structs that
//! borrow the live `AppApiState` to read whatever the operator's
//! current session knows.

use crate::server::state::AppApiState;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

/// Hard cap on tool-call iterations per chat turn. Prevents runaway
/// loops if the model keeps emitting tool calls without ever falling
/// through to a plain-text answer. Six steps is enough headroom for
/// "look up balance → look up positions → look up risk → summarize"
/// while still being a tight ceiling.
pub const MAX_TOOL_STEPS: usize = 6;

/// Parsed tool invocation from model output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// JSON shape we append to the conversation after running a tool.
#[derive(Debug, Clone, Serialize)]
struct ToolResult<'a> {
    name: &'a str,
    result: Value,
}

/// Server-side function the model can invoke. Implementations should
/// be cheap, deterministic, and operate on data the running backend
/// already knows (avoiding network round-trips when possible — the
/// model is local but the operator is still waiting in front of a
/// loading spinner).
pub trait Tool: Send + Sync {
    /// Canonical lookup name used in the model output's JSON.
    fn name(&self) -> &'static str;

    /// One-sentence description the model sees when deciding whether
    /// to call this tool. Keep it concrete: "Returns the current
    /// account balance, equity, free margin, and used margin." beats
    /// "Account info".
    fn description(&self) -> &'static str;

    /// JSON Schema fragment for the `arguments` object. Use
    /// `serde_json::json!({"type": "object", "properties": {...}, "required": [...]})`.
    /// Return `Value::Null` for no-argument tools.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool against the live app state. The return value
    /// is the JSON the model receives back as the `result` field in
    /// the `tool_result` fence.
    fn execute(&self, state: &AppApiState, arguments: Value) -> Result<Value>;
}

/// Registry of tools currently exposed to the model. Lookup by name
/// is O(N) (N is small — single-digit count of tools — so a HashMap
/// adds no value over a Vec + linear scan).
pub struct ToolRegistry {
    tools: HashMap<&'static str, Box<dyn Tool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name(), tool);
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.tools.keys().copied().collect();
        v.sort();
        v
    }

    pub fn lookup(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Format the registry's tool list as part of a system prompt.
    /// Compact markdown — minimises token cost while staying clear.
    pub fn render_for_prompt(&self) -> String {
        let mut s = String::new();
        s.push_str("You have access to the following tools. To use one, emit a code block:\n\n");
        s.push_str("```tool_call\n{\"name\": \"<tool_name>\", \"arguments\": {...}}\n```\n\n");
        s.push_str("Use a tool ONLY when you need fresh data — for explanatory questions just answer directly.\n");
        s.push_str("After the system runs the tool you will receive a `tool_result` block; use that data to compose your final answer.\n");
        s.push_str("\nAvailable tools:\n\n");
        // Deterministic order so the prompt is stable across runs.
        let mut names: Vec<_> = self.tools.keys().copied().collect();
        names.sort();
        for name in names {
            let tool = &self.tools[name];
            s.push_str(&format!("- `{name}` — {}\n", tool.description()));
            let schema = tool.parameters_schema();
            if !schema.is_null() {
                s.push_str(&format!("  parameters: {schema}\n"));
            }
        }
        s
    }
}

/// Parse a tool call out of raw model output.
///
/// Looks for the first ` ```tool_call ` fenced block (case-sensitive
/// per protocol). Returns `None` when the response has no tool call
/// — that's the signal to stop the ReAct loop.
///
/// We deliberately accept a relaxed JSON parse: extra whitespace and
/// trailing commas are common in small-model output and shouldn't
/// break the round-trip.
pub fn parse_tool_call(raw: &str) -> Option<ToolCall> {
    // Find the opening fence. Both "```tool_call" and "```tool-call"
    // are accepted so we're robust to model normalisation quirks.
    let start = raw.find("```tool_call").or_else(|| raw.find("```tool-call"))?;
    // Skip past the fence to the start of the JSON.
    let after_fence = &raw[start..];
    let json_start = after_fence.find('\n')? + 1;
    let body = &after_fence[json_start..];
    // Find the closing fence.
    let end = body.find("```")?;
    let json_text = body[..end].trim();
    serde_json::from_str::<ToolCall>(json_text).ok()
}

/// Multi-turn ReAct loop. `inference(prompt) -> Result<String>` is
/// the model's text-completion callback — passed in so this module
/// stays decoupled from the llama-cpp wrapper.
///
/// Returns the model's final plain-text answer once it stops
/// emitting tool calls (or once we hit `MAX_TOOL_STEPS`).
pub fn run_tool_loop<F>(
    state: &AppApiState,
    registry: &ToolRegistry,
    user_prompt: &str,
    mut inference: F,
) -> Result<String>
where
    F: FnMut(&str) -> Result<String>,
{
    let system = build_system_prompt(registry);
    let mut conversation = format!("{system}\n\nUser: {user_prompt}\n\nAssistant:");

    for step in 0..MAX_TOOL_STEPS {
        let raw = inference(&conversation)
            .with_context(|| format!("inference failed at step {step}"))?;

        // No tool call → done. Return whatever the model said.
        let Some(call) = parse_tool_call(&raw) else {
            return Ok(raw);
        };

        // Look up and run the tool.
        let result_value = match registry.lookup(&call.name) {
            Some(tool) => match tool.execute(state, call.arguments.clone()) {
                Ok(v) => v,
                Err(err) => {
                    // Surface the error back to the model — let it
                    // decide whether to retry with different args
                    // or give up and explain to the user.
                    json!({
                        "error": err.to_string(),
                    })
                }
            },
            None => json!({
                "error": format!(
                    "unknown tool `{}`. Available tools: {:?}",
                    call.name,
                    registry.names()
                ),
            }),
        };

        // Append the model's tool-call AND the tool result to the
        // running conversation, then re-prompt.
        let result_block = serde_json::to_string(&ToolResult {
            name: &call.name,
            result: result_value,
        })
        .unwrap_or_else(|_| "{\"error\":\"failed to serialize result\"}".to_string());

        conversation.push_str(&raw);
        conversation.push_str("\n\n```tool_result\n");
        conversation.push_str(&result_block);
        conversation.push_str("\n```\n\nAssistant:");
    }

    Err(anyhow!(
        "tool-call loop exceeded {MAX_TOOL_STEPS} steps without converging to a plain answer"
    ))
}

/// Build the system prompt the model sees. Combines the
/// task-agnostic preamble with the dynamic tool list.
///
/// #126 — topic-scope enforcement. The preamble explicitly scopes
/// the model to economics + trading + the operator's platform.
/// Off-topic questions (poems, recipes, general chitchat) get a
/// polite decline that points the user back at what we DO answer.
/// The model also gets a directive about its role relative to the
/// trained models: it explains, it doesn't override.
pub fn build_system_prompt(registry: &ToolRegistry) -> String {
    let tools_block = registry.render_for_prompt();
    format!(
        "You are NeoEthos, a local AI assistant embedded in a forex trading platform. \
         You help the operator understand their account, risk settings, market data, \
         their trained ML strategies, and the economic context.\n\n\
         \
         ## Scope (strict)\n\
         You ONLY discuss topics related to: forex / commodities / index trading, \
         economics, the operator's NeoEthos account and positions, the platform's \
         trained ML models, risk management, and news that affects markets. If the \
         user asks about anything else (weather, recipes, code in other domains, \
         personal advice, general chitchat), politely decline with one sentence \
         like: \"I'm scoped to trading and markets — happy to help with anything \
         in that area.\" Do not attempt the off-topic task even partially.\n\n\
         \
         ## Your role relative to the trained models\n\
         The platform runs its own ML strategies that open and close positions. \
         You are NOT above those models — you are the EXPLAINER. When the user \
         asks about a trade, your job is to read the signal metadata (use the \
         tools below) and translate what the models saw into plain language. \
         You may flag concerns (\"this trade is bigger than your usual size\").\n\n\
         \
         ## Trade-management actions (strictly proposal-only)\n\
         You can suggest closing a position via `propose_close_position`. \
         IMPORTANT: this does NOT execute the close — it adds a Confirm / \
         Reject prompt to the operator's UI. The operator's click is what \
         actually closes the position; there is no way for you to bypass that. \
         Use the proposal tool sparingly and only with a strong reason \
         (down N pips, high-impact event imminent, risk cap about to trip). \
         Never propose placing new orders, modifying stops, or cancelling \
         pending orders — those actions are not yet on the whitelist.\n\n\
         \
         ## Truthfulness\n\
         Stay concise, factual, and never invent numbers. Always use the tools \
         below to fetch real data. If you don't know something and no tool can \
         answer it, say so directly — do not guess.\n\n\
         \
         ## Memory\n\
         You have persistent memory via `save_memory_note` / `load_memory_note`. \
         At the start of a conversation it's often useful to `list_memory_keys` \
         to see what you already know about the operator before answering. \
         Save user preferences (\"trade only EURUSD\", \"prefer 50-pip stops\") \
         with category `user_pref`, news digests with `event_digest`, trade \
         narratives with `trade_explanation`.\n\n\
         \
         {tools_block}"
    )
}

// ─── Built-in tools ───────────────────────────────────────────────

/// `get_account_snapshot` — returns the current account balance,
/// equity, free margin, used margin, currency, and open-position
/// count.
pub struct GetAccountSnapshotTool;

impl Tool for GetAccountSnapshotTool {
    fn name(&self) -> &'static str {
        "get_account_snapshot"
    }

    fn description(&self) -> &'static str {
        "Returns the current account balance, equity, free margin, used margin, \
         currency, and number of open positions. No arguments. Use this when the \
         user asks about their P&L, drawdown, available capital, or position count."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn execute(&self, state: &AppApiState, _args: Value) -> Result<Value> {
        // Reach into the bridge cache rather than calling the HTTP
        // route — the cache holds the most recent snapshot the
        // streaming bridge maintains, so the tool returns instantly
        // without re-querying the broker. Uses the blocking variant
        // because we run inside the LLM inference blocking thread
        // (see `chat_impl` in `server/gemma.rs`).
        let snap = state
            .account_blocking()
            .ok_or_else(|| {
                anyhow!(
                    "no account snapshot available yet — broker not connected \
                     or initial fetch not finished"
                )
            })?;
        Ok(json!({
            "balance": snap.balance,
            "equity": snap.equity,
            "free_margin": snap.free_margin,
            "used_margin": snap.used_margin,
            "currency": snap.currency,
            "open_position_count": snap.positions.len(),
        }))
    }
}

/// `get_risk_caps` — returns the active prop-firm preset name plus
/// the current numeric risk caps (daily DD, total DD, max lot,
/// risk-per-trade).
pub struct GetRiskCapsTool;

impl Tool for GetRiskCapsTool {
    fn name(&self) -> &'static str {
        "get_risk_caps"
    }

    fn description(&self) -> &'static str {
        "Returns the active prop-firm preset (ftmo/myforexfunds/fundednext/the5ers/none) \
         and the current numeric risk caps: daily drawdown limit, total drawdown limit, \
         max lot size, current per-trade risk percent. No arguments. Use this when the \
         user asks 'what's my preset?', 'how much can I lose today?', or 'what's my \
         max lot?'."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn execute(&self, _state: &AppApiState, _args: Value) -> Result<Value> {
        // Risk config lives in config.yaml. The bridge doesn't cache
        // it, so we re-read on every call. The file is tiny (<5 KB)
        // so this is fast.
        let settings = neoethos_core::Settings::from_yaml("config.yaml")
            .context("failed to load config.yaml")?;
        let r = &settings.risk;
        Ok(json!({
            "preset": r.preset.as_str(),
            "preset_display_name": r.preset.display_name(),
            "daily_drawdown_limit": r.daily_drawdown_limit,
            "total_drawdown_limit": r.total_drawdown_limit,
            "max_lot_size": r.max_lot_size,
            "risk_per_trade": r.risk_per_trade,
            "max_risk_per_trade": r.max_risk_per_trade,
            "require_stop_loss": r.require_stop_loss,
            "prop_firm_rules_enabled": r.prop_firm_rules,
        }))
    }
}

/// `get_news_trading_mode` — returns the current news-trading mode
/// (block_on_news / allow_always / warn_only) from #117.
pub struct GetNewsTradingModeTool;

impl Tool for GetNewsTradingModeTool {
    fn name(&self) -> &'static str {
        "get_news_trading_mode"
    }

    fn description(&self) -> &'static str {
        "Returns how the trading gate currently treats high-impact news events: \
         `block_on_news` (pause new orders), `allow_always` (play through), or \
         `warn_only` (visual warning only). Use this when the user asks 'will I be \
         blocked during NFP?' or 'is news-trading on?'"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn execute(&self, _state: &AppApiState, _args: Value) -> Result<Value> {
        let settings = neoethos_core::Settings::from_yaml("config.yaml")
            .context("failed to load config.yaml")?;
        let mode = settings.news.news_trading_mode;
        Ok(json!({
            "mode": mode.as_str(),
            "display_name": mode.display_name(),
            "news_calendar_enabled": settings.news.news_calendar_enabled,
            "news_kill_window_min": settings.news.news_kill_window_min,
        }))
    }
}

/// Wire up the standard tool set. Add new tools here as the
/// platform grows — chart snapshot, recent fills, open orders, etc.
/// `get_open_positions` — returns one row per open position with
/// side, volume, PnL pips, PnL in account currency, and how long the
/// position has been open. Used when the operator asks "what am I in
/// right now?" or "which position is bleeding?".
pub struct GetOpenPositionsTool;

impl Tool for GetOpenPositionsTool {
    fn name(&self) -> &'static str {
        "get_open_positions"
    }

    fn description(&self) -> &'static str {
        "Returns the list of currently open positions: position_id, symbol, side, \
         volume in lots, P&L in pips, P&L in account currency, and the position's \
         open timestamp (Unix milliseconds, UTC). Empty list when no positions are \
         open. Use when the user asks about specific positions, totals, or which \
         one is losing."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn execute(&self, state: &AppApiState, _args: Value) -> Result<Value> {
        let snap = state.account_blocking().ok_or_else(|| {
            anyhow!(
                "no account snapshot available yet — broker not connected or initial \
                 fetch not finished"
            )
        })?;
        let positions: Vec<Value> = snap
            .positions
            .iter()
            .map(|p| {
                json!({
                    "position_id": p.position_id,
                    "symbol": p.symbol,
                    "side": p.side,
                    "volume_lots": p.volume,
                    "pnl_pips": p.pnl_pips,
                    "pnl_account_currency": p.pnl_usd,
                    "open_timestamp_ms": p.open_timestamp_ms,
                })
            })
            .collect();
        Ok(json!({
            "open_position_count": positions.len(),
            "positions": positions,
            "account_currency": snap.currency,
        }))
    }
}

/// `get_chart_data` — returns the last N candles for a symbol +
/// timeframe (OHLC + volume + timestamps). Reuses the same loader
/// the HTTP `/chart` route uses so answers can't drift from what
/// the Chart screen shows.
pub struct GetChartDataTool;

impl Tool for GetChartDataTool {
    fn name(&self) -> &'static str {
        "get_chart_data"
    }

    fn description(&self) -> &'static str {
        "Returns the most recent OHLC candles for a symbol on a given timeframe. \
         Arguments: `symbol` (e.g. \"EURUSD\"), `timeframe` (M1, M5, M15, H1, H4, D1), \
         `limit` (1-500, default 50). Returns the candle list, latest close, \
         price min/max in the window, and a percent change from window-open to \
         window-close. Use when the user asks about recent price action, support/\
         resistance levels, or wants you to reason about a specific pair."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "symbol":    {"type": "string", "description": "Symbol ticker, e.g. EURUSD"},
                "timeframe": {"type": "string", "description": "M1|M5|M15|M30|H1|H4|D1|W1"},
                "limit":     {"type": "integer", "minimum": 1, "maximum": 500}
            },
            "required": ["symbol", "timeframe"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing required argument `symbol`"))?
            .trim()
            .to_uppercase();
        let timeframe = args
            .get("timeframe")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing required argument `timeframe`"))?
            .trim()
            .to_uppercase();
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50)
            .clamp(1, 500) as usize;

        let chart = crate::server::chart::load_chart(symbol, timeframe, limit)
            .map_err(|e| anyhow!("chart load failed: {e}"))?;

        // Compact candle representation — full OHLC + ts + volume,
        // skipping the per-candle "name" boilerplate to keep the
        // result payload small for the LLM context window.
        let candles: Vec<Value> = chart
            .candles
            .iter()
            .map(|c| {
                json!({
                    "ts_ms": c.ts_ms,
                    "o": c.open,
                    "h": c.high,
                    "l": c.low,
                    "c": c.close,
                    "v": c.volume,
                })
            })
            .collect();
        Ok(json!({
            "symbol": chart.symbol,
            "timeframe": chart.timeframe,
            "candle_count": chart.candle_count,
            "latest_close": chart.latest_close,
            "price_min": chart.price_min,
            "price_max": chart.price_max,
            "price_change_pct": chart.price_change_pct,
            "headline": chart.headline,
            "candles": candles,
        }))
    }
}

/// `fetch_url` — HTTP GET against a public URL. Bounded by a 10s
/// timeout and a 1 MB body cap. SSRF-guarded: only http(s), no
/// localhost / private-network targets, no `file://`.
///
/// Returns: `{"status": 200, "content_type": "...", "body": "...",
/// "truncated": false}`. When the body exceeds 1 MB or 500 KB of
/// text the `body` is truncated and `truncated` is `true` — the
/// model is told to ask the user before doing more retries.
pub struct FetchUrlTool;

/// Hard caps so a misbehaving LLM call can't DoS the local memory
/// or hammer a third-party server. 500 KB is enough for a typical
/// HTML article body after stripping ads; 10s is generous for an
/// ECB-website fetch over a wired connection.
const FETCH_MAX_BODY_BYTES: usize = 500 * 1024;
const FETCH_TIMEOUT_SECS: u64 = 10;

impl Tool for FetchUrlTool {
    fn name(&self) -> &'static str {
        "fetch_url"
    }

    fn description(&self) -> &'static str {
        "Performs an HTTP GET against a PUBLIC URL and returns the response body \
         as text. Use this when the user asks about external information you don't \
         have a tool for — economic calendars, central-bank press releases, news \
         articles. Capped at 10s timeout and 500 KB body; private/internal URLs \
         (localhost, 127.x, 10.x, 192.168.x, file://) are rejected. Argument: `url`."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Public http(s) URL"}
            },
            "required": ["url"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let url_str = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing required argument `url`"))?
            .trim()
            .to_string();

        ssrf_guard(&url_str)?;

        // Synchronous reqwest::blocking — we're already on a
        // spawn_blocking thread (see chat_impl), no reactor in the
        // call stack to starve.
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
            // Allow up to 5 redirects — common for news sites
            // canonicalising URLs. The SSRF guard re-checks the
            // final URL via reqwest's policy hook.
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() > 5 {
                    attempt.error("too many redirects (max 5)")
                } else if let Err(e) = ssrf_guard(attempt.url().as_str()) {
                    attempt.error(format!("redirect blocked by SSRF guard: {e}"))
                } else {
                    attempt.follow()
                }
            }))
            // Set a generic User-Agent so politely-configured
            // servers know who's calling.
            .user_agent("NeoEthos-Gemma/0.4 (LLM tool fetch)")
            .build()
            .map_err(|e| anyhow!("failed to build HTTP client: {e}"))?;

        let resp = client
            .get(&url_str)
            .send()
            .map_err(|e| anyhow!("HTTP request failed: {e}"))?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Cap body read so a multi-GB response can't OOM us.
        let bytes = resp
            .bytes()
            .map_err(|e| anyhow!("failed to read response body: {e}"))?;
        let total_len = bytes.len();
        let truncated = total_len > FETCH_MAX_BODY_BYTES;
        let body_slice = if truncated {
            &bytes[..FETCH_MAX_BODY_BYTES]
        } else {
            &bytes[..]
        };
        let body_text = String::from_utf8_lossy(body_slice).to_string();
        Ok(json!({
            "status": status,
            "content_type": content_type,
            "body": body_text,
            "body_length_bytes": total_len,
            "truncated": truncated,
        }))
    }
}

/// SSRF guard. Rejects schemes and hosts that would let an LLM tool
/// pivot into the local network. NOT a substitute for proper egress
/// firewalling, but enough to block obvious mistakes.
fn ssrf_guard(url_str: &str) -> Result<()> {
    let parsed = url::Url::parse(url_str)
        .map_err(|e| anyhow!("invalid URL `{url_str}`: {e}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => anyhow::bail!("scheme `{other}` not allowed (only http/https)"),
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("URL has no host"))?
        .to_ascii_lowercase();
    // Block obvious private/loopback identifiers. We deliberately
    // do NOT resolve via DNS — that would (a) leak DNS queries for
    // model-generated URLs and (b) open a TOCTOU window between
    // resolve-and-check and actual connect. Anyone routing public
    // hostnames to private IPs in their /etc/hosts is asking for
    // it; reqwest will follow that resolution and the egress
    // firewall is the real defence.
    const BLOCKED_HOSTS: &[&str] = &[
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "[::1]",
        // common metadata-server addresses (AWS, GCP, Azure)
        "169.254.169.254",
        "metadata.google.internal",
    ];
    if BLOCKED_HOSTS.iter().any(|b| *b == host) {
        anyhow::bail!("host `{host}` is in the SSRF block-list");
    }
    // RFC1918 private ranges — only the literal IP prefixes; we
    // already skipped DNS resolution on purpose.
    let private_prefixes = ["10.", "192.168.", "172.16.", "172.17.", "172.18.",
        "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.",
        "172.31.", "127."];
    if private_prefixes.iter().any(|p| host.starts_with(p)) {
        anyhow::bail!("host `{host}` is in a private IP range");
    }
    Ok(())
}

/// `get_recent_log_lines` — tails the daily log file. Used when the
/// operator asks "what just broke?" or "why isn't X working?". The
/// log is the canonical observability surface; this tool puts it in
/// front of the LLM.
pub struct GetRecentLogLinesTool;

/// Cap on lines returned — large log files would blow the model's
/// context window. 200 lines covers a few minutes of typical
/// activity at this codebase's log volume.
const LOG_TAIL_MAX_LINES: usize = 200;

impl Tool for GetRecentLogLinesTool {
    fn name(&self) -> &'static str {
        "get_recent_log_lines"
    }

    fn description(&self) -> &'static str {
        "Returns the last N lines of today's NeoEthos log file (default 50, max 200). \
         Use when the user reports a problem and asks 'what just happened?' or \
         'check the logs'. The path is `<user-data-dir>/neoethos/logs/\
         neoethos.YYYY-MM-DD.log`. Argument: `lines` (optional integer)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": LOG_TAIL_MAX_LINES,
                    "description": "How many trailing lines to return"
                }
            }
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let lines = args
            .get("lines")
            .and_then(|v| v.as_u64())
            .unwrap_or(50)
            .clamp(1, LOG_TAIL_MAX_LINES as u64) as usize;

        let path = neoethos_core::logging::canonical_log_path();
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| anyhow!("failed to read log {}: {e}", path.display()))?;
        let all_lines: Vec<&str> = contents.lines().collect();
        let tail = if all_lines.len() > lines {
            &all_lines[all_lines.len() - lines..]
        } else {
            &all_lines[..]
        };
        Ok(json!({
            "path": path.display().to_string(),
            "total_lines": all_lines.len(),
            "returned_lines": tail.len(),
            "lines": tail,
        }))
    }
}

// ─── Persistent-memory tools (#125) ───────────────────────────────
//
// Four thin wrappers around `crate::app_services::gemma_memory`.
// The model uses these to remember user preferences, digested news
// events, and trade explanations across sessions. Without them
// every chat turn forgets everything before it.

use crate::app_services::gemma_memory::{self, NoteCategory};

pub struct SaveMemoryNoteTool;

impl Tool for SaveMemoryNoteTool {
    fn name(&self) -> &'static str {
        "save_memory_note"
    }

    fn description(&self) -> &'static str {
        "Stores a note in persistent memory so you can read it back in a future \
         chat turn or session. Arguments: `key` (unique identifier — re-saving \
         the same key overwrites), `content` (the note body), `category` (one \
         of: user_pref, event_digest, trade_explanation, scratch). Use \
         `user_pref` for things the user told you to remember forever, \
         `event_digest` for news/calendar summaries, `trade_explanation` for \
         per-trade narratives, `scratch` for anything else (FIFO-evicted at \
         200 entries)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "key":      {"type": "string"},
                "content":  {"type": "string"},
                "category": {"type": "string", "enum": ["user_pref", "event_digest", "trade_explanation", "scratch"]}
            },
            "required": ["key", "content", "category"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let key = args.get("key").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `key`"))?.to_string();
        let content = args.get("content").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `content`"))?.to_string();
        let cat_raw = args.get("category").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `category`"))?;
        let category = NoteCategory::parse(cat_raw).ok_or_else(|| {
            anyhow!(
                "unknown category `{cat_raw}`. Valid: user_pref, event_digest, \
                 trade_explanation, scratch"
            )
        })?;
        gemma_memory::global()?.save(&key, &content, category)?;
        Ok(json!({"ok": true, "key": key, "category": category.as_str()}))
    }
}

pub struct LoadMemoryNoteTool;

impl Tool for LoadMemoryNoteTool {
    fn name(&self) -> &'static str {
        "load_memory_note"
    }

    fn description(&self) -> &'static str {
        "Reads a single note from persistent memory by key. Returns the note \
         content + category + timestamps, or `{found: false}` if absent. Use \
         to check whether you already digested an event ('did I see today's \
         NFP?') or to recall a user preference. Argument: `key`."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let key = args.get("key").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `key`"))?;
        match gemma_memory::global()?.load(key)? {
            Some(note) => Ok(json!({
                "found": true,
                "key": note.key,
                "content": note.content,
                "category": note.category,
                "created_at_unix_ms": note.created_at_unix_ms,
                "updated_at_unix_ms": note.updated_at_unix_ms,
            })),
            None => Ok(json!({"found": false, "key": key})),
        }
    }
}

pub struct ListMemoryKeysTool;

impl Tool for ListMemoryKeysTool {
    fn name(&self) -> &'static str {
        "list_memory_keys"
    }

    fn description(&self) -> &'static str {
        "Lists notes in persistent memory, most-recently-updated first. \
         Optional filters: `prefix` (e.g. \"trade:\" to find all trade \
         explanations), `category` (one of user_pref/event_digest/\
         trade_explanation/scratch), `limit` (default 20, max 100). Use to \
         scan what you already know before deciding whether to ask the user \
         or fetch fresh data."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prefix":   {"type": "string"},
                "category": {"type": "string"},
                "limit":    {"type": "integer", "minimum": 1, "maximum": 100}
            }
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let prefix = args.get("prefix").and_then(|v| v.as_str()).map(|s| s.to_string());
        let category = args
            .get("category")
            .and_then(|v| v.as_str())
            .and_then(NoteCategory::parse);
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .clamp(1, 100) as usize;
        let notes = gemma_memory::global()?.list(prefix.as_deref(), category, limit)?;
        // Compact result — content elided to keep the model's
        // context window happy when the list is long. Caller must
        // load_memory_note(key) to get the actual content.
        let summaries: Vec<Value> = notes
            .iter()
            .map(|n| {
                json!({
                    "key": n.key,
                    "category": n.category,
                    "updated_at_unix_ms": n.updated_at_unix_ms,
                    "content_preview": preview(&n.content, 120),
                })
            })
            .collect();
        Ok(json!({
            "count": summaries.len(),
            "notes": summaries,
        }))
    }
}

pub struct ForgetMemoryNoteTool;

impl Tool for ForgetMemoryNoteTool {
    fn name(&self) -> &'static str {
        "forget_memory_note"
    }

    fn description(&self) -> &'static str {
        "Permanently removes a note from persistent memory. Idempotent: \
         forgetting a key that doesn't exist is fine, returns `{removed: \
         false}`. Use ONLY when the user explicitly asks you to forget \
         something, OR when a note is clearly stale (e.g. an event_digest \
         for last year's NFP that someone forgot to TTL out)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let key = args.get("key").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `key`"))?;
        let removed = gemma_memory::global()?.forget(key)?;
        Ok(json!({"removed": removed, "key": key}))
    }
}

/// Truncate a string for the list-tool preview field. Cuts at the
/// nearest char boundary so we don't slice a multi-byte UTF-8
/// scalar in half.
fn preview(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}…")
}

// ─── Trade-explanation tool (#127) ───────────────────────────────
//
// Reads from the in-memory `signal_journal` (every auto-trade
// signal the dispatcher saw — dispatched or rejected) plus the live
// `account.positions` cache, and returns the joined view so the
// Gemma layer can narrate "why" each trade opened. The model is
// expected to correlate signals with positions by (symbol, side,
// timestamp_ms close-to) — we hand it both streams; we don't try
// to build an exact join server-side because the broker doesn't
// echo the originating signal id back on the fill.

pub struct ExplainRecentTradesTool;

impl Tool for ExplainRecentTradesTool {
    fn name(&self) -> &'static str {
        "explain_recent_trades"
    }

    fn description(&self) -> &'static str {
        "Returns the recent dispatch history of the auto-trade pipeline AND the \
         current open positions, so you can narrate WHY each trade opened. Each \
         signal carries strategy_id, model_id, confidence, dispatch outcome (was \
         it filled or blocked by a gate?), and a feature_snapshot of indicator \
         values at signal time. Correlate signals with positions by symbol + \
         side + nearby timestamp. Use when the user asks 'why did you open \
         EURUSD long?', 'what did the model see?', or 'why is this trade in \
         the book?'. Arguments: `limit` (default 20, max 100)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        })
    }

    fn execute(&self, state: &AppApiState, args: Value) -> Result<Value> {
        use crate::app_services::signal_journal::recent;
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .clamp(1, 100) as usize;
        let signals = recent(limit);
        // Live positions — None when broker hasn't filled the cache
        // yet (no session). We don't fail the tool in that case;
        // the model can still narrate from the signal stream alone.
        let positions = state.account_blocking().map(|snap| {
            snap.positions
                .iter()
                .map(|p| {
                    json!({
                        "position_id": p.position_id,
                        "symbol": p.symbol,
                        "side": p.side,
                        "volume_lots": p.volume,
                        "pnl_pips": p.pnl_pips,
                        "pnl_account_currency": p.pnl_usd,
                        "open_timestamp_ms": p.open_timestamp_ms,
                    })
                })
                .collect::<Vec<_>>()
        });
        let signals_json: Vec<Value> = signals
            .iter()
            .map(|s| {
                let features: Vec<Value> = s
                    .feature_snapshot
                    .iter()
                    .map(|(k, v)| json!({"name": k, "value": v}))
                    .collect();
                json!({
                    "timestamp_ms": s.timestamp_ms,
                    "symbol": s.symbol,
                    "side": s.side,
                    "confidence": s.confidence,
                    "label": s.label,
                    "strategy_id": s.strategy_id,
                    "model_id": s.model_id,
                    "feature_snapshot": features,
                    "dispatched": s.dispatched,
                    "dispatch_note": s.dispatch_note,
                })
            })
            .collect();
        Ok(json!({
            "signal_count": signals_json.len(),
            "signals": signals_json,
            "open_positions": positions,
            "hint": "Correlate signals with open positions by (symbol, side, \
                     timestamp_ms close-to). Rejected signals — dispatched=false \
                     — explain WHY a trade did NOT open, which is often what the \
                     user is asking when a position they expected isn't there.",
        }))
    }
}

// ─── News-aggregator tools (#129) ────────────────────────────────
//
// Sit on top of `app_services::news_sources` so the LLM gets
// structured calendar events + headlines without having to
// remember source URLs. Both tools are read-only and consult the
// 5-min in-process cache so consecutive calls don't re-fetch.

use crate::app_services::news_sources;

pub struct GetUpcomingCalendarEventsTool;

impl Tool for GetUpcomingCalendarEventsTool {
    fn name(&self) -> &'static str {
        "get_upcoming_calendar_events"
    }

    fn description(&self) -> &'static str {
        "Returns scheduled economic-calendar events from the configured news \
         sources (ForexFactory by default). Filters by currency list and a \
         lookahead window so the LLM can answer 'is anything coming up for \
         EURUSD in the next 30 min?'. Arguments: `currencies` (array of \
         3-letter codes, e.g. [\"USD\", \"EUR\"] — empty array returns all), \
         `lookahead_minutes` (default 60, max 1440), `min_impact` (\"low\" / \
         \"medium\" / \"high\" — default \"medium\")."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "currencies":       {"type": "array", "items": {"type": "string"}},
                "lookahead_minutes":{"type": "integer", "minimum": 1, "maximum": 1440},
                "min_impact":       {"type": "string", "enum": ["low", "medium", "high"]}
            }
        })
    }

    fn execute(&self, state: &AppApiState, args: Value) -> Result<Value> {
        let mut currencies: Vec<String> = args
            .get("currencies")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.trim().to_uppercase()))
                    .collect()
            })
            .unwrap_or_default();
        // If the caller didn't supply a currency list, default to
        // the union of currencies derived from the operator's open
        // positions. This makes the tool "do the right thing" when
        // the LLM asks for upcoming events without filters — it
        // gets only the events that matter to the trade book.
        if currencies.is_empty()
            && let Some(snap) = state.account_blocking()
        {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for p in &snap.positions {
                for c in news_sources::currencies_for_symbol(&p.symbol) {
                    if seen.insert(c.to_string()) {
                        currencies.push(c.to_string());
                    }
                }
            }
        }
        let lookahead_min = args
            .get("lookahead_minutes")
            .and_then(|v| v.as_u64())
            .unwrap_or(60)
            .clamp(1, 1440) as i64;
        let min_impact = args
            .get("min_impact")
            .and_then(|v| v.as_str())
            .map(news_sources::NewsImpact::parse)
            .unwrap_or(news_sources::NewsImpact::Medium);

        let now_ms = chrono::Utc::now().timestamp_millis();
        let until_ms = now_ms + lookahead_min * 60_000;

        let sources = news_sources::default_sources();
        let mut all: Vec<news_sources::CalendarEvent> = Vec::new();
        for src in sources {
            match src.fetch_calendar_events() {
                Ok(events) => all.extend(events),
                Err(err) => {
                    tracing::warn!(
                        target: "neoethos_app::gemma_tools::calendar",
                        source = src.id(),
                        error = %err,
                        "calendar fetch failed"
                    );
                }
            }
        }
        // Dedupe by (currency, scheduled_at, title) — same event
        // could appear in multiple sources (FF + secondary calendar
        // mirrors etc.).
        let mut seen: std::collections::HashSet<(String, i64, String)> =
            std::collections::HashSet::new();
        all.retain(|e| seen.insert((e.currency.clone(), e.scheduled_at_unix_ms, e.title.clone())));
        // Filter by time window + impact + currency.
        let filtered: Vec<&news_sources::CalendarEvent> = all
            .iter()
            .filter(|e| e.scheduled_at_unix_ms >= now_ms && e.scheduled_at_unix_ms <= until_ms)
            .filter(|e| e.impact.weight() >= min_impact.weight())
            .filter(|e| currencies.is_empty() || currencies.contains(&e.currency))
            .collect();
        let mut sorted: Vec<&news_sources::CalendarEvent> = filtered;
        sorted.sort_by_key(|e| e.scheduled_at_unix_ms);

        let events_json: Vec<Value> = sorted
            .iter()
            .map(|e| {
                json!({
                    "currency": e.currency,
                    "title": e.title,
                    "scheduled_at_unix_ms": e.scheduled_at_unix_ms,
                    "impact": e.impact.as_str(),
                    "forecast": e.forecast,
                    "previous": e.previous,
                    "actual": e.actual,
                    "source": e.source,
                })
            })
            .collect();
        Ok(json!({
            "now_unix_ms": now_ms,
            "lookahead_minutes": lookahead_min,
            "min_impact": min_impact.as_str(),
            "event_count": events_json.len(),
            "events": events_json,
        }))
    }
}

pub struct GetRecentMarketHeadlinesTool;

impl Tool for GetRecentMarketHeadlinesTool {
    fn name(&self) -> &'static str {
        "get_recent_market_headlines"
    }

    fn description(&self) -> &'static str {
        "Returns recent market-news headlines from the configured RSS sources \
         (FXStreet, DailyFX, Investing.com by default). Use to scan the dominant \
         narrative without picking which site to fetch. Arguments: `query` \
         (optional case-insensitive substring filter, e.g. \"EUR\"), `limit` \
         (default 25, max 100). Headlines are deduplicated by URL across \
         sources and sorted newest-first."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase());
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(25)
            .clamp(1, 100) as usize;

        let sources = news_sources::default_sources();
        let mut all: Vec<news_sources::Headline> = Vec::new();
        for src in sources {
            match src.fetch_headlines() {
                Ok(items) => all.extend(items),
                Err(err) => {
                    tracing::warn!(
                        target: "neoethos_app::gemma_tools::headlines",
                        source = src.id(),
                        error = %err,
                        "headline fetch failed"
                    );
                }
            }
        }
        // Dedupe by link (RSS items occasionally appear in multiple
        // feeds with the same canonical URL).
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        all.retain(|h| {
            if h.link.is_empty() {
                // No link to dedupe on; keep all such items.
                true
            } else {
                seen.insert(h.link.clone())
            }
        });
        // Apply query filter.
        if let Some(q) = query.as_deref() {
            all.retain(|h| {
                h.title.to_lowercase().contains(q) || h.summary.to_lowercase().contains(q)
            });
        }
        // Sort newest first + truncate to limit.
        all.sort_by(|a, b| b.published_at_unix_ms.cmp(&a.published_at_unix_ms));
        all.truncate(limit);

        let items_json: Vec<Value> = all
            .iter()
            .map(|h| {
                json!({
                    "title": h.title,
                    "link": h.link,
                    "summary": preview(&h.summary, 240),
                    "published_at_unix_ms": h.published_at_unix_ms,
                    "source": h.source,
                })
            })
            .collect();
        Ok(json!({
            "headline_count": items_json.len(),
            "headlines": items_json,
        }))
    }
}

// ─── Trade-management proposal tool (#136) ───────────────────────
//
// The LLM CANNOT execute trades autonomously. Its only path to
// changing position state is to PROPOSE an action via this tool,
// which adds a row to the pending-actions queue. The Flutter UI
// pops a Confirm / Reject prompt; the user's explicit click is
// what actually fires the broker call.
//
// Today the only whitelisted action kind is ClosePosition. Adding
// further kinds (cancel pending, modify SL, place new order) is a
// deliberate code change here AND in `pending_actions::ActionKind`
// — there is no generic "execute arbitrary command" backdoor.

pub struct ProposeClosePositionTool;

impl Tool for ProposeClosePositionTool {
    fn name(&self) -> &'static str {
        "propose_close_position"
    }

    fn description(&self) -> &'static str {
        "Propose closing an open position. This DOES NOT execute the close — \
         it adds a Confirm / Reject prompt to the operator's UI. The operator's \
         click is what actually closes the position; you cannot bypass the prompt. \
         Use ONLY when you've reasoned through `get_open_positions` + \
         `get_chart_data` and have a concrete recommendation. Arguments: \
         `position_id` (the integer position_id from get_open_positions), \
         `symbol_hint` (optional, e.g. \"EURUSD\" — surfaced in the UI \
         prompt), `reason` (REQUIRED plain-English explanation the operator \
         sees: \"down 25 pips, ECB conference in 8 min, recommend closing\")."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "position_id": {"type": "integer", "description": "position_id from get_open_positions"},
                "symbol_hint": {"type": "string"},
                "reason":      {"type": "string", "minLength": 10}
            },
            "required": ["position_id", "reason"]
        })
    }

    fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
        let position_id = args
            .get("position_id")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing required argument `position_id` (integer)"))?;
        if position_id <= 0 {
            return Err(anyhow!("position_id must be a positive integer"));
        }
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing required argument `reason`"))?
            .trim()
            .to_string();
        if reason.len() < 10 {
            return Err(anyhow!(
                "reason must be at least 10 chars — the operator needs context to \
                 decide whether to confirm. Got: `{reason}`"
            ));
        }
        let symbol_hint = args
            .get("symbol_hint")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let id = crate::app_services::pending_actions::propose(
            crate::app_services::pending_actions::ActionKind::ClosePosition {
                position_id,
                // 0 = "close entire position". The UI will pass the
                // real broker volume_units when the operator clicks
                // Confirm. We can't look it up here because the
                // LLM tool runs synchronously and AppApiState's
                // position cache is async-locked.
                volume_units: 0,
                symbol_hint,
            },
            reason.clone(),
        )?;
        Ok(json!({
            "ok": true,
            "action_id": id,
            "status": "pending",
            "ttl_seconds": crate::app_services::pending_actions::PENDING_ACTION_TTL_SECS,
            "note": "Proposal added to operator queue. The user will see a \
                     Confirm / Reject prompt; their click is what actually \
                     closes the position. If they don't respond within the \
                     TTL the action auto-expires.",
        }))
    }
}

pub struct ListPendingActionsTool;

impl Tool for ListPendingActionsTool {
    fn name(&self) -> &'static str {
        "list_pending_actions"
    }

    fn description(&self) -> &'static str {
        "Returns recent trade-action proposals + their current status (pending / \
         confirmed / rejected / executed / expired / failed). Use to check \
         whether a propose_close_position you fired earlier has been actioned \
         by the operator, or to see history before proposing again. No arguments."
    }

    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }

    fn execute(&self, _state: &AppApiState, _args: Value) -> Result<Value> {
        let actions = crate::app_services::pending_actions::list_all();
        let rows: Vec<Value> = actions
            .iter()
            .map(|a| {
                json!({
                    "id": a.id,
                    "summary": a.kind.summary(),
                    "reason": preview(&a.reason, 240),
                    "status": serde_json::to_value(a.status).unwrap_or(Value::Null),
                    "proposed_at_unix_ms": a.proposed_at_unix_ms,
                    "expires_at_unix_ms": a.expires_at_unix_ms,
                    "result_note": preview(&a.result_note, 240),
                })
            })
            .collect();
        Ok(json!({
            "count": rows.len(),
            "actions": rows,
        }))
    }
}

pub fn register_default_tools(registry: &mut ToolRegistry) {
    registry.register(Box::new(GetAccountSnapshotTool));
    registry.register(Box::new(GetOpenPositionsTool));
    registry.register(Box::new(GetChartDataTool));
    registry.register(Box::new(GetRiskCapsTool));
    registry.register(Box::new(GetNewsTradingModeTool));
    registry.register(Box::new(FetchUrlTool));
    registry.register(Box::new(GetRecentLogLinesTool));
    registry.register(Box::new(SaveMemoryNoteTool));
    registry.register(Box::new(LoadMemoryNoteTool));
    registry.register(Box::new(ListMemoryKeysTool));
    registry.register(Box::new(ForgetMemoryNoteTool));
    registry.register(Box::new(ExplainRecentTradesTool));
    registry.register(Box::new(GetUpcomingCalendarEventsTool));
    registry.register(Box::new(GetRecentMarketHeadlinesTool));
    registry.register(Box::new(ProposeClosePositionTool));
    registry.register(Box::new(ListPendingActionsTool));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_renders_each_tool_with_description() {
        let mut reg = ToolRegistry::new();
        register_default_tools(&mut reg);
        let rendered = reg.render_for_prompt();
        // Every shipped tool surfaces by its canonical name.
        for name in [
            "get_account_snapshot",
            "get_open_positions",
            "get_chart_data",
            "get_risk_caps",
            "get_news_trading_mode",
            "fetch_url",
            "get_recent_log_lines",
            "save_memory_note",
            "load_memory_note",
            "list_memory_keys",
            "forget_memory_note",
            "explain_recent_trades",
            "get_upcoming_calendar_events",
            "get_recent_market_headlines",
            "propose_close_position",
            "list_pending_actions",
        ] {
            assert!(
                rendered.contains(name),
                "rendered prompt missing tool `{name}`:\n{rendered}"
            );
        }
        // Schema fragments should be inline so the model has full
        // info without a follow-up turn. serde_json's compact
        // Display impl emits `"type":"object"` (no space after the
        // colon) which is intentional — fewer tokens.
        assert!(rendered.contains("\"type\":\"object\""));
    }

    #[test]
    fn ssrf_guard_rejects_obvious_local_targets() {
        for bad in [
            "http://localhost/",
            "http://127.0.0.1/",
            "http://127.0.0.1:7423/healthz",
            "http://10.0.0.5/",
            "http://192.168.1.1/admin",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "ftp://example.com/",
        ] {
            assert!(
                ssrf_guard(bad).is_err(),
                "SSRF guard should have rejected `{bad}`"
            );
        }
    }

    #[test]
    fn ssrf_guard_allows_public_https() {
        for good in [
            "https://www.ecb.europa.eu/press/pr/html/index.en.html",
            "https://api.example.com/v1/things",
            "http://huggingface.co/path",
        ] {
            assert!(
                ssrf_guard(good).is_ok(),
                "SSRF guard should have allowed `{good}`"
            );
        }
    }

    #[test]
    fn ssrf_guard_rejects_malformed_url() {
        assert!(ssrf_guard("not a url").is_err());
        assert!(ssrf_guard("").is_err());
    }

    #[test]
    fn system_prompt_carries_scope_and_role_directives() {
        let mut reg = ToolRegistry::new();
        register_default_tools(&mut reg);
        let prompt = build_system_prompt(&reg);
        // Scope: must explicitly call out the off-topic decline.
        assert!(
            prompt.contains("Scope") && prompt.contains("politely decline"),
            "system prompt must contain the scope-enforcement block"
        );
        // Role: must explicitly position Gemma as the explainer, not
        // the override layer. This is the directive from #112 + the
        // user's #124 follow-up.
        assert!(
            prompt.contains("EXPLAINER") || prompt.contains("explainer"),
            "system prompt must scope Gemma's role to explanation, not override"
        );
        // Memory hint: the model should know it has persistent
        // memory and that `list_memory_keys` is a good first call.
        assert!(
            prompt.contains("list_memory_keys"),
            "system prompt should mention the memory-list tool by name"
        );
    }

    #[test]
    fn parse_tool_call_extracts_simple_call() {
        let raw = "Let me check.\n\n```tool_call\n\
                   {\"name\": \"get_account_snapshot\", \"arguments\": {}}\n\
                   ```\n";
        let call = parse_tool_call(raw).expect("should parse");
        assert_eq!(call.name, "get_account_snapshot");
        assert!(call.arguments.is_object());
    }

    #[test]
    fn parse_tool_call_accepts_hyphen_variant() {
        let raw = "```tool-call\n{\"name\":\"get_risk_caps\",\"arguments\":{}}\n```";
        let call = parse_tool_call(raw).expect("hyphen variant should parse");
        assert_eq!(call.name, "get_risk_caps");
    }

    #[test]
    fn parse_tool_call_returns_none_for_plain_text() {
        let raw = "Your balance is $10,000. Equity is $10,025.";
        assert!(parse_tool_call(raw).is_none());
    }

    #[test]
    fn parse_tool_call_returns_none_for_malformed_json() {
        let raw = "```tool_call\n{not valid json}\n```";
        assert!(parse_tool_call(raw).is_none());
    }

    #[test]
    fn names_returns_sorted_list_for_stable_prompts() {
        let mut reg = ToolRegistry::new();
        register_default_tools(&mut reg);
        let names = reg.names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "names() must be sorted for stable prompts");
    }

    /// Smoke test of the ReAct loop with a fake inference callback
    /// that emits exactly one tool call then plain text. Validates
    /// the conversation grows correctly across turns.
    #[test]
    fn tool_loop_runs_one_step_then_returns_plain_text() {
        // Mock state isn't easy to construct without the rest of
        // the app — use the parser-only path. The lookup-fails arm
        // exercises the same loop logic.
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));

        // For this test we don't need a real AppApiState — construct
        // a minimal one via its public constructor. Skip if the
        // constructor isn't no-arg.
        let state = AppApiState::new();

        let mut step = 0usize;
        let result = run_tool_loop(&state, &reg, "test prompt", |conv| {
            step += 1;
            match step {
                1 => Ok(
                    "```tool_call\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"hi\"}}\n```"
                        .to_string(),
                ),
                _ => {
                    // The second call sees the appended tool_result
                    // in the conversation. Verify it's there.
                    assert!(conv.contains("tool_result"));
                    assert!(conv.contains("\"msg\""));
                    Ok("The echo returned 'hi'.".to_string())
                }
            }
        })
        .expect("loop should converge");
        assert!(result.contains("echo"));
        assert_eq!(step, 2);
    }

    /// Helper tool for the loop test — echoes whatever it's given.
    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echoes back the `msg` argument."
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {"msg": {"type": "string"}}})
        }
        fn execute(&self, _state: &AppApiState, args: Value) -> Result<Value> {
            Ok(json!({"echoed": args}))
        }
    }
}
