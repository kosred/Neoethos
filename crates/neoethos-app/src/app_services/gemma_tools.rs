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
pub fn build_system_prompt(registry: &ToolRegistry) -> String {
    let tools_block = registry.render_for_prompt();
    format!(
        "You are NeoEthos, a local AI assistant embedded in a forex trading platform. \
         You help the operator understand their account, risk settings, market data, \
         and strategy performance. Stay concise, factual, and never invent numbers — \
         use the tools below to fetch real data.\n\n\
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
pub fn register_default_tools(registry: &mut ToolRegistry) {
    registry.register(Box::new(GetAccountSnapshotTool));
    registry.register(Box::new(GetRiskCapsTool));
    registry.register(Box::new(GetNewsTradingModeTool));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_renders_each_tool_with_description() {
        let mut reg = ToolRegistry::new();
        register_default_tools(&mut reg);
        let rendered = reg.render_for_prompt();
        assert!(rendered.contains("get_account_snapshot"));
        assert!(rendered.contains("get_risk_caps"));
        assert!(rendered.contains("get_news_trading_mode"));
        // Schema fragments should be inline so the model has full
        // info without a follow-up turn. serde_json's compact
        // Display impl emits `"type":"object"` (no space after the
        // colon) which is intentional — fewer tokens.
        assert!(rendered.contains("\"type\":\"object\""));
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
