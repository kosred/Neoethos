//! G3 — read-only tool catalog stubs.
//!
//! These tools are what Gemma calls during conversational Q&A
//! to look up live bot state. They are ALL in `ToolCategory::
//! ReadOnlyState` (or `ReadOnlyResearch` for web fetch in G5),
//! so they fire regardless of the trading-tools gate.
//!
//! ## What's in the catalog
//!
//! | Tool name                  | Returns                                       |
//! |----------------------------|-----------------------------------------------|
//! | `list_positions`           | Open positions for the active account         |
//! | `list_orders`              | Pending orders                                |
//! | `get_quote`                | Latest bid/ask for a symbol                   |
//! | `get_account_balance`      | Balance + equity + free margin                |
//! | `get_recent_predictions`   | Last N ensemble predictions for a symbol      |
//! | `explain_last_decision`    | Per-expert vote breakdown for the last bar    |
//! | `get_risk_config`          | Current Risky Mode tier / caps / blackout     |
//! | `get_news_blackout_state`  | Whether the news blackout window is active    |
//! | `get_health`               | Autonomous trader + training-job health       |
//! | `tail_log`                 | Most recent app-log entries                   |
//!
//! ## Look-ahead bias enforcement
//!
//! Every tool that returns time-series data filters at
//! `ToolContext.past_data_cutoff_unix_ms`. The G3 stubs apply
//! this filter even on placeholder data — that's the invariant
//! G6 will inherit when these get wired to real `TradingSession`
//! state.
//!
//! ## Wiring to real state
//!
//! G3 ships **stub** implementations that return deterministic
//! placeholder JSON shaped exactly like what the production
//! tools will emit. The shapes are pinned by `#[derive(Serialize,
//! Deserialize)]` round-trip tests; G6 swaps the stub body for a
//! real call into `forex_app::app_services::TradingSession`
//! without changing the wire format.

use crate::error::GemmaError;
use crate::tools::{BotTool, ToolCategory, ToolContext, ToolRegistry};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// list_positions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PositionRow {
    pub symbol: String,
    pub side: String, // "BUY" | "SELL"
    pub volume: i64,
    pub avg_price: f64,
    pub unrealised_pnl: f64,
    pub opened_at_unix_ms: i64,
}

pub struct ListPositionsTool;

impl BotTool for ListPositionsTool {
    fn name(&self) -> &'static str {
        "list_positions"
    }
    fn description(&self) -> &'static str {
        "List open positions for the active account. Read-only."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional: filter to one symbol, e.g. \"EUR/USD\"."
                }
            }
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        let filter_symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        // Stub data — replaced in G6 with TradingSession lookup.
        let stub = vec![PositionRow {
            symbol: "EUR/USD".to_string(),
            side: "BUY".to_string(),
            volume: 100_000,
            avg_price: 1.0823,
            unrealised_pnl: 12.40,
            opened_at_unix_ms: ctx.past_data_cutoff_unix_ms.saturating_sub(3600_000),
        }];
        let rows: Vec<PositionRow> = stub
            .into_iter()
            .filter(|p| {
                p.opened_at_unix_ms < ctx.past_data_cutoff_unix_ms
                    && filter_symbol.as_ref().is_none_or(|s| s == &p.symbol)
            })
            .collect();
        Ok(json!({ "account_id": ctx.account_id, "positions": rows }))
    }
}

// ---------------------------------------------------------------------------
// list_orders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrderRow {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub volume: i64,
    pub kind: String, // "MARKET" | "LIMIT" | "STOP"
    pub price: Option<f64>,
    pub status: String, // "Pending" | "Filled" | "Cancelled"
    pub submitted_at_unix_ms: i64,
}

pub struct ListOrdersTool;

impl BotTool for ListOrdersTool {
    fn name(&self) -> &'static str {
        "list_orders"
    }
    fn description(&self) -> &'static str {
        "List recent + pending orders. Read-only."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional filter: 'pending' | 'filled' | 'cancelled' | 'all'",
                    "default": "all"
                }
            }
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, _args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        let empty: Vec<OrderRow> = vec![];
        Ok(json!({ "account_id": ctx.account_id, "orders": empty }))
    }
}

// ---------------------------------------------------------------------------
// get_quote
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuoteSnapshot {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub timestamp_unix_ms: i64,
}

pub struct GetQuoteTool;

impl BotTool for GetQuoteTool {
    fn name(&self) -> &'static str {
        "get_quote"
    }
    fn description(&self) -> &'static str {
        "Latest bid/ask quote for a symbol. Returns the last \
         CLOSED-bar quote (look-ahead bias guard)."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "symbol": { "type": "string", "description": "e.g. \"EUR/USD\"" }
            },
            "required": ["symbol"]
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        let symbol =
            args.get("symbol")
                .and_then(|v| v.as_str())
                .ok_or_else(|| GemmaError::ToolDenied {
                    name: "get_quote".to_string(),
                    reason: "missing required `symbol` parameter".to_string(),
                })?;
        Ok(json!(QuoteSnapshot {
            symbol: symbol.to_string(),
            bid: 1.0823,
            ask: 1.0825,
            // Strict past-data: snap to the cutoff itself rather
            // than current wall-clock.
            timestamp_unix_ms: ctx.past_data_cutoff_unix_ms,
        }))
    }
}

// ---------------------------------------------------------------------------
// get_account_balance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AccountBalance {
    pub account_id: String,
    pub balance: f64,
    pub equity: f64,
    pub free_margin: f64,
    pub used_margin: f64,
    pub currency: String,
}

pub struct GetAccountBalanceTool;

impl BotTool for GetAccountBalanceTool {
    fn name(&self) -> &'static str {
        "get_account_balance"
    }
    fn description(&self) -> &'static str {
        "Account balance + equity + free margin. Read-only."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, _args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        Ok(json!(AccountBalance {
            account_id: ctx.account_id.clone(),
            balance: 10_000.0,
            equity: 10_012.40,
            free_margin: 9_762.40,
            used_margin: 250.0,
            currency: "EUR".to_string(),
        }))
    }
}

// ---------------------------------------------------------------------------
// get_recent_predictions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictionRow {
    pub timestamp_unix_ms: i64,
    pub symbol: String,
    pub direction: String, // "long" | "short" | "flat"
    pub confidence: f64,
}

pub struct GetRecentPredictionsTool;

impl BotTool for GetRecentPredictionsTool {
    fn name(&self) -> &'static str {
        "get_recent_predictions"
    }
    fn description(&self) -> &'static str {
        "Last N ensemble predictions for a symbol. Only entries \
         from CLOSED bars (look-ahead bias guard)."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "symbol": { "type": "string" },
                "limit":  { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 }
            },
            "required": ["symbol"]
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GemmaError::ToolDenied {
                name: "get_recent_predictions".to_string(),
                reason: "missing `symbol`".to_string(),
            })?
            .to_string();
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(10) as usize;
        let mut rows: Vec<PredictionRow> = Vec::new();
        for i in 0..limit.min(3) {
            rows.push(PredictionRow {
                timestamp_unix_ms: ctx
                    .past_data_cutoff_unix_ms
                    .saturating_sub((i as i64 + 1) * 60_000),
                symbol: symbol.clone(),
                direction: "long".to_string(),
                confidence: 0.62,
            });
        }
        // Defensive look-ahead filter even on the stub.
        rows.retain(|r| r.timestamp_unix_ms < ctx.past_data_cutoff_unix_ms);
        Ok(json!({ "symbol": symbol, "predictions": rows }))
    }
}

// ---------------------------------------------------------------------------
// explain_last_decision
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpertVote {
    pub expert_name: String,
    pub direction: String,
    pub confidence: f64,
    pub weight: f64,
}

pub struct ExplainLastDecisionTool;

impl BotTool for ExplainLastDecisionTool {
    fn name(&self) -> &'static str {
        "explain_last_decision"
    }
    fn description(&self) -> &'static str {
        "Per-expert vote breakdown for the last bar's ensemble \
         decision. Useful when the user asks 'why did the model \
         go short?'."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "symbol": { "type": "string" } },
            "required": ["symbol"]
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        let symbol = args
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GemmaError::ToolDenied {
                name: "explain_last_decision".to_string(),
                reason: "missing `symbol`".to_string(),
            })?
            .to_string();
        let votes = vec![
            ExpertVote {
                expert_name: "lightgbm".to_string(),
                direction: "long".to_string(),
                confidence: 0.71,
                weight: 0.12,
            },
            ExpertVote {
                expert_name: "xgboost".to_string(),
                direction: "long".to_string(),
                confidence: 0.65,
                weight: 0.12,
            },
            ExpertVote {
                expert_name: "tabnet".to_string(),
                direction: "flat".to_string(),
                confidence: 0.55,
                weight: 0.08,
            },
        ];
        Ok(json!({
            "symbol": symbol,
            "decision_timestamp_unix_ms": ctx.past_data_cutoff_unix_ms,
            "ensemble_direction": "long",
            "ensemble_confidence": 0.62,
            "per_expert_votes": votes
        }))
    }
}

// ---------------------------------------------------------------------------
// get_risk_config
// ---------------------------------------------------------------------------

pub struct GetRiskConfigTool;

impl BotTool for GetRiskConfigTool {
    fn name(&self) -> &'static str {
        "get_risk_config"
    }
    fn description(&self) -> &'static str {
        "Current Risky Mode configuration: tier caps, daily loss \
         budget, max concurrent trades, autonomous-only contract \
         state."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, _args: Value, _ctx: &ToolContext) -> Result<Value, GemmaError> {
        Ok(json!({
            "tier": "Defensive",
            "max_loss_per_trade_pct": 0.005,
            "max_loss_per_day_pct":   0.02,
            "max_concurrent_trades":  3,
            "autonomous_only_contract_accepted": true,
            "manual_halt_active": false,
            "hardware_halt_active": false
        }))
    }
}

// ---------------------------------------------------------------------------
// get_news_blackout_state
// ---------------------------------------------------------------------------

pub struct GetNewsBlackoutStateTool;

impl BotTool for GetNewsBlackoutStateTool {
    fn name(&self) -> &'static str {
        "get_news_blackout_state"
    }
    fn description(&self) -> &'static str {
        "Whether a news-event blackout window is currently \
         suppressing autonomous trades, and when the next one \
         starts."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, _args: Value, ctx: &ToolContext) -> Result<Value, GemmaError> {
        Ok(json!({
            "currently_active": false,
            "next_window_start_unix_ms": ctx.past_data_cutoff_unix_ms.saturating_add(7200_000),
            "next_window_reason": "FOMC"
        }))
    }
}

// ---------------------------------------------------------------------------
// get_health
// ---------------------------------------------------------------------------

pub struct GetHealthTool;

impl BotTool for GetHealthTool {
    fn name(&self) -> &'static str {
        "get_health"
    }
    fn description(&self) -> &'static str {
        "Autonomous-trader + training-job health summary."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, _args: Value, _ctx: &ToolContext) -> Result<Value, GemmaError> {
        Ok(json!({
            "autonomous_trader_running": true,
            "last_signal_unix_ms": null,
            "broker_connected": true,
            "broker_session_expires_at_unix_ms": null,
            "training_job_running": false,
            "last_training_completed_unix_ms": null
        }))
    }
}

// ---------------------------------------------------------------------------
// tail_log
// ---------------------------------------------------------------------------

pub struct TailLogTool;

impl BotTool for TailLogTool {
    fn name(&self) -> &'static str {
        "tail_log"
    }
    fn description(&self) -> &'static str {
        "Most recent N entries from the app log. Useful for \
         diagnosing why something happened."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "lines": { "type": "integer", "minimum": 1, "maximum": 200, "default": 20 }
            }
        })
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::ReadOnlyState
    }
    fn invoke(&self, args: Value, _ctx: &ToolContext) -> Result<Value, GemmaError> {
        let n = args
            .get("lines")
            .and_then(|v| v.as_i64())
            .unwrap_or(20)
            .clamp(1, 200) as usize;
        let lines: Vec<String> = vec!["forex-ai bootstrapped".to_string(); n.min(1)];
        Ok(json!({ "lines": lines }))
    }
}

// ---------------------------------------------------------------------------
// Registry factory
// ---------------------------------------------------------------------------

/// Register all G3 read-only tools onto `registry`. Idempotent
/// in the sense that calling it twice on the same registry will
/// panic via the duplicate-name guard — same behaviour as
/// `ToolRegistry::register`. The factory exists so chrome /
/// integration code doesn't have to enumerate the tools.
pub fn register_all_g3(registry: &mut ToolRegistry) {
    registry.register(Box::new(ListPositionsTool));
    registry.register(Box::new(ListOrdersTool));
    registry.register(Box::new(GetQuoteTool));
    registry.register(Box::new(GetAccountBalanceTool));
    registry.register(Box::new(GetRecentPredictionsTool));
    registry.register(Box::new(ExplainLastDecisionTool));
    registry.register(Box::new(GetRiskConfigTool));
    registry.register(Box::new(GetNewsBlackoutStateTool));
    registry.register(Box::new(GetHealthTool));
    registry.register(Box::new(TailLogTool));
}

/// Build a registry pre-loaded with all G3 read-only tools.
pub fn registry_with_g3_tools() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    register_all_g3(&mut r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext {
            past_data_cutoff_unix_ms: 1_700_000_000_000,
            account_id: "acc-1".to_string(),
            gated_tools_enabled: false,
        }
    }

    #[test]
    fn registry_contains_all_ten_g3_tools() {
        let r = registry_with_g3_tools();
        assert_eq!(r.len(), 10);
        for name in &[
            "list_positions",
            "list_orders",
            "get_quote",
            "get_account_balance",
            "get_recent_predictions",
            "explain_last_decision",
            "get_risk_config",
            "get_news_blackout_state",
            "get_health",
            "tail_log",
        ] {
            assert!(r.get(name).is_some(), "missing tool: {name}");
        }
    }

    #[test]
    fn every_g3_tool_is_read_only_state() {
        let r = registry_with_g3_tools();
        for name in r.names() {
            let t = r.get(name).unwrap();
            assert_eq!(
                t.category(),
                ToolCategory::ReadOnlyState,
                "tool {name} should be ReadOnlyState"
            );
        }
    }

    #[test]
    fn every_g3_tool_runs_with_gate_closed() {
        let r = registry_with_g3_tools();
        let ctx = ctx();
        for name in r.names() {
            let owned = name.to_string();
            // Use a minimum-valid args object — many tools need
            // a `symbol` field.
            let args = json!({ "symbol": "EUR/USD" });
            let out = r.invoke(&owned, args, &ctx);
            assert!(out.is_ok(), "tool {owned} failed: {out:?}");
        }
    }

    #[test]
    fn list_positions_filters_at_past_data_cutoff() {
        let tool = ListPositionsTool;
        let out = tool.invoke(json!({}), &ctx()).unwrap();
        let positions = out["positions"].as_array().unwrap();
        for p in positions {
            let t = p["opened_at_unix_ms"].as_i64().unwrap();
            assert!(t < 1_700_000_000_000, "position leaked future bar: {t}");
        }
    }

    #[test]
    fn list_positions_respects_symbol_filter() {
        let tool = ListPositionsTool;
        let out = tool
            .invoke(json!({ "symbol": "DOES_NOT_EXIST" }), &ctx())
            .unwrap();
        assert_eq!(out["positions"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn get_quote_requires_symbol_arg() {
        let tool = GetQuoteTool;
        let err = tool.invoke(json!({}), &ctx()).expect_err("must bail");
        assert!(matches!(err, GemmaError::ToolDenied { .. }));
    }

    #[test]
    fn get_quote_timestamp_never_exceeds_cutoff() {
        let tool = GetQuoteTool;
        let out = tool.invoke(json!({ "symbol": "EUR/USD" }), &ctx()).unwrap();
        assert!(out["timestamp_unix_ms"].as_i64().unwrap() <= 1_700_000_000_000);
    }

    #[test]
    fn get_recent_predictions_filters_future_entries() {
        let tool = GetRecentPredictionsTool;
        let out = tool
            .invoke(json!({ "symbol": "EUR/USD", "limit": 5 }), &ctx())
            .unwrap();
        let preds = out["predictions"].as_array().unwrap();
        for p in preds {
            let t = p["timestamp_unix_ms"].as_i64().unwrap();
            assert!(t < 1_700_000_000_000);
        }
    }

    #[test]
    fn explain_last_decision_returns_per_expert_votes() {
        let tool = ExplainLastDecisionTool;
        let out = tool.invoke(json!({ "symbol": "EUR/USD" }), &ctx()).unwrap();
        let votes = out["per_expert_votes"].as_array().unwrap();
        assert!(!votes.is_empty());
        for v in votes {
            assert!(v["expert_name"].is_string());
            assert!(v["direction"].is_string());
            assert!(v["confidence"].as_f64().is_some());
        }
    }

    #[test]
    fn account_balance_round_trips_through_serde() {
        let ab = AccountBalance {
            account_id: "x".to_string(),
            balance: 1.0,
            equity: 1.0,
            free_margin: 1.0,
            used_margin: 0.0,
            currency: "USD".to_string(),
        };
        let s = serde_json::to_string(&ab).unwrap();
        let back: AccountBalance = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ab);
    }

    #[test]
    fn every_tool_has_non_empty_schema() {
        let r = registry_with_g3_tools();
        for name in r.names() {
            let owned = name.to_string();
            let t = r.get(&owned).unwrap();
            let schema = t.parameters_schema();
            assert_eq!(schema["type"], "object");
        }
    }

    #[test]
    fn every_tool_has_non_empty_description() {
        let r = registry_with_g3_tools();
        for name in r.names() {
            let owned = name.to_string();
            let t = r.get(&owned).unwrap();
            assert!(
                !t.description().is_empty(),
                "tool {owned} has empty description"
            );
        }
    }
}
