//! Tool catalog — `BotTool` trait + `ToolContext` + `ToolRegistry`.
//!
//! Phase G0 — trait surface + an empty registry.
//!
//! ## Gemma's two trading roles
//!
//! Role 1 — `GemmaExpert` votes in `SoftVotingEnsemble`
//! (autonomous trading, ensemble-decided). Does NOT use this
//! catalog. See `expert.rs`.
//!
//! Role 2 — `GatedTrading` tools emit `PendingSuggestion`
//! objects; the UI Approve/Rejects; only Approve fires real
//! `submit_order` (with `OrderSource::AiSuggested`). See
//! `suggestions.rs`.
//!
//! Look-ahead bias: tools returning time-series data MUST
//! filter at `ToolContext.past_data_cutoff_unix_ms`.

use crate::error::GemmaError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// Read-only state (positions, quotes, configs).
    ReadOnlyState,
    /// Read-only research (web fetch). Gated by search provider.
    ReadOnlyResearch,
    /// Trade SUGGESTION — emits PendingSuggestion, never executes.
    GatedTrading,
    /// Non-trading config mutation.
    GatedConfig,
    /// Training-job control.
    GatedTraining,
}

impl ToolCategory {
    pub fn is_safe_by_default(self) -> bool {
        matches!(self, Self::ReadOnlyState | Self::ReadOnlyResearch)
    }
}

#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Look-ahead bias gate: time-series tools filter at < this.
    pub past_data_cutoff_unix_ms: i64,
    pub account_id: String,
    pub gated_tools_enabled: bool,
}

impl ToolContext {
    /// Test helper — never use in production code.
    pub fn for_test() -> Self {
        Self {
            past_data_cutoff_unix_ms: 0,
            account_id: "test-account".to_string(),
            gated_tools_enabled: false,
        }
    }
}

pub trait BotTool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn category(&self) -> ToolCategory;
    fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<serde_json::Value, GemmaError>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn BotTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn BotTool>) {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            panic!("ToolRegistry: duplicate tool name {name:?}");
        }
        self.tools.insert(name, tool);
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
    pub fn get(&self, name: &str) -> Option<&dyn BotTool> {
        self.tools.get(name).map(|b| b.as_ref())
    }
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    pub fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<serde_json::Value, GemmaError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| GemmaError::ToolNotFound {
                name: name.to_string(),
            })?;
        if !tool.category().is_safe_by_default() && !ctx.gated_tools_enabled {
            return Err(GemmaError::ToolDenied {
                name: name.to_string(),
                reason: format!(
                    "tool category {:?} is gated and the gate is closed",
                    tool.category()
                ),
            });
        }
        tool.invoke(args, ctx)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool {
        category: ToolCategory,
    }

    impl BotTool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "Echoes its args back."
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }
        fn category(&self) -> ToolCategory {
            self.category
        }
        fn invoke(
            &self,
            args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<serde_json::Value, GemmaError> {
            Ok(args)
        }
    }

    #[test]
    fn empty_registry_reports_zero_tools() {
        let r = ToolRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn register_then_lookup_round_trips() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::ReadOnlyState,
        }));
        assert_eq!(r.len(), 1);
        assert!(r.get("echo").is_some());
        assert_eq!(r.names(), vec!["echo"]);
    }

    #[test]
    #[should_panic(expected = "duplicate tool name")]
    fn duplicate_registration_panics() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::ReadOnlyState,
        }));
        r.register(Box::new(EchoTool {
            category: ToolCategory::ReadOnlyState,
        }));
    }

    #[test]
    fn invoke_unknown_tool_returns_tool_not_found() {
        let r = ToolRegistry::new();
        let err = r
            .invoke("nope", serde_json::json!({}), &ToolContext::for_test())
            .expect_err("bail");
        match err {
            GemmaError::ToolNotFound { name } => assert_eq!(name, "nope"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn read_only_tool_runs_even_when_gate_is_closed() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::ReadOnlyState,
        }));
        let mut ctx = ToolContext::for_test();
        ctx.gated_tools_enabled = false;
        let out = r
            .invoke("echo", serde_json::json!({"text": "hi"}), &ctx)
            .expect("ok");
        assert_eq!(out, serde_json::json!({"text": "hi"}));
    }

    #[test]
    fn gated_trading_tool_refuses_when_gate_is_closed() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::GatedTrading,
        }));
        let mut ctx = ToolContext::for_test();
        ctx.gated_tools_enabled = false;
        let err = r
            .invoke("echo", serde_json::json!({}), &ctx)
            .expect_err("bail");
        match err {
            GemmaError::ToolDenied { name, reason } => {
                assert_eq!(name, "echo");
                assert!(reason.contains("gate"));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn gated_trading_tool_runs_when_gate_is_open() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::GatedTrading,
        }));
        let mut ctx = ToolContext::for_test();
        ctx.gated_tools_enabled = true;
        assert!(r.invoke("echo", serde_json::json!({}), &ctx).is_ok());
    }

    #[test]
    fn gated_config_tool_refuses_when_gate_is_closed() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool {
            category: ToolCategory::GatedConfig,
        }));
        let ctx = ToolContext::for_test();
        assert!(r.invoke("echo", serde_json::json!({}), &ctx).is_err());
    }

    #[test]
    fn tool_category_safe_by_default_only_for_read_only() {
        assert!(ToolCategory::ReadOnlyState.is_safe_by_default());
        assert!(ToolCategory::ReadOnlyResearch.is_safe_by_default());
        assert!(!ToolCategory::GatedTrading.is_safe_by_default());
        assert!(!ToolCategory::GatedConfig.is_safe_by_default());
        assert!(!ToolCategory::GatedTraining.is_safe_by_default());
    }

    #[test]
    fn tool_context_for_test_has_zero_cutoff() {
        let ctx = ToolContext::for_test();
        assert_eq!(ctx.past_data_cutoff_unix_ms, 0);
        assert!(!ctx.gated_tools_enabled);
    }
}
