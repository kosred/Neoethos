//! Tier-1 tests 6.1.1 + 6.1.2 (design doc): the tool catalog snapshot and
//! the guard-coverage assertion. No backend needed.

use neoethos_mcp::{ControlPlane, GUARDED_TOOLS, TRADE_ROUTES};

/// Side-effect class of a tool, as annotated (design §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    /// read_only_hint=true
    Ro,
    /// read_only_hint=true, open_world_hint=true (news_briefing only)
    RoOpenWorld,
    /// read_only_hint=false, destructive_hint=false
    Mut,
    /// Mut + idempotent_hint=true (the stop tools)
    MutIdempotent,
    /// read_only_hint=false, destructive_hint=true (demo-guarded trade set,
    /// plus update_settings as an honest hint)
    Trade,
}

/// The complete expected catalog — every tool, in design §3 order. Any
/// drift (added/removed/renamed tool, changed class) fails here first.
const EXPECTED: &[(&str, Class)] = &[
    // System and health
    ("system_health", Class::Ro),
    ("hardware_info", Class::Ro),
    ("engine_status", Class::Ro),
    ("storage_paths", Class::Ro),
    ("diagnostics_report", Class::Mut),
    // Account and broker state
    ("broker_status", Class::Ro),
    ("account_snapshot", Class::Ro),
    ("list_positions", Class::Ro),
    ("broker_symbols", Class::Ro),
    ("broker_timeframes", Class::Ro),
    ("order_history", Class::Ro),
    ("cashflow", Class::Ro),
    ("expected_margin", Class::Ro),
    // Market data
    ("get_chart", Class::Ro),
    ("chart_history", Class::Ro),
    ("get_indicator", Class::Ro),
    ("intelligence", Class::Ro),
    ("live_spots", Class::Ro),
    ("spread_stats", Class::Ro),
    ("news_briefing", Class::RoOpenWorld),
    ("get_watchlist", Class::Ro),
    ("set_watchlist", Class::Mut),
    // Trading — direct orders
    ("open_position", Class::Trade),
    ("close_position", Class::Trade),
    ("modify_position_protection", Class::Trade),
    ("place_pending_order", Class::Trade),
    ("list_pending_orders", Class::Ro),
    ("cancel_pending_order", Class::Trade),
    // Trading — human-confirm queue
    ("list_pending_actions", Class::Ro),
    ("confirm_pending_action", Class::Trade),
    ("reject_pending_action", Class::Mut),
    // Autopilot
    ("autopilot_start", Class::Trade),
    ("autopilot_stop", Class::MutIdempotent),
    ("autopilot_status", Class::Ro),
    ("demo_gate_status", Class::Ro),
    ("replay_backtest", Class::Ro),
    ("parity_check", Class::Ro),
    ("tail_risk", Class::Ro),
    ("challenge_sim", Class::Ro),
    // Discovery and training
    ("discovery_start", Class::Mut),
    ("discovery_stop", Class::MutIdempotent),
    ("training_start", Class::Mut),
    ("training_stop", Class::MutIdempotent),
    ("experience_train", Class::Mut),
    // Strategy lab and portfolios
    ("promotion_gate_status", Class::Ro),
    ("promote_strategies", Class::Trade),
    ("list_portfolios", Class::Ro),
    ("list_strategies", Class::Ro),
    ("strategy_report", Class::Ro),
    ("strategy_blacklist", Class::Ro),
    // Settings and risk
    ("get_settings", Class::Ro),
    ("update_settings", Class::Trade), // hint-only: destructive UX, NOT guarded
    ("knob_catalog", Class::Ro),
    ("get_risk", Class::Ro),
    ("apply_risk_preset", Class::Trade),
    ("risky_scenarios", Class::Ro),
    // Journal and analytics
    ("journal_trades", Class::Ro),
    ("journal_stats", Class::Ro),
    ("journal_analytics", Class::Ro),
    // Data management
    ("data_coverage", Class::Ro),
    ("fetch_history", Class::Mut),
    ("import_data", Class::Mut),
];

fn listed_tools() -> Vec<rmcp::model::Tool> {
    ControlPlane::tool_router().list_all()
}

#[test]
fn catalog_has_exactly_the_59_designed_tools() {
    assert_eq!(
        EXPECTED.len(),
        62,
        "the expected table itself must list 62 tools"
    );
    let tools = listed_tools();
    assert_eq!(
        tools.len(),
        62,
        "tools/list must expose exactly the 62 designed tools, got: {:?}",
        tools.iter().map(|t| t.name.as_ref()).collect::<Vec<_>>()
    );
    for (name, _) in EXPECTED {
        assert!(
            tools.iter().any(|t| t.name.as_ref() == *name),
            "designed tool '{name}' is missing from tools/list"
        );
    }
}

#[test]
fn every_tool_has_a_description() {
    for tool in listed_tools() {
        let desc = tool
            .description
            .as_ref()
            .unwrap_or_else(|| panic!("tool '{}' has no description", tool.name));
        assert!(
            desc.trim().len() >= 20,
            "tool '{}' has a too-short description: {desc:?}",
            tool.name
        );
    }
}

#[test]
fn annotations_match_the_design_class_table() {
    let tools = listed_tools();
    for (name, class) in EXPECTED {
        let tool = tools
            .iter()
            .find(|t| t.name.as_ref() == *name)
            .unwrap_or_else(|| panic!("tool '{name}' missing"));
        let ann = tool
            .annotations
            .as_ref()
            .unwrap_or_else(|| panic!("tool '{name}' has no annotations"));
        let read_only = ann.read_only_hint;
        let destructive = ann.destructive_hint;
        let idempotent = ann.idempotent_hint;
        let open_world = ann.open_world_hint;
        match class {
            Class::Ro => {
                assert_eq!(read_only, Some(true), "'{name}' must be read-only");
                assert_ne!(destructive, Some(true), "'{name}' must not be destructive");
                assert_eq!(open_world, Some(false), "'{name}' must be closed-world");
            }
            Class::RoOpenWorld => {
                assert_eq!(read_only, Some(true), "'{name}' must be read-only");
                assert_eq!(open_world, Some(true), "'{name}' must be open-world");
            }
            Class::Mut => {
                assert_eq!(read_only, Some(false), "'{name}' must not be read-only");
                assert_eq!(destructive, Some(false), "'{name}' must not be destructive");
                assert_eq!(open_world, Some(false), "'{name}' must be closed-world");
            }
            Class::MutIdempotent => {
                assert_eq!(read_only, Some(false), "'{name}' must not be read-only");
                assert_eq!(destructive, Some(false), "'{name}' must not be destructive");
                assert_eq!(idempotent, Some(true), "'{name}' must be idempotent");
                assert_eq!(open_world, Some(false), "'{name}' must be closed-world");
            }
            Class::Trade => {
                assert_eq!(read_only, Some(false), "'{name}' must not be read-only");
                assert_eq!(
                    destructive,
                    Some(true),
                    "'{name}' must carry destructive_hint"
                );
                assert_eq!(open_world, Some(false), "'{name}' must be closed-world");
            }
        }
    }
}

/// Design §3 guard-coverage rule: the guarded set equals the set of tools
/// annotated `destructive_hint=true ∧ read_only_hint≠true`, minus
/// `update_settings` (destructive as an honest UX hint only — it executes
/// no trade). Belt and braces on top of the DemoProof compile error: a new
/// trade tool that is annotated destructive but missing from GUARDED_TOOLS
/// (or vice versa) fails here.
#[test]
fn guarded_set_equals_the_destructive_trade_set() {
    let mut destructive: Vec<String> = listed_tools()
        .iter()
        .filter(|t| {
            t.annotations
                .as_ref()
                .is_some_and(|a| a.destructive_hint == Some(true) && a.read_only_hint != Some(true))
        })
        .map(|t| t.name.to_string())
        .filter(|name| name != "update_settings")
        .collect();
    destructive.sort();

    let mut guarded: Vec<String> = GUARDED_TOOLS.iter().map(|s| s.to_string()).collect();
    guarded.sort();

    assert_eq!(
        destructive, guarded,
        "the destructive-annotated trade set and the demo-guarded set drifted apart"
    );
}

#[test]
fn trade_route_registry_covers_all_nine_guarded_paths() {
    // The route registry is what trade_post() enforces at runtime; keep it
    // in lockstep with the design's exact route list.
    let expected = [
        "/orders",
        "/orders/pending",
        "/orders/cancel",
        "/positions/close",
        "/positions/protection",
        "/actions/{id}/confirm",
        "/autonomous/start",
        "/strategy_lab/promote",
        "/risk/preset",
    ];
    assert_eq!(TRADE_ROUTES, &expected);
}
