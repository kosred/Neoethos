const CLI_MAIN: &str = include_str!("../src/main.rs");
const CLI_NATIVE_RESEARCH: &str = include_str!("../src/native_research.rs");
const TUI_APP: &str = include_str!("../src/tui/app.rs");
const TUI_PAGES: &str = include_str!("../src/tui/pages/mod.rs");
const TUI_NATIVE_RESEARCH: &str = include_str!("../src/tui/pages/native_research.rs");

#[test]
fn cli_routes_native_research_commands_only_to_the_native_http_adapter() {
    assert!(CLI_MAIN.contains("mod native_research;"));
    assert!(CLI_MAIN.contains("\"native-research\" => native_research::run(tail)"));

    for retired in [
        "/engines/discovery/start",
        "/engines/discovery/stop",
        "/engines/training/start",
        "/engines/training/stop",
        "cmd_discover(",
        "cmd_train(",
    ] {
        assert!(
            !CLI_NATIVE_RESEARCH.contains(retired),
            "native research CLI still reaches a legacy lane through {retired}"
        );
    }

    assert!(CLI_NATIVE_RESEARCH.contains("NATIVE_RESEARCH_START_ROUTE"));
    assert!(CLI_NATIVE_RESEARCH.contains("NATIVE_RESEARCH_CANCEL_ROUTE"));
    assert!(CLI_NATIVE_RESEARCH.contains("ENGINES_STATUS_ROUTE"));
    assert!(CLI_NATIVE_RESEARCH.contains("/engines/native-research/start"));
    assert!(CLI_NATIVE_RESEARCH.contains("/engines/native-research/cancel"));
    assert!(CLI_NATIVE_RESEARCH.contains("/engines/status"));
    assert!(CLI_NATIVE_RESEARCH.contains("current_lease_token_v1"));
    assert!(CLI_NATIVE_RESEARCH.contains("leaseToken"));
}

#[test]
fn tui_has_a_dedicated_native_research_page_and_cancel_is_an_http_command() {
    assert!(TUI_PAGES.contains("pub mod native_research;"));
    assert!(TUI_PAGES.contains("Page::NativeResearch"));
    assert!(TUI_APP.contains("PendingAction::NativeResearchCancel"));
    assert!(TUI_NATIVE_RESEARCH.contains("\"native-research\".to_string()"));
    assert!(TUI_NATIVE_RESEARCH.contains("\"start\".to_string()"));
    assert!(TUI_NATIVE_RESEARCH.contains("\"status\".to_string()"));
    assert!(TUI_NATIVE_RESEARCH.contains("\"cancel\".to_string()"));
    assert!(
        !TUI_NATIVE_RESEARCH.contains("stop_latest"),
        "TUI cancellation must reach the app-owned native handle, not kill a status child"
    );
    assert!(!TUI_NATIVE_RESEARCH.contains("batch-discover"));
    assert!(!TUI_NATIVE_RESEARCH.contains("trainingStart"));
}

#[test]
fn operator_output_includes_bounded_failure_and_published_evidence() {
    for marker in [
        "failureStage",
        "failureCode",
        "failureDetail",
        "leaseToken",
        "relativePath",
        "fileSha256",
        "resolvedPopulation",
        "hardGrowthCap",
        "selectedDeviceOrdinal",
        "consumerCompletionConfirmed",
        "replayIdentitySealed",
        "bounded_text",
    ] {
        assert!(
            CLI_NATIVE_RESEARCH.contains(marker),
            "CLI status renderer is missing {marker}"
        );
    }
}
