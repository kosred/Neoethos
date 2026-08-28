fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing start marker: {start}"))
        .1;
    source
        .split_once(end)
        .unwrap_or_else(|| panic!("missing end marker: {end}"))
        .0
}

#[test]
fn batch_discover_has_an_explicit_typed_population_auto_override() {
    let source = include_str!("../src/main.rs");
    let override_body = section(
        source,
        "fn apply_batch_discover_cli_overrides",
        "fn cmd_batch_discover",
    );
    let command_body = section(
        source,
        "fn cmd_batch_discover",
        "/// Print the resolved config",
    );

    assert!(override_body.contains("--population-auto"));
    assert!(override_body.contains("config.population_auto"));
    assert!(
        override_body.contains("parse::<bool>()"),
        "the CLI must reject malformed values instead of treating their presence as true"
    );
    assert!(command_body.contains("apply_batch_discover_cli_overrides"));
}

#[test]
fn tui_forwards_the_population_auto_field_to_batch_discover() {
    let form = include_str!("../src/tui/form.rs");
    let launch = include_str!("../src/tui/pages/discover.rs");
    let override_helper = section(
        launch,
        "fn append_population_auto_override",
        "pub fn launch_now",
    );
    let launch_body = section(
        launch,
        "pub fn launch_now",
        "pub(super) fn strip_ansi_for_display",
    );

    assert!(form.contains("\"Population auto\""));
    assert!(override_helper.contains("\"--population-auto\""));
    assert!(launch_body.contains("\"Population auto\""));
    assert!(launch_body.contains("append_population_auto_override"));
}
