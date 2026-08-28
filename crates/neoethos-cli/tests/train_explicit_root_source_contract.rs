use std::fs;
use std::path::Path;

fn read(relative: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let tail = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing section start `{start}`"))
        .1;
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing section end `{end}`"))
        .0
}

#[test]
fn train_command_binds_the_explicit_canonical_root_without_an_env_fallback() {
    let source = read("src/main.rs");
    let train = between(&source, "fn cmd_train", "struct StreamingArtifactBundle");

    assert!(
        train.contains("let data_root = parse_root(args, Some(&settings));"),
        "train must resolve --data-path/--root against the same settings it trains with"
    );
    assert!(
        train.contains(".with_data_root(data_root)"),
        "the resolved root must move into TrainingOrchestrator"
    );
    assert!(
        !train.contains("NEOETHOS_BOT_DATA_ROOT"),
        "the retired process-environment data-root fallback must not reappear"
    );
}

#[test]
fn tui_train_passes_root_as_an_argument_instead_of_child_environment_state() {
    let source = read("src/tui/pages/train.rs");
    let launch = between(
        &source,
        "pub fn launch_now",
        "shared.status = \"Spawned train\"",
    );

    assert!(launch.contains("\"--root\".to_string()"));
    assert!(launch.contains("shared.jobs.spawn(\"train\", args)"));
    assert!(!launch.contains("spawn_with_env"));
    assert!(!launch.contains("NEOETHOS_BOT_DATA_ROOT"));
}

#[test]
fn auto_loop_passes_its_resolved_root_to_each_training_command() {
    let source = read("src/main.rs");
    let auto_loop = between(&source, "fn cmd_auto_loop", "fn cmd_import");

    assert!(
        auto_loop.contains("\"--root\".to_string(),\n                root.clone(),"),
        "every auto-loop training unit must carry the already-resolved canonical root"
    );
    assert!(
        !auto_loop.contains("std::env::set_var(\"NEOETHOS_BOT_DATA_ROOT\""),
        "auto-loop must not mutate process environment to configure training"
    );
}
