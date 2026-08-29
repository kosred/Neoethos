use std::fs;
use std::path::Path;

fn source(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("read {}: {error}", path.display());
    })
}

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source start marker {start:?}"));
    let tail = &source[start_index..];
    let end_index = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing source end marker {end:?}"));
    &tail[..end_index]
}

fn assert_absent(label: &str, source: &str, forbidden: &[&str]) {
    for needle in forbidden {
        assert!(
            !source.contains(needle),
            "{label} must not contain pre-admission or legacy bypass {needle:?}"
        );
    }
}

#[test]
fn headless_flags_form_one_typed_ordered_pipeline_and_retain_the_handle() {
    let main = source("src/main.rs");
    let headless = between(
        &main,
        "async fn run_headless_loop",
        "pub(crate) fn app_record",
    );

    assert_absent(
        "headless startup",
        headless,
        &[
            "discover_symbols(",
            "Settings::from_yaml",
            "resolve_unique_background_dataset_identity",
            "pin_current_discovery_input",
            "start_discovery_job",
            "start_training_job",
        ],
    );
    assert!(headless.contains("run_headless_execution_pipeline_v1"));
    assert!(headless.contains("headless_execution.cancel()"));
    assert!(headless.contains("headless_execution.await_terminal().await"));
    let cancel = headless.find("headless_execution.cancel()").unwrap();
    let terminal = headless
        .find("headless_execution.await_terminal().await")
        .unwrap();
    assert!(
        cancel < terminal,
        "headless shutdown must cancel before await"
    );
}

#[test]
fn main_installs_one_native_startup_authority_before_any_api_state() {
    let main = source("src/main.rs");
    let install_call = "install_canonical_native_startup_authority_v1(&settings)";

    assert_eq!(
        main.matches(install_call).count(),
        1,
        "startup Settings must seal exactly one native authority"
    );
    let install = main.find(install_call).unwrap();
    let first_state = main
        .find("AppApiState::new()")
        .expect("main constructs at least one API state");
    assert!(
        install < first_state,
        "native startup authority must be installed before any API state captures it"
    );
    assert!(
        main[..install].contains("install_runtime_overrides_from_settings(&settings)"),
        "native authority sealing must follow the existing startup runtime installers"
    );
    assert!(
        main[..install].contains("feature = \"gpu-nvidia\""),
        "only the supported Linux NVIDIA build may install native authority"
    );
    assert_absent(
        "main native startup authority",
        &main,
        &["install_and_seal_canonical_native_runtime_authority_v1("],
    );
}

#[test]
fn validation_uses_typed_start_and_timeout_waits_for_real_release() {
    let validation = source("src/app_services/validation.rs");
    let run_one = between(&validation, "async fn run_one_tf", "fn snapshot_to_outcome");

    assert_absent(
        "validation TF run",
        run_one,
        &[
            "Settings::from_yaml",
            "DiscoveryConfig::try_from_settings",
            "resolve_unique_background_dataset_identity",
            "pin_current_discovery_input",
            "start_discovery_job",
        ],
    );
    assert!(run_one.contains("start_typed_discovery_execution_v1"));
    assert!(run_one.contains("TypedDiscoveryGenerationOverrideV1::Floor"));
    assert!(run_one.contains("handle.cancel()"));
    assert!(run_one.contains("handle.await_terminal().await"));
    let cancel = run_one.find("handle.cancel()").unwrap();
    let terminal = run_one.find("handle.await_terminal().await").unwrap();
    assert!(
        cancel < terminal,
        "timeout must request cancellation before awaiting terminal/lease release"
    );
}

#[test]
fn indirect_callers_use_typed_starts_without_json_or_prelease_data_work() {
    let supervisor = source("src/app_services/supervisor.rs");
    let execute = &supervisor[supervisor
        .find("async fn execute")
        .expect("supervisor execute")..];
    let supervisor_discovery = between(
        execute,
        "SupervisorAction::StartDiscovery { symbol, base_tf } => {",
        "SupervisorAction::StopDiscovery =>",
    );
    assert_absent(
        "supervisor discovery",
        supervisor_discovery,
        &[
            "Settings::from_yaml",
            "resolve_unique_background_dataset_identity",
            "serde_json::from_value",
            "engines_control::discovery_start(",
            "\"dataset_identity\"",
        ],
    );
    assert!(supervisor_discovery.contains("start_typed_discovery_execution_v1"));

    let supervisor_training = between(
        execute,
        "SupervisorAction::StartTraining { symbol, base_tf } => {",
        "SupervisorAction::StopTraining =>",
    );
    assert_absent(
        "supervisor training",
        supervisor_training,
        &["serde_json::from_value", "engines_control::training_start("],
    );
    assert!(supervisor_training.contains("start_typed_training_execution_v1"));

    let rediscovery = source("src/app_services/rediscovery.rs");
    let rediscovery_spawn =
        &rediscovery[rediscovery.find("pub fn spawn").expect("rediscovery spawn")..];
    assert_absent(
        "rediscovery worker",
        rediscovery_spawn,
        &[
            "Settings::from_yaml",
            "resolve_unique_background_dataset_identity",
            "serde_json::from_value",
            "engines_control::discovery_start(",
            "\"dataset_identity\"",
        ],
    );
    assert!(rediscovery_spawn.contains("start_typed_discovery_execution_v1"));
    assert!(rediscovery_spawn.contains("TypedLegacyExecutionStartErrorV1::Busy"));

    let federation = source("src/app_services/federation.rs");
    let federation_discovery = between(
        &federation,
        "// 2. Run the local discovery",
        "// 4. Submit every artifact",
    );
    assert_absent(
        "federation worker discovery",
        federation_discovery,
        &[
            "Settings::from_yaml",
            "resolve_unique_background_dataset_identity",
            "serde_json::from_value",
            "engines_control::discovery_start(",
            "StatusCode::CONFLICT",
            "\"dataset_identity\"",
        ],
    );
    assert!(federation_discovery.contains("start_typed_discovery_execution_v1"));
    assert!(federation_discovery.contains("TypedLegacyExecutionStartErrorV1::Busy"));
    assert!(federation_discovery.contains("handle.await_terminal().await"));
}

#[test]
fn public_headless_adapter_wraps_but_does_not_expose_or_detach_lane_authority() {
    let adapter = source("src/app_services/entrypoints.rs");

    assert!(adapter.contains("pub struct HeadlessExecutionHandleV1"));
    assert!(adapter.contains("run_headless_execution_pipeline_v1"));
    assert!(adapter.contains("start_typed_discovery_execution_v1"));
    assert!(adapter.contains("start_typed_training_execution_v1"));
    assert!(adapter.contains("training_after_success: intent.auto_training"));
    assert!(adapter.contains("pub fn cancel(&self)"));
    assert!(adapter.contains("pub async fn await_terminal(self)"));
    assert_absent(
        "public headless adapter",
        &adapter,
        &[
            "detach_typed_legacy_execution_observer_v1",
            "Settings::from_yaml",
            "resolve_unique_background_dataset_identity",
            "pin_current_discovery_input",
        ],
    );
}
