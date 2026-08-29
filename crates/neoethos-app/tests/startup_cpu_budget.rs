use std::path::PathBuf;
use std::process::Command;

fn source(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read startup source {}: {error}", path.display()))
}

#[test]
fn headless_app_installs_budget_before_building_tokio_runtime() {
    let source = source("src/main.rs");
    assert!(
        !source.contains("#[tokio::main]"),
        "the macro creates Tokio before NeoEthos can install its process budget"
    );

    let load = source
        .find("Settings::from_yaml")
        .expect("startup loads the operator config synchronously");
    let source_seal = source
        .find("initialize_source_seal_before_runtime")
        .expect("startup installs SourceSeal signal ownership synchronously");
    let install = source
        .find("install_process_budget")
        .expect("startup installs the one process CPU budget");
    let runtime = source
        .find("build_managed_runtime")
        .expect("startup builds the explicitly-sized Tokio runtime");
    let logging = source
        .find("setup_logging(true)")
        .expect("startup initializes logging after budget installation");
    let run = source
        .find("block_on")
        .expect("the synchronous entrypoint enters the managed runtime");
    assert!(
        source_seal < load
            && load < install
            && install < logging
            && logging < runtime
            && runtime < run
    );
}

#[test]
fn headless_app_exposes_structured_startup_diagnostics() {
    let source = source("src/main.rs");
    assert!(source.contains("startup_diagnostics_requested"));
    assert!(source.contains("format_startup_diagnostics"));
}

#[test]
fn headless_app_runtime_uses_the_resolved_parent_cap() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-app"))
        .current_dir(&repository)
        .args([
            "--config",
            "config.yaml",
            "--cpu-threads",
            "3",
            "--startup-diagnostics",
        ])
        .output()
        .expect("run headless startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected = expected_capped_workers(3);
    assert!(
        stderr.contains("schema=neoethos.startup.cpu_budget.v1")
            || stderr.contains("NEOETHOS_STARTUP_V1")
    );
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(stderr.contains(&format!("effective_worker_limit={expected}")));
    assert!(stderr.contains(&format!("runtime_worker_threads={expected}")));
    assert!(stderr.contains(
        "events=import_signal_preflight_completed,configuration_loaded,\
         parent_cpu_cap_parsed,cpu_budget_resolved,\
         cpu_budget_installed,runtime_settings_installed,tokio_runtime_built"
    ));
}

#[test]
fn headless_app_automatic_limit_leaves_exactly_two_logical_threads_free() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-app"))
        .current_dir(&repository)
        .args(["--config", "config.yaml", "--startup-diagnostics"])
        .output()
        .expect("run automatic startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected = expected_automatic_workers();
    assert!(stderr.contains("coordination_scope=process_local"));
    assert!(stderr.contains(&format!("automatic_worker_limit={expected}")));
    assert!(stderr.contains(&format!("effective_worker_limit={expected}")));
    assert!(stderr.contains(&format!("runtime_worker_threads={expected}")));
}

#[test]
fn invalid_parent_cap_fails_before_tokio_runtime_creation() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-app"))
        .current_dir(repository)
        .args([
            "--config",
            "config.yaml",
            "--cpu-threads",
            "0",
            "--startup-diagnostics",
        ])
        .output()
        .expect("run invalid startup diagnostic");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("tokio_runtime_built"));
}

fn expected_capped_workers(parent: usize) -> usize {
    let effective = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let reserved = effective.saturating_sub(1).min(2);
    (effective - reserved).min(parent)
}

fn expected_automatic_workers() -> usize {
    let effective = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    effective - effective.saturating_sub(1).min(2)
}
