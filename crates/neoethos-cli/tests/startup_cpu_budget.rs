use std::path::PathBuf;
use std::process::Command;

fn isolated_config(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "neoethos-cli-startup-budget-{}-{name}.yaml",
        std::process::id()
    ));
    std::fs::write(&path, "{}\n").expect("write isolated startup-test config");
    path
}

#[test]
fn cli_installs_budget_before_dispatch_and_reports_no_async_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read CLI entrypoint");

    let install = source
        .find("let installed = neoethos_core::execution_budget::install_process_budget(")
        .expect("CLI installs the process budget");
    let logging = install
        + source[install..]
            .find("setup_logging(false)")
            .expect("CLI initializes logging after budget installation");
    let dispatch = source
        .find("if args.len() < 2")
        .expect("CLI reaches command dispatch");
    assert!(install < logging && logging < dispatch);
    assert!(!source.contains("#[tokio::main]"));
    assert!(source.contains("startup_diagnostics_requested"));
    assert!(source.contains("StartupRuntimeKind::Synchronous"));
}

#[test]
fn cli_reports_the_installed_synchronous_budget() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = isolated_config("reports");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
        .current_dir(repository)
        .env("CONFIG_FILE", &config)
        .args(["--startup-diagnostics", "--cpu-threads", "3"])
        .output()
        .expect("run CLI startup diagnostic");
    let _ = std::fs::remove_file(config);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("executable=neoethos-cli"));
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(stderr.contains("runtime_kind=synchronous"));
    assert!(stderr.contains("runtime_worker_threads=none"));
    assert!(stderr.contains("runtime_settings_installed"));
}

#[test]
fn cli_rejects_zero_before_command_dispatch() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = isolated_config("rejects-zero");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
        .current_dir(repository)
        .env("CONFIG_FILE", &config)
        .args(["--startup-diagnostics", "--cpu-threads", "0"])
        .output()
        .expect("run invalid CLI startup diagnostic");
    let _ = std::fs::remove_file(config);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("NEOETHOS_STARTUP_V1"));
}

#[test]
fn quote_bundle_digest_is_checked_before_broker_inputs_or_output_creation() {
    let settings = isolated_config("quote-replay-digest");
    let destination = settings.with_extension("replay-output");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
        .arg("canonical-research")
        .arg("--settings-source")
        .arg(&settings)
        .arg("--quote-replay-bundle")
        .arg(&settings)
        .arg("--out-dir")
        .arg(&destination)
        .args([
            "--quote-replay-sha256",
            &"0".repeat(64),
            "--quote-replay-store",
            "absent-store",
            "--authority-root",
            "absent-authority",
            "--data-root",
            "absent-data",
            "--plan-sha256",
            "absent-plan",
            "--matrix-sha256",
            "absent-matrix",
            "--broker-symbol-contract",
            "absent-contract",
            "--broker-account-snapshot",
            "absent-account",
            "--symbol",
            "EURUSD",
            "--base-timeframe",
            "H4",
            "--max-batches",
            "1",
            "--cpu-threads",
            "2",
        ])
        .output()
        .expect("launch current CLI");
    let _ = std::fs::remove_file(settings);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("quote replay bundle SHA256 mismatch"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !destination.exists(),
        "invalid replay created research output"
    );
}

#[test]
fn discovery_executes_inside_the_exact_installed_cpu_lease() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read CLI entrypoint");
    let wrapper = source
        .split_once("fn cmd_discover(args: &[String])")
        .expect("discover wrapper")
        .1
        .split_once("fn cmd_discover_on_budgeted_pool(")
        .expect("budgeted discover body")
        .0;

    for required in [
        "installed_process_budget()",
        "effective_worker_limit",
        "CpuPermitRequest::local(",
        "BudgetedCpuExecutor::new_for_broker",
        ".execute_with_scope(lease.into_transfer()",
        "BudgetedCpuExecutor::current_pool_width()",
        "observed_width == width.get()",
        "cmd_discover_on_budgeted_pool(args, cpu)",
    ] {
        assert!(
            wrapper.contains(required),
            "Discovery CPU wrapper is missing `{required}`"
        );
    }
}
