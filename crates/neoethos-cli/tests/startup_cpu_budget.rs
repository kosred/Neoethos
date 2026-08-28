use std::path::PathBuf;
use std::process::Command;

#[test]
fn cli_installs_budget_before_dispatch_and_reports_no_async_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read CLI entrypoint");

    let install = source
        .find("install_process_budget")
        .expect("CLI installs the process budget");
    let logging = source
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
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
        .current_dir(repository)
        .args(["--startup-diagnostics", "--cpu-threads", "3"])
        .output()
        .expect("run CLI startup diagnostic");
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
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
        .current_dir(repository)
        .args(["--startup-diagnostics", "--cpu-threads", "0"])
        .output()
        .expect("run invalid CLI startup diagnostic");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("NEOETHOS_STARTUP_V1"));
}
