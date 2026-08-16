use std::path::PathBuf;
use std::process::Command;

#[test]
fn control_plane_installs_budget_before_building_tokio_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read control-plane entrypoint");

    assert!(!source.contains("#[tokio::main]"));
    let install = source
        .find("install_process_budget")
        .expect("control plane installs its process budget");
    let logging = source
        .find("tracing_subscriber::fmt")
        .expect("control plane installs tracing after its CPU budget");
    let runtime = source
        .find("build_managed_runtime")
        .expect("control plane builds an explicitly-sized Tokio runtime");
    assert!(install < logging && logging < runtime);
    assert!(source.contains("--cpu-threads"));
    assert!(source.contains("--startup-diagnostics"));
}

#[test]
fn control_plane_keeps_stdout_clean_and_sizes_tokio_from_the_budget() {
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-control-plane"))
        .args(["--cpu-threads", "3", "--startup-diagnostics"])
        .output()
        .expect("run control-plane startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "stdout is reserved for JSON-RPC");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("executable=neoethos-control-plane"));
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(stderr.contains("runtime_kind=tokio"));
    assert!(stderr.contains("tokio_runtime_built"));
}

#[test]
fn control_plane_rejects_malformed_parent_cap_before_tokio() {
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-control-plane"))
        .args(["--cpu-threads", "many", "--startup-diagnostics"])
        .output()
        .expect("run invalid control-plane startup diagnostic");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("tokio_runtime_built"));
}
