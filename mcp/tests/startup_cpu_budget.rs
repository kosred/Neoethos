use std::path::PathBuf;
use std::process::Command;

#[test]
fn sidecar_installs_budget_before_building_tokio_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read MCP sidecar entrypoint");

    assert!(!source.contains("#[tokio::main]"));
    let main = &source[source.find("fn main() -> Result<()>").expect("sync main")..];
    let config = main
        .find("load_config")
        .expect("config is loaded synchronously");
    let install = main
        .find("install_process_budget")
        .expect("MCP installs its process budget");
    let logging = main
        .find("tracing_subscriber::fmt")
        .expect("MCP installs tracing after its CPU budget");
    let runtime = main
        .find("build_managed_runtime")
        .expect("MCP builds an explicitly-sized Tokio runtime");
    assert!(config < install && install < logging && logging < runtime);
    assert!(source.contains("--startup-diagnostics"));
}

#[test]
fn sidecar_diagnostic_builds_only_the_budgeted_runtime() {
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-mcp"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["--cpu-threads=3", "--startup-diagnostics"])
        .output()
        .expect("run MCP startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("executable=neoethos-mcp"));
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(stderr.contains("runtime_kind=tokio"));
    assert!(stderr.contains("configuration_loaded,parent_cpu_cap_parsed"));
}

#[test]
fn sidecar_rejects_duplicate_parent_caps_before_tokio() {
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-mcp"))
        .args([
            "--cpu-threads=2",
            "--cpu-threads",
            "3",
            "--startup-diagnostics",
        ])
        .output()
        .expect("run invalid MCP startup diagnostic");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("may be supplied only once"));
    assert!(!stderr.contains("tokio_runtime_built"));
}
