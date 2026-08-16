use std::path::PathBuf;
use std::process::Command;

#[test]
fn mesh_installs_budget_before_building_tokio_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let source = std::fs::read_to_string(&path).expect("read mesh entrypoint");

    assert!(!source.contains("#[tokio::main]"));
    let main = &source[source.find("fn main() -> Result<()>").expect("sync main")..];
    let parse = main
        .find("parse_args")
        .expect("mesh parses config synchronously");
    let install = main
        .find("install_process_budget")
        .expect("mesh installs its process budget");
    let logging = main
        .find("tracing_subscriber::fmt")
        .expect("mesh installs tracing after its CPU budget");
    let runtime = main
        .find("build_managed_runtime")
        .expect("mesh builds an explicitly-sized Tokio runtime");
    assert!(parse < install && install < logging && logging < runtime);
    assert!(source.contains("--startup-diagnostics"));
}

#[test]
fn mesh_diagnostic_exits_before_identity_or_network_initialization() {
    let unique = format!(
        "neoethos-mesh-startup-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let data_dir = std::env::temp_dir().join(unique);
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-mesh"))
        .args([
            "--data-dir",
            data_dir.to_str().expect("UTF-8 temp path"),
            "--cpu-threads",
            "3",
            "--startup-diagnostics",
        ])
        .output()
        .expect("run mesh startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !data_dir.exists(),
        "diagnostic must not create mesh identity state"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("executable=neoethos-mesh"));
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(
        stderr.contains("events=parent_cpu_cap_parsed,cpu_budget_resolved,cpu_budget_installed")
    );
    assert!(!stderr.contains("configuration_loaded"));
    assert!(stderr.contains("tokio_runtime_built"));
}

#[test]
fn mesh_rejects_zero_before_runtime_or_network_initialization() {
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-mesh"))
        .args(["--cpu-threads", "0", "--startup-diagnostics"])
        .output()
        .expect("run invalid mesh startup diagnostic");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("tokio_runtime_built"));
}
