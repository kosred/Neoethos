use std::path::PathBuf;
use std::process::Command;

fn source(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read desktop source {}: {error}", path.display()))
}

#[test]
fn desktop_installs_and_retains_custom_runtime_before_tauri_builder() {
    let source = source("src/lib.rs");
    let run = &source[source
        .rfind("pub fn run()")
        .expect("desktop run entrypoint")..];
    let preflight = run
        .find("prepare_desktop_startup")
        .expect("desktop performs synchronous config and budget preflight");
    let builder = run
        .find("tauri::Builder::default")
        .expect("desktop starts the application builder");
    assert!(preflight < builder);

    let prepare = &source[source
        .find("fn prepare_desktop_startup")
        .expect("desktop startup preflight implementation")..];
    let config = prepare
        .find("Settings::from_yaml")
        .expect("desktop loads configuration synchronously");
    let install = prepare
        .find("install_process_budget")
        .expect("desktop installs its CPU budget");
    let runtime = prepare
        .find("DesktopRuntimeGuard::build")
        .expect("desktop builds its retained Tokio runtime");
    let logging = prepare
        .find("neoethos_core::logging::setup_logging(false)")
        .expect("desktop installs the shared release and debug logger");
    assert!(config < logging && logging < runtime);
    assert!(
        !source.contains("tauri_plugin_log::Builder"),
        "a second process-global logger would conflict with the shared tracing bridge"
    );
    let tauri_runtime = prepare
        .find("install_for_tauri")
        .expect("desktop installs the managed runtime for Tauri");
    assert!(config < install && install < runtime && runtime < tauri_runtime);
    assert!(
        tauri_runtime
            < prepare
                .find("Ok(PreparedDesktopStartup")
                .expect("completed preflight")
    );
    assert!(source.contains("DesktopRuntimeGuard"));
}

#[test]
fn desktop_entrypoint_exposes_startup_diagnostics_without_opening_a_window() {
    let main = source("src/main.rs");
    assert!(main.contains("--startup-diagnostics"));
    assert!(main.contains("startup_diagnostics"));
}

#[test]
fn desktop_first_run_seeds_exact_embedded_files_before_managed_tauri_runtime() {
    let data_root = unique_temp_root("seed");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-desktop"))
        .env("NEOETHOS_USER_DATA_DIR", &data_root)
        .env("LOG_DIR", data_root.join("logs"))
        .env_remove("RUST_LOG")
        .args(["--cpu-threads", "3", "--startup-diagnostics"])
        .output()
        .expect("run desktop startup diagnostic");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("executable=neoethos-desktop"));
    assert!(stderr.contains("runtime_kind=tauri"));
    assert!(stderr.contains("coordination_scope=managed_process_tree"));
    assert!(stderr.contains(&format!(
        "runtime_worker_threads={}",
        expected_capped_workers(3)
    )));
    assert!(stderr.contains("tauri_async_runtime_installed"));
    let log_files = std::fs::read_dir(data_root.join("logs"))
        .expect("desktop startup must create its runtime log directory")
        .map(|entry| entry.expect("runtime log entry").path())
        .collect::<Vec<_>>();
    assert_eq!(log_files.len(), 1);
    let runtime_log = std::fs::read_to_string(&log_files[0]).expect("read full runtime log");
    assert!(runtime_log.contains("Logging initialized (verbose=false)"));
    assert!(runtime_log.contains("> [SYSTEM] SUCCESS setup_logging"));

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        std::fs::read(data_root.join("config.yaml")).expect("seeded config"),
        std::fs::read(manifest.join("resources/config.yaml")).expect("embedded config source")
    );
    assert_eq!(
        std::fs::read(data_root.join("data/symbol_metadata.json")).expect("seeded symbol metadata"),
        std::fs::read(manifest.join("resources/symbol_metadata.json"))
            .expect("embedded metadata source")
    );
    remove_temp_root(&data_root);
}

#[test]
fn desktop_rejects_malformed_parent_cap_before_runtime_installation() {
    let data_root = unique_temp_root("invalid-cap");
    let output = Command::new(env!("CARGO_BIN_EXE_neoethos-desktop"))
        .env("NEOETHOS_USER_DATA_DIR", &data_root)
        .env("LOG_DIR", data_root.join("logs"))
        .env_remove("RUST_LOG")
        .args(["--cpu-threads", "invalid", "--startup-diagnostics"])
        .output()
        .expect("run invalid desktop startup diagnostic");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expects a positive integer"));
    assert!(!stderr.contains("tokio_runtime_built"));
    remove_temp_root(&data_root);
}

fn unique_temp_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "neoethos-desktop-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(root.starts_with(std::env::temp_dir()));
    root
}

fn remove_temp_root(root: &std::path::Path) {
    assert!(root.starts_with(std::env::temp_dir()));
    if root.exists() {
        std::fs::remove_dir_all(root).expect("remove isolated desktop test root");
    }
}

fn expected_capped_workers(parent: usize) -> usize {
    let effective = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let reserved = effective.saturating_sub(1).min(2);
    (effective - reserved).min(parent)
}
