use std::path::Path;

use neoethos_execution_budget::{
    DetectedCpuArchitectureV1, X8664V3FeatureSetV1, X8664V3SnapshotV1,
    evaluate_x86_64_v3_snapshot_v1,
};
use neoethos_x86_64_v3_launcher::{
    LauncherErrorCodeV1, LauncherErrorV1, PRIVATE_V3_PAYLOAD_MARKER_V1,
    PUBLIC_LAUNCHER_FAILURE_EXIT_CODE_V1, private_v3_payload_path_v1,
};

#[test]
fn derives_only_the_private_sibling_payload_for_windows_launchers() {
    assert_eq!(
        private_v3_payload_path_v1(Path::new(r"C:\NeoEthos\neoethos-cli.exe")).unwrap(),
        Path::new(r"C:\NeoEthos\neoethos-cli.x86-64-v3.exe")
    );
}

#[test]
fn derives_only_the_private_sibling_payload_for_linux_launchers() {
    assert_eq!(
        private_v3_payload_path_v1(Path::new("/opt/neoethos/neoethos-cli")).unwrap(),
        Path::new("/opt/neoethos/neoethos-cli.x86-64-v3")
    );
}

#[test]
fn refuses_a_launcher_path_without_a_file_name() {
    let error = private_v3_payload_path_v1(Path::new("/")).unwrap_err();
    assert_eq!(error.code(), LauncherErrorCodeV1::InvalidLauncherPath);
}

#[test]
fn cpu_refusal_is_preserved_as_a_fail_loud_launcher_diagnostic() {
    let preflight = evaluate_x86_64_v3_snapshot_v1(X8664V3SnapshotV1::new(
        DetectedCpuArchitectureV1::X8664,
        X8664V3FeatureSetV1::empty(),
    ))
    .unwrap_err();
    let launcher_error = LauncherErrorV1::from_cpu_preflight(preflight.clone());

    assert_eq!(
        launcher_error.code(),
        LauncherErrorCodeV1::CpuPreflightRefused
    );
    assert_eq!(launcher_error.to_string(), preflight.to_string());
    assert_eq!(
        launcher_error.exit_code(),
        PUBLIC_LAUNCHER_FAILURE_EXIT_CODE_V1
    );
}

#[test]
fn production_path_checks_cpu_before_resolving_or_starting_the_payload() {
    let source = include_str!("../src/launcher_v1.rs");
    let cpu_check = source.find("require_current_x86_64_v3_v1()").unwrap();
    let current_executable = source.find("current_exe()").unwrap();
    let child_start = source.find("Command::new(&payload_path)").unwrap();

    assert!(cpu_check < current_executable);
    assert!(current_executable < child_start);
    assert!(source.contains("args_os().skip(1)"));
    assert!(source.contains("payload_path.is_file()"));
}

#[test]
fn public_main_prints_the_typed_error_and_returns_nonzero() {
    let source = include_str!("../src/main.rs");

    assert!(source.contains("run_public_launcher_v1()"));
    assert!(source.contains("eprintln!(\"{error}\")"));
    assert!(source.contains("process::exit(error.exit_code())"));
}

#[test]
fn launcher_has_no_environment_path_or_feature_bypass() {
    let launcher_source = include_str!("../src/launcher_v1.rs");
    let main_source = include_str!("../src/main.rs");
    let manifest = include_str!("../Cargo.toml");
    let combined = format!("{launcher_source}\n{main_source}\n{manifest}");

    assert_eq!(PRIVATE_V3_PAYLOAD_MARKER_V1, ".x86-64-v3");
    for forbidden in [
        "std::env::var",
        "std::env::var_os",
        "NEOETHOS_SKIP",
        "NEOETHOS_DISABLE",
        "PATH",
        "target_feature(enable",
        "target-cpu=x86-64-v3",
    ] {
        assert!(
            !combined.contains(forbidden),
            "launcher must not contain bypass or v3 compilation token {forbidden}"
        );
    }
}
