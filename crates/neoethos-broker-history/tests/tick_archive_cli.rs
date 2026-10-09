#[test]
fn tick_archive_help_is_safe_and_exposes_explicit_scope_and_limits() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_neoethos-tick-archive"))
        .arg("--help")
        .output()
        .expect("start actual archive executable");
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for required in [
        "never places orders",
        "--account-id",
        "--symbol-id",
        "--from-ms",
        "--to-ms",
        "--max-archive-bytes",
        "--reserve-disk-bytes",
        "--stop-file",
    ] {
        assert!(help.contains(required), "missing {required}");
    }
}

#[test]
fn offline_inspector_requires_bounded_scope_and_independent_digest_without_credentials() {
    let executable = env!("CARGO_BIN_EXE_neoethos-tick-inspect");
    let output = std::process::Command::new(executable)
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for required in [
        "offline",
        "--archive",
        "--expected-page-hash-chain",
        "--from-ms",
        "--to-ms",
        "--max-quote-age-ms",
        "--max-events",
    ] {
        assert!(help.contains(required), "missing {required}");
    }
    let invalid = std::process::Command::new(executable)
        .arg("--archive")
        .arg("missing")
        .output()
        .unwrap();
    assert!(!invalid.status.success());
}
