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
