//! Source contract for the broker-truth acquisition CLI entrypoint.
//!
//! `docs/superpowers/plans/2026-08-23-broker-truth-acquisition-entrypoint.md`
//! requires a runnable one-shot acquisition binary that parses
//! `BrokerTruthAcquisitionArgsV1`, prepares the exact reviewed inputs, and
//! executes the prepared acquisition with the production cancellation token.

use std::path::PathBuf;

fn read(path: &str) -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(manifest.join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn acquisition_cli_parses_and_executes_prepared_acquisition() {
    let main = read("src/main.rs");
    assert!(main.contains("BrokerTruthAcquisitionArgsV1::try_parse_from"),
        "acquisition CLI must parse the reviewed argument set: {main}");
    assert!(main.contains("prepare_acquisition_v1"),
        "acquisition CLI must prepare the reviewed acquisition inputs: {main}");
    assert!(main.contains("execute_prepared_acquisition_v1"),
        "acquisition CLI must execute the prepared acquisition: {main}");
    assert!(main.contains("ProductionBrokerTruthCancellationV2::new"),
        "acquisition CLI must use the production cancellation token: {main}");
}

#[test]
fn acquisition_cli_reports_failures_without_success_output() {
    let main = read("src/main.rs");
    assert!(main.contains("ExitCode") || main.contains("process::exit"),
        "acquisition CLI must return a process exit code: {main}");
    assert!(main.contains("eprintln"),
        "acquisition CLI must print failures to stderr: {main}");
}
