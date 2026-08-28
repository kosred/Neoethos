use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

#[test]
fn capability_preflight_has_no_retired_cpu_escape() {
    let source = read("src/gpu_native/capability.rs");
    assert!(
        !source.contains("FallbackPolicy::AllowCpu"),
        "the retired fallback variant cannot bypass strict GPU capability checks"
    );
    let preflight = section(
        &source,
        "pub fn gpu_pipeline_preflight(",
        "#[derive(Debug, Clone, PartialEq, Eq)]",
    );
    let scan_at = preflight
        .find("let mut unsupported")
        .expect("strict preflight must scan unsupported stages");
    let decision_at = preflight
        .find("if unsupported.is_empty()")
        .expect("strict preflight must decide only from the scanned manifest");
    assert!(scan_at < decision_at);
    assert!(
        !preflight[..scan_at].contains("return Ok"),
        "backend policy cannot return success before the strict manifest scan"
    );

    let auto_test = section(
        &source,
        "fn auto_backend_cannot_bypass_strict_preflight()",
        "#[test]",
    );
    assert!(auto_test.contains("EvaluationBackend::AUTO"));
    assert!(auto_test.contains(".unwrap_err()"));
}

#[test]
fn device_identity_probe_does_not_claim_mutable_session_access() {
    let source = read("src/strict_discovery_device_route_v1.rs");
    let probe = section(
        &source,
        "for ordinal in 0..reported_device_count",
        "let observation = StrictDiscoveryProbeObservationV1",
    );
    assert!(probe.contains("let session = match neoethos_gpu_cuda::PopulationSession::create("));
    assert!(
        !probe.contains("let mut session"),
        "read-only device identity probing must not retain an unnecessary mutable binding"
    );
    assert!(probe.contains("session.read_device_identity_v1()"));
}
