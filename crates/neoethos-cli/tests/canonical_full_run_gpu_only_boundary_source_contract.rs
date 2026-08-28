use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-cli"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, rest) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source marker {start:?}"));
    rest.split_once(end)
        .unwrap_or_else(|| panic!("missing source marker {end:?} after {start:?}"))
        .0
}

fn require_before(source: &str, earlier: &str, later: &str) {
    let earlier_at = source
        .find(earlier)
        .unwrap_or_else(|| panic!("missing required earlier boundary {earlier:?}"));
    let later_at = source
        .find(later)
        .unwrap_or_else(|| panic!("missing required later boundary {later:?}"));
    assert!(
        earlier_at < later_at,
        "{earlier:?} must execute before {later:?}"
    );
}

#[test]
fn canonical_full_run_acquires_exact_ordinal_strict_gpu_permit_before_materialization() {
    let source = read("src/canonical_full_run.rs");
    let run = section(&source, "pub fn run(", "\nfn publish_full_run_artifact(");

    for required in [
        "ExactCudaDeviceOrdinalV1",
        "StrictGpuOnlyFullDiscoveryPermitV1",
        "acquire_strict_gpu_only_full_discovery_permit_v1(",
        "PipelineStage::FULL_DISCOVERY",
    ] {
        assert!(
            run.contains(required),
            "canonical full-run is missing strict GPU admission token {required:?}"
        );
    }

    for materialization in [
        "load_exact_canonical_timeframe(",
        "canonical_discovery_normalization_training_rows(",
        "CanonicalSearchInput::from_exact_series_receipt(",
    ] {
        require_before(
            run,
            "acquire_strict_gpu_only_full_discovery_permit_v1(",
            materialization,
        );
    }

    assert!(
        !run.contains("StrictGpuOnlyFullDiscoveryPermitV1 {"),
        "the CLI must consume Search's sealed permit, never construct one locally"
    );
}

#[test]
fn canonical_full_run_cannot_treat_gpu_preferred_or_allow_cpu_as_admission() {
    let source = read("src/canonical_full_run.rs");
    let run = section(&source, "pub fn run(", "\nfn publish_full_run_artifact(");

    for required in [
        "require_exact_cuda_device_ordinal_v1(",
        "require_all_full_discovery_stages_strict_gpu_v1(",
        "acquire_strict_gpu_only_full_discovery_permit_v1(",
    ] {
        assert!(
            run.contains(required),
            "canonical full-run can start without fail-closed check {required:?}"
        );
    }
    for forbidden in [
        "GPU_PREFERRED",
        "FallbackPolicy::AllowCpu",
        "DevicePreference::Auto",
        "pin_cpu_only_model_device",
    ] {
        assert!(
            !run.contains(forbidden),
            "canonical full-run retains forbidden GPU admission escape {forbidden:?}"
        );
    }
}

#[test]
fn canonical_full_run_accepts_only_a_sealed_compact_discovery_receipt() {
    let source = read("src/canonical_full_run.rs");
    let run = section(&source, "pub fn run(", "\nfn publish_full_run_artifact(");

    for required in [
        "run_canonical_trendbar_gpu_only_compact_v1(",
        "SealedCompactGpuOnlyDiscoveryReceiptV1",
        "validate_against_gpu_only_permit(",
        "CanonicalGpuOnlyFullRunArtifactV2",
        "compact_discovery_receipt:",
    ] {
        assert!(
            run.contains(required) || source.contains(required),
            "canonical full-run compact result boundary is missing {required:?}"
        );
    }

    for forbidden in [
        "let result = research.discovery_result();",
        "discovery_result: result.clone(),",
        "artifact.discovery_result.portfolio",
    ] {
        assert!(
            !source.contains(forbidden),
            "canonical full-run still returns or serializes a full Search payload via {forbidden:?}"
        );
    }
}

#[test]
fn compact_full_run_artifact_cannot_embed_heavy_search_matrices() {
    let source = read("src/canonical_full_run.rs");
    let artifact = section(
        &source,
        "struct CanonicalGpuOnlyFullRunArtifactV2 {",
        "\n}\n\n",
    );

    assert!(artifact.contains("compact_discovery_receipt: SealedCompactGpuOnlyDiscoveryReceiptV1"));
    for forbidden in [
        "FeatureFrame",
        "features:",
        "signals:",
        "trades:",
        "folds:",
        "metric_matrix",
        "DiscoveryResult",
    ] {
        assert!(
            !artifact.contains(forbidden),
            "compact full-run artifact embeds forbidden heavy field {forbidden:?}"
        );
    }
}
