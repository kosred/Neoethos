use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).unwrap_or_default()
}

fn sibling(crate_name: &str, relative: &str) -> String {
    fs::read_to_string(manifest_dir().join("..").join(crate_name).join(relative))
        .unwrap_or_default()
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?}"))
        .0
}

fn require_all(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "missing pinned-series token {token:?}"
        );
    }
}

#[test]
fn pinning_owns_exact_manifests_and_reader_leases_without_decoding_values() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    assert!(read("src/core/mod.rs").contains("pub mod pinned_canonical_series_v1;"));
    assert!(read("src/lib.rs").contains("PinnedCanonicalSeriesV1, pin_exact_canonical_series_v1"));
    require_all(
        &source,
        &[
            "pub struct PinnedCanonicalSeriesV1",
            "DatasetManifestV1",
            "DatasetGenerationLease",
            "open_exact_dataset_generation",
            "pin_exact_canonical_series_v1",
        ],
    );
    let pin = section(&source, "pub fn pin_exact_canonical_series_v1", "\n}\n");
    for forbidden in [
        "load_vortex",
        "load_exact_canonical_timeframe",
        "load_exact_dataset_series_receipt",
        "FeatureFrame",
        "Ohlcv",
    ] {
        assert!(
            !pin.contains(forbidden),
            "pinning decoded/materialized data through {forbidden:?}"
        );
    }
}

#[test]
fn cpu_decode_consumes_the_pin_and_requires_the_sealed_zero_gpu_authority() {
    let source = read("src/core/pinned_canonical_series_v1.rs");
    let decode = section(
        &source,
        "pub fn into_cpu_dataset_after_no_physical_gpu_v1",
        "\n    }\n",
    );
    require_all(
        decode,
        &[
            "self",
            "SealedCpuNoPhysicalGpuRunDeviceAdmissionV1",
            "materialize_pinned_canonical_series_v1",
        ],
    );
    let materialize = section(
        &source,
        "fn materialize_pinned_canonical_series_v1",
        "\n    }\n",
    );
    require_all(
        materialize,
        &[
            "materialize_pinned_canonical_timeframe_v1",
            "generation.lease",
        ],
    );
    let canonical = read("src/core/canonical_ohlcv.rs");
    let reopen = section(
        &canonical,
        "pub(crate) fn materialize_pinned_canonical_timeframe_v1",
        "\n}\n",
    );
    require_all(
        reopen,
        &["lease.reopen_verified()", "vortex_array_to_ohlcv"],
    );
    assert!(!reopen.contains("load_vortex"));
    for forbidden in [
        "Clone for PinnedCanonicalSeriesV1",
        "current",
        "inventory",
        "unwrap",
    ] {
        assert!(!decode.contains(forbidden), "decode escape {forbidden:?}");
    }
}

#[test]
fn app_moves_one_pin_only_inside_the_selected_prepared_factory() {
    let app = sibling("neoethos-app", "src/app_services/discovery.rs");
    require_all(
        &app,
        &[
            "PinnedCanonicalSeriesV1",
            "take_pinned_series_v1",
            "prepare_canonical_discovery_run_input_v3",
            "into_cpu_dataset_after_no_physical_gpu_v1",
        ],
    );
    let worker = section(
        &app,
        "let feature_handle = tokio::task::spawn_blocking",
        "\n        });",
    );
    let dispatch = worker
        .find("prepare_canonical_discovery_run_input_v3")
        .expect("prepared dispatcher");
    let take = worker
        .find("take_pinned_series_v1")
        .expect("move-only pin take");
    let decode = worker
        .find("into_cpu_dataset_after_no_physical_gpu_v1")
        .expect("authorized CPU decode");
    assert!(dispatch < take && take < decode);
}

#[test]
fn cli_pins_exact_leases_before_dispatch_and_never_reopens_from_a_receipt() {
    let cli = sibling("neoethos-cli", "src/main.rs");
    let full = sibling("neoethos-cli", "src/canonical_full_run.rs");
    let discover = section(
        &cli,
        "fn cmd_discover(args: &[String])",
        "\nfn cmd_batch_discover",
    );
    let full_run = section(
        &full,
        "pub fn run(args: &[String], settings: &neoethos_core::Settings)",
        "\n#[cfg(not(feature = \"gpu-nvidia-full\"))]",
    );
    for (label, source, pin_token) in [
        (
            "discover",
            discover,
            "let mut pinned_selection = pin_direct_timeframe_selection",
        ),
        (
            "full",
            full_run,
            "let pinned_series = pin_exact_canonical_series_v1",
        ),
    ] {
        require_all(
            source,
            &[
                "prepare_canonical_discovery_run_input_v3",
                "into_cpu_dataset_after_no_physical_gpu_v1",
                pin_token,
            ],
        );
        let pin = source.find(pin_token).unwrap();
        let dispatch = source
            .find("prepare_canonical_discovery_run_input_v3")
            .unwrap();
        assert!(pin < dispatch, "{label} must pin before device admission");
        assert!(
            !source.contains("CanonicalSearchInput::from_exact_series_receipt"),
            "{label} may not detach the receipt and reopen values"
        );
    }
}

#[test]
fn autoresearch_pins_before_its_top_level_route_and_repins_before_each_batch_route() {
    let source = sibling("neoethos-autoresearch", "src/runner/streaming.rs");
    require_all(
        &source,
        &[
            "pin_exact_canonical_series_v1",
            "dispatch_canonical_discovery_data_preparation_v3",
            "into_cpu_dataset_after_no_physical_gpu_v1",
            "run_prepared_streaming_working_set_v3",
        ],
    );
    let resolve = section(&source, "pub fn resolve(", "\n    /// Supply one sealed");
    let pin = resolve.find("pin_exact_canonical_series_v1").unwrap();
    let dispatch = resolve
        .find("dispatch_canonical_discovery_data_preparation_v3")
        .unwrap();
    let decode = resolve
        .find("into_cpu_dataset_after_no_physical_gpu_v1")
        .unwrap();
    assert!(pin < dispatch && dispatch < decode);
    assert!(!resolve.contains("load_dataset_for_identity_with_timeframes"));

    let execute = section(&source, "fn execute(&mut self", "\n    fn evaluate_oos(");
    let batch_dispatch = execute
        .find("run_prepared_streaming_working_set_v3")
        .unwrap();
    let batch_pin = execute[batch_dispatch..]
        .find("pin_exact_canonical_series_v1")
        .unwrap();
    let batch_decode = execute[batch_dispatch..]
        .find("into_cpu_dataset_after_no_physical_gpu_v1")
        .unwrap();
    assert!(batch_pin < batch_decode);
}
