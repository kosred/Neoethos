use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).expect("read required Search source")
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
            "missing prepared streaming token {token:?}"
        );
    }
}

#[test]
fn every_batch_dispatches_one_fresh_prepared_admission_before_materialization() {
    let source = read("src/orchestration.rs");
    let prepared = section(
        &source,
        "pub fn run_prepared_streaming_working_set_v3",
        "\n}\n",
    );
    require_all(
        prepared,
        &[
            "run_streaming_working_set_core",
            "pin_factory",
            "prepare_canonical_discovery_run_input_v3",
            "cpu_factory",
            "native_workspace_plan_factory",
            "native_factory",
            "run_prepared",
        ],
    );
    assert_eq!(
        prepared
            .matches("prepare_canonical_discovery_run_input_v3(")
            .count(),
        1,
        "one loop iteration must perform one and only one admission dispatch"
    );
    let dispatch = prepared
        .find("prepare_canonical_discovery_run_input_v3(")
        .expect("prepared dispatch");
    let pin = prepared.find("pin_factory(").expect("pre-admission pin");
    let cpu_build = prepared.find("cpu_factory(").expect("CPU batch factory");
    let native_build = prepared
        .find("native_factory(")
        .expect("native batch factory");
    assert!(pin < dispatch && dispatch < cpu_build && dispatch < native_build);
}

#[test]
fn prepared_streaming_loop_preserves_order_shuffle_and_receipt_device_binding() {
    let source = read("src/orchestration.rs");
    let prepared = section(
        &source,
        "pub fn run_prepared_streaming_working_set_v3",
        "\n}\n",
    );
    require_all(
        prepared,
        &[
            "Option<Arc<neoethos_data::core::hpc_ta::SweepBatch>>",
            "PreparedCanonicalDiscoveryRunInputV3",
            ".feature_names()",
            "run_prepared(prepared)",
        ],
    );
    for forbidden in [
        "acquire_discovery_run_device_admission_v1",
        "probe_cuda_device_count_v1",
        "device_count",
        "FeatureFrame::",
        "CanonicalSearchRunInputV2::new",
        "run_discovery_cycle_with_holdout(",
        "clone_admission",
        "reuse_admission",
    ] {
        assert!(
            !prepared.contains(forbidden),
            "prepared streaming loop contains forbidden authority/materialization escape {forbidden:?}"
        );
    }
}

#[test]
fn legacy_streaming_api_delegates_to_the_same_ordered_core() {
    let source = read("src/orchestration.rs");
    let legacy = section(&source, "pub fn run_streaming_working_set", "\n}\n");
    require_all(
        legacy,
        &[
            "run_streaming_working_set_core",
            "build_features(batch)",
            "run_cycle(&features)",
        ],
    );
}
