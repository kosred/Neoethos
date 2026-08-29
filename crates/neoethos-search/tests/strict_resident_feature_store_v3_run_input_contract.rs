use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).unwrap_or_default()
}

fn read_gpu_cuda_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join("../neoethos-gpu-cuda").join(relative))
        .unwrap_or_default()
}

fn read_data_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join("../neoethos-data").join(relative)).unwrap_or_default()
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn normalized(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing strict V3 token {token:?}");
    }
}

fn require_none(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "strict V3 run-input path contains forbidden token {token:?}"
        );
    }
}

fn combined_gpu_cuda_source() -> String {
    format!(
        "{}\n{}",
        read_gpu_cuda_or_empty("src/population.rs"),
        read_gpu_cuda_or_empty("src/resident_feature_store_v3.rs")
    )
}

#[test]
fn sealed_store_is_consumed_once_only_after_exact_identity_validation() {
    let source = read_or_empty("src/strict_resident_feature_store_v3.rs");
    assert!(
        source.contains("pub(crate) fn bind_strict_resident_feature_store_v3_run_input("),
        "missing additive strict resident Search run-input binder"
    );
    let bind = section(
        &source,
        "pub(crate) fn bind_strict_resident_feature_store_v3_run_input(",
        "\n}",
    );
    let validate = section(
        &source,
        "fn validate_strict_resident_feature_store_v3(",
        "\n}",
    );
    let compact = normalized(bind);
    let validation_and_bind = format!("{validate}\n{bind}");

    require_all(
        &validation_and_bind,
        &[
            "sealed_store: SealedGpuResidentFeatureStoreV3",
            "CanonicalSearchArtifactScopeV2",
            "admission_identity",
            "feature_plan",
            "normalization",
            "provenance",
            "content",
            "ordered_feature",
            "device",
            "build",
            "ordinal",
            "context",
            "stream",
        ],
    );
    assert!(
        !bind.contains("&SealedGpuResidentFeatureStoreV3"),
        "the sealed store must move into exactly one Search run"
    );
    let validate_at = compact
        .find("validate")
        .expect("exact identity validation must be explicit");
    let import_at = compact
        .find("import_v3")
        .expect("validated sealed store must become one opaque V3 import");
    assert!(
        validate_at < import_at,
        "validation must occur before or atomically within sealed-store import consumption"
    );
    assert!(
        compact.contains("into_") && compact.contains("import_v3"),
        "Search must consume, not borrow or clone, the sealed store into its import"
    );
    require_all(
        bind,
        &[
            "Result<StrictResidentPopulationExecutionRunV3",
            "bind_resident_feature_store_v3(resident_import, scope)",
        ],
    );
    require_none(
        bind,
        &[
            "ExactPopulationExecutionRunV1",
            "&mut",
            "acquire_strict_discovery_device_admission_v1",
            "require_exact_cuda_device_ordinal_v1",
            "selected_ordinal()",
        ],
    );
}

#[test]
fn data_seal_owns_the_only_transition_to_an_admitted_stream_import() {
    let data = read_data_or_empty("src/core/gpu_resident_feature_store_v3.rs");
    let consume = section(
        &data,
        "pub fn into_resident_feature_store_import_v3(",
        "\n}",
    );
    require_all(
        consume,
        &[
            "self",
            "validate_resident_feature_store_import_v3",
            ".import_on_admitted_run_stream_v3()",
            "ResidentFeatureStoreImportV3",
        ],
    );
    require_none(
        consume,
        &[
            "&self",
            "Context::new(",
            "Stream::new(",
            "Device::get_device(",
            "as_raw()",
            "clone()",
        ],
    );
}

#[test]
fn opaque_import_is_consumed_into_a_population_session_on_the_admitted_run_stream() {
    let search = read_or_empty("src/strict_resident_feature_store_v3.rs");
    let gpu_cuda = combined_gpu_cuda_source();
    assert!(
        gpu_cuda.contains("pub struct ResidentPopulationSessionV3"),
        "gpu-cuda lacks the opaque run-owned resident population session"
    );
    let consume = section(
        &gpu_cuda,
        "pub fn consume_into_population_session_v3(",
        "\n}",
    );

    require_all(
        &search,
        &[
            "ResidentPopulationSessionV3",
            ".consume_into_population_session_v3(",
        ],
    );
    require_all(
        consume,
        &[
            "self",
            "ResidentPopulationSessionV3",
            "admitted",
            "run_stream",
        ],
    );
    assert!(
        !consume.contains("&self"),
        "one opaque import may mint only one population session"
    );
    require_none(
        consume,
        &[
            "PopulationSession::create(",
            "Context::new(",
            "Device::get_device(",
            "create_stream",
            "cudaStreamCreate",
        ],
    );
}

#[test]
fn strict_run_input_has_no_host_materialization_transfer_or_fallback_escape() {
    let search = read_or_empty("src/strict_resident_feature_store_v3.rs");
    let gpu_cuda = combined_gpu_cuda_source();
    assert!(
        search.contains("pub(crate) fn bind_strict_resident_feature_store_v3_run_input("),
        "host-boundary refusals require the real strict Search binder"
    );
    assert!(
        gpu_cuda.contains("pub fn consume_into_population_session_v3("),
        "host-boundary refusals require the real gpu-cuda consume boundary"
    );
    let consume = if gpu_cuda.contains("pub fn consume_into_population_session_v3(") {
        section(
            &gpu_cuda,
            "pub fn consume_into_population_session_v3(",
            "\n}",
        )
    } else {
        ""
    };

    require_none(
        &search,
        &[
            "FeatureFrame",
            "Ohlcv",
            "Cow<",
            "row_window",
            "acquire_strict_discovery_device_admission_v1",
            "acquire_discovery_run_device_admission_v1",
            "as_device_ptr",
            "as_raw()",
            "CUcontext",
            "CUstream",
            "unsafe",
            "upload_dataset",
            "upload_parent_dataset_v1",
            "upload_genes",
            "upload_scenarios",
            "copy_to_device",
            "copy_to_host",
            "read_metrics",
            ".wait(",
            "H2D",
            "D2H",
            "Cpu",
            "f32",
            "fallback",
        ],
    );
    require_none(
        consume,
        &[
            "FeatureFrame",
            "Ohlcv",
            "Cow<",
            "row_window",
            "acquire_strict_discovery_device_admission_v1",
            "acquire_discovery_run_device_admission_v1",
            "PopulationSession::create(",
            "Context::new(",
            "Device::get_device(",
            "create_stream",
            "upload_dataset",
            "upload_parent_dataset_v1",
            "copy_to_device",
            "copy_to_host",
            "Cpu",
            "f32",
            "fallback",
        ],
    );
}

#[test]
fn standalone_native_run_owns_session_and_shape_without_a_host_v1_parent() {
    let search = read_or_empty("src/strict_resident_feature_store_v3.rs");
    let evidence = read_or_empty("src/population_execution_evidence_v1.rs");
    let run = section(
        &search,
        "pub(crate) struct StrictResidentPopulationExecutionRunV3 {",
        "\n}",
    );

    require_all(
        run,
        &[
            "session: Option<ResidentPopulationSessionV3>",
            "row_count: usize",
            "column_count: usize",
        ],
    );
    require_none(
        &evidence,
        &[
            "resident_feature_store_session_v3",
            "parent_row_count",
            "parent_feature_count",
        ],
    );
    require_none(
        &search,
        &[
            "ExactPopulationExecutionRunV1",
            "begin_exact_population_execution_run_v1",
            "PopulationParentDatasetV1",
        ],
    );
}

#[test]
fn armed_import_session_and_completion_lease_fail_closed_on_drop() {
    let gpu_cuda = combined_gpu_cuda_source();
    assert!(
        gpu_cuda.contains("pub struct ResidentPopulationSessionV3"),
        "gpu-cuda lacks the resident population lifetime owner"
    );
    let wrapper = section(
        &gpu_cuda,
        "pub struct ResidentPopulationSessionV3",
        "impl ResidentPopulationSessionV3",
    );
    let implementation = section(
        &gpu_cuda,
        "impl ResidentPopulationSessionV3",
        "impl Drop for ResidentPopulationSessionV3",
    );
    let lease_implementation = section(
        &gpu_cuda,
        "impl ResidentFeatureStoreConsumerLeaseV3",
        "impl Drop for ResidentFeatureStoreConsumerLeaseV3",
    );
    let drop_impl = section(
        &gpu_cuda,
        "impl Drop for ResidentPopulationSessionV3",
        "\n}",
    );

    require_all(
        wrapper,
        &[
            "PopulationSession",
            "ResidentFeatureStoreImportV3",
            "ResidentFeatureStoreConsumerLeaseV3",
        ],
    );
    require_all(implementation, &["record_consumer_completion"]);
    require_all(lease_implementation, &["completion_is_ready"]);
    require_all(
        drop_impl,
        &["arm_resident_session_leak_only_v3", "population_session"],
    );
    assert!(
        gpu_cuda.contains("#[must_use") || gpu_cuda.contains("#[must_use ="),
        "the resident population owner must be must_use"
    );
    require_none(
        &gpu_cuda,
        &[
            "impl Clone for ResidentPopulationSessionV3",
            "impl Default for ResidentPopulationSessionV3",
        ],
    );
}

#[test]
fn native_binding_borrows_parent_and_stream_and_waits_for_the_ready_event() {
    let header = read_gpu_cuda_or_empty("native/neoethos_gpu_cuda.h");
    let native = read_gpu_cuda_or_empty("native/prototype_b_population.cu");
    let stub = read_gpu_cuda_or_empty("native/stub.cpp");
    let symbol = "neoethos_gpu_cuda_population_bind_resident_feature_store_v3";

    for source in [&header, &native, &stub] {
        assert!(source.contains(symbol), "native V3 bind omits {symbol}");
    }
    require_all(
        &native,
        &[
            "cudaStreamWaitEvent(session->stream",
            "parent_ownership",
            "NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3",
            "stream_ownership",
            "STREAM_BORROWED",
        ],
    );
    let bind = section(&native, symbol, "\n}");
    require_none(
        bind,
        &[
            "cudaMalloc",
            "cudaMemcpyHostToDevice",
            "indicators_feature_major",
            "cudaStreamCreate",
            "cudaSetDevice",
        ],
    );
    let release = section(&native, "void release() {", "\n  }");
    require_all(
        release,
        &[
            "parent_ownership",
            "stream_ownership",
            "NEO_POPULATION_PARENT_OWNED_V1",
            "STREAM_OWNED",
        ],
    );
    assert!(
        !release.contains("cudaStreamSynchronize"),
        "resident-session release must never insert an implicit host synchronization"
    );
}

#[test]
fn standalone_search_gpu_cuda_feature_closes_data_v3_and_exports_the_additive_module() {
    let manifest = read_or_empty("Cargo.toml");
    let library = read_or_empty("src/lib.rs");

    assert!(
        manifest.contains("neoethos-data/gpu-cuda"),
        "Search gpu-cuda must enable the Data sealed V3 authority in standalone builds"
    );
    assert!(
        library.contains("strict_resident_feature_store_v3"),
        "Search does not export the additive strict resident V3 boundary"
    );
}
