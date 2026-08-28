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

fn read_gpu_cuda(relative: &str) -> String {
    let path = manifest_dir().join("../neoethos-gpu-cuda").join(relative);
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

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "strict resident Search bind is missing {token:?}"
        );
    }
}

#[test]
fn strict_gpu_run_consumes_the_opaque_v3_import_through_gpu_cuda_owned_binding() {
    let bind = read("src/strict_resident_feature_store_v3.rs");
    let function = section(
        &bind,
        "pub(crate) fn bind_resident_feature_store_v3(",
        "\n}",
    );

    require_all(
        function,
        &[
            "ResidentFeatureStoreImportV3",
            "StrictResidentPopulationExecutionRunV3",
            ".device_ordinal()",
            ".admission_identity_sha256()",
            ".consume_into_population_session_v3(",
        ],
    );
    assert!(
        !function.contains("&ResidentFeatureStoreImportV3"),
        "Search must consume the opaque import so it cannot outlive two consumer sessions"
    );
}

#[test]
fn strict_bind_has_no_raw_pointer_context_or_caller_mintable_authority_escape() {
    let bind = read("src/strict_resident_feature_store_v3.rs");
    for forbidden in [
        "as_device_ptr",
        "as_raw()",
        "CUcontext",
        "CUstream",
        "unsafe",
        "pub struct HardwareProbe",
        "card_present: bool",
        "requested_ordinal:",
        "device_override",
        "unwrap_or(0)",
        "std::env",
        "acquire_strict_discovery_device_admission_v1",
        "require_exact_cuda_device_ordinal_v1",
    ] {
        assert!(
            !bind.contains(forbidden),
            "strict resident bind bypasses opaque authority through {forbidden:?}"
        );
    }
}

#[test]
fn sealed_data_store_moves_through_the_retained_admitted_stream_without_handle_injection() {
    let data = fs::read_to_string(
        manifest_dir().join("../neoethos-data/src/core/gpu_resident_feature_store_v3.rs"),
    )
    .expect("read Data sealed resident store source");
    let resident = read_gpu_cuda("src/resident_feature_store_v3.rs");
    let owner_import = section(&resident, "pub fn import_on_admitted_run_stream_v3(", "\n}");
    let sealed_import = section(
        &data,
        "pub fn into_resident_feature_store_import_v3(",
        "\n}",
    );

    require_all(
        owner_import,
        &[
            "self: &Arc<Self>",
            "primary_context_for_resident_producer_v3",
            "run_stream_for_resident_producer_v3",
            "import_on_consumer_stream",
        ],
    );
    require_all(
        sealed_import,
        &[
            "self",
            "validate_resident_feature_store_import_v3",
            ".import_on_admitted_run_stream_v3()",
            "ResidentFeatureStoreImportV3",
        ],
    );
    for forbidden in [
        "Context::new(",
        "Stream::new(",
        "Device::get_device(",
        "require_exact_cuda_device_ordinal_v1",
        "as_raw()",
        "std::env",
    ] {
        assert!(
            !sealed_import.contains(forbidden),
            "sealed Data import bypasses its retained admitted route through {forbidden:?}"
        );
    }
}

#[test]
fn strict_v3_path_cannot_reach_legacy_host_gather_or_dataset_upload() {
    let bind = read("src/strict_resident_feature_store_v3.rs");
    for forbidden in [
        ".upload_dataset(",
        "neoethos_gpu_cuda_population_upload_dataset",
        "PopulationDataset::new",
        "PopulationDatasetView",
        "PopulationParentDatasetV1",
        "upload_parent_dataset_v1",
        "indicators_feature_major",
        "to_dense_samples_major",
        "feature_column(",
        "to_vec()",
    ] {
        assert!(
            !bind.contains(forbidden),
            "strict resident V3 path still reaches host gather/upload via {forbidden:?}"
        );
    }
}

#[test]
fn run_owns_the_bound_session_until_exact_consumer_completion_is_recorded() {
    let bind = read("src/strict_resident_feature_store_v3.rs");
    let evidence = read("src/population_execution_evidence_v1.rs");
    let run = section(
        &bind,
        "pub(crate) struct StrictResidentPopulationExecutionRunV3 {",
        "\n}",
    );

    require_all(
        &bind,
        &[
            "ResidentPopulationSessionV3",
            "record_consumer_completion",
            "ResidentFeatureStoreConsumerLeaseV3",
        ],
    );
    require_all(
        run,
        &[
            "session: Option<ResidentPopulationSessionV3>",
            "row_count:",
            "column_count:",
        ],
    );
    assert!(
        !bind.contains("impl Default for StrictResidentPopulationExecutionRunV3"),
        "a default native run could detach residency from the sealed route"
    );
    assert!(
        !evidence.contains("resident_feature_store_session_v3")
            && !evidence.contains("parent_row_count")
            && !evidence.contains("parent_feature_count"),
        "the host V1 execution run must not retain native V3 residency state"
    );
}

#[test]
fn strict_bind_checks_exact_scope_shape_and_route_identity_before_native_reads() {
    let bind = read("src/strict_resident_feature_store_v3.rs");
    let function = section(
        &bind,
        "pub(crate) fn bind_resident_feature_store_v3(",
        "\n}",
    );
    require_all(
        function,
        &[
            "scope.evaluated_window()",
            "resident_import.rows()",
            "resident_import.columns()",
            "scope_rows",
            "resident_columns",
            "selected_ordinal",
        ],
    );
    let validate_at = function
        .find("resident_import.rows()")
        .expect("exact resident row validation");
    let consume_at = function
        .find(".consume_into_population_session_v3(")
        .expect("gpu-cuda-owned import consumption");
    assert!(
        validate_at < consume_at,
        "Search must refuse scope/shape drift before native import consumption"
    );
}

#[test]
fn gpu_cuda_owned_wrapper_retains_population_session_import_and_completion_lease() {
    let resident = read_gpu_cuda("src/resident_feature_store_v3.rs");
    let wrapper = section(&resident, "pub struct ResidentPopulationSessionV3 {", "\n}");
    let consume = section(
        &resident,
        "pub fn consume_into_population_session_v3(",
        "\n}",
    );

    require_all(
        wrapper,
        &[
            "population_session: PopulationSession",
            "resident_import: Option<ResidentFeatureStoreImportV3>",
            "consumer_lease: Option<ResidentFeatureStoreConsumerLeaseV3>",
        ],
    );
    require_all(
        consume,
        &[
            "self",
            "Result<ResidentPopulationSessionV3",
            "bind_resident_feature_store_v3",
        ],
    );
    assert!(
        !consume.contains("&self"),
        "gpu-cuda must consume the import rather than mint detachable sessions"
    );
}

#[test]
fn native_bind_waits_on_the_actual_population_stream_and_never_frees_imported_parent() {
    let header = read_gpu_cuda("native/neoethos_gpu_cuda.h");
    let native = read_gpu_cuda("native/prototype_b_population.cu");
    let stub = read_gpu_cuda("native/stub.cpp");

    require_all(
        &header,
        &["neoethos_gpu_cuda_population_bind_resident_feature_store_v3"],
    );
    require_all(
        &stub,
        &["neoethos_gpu_cuda_population_bind_resident_feature_store_v3"],
    );
    require_all(
        &native,
        &[
            "neoethos_gpu_cuda_population_bind_resident_feature_store_v3",
            "cudaStreamWaitEvent(session->stream",
            "parent_ownership",
            "NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3",
            "NEO_POPULATION_PARENT_OWNED_V1",
        ],
    );
    let release = section(&native, "void release() {", "\n  }");
    require_all(
        release,
        &["parent_ownership", "NEO_POPULATION_PARENT_OWNED_V1"],
    );
    assert!(
        !release.contains("cudaStreamSynchronize"),
        "borrowed parent release must remain stream ordered without host synchronization"
    );
}
