use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
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

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "missing resident V3 token {token:?}"
        );
    }
}

fn require_none(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "resident V3 boundary contains forbidden token {token:?}"
        );
    }
}

#[test]
fn import_is_consumed_exactly_once_into_an_opaque_population_owner() {
    let resident = read("src/resident_feature_store_v3.rs");
    let consume = section(
        &resident,
        "pub fn consume_into_population_session_v3(",
        "\n    }",
    );
    require_all(
        consume,
        &[
            "self",
            "Result<ResidentPopulationSessionV3",
            "bind_resident_feature_store_v3",
            "admitted_run_stream",
        ],
    );
    assert!(
        !consume.contains("&self"),
        "one import must not mint multiple native population sessions"
    );
    require_none(
        consume,
        &[
            "PopulationSession::create(",
            "upload_dataset",
            "upload_parent_dataset_v1",
            "Context::new(",
            "Stream::new(",
            "Device::get_device(",
        ],
    );
}

#[test]
fn resident_bind_marks_the_borrowed_dataset_ready_for_gene_upload() {
    let population = read("src/population.rs");
    let rust_bind = section(&population, "fn bind_resident_feature_store_v3(", "\n    }");
    require_all(
        rust_bind,
        &[
            "resident_parent_shape_v3: Some((rows, feature_count))",
            "dataset_uploaded: true",
        ],
    );
    require_none(
        rust_bind,
        &[
            "upload_dataset(",
            "upload_parent_dataset_v1(",
            "dataset_uploaded: false",
        ],
    );
}

#[test]
fn consume_revalidates_the_exact_admitted_context_stream_device_and_shape() {
    let resident = read("src/resident_feature_store_v3.rs");
    let consume = section(
        &resident,
        "pub fn consume_into_population_session_v3(",
        "\n    }",
    );
    require_all(
        consume,
        &[
            "admission_identity_sha256",
            "device_identity",
            "device_ordinal",
            "consumer_context",
            "consumer_stream",
            "producer_stream",
            "parent_dataset_layout",
            "canonical_content_merkle",
            "rows",
            "columns",
            "SMC_SLOTS_V3",
        ],
    );
    require_all(
        &resident,
        &[
            "PrimaryContextMismatch",
            "ProducerStreamMismatch",
            "DeviceMismatch",
        ],
    );
}

#[test]
fn raw_device_handles_never_escape_the_gpu_cuda_owned_consume_boundary() {
    let resident = read("src/resident_feature_store_v3.rs");
    require_none(
        &resident,
        &[
            "pub fn raw_device_pointer",
            "pub fn raw_context",
            "pub fn raw_stream",
            "pub fn ready_event_raw",
            "impl Clone for ResidentFeatureStoreImportV3",
            "impl Clone for ResidentPopulationSessionV3",
        ],
    );
    require_all(
        &resident,
        &[
            "pub(crate) fn primary_context_for_resident_producer_v3",
            "pub(crate) fn run_stream_for_resident_producer_v3",
        ],
    );
}

#[test]
fn retained_run_stream_drops_before_its_primary_context() {
    let resident = read("src/resident_feature_store_v3.rs");
    for declaration in [
        section(&resident, "pub struct GpuOnlyRunDeviceAdmissionV3 {", "\n}"),
        section(
            &resident,
            "pub(crate) struct FullDiscoveryRunDeviceAdmissionRequestV3 {",
            "\n}",
        ),
    ] {
        let stream_at = declaration
            .find("run_stream: Arc<Stream>")
            .expect("retained run stream field");
        let context_at = declaration
            .find("primary_context: Arc<Context>")
            .expect("retained primary context field");
        assert!(
            stream_at < context_at,
            "Rust field drop order must destroy the run stream before its primary context"
        );
    }
}

#[test]
fn rust_ffi_descriptor_binds_bar_major_values_validity_parent_and_event() {
    let population = read("src/population.rs");
    let descriptor = section(&population, "struct RawResidentFeatureStoreBindV3 {", "\n}");
    require_all(
        descriptor,
        &[
            "abi_version:",
            "selected_device_ordinal:",
            "row_count:",
            "feature_count:",
            "smc_slots:",
            "packed_validity_bytes:",
            "close:",
            "high:",
            "low:",
            "indicators_bar_major:",
            "indicators_validity_u4:",
            "months:",
            "days:",
            "timestamps:",
            "smc_rows:",
            "admitted_primary_context:",
            "admitted_run_stream:",
            "ready_event:",
            "device_uuid:",
            "admission_identity_sha256:",
            "canonical_content_merkle:",
        ],
    );
}

#[test]
fn native_calendar_and_timestamp_storage_uses_fixed_width_int64_t() {
    let native = read("native/prototype_b_population.cu");
    let header = read("native/neoethos_gpu_cuda.h");
    let layout = read("native/layout_asserts.cpp");
    let dataset = section(&native, "struct DeviceDataset {", "\n};");
    let session = section(&native, "struct NeoCudaPopulationSession {", "\n};");
    let resident_bind = section(
        &native,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    for source in [dataset, session] {
        require_all(
            source,
            &[
                "std::int64_t* months",
                "std::int64_t* days",
                "std::int64_t* timestamps",
            ],
        );
        require_none(
            source,
            &[
                "long long* months",
                "long long* days",
                "long long* timestamps",
            ],
        );
    }
    require_all(
        resident_bind,
        &[
            "const_cast<std::int64_t*>(resident->months)",
            "const_cast<std::int64_t*>(resident->days)",
            "const_cast<std::int64_t*>(resident->timestamps)",
        ],
    );
    require_none(resident_bind, &["const_cast<long long*>"]);
    require_all(&native, &["sizeof(std::int64_t)"]);
    require_all(
        &header,
        &[
            "const std::int64_t* months",
            "const std::int64_t* days",
            "const std::int64_t* timestamps",
        ],
    );
    for member in ["months", "days", "timestamps"] {
        assert!(
            layout.contains(&format!(
                "decltype(NeoPopulationResidentFeatureStoreV3::{member})"
            )),
            "native layout omits the fixed-width {member} pointer assertion"
        );
    }
    assert_eq!(
        layout.matches("const std::int64_t*>").count(),
        3,
        "resident V3 calendar/timestamp layout must assert three exact int64_t pointer types"
    );
}

#[test]
fn native_bind_uses_the_borrowed_run_stream_and_queues_the_ready_wait() {
    let native = read("native/prototype_b_population.cu");
    let bind = section(
        &native,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    require_all(
        bind,
        &[
            "session->stream = resident->admitted_run_stream",
            "cudaStreamWaitEvent(session->stream, resident->ready_event, 0)",
            "NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3",
            "NEO_POPULATION_STREAM_BORROWED",
            "indicators_bar_major",
            "indicators_validity_u4",
        ],
    );
    require_none(
        bind,
        &[
            "cudaStreamCreate",
            "cudaSetDevice",
            "cudaMemcpyHostToDevice",
            "transpose",
            "indicators_feature_major",
            "upload_dataset",
        ],
    );
}

#[test]
fn native_bind_refuses_identity_shape_and_pointer_drift_before_any_wait() {
    let native = read("native/prototype_b_population.cu");
    let bind = section(
        &native,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    require_all(
        bind,
        &[
            "NEOETHOS_GPU_ABI_VERSION",
            "cudaGetDevice(&current_device)",
            "cudaGetDeviceProperties",
            "resident->selected_device_ordinal",
            "resident->device_uuid",
            "resident->row_count",
            "resident->feature_count",
            "resident->smc_slots",
            "resident->packed_validity_bytes",
            "hash_is_nonzero_v3",
        ],
    );
    let validate_at = bind
        .find("cudaGetDevice(&current_device)")
        .expect("native device validation");
    let wait_at = bind
        .find("cudaStreamWaitEvent(session->stream, resident->ready_event, 0)")
        .expect("native ready-event wait");
    assert!(
        validate_at < wait_at,
        "native reads may not precede validation"
    );
}

#[test]
fn release_frees_only_owned_parent_and_stream_storage() {
    let native = read("native/prototype_b_population.cu");
    let release = section(&native, "  void release() {", "\n  }");
    require_all(
        release,
        &[
            "parent_ownership == NEO_POPULATION_PARENT_OWNED_V1",
            "stream_ownership == NEO_POPULATION_STREAM_OWNED",
            "indicators_validity_u4 = nullptr",
            "stream = nullptr",
        ],
    );
    assert!(
        !release.contains("cudaStreamSynchronize"),
        "resident teardown must never insert a host synchronization"
    );
}

#[test]
fn v1_upload_and_v3_borrowed_parent_ownership_are_distinct() {
    let native = read("native/prototype_b_population.cu");
    let upload = section(
        &native,
        "neoethos_gpu_cuda_population_upload_parent_v1(",
        "\n}",
    );
    let bind = section(
        &native,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    require_all(upload, &["NEO_POPULATION_PARENT_OWNED_V1"]);
    require_all(bind, &["NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3"]);
    assert!(
        upload.contains("copy_to_device") && !bind.contains("copy_to_device"),
        "the V3 bind must not masquerade a V1 parent upload as residency"
    );
}

#[test]
fn borrowed_resident_session_refuses_every_legacy_parent_upload_entrypoint() {
    let native = read("native/prototype_b_population.cu");
    for symbol in [
        "neoethos_gpu_cuda_population_upload_dataset(",
        "neoethos_gpu_cuda_population_upload_parent_v1(",
    ] {
        let upload = section(&native, symbol, "\n}");
        require_all(
            upload,
            &[
                "session->parent_ownership == NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3",
                "NEO_POPULATION_STATUS_INVALID_ARGUMENT",
            ],
        );
    }
}

#[test]
fn header_stub_and_layout_pin_the_additive_abi_without_fake_success() {
    let header = read("native/neoethos_gpu_cuda.h");
    let stub = read("native/stub.cpp");
    let layout = read("native/layout_asserts.cpp");
    for source in [&header, &stub, &layout] {
        require_all(
            source,
            &["neoethos_gpu_cuda_population_bind_resident_feature_store_v3"],
        );
    }
    require_all(
        &header,
        &[
            "struct NeoPopulationResidentFeatureStoreV3",
            "NEO_POPULATION_PARENT_OWNED_V1",
            "NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3",
            "NEO_POPULATION_STREAM_OWNED",
            "NEO_POPULATION_STREAM_BORROWED",
        ],
    );
    let stub_bind = section(
        &stub,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    require_all(
        stub_bind,
        &["NEO_POPULATION_STATUS_UNSUPPORTED", "return nullptr"],
    );
}

#[test]
fn completion_lease_owns_native_session_and_store_lifetime_until_event_ready() {
    let resident = read("src/resident_feature_store_v3.rs");
    let wrapper = section(
        &resident,
        "pub struct ResidentPopulationSessionV3 {",
        "impl ResidentPopulationSessionV3",
    );
    let lease_lifetime = section(&resident, "struct ResidentConsumerLifetimeV3 {", "\n}");
    require_all(
        wrapper,
        &[
            "population_session: PopulationSession",
            "resident_import: Option<ResidentFeatureStoreImportV3>",
            "consumer_lease: Option<ResidentFeatureStoreConsumerLeaseV3>",
        ],
    );
    require_all(
        lease_lifetime,
        &["population_session: Option<PopulationSession>"],
    );
    require_all(
        &resident,
        &[
            "record_consumer_completion",
            "attach_population_session_v3",
            "completion_is_ready",
            "authorize_resident_session_destroy_v3",
        ],
    );
}

#[test]
fn ambiguous_drop_leaks_both_native_session_and_import_instead_of_freeing_live_data() {
    let resident = read("src/resident_feature_store_v3.rs");
    let wrapper_drop = section(
        &resident,
        "impl Drop for ResidentPopulationSessionV3",
        "\n}",
    );
    let lease_drop = section(
        &resident,
        "impl Drop for ResidentFeatureStoreConsumerLeaseV3",
        "\n}",
    );
    require_all(
        wrapper_drop,
        &[
            "arm_resident_session_leak_only_v3",
            "resident_import.is_some()",
        ],
    );
    require_all(
        lease_drop,
        &[
            "consumer_completion_event.query()",
            "leak_owner_if_consumer_is_still_running",
        ],
    );
    require_none(
        &format!("{wrapper_drop}\n{lease_drop}"),
        &["synchronize", "unwrap_or(true)", "unwrap_or(false)"],
    );
}

#[test]
fn resident_bind_has_no_cpu_f32_reprobe_or_fallback_route() {
    let resident = read("src/resident_feature_store_v3.rs");
    let population = read("src/population.rs");
    let native = read("native/prototype_b_population.cu");
    let rust_bind = section(&population, "fn bind_resident_feature_store_v3(", "\n    }");
    let consume = section(
        &resident,
        "pub fn consume_into_population_session_v3(",
        "\n    }",
    );
    let native_bind = section(
        &native,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
        "\n}",
    );
    let combined = format!("{rust_bind}\n{consume}\n{native_bind}");
    require_none(
        &combined,
        &[
            "f32",
            "Cpu",
            "fallback",
            "reprobe",
            "probe_cuda",
            "device_count",
            "read_metrics",
            "copy_to_host",
        ],
    );
}
