use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read_required(relative: &str) -> String {
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
            "resident generation post-GA A1 bridge is missing {token:?}"
        );
    }
}

#[test]
fn sealed_generation_owner_moves_in_place_into_one_non_mintable_post_ga_typestate() {
    let rust = read_required("src/resident_generation_v1.rs");
    let bridge = section(
        &rust,
        "pub(crate) struct ResidentGenerationPostGaInPlaceRunV1 {",
        "\n}",
    );
    require_all(
        bridge,
        &[
            "run: Option<ResidentGenerationDeviceRunV1>",
            "dependency: RawReadyEventV1",
            "content_authority: ResidentGenerationPostGaContentAuthorityV1",
            "receipt: RawPostGaInPlaceReceiptV1",
            "workspace_state: ResidentGenerationPostGaWorkspaceStateV1",
        ],
    );
    assert!(
        !bridge.contains("pub "),
        "post-GA in-place typestate fields must stay private"
    );

    let begin = section(
        &rust,
        "pub(crate) fn begin_resident_post_ga_in_place_v1(",
        "\n}\n\n",
    );
    require_all(
        begin,
        &[
            "input: ResidentGenerationPostGaInputV1",
            "let ResidentGenerationPostGaInputV1 {",
            "ready.into_parts_v1()?",
            "require_run_state_v1(&run, ResidentGenerationRunStateV1::Sealed)?",
            "validate_post_ga_content_identity_v1(",
            "run.state = ResidentGenerationRunStateV1::Poisoned",
            "ffi_begin_resident_post_ga_in_place_v1(",
            "validate_post_ga_in_place_receipt_v1(",
            "run.state = ResidentGenerationRunStateV1::PostGaInPlace",
            "ResidentGenerationPostGaWorkspaceStateV1::GenerationStoresOnly",
        ],
    );
    assert!(
        begin.find("run.state = ResidentGenerationRunStateV1::Poisoned")
            < begin.find("ffi_begin_resident_post_ga_in_place_v1("),
        "ambiguous native launch must poison the owner before the FFI call"
    );
    assert!(
        begin.find("ffi_begin_resident_post_ga_in_place_v1(")
            < begin.find("run.state = ResidentGenerationRunStateV1::PostGaInPlace"),
        "PostGaInPlace may be published only after native receipt validation"
    );
    require_all(
        &rust,
        &[
            "#[must_use = \"post-GA in-place work owns the generation run until a resident stage consumes it\"]",
            "PostGaInPlace",
            "ResidentGenerationPostGaWorkspaceStateV1",
            "GenerationStoresOnly",
        ],
    );
    for forbidden in [
        "impl Clone for ResidentGenerationPostGaInPlaceRunV1",
        "impl Copy for ResidentGenerationPostGaInPlaceRunV1",
        "impl Default for ResidentGenerationPostGaInPlaceRunV1",
        "Deserialize",
        "pub fn from_raw",
        "pub fn raw_",
        "pub fn device_ptr",
    ] {
        assert!(
            !rust.contains(forbidden),
            "post-GA owner can be minted or exposes raw authority via {forbidden:?}"
        );
    }
}

#[test]
fn private_abi_returns_only_same_run_identity_and_zero_additional_allocation_receipt() {
    let rust = read_required("src/resident_generation_v1.rs");
    let abi = read_required("native/resident_generation_v1_abi.cuh");

    require_all(
        &abi,
        &[
            "struct NeoResidentGenerationPostGaInPlaceReceiptV1",
            "std::uint64_t ready_event_id;",
            "std::uint64_t current_generation_index;",
            "std::uint64_t same_stream_enqueue_count;",
            "std::uint64_t logical_population_count;",
            "std::uint64_t retained_evaluation_capacity;",
            "std::uint64_t generation_allocation_total_device_bytes;",
            "std::uint64_t additional_allocation_count;",
            "std::uint64_t additional_device_bytes;",
            "std::uint64_t gene_content_identity_handle;",
            "std::uint64_t metric_content_identity_handle;",
            "std::uint64_t generation_receipt_identity_handle;",
            "begin_resident_post_ga_in_place_v1(",
            "NeoResidentGenerationRunV1* run",
            "const NeoResidentGenerationReadyEventV1* dependency",
            "NeoResidentGenerationPostGaInPlaceReceiptV1* receipt",
        ],
    );
    for forbidden in [
        "NeoResidentGenerationRunV1** post_ga_run",
        "cudaStream_t post_ga_stream;",
        "cudaEvent_t post_ga_event;",
        "void* gene_scalars_device;",
        "void* metric_rows_device;",
        "void* decision_keys_device;",
    ] {
        assert!(
            !abi.contains(forbidden),
            "private post-GA receipt exposes or duplicates native authority via {forbidden:?}"
        );
    }

    require_all(
        &rust,
        &[
            "struct RawPostGaInPlaceReceiptV1",
            "const _: [(); 96] = [(); std::mem::size_of::<RawPostGaInPlaceReceiptV1>()];",
            "ffi_begin_resident_post_ga_in_place_v1",
            "additional_allocation_count == 0",
            "additional_device_bytes == 0",
            "let exact_generation_allocation =",
            "receipt.generation_allocation_total_device_bytes",
            "run.allocation.total_device_bytes()",
            "|| !exact_generation_allocation",
        ],
    );
}

#[test]
fn native_bridge_reuses_the_sealed_generation_run_event_stream_and_allocation_in_place() {
    let cuda = read_required("native/resident_generation_v1.cu");
    let begin = section(
        &cuda,
        "extern \"C\" std::int32_t begin_resident_post_ga_in_place_v1(",
        "\n}\n\n",
    );
    require_all(
        begin,
        &[
            "run->sealed",
            "!run->post_ga_in_place_bound",
            "dependency->event_id == run->next_event_id",
            "dependency->generation_index == run->current_generation_index",
            "dependency->same_stream_enqueue_count == run->same_stream_enqueue_count",
            "consume_resident_generation_event_dependency_v1(run)",
            "run->post_ga_in_place_bound = true",
            "receipt->ready_event_id = dependency->event_id",
            "receipt->current_generation_index = run->current_generation_index",
            "receipt->same_stream_enqueue_count = run->same_stream_enqueue_count",
            "receipt->logical_population_count = run->logical_population_count",
            "receipt->retained_evaluation_capacity = run->retained_evaluation_capacity",
            "receipt->generation_allocation_total_device_bytes",
            "run->allocation.total_device_bytes;",
            "receipt->additional_allocation_count = 0",
            "receipt->additional_device_bytes = 0",
        ],
    );
    require_all(
        &cuda,
        &[
            "run->gene_scalars_device != nullptr",
            "run->gene_indices_device != nullptr",
            "run->gene_weights_device != nullptr",
            "run->metric_rows_device != nullptr",
            "run->resident_decision_keys_device != nullptr",
            "generation_content_identity_handle_v1(run->run_token, 1)",
            "generation_content_identity_handle_v1(run->run_token, 2)",
            "generation_content_identity_handle_v1(run->run_token, 3)",
        ],
    );
    for forbidden in [
        "new ",
        "delete ",
        "cudaMalloc",
        "cudaFree",
        "cudaEventCreate",
        "cudaEventDestroy",
        "cudaStreamCreate",
        "cudaSetDevice",
        "cudaMemcpy",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ] {
        assert!(
            !begin.contains(forbidden),
            "in-place native bridge creates, transfers, synchronizes, or frees via {forbidden:?}"
        );
    }
}

#[test]
fn bridge_retains_all_lifetimes_and_stops_at_the_charged_workspace_frontier() {
    let rust = read_required("src/resident_generation_v1.rs");
    let bridge = section(
        &rust,
        "pub(crate) struct ResidentGenerationPostGaInPlaceRunV1 {",
        "\n}",
    );
    require_all(
        &rust,
        &[
            "population_session_import: Option<ResidentGenerationPopulationSessionImportV1>",
            "dependency_lifetime_owners: Vec<Box<dyn Any>>",
            "impl Drop for ResidentGenerationDeviceRunV1",
            "leak_live_native_generation_run_v1(self)",
            "ResidentGenerationPostGaWorkspaceStateV1::GenerationStoresOnly",
        ],
    );
    for forbidden in [
        "pub fn native_run",
        "pub fn admitted_stream",
        "pub fn ready_event",
        "pub fn primary_context",
        "pub fn gene_store",
        "pub fn metric_store",
        "pub fn decision_key_store",
    ] {
        assert!(
            !bridge.contains(forbidden),
            "bridge exposes lifetime-bound resident authority via {forbidden:?}"
        );
    }
    for forbidden in [
        "execute_gpu_post_ga_pipeline_v1(",
        "SealedResidentPostGaOutcomeV1",
        "post_ga_kernel_completion_receipt",
        "post_ga_workspace_state: Charged",
    ] {
        assert!(
            !rust.contains(forbidden),
            "A1 bridge falsely claims work beyond the charged workspace frontier via {forbidden:?}"
        );
    }
}
