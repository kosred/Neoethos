use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read(path: impl AsRef<Path>) -> String {
    let path = workspace_root().join(path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn before(haystack: &str, left: &str, right: &str) {
    let left_index = haystack
        .find(left)
        .unwrap_or_else(|| panic!("missing left token `{left}`"));
    let right_index = haystack
        .find(right)
        .unwrap_or_else(|| panic!("missing right token `{right}`"));
    assert!(left_index < right_index, "`{left}` must precede `{right}`");
}

#[test]
fn strict_phase_one_is_opaque_and_refuses_incomplete_producers_before_device_work() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    assert!(source.contains("pub struct GpuOnlyFeatureMaterializationAdmissionV3"));
    assert!(!source.contains("pub fn new("));
    assert!(!source.contains("pub fn seal("));
    assert!(!source.contains("supports_gpu"));
    assert!(!source.contains("GpuOnlyResidentAdmissionV3) ->"));
    assert!(source.contains("ResidentFeatureProducerV3::ALL"));
    assert!(source.contains("MissingProducerCapabilities"));
    for required in [
        "ClassicTa",
        "Smc",
        "Quant",
        "Session",
        "Regime",
        "Footprint",
        "HigherTimeframeAlignment",
        "RobustNormalization",
        "CanonicalContentSha256",
        "FeatureMajorToBarMajor",
    ] {
        assert!(
            source.contains(required),
            "phase-one capability authority omitted {required}"
        );
    }
    before(
        &source,
        "require_complete_resident_producer_manifest_v3",
        "bind_gpu_only_run_device_v3",
    );
    assert!(source.contains("preflight_gpu_only_feature_recipe_v3"));
    assert!(source.contains("GpuOnlyRunDeviceAdmissionV3"));
    assert!(!source.contains("Context::new"));
    assert!(!source.contains("Device::get_device"));
    assert!(!source.contains("probe_selected_cuda_device_v3"));
}

#[test]
fn selected_device_build_requires_real_cc_sass_and_manifest_provenance() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let full_plan = read("crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs");
    let run_admission = read("crates/neoethos-gpu-cuda/src/run_device_admission_v1.rs");
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let vector_ta_loader = read("vendor/vector-ta-0.2.9-patched/src/cuda/module_loader.rs");
    for token in [
        "pub struct GpuOnlyRunDeviceAdmissionV3",
        "admission_identity_sha256",
        "seal_gpu_only_run_device_admission_v3",
    ] {
        assert!(
            source.contains(token),
            "missing device/build proof `{token}`"
        );
    }
    assert!(contracts.contains("let expected_sass_target = format!("));
    assert!(contracts.contains("if native_sass_target != expected_sass_target"));
    assert!(contracts.contains("NativeSassTargetMismatch"));
    assert!(vector_ta_loader.contains("pub const COMPILED_ARCHS"));
    assert!(run_admission.contains("cuda_build_manifest_v1()"));
    assert!(run_admission.contains("sass_targets"));
    assert!(run_admission.contains("ptx_targets"));
    assert!(run_admission.contains("CUDA build must contain exact SASS and no PTX fallback"));
    assert!(run_admission.contains("let sass_target = format!(\"sm_{major}{minor}\")"));
    assert!(full_plan.contains("into_gpu_only_run_device_admission_v3"));
    assert!(full_plan.contains("seal_gpu_only_run_device_admission_v3("));
    assert!(run_admission.contains("cuDriverGetVersion="));
    assert!(run_admission.contains("cuCtxGetApiVersion="));
    assert!(!source.contains("pub fn acquire_gpu_only_run_device_admission_v3"));
    assert!(!source.contains("unwrap_or(\"sm_"));
    assert!(!source.contains("unwrap_or_default().contains"));
}

#[test]
fn opaque_data_authority_derives_working_set_and_batch_peak_from_current_runtime() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let full_plan = read("crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs");
    for token in [
        "derive_exact_resident_working_set_v3",
        "ResolvedResidentProducerBatchMemoryV3",
        "batch.column_count > 64",
        "value_and_logical_validity_bytes",
        "additional_retained_bytes",
        "exact_retained_bytes",
        "max_live_producer_bytes",
        "max_live_producer_scratch_bytes",
        "pointer_and_schema_metadata_bytes",
        "run_device.allocator_context_reserve_bytes()",
        "phase_one_free_bytes_snapshot",
        "EXACT_ALLOCATOR_RESERVE_POLICY_V3",
    ] {
        assert!(
            source.contains(token),
            "missing derived runtime authority `{token}`"
        );
    }
    let bind_body = source
        .split_once("pub(crate) fn bind_gpu_only_run_device_v3(")
        .expect("Data V3 run-device bind")
        .1
        .split_once("pub(crate) fn seal_gpu_resident_feature_store_v3(")
        .expect("Data V3 bind boundary")
        .0;
    before(
        bind_body,
        "let GpuOnlyFeatureRecipePreflightV3 {",
        "let working_set = derive_exact_resident_working_set_v3(",
    );
    assert!(runtime.contains("RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3"));
    assert!(runtime.contains("request.allocator_context_reserve_bytes"));
    assert!(full_plan.contains("allocator_context_reserve_bytes: preflight"));
    assert!(full_plan.contains("into_gpu_only_run_device_admission_v3"));
    assert!(runtime.contains("mem_get_info()"));
    assert!(!source.contains("mem_get_info()"));
    assert!(source.contains("run_device.admission_identity_sha256()"));
    assert!(!source.contains("plan.admission_identity_sha256"));
    assert!(!source.contains("pub working_set:"));
    assert!(!source.contains("pub allocator_context_reserve_bytes:"));
    assert!(!source.contains("pub device_free_bytes_snapshot:"));
}

#[test]
fn data_compile_frontiers_keep_one_capability_authority_and_typed_runtime_evidence() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    assert!(source.contains("producer_capabilities: ResidentProducerCapabilityManifestV3"));
    assert!(source.contains("current_resident_producer_capabilities_v3()?"));
    assert_eq!(
        source
            .matches("require_complete_resident_producer_manifest_v3(")
            .count(),
        3,
        "one definition, one crate-owned factory call and one exact fail-closed unit test are required"
    );
    assert!(!source.contains("std::mem::take(&mut plan.producer_capabilities)"));
    assert!(
        source.contains("retained_parent_dataset_bytes: evidence.retained_parent_dataset_bytes,")
    );
    assert!(!source.contains("retained_parent_dataset_bytes: checked_u64("));
}

#[test]
fn one_shot_run_device_and_final_event_authority_are_moved_never_caller_minted() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let smc_runtime = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    for token in [
        "GpuOnlyFeatureMaterializationSealTokenV3",
        "begin_materialization",
        "begin_resident_smc_store_v3(",
        "self.run_device",
        "owner.admission_identity_sha256()",
        "owner.device_identity()",
        "let ready_event = owner.ready_event_contract()?",
        "let parent_dataset = owner.parent_dataset_layout().clone()",
    ] {
        assert!(
            source.contains(token),
            "missing one-shot Data token `{token}`"
        );
    }
    assert!(smc_runtime.contains("ResidentFeatureStoreAssemblerV3::new("));
    for token in [
        "run_device: Option<GpuOnlyRunDeviceAdmissionV3>",
        "admission_identity_sha256",
        "run_stream_process_token",
        "ready_event_contract",
        "sealed store retains one-shot run-device admission",
    ] {
        assert!(
            runtime.contains(token),
            "missing one-shot runtime token `{token}`"
        );
    }
    assert!(!source.contains("ready_event: ResidentReadyEventV3"));
    assert!(!source.contains("ordered_feature_names: Vec<String>"));
    assert!(!source.contains("parent_dataset: ResidentParentDatasetLayoutV3"));
    assert!(!source.contains("plan.admission_identity_sha256"));
    assert!(!runtime.contains("pub fn primary_context(&self)"));
    assert!(!runtime.contains("pub fn run_stream(&self)"));
}

#[test]
fn assembler_appends_one_exact_monotonic_batch_and_retires_it_before_the_next() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    for token in [
        "pub struct ResidentFeatureStoreAssemblerV3",
        "pub unsafe trait ResidentF64FeatureBatchV3",
        "ResidentFeatureColumnBindingV3",
        "expected_column_bindings",
        "next_destination_column",
        "pending_batch",
        "append_batch",
        "try_retire_completed_batch",
        "enqueue_nonblocking_release",
        "max_live_producer_bytes",
        "producer_batch_count",
        "ResidentAppendTransactionV3::new(batch)",
        "transaction.ready_event().record",
        "pending.batch_ready_event.query",
        "host_pointer_tables",
        "pointer_tables",
    ] {
        assert!(
            runtime.contains(token),
            "missing incremental invariant `{token}`"
        );
    }
    assert!(!runtime.contains("Vec<Box<dyn ResidentF64FeatureBatchV3>>"));
    assert!(!runtime.contains("feature_sources:"));
    assert!(!runtime.contains("release_is_non_synchronizing"));
    before(
        &runtime,
        "append_batch",
        "ResidentAppendTransactionV3::new(batch)",
    );
    before(
        &runtime,
        "ResidentAppendTransactionV3::new(batch)",
        "if self.pending_batch.is_some()",
    );
    before(
        &runtime,
        "expected_column_bindings",
        "transaction.ready_event().record",
    );
    before(
        &runtime,
        "pending.batch_ready_event.query",
        "pending.release",
    );
    before(&runtime, "try_retire_completed_batch", "pub fn seal(");
    assert!(runtime.contains("return Ok(false);"));
    assert_eq!(
        runtime
            .matches("let (host_pointer_tables, pointer_tables) = compact_device_buffer_from_slice_async(")
            .count(),
        1,
        "each producer batch uses one retained pinned metadata transfer"
    );
    assert!(runtime.contains("self.next_destination_column == self.total_columns"));
    assert!(runtime.contains("self.pending_batch.is_none()"));
}

#[test]
fn resident_peak_is_one_final_store_plus_u4_and_max_live_batch_never_all_sources() {
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    for token in [
        "final_bar_major_value_bytes",
        "packed_validity_logical_bytes",
        "packed_validity_allocated_bytes",
        "parent_ohlcv_bytes",
        "parent_clock_bytes",
        "parent_smc_bytes",
        "parent_dataset_bytes",
        "active_view_indices_bytes",
        "lazy_view_indices_capacity_bytes",
        "max_live_producer_bytes",
        "pointer_and_schema_metadata_bytes",
        "merkle_scratch_bytes",
        "allocator_context_reserve_bytes",
        "device_free_bytes_snapshot",
        "pre_materialization_free_bytes_snapshot",
        "post_parent_free_bytes_snapshot",
        "retained_parent_dataset_bytes",
        "remaining_peak_after_parent_bytes",
        "full_feature_major_staging_bytes: 0",
    ] {
        assert!(
            contracts.contains(token),
            "missing exact VRAM receipt `{token}`"
        );
    }
    assert!(!contracts.contains("free / 10"));
    assert!(!contracts.contains("* 7 / 10"));
    assert!(!contracts.contains("permanent_view_indices_bytes"));
    assert!(contracts.contains("active_view_indices_bytes = 0"));
    assert!(contracts.contains("lazy_view_indices_capacity_bytes = 0"));
    assert_eq!(
        runtime
            .matches("StreamOrderedDeviceBufferV3::<f64>::uninitialized_async(")
            .count(),
        1,
        "strict production owns exactly one full final f64 allocation"
    );
    assert!(!runtime.contains("canonical_feature_major_values"));
    assert!(!runtime.contains("all_producer_sources"));
    assert!(runtime.contains("let merkle_scratch_level_bytes ="));
    assert!(
        runtime.contains("let merkle_scratch_bytes = merkle_scratch_level_bytes.checked_mul(2)")
    );
    assert_eq!(
        runtime
            .matches("merkle_scratch_level_bytes,\n            Arc::clone")
            .count(),
        2,
        "both Merkle scratch allocations must use the one-level extent"
    );
    assert!(
        runtime
            .contains("merkle_scratch_bytes,\n            pre_materialization_free_bytes_snapshot")
    );
    assert!(runtime.contains("StreamOrderedDeviceBufferV3::<u8>::uninitialized_async"));
    assert!(!runtime.contains("StreamOrderedDeviceBufferV3::<u8>::zeroed_async"));
    assert!(runtime.contains("validity_initialization_count: 1"));
    assert!(runtime.contains("packed_validity_allocated_bytes"));
    assert!(runtime.contains("mem_get_info()"));
    assert!(runtime.contains("runtime_pointer_and_schema_metadata_bytes_v3"));
    assert!(runtime.contains("working_set.reserve_policy_id()"));
    assert!(!runtime.contains("observed_free_bytes != working_set.device_free_bytes_snapshot()"));
    before(
        &runtime,
        "validate_parent_extents(parent_source, rows)?",
        "let (observed_free_bytes, _) = mem_get_info()?",
    );
    before(
        &runtime,
        "let (observed_free_bytes, _) = mem_get_info()?",
        "StreamOrderedDeviceBufferV3::<f64>::uninitialized_async(",
    );
    before(
        &runtime,
        "let (observed_free_bytes, _) = mem_get_info()?",
        "pub fn append_batch",
    );
}

#[test]
fn u4_pack_is_lossless_word_padded_zeroed_once_and_checks_every_producer_code() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_feature_store_v3.cu");
    for token in [
        "neoethos_resident_initialize_validity_u4_v3",
        "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3",
        "pack_sources_to_bar_major_f64_u4_v3",
        "pack_batch_boundary_validity_u4_v3",
        "source_validity_addresses",
        "validity_code_error",
        "code > 9U",
        "atomicOr",
        "allocated_bytes % sizeof(unsigned int)",
        "rows > std::numeric_limits<std::size_t>::max() /",
    ] {
        assert!(
            native.contains(token),
            "missing exact u4 boundary `{token}`"
        );
    }
    assert_eq!(
        runtime
            .matches("neoethos_resident_initialize_validity_u4_v3(")
            .count(),
        2,
        "one declaration plus one initialization call is required"
    );
    assert!(runtime.contains("let mut validity_code_error = [0_u32; 1]"));
    assert!(runtime.contains(".copy_to(&mut validity_code_error)"));
    assert!(runtime.contains("InvalidProducerValidityCode"));
    assert!(!native.contains("canonical_nan"));
    assert!(!native.contains("isfinite("));
    assert!(!native.contains("isnan("));
}

#[test]
fn exact_v3_merkle_is_parallel_preserves_raw_bits_and_copies_only_compact_root() {
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_feature_store_v3.cu");
    for token in [
        "canonical_feature_merkle_sha256_host_oracle_v3",
        "CANONICAL_FEATURE_MERKLE_LEAF_DOMAIN_V3",
        "CANONICAL_FEATURE_MERKLE_NODE_DOMAIN_V3",
        "CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V3",
        "pack_logical_validity_u4_v3",
        "logical_validity_code",
    ] {
        assert!(contracts.contains(token), "missing V3 CPU oracle `{token}`");
    }
    for token in [
        "canonical_feature_merkle_leaf_sha256_v3",
        "canonical_feature_merkle_reduce_sha256_v3",
        "canonical_feature_merkle_root_sha256_v3",
        "search_bar_major_value_bits[cell]",
        "search_bar_major_validity_u4[cell / 2U]",
    ] {
        assert!(
            native.contains(token),
            "missing parallel device V3 `{token}`"
        );
    }
    assert!(!native.contains("<<<1, 1"));
    assert!(runtime.contains("Mutex<Option<ResidentFeatureCompactHashesV3>>"));
    assert!(runtime.contains(".copy_to(canonical_content_merkle.as_mut_slice())"));
    assert!(runtime.contains("canonical_root_readback_count: 1"));
    assert!(runtime.contains("if self.validity_error_readback_count == 0"));
    assert!(runtime.contains("else if self.validity_error_readback_count != 1"));
    assert!(runtime.contains("validity_error_readback_count: self.validity_error_readback_count"));
    assert!(runtime.contains("validity_error_d2h_bytes: self.validity_error_d2h_bytes"));
    assert!(runtime.contains("compact_control_plane_d2h_bytes"));
    for forbidden in [
        "search_bar_major_values.copy_to",
        "search_bar_major_validity_u4.copy_to",
        "timestamp_source.buffer().copy_to",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "full payload D2H via `{forbidden}`"
        );
    }
    assert!(!runtime.contains("search_bar_major_values.async_copy_from"));
}

#[test]
fn opaque_import_exposes_no_raw_buffers_or_parent_before_population_consume_seam() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let owner = runtime
        .split_once("impl ResidentFeatureStoreOwnerV3 {")
        .expect("resident owner implementation")
        .1
        .split_once("impl Drop for ResidentFeatureStoreOwnerV3")
        .expect("resident owner implementation boundary")
        .0;
    let import = runtime
        .split_once("impl ResidentFeatureStoreImportV3 {")
        .expect("resident import implementation")
        .1
        .split_once("impl Drop for ResidentFeatureStoreImportV3")
        .expect("resident import implementation boundary")
        .0;

    for forbidden in [
        "pub(crate) fn bar_major_values(",
        "pub(crate) fn bar_major_validity_u4(",
    ] {
        assert!(
            !owner.contains(forbidden),
            "owner must not expose raw resident buffer getter `{forbidden}`"
        );
    }
    for forbidden in [
        "pub(crate) fn bar_major_values(",
        "pub(crate) fn bar_major_validity_u4(",
        "pub(crate) fn parent_source(",
    ] {
        assert!(
            !import.contains(forbidden),
            "opaque import must not expose `{forbidden}` before consume_into_population_session_v3"
        );
    }
    assert!(owner.contains("pub fn parent_dataset_layout(&self)"));
    assert!(import.contains("pub fn record_consumer_completion("));
}

#[test]
fn same_primary_context_event_wait_owns_every_allocation_through_consumer_completion() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    assert!(data.contains("pub struct SealedGpuResidentFeatureStoreV3"));
    assert!(data.contains("Arc<ResidentFeatureStoreOwnerV3>"));
    assert!(!data.contains("pub raw_pointer"));
    assert!(!data.contains("pub device_ptr"));
    for token in [
        "Arc<Context>",
        "Arc<Stream>",
        "cuStreamWaitEvent",
        "ResidentFeatureStoreConsumerLeaseV3",
        "consumer_completion_event.record",
        "consumer_completion_event.query",
        "leak_owner_if_consumer_is_still_running",
        "ResidentProducerReadyEventV3",
        "wait_before_read",
        "ResidentAppendTransactionV3",
        "ResidentSealTransactionV3",
        "StreamOrderedDeviceBufferV3",
        "is_owned_by_stream",
        "drop_async",
    ] {
        assert!(
            runtime.contains(token),
            "missing event-owned handoff `{token}`"
        );
    }
    assert!(!runtime.contains("stream.synchronize()"));
    assert!(!runtime.contains("consumer_stream.synchronize"));
    assert!(!runtime.contains("fn release(\n        self,\n        _stream: &Stream"));
    before(
        &runtime,
        "cuStreamWaitEvent",
        "consumer_completion_event.record",
    );
    before(
        &runtime,
        "let owner = self.owner.as_ref()",
        "self.owner.take()",
    );
}

#[test]
fn native_v3_is_build_linked_and_rust_exported_not_merely_present_as_source() {
    let build = read("crates/neoethos-gpu-cuda/build.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");
    let manifest = read("crates/neoethos-gpu-cuda/Cargo.toml");
    let contract_library = read("crates/neoethos-gpu-contracts/src/lib.rs");
    let contract_manifest = read("crates/neoethos-gpu-contracts/Cargo.toml");
    let data_core = read("crates/neoethos-data/src/core/mod.rs");
    let data_library = read("crates/neoethos-data/src/lib.rs");
    let data_manifest = read("crates/neoethos-data/Cargo.toml");
    for token in [
        "native/resident_feature_store_v3.cu",
        "resident_feature_store_v3",
    ] {
        assert!(build.contains(token), "gpu-cuda build omits `{token}`");
    }
    assert!(library.contains("#[cfg(feature = \"cuda\")]\npub mod resident_feature_store_v3;"));
    assert!(manifest.contains("cuda = [\"dep:cust\", \"dep:vector-ta\"]"));
    assert!(manifest.contains("cust = { version = \"0.3.2\", optional = true }"));
    assert!(manifest.contains("sha2.workspace = true"));
    assert!(contract_library.contains("pub mod resident_feature_store_v3;"));
    assert!(contract_manifest.contains("sha2.workspace = true"));
    assert!(
        data_core
            .contains("#[cfg(feature = \"gpu-cuda\")]\npub mod gpu_resident_feature_store_v3;")
    );
    assert!(data_library.contains("pub use crate::core::gpu_resident_feature_store_v3::{"));
    assert!(data_library.contains("GpuOnlyFeatureMaterializationAdmissionV3"));
    assert!(data_library.contains("SealedGpuResidentFeatureStoreV3"));
    for dependency in [
        "neoethos-gpu-contracts = { path = \"../neoethos-gpu-contracts\", optional = true }",
        "neoethos-gpu-cuda = { path = \"../neoethos-gpu-cuda\", optional = true }",
        "dep:neoethos-gpu-contracts",
        "dep:neoethos-gpu-cuda",
        "neoethos-gpu-cuda/cuda",
    ] {
        assert!(
            data_manifest.contains(dependency),
            "Data V3 dependency/export seam omits `{dependency}`"
        );
    }
}

#[test]
fn native_v3_refuses_null_default_stream_and_selected_device_grid_overflow() {
    let native = read("crates/neoethos-gpu-cuda/native/resident_feature_store_v3.cu");
    for token in [
        "stream == nullptr",
        "source_addresses == nullptr",
        "source_validity_addresses == nullptr",
        "search_bar_major_values == nullptr",
        "merkle_scratch_a == nullptr",
        "digest == nullptr",
        "cudaGetDevice(&device)",
        "cudaGetDeviceProperties(&properties, device)",
        "properties.maxGridSize[0]",
        "properties.maxGridSize[1]",
        "validate_current_device_grid",
    ] {
        assert!(native.contains(token), "missing native refusal `{token}`");
    }
}

#[test]
fn cupqc_is_optional_and_portable_in_tree_v3_is_mandatory() {
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    assert!(contracts.contains("PORTABLE_CUDA_SHA256_AUTHORITY_V3"));
    assert!(contracts.contains("optional_cupqc_acceleration"));
    assert!(contracts.contains("target_os == \"linux\""));
    assert!(contracts.contains("70 | 75 | 80 | 86 | 87 | 89 | 90"));
    assert!(!contracts.contains("100 | 120"));
    assert!(contracts.contains("CuPqcHostCompilerV3::Gcc"));
    assert!(contracts.contains("CuPqcHostCompilerV3::Clang"));
    assert!(!contracts.contains("CuPqcHostCompilerV3::Msvc |"));
    assert!(!contracts.contains("resident_feature_store_v1"));
}

#[test]
fn strict_data_staged_entrypoints_own_the_entire_resident_materialization_sequence() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let library = read("crates/neoethos-data/src/lib.rs");
    let staged_materializer = source
        .split_once("fn materialize_prepared_gpu_only_feature_store_on_run_device_v3(")
        .expect("strict staged Data V3 materializer")
        .1;
    for token in [
        "pub fn prepare_gpu_only_feature_materialization_v3(",
        "pub fn materialize_gpu_only_feature_store_v3(",
        "pub fn materialize_prepared_gpu_only_feature_store_v3(",
        "fn materialize_prepared_gpu_only_feature_store_on_run_device_v3(",
        "workspace_preflight: PreparedGpuOnlyFeatureWorkspacePreflightV3",
        "admitted_run: AdmittedNativeCudaFullDiscoveryRunV1",
        "CrateOwnedResidentProducerFactoryV3::resolve(",
        "preflight_gpu_only_feature_recipe_v3(plan)?",
        "admitted_run.into_gpu_only_run_device_admission_v3()",
        "bind_gpu_only_run_device_v3(preflight, run_device)?",
        "producers.take_smc_materialization()?",
        "admission.begin_materialization(smc_materialization)?",
        "pending_smc_batch.append_to(&mut assembler)?",
        "assembler.try_retire_completed_batch()?",
        "let owner = assembler.seal()?",
        "seal_gpu_resident_feature_store_v3(",
    ] {
        assert!(
            source.contains(token),
            "missing owned orchestration `{token}`"
        );
    }
    for (left, right) in [
        (
            "CrateOwnedResidentProducerFactoryV3::resolve(",
            "admitted_run.into_gpu_only_run_device_admission_v3()",
        ),
        (
            "admitted_run.into_gpu_only_run_device_admission_v3()",
            "admission.begin_materialization(smc_materialization)?",
        ),
        (
            "admission.begin_materialization(smc_materialization)?",
            "pending_smc_batch.append_to(&mut assembler)?",
        ),
        (
            "pending_smc_batch.append_to(&mut assembler)?",
            "let owner = assembler.seal()?",
        ),
        (
            "let owner = assembler.seal()?",
            "seal_gpu_resident_feature_store_v3(",
        ),
    ] {
        let ordering_source = if left == "let owner = assembler.seal()?" {
            staged_materializer
        } else {
            &source
        };
        before(ordering_source, left, right);
    }
    assert!(library.contains("materialize_gpu_only_feature_store_v3"));
    assert!(!source.contains("pub Box<dyn Resident"));
    assert!(!source.contains("pub fn from_raw"));
    assert!(!source.contains("pub fn from_device_ptr"));
    assert!(!source.contains("compute_hpc_feature_frame_sized("));
}

#[test]
fn crate_owned_a2_factory_seals_every_capability_before_the_run_device() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for token in [
        "struct CrateOwnedResidentProducerFactoryV3",
        "struct CrateOwnedResidentMaterializationV3",
        "current_resident_producer_capabilities_v3",
        "ResidentFeatureProducerV3::ALL",
        "require_complete_resident_producer_manifest_v3",
        "MissingProducerCapabilities",
        "A2ProducerFactoryNotIntegrated",
        "current_a2_capability_census_is_complete_and_canonical",
        "ResidentSmcMaterializationV3",
        "PendingResidentSmcBatchV3",
    ] {
        assert!(
            source.contains(token),
            "missing crate-owned A2 authority `{token}`"
        );
    }
    before(
        &source,
        "current_resident_producer_capabilities_v3",
        "admitted_run.into_gpu_only_run_device_admission_v3()",
    );
    before(
        &source,
        "require_complete_resident_producer_manifest_v3",
        "admitted_run.into_gpu_only_run_device_admission_v3()",
    );
    assert!(!source.contains("supports_gpu"));
    assert!(!source.contains("CPU fallback"));
    assert!(!source.contains("HostF64"));
    assert!(!source.contains("FeatureFrame"));
    assert!(!source.contains("Box<dyn ResidentParentDatasetSourceV3>"));
    assert!(!source.contains("Box<dyn ResidentF64FeatureBatchV3>"));
}

#[test]
fn htf_runtime_receipt_survives_append_and_is_revalidated_at_import() {
    let native = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for token in [
        "higher_timeframe_runtime_receipt_v3: Option<ResidentHigherTimeframeRuntimeReceiptV3>",
        "self.higher_timeframe_runtime_receipt_v3 = Some(receipt.clone())",
        "higher_timeframe_runtime_receipt_v3: self.higher_timeframe_runtime_receipt_v3.clone()",
    ] {
        assert!(
            native.contains(token),
            "missing retained HTF receipt token `{token}`"
        );
    }
    for token in [
        "higher_timeframe_runtime_receipt_v3",
        "receipt.parent_count() == self.resident_sources.direct_parents().len()",
        "receipt.parent_feature_column_count() == htf_route_count",
        "receipt.feature_value_d2h_bytes() == 0",
        "receipt.host_synchronize_count() == 0",
        "!htf_runtime_evidence_matches",
    ] {
        assert!(
            data.contains(token),
            "missing sealed HTF evidence token `{token}`"
        );
    }
}
