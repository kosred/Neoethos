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

fn signature<'a>(source: &'a str, declaration: &str) -> &'a str {
    let (_, tail) = source
        .split_once(declaration)
        .unwrap_or_else(|| panic!("missing declaration {declaration:?}"));
    tail.split_once('{')
        .unwrap_or_else(|| panic!("missing body for declaration {declaration:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

fn declaration_window<'a>(source: &'a str, declaration: &str) -> &'a str {
    let start = source
        .find(declaration)
        .unwrap_or_else(|| panic!("missing declaration {declaration:?}"));
    let before = source[..start]
        .rfind("\n\n")
        .map(|offset| offset + 2)
        .unwrap_or(0);
    let body_end = source[start..]
        .find("\n}")
        .map(|offset| start + offset + 2)
        .unwrap_or_else(|| panic!("unterminated declaration {declaration:?}"));
    &source[before..body_end]
}

#[test]
fn crate_exports_the_full_discovery_workspace_plan_and_admitted_run_boundary() {
    let lib = read("src/lib.rs");
    require_all(
        &lib,
        &[
            "mod full_discovery_workspace_plan_v1;",
            "SealedFullDiscoveryGpuWorkspacePlanV1",
            "AdmittedFullDiscoveryGpuRunV1",
            "bind_full_discovery_workspace_plan_v1",
        ],
    );
}

#[test]
fn full_plan_covers_every_resident_discovery_stage_before_materialization() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let bundle = section(
        &source,
        "struct FullDiscoveryWorkspacePreflightBundleV1 {",
        "\n}",
    );
    require_all(
        bundle,
        &[
            "resident_feature_store:",
            "population_parent_and_views:",
            "resident_genetic_evolution:",
            "walk_forward_validation:",
            "cpcv_and_pbo:",
            "outer_holdout_and_oos:",
            "portfolio_constraints:",
            "robustness_tails:",
            "final_compact_readback:",
            "workspace_semantics:",
        ],
    );
    require_all(
        &source,
        &[
            "MissingStageRequirement",
            "FULL_DISCOVERY_WORKSPACE_PLAN_SCHEMA_V1",
            "DISCOVERY_SEMANTICS_VERSION",
        ],
    );
}

#[test]
fn plan_models_always_resident_bytes_reusable_phase_arena_and_bounded_readback() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let plan = section(
        &source,
        "pub struct SealedFullDiscoveryGpuWorkspacePlanV1 {",
        "\n}",
    );
    require_all(
        plan,
        &[
            "always_resident_bytes:",
            "reusable_phase_arena_bytes:",
            "bounded_final_readback_bytes:",
            "phase_lifetime_plan:",
            "workspace_plan_identity_sha256:",
        ],
    );
    require_all(
        &source,
        &[
            "struct FullDiscoveryWorkspacePhaseV1",
            "struct PhaseArenaReuseProofV1",
            "producer_completion_event_identity_sha256:",
            "consumer_wait_event_identity_sha256:",
            "require_non_overlapping_interval_v1(",
        ],
    );
}

#[test]
fn plan_is_sealed_from_opaque_preflights_not_raw_bytes_hashes_or_ordinals() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let seal_signature = signature(&source, "fn seal_full_discovery_gpu_workspace_plan_v1(");
    require_all(
        seal_signature,
        &["preflight: FullDiscoveryWorkspacePreflightBundleV1"],
    );
    for forbidden in [
        "ordinal:",
        "reserve_bytes:",
        "workspace_bytes:",
        "identity_sha256:",
        "hash: String",
        "[u8; 32]",
    ] {
        assert!(
            !seal_signature.contains(forbidden),
            "caller supplied workspace authority via {forbidden:?}"
        );
    }
    for forbidden in [
        "pub fn from_bytes",
        "pub fn from_ordinal",
        "pub fn from_reserve",
        "pub fn from_sha256",
        "pub fn unchecked",
    ] {
        assert!(
            !source.contains(forbidden),
            "caller could mint a full-workspace plan via {forbidden:?}"
        );
    }
}

#[test]
fn mutually_exclusive_validation_and_robustness_scratch_share_one_checked_max_arena() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let arena = section(&source, "fn seal_mutually_exclusive_phase_arena_v1(", "\n}");
    require_all(
        arena,
        &[
            "walk_forward_validation",
            "cpcv_and_pbo",
            "outer_holdout_and_oos",
            "robustness_tails",
            "checked_max_mutually_exclusive_phase_bytes_v1(",
            "require_non_overlapping_interval_v1(",
            "PhaseArenaReuseProofV1",
        ],
    );
    for forbidden in [".sum(", "checked_add", "saturating_add", "+="] {
        assert!(
            !arena.contains(forbidden),
            "mutually exclusive phase scratch was permanently summed via {forbidden:?}"
        );
    }
}

#[test]
fn total_extent_checked_adds_only_resident_arena_and_bounded_readback_classes() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let sealer = section(
        &source,
        "fn seal_full_discovery_gpu_workspace_plan_v1(",
        "\n}",
    );
    require_all(
        sealer,
        &[
            "always_resident_bytes",
            "reusable_phase_arena_bytes",
            "bounded_final_readback_bytes",
            "always_resident_bytes.checked_add(reusable_phase_arena_bytes)",
            ".checked_add(bounded_final_readback_bytes)",
            "WorkspaceExtentOverflow",
            "required_workspace_bytes",
        ],
    );
    for forbidden in [
        "walk_forward_validation.scratch_bytes().checked_add",
        "cpcv_and_pbo.scratch_bytes().checked_add",
        "outer_holdout_and_oos.scratch_bytes().checked_add",
        "robustness_tails.scratch_bytes().checked_add",
        "saturating_add",
        "unwrap_or(u64::MAX)",
        "CONSERVATIVE_BATCH",
        "OCCUPANCY_KNEE",
    ] {
        assert!(
            !sealer.contains(forbidden),
            "non-authoritative permanent stage sum remained via {forbidden:?}"
        );
    }
}

#[test]
fn binding_consumes_the_one_shot_admission_and_the_sealed_full_plan() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let bind_signature = signature(&source, "pub fn bind_full_discovery_workspace_plan_v1(");
    require_all(
        bind_signature,
        &[
            "admission: SealedDiscoveryRunDeviceAdmissionV1",
            "plan: SealedFullDiscoveryGpuWorkspacePlanV1",
            "Result<AdmittedFullDiscoveryGpuRunV1",
        ],
    );
    for forbidden in [
        "&SealedDiscoveryRunDeviceAdmissionV1",
        "&SealedFullDiscoveryGpuWorkspacePlanV1",
    ] {
        assert!(
            !bind_signature.contains(forbidden),
            "full-run authority was borrowed instead of consumed via {forbidden:?}"
        );
    }
}

#[test]
fn binding_uses_the_admissions_actual_memory_snapshot_and_fails_loud_on_no_room() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let binding = section(
        &source,
        "pub fn bind_full_discovery_workspace_plan_v1(",
        "\n}",
    );
    require_all(
        binding,
        &[
            "free_memory_bytes_snapshot",
            "required_workspace_bytes",
            "checked_sub",
            "InsufficientExactOrdinalMemory",
            "admission_identity_sha256",
            "workspace_plan_identity_sha256",
        ],
    );
    for forbidden in [
        "try_smaller_device",
        "RecomputeOnCpu",
        "FallbackDecision",
        "unwrap_or",
        "default_reserve",
    ] {
        assert!(
            !binding.contains(forbidden),
            "capacity failure escaped the selected ordinal via {forbidden:?}"
        );
    }
}

#[test]
fn full_plan_binding_never_reprobes_or_creates_a_second_context_or_stream() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    for forbidden in [
        "probe_physical_gpu_inventory_v1",
        "probe_cuda_device_count_v1",
        "Device::get_device",
        "Context::new",
        "Context::retain_primary",
        "Stream::new",
    ] {
        assert!(
            !source.contains(forbidden),
            "full-plan binding recreated run-device state via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "admission.probe_counters()",
            "require_exact_single_run_device_acquisition_v1()",
        ],
    );
    for forbidden in [
        "physical_inventory_probe_count: 1",
        "cuda_enumeration_count: 1",
        "primary_context_acquisition_count: 1",
        "run_stream_creation_count: 1",
    ] {
        assert!(
            !source.contains(forbidden),
            "workspace binding fabricated a probe counter via {forbidden:?}"
        );
    }
}

#[test]
fn admitted_native_run_carries_the_same_context_stream_device_build_and_plan() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let native = section(
        &source,
        "struct AdmittedNativeCudaFullDiscoveryRunV1 {",
        "\n}",
    );
    require_all(
        native,
        &[
            "physical_inventory_identity_sha256:",
            "admission_identity_sha256:",
            "workspace_plan_identity_sha256:",
            "device_uuid:",
            "pci_identity:",
            "primary_context:",
            "run_stream:",
            "cuda_build_identity:",
            "free_memory_bytes_snapshot:",
        ],
    );
    require_all(
        &source,
        &[
            "into_gpu_only_run_device_admission_v3",
            "GpuOnlyRunDeviceAdmissionV3",
        ],
    );
}

#[test]
fn admitted_native_run_preserves_selected_device_ordinal_as_read_only_evidence() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let native = section(
        &source,
        "struct AdmittedNativeCudaFullDiscoveryRunV1 {",
        "\n}",
    );
    require_all(native, &["selected_device_ordinal: u32"]);
    assert!(
        !native.contains("pub selected_device_ordinal:"),
        "selected ordinal field must remain sealed behind read-only evidence"
    );

    let binding = section(
        &source,
        "fn bind_native_full_discovery_workspace_v1(",
        "\n}",
    );
    require_all(binding, &["selected_device_ordinal: admission.ordinal"]);

    require_all(
        &source,
        &[
            "pub const fn selected_device_ordinal(&self) -> u32",
            "self.selected_device_ordinal",
        ],
    );
    for forbidden in [
        "pub fn set_selected_device_ordinal",
        "pub fn selected_device_ordinal_mut",
        "pub fn from_selected_device_ordinal",
    ] {
        assert!(
            !source.contains(forbidden),
            "selected ordinal authority became caller-mutable via {forbidden:?}"
        );
    }
}

#[test]
fn one_shot_admission_binds_distinct_driver_and_context_api_versions() {
    let admission = read("src/run_device_admission_v1.rs");
    let sealed_native = section(
        &admission,
        "struct SealedNativeCudaRunDeviceAdmissionV1 {",
        "\n}",
    );
    require_all(
        sealed_native,
        &["driver_version: String", "context_api_version: String"],
    );

    let sealer = section(
        &admission,
        "fn seal_native_cuda_run_device_admission_v1(",
        "\n}",
    );
    require_all(
        sealer,
        &[
            "CudaApiVersion::get()",
            "primary_context",
            ".get_api_version()",
            "cuDriverGetVersion={}.{}",
            "cuCtxGetApiVersion={}.{}",
        ],
    );
    require_all(
        &admission,
        &[
            "driver_version.as_bytes()",
            "context_api_version.as_bytes()",
        ],
    );
    for forbidden in [
        "context_api_version = driver_version",
        "context_api_version = candidate.cuda_build_identity.nvcc_version",
        "context_api_version = env!",
    ] {
        assert!(
            !sealer.contains(forbidden),
            "context runtime identity was reconstructed via {forbidden:?}"
        );
    }

    let full_plan = read("src/full_discovery_workspace_plan_v1.rs");
    let admitted = section(
        &full_plan,
        "struct AdmittedNativeCudaFullDiscoveryRunV1 {",
        "\n}",
    );
    require_all(admitted, &["context_api_version: String"]);
    let binding = section(
        &full_plan,
        "fn bind_native_full_discovery_workspace_v1(",
        "\n}",
    );
    require_all(
        binding,
        &["context_api_version: admission.context_api_version"],
    );
}

#[test]
fn admitted_full_workspace_consumes_into_one_opaque_v3_run_device_carrier() {
    let full_plan = read("src/full_discovery_workspace_plan_v1.rs");
    let conversion = section(
        &full_plan,
        "pub fn into_gpu_only_run_device_admission_v3(",
        "\n    }",
    );
    require_all(
        conversion,
        &[
            "seal_gpu_only_run_device_admission_v3(",
            "FullDiscoveryRunDeviceAdmissionRequestV3",
            "source_admission_identity_sha256:",
            "workspace_plan_identity_sha256,",
            "selected_device_ordinal,",
            "vector_ta_build_sha256,",
            "gpu_cuda_build_sha256,",
            "exact_math_authority,",
            "primary_context,",
            "run_stream,",
        ],
    );
    for forbidden in [
        "let _ = self",
        "ResidentStoreCarrierNotIntegrated",
        "probe_physical_gpu_inventory_v1",
        "probe_cuda_device_count_v1",
        "Device::get_device",
        "Context::new",
        "Stream::new",
    ] {
        assert!(
            !conversion.contains(forbidden),
            "V3 carrier conversion bypasses the admitted run via {forbidden:?}"
        );
    }

    let resident = read("src/resident_feature_store_v3.rs");
    let request = section(
        &resident,
        "struct FullDiscoveryRunDeviceAdmissionRequestV3 {",
        "\n}",
    );
    require_all(
        request,
        &[
            "source_admission_identity_sha256:",
            "workspace_plan_identity_sha256:",
            "selected_device_ordinal:",
            "driver_version:",
            "context_api_version:",
            "vector_ta_build_sha256:",
            "gpu_cuda_build_sha256:",
            "exact_math_authority:",
            "primary_context:",
            "run_stream:",
        ],
    );
    assert!(
        !request.contains("pub struct"),
        "the full-workspace carrier request must remain crate-private"
    );

    let sealer = section(
        &resident,
        "fn seal_gpu_only_run_device_admission_v3(",
        "\n}",
    );
    require_all(
        sealer,
        &[
            "CudaPrimaryContextBuildIdentityV3::new(",
            "process_handle_token_v3(",
            "request.driver_version",
            "request.context_api_version",
            "request.vector_ta_build_sha256",
            "request.gpu_cuda_build_sha256",
            "request.exact_math_authority",
            "RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3",
            "GpuOnlyRunDeviceAdmissionV3 {",
        ],
    );
    require_all(
        &resident,
        &[
            "fn hash_gpu_only_run_device_admission_v3(",
            "request.source_admission_identity_sha256",
            "request.workspace_plan_identity_sha256",
        ],
    );
}

#[test]
fn workspace_hash_has_one_typed_input_and_v3_request_uses_shorthand_fields() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let hash_input = section(
        &source,
        "struct FullDiscoveryWorkspacePlanHashInputV1<'a> {",
        "\n}",
    );
    require_all(
        hash_input,
        &[
            "always_resident_bytes:",
            "reusable_phase_arena_bytes:",
            "bounded_final_readback_bytes:",
            "required_workspace_bytes:",
            "phases:",
            "reuse_proof:",
            "component_identity_sha256:",
            "exact_math_authority:",
        ],
    );
    let signature = signature(&source, "fn hash_workspace_plan_v1(");
    assert!(
        signature.contains("input: &FullDiscoveryWorkspacePlanHashInputV1<'_>"),
        "workspace hash must take one typed input instead of an argument fan-out"
    );
    require_all(
        &source,
        &["hash_workspace_plan_v1(&FullDiscoveryWorkspacePlanHashInputV1 {"],
    );
    for redundant in [
        "workspace_plan_identity_sha256: workspace_plan_identity_sha256",
        "selected_device_ordinal: selected_device_ordinal",
        "primary_context: primary_context",
        "run_stream: run_stream",
        "vector_ta_build_sha256: vector_ta_build_sha256",
        "gpu_cuda_build_sha256: gpu_cuda_build_sha256",
        "exact_math_authority: exact_math_authority",
    ] {
        assert!(
            !source.contains(redundant),
            "redundant V3 request field initializer remained: {redundant}"
        );
    }
}

#[test]
fn plan_and_admitted_run_are_move_only_and_cannot_be_rehydrated_from_receipts() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    for declaration in [
        "pub struct SealedFullDiscoveryGpuWorkspacePlanV1 {",
        "pub enum AdmittedFullDiscoveryGpuRunV1 {",
        "struct AdmittedNativeCudaFullDiscoveryRunV1 {",
    ] {
        let window = declaration_window(&source, declaration);
        for forbidden in ["Clone", "Default", "Deserialize"] {
            assert!(
                !window.contains(forbidden),
                "full-run authority gained reconstructible trait {forbidden:?}"
            );
        }
    }
    for forbidden in [
        "from_receipt",
        "from_identity_sha256",
        "from_workspace_plan_sha256",
        "impl Clone for AdmittedFullDiscoveryGpuRunV1",
    ] {
        assert!(
            !source.contains(forbidden),
            "durable evidence could rehydrate authority via {forbidden:?}"
        );
    }
}

#[test]
fn completion_receipt_distinguishes_zero_intermediate_d2h_from_one_bounded_final_readback() {
    let source = read("src/full_discovery_workspace_plan_v1.rs");
    let receipt = section(&source, "pub struct FullDiscoveryGpuRunReceiptV1 {", "\n}");
    require_all(
        receipt,
        &[
            "intermediate_device_to_host_count:",
            "intermediate_device_to_host_bytes:",
            "final_compact_readback_count:",
            "final_compact_readback_bytes:",
            "final_compact_readback_limit_bytes:",
        ],
    );
    let sealer = section(&source, "fn seal_full_discovery_gpu_run_receipt_v1(", "\n}");
    require_all(
        sealer,
        &[
            "intermediate_device_to_host_count == 0",
            "intermediate_device_to_host_bytes == 0",
            "final_compact_readback_count == 1",
            "final_compact_readback_bytes <= final_compact_readback_limit_bytes",
        ],
    );
}
