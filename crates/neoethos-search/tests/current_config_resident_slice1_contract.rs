use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve repository root")
}

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn body_after<'a>(source: &'a str, marker: &str) -> &'a str {
    source
        .split_once(marker)
        .unwrap_or_else(|| panic!("missing source marker {marker}"))
        .1
}

#[test]
fn full_workspace_seals_and_exports_exact_trim_and_total_reserves() {
    let source = read("crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs");
    let sealed = body_after(
        &source,
        "pub struct SealedFullDiscoveryGpuWorkspacePlanV1 {",
    );
    for field in [
        "trim_prefilter_reserved_bytes: u64",
        "required_workspace_bytes: u64",
        "workspace_plan_identity_sha256: [u8; 32]",
    ] {
        assert!(
            sealed.contains(field),
            "sealed workspace is missing {field}"
        );
    }
    let conversion = body_after(&source, "pub fn into_gpu_only_run_device_admission_v3(");
    for field in [
        "trim_prefilter_reserved_bytes",
        "required_workspace_bytes",
        "full_discovery_trim_admission",
    ] {
        assert!(
            conversion.contains(field),
            "full-workspace conversion drops {field}"
        );
    }
}

#[test]
fn gpu_only_admission_retains_trim_reserve_without_public_raw_handles() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let admission = body_after(&source, "pub struct GpuOnlyRunDeviceAdmissionV3 {");
    assert!(admission.contains("full_discovery_trim_admission"));
    assert!(admission.contains("required_workspace_bytes"));
    assert!(admission.contains("trim_prefilter_reserved_bytes"));
    assert!(!admission.contains("pub admitted_run_stream: *mut"));
    assert!(!admission.contains("pub primary_context: *mut"));
}

#[test]
fn resident_store_is_move_consumed_into_trim_before_population() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let conversion = body_after(&source, "pub fn consume_into_resident_trim_prefilter_v1(");
    let population = conversion
        .find("pub fn consume_into_population_session_v3(")
        .expect("existing direct population conversion remains source-visible");
    let conversion = &conversion[..population];
    for requirement in [
        "self.owner.take()",
        "self.consumer_context.take()",
        "self.consumer_stream.take()",
        "ResidentTrimPrefilterParentImportV1",
        "SealedResidentColumnClassificationV1",
        "ResidentTrimPrefilterFullDiscoveryAdmissionV1",
    ] {
        assert!(
            conversion.contains(requirement),
            "trim conversion is missing {requirement}"
        );
    }
    assert!(!conversion.contains("copy_to("));
    assert!(!conversion.contains("synchronize("));
}

#[test]
fn sealed_trim_views_have_one_move_only_population_consumer() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_trim_prefilter_v1.rs");
    assert_eq!(
        source
            .matches("pub fn consume_into_population_session_v3(")
            .count(),
        1,
        "there must be one opaque trim-to-population ownership transfer"
    );
    let consumer = body_after(&source, "pub fn consume_into_population_session_v3(");
    for requirement in [
        "self.parent_import.take()",
        "self.sealed_schema.take()",
        "self.full_admission.take()",
        "selected_compact_to_parent_columns_device",
        "selected_column_count_device",
        "trim_prefilter_ready_event",
    ] {
        assert!(
            consumer.contains(requirement),
            "trim-to-population transfer is missing {requirement}"
        );
    }
    assert!(!consumer.contains("cudaMemcpyDeviceToHost"));
    assert!(!consumer.contains("synchronize("));
}

#[test]
fn production_native_discovery_uses_the_bounded_two_pass_screening_owner() {
    let source = read("crates/neoethos-search/src/canonical_native_discovery_run_v1.rs");
    for requirement in [
        "prepare_resident_feature_screening_v2",
        "seal_feature_screening_workspace_plan_v2",
        "bind_feature_screening_gpu_workspace_plan_v2",
        "begin_prepared_gpu_only_feature_two_pass_v2",
        "stream_score_batches_v2",
        "seal_selected_map_v2",
        "prepare_compact_selected_store_v2",
        "prepare_compact_selected_canonical_trendbar_research_run_input_capped_v6",
        "materialize_compact_selected_store_v2",
    ] {
        assert!(
            source.contains(requirement),
            "prepared native path is missing {requirement}"
        );
    }
    for obsolete in [
        "GpuNativeTrimPrefilterViewIdentityV3",
        "begin_gpu_resident_trim_prefilter_view_v1",
        "execute_gpu_resident_trim_prefilter_view_v1",
        "seal_gpu_resident_trim_prefilter_view_v1",
        "materialize_prepared_gpu_only_feature_store_for_data_population_v3",
    ] {
        assert!(
            !source.contains(obsolete),
            "production native path still exposes superseded route {obsolete}"
        );
    }
}

#[test]
fn cpu_and_resident_trim_share_one_schema_classification_authority() {
    let shared = read("crates/neoethos-search/src/prefilter_schema_v1.rs");
    for requirement in [
        "PREFILTER_STATE_FAMILIES_V1",
        "is_prefilter_state_column_v1",
        "timeframe_group_v1",
        "template_feature_indices",
        "seal_prefilter_column_classification_v1",
        "column_classification_content_sha256",
    ] {
        assert!(
            shared.contains(requirement),
            "shared classification authority is missing {requirement}"
        );
    }
    let discovery = read("crates/neoethos-search/src/discovery.rs");
    assert!(discovery.contains("prefilter_schema_v1::is_prefilter_state_column_v1"));
    assert!(discovery.contains("prefilter_schema_v1::timeframe_group_v1"));
    assert!(!discovery.contains("fn is_prefilter_state_column("));
    assert!(!discovery.contains("fn timeframe_group("));
    let discovery_tests = read("crates/neoethos-search/src/discovery_tests.rs");
    assert!(!discovery_tests.contains("timeframe_group(n)"));
}

#[test]
fn trim_preflight_is_a_distinct_native_calibrated_workspace_extent() {
    let source = read("crates/neoethos-gpu-cuda/src/full_discovery_workspace_plan_v1.rs");
    assert!(source.contains("OpaqueResidentTrimPrefilterPreflightV1"));
    for requirement in [
        "resident_trim_prefilter: OpaqueResidentTrimPrefilterPreflightV1",
        "peak_device_bytes",
        "retained_view_device_bytes",
        "cub_select_scratch_bytes",
        "cub_radix_sort_scratch_bytes",
        "population_overlap_device_bytes",
        "native_query_identity_sha256",
        "calibration_identity_sha256",
        "preflight_identity_sha256",
        "require_resident_trim_prefilter_preflight_v1",
    ] {
        assert!(
            source.contains(requirement),
            "trim preflight authority is missing {requirement}"
        );
    }
    assert!(source.contains("Self::ResidentTrimPrefilter => \"resident-trim-prefilter\""));
    assert!(!source.contains(
        "let trim_prefilter_reserved_bytes = preflight.population_parent_and_views.device_bytes;"
    ));
    assert!(!source.contains("trim_prefilter_reserved_bytes: 1"));
    let screening = read("crates/neoethos-search/src/gpu_resident_feature_screening_v2.rs");
    let preparation = body_after(
        &screening,
        "pub(crate) fn prepare_resident_feature_screening_v2(",
    );
    let trim_preflight = preparation
        .find("preflight_resident_trim_prefilter_workspace_v2(")
        .expect("production screening must query the exact native trim workspace");
    let sealed_preparation = preparation
        .find("Ok(PreparedResidentFeatureScreeningV2")
        .expect("production screening must seal its prepared authority");
    assert!(
        trim_preflight < sealed_preparation,
        "native trim calibration must finish before the screening authority is sealed"
    );

    let production = read("crates/neoethos-search/src/canonical_native_discovery_run_v1.rs");
    let workspace_seal = production
        .find("seal_feature_screening_workspace_plan_v2(")
        .expect("production must seal the calibrated screening workspace");
    let run_begin = production
        .find("begin_prepared_gpu_only_feature_two_pass_v2(")
        .expect("production must begin the bounded two-pass run");
    assert!(
        workspace_seal < run_begin,
        "the calibrated trim extent must be admitted before any screening allocation"
    );
}
