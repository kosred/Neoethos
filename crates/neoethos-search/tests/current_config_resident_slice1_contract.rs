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
fn prepared_native_discovery_uses_real_trim_owner_not_identity_placeholder() {
    let source = read("crates/neoethos-search/src/prepared_discovery_run_input_v3.rs");
    assert!(!source.contains("fn seal_gpu_native_trim_prefilter_view_identity_v3("));
    assert!(!source.contains("GpuNativeTrimPrefilterViewIdentityV3"));
    for requirement in [
        "seal_current_config_resident_search_plan_v1",
        "begin_gpu_resident_trim_prefilter_view_v1",
        "execute_gpu_resident_trim_prefilter_view_v1",
        "seal_gpu_resident_trim_prefilter_view_v1",
        "consume_into_population_session_v3",
    ] {
        assert!(
            source.contains(requirement),
            "prepared native path is missing {requirement}"
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
}
