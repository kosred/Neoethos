use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read_or_empty(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative)).unwrap_or_default()
}

fn normalized(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
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
        assert!(source.contains(token), "missing preflight token {token:?}");
    }
}

#[test]
fn preflight_owns_the_exact_pin_and_complete_census_without_decoding_values() {
    let source = read_or_empty("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    require_all(
        &source,
        &[
            "pub struct PreparedGpuOnlyFeatureWorkspacePreflightV3",
            "source_descriptor: PinnedResidentCanonicalSourceDescriptorV1",
            "producer_capabilities: ResidentProducerCapabilityManifestV3",
            "pub fn preflight_gpu_only_feature_workspace_v3(",
            "seal_current_resident_producer_capability_manifest_v3()",
            "into_resident_feature_recipe_assembly_v4",
        ],
    );
    require_all(
        &normalized(&source),
        &[
            "pinned_series.row_count(base_timeframe)",
            "pinned_series.into_resident_source_descriptor_v1()",
        ],
    );

    let constructor = section(
        &source,
        "pub fn preflight_gpu_only_feature_workspace_v3(",
        "\n}",
    );
    for forbidden in [
        "CanonicalOhlcvFrame",
        "FeatureFrame",
        "materialize_pinned_canonical_timeframe_v1",
        "into_cpu_dataset_after_no_physical_gpu_v1",
        "vortex_array_to_ohlcv",
        "load_exact_canonical_timeframe",
    ] {
        assert!(
            !constructor.contains(forbidden),
            "metadata-only preflight decoded or materialized values via {forbidden:?}"
        );
    }
}

#[test]
fn preflight_authority_is_move_only_and_not_rehydratable_from_caller_evidence() {
    let source = read_or_empty("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let prepared = section(
        &source,
        "pub struct PreparedGpuOnlyFeatureWorkspacePreflightV3 {",
        "\n}",
    );
    assert!(
        !prepared.contains("pub "),
        "prepared preflight fields must remain private"
    );
    for forbidden in [
        "impl Clone for PreparedGpuOnlyFeatureWorkspacePreflightV3",
        "impl Default for PreparedGpuOnlyFeatureWorkspacePreflightV3",
        "Deserialize for PreparedGpuOnlyFeatureWorkspacePreflightV3",
        "Serialize for PreparedGpuOnlyFeatureWorkspacePreflightV3",
        "from_raw",
        "from_bytes",
        "from_hash",
        "from_sha256",
        "from_ordinal",
    ] {
        assert!(
            !source.contains(forbidden),
            "prepared authority can be reconstructed via {forbidden:?}"
        );
    }

    let signature = section(
        &source,
        "pub fn preflight_gpu_only_feature_workspace_v3(",
        ") ->",
    );
    require_all(
        signature,
        &[
            "pinned_series: PinnedCanonicalSeriesV1",
            "base_timeframe: CanonicalTimeframe",
            "profile: FeatureProfile",
            "budget_rows: usize",
        ],
    );
    for forbidden in [
        "[u8; 32]",
        "String",
        "workspace_bytes",
        "device_ordinal",
        "context",
        "stream",
    ] {
        assert!(
            !signature.contains(forbidden),
            "caller can supply preflight authority via {forbidden:?}"
        );
    }
}

#[test]
fn ordered_producer_and_component_receipt_backlogs_are_frozen_exactly() {
    let source = read_or_empty("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let producers = section(
        &source,
        "pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3:",
        "];",
    );
    assert_eq!(
        producers.matches("ResidentFeatureProducerV3::").count(),
        0,
        "the current ordered missing-producer frontier changed"
    );

    let receipts = section(
        &source,
        "pub const CURRENT_PENDING_FEATURE_WORKSPACE_RECEIPTS_V3:",
        "];",
    );
    assert_eq!(
        receipts
            .matches("GpuOnlyFeatureWorkspaceReceiptBacklogV3::")
            .count(),
        0,
        "the ordered component-receipt backlog changed"
    );
}

#[test]
fn prepared_preflight_moves_the_descriptor_and_split_into_recipe_assembly_once() {
    let source = read_or_empty("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let recipe = read_or_empty("src/core/gpu_resident_feature_recipe_v4.rs");
    let method = section(
        &source,
        "pub(crate) fn into_resident_feature_recipe_assembly_v4(",
        "\n    }",
    );
    require_all(
        method,
        &[
            "let Self {",
            "source_descriptor",
            "robust_normalization_split",
            "PreparedResidentFeatureRecipeAssemblyV4::from_workspace_preflight(",
        ],
    );
    for forbidden in ["clone()", "from_hash", "from_bytes", "[u8; 32]"] {
        assert!(
            !method.contains(forbidden),
            "prepared recipe assembly rehydrates authority through {forbidden:?}"
        );
    }
    require_all(
        &recipe,
        &[
            "pub(crate) struct PreparedResidentFeatureRecipeAssemblyV4",
            "source_descriptor: PinnedResidentCanonicalSourceDescriptorV1",
            "robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2",
            "column_schema: ResidentColumnSchemaAssemblerV4",
        ],
    );
}

#[test]
fn data_seals_its_current_census_before_any_one_shot_run_carrier_is_consumed() {
    let resident = read_or_empty("src/core/gpu_resident_feature_store_v3.rs");
    require_all(
        &resident,
        &[
            "pub(crate) fn seal_current_resident_producer_capability_manifest_v3(",
            "require_complete_resident_producer_manifest_v3(",
            "current_resident_producer_capabilities_v3()?",
        ],
    );
    let materialize = section(
        &resident,
        "pub fn materialize_gpu_only_feature_store_v3(",
        "\n}",
    );
    let resolve = materialize
        .find("CrateOwnedResidentProducerFactoryV3::resolve")
        .expect("Data must resolve the complete producer census");
    let consume = materialize
        .find("admitted_run.into_gpu_only_run_device_admission_v3()")
        .expect("Data must consume the one-shot run carrier");
    assert!(
        resolve < consume,
        "producer preflight must fail before the one-shot carrier is consumed"
    );
}

#[test]
fn data_exports_only_the_preflight_token_and_descriptive_backlog_surface() {
    let core = read_or_empty("src/core/mod.rs");
    let library = read_or_empty("src/lib.rs");
    assert!(
        core.contains("pub mod gpu_only_feature_workspace_preflight_v3;"),
        "Data core does not register the preflight module"
    );
    require_all(
        &library,
        &[
            "PreparedGpuOnlyFeatureWorkspacePreflightV3",
            "GpuOnlyFeatureWorkspaceReceiptBacklogV3",
            "CURRENT_PENDING_RESIDENT_PRODUCERS_V3",
            "CURRENT_PENDING_FEATURE_WORKSPACE_RECEIPTS_V3",
            "preflight_gpu_only_feature_workspace_v3",
        ],
    );
}

#[test]
fn completed_workspace_preflight_moves_directly_into_the_resident_materializer() {
    let preflight = read_or_empty("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let materializer = read_or_empty("src/core/gpu_resident_feature_store_v3.rs");
    require_all(
        &preflight,
        &[
            "seal_current_resident_producer_capability_manifest_v3()?",
            "into_resident_feature_recipe_assembly_v4(",
            "PreparedResidentFeatureRecipeAssemblyV4::from_workspace_preflight(",
        ],
    );
    require_all(
        &materializer,
        &[
            "workspace_preflight: PreparedGpuOnlyFeatureWorkspacePreflightV3",
            "CrateOwnedResidentProducerFactoryV3::resolve(workspace_preflight)?",
            "preflight_resident_higher_timeframe_alignment_v3(",
            "prepared_htf_append.append_to(&mut assembler)?",
            "seal_token.apply_resident_robust_normalization_v2(&mut assembler)?",
        ],
    );
}
