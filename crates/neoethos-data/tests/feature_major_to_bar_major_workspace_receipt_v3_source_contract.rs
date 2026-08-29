use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative:?}: {error}"))
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
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

#[test]
fn capability_is_bound_to_the_real_in_tree_cuda_layout_implementation() {
    let runtime = read("../neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let native = read("../neoethos-gpu-cuda/native/resident_feature_store_v3.cu");

    require_all(
        &runtime,
        &[
            "pub fn resident_feature_major_to_bar_major_capability_v3()",
            "ResidentFeatureProducerV3::FeatureMajorToBarMajor",
            "include_bytes!(\"resident_feature_store_v3.rs\")",
            "include_bytes!(\"../native/resident_feature_store_v3.cu\")",
            "FEATURE_MAJOR_TO_BAR_MAJOR_EXACT_AUTHORITY_V3",
        ],
    );
    require_all(
        &native,
        &[
            "pack_sources_to_bar_major_f64_u4_v3",
            "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3",
            "search_bar_major_values",
            "search_bar_major_validity_u4",
        ],
    );
}

#[test]
fn data_seals_private_allocation_lifetime_and_component_receipts_from_its_plan() {
    let source = read("src/core/gpu_resident_feature_store_v3.rs");
    require_all(
        &source,
        &[
            "pub(crate) struct FeatureMajorToBarMajorAllocationReceiptV3",
            "pub(crate) struct FeatureMajorToBarMajorLifetimeReceiptV3",
            "pub(crate) struct SealedFeatureMajorToBarMajorComponentReceiptV3",
            "fn seal_feature_major_to_bar_major_component_receipt_v3(",
            "plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3",
            "resident_feature_major_to_bar_major_capability_v3()?",
            "FeatureMajorToBarMajorLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion",
            "full_feature_major_staging_bytes: 0",
        ],
    );

    for receipt in [
        "FeatureMajorToBarMajorAllocationReceiptV3",
        "FeatureMajorToBarMajorLifetimeReceiptV3",
        "SealedFeatureMajorToBarMajorComponentReceiptV3",
    ] {
        let declaration = format!("pub(crate) struct {receipt} {{");
        let fields = section(&source, &declaration, "\n}");
        assert!(
            !fields.contains("pub "),
            "{receipt} fields must stay private"
        );
        for forbidden in ["Clone", "Serialize", "Deserialize", "Default"] {
            assert!(
                !source.contains(&format!("{forbidden} for {receipt}")),
                "{receipt} must not be caller-rehydratable via {forbidden}"
            );
        }
    }

    let signature = section(
        &source,
        "fn seal_feature_major_to_bar_major_component_receipt_v3(",
        ") ->",
    );
    assert!(signature.contains("plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3"));
    for forbidden in [
        "[u8; 32]",
        "String",
        "device_bytes",
        "workspace_bytes",
        "ordinal",
        "context",
        "stream",
    ] {
        assert!(
            !signature.contains(forbidden),
            "component receipt accepts caller evidence via {forbidden:?}"
        );
    }
}

#[test]
fn receipt_moves_through_preflight_admission_and_seal_then_checks_real_evidence() {
    let source = read("src/core/gpu_resident_feature_store_v3.rs");
    let compact = normalized(&source);
    require_all(
        &compact,
        &[
            "GpuOnlyFeatureRecipePreflightV3{plan,footprint,feature_major_to_bar_major,canonical_content_sha256,}",
            "GpuOnlyFeatureMaterializationAdmissionV3{authority:DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3,contract,run_device,footprint,feature_major_to_bar_major,canonical_content_sha256,}",
            "GpuOnlyFeatureMaterializationSealTokenV3{authority:self.authority,contract:self.contract,footprint:self.footprint,feature_major_to_bar_major:self.feature_major_to_bar_major,canonical_content_sha256:self.canonical_content_sha256,}",
            "feature_major_to_bar_major.validate_working_set(&working_set)?",
            "feature_major_to_bar_major.validate_runtime_evidence(&evidence,&ready_event)?",
        ],
    );
    assert!(
        !source.contains("feature_major_to_bar_major.clone()"),
        "the component receipt must move instead of cloning"
    );
}

#[test]
fn production_census_retains_the_real_layout_producer_after_later_receipts() {
    let resident = read("src/core/gpu_resident_feature_store_v3.rs");
    let preflight = read("src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let census = section(
        &resident,
        "fn current_resident_producer_capabilities_v3()",
        "\n}",
    );
    require_all(
        census,
        &[
            "resident_classic_ta_capability_v3()?",
            "resident_smc_capability_v3()?",
            "resident_feature_major_to_bar_major_capability_v3()?",
        ],
    );

    let pending = section(
        &preflight,
        "pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3:",
        "];",
    );
    for producer in [
        "ResidentFeatureProducerV3::Quant",
        "ResidentFeatureProducerV3::Session",
        "ResidentFeatureProducerV3::Regime",
        "ResidentFeatureProducerV3::HigherTimeframeAlignment",
        "ResidentFeatureProducerV3::RobustNormalization",
    ] {
        assert!(pending.contains(producer), "missing pending {producer}");
    }
    assert_eq!(pending.matches("ResidentFeatureProducerV3::").count(), 5);
    assert!(!pending.contains("CanonicalContentSha256"));
    assert!(!pending.contains("FeatureMajorToBarMajor"));
}

#[test]
fn complete_workspace_component_stays_red_but_requires_the_typed_layout_receipt() {
    let source = read("tests/gpu_only_feature_workspace_preflight_v3_source_contract.rs");
    let red = section(
        &source,
        "fn red_data_component_receipt_must_consume_the_preflight_without_caller_evidence()",
        "\n}",
    );
    require_all(
        red,
        &[
            "pub(crate) struct SealedGpuOnlyFeatureWorkspaceComponentV3",
            "feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3",
        ],
    );
}
