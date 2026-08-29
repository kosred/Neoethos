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
fn capability_is_bound_to_the_actual_parallel_merkle_cuda_implementation() {
    let runtime = read("../neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let native = read("../neoethos-gpu-cuda/native/resident_feature_store_v3.cu");
    require_all(
        &runtime,
        &[
            "pub fn resident_canonical_content_sha256_capability_v3()",
            "ResidentFeatureProducerV3::CanonicalContentSha256",
            "PORTABLE_CUDA_SHA256_AUTHORITY_V3",
            "include_bytes!(\"resident_feature_store_v3.rs\")",
            "include_bytes!(\"../native/resident_feature_store_v3.cu\")",
        ],
    );
    require_all(
        &native,
        &[
            "canonical_feature_merkle_leaf_sha256_v3",
            "canonical_feature_merkle_reduce_sha256_v3",
            "canonical_feature_merkle_root_sha256_v3",
            "neoethos_resident_canonical_merkle_sha256_v3",
        ],
    );
}

#[test]
fn data_seals_private_root_scratch_lifetime_and_component_receipts() {
    let source = read("src/core/gpu_resident_feature_store_v3.rs");
    require_all(
        &source,
        &[
            "pub(crate) struct CanonicalContentSha256AllocationReceiptV3",
            "pub(crate) struct CanonicalContentSha256LifetimeReceiptV3",
            "pub(crate) struct SealedCanonicalContentSha256ComponentReceiptV3",
            "fn seal_canonical_content_sha256_component_receipt_v3(",
            "plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3",
            "resident_canonical_content_sha256_capability_v3()?",
            "CanonicalContentSha256ScratchLifetimeV3::ThroughFinalReadyAndCompactRootReadback",
            "CanonicalContentSha256RootLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion",
        ],
    );
    let signature = section(
        &source,
        "fn seal_canonical_content_sha256_component_receipt_v3(",
        ") ->",
    );
    assert!(signature.contains("plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3"));
    for forbidden in [
        "[u8; 32]",
        "String",
        "device_bytes",
        "scratch_bytes",
        "workspace_bytes",
        "ordinal",
        "context",
        "stream",
    ] {
        assert!(
            !signature.contains(forbidden),
            "canonical receipt accepts caller evidence via {forbidden:?}"
        );
    }
}

#[test]
fn real_owner_evidence_covers_root_scratch_readback_and_event_lifetimes() {
    let runtime = read("../neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let data = read("src/core/gpu_resident_feature_store_v3.rs");
    let compact_data = normalized(&data);
    require_all(
        &runtime,
        &[
            "merkle_leaf_count: usize",
            "merkle_scratch_bytes: usize",
            "canonical_root_device_bytes:",
            ".canonical_content_merkle",
            "hash_transient: Mutex<Option<ResidentHashTransientV3>>",
            "retire_hash_transient_after_ready",
            "canonical_root_readback_count: 1",
            "canonical_root_d2h_bytes: SHA256_BYTES",
        ],
    );
    require_all(
        &compact_data,
        &[
            "canonical_content_sha256.validate_working_set(&working_set)?",
            "canonical_content_sha256.validate_runtime_evidence(&evidence,&ready_event)?",
            "evidence.merkle_leaf_count",
            "evidence.merkle_scratch_bytes",
            "evidence.canonical_root_device_bytes",
            "evidence.canonical_root_readback_count",
            "evidence.canonical_root_d2h_bytes",
        ],
    );
}

#[test]
fn receipt_moves_with_the_existing_component_authority_without_clone_or_rehydration() {
    let source = read("src/core/gpu_resident_feature_store_v3.rs");
    let compact = normalized(&source);
    require_all(
        &compact,
        &[
            "GpuOnlyFeatureRecipePreflightV3{plan,footprint,regime,robust_normalization,feature_major_to_bar_major,canonical_content_sha256,}",
            "GpuOnlyFeatureMaterializationAdmissionV3{authority:DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3,contract,run_device,footprint,regime,robust_normalization,feature_major_to_bar_major,canonical_content_sha256,}",
            "GpuOnlyFeatureMaterializationSealTokenV3{authority:self.authority,contract:self.contract,footprint:self.footprint,regime:self.regime,robust_normalization:self.robust_normalization,feature_major_to_bar_major:self.feature_major_to_bar_major,canonical_content_sha256:self.canonical_content_sha256,}",
            "canonical_content_sha256:seal_token.canonical_content_sha256,",
        ],
    );
    assert!(!source.contains("canonical_content_sha256.clone()"));
    for forbidden in ["from_raw", "from_bytes", "from_hash", "from_sha256"] {
        assert!(
            !source.contains(&format!(
                "{forbidden}_canonical_content_sha256_component_receipt"
            )),
            "canonical receipt can be rehydrated via {forbidden}"
        );
    }
}

#[test]
fn production_census_retains_canonical_receipt_after_later_producers_advance() {
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
            "resident_robust_normalization_capability_v2()?",
            "resident_canonical_content_sha256_capability_v3()?",
            "resident_feature_major_to_bar_major_capability_v3()?",
        ],
    );
    let canonical = census
        .find("resident_canonical_content_sha256_capability_v3()?")
        .expect("canonical capability");
    let layout = census
        .find("resident_feature_major_to_bar_major_capability_v3()?")
        .expect("layout capability");
    assert!(
        canonical < layout,
        "producer census must preserve enum order"
    );

    let pending = section(
        &preflight,
        "pub const CURRENT_PENDING_RESIDENT_PRODUCERS_V3:",
        "];",
    );
    for producer in [
        "ResidentFeatureProducerV3::Quant",
        "ResidentFeatureProducerV3::Session",
        "ResidentFeatureProducerV3::HigherTimeframeAlignment",
    ] {
        assert!(pending.contains(producer), "missing pending {producer}");
    }
    assert_eq!(pending.matches("ResidentFeatureProducerV3::").count(), 3);
    assert!(!pending.contains("ResidentFeatureProducerV3::RobustNormalization"));
    assert!(!pending.contains("CanonicalContentSha256"));
    assert!(!pending.contains("FeatureMajorToBarMajor"));
}

#[test]
fn complete_workspace_component_remains_red_and_requires_both_real_receipts() {
    let source = read("tests/gpu_only_feature_workspace_preflight_v3_source_contract.rs");
    let red = section(
        &source,
        "fn red_data_component_receipt_must_consume_the_preflight_without_caller_evidence()",
        "\n}",
    );
    require_all(
        red,
        &[
            "feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3",
            "canonical_content_sha256: SealedCanonicalContentSha256ComponentReceiptV3",
        ],
    );
}
