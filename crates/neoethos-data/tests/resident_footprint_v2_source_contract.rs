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
        assert!(source.contains(token), "missing Footprint token {token:?}");
    }
}

#[test]
fn cpu_oracle_freezes_semantic_v2_order_math_and_validity() {
    let cpu = read("crates/neoethos-data/src/core/footprint_features.rs");
    let registry = read("crates/neoethos-data/src/core/feature_registry.rs");
    require_all(
        &cpu,
        &[
            "pub const FOOTPRINT_SEMANTIC_VERSION: u32 = 2;",
            "pub const FOOTPRINT_FEATURE_NAMES: [&str; 7]",
            "pub const FOOTPRINT_CPU_ORACLE_AUTHORITY_V2: &str",
            "const W: usize = 96;",
            "const CORR_W: usize = 48;",
            "const DELTA_W: usize = 24;",
            "const EPS: f64 = 1e-12;",
            "FeatureCellValidity::Warmup",
            "FeatureCellValidity::MissingInput",
            "FeatureCellValidity::ZeroDenominator",
            "validate_canonical_millisecond_timestamps(timestamps)?",
        ],
    );
    let names = section(&cpu, "pub const FOOTPRINT_FEATURE_NAMES: [&str; 7]", "];");
    for name in [
        "fp_volume_z",
        "fp_absorption",
        "fp_effort_result_div",
        "fp_climax",
        "fp_delta_proxy",
        "fp_volprice_corr",
        "fp_fix_window",
    ] {
        assert!(names.contains(name), "CPU oracle omitted {name}");
    }
    assert_eq!(names.matches("\"fp_").count(), 7);
    assert!(registry.contains("super::footprint_features::FOOTPRINT_SEMANTIC_VERSION"));
}

#[test]
fn native_cuda_emits_the_complete_seven_column_family_without_host_values() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_footprint_v2.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_footprint_v2.cu");
    require_all(
        &runtime,
        &[
            "pub const RESIDENT_FOOTPRINT_COLUMN_NAMES_V2: [&str; 7]",
            "pub fn resident_footprint_capability_v2()",
            "pub(crate) fn launch_resident_footprint_v2(",
            "ResidentParentDatasetSourceV3",
            "ResidentProducerReadyEventV3::record(",
            "unsafe impl ResidentF64FeatureBatchV3 for ResidentFootprintFeatureBatchV2",
            "retained_feature_device_bytes",
            "prefix_scratch_device_bytes",
            "parent_input_h2d_bytes: 0",
            "feature_value_d2h_bytes: 0",
            "producer_ready_event_count: 1",
            "native_launch_count: 2",
        ],
    );
    require_all(
        &native,
        &[
            "neoethos_resident_footprint_f64_v2",
            "resident_footprint_prefix_v2",
            "resident_footprint_features_v2",
            "kFootprintColumnsV2 = 7",
            "kPrefixSeriesV2 = 8",
            "kRollingWindowV2 = 96",
            "kCorrelationWindowV2 = 48",
            "kDeltaWindowV2 = 24",
            "kWarmupV2 = 1",
            "kZeroDenominatorV2 = 5",
            "canonical_nan_v2",
        ],
    );
    for forbidden in [
        "compute_footprint_feature_columns",
        "FeatureFrame",
        "copy_to(",
        "stream.synchronize()",
        "Context::new(",
        "Stream::new(",
        "fallback_",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "resident Footprint reintroduced forbidden host seam {forbidden:?}"
        );
    }
}

#[test]
fn runtime_receipt_is_exact_same_carrier_allocation_and_event_evidence() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_footprint_v2.rs");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let compact = normalized(&runtime);
    require_all(
        &compact,
        &[
            "feature_cells.checked_mul(std::mem::size_of::<f64>()+std::mem::size_of::<u8>())",
            "rows.checked_add(1)",
            ".checked_mul(FOOTPRINT_PREFIX_SERIES_V2)",
            ".checked_mul(std::mem::size_of::<f64>())",
            "parent.producer_context().as_raw()!=context.as_raw()",
            "parent.producer_stream().as_inner()!=stream.as_inner()",
            "parent.device_ordinal()!=device_ordinal",
            "parent.producer_ready_event().wait_before_read(",
        ],
    );
    require_all(
        &store,
        &[
            "pub fn append_resident_footprint_v2(",
            "launch_resident_footprint_v2(run_device, parent, bindings)?",
            "footprint_runtime_receipt_v2",
        ],
    );
}

#[test]
fn data_owns_move_only_footprint_preflight_lifetime_and_runtime_validation() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let compact = normalized(&data);
    require_all(
        &data,
        &[
            "pub(crate) struct FootprintAllocationReceiptV2",
            "pub(crate) struct FootprintLifetimeReceiptV2",
            "pub(crate) struct SealedFootprintComponentReceiptV2",
            "fn seal_footprint_component_receipt_v2(",
            "fn validate_runtime_evidence(",
            "ResidentFootprintRuntimeReceiptV2",
            "FootprintScratchLifetimeV2::ThroughProducerPackReadyEvent",
            "FootprintOutputLifetimeV2::ThroughProducerPackReadyEvent",
        ],
    );
    require_all(
        &compact,
        &[
            "footprint.validate_working_set(&working_set)?",
            "footprint.validate_runtime_evidence(",
            "footprint:seal_token.footprint,",
        ],
    );
    for forbidden in [
        "impl Clone for SealedFootprintComponentReceiptV2",
        "from_raw_footprint",
        "from_bytes_footprint",
        "from_hash_footprint",
        "from_sha256_footprint",
    ] {
        assert!(
            !data.contains(forbidden),
            "Footprint authority is caller-rehydratable via {forbidden:?}"
        );
    }
}

#[test]
fn capability_census_advances_to_five_of_ten_only_after_receipt_wiring() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let preflight =
        read("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let census = section(
        &data,
        "fn current_resident_producer_capabilities_v3()",
        "\n}",
    );
    require_all(
        census,
        &[
            "resident_classic_ta_capability_v3()?",
            "resident_smc_capability_v3()?",
            "resident_footprint_capability_v2()?",
            "resident_canonical_content_sha256_capability_v3()?",
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
    assert!(!pending.contains("ResidentFeatureProducerV3::Footprint"));
}

#[test]
fn cuda_translation_unit_is_build_linked_and_full_workspace_stays_red() {
    let build = read("crates/neoethos-gpu-cuda/build.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");
    let red = read(
        "crates/neoethos-data/tests/gpu_only_feature_workspace_preflight_v3_source_contract.rs",
    );
    assert!(build.contains("native/resident_footprint_v2.cu"));
    assert!(library.contains("#[cfg(feature = \"cuda\")]\npub mod resident_footprint_v2;"));
    assert!(red.contains("red_data_component_receipt_must_consume_the_preflight"));
    assert!(red.contains("#[ignore = \"RED: complete producers"));
    assert!(red.contains("footprint: SealedFootprintComponentReceiptV2"));
}
