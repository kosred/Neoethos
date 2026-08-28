use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read_required_source(relative: &str) -> String {
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
            "full-discovery GPU authority is missing {token:?}"
        );
    }
}

fn require_before(source: &str, earlier: &str, later: &str) {
    let earlier_at = source
        .find(earlier)
        .unwrap_or_else(|| panic!("missing earlier source boundary {earlier:?}"));
    let later_at = source
        .find(later)
        .unwrap_or_else(|| panic!("missing later source boundary {later:?}"));
    assert!(
        earlier_at < later_at,
        "{earlier:?} must execute before {later:?}"
    );
}

#[test]
fn canonical_full_discovery_stage_set_is_exactly_the_sixteen_named_stages() {
    let capability_source = read_required_source("src/gpu_native/capability.rs");
    let authority_source = read_required_source("src/full_discovery_gpu_stage_authority_v2.rs");
    let full_discovery = section(
        &capability_source,
        "pub const FULL_DISCOVERY: [Self; 16] = [",
        "];",
    );
    let stages = [
        "FeaturePreparation",
        "GaGenerationSelection",
        "PopulationEvaluation",
        "SignalAndMinTradeFilter",
        "QualityScreen",
        "MonteCarlo",
        "PropFirmWindow",
        "CandidateCorrelation",
        "WalkForward",
        "Cpcv",
        "Pbo",
        "RobustnessPermutationPlateau",
        "RiskDiagnostics",
        "CanonicalReplay",
        "ForwardTailReplay",
        "SurvivorRanking",
    ];

    assert_eq!(
        full_discovery.matches("Self::").count(),
        stages.len(),
        "FULL_DISCOVERY must contain exactly sixteen entries"
    );
    for stage in stages {
        assert_eq!(
            full_discovery.matches(&format!("Self::{stage}")).count(),
            1,
            "FULL_DISCOVERY must contain {stage} exactly once"
        );
    }

    require_all(
        &authority_source,
        &[
            "PipelineStage::FULL_DISCOVERY",
            "FULL_DISCOVERY_STAGE_COUNT_V2",
            "FullDiscoveryStageCountMismatch",
        ],
    );
}

#[test]
fn permit_is_opaque_and_binds_exact_device_build_input_plan_and_stage_identities() {
    let authority_source = read_required_source("src/full_discovery_gpu_stage_authority_v2.rs");
    let permit = section(
        &authority_source,
        "pub struct StrictGpuOnlyFullDiscoveryPermitV2 {",
        "\n}",
    );
    require_all(
        permit,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "canonical_search_input_receipt_sha256: String",
            "resident_input_content_sha256: String",
            "resolved_search_config_sha256: String",
            "full_discovery_stage_manifest_sha256: String",
            "permit_identity_sha256: String",
        ],
    );
    assert!(
        !permit.contains("pub "),
        "permit fields must remain private and non-caller-mintable"
    );

    for forbidden in [
        "Deserialize",
        "impl Default for StrictGpuOnlyFullDiscoveryPermitV2",
        "pub fn new(",
        "pub fn from_",
        "From<",
        "unsafe",
    ] {
        assert!(
            !authority_source.contains(forbidden),
            "opaque permit exposes forbidden construction path {forbidden:?}"
        );
    }
}

#[test]
fn acquisition_consumes_sealed_authority_instead_of_caller_supplied_facts() {
    let authority_source = read_required_source("src/full_discovery_gpu_stage_authority_v2.rs");
    let acquisition = section(
        &authority_source,
        "pub fn acquire_strict_gpu_only_full_discovery_permit_v2(",
        ") -> Result<StrictGpuOnlyFullDiscoveryPermitV2",
    );
    require_all(
        acquisition,
        &[
            "sealed_resident_input_authority:",
            "resolved_full_discovery_plan_authority:",
        ],
    );
    for forbidden in [
        "card_present: bool",
        "all_stages_strict: bool",
        "allow_cpu: bool",
        "selected_cuda_ordinal: u32",
        "cuda_build_manifest_sha256: &str",
        "canonical_search_input_receipt_sha256: &str",
        "resident_input_content_sha256: &str",
        "stage_manifest_sha256: &str",
    ] {
        assert!(
            !acquisition.contains(forbidden),
            "caller can supply authority fact via {forbidden:?}"
        );
    }

    require_all(
        &authority_source,
        &[
            "bind_exact_ordinal_device_build_and_input_v2(",
            "validate_resolved_plan_against_resident_input_v2(",
            "compute_full_discovery_stage_manifest_identity_v2(",
            "compute_strict_gpu_only_full_discovery_permit_identity_v2(",
        ],
    );
}

#[test]
fn card_present_execution_rejects_cpu_allow_cpu_and_untyped_preference_escapes() {
    let authority_source = read_required_source("src/full_discovery_gpu_stage_authority_v2.rs");
    require_all(
        &authority_source,
        &[
            "CardPresentCpuExecutionForbidden",
            "CardPresentAllowCpuForbidden",
            "ExactCudaOrdinalRequired",
            "reject_card_present_cpu_or_fallback_v2(",
            "FallbackPolicy::AllowCpu",
        ],
    );
    for forbidden in [
        "cpu_forced",
        "cpu-forced",
        "GPU_PREFERRED",
        "gpu_pipeline_preflight(",
        "stage1_baseline()",
        "unwrap_or_default",
    ] {
        assert!(
            !authority_source.contains(forbidden),
            "strict authority retains forbidden escape {forbidden:?}"
        );
    }
}

#[test]
fn every_stage_must_be_strict_before_the_permit_can_be_constructed() {
    let authority_source = read_required_source("src/full_discovery_gpu_stage_authority_v2.rs");
    require_all(
        &authority_source,
        &[
            "require_all_full_discovery_stages_strict_gpu_v2(",
            "StageGpuCapability::StrictGpu",
            "StageGpuCapability::HybridOnly",
            "StageGpuCapability::CpuOnly",
            "StageGpuCapability::Unsupported",
            "MissingStageCapability",
            "NonStrictFullDiscoveryStage",
        ],
    );

    let acquisition = section(
        &authority_source,
        "pub fn acquire_strict_gpu_only_full_discovery_permit_v2(",
        "\n}",
    );
    require_before(
        acquisition,
        "reject_card_present_cpu_or_fallback_v2(",
        "require_all_full_discovery_stages_strict_gpu_v2(",
    );
    require_before(
        acquisition,
        "require_all_full_discovery_stages_strict_gpu_v2(",
        "StrictGpuOnlyFullDiscoveryPermitV2 {",
    );
}

#[test]
fn card_present_strict_gpu_sizing_never_routes_a_small_fit_toward_cpu() {
    let adapter = read_required_source("src/gpu_native/prototype_b_population_eval.rs");
    let sizing = section(
        &adapter,
        "fn candidates_for_free_memory(",
        "\n}\n\n/// The `max_events`",
    );
    for forbidden in [
        "the CPU lane is the honest",
        "if fits < 16 {",
        "return Sizing::NoRoom",
    ] {
        assert!(
            !sizing.contains(forbidden),
            "card-present strict GPU sizing retains CPU-oriented tiny-fit path {forbidden:?}"
        );
    }
    require_all(
        sizing,
        &[
            "StrictGpuMinimumBatchNotResident",
            "selected_cuda_ordinal",
            "cuda_build_manifest_sha256",
        ],
    );
}
