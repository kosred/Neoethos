use std::fs;
use std::path::{Path, PathBuf};

fn collect_rust_sources(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            collect_rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn active_production_sources_have_no_v1_core_receipt_alias_or_stale_outer_wrapper() {
    let crates_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("search crate has crates parent");
    let mut sources = Vec::new();
    for crate_name in [
        "neoethos-search",
        "neoethos-cli",
        "neoethos-app",
        "neoethos-autoresearch",
        "neoethos-broker-truth-acquire",
    ] {
        collect_rust_sources(&crates_root.join(crate_name).join("src"), &mut sources);
    }
    sources.sort();

    let retired = [
        "CanonicalSearchInputReceiptV1",
        "CanonicalSearchRunInputV1",
        "CanonicalSearchArtifactScopeV1",
        "CanonicalSearchArtifactEnvelopeV1",
        "PromotionSummaryAuthorityPayloadV2",
        "PROMOTION_SUMMARY_ARTIFACT_KIND_V2",
        "CanonicalTrendbarResearchExecutionContractV2",
        "CanonicalTrendbarResearchDiscoveryResultV2",
        "PromotionBatchBindingV4",
        "HistoricalResearchArtifactV1",
        "HistoricalResearchRequestV1",
        "HistoricalCandidateScanRequestV1",
        "HistoricalCandidateScanContractV1",
        "HistoricalCandidateScanResultV1",
        "HistoricalCandidateResultV1",
        "DiscoveryValidationSnapshotManifestV1",
        "ValidatedDiscoveryValidationSnapshotV1",
        "StoredPromotionSummaryAuthorityV2",
        "strict v2 live portfolio",
        "strict v2 portfolio",
        "promotion_summary_v2",
        "model_targets v2",
        "discovery_ledger.v2.json",
        "trial_returns.v2.json",
        "neoethos.live-portfolio-authority.v2",
        "neoethos.autoresearch.promotion_evidence.v4",
        "strict v2 promotion gate",
        "composite-v2",
    ];
    for path in sources {
        let source = fs::read_to_string(&path).expect("read Rust production source");
        for token in retired {
            assert!(
                !source.contains(token),
                "active production source {} retains stale receipt wrapper `{token}`",
                path.display()
            );
        }
    }
}

#[test]
fn canonical_feature_execution_authority_uses_production_dispatch_and_strict_gpu_lane() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("neoethos-data")
            .join("src")
            .join("core")
            .join("hpc_ta.rs"),
    )
    .expect("read canonical feature execution authority source");

    for required in [
        "pub fn resolved_canonical_feature_execution_authority_v1(",
        "let policy = resolved_indicator_compute_policy();",
        "match detect_best_kernel()",
        "Kernel::Scalar => ResolvedCanonicalFeatureMathLaneV1::CpuScalar",
        "Kernel::Avx2 => ResolvedCanonicalFeatureMathLaneV1::CpuAvx2Fma",
        "ResolvedCanonicalFeatureMathLaneV1::CpuAvx512F64Avx2FmaDqVlBw",
        "ResolvedCanonicalFeatureMathLaneV1::GpuCudaF64Strict",
        "VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1",
        "VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1",
    ] {
        assert!(
            source.contains(required),
            "canonical feature execution authority is missing production dispatch fact `{required}`"
        );
    }
    for forbidden in [
        "is_x86_feature_detected!(\"avx2\")",
        "is_x86_feature_detected!(\"avx512f\")",
        "GpuOnly => (ResolvedCanonicalFeatureMathLaneV1::Cpu",
    ] {
        assert!(
            !source.contains(forbidden),
            "canonical receipt authority reimplemented or weakened production dispatch via `{forbidden}`"
        );
    }
}

#[test]
fn canonical_feature_policy_is_frozen_before_bits_and_conflicts_fail_closed() {
    let crates_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("search crate has crates parent");
    let data_source = fs::read_to_string(
        crates_root
            .join("neoethos-data")
            .join("src")
            .join("core")
            .join("hpc_ta.rs"),
    )
    .expect("read indicator policy authority source");
    let backend_source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("backend.rs"),
    )
    .expect("read search backend source");

    assert!(
        data_source.contains("*POLICY_OVERRIDE.get_or_init(|| IndicatorComputePolicy::Auto)"),
        "the first canonical feature-policy read must freeze Auto before feature bits are built"
    );
    for required in [
        "BackendConfigError::IndicatorPolicyAlreadyResolved",
        "Err(active) if active == policy",
        "Err(active) =>",
    ] {
        assert!(
            backend_source.contains(required),
            "backend install must fail closed on a different already-selected feature lane: \
             missing `{required}`"
        );
    }
    let set_position = backend_source
        .find("set_indicator_compute_policy(policy)")
        .expect("backend installs the canonical feature policy");
    let backend_position = backend_source
        .find("install_evaluation_backend(backend)?")
        .expect("backend installs the evaluation policy");
    assert!(
        set_position < backend_position,
        "a conflicting canonical feature policy must be refused before mutating the search backend"
    );
    assert!(
        !backend_source
            .contains("let _ = neoethos_data::core::hpc_ta::set_indicator_compute_policy(policy)"),
        "backend installation still discards a conflicting canonical feature-policy result"
    );
}

#[test]
fn canonical_search_input_v2_binds_exact_content_and_runtime_math_authority() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("data_selection.rs"),
    )
    .expect("read canonical search receipt source");

    for required in [
        "pub struct CanonicalSearchInputReceiptV2",
        "b\"neoethos.canonical-search-input-receipt.v2\\0\"",
        "pub struct CanonicalSearchRunInputV2<'a>",
        "pub struct CanonicalSearchArtifactScopeV2",
        "b\"neoethos.canonical-search-artifact-scope.v2\\0\"",
        "pub struct CanonicalSearchArtifactEnvelopeV2",
        "feature_content_sha256: String",
        "feature_execution: CanonicalFeatureExecutionReceiptV1",
        "pub struct CanonicalFeatureExecutionReceiptV1",
        "vector_ta_math_authority: String",
        "selected_lane: CanonicalFeatureMathLaneV1",
        "timestamp.to_le_bytes()",
        "name.as_bytes()",
        "value.to_bits().to_le_bytes()",
        "validity.code()",
        "resolved_canonical_feature_execution_authority_v1()",
    ] {
        assert!(
            source.contains(required),
            "V2 receipt is missing exact content/math contract `{required}`"
        );
    }
    for retired in [
        "pub struct CanonicalSearchInputReceiptV1",
        "type CanonicalSearchInputReceiptV1",
        "b\"neoethos.canonical-search-input-receipt.v1\\0\"",
        "pub struct CanonicalSearchRunInputV1",
        "type CanonicalSearchRunInputV1",
        "pub struct CanonicalSearchArtifactScopeV1",
        "type CanonicalSearchArtifactScopeV1",
        "b\"neoethos.canonical-search-artifact-scope.v1\\0\"",
        "pub struct CanonicalSearchArtifactEnvelopeV1",
        "type CanonicalSearchArtifactEnvelopeV1",
    ] {
        assert!(
            !source.contains(retired),
            "retired V1 receipt compatibility path remains active: `{retired}`"
        );
    }
}

#[test]
fn canonical_search_receipt_is_strict_segment_complete_and_content_addressed() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("data_selection.rs"),
    )
    .expect("read canonical search receipt source");

    for required in [
        "#[serde(deny_unknown_fields)]",
        "segments: Vec<CanonicalSearchSourceSegmentReceiptV1>",
        "pub fn validate_against(",
        "pub fn identity_sha256(",
        "pub fn to_json_bytes(",
        "pub fn from_json_bytes(",
        "pub struct CanonicalSearchRunInputV2<'a>",
        "pub fn new(\n        receipt: CanonicalSearchInputReceiptV2,",
        "pub struct CanonicalSearchArtifactScopeV2",
        "pub struct CanonicalSearchArtifactEnvelopeV2",
        "search_config_hash: String",
        "pub fn search_config_hash(&self) -> &str",
        "pub struct CanonicalSearchEvaluatedWindowV1",
        "pub enum CanonicalSearchWindowRoleV1",
        "pub fn validate_against_receipt(",
        "base_frame: &'a CanonicalOhlcvFrame",
    ] {
        assert!(
            source.contains(required),
            "canonical search receipt is missing strict contract `{required}`"
        );
    }
    assert!(
        !source.contains(
            "pub fn new(\n        receipt: CanonicalSearchInputReceiptV2,\n        features: &'a FeatureFrame,\n        ohlcv: &'a Ohlcv"
        ),
        "production run input still accepts mutable receipt-free OHLCV values"
    );
}

#[test]
fn public_discovery_boundary_and_result_cannot_drop_the_exact_receipt() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("discovery.rs"),
    )
    .expect("read discovery source");

    assert!(
        source.contains("pub search_input_receipt: CanonicalSearchInputReceiptV2"),
        "DiscoveryResult does not own the exact input receipt"
    );
    assert!(
        source.contains("pub search_config_hash: String"),
        "DiscoveryResult does not own the exact resolved search-config identity"
    );
    assert!(
        source
            .matches("input: &CanonicalSearchRunInputV2<'_>")
            .count()
            >= 4,
        "every public discovery-cycle variant must accept the typed receipt-bound input"
    );
    for retired in [
        "pub fn run_discovery_cycle(\n    features: &FeatureFrame",
        "pub fn run_discovery_cycle_with_progress<F>(\n    features: &FeatureFrame",
        "pub fn run_discovery_cycle_with_holdout(\n    features: &FeatureFrame",
        "pub fn run_discovery_cycle_with_holdout_and_progress<F>(\n    features: &FeatureFrame",
    ] {
        assert!(
            !source.contains(retired),
            "public discovery still exposes receipt-free input `{retired}`"
        );
    }
    assert!(
        source
            .matches("CanonicalSearchArtifactEnvelopeV2::new(")
            .count()
            >= 2,
        "portfolio and promotion-summary writers must embed their receipt/window authority"
    );
    assert!(
        !source.contains("write_json_atomic(path, &exports)"),
        "portfolio writer still persists a receipt-free raw array"
    );
}

#[test]
fn live_portfolio_is_a_strict_v3_artifact_bound_to_receipt_and_config() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("live_portfolio.rs"),
    )
    .expect("read live portfolio source");

    for required in [
        "pub const LIVE_PORTFOLIO_SCHEMA_VERSION: u32 = 3;",
        "#[serde(deny_unknown_fields)]",
        "pub search_scope: CanonicalSearchArtifactScopeV2",
        "pub search_config_hash: String",
        "pub fn validate(&self) -> anyhow::Result<()>",
        "artifact.validate()?;",
    ] {
        assert!(
            source.contains(required),
            "live portfolio is missing strict receipt/config contract `{required}`"
        );
    }
    assert!(
        !source.contains("#[serde(default)]\n    pub cost_band"),
        "live portfolio still silently accepts the receipt-free legacy schema"
    );
}

#[test]
fn receipt_bound_cache_files_move_to_v3_without_legacy_current_fallback() {
    let search_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let ledger =
        fs::read_to_string(search_src.join("discovery_ledger.rs")).expect("read discovery ledger");
    let returns =
        fs::read_to_string(search_src.join("trial_returns.rs")).expect("read trial returns");

    for (name, source, schema, file) in [
        (
            "discovery ledger",
            ledger.as_str(),
            "neoethos.discovery_search_ledger.v3",
            "discovery_ledger.v3.json",
        ),
        (
            "trial returns",
            returns.as_str(),
            "neoethos.trial_returns.v3",
            "trial_returns.v3.json",
        ),
    ] {
        assert!(
            source.contains(schema) && source.contains(file),
            "{name} did not migrate its receipt-owning schema/path to V3"
        );
        let legacy_schema = schema.replace(".v3", ".v2");
        let legacy_file = file.replace(".v3", ".v2");
        assert!(
            !source.contains(legacy_schema.as_str()) && !source.contains(legacy_file.as_str()),
            "{name} retains a silent V2 cache alias/fallback"
        );
    }
}
