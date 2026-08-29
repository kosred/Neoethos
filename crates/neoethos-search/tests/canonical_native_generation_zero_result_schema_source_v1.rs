fn assert_field_census_v1(source: &str, declaration: &str, fields: &[&str]) {
    let body = source
        .split_once(declaration)
        .unwrap_or_else(|| panic!("missing wire declaration {declaration}"))
        .1
        .split_once("\n}")
        .unwrap_or_else(|| panic!("unterminated wire declaration {declaration}"))
        .0;
    let mut cursor = 0;
    for field in fields {
        let offset = body[cursor..]
            .find(field)
            .unwrap_or_else(|| panic!("missing or reordered field {field} in {declaration}"));
        cursor += offset + field.len();
    }
    let actual_count = body
        .lines()
        .map(str::trim)
        .filter(|line| line.ends_with(',') && line.contains(": "))
        .count();
    assert_eq!(
        actual_count,
        fields.len(),
        "field census drifted in {declaration}"
    );
}

fn assert_no_field_level_serde_attributes_v1(source: &str, declaration: &str) {
    let body = source
        .split_once(declaration)
        .unwrap_or_else(|| panic!("missing wire declaration {declaration}"))
        .1
        .split_once("\n}")
        .unwrap_or_else(|| panic!("unterminated wire declaration {declaration}"))
        .0;
    assert!(
        !body.contains("#[serde("),
        "field-level serde attribute drifted in {declaration}"
    );
}

fn sealed_view_clone_copy_guard_v1(sealed: &str) -> Result<(), &'static str> {
    let view_declaration = "struct CanonicalNativeGenerationZeroResearchResultViewV1<'a> {";
    let view_prefix = sealed
        .split_once(view_declaration)
        .ok_or("sealed view declaration is missing")?
        .0;
    let contiguous_attributes = view_prefix
        .rsplit_once("\n}")
        .map_or(view_prefix, |(_, tail)| tail);
    for derive in contiguous_attributes
        .lines()
        .filter(|line| line.trim_start().starts_with("#[derive("))
    {
        if derive.contains("Clone") || derive.contains("Copy") {
            return Err("sealed view derives Clone or Copy through a multi-trait derive");
        }
    }
    let compact_source: String = sealed
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect();
    if compact_source.contains("CloneforCanonicalNativeGenerationZeroResearchResultViewV1")
        || compact_source.contains("CopyforCanonicalNativeGenerationZeroResearchResultViewV1")
    {
        return Err("sealed view has a manual Clone or Copy implementation");
    }
    Ok(())
}

#[test]
fn result_source_separates_pre_v5_admission_from_post_run_sealing() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let preflight = source
        .split_once("// BEGIN CANONICAL_NATIVE_GEN0_PREFLIGHT_V1")
        .expect("pre-V5 result admission region")
        .1
        .split_once("// END CANONICAL_NATIVE_GEN0_PREFLIGHT_V1")
        .expect("pre-V5 result admission end")
        .0;
    assert!(
        !source.contains("#![allow(dead_code)]"),
        "2A2 must consume the private size planner without a module dead-code waiver"
    );
    for required in [
        "CanonicalNativeDiscoveryRequestV1",
        "prepared_feature_count",
        "CanonicalNativeGenerationZeroResultSizePlanV1",
        "CanonicalNativeGenerationZeroFixedMetadataShapeV1",
        "RESIDENT_POPULATION_SIZING_RECEIPT_V2_JSON_UPPER_BOUND_BYTES_V1",
        "checked_native_v3_receipt_json_upper_bound_bytes_v1",
        "EMPTY_POPULATION_ARRAY_REPLACEMENT_BYTES_V1",
        "COST_BAND_OPTION_JSON_UPPER_BOUND_BYTES_V1",
        "ADAPTIVE_TOKEN_OPTION_JSON_UPPER_BOUND_BYTES_V1",
        "EVIDENCE_IDENTITY_JSON_STRING_BYTES_V1",
    ] {
        assert!(preflight.contains(required), "preflight omits {required}");
    }
    for forbidden in [
        "CanonicalGpuResidentSearchInputReceiptV3",
        "ResidentPopulationAutoSizingReceiptV2",
        "ResidentGenerationZeroMilestoneV1",
        "SearchResult",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "pre-V5 Pcap depends on post-run authority {forbidden}"
        );
    }
    assert!(
        preflight.contains("pub(crate) fn preflight_canonical_native_generation_zero_result_v1")
    );
    let high_level_preflight_signature = preflight
        .split_once("pub(crate) fn preflight_canonical_native_generation_zero_result_v1(")
        .expect("high-level preflight signature")
        .1
        .split_once(") ->")
        .expect("high-level preflight return type")
        .0;
    let compact_preflight_signature: String = high_level_preflight_signature
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect();
    assert_eq!(
        compact_preflight_signature,
        "request:&CanonicalNativeDiscoveryRequestV1,prepared_feature_count:usize,",
        "high-level preflight ownership/order drifted"
    );
    assert!(preflight.contains("pub(crate) struct CanonicalNativeGenerationZeroResultPreflightV1"));
    assert!(!preflight.contains("pub struct CanonicalNativeGenerationZeroResultPreflightV1"));
    assert!(preflight.contains("struct CanonicalNativeGenerationZeroFixedMetadataShapeV1 {"));
    assert!(
        !preflight.contains("pub(crate) struct CanonicalNativeGenerationZeroFixedMetadataShapeV1")
    );
    assert!(!preflight.contains("pub struct CanonicalNativeGenerationZeroFixedMetadataShapeV1"));
    assert!(preflight.contains("fn checked_preflight_from_fixed_metadata_shape_v1("));
    assert!(!preflight.contains("pub(crate) fn checked_preflight_from_fixed_metadata_shape_v1"));
    assert!(!preflight.contains("pub fn checked_preflight_from_fixed_metadata_shape_v1"));

    let root = include_str!("../src/lib.rs");
    let result_exports = root
        .split_once("pub use canonical_native_generation_zero_result_v1::{")
        .expect("result export block")
        .1
        .split_once("};")
        .expect("result export block end")
        .0;
    let exported: Vec<_> = result_exports
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    assert_eq!(
        exported,
        [
            "CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1",
            "CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1",
        ],
        "schema/version must remain the complete public result-module export surface"
    );
}

#[test]
fn sealed_view_clone_copy_guard_rejects_attached_derive_orders_and_manual_impls() {
    let valid = "fn prior() {}\n#[derive(Debug)]\npub(crate) struct CanonicalNativeGenerationZeroResearchResultViewV1<'a> {\n    milestone: &'a (),\n}\n";
    sealed_view_clone_copy_guard_v1(valid).unwrap();
    for invalid in [
        valid.replace("#[derive(Debug)]", "#[derive(Debug, Clone)]"),
        valid.replace("#[derive(Debug)]", "#[derive(Clone, Debug)]"),
        valid.replace("#[derive(Debug)]", "#[derive(Debug, Copy)]"),
        format!(
            "{valid}impl<'a> Clone for CanonicalNativeGenerationZeroResearchResultViewV1<'a> {{ fn clone(&self) -> Self {{ unreachable!() }} }}"
        ),
        format!(
            "{valid}impl<'a> Copy for CanonicalNativeGenerationZeroResearchResultViewV1<'a> {{}}"
        ),
    ] {
        assert!(sealed_view_clone_copy_guard_v1(&invalid).is_err());
    }
}

#[test]
fn sealed_view_is_borrow_only_for_population_payload_and_validates_dual_receipts() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let sealed = source
        .split_once("// BEGIN CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1")
        .expect("sealed-result region")
        .1
        .split_once("// END CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1")
        .expect("sealed-result end")
        .0;
    for required in [
        "CanonicalNativeGenerationZeroResearchResultViewV1<'a>",
        "seal_canonical_native_generation_zero_research_result_v1",
        "checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1",
        "&'a ResidentGenerationZeroMilestoneV1",
        "CanonicalGpuResidentSearchInputReceiptV3",
        "ResidentPopulationAutoSizingReceiptV2",
        "CanonicalTrendbarResearchExecutionContractV3",
        "EvaluationConfig",
        "validate_self_v2",
        "validate_financial_authority_against_pinned_source_projection_v2",
        "checked_from_session_extents_v1",
        "validate_population_payload_v1",
        "validate_execution_facts_v1",
        "RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2",
        "ga_fitness",
        "ga_fitness_growth",
        "EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1",
    ] {
        assert!(sealed.contains(required), "sealed view omits {required}");
    }
    for forbidden in [
        "search_result.clone",
        "genes.clone",
        "metrics.clone",
        "milestone.search_result().clone",
        "milestone.search_result().to_owned",
        "SearchResult::clone",
        "search_result: SearchResult",
        ".cloned()",
        ".genes.to_vec",
        ".metrics.to_vec",
        "metrics_receipt_identities_sha256().clone",
        "metrics_receipt_identities_sha256().to_vec",
        "collect::<Vec",
        "collect::<Vec<String>>",
        "collect::<Vec<Gene>>",
        "collect::<Vec<[f64; 11]>>",
        "Vec::from(",
        "extend_from_slice",
        "into_boxed_slice",
        "Box::from(",
        "Arc::from(",
        "serde_json::Value",
        "serde_json::to_value",
        "serde_json::to_vec",
        "serde_json::to_string",
        "to_value",
        "to_vec",
        "to_string",
        "Vec<u8>",
        "CanonicalTrendbarResearchDiscoveryResultV3",
        "DiscoveryResult",
    ] {
        assert!(
            !sealed.contains(forbidden),
            "sealed view contains forbidden payload authority {forbidden}"
        );
    }
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroResearchResultViewV1<'a> {",
        &[
            "preflight: CanonicalNativeGenerationZeroResultPreflightV1,",
            "request_evidence: CanonicalNativeGenerationZeroRequestEvidenceV1,",
            "financial_execution_contract_v3: CanonicalTrendbarResearchExecutionContractV3,",
            "native_input_receipt_v3: CanonicalGpuResidentSearchInputReceiptV3,",
            "population_sizing_receipt_v2: ResidentPopulationAutoSizingReceiptV2,",
            "evaluation_evidence_v1: CanonicalNativeGenerationZeroEvaluationEvidenceV1,",
            "milestone: &'a ResidentGenerationZeroMilestoneV1,",
            "evidence_identity_sha256: String,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroRequestEvidenceV1 {",
        &[
            "execution_scope: CanonicalNativeExecutionScopeV1,",
            "artifact_class: HistoricalResearchArtifactClassV1,",
            "promotion_eligibility: HistoricalResearchPromotionEligibilityV1,",
            "authorization_issued: bool,",
            "contract_artifact_reference_schema: String,",
            "contract_artifact_reference_version: u16,",
            "contract_artifact_relative_path: String,",
            "contract_artifact_expected_sha256: String,",
            "contract_artifact_exact_file_sha256: String,",
            "contract_artifact_exact_file_byte_count: u64,",
            "contract_domain_identity_sha256: String,",
            "startup_settings_id: String,",
            "runtime_install_receipt_id: String,",
            "generation_zero_runtime_authority_id: String,",
            "unused_full_search_scope_id: String,",
            "raw_generations: usize,",
            "clamped_generations: usize,",
            "cost_band_status: CanonicalNativeCostBandStatusV1,",
            "cost_band: Option<(f64, f64)>,",
            "configured_population_cap: usize,",
            "resolved_population_cap: usize,",
            "term_cap: usize,",
            "string_bytes_cap: usize,",
            "vector_elements_cap: usize,",
            "source_count_cap: usize,",
            "result_bytes_cap: u64,",
        ],
    );
    assert!(
        sealed.contains("pub(crate) struct CanonicalNativeGenerationZeroResearchResultViewV1<'a>")
    );
    assert!(source.contains("pub(crate) struct CanonicalNativeGenerationZeroCompactJsonSealV1"));
    assert!(
        source.contains("pub(crate) enum CanonicalNativeGenerationZeroResultErrorV1")
            || source.contains("pub(crate) struct CanonicalNativeGenerationZeroResultErrorV1")
    );
    assert!(
        sealed.contains("pub(crate) fn seal_canonical_native_generation_zero_research_result_v1")
    );
    assert!(sealed.contains(
        "pub(crate) fn checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1"
    ));
    let high_level_signature_tail = sealed
        .split_once("pub(crate) fn seal_canonical_native_generation_zero_research_result_v1<'a>(")
        .expect("high-level result sealer signature")
        .1;
    let (high_level_signature, high_level_return) = high_level_signature_tail
        .split_once(") ->")
        .expect("high-level result sealer return type");
    let compact_signature: String = high_level_signature
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect();
    assert_eq!(
        compact_signature,
        "request:&CanonicalNativeDiscoveryRequestV1,preflight:CanonicalNativeGenerationZeroResultPreflightV1,financial_contract:CanonicalTrendbarResearchExecutionContractV3,native_receipt_v3:CanonicalGpuResidentSearchInputReceiptV3,sizing_receipt_v2:ResidentPopulationAutoSizingReceiptV2,evaluation_config:EvaluationConfig,milestone:&'aResidentGenerationZeroMilestoneV1,",
        "high-level sealer ownership/order drifted"
    );
    assert!(
        high_level_return
            .split_once('{')
            .expect("high-level result sealer body")
            .0
            .contains("CanonicalNativeGenerationZeroResearchResultViewV1<'a>")
    );
    assert!(!sealed.contains("pub struct CanonicalNativeGenerationZeroResearchResultViewV1<'a>"));
    assert!(!sealed.contains("pub fn seal_canonical_native_generation_zero_research_result_v1"));

    sealed_view_clone_copy_guard_v1(sealed).unwrap();

    for private_wire in [
        "IdentityMaterialV1<'a>",
        "ResultWireV1<'a>",
        "CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a>",
        "CanonicalNativeGenerationZeroContractArtifactWireV1<'a>",
        "CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a>",
        "CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a>",
        "CanonicalNativeGenerationZeroCostBandWireV1",
        "CanonicalNativeGenerationZeroLimitsWireV1",
        "CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a>",
        "CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a>",
        "CanonicalNativeGenerationZeroPopulationSizingWireV1<'a>",
        "CanonicalNativeGenerationZeroEvaluationSnapshotV1",
        "CanonicalNativeGenerationZeroEvaluationWireV1<'a>",
        "CanonicalNativeGenerationZeroCompletionWireV1",
        "CanonicalNativeGenerationZeroReplayWireV1",
    ] {
        assert!(
            !sealed.contains(&format!("pub struct {private_wire}"))
                && !sealed.contains(&format!("pub(crate) struct {private_wire}")),
            "wire helper leaked visibility: {private_wire}"
        );
    }

    for required in [
        "serde_json::Serializer",
        "Sha256",
        "CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_IDENTITY_DOMAIN_V1",
    ] {
        assert!(
            sealed.contains(required),
            "typed identity stream omits {required}"
        );
    }

    let validator = sealed
        .split_once("fn validate_result_native_receipt_v3(")
        .expect("native V3 result validator")
        .1
        .split_once("\n}")
        .expect("native V3 result validator end")
        .0;
    let compact_validator: String = validator
        .chars()
        .filter(|value| !value.is_whitespace())
        .collect();
    assert!(compact_validator.contains("receipt.source_bindings().len()"));
    assert!(compact_validator.contains("receipt.source_bindings().iter().try_fold("));
    assert!(compact_validator.contains("binding.segments().len()"));
    assert!(compact_validator.contains("checked_add"));
    assert!(compact_validator.contains("validate_native_v3_source_shape_counts_v1("));
    assert!(sealed.contains("fn validate_native_v3_source_shape_counts_v1("));
    assert!(!sealed.contains("pub(crate) fn validate_native_v3_source_shape_counts_v1("));

    for required in [
        "struct LowerHexSha256SliceV1<'a>(&'a [[u8; 32]]);",
        "impl Serialize for LowerHexSha256SliceV1<'_>",
        "serialize_seq",
        "[0_u8; 64]",
    ] {
        assert!(
            sealed.contains(required),
            "borrowed lowerhex receipt wire omits {required}"
        );
    }
}

#[test]
fn compact_wire_field_names_types_and_order_are_frozen_for_b_empty() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let sealed = source
        .split_once("// BEGIN CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1")
        .expect("sealed-result region")
        .1
        .split_once("// END CANONICAL_NATIVE_GEN0_SEALED_VIEW_V1")
        .expect("sealed-result end")
        .0;

    assert_field_census_v1(
        sealed,
        "struct IdentityMaterialV1<'a> {",
        &[
            "schema: &'static str,",
            "version: u16,",
            "scope: CanonicalNativeExecutionScopeV1,",
            "artifact_class: HistoricalResearchArtifactClassV1,",
            "promotion_eligibility: HistoricalResearchPromotionEligibilityV1,",
            "authorization_issued: bool,",
            "contract_artifact: CanonicalNativeGenerationZeroContractArtifactWireV1<'a>,",
            "runtime_authority: CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a>,",
            "unused_full_search: CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a>,",
            "cost_band_status: CanonicalNativeGenerationZeroCostBandWireV1,",
            "limits: CanonicalNativeGenerationZeroLimitsWireV1,",
            "financial_provenance_only: CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a>,",
            "evaluated_native_input: CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a>,",
            "population_sizing: CanonicalNativeGenerationZeroPopulationSizingWireV1<'a>,",
            "generation_zero_evaluation: CanonicalNativeGenerationZeroEvaluationWireV1<'a>,",
            "residency_counters: CanonicalNativeGenerationZeroResidencyCountersSnapshotV1,",
            "completion: CanonicalNativeGenerationZeroCompletionWireV1,",
            "replay: CanonicalNativeGenerationZeroReplayWireV1,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct ResultWireV1<'a> {",
        &[
            "identity_material: IdentityMaterialV1<'a>,",
            "evidence_identity_sha256: &'a str,",
        ],
    );
    assert!(sealed.contains("#[serde(flatten)]\n    identity_material:"));
    assert!(!sealed.contains("Deserialize"));
    assert_eq!(
        sealed.matches("#[serde(flatten)]").count(),
        1,
        "only ResultWireV1 may flatten IdentityMaterialV1"
    );
    let result_wire_body = sealed
        .split_once("struct ResultWireV1<'a> {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    let result_wire_field_attributes: Vec<_> = result_wire_body
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("#[serde("))
        .collect();
    assert_eq!(result_wire_field_attributes, ["#[serde(flatten)]"]);
    for forbidden in ["#[serde(skip", "#[serde(rename", "#[serde(untagged)]"] {
        assert!(
            !sealed.contains(forbidden),
            "wire schema contains forbidden serde drift {forbidden}"
        );
    }

    for declaration in [
        "struct IdentityMaterialV1<'a> {",
        "struct ResultWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroContractArtifactWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroLimitsWireV1 {",
        "struct CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroCostBandWireV1 {",
        "struct CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroPopulationSizingWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroEvaluationSnapshotV1 {",
        "struct CanonicalNativeGenerationZeroEvaluationWireV1<'a> {",
        "struct CanonicalNativeGenerationZeroCompletionWireV1 {",
        "struct CanonicalNativeGenerationZeroReplayWireV1 {",
    ] {
        if declaration != "struct ResultWireV1<'a> {" {
            assert_no_field_level_serde_attributes_v1(sealed, declaration);
        }
    }

    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a> {",
        &[
            "schema: &'static str,",
            "version: u16,",
            "relative_path: &'a str,",
            "expected_sha256: &'a str,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroContractArtifactWireV1<'a> {",
        &[
            "reference: CanonicalNativeGenerationZeroArtifactReferenceWireV1<'a>,",
            "exact_file_sha256: &'a str,",
            "exact_file_byte_count: u64,",
            "contract_domain_identity_sha256: &'a str,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroLimitsWireV1 {",
        &[
            "configured_population_cap: usize,",
            "resolved_population_cap: usize,",
            "term_cap: usize,",
            "string_bytes_cap: usize,",
            "vector_elements_cap: usize,",
            "source_count_cap: usize,",
            "result_bytes_cap: u64,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroRuntimeAuthorityWireV1<'a> {",
        &[
            "startup_settings_id: &'a str,",
            "runtime_install_receipt_id: &'a str,",
            "generation_zero_runtime_authority_id: &'a str,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroUnusedFullSearchWireV1<'a> {",
        &[
            "scope_id: &'a str,",
            "raw_generations: usize,",
            "clamped_generations: usize,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroCostBandWireV1 {",
        &[
            "status: CanonicalNativeCostBandStatusV1,",
            "cost: Option<(f64, f64)>,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroFinancialProvenanceWireV1<'a> {",
        &[
            "contract: &'a CanonicalTrendbarResearchExecutionContractV3,",
            "cpu_receipt_id: &'a str,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroEvaluatedNativeInputWireV1<'a> {",
        &[
            "receipt_v3: &'a CanonicalGpuResidentSearchInputReceiptV3,",
            "receipt_id: &'a str,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroPopulationSizingWireV1<'a> {",
        &[
            "receipt_v2: &'a ResidentPopulationAutoSizingReceiptV2,",
            "receipt_id: &'a str,",
            "prepared_feature_count: usize,",
            "raw_configured_max_indicators: usize,",
            "resolved_max_indicators: usize,",
            "term_cap: usize,",
            "configured_population: usize,",
            "resolved_population: usize,",
            "population_cap: usize,",
            "hard_growth_cap: usize,",
            "max_concurrent_scenario_count: usize,",
            "stage1_row_start: usize,",
            "stage1_row_end: usize,",
            "selected_device_ordinal: u32,",
            "metrics_receipt_identities_sha256: LowerHexSha256SliceV1<'a>,",
            "adaptive_token_identity_sha256: Option<LowerHexSha256V1>,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroEvaluationSnapshotV1 {",
        &[
            "symbol: String,",
            "account_currency: String,",
            "max_hold_bars: usize,",
            "trailing_enabled: bool,",
            "trailing_atr_multiplier: f64,",
            "trailing_be_trigger_r: f64,",
            "trailing_min_lock_pips: f64,",
            "pip_value: f64,",
            "spread_pips: f64,",
            "commission_per_trade: f64,",
            "pip_value_per_lot: f64,",
            "swap_long_pips_per_day: f64,",
            "swap_short_pips_per_day: f64,",
            "pnl_conversion_fee_rate: f64,",
            "smc_gate_threshold: f64,",
            "smc_weight_ob: f64,",
            "smc_weight_fvg: f64,",
            "smc_weight_liq: f64,",
            "smc_weight_mtf: f64,",
            "smc_weight_premium: f64,",
            "smc_weight_inducement: f64,",
            "smc_weight_bos: f64,",
            "smc_weight_choch: f64,",
            "smc_weight_eqh: f64,",
            "smc_weight_eql: f64,",
            "smc_weight_displacement: f64,",
            "growth_objective: bool,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroEvaluationWireV1<'a> {",
        &[
            "snapshot_v1: &'a CanonicalNativeGenerationZeroEvaluationSnapshotV1,",
            "snapshot_identity_sha256: &'a str,",
            "scoring_objective: CanonicalNativeGenerationZeroScoringObjectiveV1,",
            "effective_smc_gate_threshold: f64,",
            "effective_smc_gate_source: &'static str,",
            "genes: &'a [Gene],",
            "metrics: &'a [[f64; 11]],",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroCompletionWireV1 {",
        &[
            "engine: &'static str,",
            "consumer_completion_confirmed: bool,",
        ],
    );
    assert_field_census_v1(
        sealed,
        "struct CanonicalNativeGenerationZeroReplayWireV1 {",
        &["replay_identity_sealed: bool,"],
    );
}

#[test]
fn embedded_v2_v3_and_counter_schema_censuses_are_exact_and_ordered() {
    let v2 = include_str!("../src/resident_population_auto_sizing_receipt_v2.rs");
    assert!(v2.contains(
        "#[derive(Clone, Debug, PartialEq, Eq, Serialize)]\npub struct ResidentPopulationAutoSizingReceiptV2 {"
    ));
    assert!(v2.contains(
        "#[cfg(all(test, feature = \"gpu-cuda\"))]\n    pub(crate) fn canonical_result_fixture_receipt_v2("
    ));
    assert!(v2.contains(
        "#[cfg(all(test, feature = \"gpu-cuda\"))]\n    pub(crate) fn canonical_result_maximum_json_receipt_v2_for_test("
    ));
    assert_field_census_v1(
        v2,
        "pub struct ResidentPopulationAutoSizingReceiptV2 {",
        &[
            "schema_version: u16,",
            "population_auto: bool,",
            "configured_population: u64,",
            "resolved_population: u64,",
            "resident_parent_rows: u64,",
            "feature_count: u64,",
            "evaluation_rows: u64,",
            "month_capacity: u64,",
            "requested_max_indicators: u64,",
            "term_cap: u64,",
            "stage1_role: String,",
            "stage1_row_start: u64,",
            "stage1_row_end: u64,",
            "migration_enabled_for_run: bool,",
            "adaptive_stops_requested_for_run: bool,",
            "adaptive_base_effective_for_stage1: bool,",
            "adaptive_resolution_reason: String,",
            "resident_adaptive_semantic_v1: String,",
            "stop_target_log_operation_schedule_v3: String,",
            "resident_adaptive_request_identity_sha256: [u8; 32],",
            "adaptive_pip_size_bits: u64,",
            "pip_value_per_lot_bits: u64,",
            "financial_authority_identity_sha256: String,",
            "financial_input_receipt_sha256: String,",
            "financial_source_projection_identity_sha256: [u8; 32],",
            "evaluation_symbol: String,",
            "evaluation_account_currency: String,",
            "adaptive_rr_bits: u64,",
            "adaptive_tail_max_bars: u64,",
            "adaptive_tail_step: u64,",
            "max_ordered_index_count: u64,",
            "max_adaptive_row_count: u64,",
            "selected_device_ordinal: u32,",
            "pre_materialization_free_bytes_snapshot: u64,",
            "allocator_context_reserve_bytes: u64,",
            "allocator_context_reserve_policy: String,",
            "admission_identity_sha256: String,",
            "native_preflight_facts_identity_sha256: String,",
            "cuda_build_manifest_sha256: String,",
            "cuda_build_artifact_sha256: String,",
            "data_peak_device_bytes: u64,",
            "data_steady_device_bytes: u64,",
            "gene_store_device_bytes: u64,",
            "metrics_scenario_device_bytes: u64,",
            "max_concurrent_scenario_count: u64,",
            "bounded_host_metric_readback_bytes: u64,",
            "required_device_bytes_excluding_reserve: u64,",
            "required_device_bytes_including_reserve: u64,",
            "raw_time_cap: u64,",
            "effective_time_cap: u64,",
            "hard_growth_cap: u64,",
            "memory_one_launch_population_cap: u64,",
            "growth_cap: u64,",
            "resolution_reason: String,",
            "workspace_plan_identity_sha256: String,",
            "population_sizing_authority_sha256: String,",
            "data_extent_identity_sha256: String,",
            "identity_sha256: String,",
        ],
    );
    assert_no_field_level_serde_attributes_v1(
        v2,
        "pub struct ResidentPopulationAutoSizingReceiptV2 {",
    );

    let v3 = include_str!("../src/data_selection.rs");
    assert!(v3.contains(
        "#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]\n#[serde(deny_unknown_fields)]\npub struct CanonicalGpuResidentSearchInputReceiptV3 {"
    ));
    for declaration in [
        "pub struct CanonicalFeatureExecutionReceiptV1 {",
        "pub struct CanonicalSearchSourceBindingReceiptV1 {",
        "pub struct CanonicalSearchSourceSegmentReceiptV1 {",
    ] {
        let prefix = v3
            .split_once(declaration)
            .unwrap_or_else(|| panic!("missing nested V3 declaration {declaration}"))
            .0;
        assert!(
            prefix.ends_with("#[serde(deny_unknown_fields)]\n"),
            "nested V3 declaration lost deny_unknown_fields: {declaration}"
        );
    }
    assert_field_census_v1(
        v3,
        "pub struct CanonicalGpuResidentSearchInputReceiptV3 {",
        &[
            "schema_version: u16,",
            "anchor_dataset_identity: String,",
            "feature_plan_identity: String,",
            "feature_provenance_identity: String,",
            "content_merkle_algorithm: String,",
            "feature_content_merkle_sha256: String,",
            "normalization_fit_sha256: String,",
            "row_count: u64,",
            "column_count: u64,",
            "feature_execution: CanonicalFeatureExecutionReceiptV1,",
            "source_bindings: Vec<CanonicalSearchSourceBindingReceiptV1>,",
        ],
    );
    for declaration in [
        "pub struct CanonicalGpuResidentSearchInputReceiptV3 {",
        "pub struct CanonicalFeatureExecutionReceiptV1 {",
        "pub struct CanonicalSearchSourceBindingReceiptV1 {",
        "pub struct CanonicalSearchSourceSegmentReceiptV1 {",
    ] {
        assert_no_field_level_serde_attributes_v1(v3, declaration);
    }
    assert_field_census_v1(
        v3,
        "pub struct CanonicalFeatureExecutionReceiptV1 {",
        &[
            "schema_version: u16,",
            "compute_policy: CanonicalFeatureComputePolicyV1,",
            "vector_ta_math_authority: String,",
            "selected_lane: CanonicalFeatureMathLaneV1,",
        ],
    );
    assert_field_census_v1(
        v3,
        "pub struct CanonicalSearchSourceBindingReceiptV1 {",
        &[
            "source_node_id: String,",
            "dataset_identity: String,",
            "manifest_schema_id: String,",
            "manifest_sha256: String,",
            "generation_id: String,",
            "vortex_sha256: String,",
            "bar_timestamp_convention: String,",
            "segments: Vec<CanonicalSearchSourceSegmentReceiptV1>,",
        ],
    );
    assert_field_census_v1(
        v3,
        "pub struct CanonicalSearchSourceSegmentReceiptV1 {",
        &[
            "row_start: u64,",
            "row_end: u64,",
            "timestamp_start_ms: i64,",
            "timestamp_end_ms: i64,",
        ],
    );

    let counters = include_str!("../../neoethos-gpu-cuda/src/population.rs");
    assert!(counters.contains(
        "#[repr(C)]\n#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]\npub struct PopulationResidencyCountersV1 {"
    ));
    assert_field_census_v1(
        counters,
        "pub struct PopulationResidencyCountersV1 {",
        &[
            "parent_upload_count: u64,",
            "parent_upload_bytes: u64,",
            "view_binding_count: u64,",
            "full_binding_count: u64,",
            "range_binding_count: u64,",
            "ordered_binding_count: u64,",
            "ordered_index_upload_bytes: u64,",
            "adaptive_upload_bytes: u64,",
            "stream_creation_count: u64,",
            "explicit_synchronization_count: u64,",
            "metric_rows_readback_count: u64,",
            "metric_rows_readback_rows: u64,",
            "metric_rows_readback_bytes: u64,",
            "diagnostic_readback_count: u64,",
            "diagnostic_readback_rows: u64,",
            "diagnostic_readback_bytes: u64,",
            "accepted_trade_total_readback_count: u64,",
            "accepted_trade_total_readback_bytes: u64,",
        ],
    );
}

#[test]
fn compact_writer_is_streaming_and_chunk_2a2_has_no_publisher() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let writer = source
        .split_once("// BEGIN CANONICAL_NATIVE_GEN0_STREAMING_WRITER_V1")
        .expect("streaming-writer region")
        .1
        .split_once("// END CANONICAL_NATIVE_GEN0_STREAMING_WRITER_V1")
        .expect("streaming-writer end")
        .0;
    for required in ["W: Write", "serde_json::Serializer", "Sha256"] {
        assert!(writer.contains(required), "writer omits {required}");
    }
    assert!(
        writer.contains("pub(crate) fn write_canonical_native_generation_zero_research_result_v1")
    );
    assert!(!writer.contains("pub fn write_canonical_native_generation_zero_research_result_v1"));
    for forbidden in [
        "Vec<u8>",
        "serde_json::Value",
        "serde_json::to_vec",
        "serde_json::to_string",
        "to_value",
        "to_vec",
        "to_string",
        ".cloned()",
        "collect::<Vec",
        "collect::<Vec<String>>",
        "collect::<Vec<Gene>>",
        "collect::<Vec<[f64; 11]>>",
        "Vec::from(",
        "extend_from_slice",
        "into_boxed_slice",
        "Box::from(",
        "Arc::from(",
        "create_new",
        "native-discovery/v1",
        "current/latest",
    ] {
        assert!(
            !writer.contains(forbidden),
            "Chunk 2A2 writer contains forbidden buffer/publisher seam {forbidden}"
        );
    }
    assert!(
        !source.lines().any(|line| {
            line.trim_start().starts_with("pub fn")
                && line.contains("fixed_metadata_upper_bound_with_empty_arrays_bytes")
        }),
        "raw fixed-metadata sizing authority became public"
    );
}
