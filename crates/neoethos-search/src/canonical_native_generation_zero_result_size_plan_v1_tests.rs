use crate::canonical_native_discovery_request_v1::{
    MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1, MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    MAX_CANONICAL_NATIVE_GEN0_TERMS_V1,
};
use crate::canonical_native_generation_zero_result_v1::{
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
    CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1,
    CanonicalNativeGenerationZeroResultSizePlanV1,
};

const FIXED_EMPTY_BYTES: u64 = 1_000;

fn plan(
    feature_count: usize,
    raw_max_indicators: usize,
    configured_population: usize,
    fixed_empty_bytes: u64,
) -> CanonicalNativeGenerationZeroResultSizePlanV1 {
    CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        feature_count,
        raw_max_indicators,
        configured_population,
        fixed_empty_bytes,
    )
    .expect("valid Generation-zero size plan")
}

#[test]
fn reserves_the_final_result_schema_without_constructing_a_result() {
    assert_eq!(
        CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
        "neoethos.canonical-native-generation-zero-research-result.v1"
    );
    assert_eq!(
        CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
        1
    );
}

#[test]
fn raw_zero_resolves_to_prepared_feature_count_and_keeps_the_template_floor() {
    let size = plan(5, 0, 10, FIXED_EMPTY_BYTES);
    assert_eq!(size.prepared_feature_count(), 5);
    assert_eq!(size.raw_configured_max_indicators(), 0);
    assert_eq!(size.resolved_max_indicators(), 5);
    assert_eq!(size.term_cap(), 5);
    assert_eq!(size.per_population_upper_bound_bytes(), 1_672);
}

#[test]
fn term_resolution_covers_raw_above_features_and_the_4096_boundary() {
    let raw_above_features = plan(7, 4_096, 10, FIXED_EMPTY_BYTES);
    assert_eq!(raw_above_features.resolved_max_indicators(), 4_096);
    assert_eq!(raw_above_features.term_cap(), 7);

    let boundary = plan(4_096, 4_096, 10, FIXED_EMPTY_BYTES);
    assert_eq!(boundary.term_cap(), 4_096);
    assert_eq!(boundary.per_population_upper_bound_bytes(), 189_858);
}

#[test]
fn invalid_feature_and_raw_term_inputs_fail_before_any_size_plan_exists() {
    for (features, raw_max) in [(0, 0), (4_097, 0), (5, 4_097)] {
        let error = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
            features,
            raw_max,
            10,
            FIXED_EMPTY_BYTES,
        )
        .expect_err("invalid F/raw max-indicators must fail");
        assert_eq!(
            error.code(),
            CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput
        );
    }
}

#[test]
fn configured_population_uses_the_existing_ten_candidate_floor() {
    for configured_population in [0, 1, 9] {
        let error = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
            5,
            0,
            configured_population,
            FIXED_EMPTY_BYTES,
        )
        .expect_err("meaningless configured population must fail");
        assert_eq!(
            error.code(),
            CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput
        );
    }
    assert_eq!(
        plan(5, 0, 10, FIXED_EMPTY_BYTES).configured_population(),
        10
    );
}

#[test]
fn fixed_metadata_bound_must_include_three_empty_arrays_and_fit_the_envelope() {
    let subtraction = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(5, 0, 10, 2)
        .expect_err("subtracting the three empty-array closers must not underflow");
    assert_eq!(
        subtraction.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow
    );

    let over_cap = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        5,
        0,
        10,
        MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1 + 1,
    )
    .expect_err("fixed metadata cannot already exceed the result envelope");
    assert_eq!(
        over_cap.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput
    );
}

#[test]
fn a_fixed_bound_that_leaves_no_complete_population_row_fails_by_name() {
    let error = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        5,
        0,
        10,
        MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    )
    .expect_err("the capacity cannot resolve to zero");
    assert_eq!(
        error.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::PopulationCapacityZero
    );
}

#[test]
fn exact_upper_bounds_and_capacity_match_reviewed_reference_values() {
    let five_terms = plan(5, 0, 10, FIXED_EMPTY_BYTES);
    assert_eq!(
        five_terms.fixed_metadata_without_empty_array_closers_bytes(),
        997
    );
    assert_eq!(
        five_terms.checked_upper_bound_for_population(10).unwrap(),
        17_717
    );
    assert_eq!(five_terms.population_cap(), 321_094);

    let maximum_terms = plan(4_096, 4_096, 10, FIXED_EMPTY_BYTES);
    assert_eq!(
        maximum_terms
            .checked_upper_bound_for_population(10)
            .unwrap(),
        1_899_577
    );
    assert_eq!(maximum_terms.population_cap(), 2_827);
}

#[test]
fn checked_upper_bound_rejects_zero_and_multiplication_overflow() {
    let size = plan(5, 0, 10, FIXED_EMPTY_BYTES);
    let zero = size
        .checked_upper_bound_for_population(0)
        .expect_err("zero population is not a result");
    assert_eq!(
        zero.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput
    );
    let overflow = size
        .checked_upper_bound_for_population(usize::MAX)
        .expect_err("P multiplied by per-P bytes must be checked");
    assert_eq!(
        overflow.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ArithmeticOverflow
    );
}

#[test]
fn checked_upper_bound_rejects_population_cap_plus_one_by_name() {
    let size = plan(5, 0, 10, FIXED_EMPTY_BYTES);
    let error = size
        .checked_upper_bound_for_population(size.population_cap() + 1)
        .expect_err("an admitted bound cannot be calculated above Pcap");
    assert_eq!(
        error.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ConfiguredPopulationExceedsCapacity
    );
}

#[test]
fn configured_population_passes_at_capacity_and_is_never_silently_shrunk() {
    let capacity = plan(5, 0, 10, FIXED_EMPTY_BYTES).population_cap();
    let exact = plan(5, 0, capacity, FIXED_EMPTY_BYTES);
    assert_eq!(exact.configured_population(), capacity);
    assert_eq!(
        exact.configured_population_upper_bound_bytes(),
        exact.checked_upper_bound_for_population(capacity).unwrap()
    );

    let error = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        5,
        0,
        capacity + 1,
        FIXED_EMPTY_BYTES,
    )
    .expect_err("configured P above Pcap must not be shrunk");
    assert_eq!(
        error.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::ConfiguredPopulationExceedsCapacity
    );
}

#[test]
fn configured_population_cannot_exceed_the_named_v1_ceiling() {
    let error = CanonicalNativeGenerationZeroResultSizePlanV1::checked_new(
        5,
        0,
        MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1 + 1,
        FIXED_EMPTY_BYTES,
    )
    .expect_err("configured population ceiling must be explicit");
    assert_eq!(
        error.code(),
        CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1::InvalidInput
    );
    assert_eq!(MAX_CANONICAL_NATIVE_GEN0_TERMS_V1, 4_096);
}

#[test]
fn planner_source_has_no_result_payload_or_receipt_authority() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let planner = source
        .split_once("pub(crate) struct CanonicalNativeGenerationZeroResultSizePlanV1")
        .expect("planner start")
        .1
        .split_once("// END CANONICAL_NATIVE_GENERATION_ZERO_SIZE_PLANNER_V1")
        .expect("planner end")
        .0;
    for forbidden in [
        "SearchResult",
        ".genes",
        ".metrics",
        ".clone()",
        "serde_json::to_vec",
        "ResidentPopulationAutoSizingReceiptV2",
        "CanonicalGpuResidentSearchInputReceiptV3",
    ] {
        assert!(
            !planner.contains(forbidden),
            "size-only source must not contain {forbidden}"
        );
    }
}
