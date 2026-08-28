#[allow(unused_imports)]
use super::*;
use crate::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
use neoethos_gpu_cuda::PopulationMetricsOnlyPlanV1;

fn valid_execution_facts_v1() -> CanonicalNativeGenerationZeroExecutionFactsV1 {
    let population = 10_usize;
    let scenario_count = 4_usize;
    let launch_count = population.div_ceil(scenario_count);
    let metric_bytes = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(population, 12)
        .unwrap()
        .metric_rows_bytes();
    CanonicalNativeGenerationZeroExecutionFactsV1 {
        prepared_feature_count: 5,
        native_receipt_feature_count: 5,
        request_raw_configured_max_indicators: 5,
        sizing_requested_max_indicators: 5,
        preflight_term_cap: 5,
        sizing_term_cap: 5,
        milestone_term_cap: 5,
        request_configured_population: 10,
        sizing_configured_population: 10,
        sizing_resolved_population: population,
        milestone_resolved_population: population,
        population_cap: 100,
        hard_growth_cap: 100,
        max_concurrent_scenario_count: scenario_count,
        month_capacity: 12,
        sizing_stage1_row_start: 100,
        sizing_stage1_row_end: 600,
        milestone_stage1_row_start: 100,
        milestone_stage1_row_end: 600,
        sizing_selected_device_ordinal: 0,
        milestone_selected_device_ordinal: 0,
        native_input_receipt_identity_sha256: "a".repeat(64),
        milestone_native_input_receipt_identity_sha256: "a".repeat(64),
        population_sizing_receipt_identity_sha256: "b".repeat(64),
        milestone_population_sizing_receipt_identity_sha256: "b".repeat(64),
        adaptive_base_effective_for_stage1: false,
        sizing_resident_adaptive_request_identity_sha256: [0; 32],
        milestone_adaptive_token_identity_sha256: None,
        metrics_receipt_identities_sha256: vec![[1; 32]; launch_count],
        counters: CanonicalNativeGenerationZeroResidencyCountersSnapshotV1 {
            parent_upload_count: 0,
            parent_upload_bytes: 0,
            view_binding_count: 1,
            full_binding_count: 1,
            range_binding_count: 0,
            ordered_binding_count: 0,
            ordered_index_upload_bytes: 0,
            adaptive_upload_bytes: 0,
            stream_creation_count: 0,
            explicit_synchronization_count: launch_count as u64,
            metric_rows_readback_count: launch_count as u64,
            metric_rows_readback_rows: population as u64,
            metric_rows_readback_bytes: metric_bytes,
            diagnostic_readback_count: 0,
            diagnostic_readback_rows: 0,
            diagnostic_readback_bytes: 0,
            accepted_trade_total_readback_count: 0,
            accepted_trade_total_readback_bytes: 0,
        },
        engine: "CudaNativeF64",
        consumer_completion_confirmed: true,
        replay_identity_sealed: false,
    }
}

fn refresh_population_execution_facts_v1(
    facts: &mut CanonicalNativeGenerationZeroExecutionFactsV1,
) {
    facts.milestone_resolved_population = facts.sizing_resolved_population;
    let launches = facts
        .sizing_resolved_population
        .div_ceil(facts.max_concurrent_scenario_count);
    facts.metrics_receipt_identities_sha256 = vec![[1; 32]; launches];
    facts.counters.explicit_synchronization_count = launches as u64;
    facts.counters.metric_rows_readback_count = launches as u64;
    facts.counters.metric_rows_readback_rows = facts.sizing_resolved_population as u64;
    facts.counters.metric_rows_readback_bytes =
        PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            facts.sizing_resolved_population,
            facts.month_capacity as u32,
        )
        .unwrap()
        .metric_rows_bytes();
}

#[test]
fn execution_facts_bind_p_f_t_k_s_l_and_authoritative_metric_bytes() {
    let facts = valid_execution_facts_v1();
    validate_execution_facts_v1(&facts).expect("valid Gen0 execution facts");
    assert_eq!(
        facts.metrics_receipt_identities_sha256.len(),
        facts
            .sizing_resolved_population
            .div_ceil(facts.max_concurrent_scenario_count)
    );

    let mutations: &[fn(&mut CanonicalNativeGenerationZeroExecutionFactsV1)] = &[
        |facts| facts.prepared_feature_count = 4,
        |facts| facts.native_receipt_feature_count = 4,
        |facts| facts.request_raw_configured_max_indicators = 4,
        |facts| facts.sizing_requested_max_indicators = 4,
        |facts| facts.preflight_term_cap = 4,
        |facts| facts.sizing_term_cap = 4,
        |facts| facts.milestone_term_cap = 4,
        |facts| facts.request_configured_population = 11,
        |facts| facts.sizing_configured_population = 11,
        |facts| facts.sizing_resolved_population = 11,
        |facts| facts.milestone_resolved_population = 11,
        |facts| facts.population_cap = 9,
        |facts| facts.hard_growth_cap = 9,
        |facts| facts.max_concurrent_scenario_count = 0,
        |facts| facts.month_capacity = 0,
        |facts| facts.sizing_stage1_row_end = facts.sizing_stage1_row_start,
        |facts| facts.milestone_stage1_row_end -= 1,
        |facts| facts.sizing_selected_device_ordinal = 1,
        |facts| facts.milestone_selected_device_ordinal = 1,
        |facts| facts.native_input_receipt_identity_sha256 = "A".repeat(64),
        |facts| facts.milestone_native_input_receipt_identity_sha256 = "c".repeat(64),
        |facts| facts.population_sizing_receipt_identity_sha256 = "0".repeat(64),
        |facts| facts.milestone_population_sizing_receipt_identity_sha256 = "c".repeat(64),
        |facts| {
            facts.metrics_receipt_identities_sha256.pop();
        },
        |facts| facts.metrics_receipt_identities_sha256[0] = [0; 32],
        |facts| facts.engine = "CpuOnly",
        |facts| facts.consumer_completion_confirmed = false,
        |facts| facts.replay_identity_sealed = true,
    ];
    for mutate in mutations {
        let mut invalid = valid_execution_facts_v1();
        mutate(&mut invalid);
        assert!(validate_execution_facts_v1(&invalid).is_err());
    }

    let mut nonzero_ordinal = valid_execution_facts_v1();
    nonzero_ordinal.sizing_selected_device_ordinal = 1;
    nonzero_ordinal.milestone_selected_device_ordinal = 1;
    validate_execution_facts_v1(&nonzero_ordinal)
        .expect("a coherent nonzero CUDA ordinal is valid execution evidence");
}

#[test]
fn zero_indicator_sentinel_binds_the_resolved_feature_count_not_raw_zero() {
    let mut facts = valid_execution_facts_v1();
    facts.request_raw_configured_max_indicators = 0;
    facts.sizing_requested_max_indicators = facts.prepared_feature_count;
    validate_execution_facts_v1(&facts)
        .expect("raw zero resolves to exact prepared F before V5 sizing");

    facts.sizing_requested_max_indicators -= 1;
    assert!(validate_execution_facts_v1(&facts).is_err());
}

#[test]
fn population_cap_binding_accepts_both_hard_cap_branches_and_never_shrinks_configured_p() {
    assert_eq!(RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2, 16_384);
    for (population_cap, expected_hard_cap) in [
        (4_096, 4_096),
        (20_000, RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2),
    ] {
        let mut facts = valid_execution_facts_v1();
        facts.population_cap = population_cap;
        facts.hard_growth_cap = expected_hard_cap;
        validate_execution_facts_v1(&facts).expect("hard cap is min(global, Pcap)");
    }

    let mut auto_grew = valid_execution_facts_v1();
    auto_grew.request_configured_population = 10;
    auto_grew.sizing_configured_population = 10;
    auto_grew.sizing_resolved_population = 20;
    refresh_population_execution_facts_v1(&mut auto_grew);
    validate_execution_facts_v1(&auto_grew).expect("coherent auto growth P != configured P");

    let mut exact_cap = valid_execution_facts_v1();
    exact_cap.request_configured_population = 100;
    exact_cap.sizing_configured_population = 100;
    exact_cap.sizing_resolved_population = 100;
    exact_cap.population_cap = 100;
    exact_cap.hard_growth_cap = 100;
    refresh_population_execution_facts_v1(&mut exact_cap);
    validate_execution_facts_v1(&exact_cap).expect("configured P exactly at Pcap is not shrunk");

    let mut above_growth_no_shrink = valid_execution_facts_v1();
    above_growth_no_shrink.request_configured_population = 20_000;
    above_growth_no_shrink.sizing_configured_population = 20_000;
    above_growth_no_shrink.sizing_resolved_population = 20_000;
    above_growth_no_shrink.population_cap = 30_000;
    above_growth_no_shrink.hard_growth_cap = RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2;
    refresh_population_execution_facts_v1(&mut above_growth_no_shrink);
    validate_execution_facts_v1(&above_growth_no_shrink)
        .expect("configured P above the growth cap is admitted without shrink below external Pcap");

    for (population_cap, hard_growth_cap) in [
        (4_096, 4_097),
        (4_096, 4_095),
        (20_000, 20_000),
        (20_000, RESIDENT_POPULATION_AUTO_HARD_GROWTH_CAP_V2 - 1),
    ] {
        let mut wrong = valid_execution_facts_v1();
        wrong.population_cap = population_cap;
        wrong.hard_growth_cap = hard_growth_cap;
        assert!(validate_execution_facts_v1(&wrong).is_err());
    }
    let mut above_cap = valid_execution_facts_v1();
    above_cap.request_configured_population = 101;
    above_cap.sizing_configured_population = 101;
    above_cap.sizing_resolved_population = 101;
    above_cap.population_cap = 100;
    above_cap.hard_growth_cap = 100;
    refresh_population_execution_facts_v1(&mut above_cap);
    assert!(validate_execution_facts_v1(&above_cap).is_err());
}

#[test]
fn adaptive_request_and_domain_separated_token_are_present_nonzero_iff_effective() {
    let fallback = valid_execution_facts_v1();
    validate_execution_facts_v1(&fallback).expect("fallback carries no adaptive token");

    let mut effective = valid_execution_facts_v1();
    effective.adaptive_base_effective_for_stage1 = true;
    effective.sizing_resident_adaptive_request_identity_sha256 = [7; 32];
    effective.milestone_adaptive_token_identity_sha256 = Some([8; 32]);
    validate_execution_facts_v1(&effective)
        .expect("effective adaptive request and token identities use distinct hash domains");

    let mutations: &[fn(&mut CanonicalNativeGenerationZeroExecutionFactsV1)] = &[
        |facts| facts.milestone_adaptive_token_identity_sha256 = Some([7; 32]),
        |facts| facts.sizing_resident_adaptive_request_identity_sha256 = [7; 32],
        |facts| {
            facts.adaptive_base_effective_for_stage1 = true;
            facts.sizing_resident_adaptive_request_identity_sha256 = [7; 32];
        },
        |facts| {
            facts.adaptive_base_effective_for_stage1 = true;
            facts.milestone_adaptive_token_identity_sha256 = Some([7; 32]);
        },
        |facts| {
            facts.adaptive_base_effective_for_stage1 = true;
            facts.sizing_resident_adaptive_request_identity_sha256 = [7; 32];
            facts.milestone_adaptive_token_identity_sha256 = Some([0; 32]);
        },
        |facts| {
            facts.adaptive_base_effective_for_stage1 = true;
            facts.sizing_resident_adaptive_request_identity_sha256 = [7; 32];
            facts.milestone_adaptive_token_identity_sha256 = Some([7; 32]);
        },
    ];
    for mutate in mutations {
        let mut invalid = valid_execution_facts_v1();
        mutate(&mut invalid);
        assert!(validate_execution_facts_v1(&invalid).is_err());
    }
}

#[test]
fn effective_smc_gate_binds_finite_actual_bits_and_versioned_runtime_source() {
    let source = EFFECTIVE_SMC_GATE_SOURCE_GENETIC_SEARCH_RUNTIME_START_GENERATION_ZERO_V1;
    assert_eq!(source, "genetic_search_runtime_start_generation_zero_v1");
    let evidence = CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
        0.5,
        source,
        &"a".repeat(64),
        &"b".repeat(64),
        &"c".repeat(64),
    )
    .unwrap();
    assert_eq!(
        evidence.effective_smc_gate_threshold().to_bits(),
        0.5_f64.to_bits()
    );
    assert_eq!(evidence.source(), source);

    let repeated = CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
        0.5,
        source,
        &"a".repeat(64),
        &"b".repeat(64),
        &"c".repeat(64),
    )
    .unwrap();
    assert_eq!(evidence.identity_sha256(), repeated.identity_sha256());

    let adjacent_gate = f64::from_bits(0.5_f64.to_bits() + 1);
    let adjacent = CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
        adjacent_gate,
        source,
        &"a".repeat(64),
        &"b".repeat(64),
        &"c".repeat(64),
    )
    .unwrap();
    assert_ne!(evidence.identity_sha256(), adjacent.identity_sha256());
    for (startup, installation, runtime) in [
        ("d".repeat(64), "b".repeat(64), "c".repeat(64)),
        ("a".repeat(64), "d".repeat(64), "c".repeat(64)),
        ("a".repeat(64), "b".repeat(64), "d".repeat(64)),
    ] {
        let changed = CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
            0.5,
            source,
            &startup,
            &installation,
            &runtime,
        )
        .unwrap();
        assert_ne!(evidence.identity_sha256(), changed.identity_sha256());
    }

    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
                invalid,
                source,
                &"a".repeat(64),
                &"b".repeat(64),
                &"c".repeat(64),
            )
            .is_err()
        );
    }
    assert!(
        CanonicalNativeGenerationZeroEffectiveSmcGateEvidenceV1::checked_new(
            0.5,
            "evaluation_config_smc_gate_threshold",
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
        )
        .is_err()
    );
}

#[test]
fn every_residency_counter_is_checked_against_generation_zero_semantics() {
    let mutations: &[fn(&mut CanonicalNativeGenerationZeroResidencyCountersSnapshotV1)] = &[
        |value| value.parent_upload_count = 1,
        |value| value.parent_upload_bytes = 1,
        |value| value.view_binding_count = 2,
        |value| value.full_binding_count = 0,
        |value| value.range_binding_count = 1,
        |value| value.ordered_binding_count = 1,
        |value| value.ordered_index_upload_bytes = 1,
        |value| value.adaptive_upload_bytes = 1,
        |value| value.stream_creation_count = 1,
        |value| value.explicit_synchronization_count -= 1,
        |value| value.metric_rows_readback_count -= 1,
        |value| value.metric_rows_readback_rows -= 1,
        |value| value.metric_rows_readback_bytes -= 1,
        |value| value.diagnostic_readback_count = 1,
        |value| value.diagnostic_readback_rows = 1,
        |value| value.diagnostic_readback_bytes = 1,
        |value| value.accepted_trade_total_readback_count = 1,
        |value| value.accepted_trade_total_readback_bytes = 1,
    ];
    assert_eq!(mutations.len(), 18);
    for mutate in mutations {
        let mut invalid = valid_execution_facts_v1();
        mutate(&mut invalid.counters);
        assert!(validate_execution_facts_v1(&invalid).is_err());
    }
}

#[test]
fn one_contiguous_stage1_range_binding_is_valid_execution_evidence() {
    let mut range = valid_execution_facts_v1();
    range.counters.full_binding_count = 0;
    range.counters.range_binding_count = 1;
    validate_execution_facts_v1(&range)
        .expect("one contiguous range binding is the real Stage1 execution shape");

    let mut none = valid_execution_facts_v1();
    none.counters.full_binding_count = 0;
    assert!(validate_execution_facts_v1(&none).is_err());

    let mut both = valid_execution_facts_v1();
    both.counters.range_binding_count = 1;
    assert!(validate_execution_facts_v1(&both).is_err());
}

#[test]
fn research_only_policy_flags_are_all_fail_closed() {
    let valid = CanonicalNativeGenerationZeroPolicyFactsV1 {
        execution_scope: crate::canonical_native_discovery_request_v1::
            CanonicalNativeExecutionScopeV1::GenerationZeroOnly,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility:
            HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        cost_band_status: crate::canonical_native_discovery_request_v1::
            CanonicalNativeCostBandStatusV1::UnusedGenerationZero,
        consumer_completion_confirmed: true,
        replay_identity_sealed: false,
    };
    validate_policy_facts_v1(&valid).unwrap();
    for mutate in [
        |facts: &mut CanonicalNativeGenerationZeroPolicyFactsV1| {
            facts.authorization_issued = true;
        },
        |facts: &mut CanonicalNativeGenerationZeroPolicyFactsV1| {
            facts.consumer_completion_confirmed = false;
        },
        |facts: &mut CanonicalNativeGenerationZeroPolicyFactsV1| {
            facts.replay_identity_sealed = true;
        },
    ] {
        let mut invalid = valid;
        mutate(&mut invalid);
        assert!(validate_policy_facts_v1(&invalid).is_err());
    }
}

#[test]
fn safety_enums_and_versioned_objective_have_exact_compact_json_literals() {
    assert_eq!(
        serde_json::to_string(
            &crate::canonical_native_discovery_request_v1::
                CanonicalNativeExecutionScopeV1::GenerationZeroOnly,
        )
        .unwrap(),
        "\"generation_zero_only\""
    );
    assert_eq!(
        serde_json::to_string(&HistoricalResearchArtifactClassV1::ResearchOnly).unwrap(),
        "\"research_only\""
    );
    assert_eq!(
        serde_json::to_string(&HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,)
            .unwrap(),
        "\"not_promotion_eligible\""
    );
    assert_eq!(
        serde_json::to_string(
            &crate::canonical_native_discovery_request_v1::
                CanonicalNativeCostBandStatusV1::UnusedGenerationZero,
        )
        .unwrap(),
        "\"unused_generation_zero\""
    );
    assert_eq!(
        serde_json::to_string(&CanonicalNativeGenerationZeroScoringObjectiveV1::PropConsistencyV4,)
            .unwrap(),
        "\"prop_consistency_v4\""
    );
    assert_eq!(
        serde_json::to_string(
            &CanonicalNativeGenerationZeroScoringObjectiveV1::RiskyKellyGrowthV5,
        )
        .unwrap(),
        "\"risky_kelly_growth_v5\""
    );
}
