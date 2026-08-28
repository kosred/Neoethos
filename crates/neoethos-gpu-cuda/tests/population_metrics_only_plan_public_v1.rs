use neoethos_gpu_cuda::{
    PopulationGeneStorePlanV1, PopulationMetricsOnlyPlanV1, PopulationParentDevicePlanV1,
};

#[test]
fn strict_month_240_plan_is_public_exact_and_has_no_outcomes() {
    let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(1, 240)
        .expect("the public strict metrics-only planner must accept one scenario");

    assert_eq!(plan.metric_rows_bytes(), 104);
    assert_eq!(plan.monthly_pnls_bytes(), 1_920);
    assert_eq!(plan.month_start_equities_bytes(), 1_920);
    assert_eq!(plan.scenario_descriptor_bytes(), 56);
    assert_eq!(plan.total_device_bytes(), 4_000);
    assert_eq!(plan.outcome_bytes(), 0);
    assert_eq!(plan.accepted_trade_total_bytes(), 0);

    let stale_outcome_charge = 8_192u64 * 72;
    assert_eq!(stale_outcome_charge + plan.total_device_bytes(), 593_824);
    assert_ne!(
        plan.total_device_bytes(),
        stale_outcome_charge + plan.total_device_bytes(),
        "the strict plan must never resurrect the compatibility outcome array"
    );
}

#[test]
fn strict_parent_and_gene_plans_match_the_native_allocations() {
    let parent = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(262_290, 1_800)
        .expect("M15 parent extents must have a checked plan");
    assert_eq!(
        parent.total_device_bytes(),
        (8 * 1_800 + 76) * 262_290,
        "the always-allocated view index must not disappear from parent sizing"
    );
    assert_eq!(parent.view_indices_bytes(), 8 * 262_290);

    let genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(200, 3_200)
        .expect("200 genes with sixteen terms each must have a checked plan");
    assert_eq!(genes.total_device_bytes(), 63 * 200 + 12 * 3_200 + 92);
}
