fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing start marker: {start}"))
        .1;
    source
        .split_once(end)
        .unwrap_or_else(|| panic!("missing end marker: {end}"))
        .0
}

#[test]
fn public_discovery_preflights_the_strict_native_population_stage_not_a_fake_full_gpu_pipeline() {
    let source = include_str!("../src/discovery.rs");
    let body = section(
        source,
        "fn run_discovery_cycle_with_holdout_and_progress_authorized",
        "outer OOS holdout: discovery sees only",
    );

    assert!(
        !body.contains("FULL_DISCOVERY"),
        "the mixed CPU/GPU Discovery manifest cannot be admitted as an all-strict-GPU pipeline"
    );
    assert!(body.contains("PipelineStage::PopulationEvaluation"));
    assert!(
        body.contains("FeaturePreparation") && body.contains("CpuOnly"),
        "the public path must report its remaining host feature-preparation stage"
    );
}
