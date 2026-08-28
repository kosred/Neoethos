#![cfg(feature = "cuda-device-fixtures")]

use neoethos_gpu_cuda::resident_search_v2::{
    ResidentSearchFixturePlanV2, ResidentSearchStateV2, ResidentSearchV2Error,
    resident_search_v2_production_readiness,
};
use neoethos_gpu_cuda::{PopulationDatasetView, PopulationSession, SMC_SLOTS};

fn move_run(
    run: neoethos_gpu_cuda::resident_search_v2::ResidentSearchRunV2,
) -> neoethos_gpu_cuda::resident_search_v2::ResidentSearchRunV2 {
    run
}

#[test]
fn production_admission_fails_closed_until_v2_semantics_are_ready() {
    let readiness = resident_search_v2_production_readiness();
    assert!(!readiness.exact_generation_semantics());
    assert!(!readiness.device_resident_generation_advance());
    assert!(readiness.device_owned_search_control());
    assert!(!readiness.immutable_scenario_admission());
    assert!(!readiness.whole_workspace_preallocated());
    assert!(!readiness.unified_device_fault_authority());
    assert!(readiness.native_bridge_production_sealed());
    assert!(readiness.terminal_cleanup_lease());
    assert!(!readiness.production_ready());
    let session = PopulationSession::create(0, 1).expect("create CUDA population session");
    let result = session.begin_resident_search_v2();
    assert!(matches!(
        result,
        Err(ResidentSearchV2Error::ResidentGenerationSemanticsNotProductionReady)
    ));
}

#[test]
fn boxed_ready_receipt_survives_owner_moves_and_native_pointer_validation() {
    const BARS: usize = 64;
    const FEATURES: usize = 8;
    const POPULATION: usize = 4;

    let close = (0..BARS)
        .map(|bar| 1.1 + bar as f64 * 0.0001)
        .collect::<Vec<_>>();
    let high = close.iter().map(|value| value + 0.0005).collect::<Vec<_>>();
    let low = close.iter().map(|value| value - 0.0005).collect::<Vec<_>>();
    let indicators = (0..FEATURES * BARS)
        .map(|index| if index % 3 == 0 { 0.25 } else { -0.125 })
        .collect::<Vec<_>>();
    let months = vec![202_401_i64; BARS];
    let days = vec![1_i64; BARS];
    let timestamps = (0..BARS)
        .map(|bar| i64::try_from(bar).expect("bar fits i64") * 60_000)
        .collect::<Vec<_>>();
    let smc_rows = vec![0_i8; BARS * SMC_SLOTS];

    let mut session = PopulationSession::create(0, 1).expect("create CUDA population session");
    session
        .upload_dataset(PopulationDatasetView {
            close: &close,
            high: &high,
            low: &low,
            indicators: &indicators,
            feature_count: FEATURES,
            months: &months,
            days: &days,
            timestamps: &timestamps,
            smc_rows: &smc_rows,
            adaptive_base_pips: None,
        })
        .expect("upload fixture dataset once");

    let plan = ResidentSearchFixturePlanV2::new(POPULATION, FEATURES)
        .expect("seal research-only fixture plan");
    let smc_weights = [
        0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375,
    ];
    let run = session
        .begin_resident_search_fixture_v2(plan, smc_weights, false)
        .expect("initialize typed resident fixture owner");
    let mut moved = move_run(move_run(run));

    assert_eq!(moved.state_v2(), ResidentSearchStateV2::Active);
    assert!(moved.ready_receipt_address_is_stable_v2());
    let view = moved
        .refresh_current_gene_view_v2()
        .expect("native exact-pointer export after Rust owner moves");
    assert_eq!(view.generation_index(), 0);
    assert_eq!(view.store_epoch(), 1);
    assert_eq!(view.logical_population_count(), POPULATION as u64);
    assert_eq!(view.feature_count(), FEATURES as u64);
    assert_eq!(view.max_terms_per_gene(), 3);
    assert_eq!(view.smc_flag_count(), SMC_SLOTS as u32);
    assert_ne!(view.run_token(), 0);

    let session = moved
        .close_fixture_v2()
        .expect("same-stream generation release after pointer fixture");
    drop(session);
}
