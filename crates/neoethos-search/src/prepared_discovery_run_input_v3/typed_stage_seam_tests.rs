use std::cell::Cell;
use std::error::Error as _;

use super::{
    PreparedCanonicalDiscoveryRunInputV5, ResidentGenerationZeroMilestoneV1,
    ResidentGenerationZeroStageErrorV1, checked_v5_max_resolved_population_v1,
    prepare_prepared_canonical_trendbar_research_run_input_capped_v5,
    run_generation_zero_pre_launch_gate_v1,
    run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5,
    run_prepared_canonical_trendbar_research_generation_zero_typed_v5,
    run_prepared_canonical_trendbar_research_generation_zero_v5,
};
use crate::{DiscoveryConfig, DiscoveryProgress};

type AnyhowRunnerV5 = fn(
    PreparedCanonicalDiscoveryRunInputV5,
    fn(DiscoveryProgress),
) -> anyhow::Result<ResidentGenerationZeroMilestoneV1>;

type TypedRunnerV5 =
    fn(
        PreparedCanonicalDiscoveryRunInputV5,
        fn(DiscoveryProgress),
    ) -> Result<ResidentGenerationZeroMilestoneV1, ResidentGenerationZeroStageErrorV1>;

type GatedTypedRunnerV5 =
    fn(
        PreparedCanonicalDiscoveryRunInputV5,
        fn(DiscoveryProgress),
        fn() -> anyhow::Result<()>,
    ) -> Result<ResidentGenerationZeroMilestoneV1, ResidentGenerationZeroStageErrorV1>;

type NativeFactoryV5 = fn(
    neoethos_data::PreparedGpuOnlyFeatureMaterializationV3,
    neoethos_gpu_cuda::AdmittedNativeCudaDataPopulationRunV1,
) -> anyhow::Result<(
    crate::data_selection::CanonicalGpuResidentSearchInputReceiptV3,
    neoethos_data::SealedGpuResidentFeatureStoreV3,
)>;

type CappedPrepareV5 = fn(
    &DiscoveryConfig,
    &crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    neoethos_data::PreparedGpuOnlyFeatureMaterializationV3,
    usize,
    NativeFactoryV5,
) -> anyhow::Result<PreparedCanonicalDiscoveryRunInputV5>;

#[test]
fn typed_sibling_and_public_anyhow_wrapper_keep_the_frozen_v5_signatures() {
    let _: AnyhowRunnerV5 =
        run_prepared_canonical_trendbar_research_generation_zero_v5::<fn(DiscoveryProgress)>;
    let _: TypedRunnerV5 =
        run_prepared_canonical_trendbar_research_generation_zero_typed_v5::<fn(DiscoveryProgress)>;
    let _: GatedTypedRunnerV5 =
        run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5::<
            fn(DiscoveryProgress),
            fn() -> anyhow::Result<()>,
        >;
    let _: CappedPrepareV5 =
        prepare_prepared_canonical_trendbar_research_run_input_capped_v5::<NativeFactoryV5>;
}

#[test]
fn callback_time_cancellation_rejects_the_pre_launch_gate_before_any_launch() {
    let cancelled = Cell::new(false);
    let launch_count = Cell::new(0_usize);

    assert!(!cancelled.get(), "the executor's earlier probe accepted");
    let progress = |_event: DiscoveryProgress| cancelled.set(true);
    progress(DiscoveryProgress::SearchStarted {
        population: 200,
        generations: 0,
        max_indicators: 5,
    });

    let outcome = run_generation_zero_pre_launch_gate_v1(|| {
        if cancelled.get() {
            Err(anyhow::anyhow!("cancelled by progress callback"))
        } else {
            Ok(())
        }
    })
    .and_then(|()| {
        launch_count.set(launch_count.get() + 1);
        Ok(())
    });

    assert!(matches!(
        outcome,
        Err(ResidentGenerationZeroStageErrorV1::PreLaunchGate(_))
    ));
    assert_eq!(launch_count.get(), 0);

    let source = include_str!("../prepared_discovery_run_input_v3.rs");
    let gated_runner = source
        .split_once(
            "pub(crate) fn run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5",
        )
        .expect("gated typed V5 runner")
        .1;
    let progress_position = gated_runner
        .find("progress_fn(DiscoveryProgress::SearchStarted")
        .expect("progress callback before launch");
    let gate_position = gated_runner
        .find("run_generation_zero_pre_launch_gate_v1(pre_launch_gate)")
        .expect("pre-launch gate");
    let launch_position = gated_runner
        .find("consume_strict_resident_population_execution_run_v3")
        .expect("resident CUDA launch boundary");
    assert!(progress_position < gate_position && gate_position < launch_position);
}

#[test]
fn capped_v5_accepts_the_exact_smaller_or_global_bound_and_rejects_overflowing_configured_p() {
    assert_eq!(
        checked_v5_max_resolved_population_v1(200, 4_096).expect("smaller preflight cap"),
        4_096
    );
    assert_eq!(
        checked_v5_max_resolved_population_v1(200, 16_384).expect("global cap"),
        16_384
    );
    assert_eq!(
        checked_v5_max_resolved_population_v1(200, usize::MAX)
            .expect("external cap above the global ceiling"),
        16_384
    );
    assert_eq!(
        checked_v5_max_resolved_population_v1(20_000, 30_000)
            .expect("configured population remains valid below the raw external cap"),
        16_384
    );
    assert!(checked_v5_max_resolved_population_v1(4_097, 4_096).is_err());
    assert!(checked_v5_max_resolved_population_v1(200, 0).is_err());
}

#[test]
fn stage_error_preserves_evaluation_and_consumer_completion_as_distinct_variants() {
    let evaluation = ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(anyhow::anyhow!(
        "evaluation sentinel"
    ));
    assert!(matches!(
        &evaluation,
        ResidentGenerationZeroStageErrorV1::GenerationZeroEvaluation(_)
    ));
    assert_eq!(
        evaluation.to_string(),
        "resident Generation-0 evaluation failed: evaluation sentinel"
    );
    assert_eq!(
        evaluation.source().map(ToString::to_string).as_deref(),
        Some("evaluation sentinel")
    );

    let completion = ResidentGenerationZeroStageErrorV1::ConsumerCompletion(anyhow::anyhow!(
        "completion sentinel"
    ));
    assert!(matches!(
        &completion,
        ResidentGenerationZeroStageErrorV1::ConsumerCompletion(_)
    ));
    assert_eq!(
        completion.to_string(),
        "resident Generation-0 consumer completion failed: completion sentinel"
    );
    assert_eq!(
        completion.source().map(ToString::to_string).as_deref(),
        Some("completion sentinel")
    );

    let pre_launch =
        ResidentGenerationZeroStageErrorV1::PreLaunchGate(anyhow::anyhow!("cancellation sentinel"));
    assert_eq!(
        pre_launch.to_string(),
        "resident Generation-0 pre-launch gate rejected: cancellation sentinel"
    );
}
