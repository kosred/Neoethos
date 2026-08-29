//! Strict-GPU generation-stage orchestration.
//!
//! This module is intentionally not exported until the lower gpu-cuda
//! resident-generation ABI and the complete sixteen-stage permit are compiled
//! and device-validated.  It describes one move-only, stream-ordered generation
//! run; it never reconstructs either authority from caller-supplied hashes.

use sha2::{Digest, Sha256};

use crate::full_discovery_gpu_stage_authority_v2::StrictGpuOnlyFullDiscoveryPermitV2;
use neoethos_gpu_cuda::resident_generation_v1::{
    ActualResidentGenerationAllocationPlanV1, CubScratchArenaV1, DeviceParityMetricRowsV1,
    DevicePointerV1, FinalBoundedGenerationDiagnosticsReceiptV1, GpuPopulationEvaluationModeV1,
    GpuPopulationEvaluationRequestV1, GpuResidentCounterRngAuthorityV1, GpuResidentGeneStoreV1,
    GpuResidentGenerationReadyEventV1, GpuResidentMetricRowsHandleV1, GpuResidentMetricStoreV1,
    GpuResidentOffspringStoreV1, GpuResidentPopulationStoreV1, GpuResidentRankStoreV1,
    GpuResidentSelectionStoreV1, GpuRunStreamV1, MonthlyMeanVarianceOrderV1,
    OnlineDailyMetricAccumulatorV1, OnlineMonthlyMetricAccumulatorV1,
    OnlineStrategyMetricAccumulatorV1, ResidentGenerationDeviceErrorV1,
    ResidentGenerationStoreBundleV1, ResidentPostGaInputV1,
    allocate_gpu_resident_generation_stores_v1, enqueue_population_metrics_only_on_run_stream_v1,
    launch_device_crossover_v1, launch_device_gene_hash_v1, launch_device_mutation_v1,
    launch_device_parent_selection_v1, rank_and_select_after_event_dependency_v1,
    reduce_final_generation_diagnostics_on_device_v1, rotate_resident_generation_stores_v1,
};

const GENERATION_RUN_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.search.gpu-resident-generation-run.v1\0";
const GENERATION_MEMORY_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.search.gpu-generation-memory-plan.v1\0";
const RNG_MAPPING_IDENTITY_DOMAIN_V1: &[u8] = b"neoethos.search.philox4x32-10-counter-mapping.v1\0";
const RANK_SEMANTICS_IDENTITY_DOMAIN_V1: &[u8] =
    b"neoethos.search.gpu-generation-rank-semantics.v1\0";

const METRIC_ROW_BYTES_PER_CANDIDATE_V1: u64 = 104;
const SCENARIO_DESCRIPTOR_BYTES_PER_CANDIDATE_V1: u64 = 56;
const F64_BYTES_V1: u64 = 8;
const TRADE_COUNT_METRIC_INDEX_V1: usize = 8;

/// This is evidence for a particular admitted device plan, never a universal
/// admission floor and never an occupancy heuristic.
const DEVICE_PLAN_EVIDENCE_POPULATION_CAPACITY_16384_V1: usize = 16_384;

#[derive(Debug)]
pub enum GpuResidentGenerationErrorV1 {
    CardPresentCpuGenerationForbidden,
    CardPresentAllowCpuGenerationForbidden,
    ExactCudaOrdinalRequired,
    AuthorityIdentityMismatch(&'static str),
    InvalidGenerationPlan(&'static str),
    GenerationStoreAllocationMismatch,
    GenerationTransferAccountingMismatch,
    CounterMappingOverflow,
    MetricsOnlyCapacityArithmeticOverflow,
    MetricsOnlyCapacityUnavailable,
    MetricsOnlyActualPlanMismatch(&'static str),
    InvalidRunIdentity,
    GenerationNotComplete,
    GenerationAlreadyComplete,
    SelectedSurvivorDiagnosticsForbiddenDuringGeneration,
    ExactStrategyMetricsParityMismatch,
    Device(ResidentGenerationDeviceErrorV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerationPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CounterRngAlgorithmV1 {
    Philox4x32_10,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OfficialGpuPrimitiveAuthorityV1 {
    NvidiaCubDeviceRadixSortPairs,
    NvidiaCubDeviceSelectFlagged,
    NvidiaCubDeviceRunLengthEncode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceParityPurposeV1 {
    DeviceParityTestOnly,
}

#[derive(Clone, Debug)]
pub struct ResolvedGpuGenerationPlanAuthorityV1 {
    generation_count: usize,
    logical_population_count: usize,
    retained_batch_capacity: usize,
    month_capacity: usize,
    search_seed: u64,
    strategy_gene_schema_sha256: String,
    fitness_ordering_semantics_sha256: String,
    crossover_semantics_sha256: String,
    mutation_semantics_sha256: String,
    strategy_metrics_semantics_sha256: String,
}

#[derive(Clone, Debug)]
struct GenerationPermitBindingV1 {
    selected_cuda_ordinal: u32,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    canonical_search_input_receipt_sha256: String,
    resident_input_content_sha256: String,
    strategy_gene_schema_sha256: String,
    fitness_ordering_semantics_sha256: String,
    crossover_semantics_sha256: String,
    mutation_semantics_sha256: String,
    strategy_metrics_semantics_sha256: String,
    rng_mapping_identity_sha256: String,
    rank_semantics_identity_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GenerationTransferAccountingV1 {
    per_generation_metric_rows_readback_count: u64,
    per_generation_explicit_synchronization_count: u64,
    per_generation_host_decision_count: u64,
    per_generation_host_wait_count: u64,
    host_to_device_transfer_count: u64,
    device_to_host_transfer_count: u64,
    final_compact_readback_count: u64,
}

impl GenerationTransferAccountingV1 {
    const fn strict_resident() -> Self {
        Self {
            per_generation_metric_rows_readback_count: 0,
            per_generation_explicit_synchronization_count: 0,
            per_generation_host_decision_count: 0,
            per_generation_host_wait_count: 0,
            host_to_device_transfer_count: 0,
            device_to_host_transfer_count: 0,
            final_compact_readback_count: 0,
        }
    }

    fn validate(self) -> Result<(), GpuResidentGenerationErrorV1> {
        if !(self.per_generation_metric_rows_readback_count == 0
            && self.per_generation_explicit_synchronization_count == 0
            && self.per_generation_host_decision_count == 0
            && self.per_generation_host_wait_count == 0
            && self.host_to_device_transfer_count == 0
            && self.device_to_host_transfer_count == 0
            && self.final_compact_readback_count == 0)
        {
            return Err(GpuResidentGenerationErrorV1::GenerationTransferAccountingMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GpuGenerationMemoryPlanReceiptV1 {
    selected_cuda_ordinal: u32,
    cuda_build_manifest_sha256: String,
    actual_plan_identity_sha256: String,
    metrics_only_fixed_bytes: u64,
    metrics_only_bytes_per_candidate: u64,
    monthly_pnls_bytes_per_candidate: u64,
    month_start_equities_bytes_per_candidate: u64,
    survivor_trade_ledger_bytes: u64,
    logical_population_count: usize,
    retained_batch_capacity: usize,
    active_scenarios: usize,
    generation_chunk_count: usize,
    device_plan_capacity_evidence_16384: bool,
    memory_receipt_identity_sha256: String,
}

#[derive(Debug)]
pub struct GpuResidentGenerationRunV1 {
    population_store: GpuResidentPopulationStoreV1,
    gene_store: GpuResidentGeneStoreV1,
    offspring_store: GpuResidentOffspringStoreV1,
    metric_store: GpuResidentMetricStoreV1,
    rank_store: GpuResidentRankStoreV1,
    selection_store: GpuResidentSelectionStoreV1,
    rng_authority: GpuResidentCounterRngAuthorityV1,
    cub_scratch_arena: CubScratchArenaV1,
    run_stream: GpuRunStreamV1,
    run_identity_sha256: String,
    permit: StrictGpuOnlyFullDiscoveryPermitV2,
    binding: GenerationPermitBindingV1,
    plan: ResolvedGpuGenerationPlanAuthorityV1,
    memory_receipt: GpuGenerationMemoryPlanReceiptV1,
    transfer_accounting: GenerationTransferAccountingV1,
    generation_store_allocation_count: u32,
    next_generation_index: usize,
    final_ready_event: Option<GpuResidentGenerationReadyEventV1>,
    final_diagnostics: Option<FinalBoundedGenerationDiagnosticsReceiptV1>,
}

#[derive(Debug)]
pub struct SealedResidentGenerationOutcomeV1 {
    population_store: GpuResidentPopulationStoreV1,
    gene_store: GpuResidentGeneStoreV1,
    metric_store: GpuResidentMetricStoreV1,
    rank_store: GpuResidentRankStoreV1,
    selection_store: GpuResidentSelectionStoreV1,
    cub_scratch_arena: CubScratchArenaV1,
    run_stream: GpuRunStreamV1,
    final_ready_event: GpuResidentGenerationReadyEventV1,
    final_diagnostics: FinalBoundedGenerationDiagnosticsReceiptV1,
    permit: StrictGpuOnlyFullDiscoveryPermitV2,
    resident_generation_outcome_identity_sha256: String,
    artifact_class: GenerationArtifactClassV1,
    promotion_eligibility: GenerationPromotionEligibilityV1,
}

impl SealedResidentGenerationOutcomeV1 {
    pub(crate) fn consume_in_gpu_post_ga_pipeline_v1(self) -> ResidentPostGaInputV1 {
        ResidentPostGaInputV1::consume_sealed_generation_v1(
            self.population_store,
            self.gene_store,
            self.metric_store,
            self.rank_store,
            self.selection_store,
            self.cub_scratch_arena,
            self.run_stream,
            self.final_ready_event,
            self.final_diagnostics,
            self.permit,
            self.resident_generation_outcome_identity_sha256,
            self.artifact_class == GenerationArtifactClassV1::ResearchOnly,
            self.promotion_eligibility == GenerationPromotionEligibilityV1::NotPromotionEligible,
        )
    }
}

pub(crate) fn begin_gpu_resident_generation_run_v1(
    permit: StrictGpuOnlyFullDiscoveryPermitV2,
    resolved_plan: ResolvedGpuGenerationPlanAuthorityV1,
    actual_allocation_plan: ActualResidentGenerationAllocationPlanV1,
) -> Result<GpuResidentGenerationRunV1, GpuResidentGenerationErrorV1> {
    let binding = validate_generation_plan_against_permit_v1(&permit, &resolved_plan)?;
    let memory_receipt = validate_metrics_only_capacity_from_actual_plan_v1(
        &actual_allocation_plan,
        &binding,
        resolved_plan.logical_population_count,
        resolved_plan.retained_batch_capacity,
        resolved_plan.month_capacity,
    )?;
    let run_identity_sha256 =
        compute_generation_run_identity_v1(&binding, &resolved_plan, &memory_receipt);
    let ResidentGenerationStoreBundleV1 {
        population_store,
        gene_store,
        offspring_store,
        metric_store,
        rank_store,
        selection_store,
        rng_authority,
        cub_scratch_arena,
        run_stream,
        generation_store_allocation_count,
    } = allocate_gpu_resident_generation_stores_v1(
        actual_allocation_plan,
        resolved_plan.logical_population_count,
        resolved_plan.retained_batch_capacity,
        resolved_plan.search_seed,
        &run_identity_sha256,
    )
    .map_err(GpuResidentGenerationErrorV1::Device)?;
    if generation_store_allocation_count != 1 {
        return Err(GpuResidentGenerationErrorV1::GenerationStoreAllocationMismatch);
    }
    let run_identity_bytes = decode_sha256_hex_v1(&run_identity_sha256)?;
    let first_counter =
        checked_counter_mapping_v1(resolved_plan.search_seed, &run_identity_bytes, 0, 0, 0, 0)?;
    rng_authority
        .validate_counter_mapping_v1(
            CounterRngAlgorithmV1::Philox4x32_10,
            first_counter.words,
            &binding.rng_mapping_identity_sha256,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;

    Ok(GpuResidentGenerationRunV1 {
        population_store,
        gene_store,
        offspring_store,
        metric_store,
        rank_store,
        selection_store,
        rng_authority,
        cub_scratch_arena,
        run_stream,
        run_identity_sha256,
        permit,
        binding,
        plan: resolved_plan,
        memory_receipt,
        transfer_accounting: GenerationTransferAccountingV1::strict_resident(),
        generation_store_allocation_count,
        next_generation_index: 0,
        final_ready_event: None,
        final_diagnostics: None,
    })
}

fn execute_resident_generation_v1(
    run: &mut GpuResidentGenerationRunV1,
) -> Result<(), GpuResidentGenerationErrorV1> {
    if run.next_generation_index != 0 {
        return Err(GpuResidentGenerationErrorV1::GenerationAlreadyComplete);
    }
    if run.generation_store_allocation_count != 1 {
        return Err(GpuResidentGenerationErrorV1::GenerationStoreAllocationMismatch);
    }
    let trade_count_metric_index = TRADE_COUNT_METRIC_INDEX_V1;
    if !(trade_count_metric_index == 8) {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyActualPlanMismatch(
            "trade count metric index changed",
        ));
    }
    let evaluation_request = GpuPopulationEvaluationRequestV1 {
        mode: GpuPopulationEvaluationModeV1::MetricsOnly,
        outcomes_device_ptr: DevicePointerV1::Null,
        accepted_trade_total_device_ptr: DevicePointerV1::Null,
        outcome_seed_kernel_launch_count: 0,
        outcome_write_count: 0,
        accepted_trade_total_atomic_count: 0,
        accepted_trade_total_d2h_count: 0,
        strategy_metrics: OnlineStrategyMetricAccumulatorV1::resident_exact_v1(),
        daily_metrics: OnlineDailyMetricAccumulatorV1::resident_exact_v1(),
        monthly_metrics: OnlineMonthlyMetricAccumulatorV1::resident_exact_v1(),
        monthly_pnls_device_array: run.metric_store.monthly_pnls_device_array(),
        month_start_equities_device_array: run.metric_store.month_start_equities_device_array(),
        monthly_mean_variance_order: MonthlyMeanVarianceOrderV1::ExactTwoPass,
        trade_count_metric_index,
    };
    if evaluation_request.mode == GpuPopulationEvaluationModeV1::SelectedSurvivorDiagnostics {
        return Err(
            GpuResidentGenerationErrorV1::SelectedSurvivorDiagnosticsForbiddenDuringGeneration,
        );
    }
    if !(evaluation_request.outcome_seed_kernel_launch_count == 0
        && evaluation_request.outcome_write_count == 0
        && evaluation_request.accepted_trade_total_atomic_count == 0
        && evaluation_request.accepted_trade_total_d2h_count == 0)
    {
        return Err(GpuResidentGenerationErrorV1::GenerationTransferAccountingMismatch);
    }

    while run.next_generation_index < run.plan.generation_count {
        let generation_index = run.next_generation_index;
        let mut previous_chunk_event = run.final_ready_event.take();
        for chunk_index in 0..run.memory_receipt.generation_chunk_count {
            let active_range = checked_generation_chunk_range_v1(
                run.plan.logical_population_count,
                run.plan.retained_batch_capacity,
                chunk_index,
            )?;
            let (metric_rows, ready_event) = enqueue_default_production_metrics_only_v1(
                &mut run.run_stream,
                &run.population_store,
                &mut run.metric_store,
                active_range,
                previous_chunk_event,
                &evaluation_request,
            )?;
            previous_chunk_event = Some(
                run.metric_store
                    .retain_chunk_rows_v1(metric_rows, ready_event)
                    .map_err(GpuResidentGenerationErrorV1::Device)?,
            );
        }

        let rank_ready = rank_and_select_after_event_dependency_v1(
            &mut run.run_stream,
            &run.metric_store,
            &mut run.rank_store,
            &mut run.selection_store,
            previous_chunk_event.ok_or(GpuResidentGenerationErrorV1::InvalidGenerationPlan(
                "every generation requires at least one exact chunk",
            ))?,
            OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRadixSortPairs,
            OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged,
            canonical_f64_total_order_key_v1,
            stable_tie_break_gene_identity_v1,
            &run.binding.rank_semantics_identity_sha256,
            &mut run.cub_scratch_arena,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;
        let parent_ready = launch_device_parent_selection_v1(
            &mut run.run_stream,
            &run.gene_store,
            &run.selection_store,
            &run.rng_authority,
            generation_index,
            rank_ready,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;
        let crossover_ready = launch_device_crossover_v1(
            &mut run.run_stream,
            &run.gene_store,
            &mut run.offspring_store,
            &run.rng_authority,
            generation_index,
            parent_ready,
            &run.binding.crossover_semantics_sha256,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;
        let mutation_ready = launch_device_mutation_v1(
            &mut run.run_stream,
            &mut run.offspring_store,
            &run.rng_authority,
            generation_index,
            crossover_ready,
            &run.binding.mutation_semantics_sha256,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;
        let gene_hash_ready = launch_device_gene_hash_v1(
            &mut run.run_stream,
            &run.offspring_store,
            mutation_ready,
            &run.binding.strategy_gene_schema_sha256,
        )
        .map_err(GpuResidentGenerationErrorV1::Device)?;
        let dedup_ready = run
            .selection_store
            .enqueue_device_dedup_v1(
                &mut run.run_stream,
                &run.offspring_store,
                gene_hash_ready,
                OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode,
                OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged,
                &mut run.cub_scratch_arena,
            )
            .map_err(GpuResidentGenerationErrorV1::Device)?;
        run.final_ready_event = Some(
            rotate_resident_generation_stores_v1(
                &mut run.run_stream,
                &mut run.population_store,
                &mut run.gene_store,
                &mut run.offspring_store,
                dedup_ready,
            )
            .map_err(GpuResidentGenerationErrorV1::Device)?,
        );
        run.next_generation_index = run
            .next_generation_index
            .checked_add(1)
            .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    }

    run.transfer_accounting.validate()?;
    let (final_diagnostics, final_ready_event) = reduce_final_generation_diagnostics_on_device_v1(
        &mut run.run_stream,
        &run.population_store,
        &run.metric_store,
        run.final_ready_event
            .take()
            .ok_or(GpuResidentGenerationErrorV1::InvalidGenerationPlan(
                "the completed generation run has no resident ready event",
            ))?,
    )
    .map_err(GpuResidentGenerationErrorV1::Device)?;
    run.final_diagnostics = Some(final_diagnostics);
    run.final_ready_event = Some(final_ready_event);
    Ok(())
}

fn enqueue_default_production_metrics_only_v1(
    run_stream: &mut GpuRunStreamV1,
    population_store: &GpuResidentPopulationStoreV1,
    metric_store: &mut GpuResidentMetricStoreV1,
    active_range: core::ops::Range<usize>,
    dependency: Option<GpuResidentGenerationReadyEventV1>,
    request: &GpuPopulationEvaluationRequestV1,
) -> Result<
    (
        GpuResidentMetricRowsHandleV1,
        GpuResidentGenerationReadyEventV1,
    ),
    GpuResidentGenerationErrorV1,
> {
    enqueue_population_metrics_only_on_run_stream_v1(
        run_stream,
        population_store,
        metric_store,
        active_range,
        dependency,
        request,
    )
    .map_err(GpuResidentGenerationErrorV1::Device)
}

pub(crate) fn seal_resident_generation_outcome_v1(
    run: GpuResidentGenerationRunV1,
) -> Result<SealedResidentGenerationOutcomeV1, GpuResidentGenerationErrorV1> {
    if run.next_generation_index != run.plan.generation_count {
        return Err(GpuResidentGenerationErrorV1::GenerationNotComplete);
    }
    run.transfer_accounting.validate()?;
    let final_ready_event = run
        .final_ready_event
        .ok_or(GpuResidentGenerationErrorV1::GenerationNotComplete)?;
    let final_diagnostics = run
        .final_diagnostics
        .ok_or(GpuResidentGenerationErrorV1::GenerationNotComplete)?;
    let resident_generation_outcome_identity_sha256 = hash_fields_v1(
        b"neoethos.search.sealed-resident-generation-outcome.v1\0",
        &[
            run.run_identity_sha256.as_bytes(),
            run.memory_receipt.memory_receipt_identity_sha256.as_bytes(),
            run.binding.rank_semantics_identity_sha256.as_bytes(),
            &run.next_generation_index.to_le_bytes(),
        ],
    );

    Ok(SealedResidentGenerationOutcomeV1 {
        population_store: run.population_store,
        gene_store: run.gene_store,
        metric_store: run.metric_store,
        rank_store: run.rank_store,
        selection_store: run.selection_store,
        cub_scratch_arena: run.cub_scratch_arena,
        run_stream: run.run_stream,
        final_ready_event,
        final_diagnostics,
        permit: run.permit,
        resident_generation_outcome_identity_sha256,
        artifact_class: GenerationArtifactClassV1::ResearchOnly,
        promotion_eligibility: GenerationPromotionEligibilityV1::NotPromotionEligible,
    })
}

fn validate_generation_plan_against_permit_v1(
    permit: &StrictGpuOnlyFullDiscoveryPermitV2,
    plan: &ResolvedGpuGenerationPlanAuthorityV1,
) -> Result<GenerationPermitBindingV1, GpuResidentGenerationErrorV1> {
    if plan.generation_count == 0
        || plan.logical_population_count == 0
        || plan.retained_batch_capacity == 0
        || plan.month_capacity == 0
    {
        return Err(GpuResidentGenerationErrorV1::InvalidGenerationPlan(
            "generation, logical population, retained batch and month capacities must be non-zero",
        ));
    }
    if plan.retained_batch_capacity > plan.logical_population_count {
        return Err(GpuResidentGenerationErrorV1::InvalidGenerationPlan(
            "retained batch capacity cannot exceed the logical population",
        ));
    }
    if permit.selected_cuda_ordinal().is_none() {
        return Err(GpuResidentGenerationErrorV1::ExactCudaOrdinalRequired);
    }
    permit
        .require_card_present_strict_gpu_v2()
        .map_err(|kind| match kind {
            crate::full_discovery_gpu_stage_authority_v2::StrictGpuRouteRefusalV2::CpuRoute => {
                GpuResidentGenerationErrorV1::CardPresentCpuGenerationForbidden
            }
            crate::full_discovery_gpu_stage_authority_v2::StrictGpuRouteRefusalV2::CpuAllowed => {
                GpuResidentGenerationErrorV1::CardPresentAllowCpuGenerationForbidden
            }
        })?;
    require_same_identity_v1(
        "strategy gene schema",
        permit.strategy_gene_schema_sha256(),
        &plan.strategy_gene_schema_sha256,
    )?;
    require_same_identity_v1(
        "fitness ordering semantics",
        permit.fitness_ordering_semantics_sha256(),
        &plan.fitness_ordering_semantics_sha256,
    )?;
    require_same_identity_v1(
        "crossover semantics",
        permit.crossover_semantics_sha256(),
        &plan.crossover_semantics_sha256,
    )?;
    require_same_identity_v1(
        "mutation semantics",
        permit.mutation_semantics_sha256(),
        &plan.mutation_semantics_sha256,
    )?;
    require_same_identity_v1(
        "strategy metrics semantics",
        permit.strategy_metrics_semantics_sha256(),
        &plan.strategy_metrics_semantics_sha256,
    )?;

    let rng_mapping_identity_sha256 = compute_rng_mapping_identity_v1();
    let rank_semantics_identity_sha256 = compute_rank_semantics_identity_v1();
    Ok(GenerationPermitBindingV1 {
        selected_cuda_ordinal: permit
            .selected_cuda_ordinal()
            .ok_or(GpuResidentGenerationErrorV1::ExactCudaOrdinalRequired)?,
        cuda_device_identity_sha256: permit.cuda_device_identity_sha256().to_owned(),
        cuda_build_manifest_sha256: permit.cuda_build_manifest_sha256().to_owned(),
        canonical_search_input_receipt_sha256: permit
            .canonical_search_input_receipt_sha256()
            .to_owned(),
        resident_input_content_sha256: permit.resident_input_content_sha256().to_owned(),
        strategy_gene_schema_sha256: plan.strategy_gene_schema_sha256.clone(),
        fitness_ordering_semantics_sha256: plan.fitness_ordering_semantics_sha256.clone(),
        crossover_semantics_sha256: plan.crossover_semantics_sha256.clone(),
        mutation_semantics_sha256: plan.mutation_semantics_sha256.clone(),
        strategy_metrics_semantics_sha256: plan.strategy_metrics_semantics_sha256.clone(),
        rng_mapping_identity_sha256,
        rank_semantics_identity_sha256,
    })
}

fn compute_generation_run_identity_v1(
    binding: &GenerationPermitBindingV1,
    plan: &ResolvedGpuGenerationPlanAuthorityV1,
    memory: &GpuGenerationMemoryPlanReceiptV1,
) -> String {
    hash_fields_v1(
        GENERATION_RUN_IDENTITY_DOMAIN_V1,
        &[
            &binding.selected_cuda_ordinal.to_le_bytes(),
            binding.cuda_device_identity_sha256.as_bytes(),
            binding.cuda_build_manifest_sha256.as_bytes(),
            binding.canonical_search_input_receipt_sha256.as_bytes(),
            binding.resident_input_content_sha256.as_bytes(),
            binding.strategy_gene_schema_sha256.as_bytes(),
            binding.fitness_ordering_semantics_sha256.as_bytes(),
            binding.crossover_semantics_sha256.as_bytes(),
            binding.mutation_semantics_sha256.as_bytes(),
            binding.strategy_metrics_semantics_sha256.as_bytes(),
            binding.rng_mapping_identity_sha256.as_bytes(),
            binding.rank_semantics_identity_sha256.as_bytes(),
            &plan.search_seed.to_le_bytes(),
            &plan.generation_count.to_le_bytes(),
            &plan.logical_population_count.to_le_bytes(),
            memory.memory_receipt_identity_sha256.as_bytes(),
        ],
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PhiloxCounterV1 {
    words: [u32; 4],
}

fn checked_counter_mapping_v1(
    search_seed: u64,
    run_identity_sha256: &[u8; 32],
    generation_index: usize,
    candidate_identity: u64,
    genetic_operator_identity: u32,
    draw_index: u64,
) -> Result<PhiloxCounterV1, GpuResidentGenerationErrorV1> {
    let _algorithm = CounterRngAlgorithmV1::Philox4x32_10;
    let generation_word = u32::try_from(generation_index)
        .map_err(|_| GpuResidentGenerationErrorV1::CounterMappingOverflow)?;
    let draw_word = u32::try_from(draw_index)
        .map_err(|_| GpuResidentGenerationErrorV1::CounterMappingOverflow)?;
    let run_word = u32::from_le_bytes([
        run_identity_sha256[0],
        run_identity_sha256[1],
        run_identity_sha256[2],
        run_identity_sha256[3],
    ]);
    Ok(PhiloxCounterV1 {
        words: [
            generation_word ^ run_word,
            candidate_identity as u32,
            (candidate_identity >> 32) as u32 ^ genetic_operator_identity,
            draw_word ^ search_seed as u32 ^ (search_seed >> 32) as u32,
        ],
    })
}

fn compute_rng_mapping_identity_v1() -> String {
    hash_fields_v1(
        RNG_MAPPING_IDENTITY_DOMAIN_V1,
        &[
            b"CounterRngAlgorithmV1::Philox4x32_10",
            b"search_seed/run_identity_sha256/generation_index/candidate_identity/genetic_operator_identity/draw_index",
        ],
    )
}

fn canonical_f64_total_order_key_v1(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn stable_tie_break_gene_identity_v1(gene_identity: u64) -> u64 {
    gene_identity
}

fn compute_rank_semantics_identity_v1() -> String {
    hash_fields_v1(
        RANK_SEMANTICS_IDENTITY_DOMAIN_V1,
        &[
            b"canonical_f64_total_order_key_v1",
            b"stable_tie_break_gene_identity_v1",
            b"NvidiaCubDeviceRadixSortPairs/run_to_run/same-build/same-device",
        ],
    )
}

fn checked_metrics_only_bytes_without_trade_outcomes_v1(
    month_capacity: usize,
    retained_batch_capacity: usize,
) -> Result<(u64, u64, u64, u64), GpuResidentGenerationErrorV1> {
    let month_capacity = u64::try_from(month_capacity)
        .map_err(|_| GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let retained_batch_capacity = u64::try_from(retained_batch_capacity)
        .map_err(|_| GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let monthly_pnls_bytes_per_candidate = month_capacity
        .checked_mul(F64_BYTES_V1)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let month_start_equities_bytes_per_candidate = month_capacity
        .checked_mul(F64_BYTES_V1)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let metrics_only_bytes_per_candidate = METRIC_ROW_BYTES_PER_CANDIDATE_V1
        .checked_add(SCENARIO_DESCRIPTOR_BYTES_PER_CANDIDATE_V1)
        .and_then(|bytes| bytes.checked_add(monthly_pnls_bytes_per_candidate))
        .and_then(|bytes| bytes.checked_add(month_start_equities_bytes_per_candidate))
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let retained_candidate_bytes = metrics_only_bytes_per_candidate
        .checked_mul(retained_batch_capacity)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    Ok((
        metrics_only_bytes_per_candidate,
        monthly_pnls_bytes_per_candidate,
        month_start_equities_bytes_per_candidate,
        retained_candidate_bytes,
    ))
}

fn validate_metrics_only_capacity_from_actual_plan_v1(
    actual_plan: &ActualResidentGenerationAllocationPlanV1,
    binding: &GenerationPermitBindingV1,
    logical_population_count: usize,
    retained_batch_capacity: usize,
    month_capacity: usize,
) -> Result<GpuGenerationMemoryPlanReceiptV1, GpuResidentGenerationErrorV1> {
    if !(retained_batch_capacity >= 1) {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyCapacityUnavailable);
    }
    let active_scenarios = logical_population_count.min(retained_batch_capacity);
    if !(active_scenarios <= retained_batch_capacity) || active_scenarios == 0 {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyCapacityUnavailable);
    }
    let generation_chunk_count =
        checked_generation_chunk_count_v1(logical_population_count, retained_batch_capacity)?;
    let covered_logical_population = checked_covered_population_v1(
        logical_population_count,
        retained_batch_capacity,
        generation_chunk_count,
    )?;
    if !(covered_logical_population == logical_population_count) {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyActualPlanMismatch(
            "exact chunk schedule does not cover the logical population",
        ));
    }
    let (
        metrics_only_bytes_per_candidate,
        monthly_pnls_bytes_per_candidate,
        month_start_equities_bytes_per_candidate,
        retained_candidate_bytes,
    ) = checked_metrics_only_bytes_without_trade_outcomes_v1(
        month_capacity,
        retained_batch_capacity,
    )?;
    let metrics_only_fixed_bytes = actual_plan.metrics_only_fixed_bytes();
    let required_device_bytes = metrics_only_fixed_bytes
        .checked_add(retained_candidate_bytes)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let reusable_device_bytes = actual_plan
        .same_context_free_bytes()
        .checked_sub(actual_plan.full_discovery_reserve_bytes())
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityUnavailable)?;
    if required_device_bytes > reusable_device_bytes {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyCapacityUnavailable);
    }
    if actual_plan.retained_batch_capacity() != retained_batch_capacity
        || actual_plan.metrics_only_bytes_per_candidate() != metrics_only_bytes_per_candidate
        || actual_plan.retained_candidate_bytes() != retained_candidate_bytes
        || !actual_plan_is_strict_metrics_only_v1(actual_plan)
    {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyActualPlanMismatch(
            "actual retained allocation differs from the checked metrics-only plan",
        ));
    }
    Ok(seal_generation_memory_receipt_v1(
        actual_plan,
        binding,
        GenerationMemorySizingV1 {
            metrics_only_fixed_bytes,
            metrics_only_bytes_per_candidate,
            monthly_pnls_bytes_per_candidate,
            month_start_equities_bytes_per_candidate,
            logical_population_count,
            retained_batch_capacity,
            active_scenarios,
            generation_chunk_count,
        },
    ))
}

struct GenerationMemorySizingV1 {
    metrics_only_fixed_bytes: u64,
    metrics_only_bytes_per_candidate: u64,
    monthly_pnls_bytes_per_candidate: u64,
    month_start_equities_bytes_per_candidate: u64,
    logical_population_count: usize,
    retained_batch_capacity: usize,
    active_scenarios: usize,
    generation_chunk_count: usize,
}

fn seal_generation_memory_receipt_v1(
    actual_plan: &ActualResidentGenerationAllocationPlanV1,
    binding: &GenerationPermitBindingV1,
    sizing: GenerationMemorySizingV1,
) -> GpuGenerationMemoryPlanReceiptV1 {
    let survivor_trade_ledger_bytes = 0;
    let device_plan_capacity_evidence_16384 =
        device_plan_capacity_evidence_16384_v1(sizing.retained_batch_capacity);
    let actual_plan_identity_sha256 = actual_plan.actual_plan_identity_sha256().to_owned();
    let memory_receipt_identity_sha256 = hash_fields_v1(
        GENERATION_MEMORY_IDENTITY_DOMAIN_V1,
        &[
            &binding.selected_cuda_ordinal.to_le_bytes(),
            binding.cuda_build_manifest_sha256.as_bytes(),
            actual_plan_identity_sha256.as_bytes(),
            &sizing.metrics_only_fixed_bytes.to_le_bytes(),
            &sizing.metrics_only_bytes_per_candidate.to_le_bytes(),
            &sizing.monthly_pnls_bytes_per_candidate.to_le_bytes(),
            &sizing
                .month_start_equities_bytes_per_candidate
                .to_le_bytes(),
            &survivor_trade_ledger_bytes.to_le_bytes(),
            &sizing.logical_population_count.to_le_bytes(),
            &sizing.retained_batch_capacity.to_le_bytes(),
            &sizing.active_scenarios.to_le_bytes(),
            &sizing.generation_chunk_count.to_le_bytes(),
        ],
    );
    GpuGenerationMemoryPlanReceiptV1 {
        selected_cuda_ordinal: binding.selected_cuda_ordinal,
        cuda_build_manifest_sha256: binding.cuda_build_manifest_sha256.clone(),
        actual_plan_identity_sha256,
        metrics_only_fixed_bytes: sizing.metrics_only_fixed_bytes,
        metrics_only_bytes_per_candidate: sizing.metrics_only_bytes_per_candidate,
        monthly_pnls_bytes_per_candidate: sizing.monthly_pnls_bytes_per_candidate,
        month_start_equities_bytes_per_candidate: sizing.month_start_equities_bytes_per_candidate,
        survivor_trade_ledger_bytes,
        logical_population_count: sizing.logical_population_count,
        retained_batch_capacity: sizing.retained_batch_capacity,
        active_scenarios: sizing.active_scenarios,
        generation_chunk_count: sizing.generation_chunk_count,
        device_plan_capacity_evidence_16384,
        memory_receipt_identity_sha256,
    }
}

fn actual_plan_is_strict_metrics_only_v1(
    actual_plan: &ActualResidentGenerationAllocationPlanV1,
) -> bool {
    actual_plan.survivor_trade_ledger_bytes() == 0
}

const fn device_plan_capacity_evidence_16384_v1(retained_batch_capacity: usize) -> bool {
    retained_batch_capacity >= DEVICE_PLAN_EVIDENCE_POPULATION_CAPACITY_16384_V1
}

fn checked_generation_chunk_count_v1(
    logical_population_count: usize,
    retained_batch_capacity: usize,
) -> Result<usize, GpuResidentGenerationErrorV1> {
    if logical_population_count == 0 || retained_batch_capacity == 0 {
        return Err(GpuResidentGenerationErrorV1::MetricsOnlyCapacityUnavailable);
    }
    logical_population_count
        .checked_add(retained_batch_capacity - 1)
        .and_then(|value| value.checked_div(retained_batch_capacity))
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)
}

fn checked_generation_chunk_range_v1(
    logical_population_count: usize,
    retained_batch_capacity: usize,
    chunk_index: usize,
) -> Result<core::ops::Range<usize>, GpuResidentGenerationErrorV1> {
    let start = chunk_index
        .checked_mul(retained_batch_capacity)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    let end = start
        .checked_add(retained_batch_capacity)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?
        .min(logical_population_count);
    if start >= end {
        return Err(GpuResidentGenerationErrorV1::InvalidGenerationPlan(
            "generation chunk is empty or outside the logical population",
        ));
    }
    Ok(start..end)
}

fn checked_covered_population_v1(
    logical_population_count: usize,
    retained_batch_capacity: usize,
    generation_chunk_count: usize,
) -> Result<usize, GpuResidentGenerationErrorV1> {
    let last_chunk_index = generation_chunk_count
        .checked_sub(1)
        .ok_or(GpuResidentGenerationErrorV1::MetricsOnlyCapacityArithmeticOverflow)?;
    Ok(checked_generation_chunk_range_v1(
        logical_population_count,
        retained_batch_capacity,
        last_chunk_index,
    )?
    .end)
}

fn require_same_identity_v1(
    _label: &'static str,
    authoritative: &str,
    planned: &str,
) -> Result<(), GpuResidentGenerationErrorV1> {
    if authoritative.len() != 64
        || !authoritative.bytes().all(|byte| byte.is_ascii_hexdigit())
        || authoritative != planned
    {
        return Err(GpuResidentGenerationErrorV1::AuthorityIdentityMismatch(
            _label,
        ));
    }
    Ok(())
}

fn hash_fields_v1(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hex_lower_v1(&hasher.finalize())
}

fn decode_sha256_hex_v1(value: &str) -> Result<[u8; 32], GpuResidentGenerationErrorV1> {
    if value.len() != 64 {
        return Err(GpuResidentGenerationErrorV1::InvalidRunIdentity);
    }
    let mut bytes = [0_u8; 32];
    for (index, output) in bytes.iter_mut().enumerate() {
        let start = index * 2;
        *output = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| GpuResidentGenerationErrorV1::InvalidRunIdentity)?;
    }
    Ok(bytes)
}

fn hex_lower_v1(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(all(test, feature = "gpu-b-native"))]
fn read_metrics_for_device_parity_test_only_v1(
    mut resident_metrics: DeviceParityMetricRowsV1,
    expected: &[neoethos_core::StrategyMetrics],
) -> Result<(), GpuResidentGenerationErrorV1> {
    let _purpose = DeviceParityPurposeV1::DeviceParityTestOnly;
    resident_metrics
        .wait()
        .map_err(GpuResidentGenerationErrorV1::Device)?;
    let actual = resident_metrics
        .read_metrics()
        .map_err(GpuResidentGenerationErrorV1::Device)?;
    validate_online_metrics_against_canonical_strategy_metrics_v1(&actual, expected)
}

#[cfg(all(test, feature = "gpu-b-native"))]
fn validate_online_metrics_against_canonical_strategy_metrics_v1(
    actual: &[neoethos_core::StrategyMetrics],
    expected: &[neoethos_core::StrategyMetrics],
) -> Result<(), GpuResidentGenerationErrorV1> {
    if actual != expected {
        return Err(GpuResidentGenerationErrorV1::ExactStrategyMetricsParityMismatch);
    }
    Ok(())
}
