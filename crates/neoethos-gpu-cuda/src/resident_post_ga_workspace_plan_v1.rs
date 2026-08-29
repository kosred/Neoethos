//! Source-only post-GA workspace authority.
//!
//! This module deliberately has no crate export or production caller yet. It
//! plans bytes from sealed metadata produced by the admitted CUDA run; it does
//! not inspect live device ownership and never asks the driver for a newer
//! memory snapshot. A future integration must make the sealed ingress types
//! constructible only by their owning runtime modules.

use sha2::{Digest, Sha256};

pub(crate) const RESIDENT_POST_GA_WORKSPACE_SEMANTICS_V1: &str =
    "neoethos.resident-post-ga-workspace-plan.v1";
const DEVICE_ALIGNMENT_BYTES_V1: u64 = 256;
const POPULATION_METRIC_ROW_BYTES_V1: u64 = 104;
const POPULATION_SCENARIO_DESCRIPTOR_BYTES_V1: u64 = 56;
const POPULATION_F64_BYTES_V1: u64 = 8;
const NEO_POPULATION_OUTCOME_BYTES_V1: u64 = 72;
const QUALITY_BOOTSTRAP_ITERATIONS_V1: u64 = 1_000;
const QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1: u64 = 5;
const TERNARY_BITS_PER_CELL_V1: u64 = 2;
const TERNARY_NEGATIVE_BITS_V1: u8 = 0b00;
const TERNARY_ZERO_BITS_V1: u8 = 0b01;
const TERNARY_POSITIVE_BITS_V1: u8 = 0b10;
const TERNARY_INVALID_BITS_V1: u8 = 0b11;

type IdentitySha256V1 = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentPostGaWorkspacePlanErrorV1 {
    ArithmeticOverflow(&'static str),
    ZeroExtent(&'static str),
    ZeroIdentity(&'static str),
    IdentityMismatch(&'static str),
    GenerationComponentMismatch,
    InvalidSignalAuthority,
    InvalidQualityAuthority,
    InvalidScenarioCoverage,
    InvalidMetricsWorkspaceReceipt,
    InvalidCubScratchReceipt,
    InvalidPhaseEventProof,
    LedgerCapacityExceeded,
    LedgerReceiptMismatch,
    GenerationReceiptMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InheritedGenerationAllocationDispositionV1 {
    ReplacesResidentGeneticEvolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetricsWorkspaceDispositionV1 {
    ReuseInheritedExactCapacity,
    DedicatedContiguousCharge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CubScratchDispositionV1 {
    ReuseInheritedGenerationScratch,
    DedicatedContiguousPostGaScratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QualityAccumulatorSemanticsV1 {
    DeterministicDailyBlockBootstrapV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QualityPnlBranchV1 {
    DenseDailyPnl,
    BoundedTradePnlFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectedLedgerPassV1 {
    ExactTradeCount,
    CubExclusiveScan,
    CheckedBudget,
    CompactReplayWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentPostGaPhaseKindV1 {
    GenerationReady,
    QualityAndMonteCarlo,
    PortfolioConstraints,
    SelectedLedgerAndValidation,
    RobustnessTail,
    FinalCompactSeal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentPostGaWorkspaceAuthorityV1 {
    ResearchOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentPostGaPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentPostGaIntegrationStateV1 {
    Unwired,
}

/// Exact byte projection of the one native allocation retained by Generation
/// V1. The fields mirror the checked native layout and stay private.
pub(crate) struct ResidentGenerationAllocationChargeProjectionV1 {
    logical_gene_scalar_bytes: u64,
    logical_gene_index_bytes: u64,
    logical_gene_weight_bytes: u64,
    offspring_bytes: u64,
    metric_row_bytes: u64,
    rank_key_bytes: u64,
    selection_bytes: u64,
    dedup_hash_bytes: u64,
    cub_scratch_bytes: u64,
    retained_evaluation_workspace_bytes: u64,
    total_device_bytes: u64,
}

/// Move-only generation charge sealed by the future Generation V1 adapter.
pub(crate) struct SealedResidentGenerationAllocationChargeProjectionV1 {
    projection: ResidentGenerationAllocationChargeProjectionV1,
    selected_cuda_ordinal: u32,
    device_uuid_sha256: IdentitySha256V1,
    primary_context_identity_sha256: IdentitySha256V1,
    run_stream_identity_sha256: IdentitySha256V1,
    cuda_build_manifest_sha256: IdentitySha256V1,
    generation_semantics_sha256: IdentitySha256V1,
    generation_allocation_receipt_sha256: IdentitySha256V1,
}

/// Exact retained population evaluator workspace owned by the same A1
/// lifetime. It is not part of the Generation V1 native allocation above.
pub(crate) struct SealedMetricsOnlyWorkspaceReceiptV1 {
    scenario_capacity: u64,
    month_capacity: u64,
    metric_rows_bytes: u64,
    monthly_pnl_bytes: u64,
    month_start_equity_bytes: u64,
    scenario_descriptor_bytes: u64,
    total_device_bytes: u64,
    outcome_bytes: u64,
    accepted_trade_total_bytes: u64,
    primary_context_identity_sha256: IdentitySha256V1,
    run_stream_identity_sha256: IdentitySha256V1,
    cuda_build_manifest_sha256: IdentitySha256V1,
    receipt_identity_sha256: IdentitySha256V1,
}

/// Exact CUB query made on the admitted stream and pinned build.
pub(crate) struct SealedCubScratchQueryReceiptV1 {
    inherited_generation_cub_scratch_bytes: u64,
    post_ga_required_cub_scratch_bytes: u64,
    cub_query_receipt_sha256: IdentitySha256V1,
    cuda_toolkit_build_sha256: IdentitySha256V1,
    cccl_build_sha256: IdentitySha256V1,
    cuda_build_manifest_sha256: IdentitySha256V1,
    same_admitted_stream_sha256: IdentitySha256V1,
}

/// An algorithm-specific exact charge. The producing CUDA stage owns its
/// layout semantics; this planner only accepts the sealed charge and identity.
pub(crate) struct SealedExactPostGaAlgorithmChargeV1 {
    device_bytes: u64,
    algorithm_semantics_sha256: IdentitySha256V1,
    algorithm_build_sha256: IdentitySha256V1,
    charge_receipt_sha256: IdentitySha256V1,
}

/// All content- and configuration-derived extents needed by the post-GA plan.
/// No constructor is provided in this unwired boundary.
pub(crate) struct SealedPostGaResolvedExtentsV1 {
    selected_cuda_ordinal: u32,
    device_uuid_sha256: IdentitySha256V1,
    primary_context_identity_sha256: IdentitySha256V1,
    run_stream_identity_sha256: IdentitySha256V1,
    cuda_build_manifest_sha256: IdentitySha256V1,
    generation_semantics_sha256: IdentitySha256V1,
    scoring_semantics_sha256: IdentitySha256V1,
    novelty_semantics_sha256: IdentitySha256V1,
    resolved_config_sha256: IdentitySha256V1,
    canonical_input_receipt_sha256: IdentitySha256V1,
    logical_candidate_count: u64,
    survivor_capacity: u64,
    row_count: u64,
    active_candidate_count: u64,
    active_chunk_count: u64,
    exact_logical_candidate_coverage: bool,
    selected_portfolio_capacity: u64,
    sealed_total_trade_count_ceiling: u64,
    quality_accumulator_bytes_per_candidate: u64,
    quality_accumulator_layout_sha256: IdentitySha256V1,
    quality_rng_counter_mapping_sha256: IdentitySha256V1,
    sealed_max_distinct_traded_day_capacity: u64,
    sealed_max_trade_count_per_active_chunk: u64,
    month_capacity: u64,
    parameter_mc_active_scenario_capacity: u64,
    exact_parameter_mc_scenario_count: u64,
    exact_sensitivity_scenario_count: u64,
    exact_cost_band_scenario_count: u64,
    exact_parameter_mc_chunk_count: u64,
    exact_scenario_chunk_coverage: bool,
    compact_candidate_to_parent_map_sha256: IdentitySha256V1,
    candidate_order_sha256: IdentitySha256V1,
    row_order_sha256: IdentitySha256V1,
    signal_semantics_sha256: IdentitySha256V1,
    selected_gene_order_sha256: IdentitySha256V1,
    settings_cost_conversion_sha256: IdentitySha256V1,
    rng_counter_mapping_sha256: IdentitySha256V1,
    native_build_sha256: IdentitySha256V1,
    portfolio_workspace: SealedExactPostGaAlgorithmChargeV1,
    validation_workspace: SealedExactPostGaAlgorithmChargeV1,
    robustness_workspace: SealedExactPostGaAlgorithmChargeV1,
    bounded_final_compact_readback_bytes: u64,
    final_compact_result_semantics_sha256: IdentitySha256V1,
}

pub(crate) struct ResidentPostGaPhaseEventEdgeV1 {
    producer_phase: ResidentPostGaPhaseKindV1,
    consumer_phase: ResidentPostGaPhaseKindV1,
    producer_event_identity_sha256: IdentitySha256V1,
    consumer_dependency_identity_sha256: IdentitySha256V1,
    same_primary_context_identity_sha256: IdentitySha256V1,
    same_run_stream_identity_sha256: IdentitySha256V1,
    typed_non_overlap_proof: bool,
}

pub(crate) struct ResidentPostGaEventLifetimeProofV1 {
    primary_context_identity_sha256: IdentitySha256V1,
    run_stream_identity_sha256: IdentitySha256V1,
    generation_ready_event_identity_sha256: IdentitySha256V1,
    quality_complete_event_identity_sha256: IdentitySha256V1,
    portfolio_complete_event_identity_sha256: IdentitySha256V1,
    validation_complete_event_identity_sha256: IdentitySha256V1,
    robustness_complete_event_identity_sha256: IdentitySha256V1,
    ordered_edges: [ResidentPostGaPhaseEventEdgeV1; 5],
}

pub(crate) struct ResidentPostGaWorkspacePreflightV1 {
    inherited_generation: SealedResidentGenerationAllocationChargeProjectionV1,
    inherited_metrics_workspace: SealedMetricsOnlyWorkspaceReceiptV1,
    cub_scratch: SealedCubScratchQueryReceiptV1,
    resolved: SealedPostGaResolvedExtentsV1,
    event_lifetimes: ResidentPostGaEventLifetimeProofV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedTernarySurvivorStoreChargeV1 {
    negative_encoding: u8,
    zero_encoding: u8,
    positive_encoding: u8,
    invalid_encoding: u8,
    signal_cell_count: u64,
    signal_bit_count: u64,
    packed_signal_bytes: u64,
    compact_to_parent_map_bytes: u64,
    candidate_order_bytes: u64,
    invalid_code_faults_and_invalidates_receipt: bool,
    compact_candidate_to_parent_map_sha256: IdentitySha256V1,
    candidate_order_sha256: IdentitySha256V1,
    row_order_sha256: IdentitySha256V1,
    signal_semantics_sha256: IdentitySha256V1,
    total_device_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QualityAccumulatorChargeV1 {
    semantics: QualityAccumulatorSemanticsV1,
    quality_accumulator_layout_sha256: IdentitySha256V1,
    quality_accumulator_bytes: u64,
    dense_day_pnl_bytes: u64,
    traded_day_flags_bytes: u64,
    bounded_trade_pnl_bytes: u64,
    branch_tag_bytes: u64,
    sealed_day_count_bytes: u64,
    bootstrap_drawdown_bytes: u64,
    active_candidate_count: u64,
    active_chunk_count: u64,
    exact_logical_candidate_coverage: bool,
    chronological_traded_day_compaction: bool,
    sealed_max_distinct_traded_day_capacity: u64,
    sealed_max_trade_count_per_active_chunk: u64,
    candidate_order_sha256: IdentitySha256V1,
    quality_rng_counter_mapping_sha256: IdentitySha256V1,
    persistent_device_bytes: u64,
    phase_device_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParameterMonteCarloWorkspaceChargeV1 {
    disposition: MetricsWorkspaceDispositionV1,
    exact_parameter_mc_scenario_count: u64,
    exact_sensitivity_scenario_count: u64,
    exact_cost_band_scenario_count: u64,
    active_scenario_capacity: u64,
    scenario_chunk_count: u64,
    exact_scenario_chunk_coverage: bool,
    metric_rows_bytes: u64,
    monthly_pnl_bytes: u64,
    month_start_equity_bytes: u64,
    scenario_descriptor_bytes: u64,
    required_metrics_workspace_bytes: u64,
    dedicated_metrics_workspace_bytes: u64,
    outcome_bytes: u64,
    accepted_trade_total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactSelectedTradeLedgerCeilingV1 {
    selected_capacity: u64,
    total_trade_count_ceiling: u64,
    selected_trade_count_bytes: u64,
    selected_trade_offset_bytes: u64,
    selected_trade_outcome_bytes: u64,
    sealed_compact_ledger_byte_ceiling: u64,
    passes: [SelectedLedgerPassV1; 4],
    selected_gene_order_sha256: IdentitySha256V1,
    settings_cost_conversion_sha256: IdentitySha256V1,
    rng_counter_mapping_sha256: IdentitySha256V1,
    native_build_sha256: IdentitySha256V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CubScratchChargeV1 {
    disposition: CubScratchDispositionV1,
    inherited_generation_cub_scratch_bytes: u64,
    post_ga_required_cub_scratch_bytes: u64,
    dedicated_cub_scratch_bytes: u64,
    cub_query_receipt_sha256: IdentitySha256V1,
    cuda_toolkit_build_sha256: IdentitySha256V1,
    cccl_build_sha256: IdentitySha256V1,
    cuda_build_manifest_sha256: IdentitySha256V1,
    same_admitted_stream_sha256: IdentitySha256V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentPostGaPhaseArenaChargeV1 {
    quality_and_monte_carlo_phase_bytes: u64,
    portfolio_constraints_phase_bytes: u64,
    selected_ledger_and_validation_phase_bytes: u64,
    robustness_tail_phase_bytes: u64,
    phase_arena_device_bytes: u64,
}

#[must_use = "a post-GA workspace plan is research authority, not a promotion permit"]
pub(crate) struct ResidentPostGaWorkspacePlanV1 {
    authority: ResidentPostGaWorkspaceAuthorityV1,
    promotion_eligibility: ResidentPostGaPromotionEligibilityV1,
    integration_state: ResidentPostGaIntegrationStateV1,
    inherited_generation_disposition: InheritedGenerationAllocationDispositionV1,
    inherited_generation_charge_count: u8,
    inherited_generation: ResidentGenerationAllocationChargeProjectionV1,
    inherited_metrics_workspace_device_bytes: u64,
    survivor_signals: PackedTernarySurvivorStoreChargeV1,
    quality_accumulator: QualityAccumulatorChargeV1,
    parameter_monte_carlo: ParameterMonteCarloWorkspaceChargeV1,
    selected_trade_ledger: CompactSelectedTradeLedgerCeilingV1,
    cub_scratch: CubScratchChargeV1,
    event_lifetimes: ResidentPostGaEventLifetimeProofV1,
    phase_arena: ResidentPostGaPhaseArenaChargeV1,
    always_resident_device_bytes: u64,
    bounded_final_compact_readback_bytes: u64,
    total_device_bytes: u64,
    workspace_plan_identity_sha256: IdentitySha256V1,
    generation_allocation_receipt_sha256: IdentitySha256V1,
}

/// A future A1 adapter will return this private receipt after the live
/// generation owner has been moved into its post-GA typestate.
pub(crate) struct A1GenerationAllocationRuntimeReceiptV1 {
    generation_allocation_total_device_bytes: u64,
    generation_component_sum: u64,
    generation_allocation_receipt_sha256: IdentitySha256V1,
    workspace_plan_identity_sha256: IdentitySha256V1,
}

/// Runtime-selected compact ledger evidence. It carries identities and counts,
/// never storage ownership.
pub(crate) struct CompactSelectedTradeLedgerRuntimeReceiptV1 {
    selected_count: u64,
    exact_total_trade_count: u64,
    exact_compact_ledger_bytes: u64,
    written_total_trade_count: u64,
    first_pass_count_sha256: IdentitySha256V1,
    written_segment_length_sha256: IdentitySha256V1,
    selected_gene_order_sha256: IdentitySha256V1,
    settings_cost_conversion_sha256: IdentitySha256V1,
    rng_counter_mapping_sha256: IdentitySha256V1,
    native_build_sha256: IdentitySha256V1,
}

/// Per-candidate device-produced evidence selecting the canonical quality
/// bootstrap branch. The receipt cannot be caller-created outside this module.
pub(crate) struct SealedQualityCandidateBranchReceiptV1 {
    candidate_identity_sha256: IdentitySha256V1,
    candidate_order_sha256: IdentitySha256V1,
    quality_accumulator_layout_sha256: IdentitySha256V1,
    quality_rng_counter_mapping_sha256: IdentitySha256V1,
    sealed_distinct_traded_day_count: u64,
    branch: QualityPnlBranchV1,
}

pub(crate) fn checked_plan_resident_post_ga_workspace_v1(
    preflight: ResidentPostGaWorkspacePreflightV1,
) -> Result<ResidentPostGaWorkspacePlanV1, ResidentPostGaWorkspacePlanErrorV1> {
    let ResidentPostGaWorkspacePreflightV1 {
        inherited_generation,
        inherited_metrics_workspace,
        cub_scratch,
        resolved,
        event_lifetimes,
    } = preflight;

    validate_cross_receipt_identity_v1(
        &inherited_generation,
        &inherited_metrics_workspace,
        &cub_scratch,
        &resolved,
    )?;
    let sealed_generation = inherited_generation;
    let inherited_generation = &sealed_generation.projection;
    let generation_component_sum = checked_sum_generation_components_v1(inherited_generation)?;
    if !(generation_component_sum == inherited_generation.total_device_bytes) {
        return Err(ResidentPostGaWorkspacePlanErrorV1::GenerationComponentMismatch);
    }
    validate_metrics_only_receipt_v1(&inherited_metrics_workspace)?;
    validate_phase_event_lifetimes_v1(&event_lifetimes, &resolved)?;
    validate_resolved_extents_v1(&resolved)?;

    let survivor_signals = checked_packed_ternary_survivor_store_v1(&resolved)?;
    let quality_accumulator = checked_quality_accumulator_charge_v1(&resolved)?;
    let parameter_monte_carlo =
        checked_parameter_mc_workspace_charge_v1(&resolved, &inherited_metrics_workspace)?;
    let selected_trade_ledger = checked_compact_selected_trade_ledger_ceiling_v1(&resolved)?;
    let cub_scratch = checked_cub_scratch_charge_v1(&cub_scratch)?;

    let always_resident_device_bytes = checked_add(
        inherited_generation.total_device_bytes,
        inherited_metrics_workspace.total_device_bytes,
    )
    .and_then(|total| checked_add(total, survivor_signals.total_device_bytes))
    .and_then(|total| checked_add(total, quality_accumulator.persistent_device_bytes))?;
    let phase_arena = checked_phase_arena_charge_v1(
        &resolved,
        &quality_accumulator,
        &parameter_monte_carlo,
        &selected_trade_ledger,
        &cub_scratch,
        &event_lifetimes,
    )?;
    let phase_arena_device_bytes = phase_arena.phase_arena_device_bytes;
    let resident_and_arena = checked_add(always_resident_device_bytes, phase_arena_device_bytes)?;
    let total_device_bytes = checked_add(
        resident_and_arena,
        resolved.bounded_final_compact_readback_bytes,
    )?;

    let workspace_plan_identity_sha256 = hash_workspace_plan_identity_v1(
        &sealed_generation,
        &inherited_metrics_workspace,
        &resolved,
        &survivor_signals,
        &quality_accumulator,
        &parameter_monte_carlo,
        &selected_trade_ledger,
        &cub_scratch,
        &event_lifetimes,
        &phase_arena,
        always_resident_device_bytes,
        total_device_bytes,
    );

    Ok(ResidentPostGaWorkspacePlanV1 {
        authority: ResidentPostGaWorkspaceAuthorityV1::ResearchOnly,
        promotion_eligibility: ResidentPostGaPromotionEligibilityV1::NotPromotionEligible,
        integration_state: ResidentPostGaIntegrationStateV1::Unwired,
        inherited_generation_disposition:
            InheritedGenerationAllocationDispositionV1::ReplacesResidentGeneticEvolution,
        inherited_generation_charge_count: 1,
        inherited_generation: sealed_generation.projection,
        inherited_metrics_workspace_device_bytes: inherited_metrics_workspace.total_device_bytes,
        survivor_signals,
        quality_accumulator,
        parameter_monte_carlo,
        selected_trade_ledger,
        cub_scratch,
        event_lifetimes,
        phase_arena,
        always_resident_device_bytes,
        bounded_final_compact_readback_bytes: resolved.bounded_final_compact_readback_bytes,
        total_device_bytes,
        workspace_plan_identity_sha256,
        generation_allocation_receipt_sha256: sealed_generation
            .generation_allocation_receipt_sha256,
    })
}

pub(crate) fn validate_a1_generation_receipt_against_workspace_plan_v1(
    receipt: &A1GenerationAllocationRuntimeReceiptV1,
    plan: &ResidentPostGaWorkspacePlanV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    let component_sum = checked_sum_generation_components_v1(&plan.inherited_generation)?;
    if receipt.generation_allocation_total_device_bytes
        != plan.inherited_generation.total_device_bytes
        || receipt.generation_component_sum != component_sum
        || receipt.generation_allocation_receipt_sha256 != plan.generation_allocation_receipt_sha256
        || receipt.workspace_plan_identity_sha256 != plan.workspace_plan_identity_sha256
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::GenerationReceiptMismatch);
    }
    Ok(())
}

pub(crate) fn validate_compact_ledger_runtime_receipt_v1(
    receipt: &CompactSelectedTradeLedgerRuntimeReceiptV1,
    plan: &ResidentPostGaWorkspacePlanV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    let selected_count = receipt.selected_count;
    let exact_total_trade_count = receipt.exact_total_trade_count;
    let exact_compact_ledger_bytes =
        checked_compact_ledger_content_bytes_v1(selected_count, exact_total_trade_count)?;
    let sealed_compact_ledger_byte_ceiling = plan
        .selected_trade_ledger
        .sealed_compact_ledger_byte_ceiling;
    let written_total_trade_count = receipt.written_total_trade_count;
    if selected_count > plan.selected_trade_ledger.selected_capacity
        || exact_total_trade_count > plan.selected_trade_ledger.total_trade_count_ceiling
        || !(exact_compact_ledger_bytes <= sealed_compact_ledger_byte_ceiling)
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::LedgerCapacityExceeded);
    }
    if receipt.exact_compact_ledger_bytes != exact_compact_ledger_bytes
        || !(written_total_trade_count == exact_total_trade_count)
        || is_zero_identity_v1(receipt.first_pass_count_sha256)
        || is_zero_identity_v1(receipt.written_segment_length_sha256)
        || receipt.first_pass_count_sha256 != receipt.written_segment_length_sha256
        || receipt.selected_gene_order_sha256
            != plan.selected_trade_ledger.selected_gene_order_sha256
        || receipt.settings_cost_conversion_sha256
            != plan.selected_trade_ledger.settings_cost_conversion_sha256
        || receipt.rng_counter_mapping_sha256
            != plan.selected_trade_ledger.rng_counter_mapping_sha256
        || receipt.native_build_sha256 != plan.selected_trade_ledger.native_build_sha256
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::LedgerReceiptMismatch);
    }
    Ok(())
}

pub(crate) fn resolve_quality_pnl_branch_v1(
    sealed_distinct_traded_day_count: u64,
) -> QualityPnlBranchV1 {
    if sealed_distinct_traded_day_count >= QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1 {
        QualityPnlBranchV1::DenseDailyPnl
    } else if sealed_distinct_traded_day_count < QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1 {
        QualityPnlBranchV1::BoundedTradePnlFallback
    } else {
        unreachable!("the sealed day-count predicates are exhaustive")
    }
}

pub(crate) fn validate_quality_candidate_branch_receipt_v1(
    receipt: &SealedQualityCandidateBranchReceiptV1,
    plan: &ResidentPostGaWorkspacePlanV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    let expected_branch = resolve_quality_pnl_branch_v1(receipt.sealed_distinct_traded_day_count);
    if is_zero_identity_v1(receipt.candidate_identity_sha256)
        || receipt.candidate_order_sha256 != plan.quality_accumulator.candidate_order_sha256
        || receipt.quality_accumulator_layout_sha256
            != plan.quality_accumulator.quality_accumulator_layout_sha256
        || receipt.quality_rng_counter_mapping_sha256
            != plan.quality_accumulator.quality_rng_counter_mapping_sha256
        || receipt.sealed_distinct_traded_day_count
            > plan
                .quality_accumulator
                .sealed_max_distinct_traded_day_capacity
        || receipt.branch != expected_branch
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidQualityAuthority);
    }
    Ok(())
}

fn checked_sum_generation_components_v1(
    inherited_generation: &ResidentGenerationAllocationChargeProjectionV1,
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    [
        inherited_generation.logical_gene_scalar_bytes,
        inherited_generation.logical_gene_index_bytes,
        inherited_generation.logical_gene_weight_bytes,
        inherited_generation.offspring_bytes,
        inherited_generation.metric_row_bytes,
        inherited_generation.rank_key_bytes,
        inherited_generation.selection_bytes,
        inherited_generation.dedup_hash_bytes,
        inherited_generation.cub_scratch_bytes,
        inherited_generation.retained_evaluation_workspace_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, checked_add)
}

fn checked_packed_ternary_survivor_store_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<PackedTernarySurvivorStoreChargeV1, ResidentPostGaWorkspacePlanErrorV1> {
    let signal_cell_count =
        checked_signal_cell_count_v1(resolved.survivor_capacity, resolved.row_count)?;
    let signal_bit_count = checked_mul(signal_cell_count, TERNARY_BITS_PER_CELL_V1)?;
    let packed_signal_bytes =
        checked_align_device_bytes_v1(checked_ceil_div_v1(signal_bit_count, 8)?)?;
    let compact_to_parent_map_bytes = checked_align_device_bytes_v1(checked_mul(
        resolved.survivor_capacity,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let candidate_order_bytes = checked_align_device_bytes_v1(checked_mul(
        resolved.survivor_capacity,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let total_device_bytes = checked_add(packed_signal_bytes, compact_to_parent_map_bytes)
        .and_then(|total| checked_add(total, candidate_order_bytes))?;

    Ok(PackedTernarySurvivorStoreChargeV1 {
        negative_encoding: TERNARY_NEGATIVE_BITS_V1,
        zero_encoding: TERNARY_ZERO_BITS_V1,
        positive_encoding: TERNARY_POSITIVE_BITS_V1,
        invalid_encoding: TERNARY_INVALID_BITS_V1,
        signal_cell_count,
        signal_bit_count,
        packed_signal_bytes,
        compact_to_parent_map_bytes,
        candidate_order_bytes,
        invalid_code_faults_and_invalidates_receipt: true,
        compact_candidate_to_parent_map_sha256: resolved.compact_candidate_to_parent_map_sha256,
        candidate_order_sha256: resolved.candidate_order_sha256,
        row_order_sha256: resolved.row_order_sha256,
        signal_semantics_sha256: resolved.signal_semantics_sha256,
        total_device_bytes,
    })
}

fn checked_signal_cell_count_v1(
    survivor_capacity: u64,
    row_count: u64,
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    if survivor_capacity == 0 || row_count == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "packed ternary survivor store",
        ));
    }
    checked_mul(survivor_capacity, row_count)
}

fn checked_quality_accumulator_charge_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<QualityAccumulatorChargeV1, ResidentPostGaWorkspacePlanErrorV1> {
    let quality_accumulator_bytes = checked_align_device_bytes_v1(checked_mul(
        resolved.survivor_capacity,
        resolved.quality_accumulator_bytes_per_candidate,
    )?)?;
    let active_day_cells = checked_mul(
        resolved.active_candidate_count,
        resolved.sealed_max_distinct_traded_day_capacity,
    )?;
    let dense_day_pnl_bytes =
        checked_align_device_bytes_v1(checked_mul(active_day_cells, POPULATION_F64_BYTES_V1)?)?;
    let traded_day_flags_bytes = checked_align_device_bytes_v1(active_day_cells)?;
    let dense_branch_bytes = dense_day_pnl_bytes.checked_add(traded_day_flags_bytes);
    let dense_branch_bytes = dense_branch_bytes.ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("dense quality branch"),
    )?;
    let active_trade_cells = resolved.sealed_max_trade_count_per_active_chunk;
    let bounded_trade_pnl_bytes =
        checked_align_device_bytes_v1(checked_mul(active_trade_cells, POPULATION_F64_BYTES_V1)?)?;
    let branch_tag_bytes = checked_align_device_bytes_v1(resolved.active_candidate_count)?;
    let sealed_day_count_bytes = checked_align_device_bytes_v1(checked_mul(
        resolved.active_candidate_count,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let bootstrap_drawdown_bytes = checked_align_device_bytes_v1(checked_mul(
        checked_mul(
            resolved.active_candidate_count,
            QUALITY_BOOTSTRAP_ITERATIONS_V1,
        )?,
        POPULATION_F64_BYTES_V1,
    )?)?;
    let branch_arena_bytes = dense_branch_bytes.max(bounded_trade_pnl_bytes);
    let phase_device_bytes = checked_add(branch_tag_bytes, sealed_day_count_bytes)
        .and_then(|total| checked_add(total, bootstrap_drawdown_bytes))
        .and_then(|total| checked_add(total, branch_arena_bytes))?;

    Ok(QualityAccumulatorChargeV1 {
        semantics: QualityAccumulatorSemanticsV1::DeterministicDailyBlockBootstrapV1,
        quality_accumulator_layout_sha256: resolved.quality_accumulator_layout_sha256,
        quality_accumulator_bytes,
        dense_day_pnl_bytes,
        traded_day_flags_bytes,
        bounded_trade_pnl_bytes,
        branch_tag_bytes,
        sealed_day_count_bytes,
        bootstrap_drawdown_bytes,
        active_candidate_count: resolved.active_candidate_count,
        active_chunk_count: resolved.active_chunk_count,
        exact_logical_candidate_coverage: resolved.exact_logical_candidate_coverage,
        chronological_traded_day_compaction: true,
        sealed_max_distinct_traded_day_capacity: resolved.sealed_max_distinct_traded_day_capacity,
        sealed_max_trade_count_per_active_chunk: resolved.sealed_max_trade_count_per_active_chunk,
        candidate_order_sha256: resolved.candidate_order_sha256,
        quality_rng_counter_mapping_sha256: resolved.quality_rng_counter_mapping_sha256,
        persistent_device_bytes: quality_accumulator_bytes,
        phase_device_bytes,
    })
}

fn checked_parameter_mc_workspace_charge_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
    inherited: &SealedMetricsOnlyWorkspaceReceiptV1,
) -> Result<ParameterMonteCarloWorkspaceChargeV1, ResidentPostGaWorkspacePlanErrorV1> {
    let total_resolved_scenarios = checked_add(
        resolved.exact_parameter_mc_scenario_count,
        resolved.exact_sensitivity_scenario_count,
    )
    .and_then(|total| checked_add(total, resolved.exact_cost_band_scenario_count))?;
    if total_resolved_scenarios == 0 || resolved.parameter_mc_active_scenario_capacity == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "parameter Monte Carlo scenarios",
        ));
    }
    let expected_chunk_count = checked_ceil_div_v1(
        total_resolved_scenarios,
        resolved.parameter_mc_active_scenario_capacity,
    )?;
    if !resolved.exact_scenario_chunk_coverage
        || expected_chunk_count != resolved.exact_parameter_mc_chunk_count
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidScenarioCoverage);
    }

    let scenario_count = resolved.parameter_mc_active_scenario_capacity;
    let month_capacity = resolved.month_capacity;
    let metric_rows_bytes = scenario_count.checked_mul(POPULATION_METRIC_ROW_BYTES_V1);
    let metric_rows_bytes = metric_rows_bytes.ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("parameter MC metric rows"),
    )?;
    let monthly_element_count = scenario_count.checked_mul(month_capacity).ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("parameter MC month cells"),
    )?;
    let monthly_pnl_bytes = checked_mul(monthly_element_count, POPULATION_F64_BYTES_V1)?;
    let month_start_equity_bytes = checked_mul(monthly_element_count, POPULATION_F64_BYTES_V1)?;
    let scenario_descriptor_bytes =
        checked_mul(scenario_count, POPULATION_SCENARIO_DESCRIPTOR_BYTES_V1)?;
    let required_metrics_workspace_bytes = checked_mul(
        scenario_count,
        checked_metrics_only_bytes_per_scenario_v1(month_capacity)?,
    )?;

    let sealed_metrics_workspace_capacity = inherited.scenario_capacity;
    let can_reuse = sealed_metrics_workspace_capacity >= scenario_count
        && inherited.month_capacity == month_capacity
        && inherited.total_device_bytes >= required_metrics_workspace_bytes;
    let (disposition, dedicated_metrics_workspace_bytes) = if can_reuse {
        (
            MetricsWorkspaceDispositionV1::ReuseInheritedExactCapacity,
            0,
        )
    } else {
        let dedicated_metrics_workspace_bytes = required_metrics_workspace_bytes;
        (
            MetricsWorkspaceDispositionV1::DedicatedContiguousCharge,
            dedicated_metrics_workspace_bytes,
        )
    };

    Ok(ParameterMonteCarloWorkspaceChargeV1 {
        disposition,
        exact_parameter_mc_scenario_count: resolved.exact_parameter_mc_scenario_count,
        exact_sensitivity_scenario_count: resolved.exact_sensitivity_scenario_count,
        exact_cost_band_scenario_count: resolved.exact_cost_band_scenario_count,
        active_scenario_capacity: scenario_count,
        scenario_chunk_count: resolved.exact_parameter_mc_chunk_count,
        exact_scenario_chunk_coverage: resolved.exact_scenario_chunk_coverage,
        metric_rows_bytes,
        monthly_pnl_bytes,
        month_start_equity_bytes,
        scenario_descriptor_bytes,
        required_metrics_workspace_bytes,
        dedicated_metrics_workspace_bytes,
        outcome_bytes: 0,
        accepted_trade_total_bytes: 0,
    })
}

fn checked_metrics_only_bytes_per_scenario_v1(
    month_capacity: u64,
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    if month_capacity == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "metrics-only month capacity",
        ));
    }
    let monthly_pair = 2_u64.checked_mul(month_capacity).ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("metrics-only monthly pair"),
    )?;
    checked_add(
        POPULATION_METRIC_ROW_BYTES_V1,
        POPULATION_SCENARIO_DESCRIPTOR_BYTES_V1,
    )
    .and_then(|total| {
        checked_mul(monthly_pair, POPULATION_F64_BYTES_V1)
            .and_then(|monthly| checked_add(total, monthly))
    })
}

fn checked_compact_selected_trade_ledger_ceiling_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<CompactSelectedTradeLedgerCeilingV1, ResidentPostGaWorkspacePlanErrorV1> {
    let selected_count = resolved.selected_portfolio_capacity;
    let exact_total_trade_count = resolved.sealed_total_trade_count_ceiling;
    if selected_count == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "selected portfolio capacity",
        ));
    }
    let selected_trade_count_bytes = checked_align_device_bytes_v1(checked_mul(
        selected_count,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let offset_count = selected_count.checked_add(1).ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("selected ledger offsets"),
    )?;
    let selected_trade_offset_bytes = checked_align_device_bytes_v1(checked_mul(
        offset_count,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let selected_trade_outcome_bytes =
        exact_total_trade_count.checked_mul(NEO_POPULATION_OUTCOME_BYTES_V1);
    let selected_trade_outcome_bytes =
        checked_align_device_bytes_v1(selected_trade_outcome_bytes.ok_or(
            ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("selected ledger outcomes"),
        )?)?;
    let sealed_compact_ledger_byte_ceiling =
        checked_add(selected_trade_count_bytes, selected_trade_offset_bytes)
            .and_then(|total| checked_add(total, selected_trade_outcome_bytes))?;

    Ok(CompactSelectedTradeLedgerCeilingV1 {
        selected_capacity: selected_count,
        total_trade_count_ceiling: exact_total_trade_count,
        selected_trade_count_bytes,
        selected_trade_offset_bytes,
        selected_trade_outcome_bytes,
        sealed_compact_ledger_byte_ceiling,
        passes: [
            SelectedLedgerPassV1::ExactTradeCount,
            SelectedLedgerPassV1::CubExclusiveScan,
            SelectedLedgerPassV1::CheckedBudget,
            SelectedLedgerPassV1::CompactReplayWrite,
        ],
        selected_gene_order_sha256: resolved.selected_gene_order_sha256,
        settings_cost_conversion_sha256: resolved.settings_cost_conversion_sha256,
        rng_counter_mapping_sha256: resolved.rng_counter_mapping_sha256,
        native_build_sha256: resolved.native_build_sha256,
    })
}

fn checked_compact_ledger_content_bytes_v1(
    selected_count: u64,
    exact_total_trade_count: u64,
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    let count_bytes = checked_align_device_bytes_v1(checked_mul(
        selected_count,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let offset_count = selected_count.checked_add(1).ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("runtime ledger offsets"),
    )?;
    let offset_bytes = checked_align_device_bytes_v1(checked_mul(
        offset_count,
        std::mem::size_of::<u64>() as u64,
    )?)?;
    let outcome_bytes = exact_total_trade_count.checked_mul(NEO_POPULATION_OUTCOME_BYTES_V1);
    let outcome_bytes = checked_align_device_bytes_v1(outcome_bytes.ok_or(
        ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow("runtime ledger outcomes"),
    )?)?;
    checked_add(count_bytes, offset_bytes).and_then(|total| checked_add(total, outcome_bytes))
}

fn checked_cub_scratch_charge_v1(
    receipt: &SealedCubScratchQueryReceiptV1,
) -> Result<CubScratchChargeV1, ResidentPostGaWorkspacePlanErrorV1> {
    require_nonzero_identity_v1(receipt.cub_query_receipt_sha256, "CUB query receipt")?;
    require_nonzero_identity_v1(receipt.cuda_toolkit_build_sha256, "CUDA toolkit build")?;
    require_nonzero_identity_v1(receipt.cccl_build_sha256, "CCCL build")?;
    require_nonzero_identity_v1(receipt.same_admitted_stream_sha256, "CUB admitted stream")?;
    if receipt.post_ga_required_cub_scratch_bytes == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidCubScratchReceipt);
    }

    let inherited_generation_cub_scratch_bytes = receipt.inherited_generation_cub_scratch_bytes;
    let post_ga_required_cub_scratch_bytes = receipt.post_ga_required_cub_scratch_bytes;
    let (disposition, dedicated_cub_scratch_bytes) =
        if inherited_generation_cub_scratch_bytes >= post_ga_required_cub_scratch_bytes {
            (CubScratchDispositionV1::ReuseInheritedGenerationScratch, 0)
        } else {
            let dedicated_cub_scratch_bytes = post_ga_required_cub_scratch_bytes;
            (
                CubScratchDispositionV1::DedicatedContiguousPostGaScratch,
                dedicated_cub_scratch_bytes,
            )
        };

    Ok(CubScratchChargeV1 {
        disposition,
        inherited_generation_cub_scratch_bytes,
        post_ga_required_cub_scratch_bytes,
        dedicated_cub_scratch_bytes,
        cub_query_receipt_sha256: receipt.cub_query_receipt_sha256,
        cuda_toolkit_build_sha256: receipt.cuda_toolkit_build_sha256,
        cccl_build_sha256: receipt.cccl_build_sha256,
        cuda_build_manifest_sha256: receipt.cuda_build_manifest_sha256,
        same_admitted_stream_sha256: receipt.same_admitted_stream_sha256,
    })
}

fn checked_phase_arena_charge_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
    quality: &QualityAccumulatorChargeV1,
    parameter_mc: &ParameterMonteCarloWorkspaceChargeV1,
    ledger: &CompactSelectedTradeLedgerCeilingV1,
    cub: &CubScratchChargeV1,
    event_lifetimes: &ResidentPostGaEventLifetimeProofV1,
) -> Result<ResidentPostGaPhaseArenaChargeV1, ResidentPostGaWorkspacePlanErrorV1> {
    validate_phase_event_lifetimes_v1(event_lifetimes, resolved)?;
    let quality_and_monte_carlo_phase_bytes = checked_add(
        quality.phase_device_bytes,
        parameter_mc.dedicated_metrics_workspace_bytes,
    )
    .and_then(|total| checked_add(total, cub.dedicated_cub_scratch_bytes))?;
    let portfolio_constraints_phase_bytes = checked_add(
        resolved.portfolio_workspace.device_bytes,
        cub.dedicated_cub_scratch_bytes,
    )?;
    let selected_ledger_and_validation_phase_bytes = checked_add(
        ledger.sealed_compact_ledger_byte_ceiling,
        resolved.validation_workspace.device_bytes,
    )
    .and_then(|total| checked_add(total, cub.dedicated_cub_scratch_bytes))?;
    let robustness_tail_phase_bytes = checked_add(
        resolved.robustness_workspace.device_bytes,
        cub.dedicated_cub_scratch_bytes,
    )?;
    let phase_arena_device_bytes = checked_max_phase_arena_bytes_v1([
        quality_and_monte_carlo_phase_bytes,
        portfolio_constraints_phase_bytes,
        selected_ledger_and_validation_phase_bytes,
        robustness_tail_phase_bytes,
    ])?;

    Ok(ResidentPostGaPhaseArenaChargeV1 {
        quality_and_monte_carlo_phase_bytes,
        portfolio_constraints_phase_bytes,
        selected_ledger_and_validation_phase_bytes,
        robustness_tail_phase_bytes,
        phase_arena_device_bytes,
    })
}

fn checked_max_phase_arena_bytes_v1(
    phase_bytes: [u64; 4],
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    phase_bytes
        .into_iter()
        .max()
        .filter(|bytes| *bytes != 0)
        .ok_or(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "post-GA phase arena",
        ))
}

fn validate_cross_receipt_identity_v1(
    generation: &SealedResidentGenerationAllocationChargeProjectionV1,
    metrics: &SealedMetricsOnlyWorkspaceReceiptV1,
    cub: &SealedCubScratchQueryReceiptV1,
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    for (identity, label) in [
        (resolved.device_uuid_sha256, "device UUID"),
        (
            resolved.primary_context_identity_sha256,
            "primary context identity",
        ),
        (resolved.run_stream_identity_sha256, "run stream identity"),
        (resolved.cuda_build_manifest_sha256, "CUDA build manifest"),
        (resolved.generation_semantics_sha256, "generation semantics"),
        (resolved.scoring_semantics_sha256, "scoring semantics"),
        (resolved.novelty_semantics_sha256, "novelty semantics"),
        (resolved.resolved_config_sha256, "resolved config"),
        (
            resolved.canonical_input_receipt_sha256,
            "canonical input receipt",
        ),
    ] {
        require_nonzero_identity_v1(identity, label)?;
    }
    let identity_matches = generation.selected_cuda_ordinal == resolved.selected_cuda_ordinal
        && generation.device_uuid_sha256 == resolved.device_uuid_sha256
        && generation.primary_context_identity_sha256 == resolved.primary_context_identity_sha256
        && generation.run_stream_identity_sha256 == resolved.run_stream_identity_sha256
        && generation.cuda_build_manifest_sha256 == resolved.cuda_build_manifest_sha256
        && generation.generation_semantics_sha256 == resolved.generation_semantics_sha256
        && metrics.primary_context_identity_sha256 == resolved.primary_context_identity_sha256
        && metrics.run_stream_identity_sha256 == resolved.run_stream_identity_sha256
        && metrics.cuda_build_manifest_sha256 == resolved.cuda_build_manifest_sha256
        && cub.cuda_build_manifest_sha256 == resolved.cuda_build_manifest_sha256
        && cub.same_admitted_stream_sha256 == resolved.run_stream_identity_sha256
        && cub.inherited_generation_cub_scratch_bytes == generation.projection.cub_scratch_bytes;
    if !identity_matches {
        return Err(ResidentPostGaWorkspacePlanErrorV1::IdentityMismatch(
            "post-GA sealed receipts do not name one admitted run",
        ));
    }
    Ok(())
}

fn validate_metrics_only_receipt_v1(
    receipt: &SealedMetricsOnlyWorkspaceReceiptV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    let per_scenario = checked_metrics_only_bytes_per_scenario_v1(receipt.month_capacity)?;
    let exact_total = checked_mul(receipt.scenario_capacity, per_scenario)?;
    let exact_metric_rows = checked_mul(receipt.scenario_capacity, POPULATION_METRIC_ROW_BYTES_V1)?;
    let exact_month_cells = checked_mul(receipt.scenario_capacity, receipt.month_capacity)?;
    let exact_monthly_pnl = checked_mul(exact_month_cells, POPULATION_F64_BYTES_V1)?;
    let exact_month_start_equity = checked_mul(exact_month_cells, POPULATION_F64_BYTES_V1)?;
    let exact_scenario_descriptors = checked_mul(
        receipt.scenario_capacity,
        POPULATION_SCENARIO_DESCRIPTOR_BYTES_V1,
    )?;
    let component_total = checked_add(receipt.metric_rows_bytes, receipt.monthly_pnl_bytes)
        .and_then(|total| checked_add(total, receipt.month_start_equity_bytes))
        .and_then(|total| checked_add(total, receipt.scenario_descriptor_bytes))?;
    if receipt.scenario_capacity == 0
        || receipt.outcome_bytes != 0
        || receipt.accepted_trade_total_bytes != 0
        || receipt.metric_rows_bytes != exact_metric_rows
        || receipt.monthly_pnl_bytes != exact_monthly_pnl
        || receipt.month_start_equity_bytes != exact_month_start_equity
        || receipt.scenario_descriptor_bytes != exact_scenario_descriptors
        || receipt.total_device_bytes != exact_total
        || component_total != exact_total
        || is_zero_identity_v1(receipt.receipt_identity_sha256)
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidMetricsWorkspaceReceipt);
    }
    Ok(())
}

fn validate_resolved_extents_v1(
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    if resolved.logical_candidate_count == 0
        || resolved.survivor_capacity == 0
        || resolved.row_count == 0
        || resolved.active_candidate_count == 0
        || resolved.active_chunk_count == 0
        || resolved.selected_portfolio_capacity == 0
        || resolved.quality_accumulator_bytes_per_candidate == 0
        || resolved.sealed_max_distinct_traded_day_capacity == 0
        || resolved.sealed_max_trade_count_per_active_chunk == 0
        || resolved.month_capacity == 0
        || resolved.bounded_final_compact_readback_bytes == 0
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "resolved post-GA extent",
        ));
    }
    let expected_candidate_chunks = checked_ceil_div_v1(
        resolved.logical_candidate_count,
        resolved.active_candidate_count,
    )?;
    if !resolved.exact_logical_candidate_coverage
        || expected_candidate_chunks != resolved.active_chunk_count
        || resolved.selected_portfolio_capacity > resolved.survivor_capacity
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidScenarioCoverage);
    }
    for (identity, label) in [
        (
            resolved.quality_accumulator_layout_sha256,
            "quality accumulator layout",
        ),
        (
            resolved.quality_rng_counter_mapping_sha256,
            "quality RNG counter mapping",
        ),
        (
            resolved.compact_candidate_to_parent_map_sha256,
            "compact candidate parent map",
        ),
        (resolved.candidate_order_sha256, "candidate order"),
        (resolved.row_order_sha256, "row order"),
        (resolved.signal_semantics_sha256, "signal semantics"),
        (resolved.selected_gene_order_sha256, "selected gene order"),
        (
            resolved.settings_cost_conversion_sha256,
            "settings cost conversion",
        ),
        (
            resolved.rng_counter_mapping_sha256,
            "ledger RNG counter mapping",
        ),
        (resolved.native_build_sha256, "native build"),
        (
            resolved.final_compact_result_semantics_sha256,
            "final compact result semantics",
        ),
    ] {
        require_nonzero_identity_v1(identity, label)?;
    }
    validate_algorithm_charge_v1(&resolved.portfolio_workspace)?;
    validate_algorithm_charge_v1(&resolved.validation_workspace)?;
    validate_algorithm_charge_v1(&resolved.robustness_workspace)?;
    Ok(())
}

fn validate_algorithm_charge_v1(
    charge: &SealedExactPostGaAlgorithmChargeV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    if charge.device_bytes == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "post-GA algorithm charge",
        ));
    }
    require_nonzero_identity_v1(charge.algorithm_semantics_sha256, "algorithm semantics")?;
    require_nonzero_identity_v1(charge.algorithm_build_sha256, "algorithm build")?;
    require_nonzero_identity_v1(charge.charge_receipt_sha256, "algorithm charge receipt")
}

fn validate_phase_event_lifetimes_v1(
    proof: &ResidentPostGaEventLifetimeProofV1,
    resolved: &SealedPostGaResolvedExtentsV1,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    if proof.primary_context_identity_sha256 != resolved.primary_context_identity_sha256
        || proof.run_stream_identity_sha256 != resolved.run_stream_identity_sha256
    {
        return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidPhaseEventProof);
    }
    let expected = [
        (
            ResidentPostGaPhaseKindV1::GenerationReady,
            ResidentPostGaPhaseKindV1::QualityAndMonteCarlo,
            proof.generation_ready_event_identity_sha256,
        ),
        (
            ResidentPostGaPhaseKindV1::QualityAndMonteCarlo,
            ResidentPostGaPhaseKindV1::PortfolioConstraints,
            proof.quality_complete_event_identity_sha256,
        ),
        (
            ResidentPostGaPhaseKindV1::PortfolioConstraints,
            ResidentPostGaPhaseKindV1::SelectedLedgerAndValidation,
            proof.portfolio_complete_event_identity_sha256,
        ),
        (
            ResidentPostGaPhaseKindV1::SelectedLedgerAndValidation,
            ResidentPostGaPhaseKindV1::RobustnessTail,
            proof.validation_complete_event_identity_sha256,
        ),
        (
            ResidentPostGaPhaseKindV1::RobustnessTail,
            ResidentPostGaPhaseKindV1::FinalCompactSeal,
            proof.robustness_complete_event_identity_sha256,
        ),
    ];
    let event_identities = [
        proof.generation_ready_event_identity_sha256,
        proof.quality_complete_event_identity_sha256,
        proof.portfolio_complete_event_identity_sha256,
        proof.validation_complete_event_identity_sha256,
        proof.robustness_complete_event_identity_sha256,
    ];
    for (index, identity) in event_identities.iter().enumerate() {
        if is_zero_identity_v1(*identity)
            || event_identities[..index]
                .iter()
                .any(|prior| prior == identity)
        {
            return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidPhaseEventProof);
        }
    }
    for (edge, (producer, consumer, expected_event)) in proof.ordered_edges.iter().zip(expected) {
        let producer_event_identity_sha256 = edge.producer_event_identity_sha256;
        let consumer_dependency_identity_sha256 = edge.consumer_dependency_identity_sha256;
        if edge.producer_phase != producer
            || edge.consumer_phase != consumer
            || producer_event_identity_sha256 != expected_event
            || !(producer_event_identity_sha256 == consumer_dependency_identity_sha256)
            || edge.same_primary_context_identity_sha256 != proof.primary_context_identity_sha256
            || edge.same_run_stream_identity_sha256 != proof.run_stream_identity_sha256
            || !edge.typed_non_overlap_proof
            || is_zero_identity_v1(expected_event)
        {
            return Err(ResidentPostGaWorkspacePlanErrorV1::InvalidPhaseEventProof);
        }
    }
    Ok(())
}

fn hash_workspace_plan_identity_v1(
    generation: &SealedResidentGenerationAllocationChargeProjectionV1,
    metrics: &SealedMetricsOnlyWorkspaceReceiptV1,
    resolved: &SealedPostGaResolvedExtentsV1,
    signals: &PackedTernarySurvivorStoreChargeV1,
    quality: &QualityAccumulatorChargeV1,
    parameter_mc: &ParameterMonteCarloWorkspaceChargeV1,
    ledger: &CompactSelectedTradeLedgerCeilingV1,
    cub: &CubScratchChargeV1,
    event_lifetimes: &ResidentPostGaEventLifetimeProofV1,
    phase_arena: &ResidentPostGaPhaseArenaChargeV1,
    always_resident_device_bytes: u64,
    total_device_bytes: u64,
) -> IdentitySha256V1 {
    let mut hasher = Sha256::new();
    hasher.update(RESIDENT_POST_GA_WORKSPACE_SEMANTICS_V1.as_bytes());
    for identity in [
        generation.device_uuid_sha256,
        generation.primary_context_identity_sha256,
        generation.run_stream_identity_sha256,
        generation.cuda_build_manifest_sha256,
        generation.generation_semantics_sha256,
        generation.generation_allocation_receipt_sha256,
        metrics.primary_context_identity_sha256,
        metrics.run_stream_identity_sha256,
        metrics.cuda_build_manifest_sha256,
        metrics.receipt_identity_sha256,
        resolved.device_uuid_sha256,
        resolved.primary_context_identity_sha256,
        resolved.run_stream_identity_sha256,
        resolved.cuda_build_manifest_sha256,
        resolved.generation_semantics_sha256,
        resolved.scoring_semantics_sha256,
        resolved.novelty_semantics_sha256,
        resolved.resolved_config_sha256,
        resolved.canonical_input_receipt_sha256,
        resolved.quality_accumulator_layout_sha256,
        resolved.quality_rng_counter_mapping_sha256,
        resolved.compact_candidate_to_parent_map_sha256,
        resolved.candidate_order_sha256,
        resolved.row_order_sha256,
        resolved.signal_semantics_sha256,
        resolved.selected_gene_order_sha256,
        resolved.settings_cost_conversion_sha256,
        resolved.rng_counter_mapping_sha256,
        resolved.native_build_sha256,
        resolved.portfolio_workspace.algorithm_semantics_sha256,
        resolved.portfolio_workspace.algorithm_build_sha256,
        resolved.portfolio_workspace.charge_receipt_sha256,
        resolved.validation_workspace.algorithm_semantics_sha256,
        resolved.validation_workspace.algorithm_build_sha256,
        resolved.validation_workspace.charge_receipt_sha256,
        resolved.robustness_workspace.algorithm_semantics_sha256,
        resolved.robustness_workspace.algorithm_build_sha256,
        resolved.robustness_workspace.charge_receipt_sha256,
        resolved.final_compact_result_semantics_sha256,
        signals.compact_candidate_to_parent_map_sha256,
        signals.candidate_order_sha256,
        signals.row_order_sha256,
        signals.signal_semantics_sha256,
        quality.quality_accumulator_layout_sha256,
        quality.candidate_order_sha256,
        quality.quality_rng_counter_mapping_sha256,
        ledger.selected_gene_order_sha256,
        ledger.settings_cost_conversion_sha256,
        ledger.rng_counter_mapping_sha256,
        ledger.native_build_sha256,
        cub.cub_query_receipt_sha256,
        cub.cuda_toolkit_build_sha256,
        cub.cccl_build_sha256,
        cub.cuda_build_manifest_sha256,
        cub.same_admitted_stream_sha256,
        event_lifetimes.primary_context_identity_sha256,
        event_lifetimes.run_stream_identity_sha256,
        event_lifetimes.generation_ready_event_identity_sha256,
        event_lifetimes.quality_complete_event_identity_sha256,
        event_lifetimes.portfolio_complete_event_identity_sha256,
        event_lifetimes.validation_complete_event_identity_sha256,
        event_lifetimes.robustness_complete_event_identity_sha256,
    ] {
        hasher.update(identity);
    }
    hasher.update(generation.selected_cuda_ordinal.to_le_bytes());
    hasher.update(resolved.selected_cuda_ordinal.to_le_bytes());
    for value in [
        generation.projection.logical_gene_scalar_bytes,
        generation.projection.logical_gene_index_bytes,
        generation.projection.logical_gene_weight_bytes,
        generation.projection.offspring_bytes,
        generation.projection.metric_row_bytes,
        generation.projection.rank_key_bytes,
        generation.projection.selection_bytes,
        generation.projection.dedup_hash_bytes,
        generation.projection.cub_scratch_bytes,
        generation.projection.retained_evaluation_workspace_bytes,
        generation.projection.total_device_bytes,
        metrics.scenario_capacity,
        metrics.month_capacity,
        metrics.metric_rows_bytes,
        metrics.monthly_pnl_bytes,
        metrics.month_start_equity_bytes,
        metrics.scenario_descriptor_bytes,
        metrics.total_device_bytes,
        metrics.outcome_bytes,
        metrics.accepted_trade_total_bytes,
        resolved.logical_candidate_count,
        resolved.survivor_capacity,
        resolved.row_count,
        resolved.active_candidate_count,
        resolved.active_chunk_count,
        resolved.selected_portfolio_capacity,
        resolved.sealed_total_trade_count_ceiling,
        resolved.quality_accumulator_bytes_per_candidate,
        resolved.sealed_max_distinct_traded_day_capacity,
        resolved.sealed_max_trade_count_per_active_chunk,
        resolved.month_capacity,
        resolved.parameter_mc_active_scenario_capacity,
        resolved.exact_parameter_mc_scenario_count,
        resolved.exact_sensitivity_scenario_count,
        resolved.exact_cost_band_scenario_count,
        resolved.exact_parameter_mc_chunk_count,
        resolved.portfolio_workspace.device_bytes,
        resolved.validation_workspace.device_bytes,
        resolved.robustness_workspace.device_bytes,
        signals.signal_cell_count,
        signals.signal_bit_count,
        signals.packed_signal_bytes,
        signals.compact_to_parent_map_bytes,
        signals.candidate_order_bytes,
        signals.total_device_bytes,
        quality.quality_accumulator_bytes,
        quality.dense_day_pnl_bytes,
        quality.traded_day_flags_bytes,
        quality.bounded_trade_pnl_bytes,
        quality.branch_tag_bytes,
        quality.sealed_day_count_bytes,
        quality.bootstrap_drawdown_bytes,
        quality.active_candidate_count,
        quality.active_chunk_count,
        quality.sealed_max_distinct_traded_day_capacity,
        quality.sealed_max_trade_count_per_active_chunk,
        quality.persistent_device_bytes,
        quality.phase_device_bytes,
        parameter_mc.exact_parameter_mc_scenario_count,
        parameter_mc.exact_sensitivity_scenario_count,
        parameter_mc.exact_cost_band_scenario_count,
        parameter_mc.active_scenario_capacity,
        parameter_mc.scenario_chunk_count,
        parameter_mc.metric_rows_bytes,
        parameter_mc.monthly_pnl_bytes,
        parameter_mc.month_start_equity_bytes,
        parameter_mc.scenario_descriptor_bytes,
        parameter_mc.required_metrics_workspace_bytes,
        parameter_mc.dedicated_metrics_workspace_bytes,
        parameter_mc.outcome_bytes,
        parameter_mc.accepted_trade_total_bytes,
        ledger.selected_capacity,
        ledger.total_trade_count_ceiling,
        ledger.selected_trade_count_bytes,
        ledger.selected_trade_offset_bytes,
        ledger.selected_trade_outcome_bytes,
        ledger.sealed_compact_ledger_byte_ceiling,
        cub.inherited_generation_cub_scratch_bytes,
        cub.post_ga_required_cub_scratch_bytes,
        cub.dedicated_cub_scratch_bytes,
        phase_arena.quality_and_monte_carlo_phase_bytes,
        phase_arena.portfolio_constraints_phase_bytes,
        phase_arena.selected_ledger_and_validation_phase_bytes,
        phase_arena.robustness_tail_phase_bytes,
        phase_arena.phase_arena_device_bytes,
        always_resident_device_bytes,
        resolved.bounded_final_compact_readback_bytes,
        total_device_bytes,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update([
        signals.negative_encoding,
        signals.zero_encoding,
        signals.positive_encoding,
        signals.invalid_encoding,
        u8::from(signals.invalid_code_faults_and_invalidates_receipt),
        u8::from(quality.exact_logical_candidate_coverage),
        u8::from(quality.chronological_traded_day_compaction),
        u8::from(parameter_mc.exact_scenario_chunk_coverage),
        parameter_mc.disposition as u8,
        cub.disposition as u8,
        quality.semantics as u8,
    ]);
    for pass in ledger.passes {
        hasher.update([pass as u8]);
    }
    for edge in &event_lifetimes.ordered_edges {
        hasher.update([edge.producer_phase as u8, edge.consumer_phase as u8]);
        hasher.update(edge.producer_event_identity_sha256);
        hasher.update(edge.consumer_dependency_identity_sha256);
        hasher.update(edge.same_primary_context_identity_sha256);
        hasher.update(edge.same_run_stream_identity_sha256);
        hasher.update([u8::from(edge.typed_non_overlap_proof)]);
    }
    hasher.finalize().into()
}

fn checked_add(lhs: u64, rhs: u64) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    lhs.checked_add(rhs)
        .ok_or(ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow(
            "u64 addition",
        ))
}

fn checked_mul(lhs: u64, rhs: u64) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    lhs.checked_mul(rhs)
        .ok_or(ResidentPostGaWorkspacePlanErrorV1::ArithmeticOverflow(
            "u64 multiplication",
        ))
}

fn checked_ceil_div_v1(
    numerator: u64,
    denominator: u64,
) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    if denominator == 0 {
        return Err(ResidentPostGaWorkspacePlanErrorV1::ZeroExtent(
            "ceil-div denominator",
        ));
    }
    let adjusted = checked_add(numerator, denominator - 1)?;
    Ok(adjusted / denominator)
}

fn checked_align_device_bytes_v1(bytes: u64) -> Result<u64, ResidentPostGaWorkspacePlanErrorV1> {
    if bytes == 0 {
        return Ok(0);
    }
    checked_ceil_div_v1(bytes, DEVICE_ALIGNMENT_BYTES_V1)
        .and_then(|blocks| checked_mul(blocks, DEVICE_ALIGNMENT_BYTES_V1))
}

fn require_nonzero_identity_v1(
    identity: IdentitySha256V1,
    label: &'static str,
) -> Result<(), ResidentPostGaWorkspacePlanErrorV1> {
    if is_zero_identity_v1(identity) {
        Err(ResidentPostGaWorkspacePlanErrorV1::ZeroIdentity(label))
    } else {
        Ok(())
    }
}

fn is_zero_identity_v1(identity: IdentitySha256V1) -> bool {
    identity == [0; 32]
}
