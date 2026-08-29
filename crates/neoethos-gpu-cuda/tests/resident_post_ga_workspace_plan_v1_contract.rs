use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read_required(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "resident post-GA workspace plan is missing {token:?}"
        );
    }
}

fn forbid_all(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "resident post-GA workspace plan contains forbidden authority {token:?}"
        );
    }
}

fn workspace_source() -> String {
    read_required("src/resident_post_ga_workspace_plan_v1.rs")
}

#[test]
fn preflight_consumes_sealed_byte_identity_authority_without_live_owner_or_raw_handles() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "RESIDENT_POST_GA_WORKSPACE_SEMANTICS_V1",
            "ResidentGenerationAllocationChargeProjectionV1",
            "SealedResidentGenerationAllocationChargeProjectionV1",
            "SealedPostGaResolvedExtentsV1",
            "SealedCubScratchQueryReceiptV1",
            "checked_plan_resident_post_ga_workspace_v1",
            "validate_a1_generation_receipt_against_workspace_plan_v1",
            "selected_cuda_ordinal",
            "device_uuid_sha256",
            "primary_context_identity_sha256",
            "run_stream_identity_sha256",
            "cuda_build_manifest_sha256",
            "generation_semantics_sha256",
            "scoring_semantics_sha256",
            "novelty_semantics_sha256",
            "resolved_config_sha256",
            "canonical_input_receipt_sha256",
        ],
    );
    forbid_all(
        &source,
        &[
            "ResidentGenerationPostGaInPlaceRunV1,",
            "ResidentGenerationPostGaInPlaceRunV1)",
            "*mut ",
            "*const ",
            "NonNull<",
            "device_ptr",
            "raw_handle",
            "stream_handle",
            "event_handle",
            "context_handle",
            "cudaMemGetInfo",
            "mem_get_info",
        ],
    );

    let projection = section(
        &source,
        "struct ResidentGenerationAllocationChargeProjectionV1 {",
        "\n}",
    );
    assert!(
        !projection.contains("pub "),
        "sealed byte projection fields must not be caller-mintable"
    );
}

#[test]
fn inherited_generation_allocation_is_componentized_and_charged_exactly_once() {
    let source = workspace_source();
    let projection = section(
        &source,
        "struct ResidentGenerationAllocationChargeProjectionV1 {",
        "\n}",
    );
    require_all(
        projection,
        &[
            "logical_gene_scalar_bytes: u64",
            "logical_gene_index_bytes: u64",
            "logical_gene_weight_bytes: u64",
            "offspring_bytes: u64",
            "metric_row_bytes: u64",
            "rank_key_bytes: u64",
            "selection_bytes: u64",
            "dedup_hash_bytes: u64",
            "cub_scratch_bytes: u64",
            "retained_evaluation_workspace_bytes: u64",
            "total_device_bytes: u64",
        ],
    );
    require_all(
        &source,
        &[
            "InheritedGenerationAllocationDispositionV1::ReplacesResidentGeneticEvolution",
            "inherited_generation_charge_count: 1",
            "checked_sum_generation_components_v1",
            "inherited_generation.total_device_bytes",
            "generation_component_sum == inherited_generation.total_device_bytes",
            "always_resident_device_bytes",
        ],
    );
    forbid_all(
        &source,
        &[
            "inherited_generation_charge_count: 2",
            "generation_store_device_bytes + inherited_generation.total_device_bytes",
            "saturating_add",
            "saturating_mul",
        ],
    );
}

#[test]
fn survivor_signal_store_is_lossless_packed_two_bit_ternary_with_order_authority() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "PackedTernarySurvivorStoreChargeV1",
            "TERNARY_NEGATIVE_BITS_V1: u8 = 0b00",
            "TERNARY_ZERO_BITS_V1: u8 = 0b01",
            "TERNARY_POSITIVE_BITS_V1: u8 = 0b10",
            "TERNARY_INVALID_BITS_V1: u8 = 0b11",
            "TERNARY_BITS_PER_CELL_V1: u64 = 2",
            "checked_signal_cell_count_v1",
            "checked_ceil_div_v1(signal_bit_count, 8)",
            "checked_align_device_bytes_v1",
            "invalid_code_faults_and_invalidates_receipt: true",
            "compact_candidate_to_parent_map_sha256",
            "candidate_order_sha256",
            "row_order_sha256",
            "signal_semantics_sha256",
        ],
    );
    forbid_all(
        &source,
        &[
            "Vec<Vec<i8>>",
            "Vec<i8>",
            "signal_cell_count / 4",
            "TERNARY_INVALID_BITS_V1: u8 = 0b00",
        ],
    );
}

#[test]
fn quality_accumulator_has_exact_versioned_day_count_branch_and_chunk_coverage() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "QUALITY_BOOTSTRAP_ITERATIONS_V1: u64 = 1_000",
            "QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1: u64 = 5",
            "QualityAccumulatorSemanticsV1",
            "QualityAccumulatorChargeV1",
            "quality_accumulator_layout_sha256",
            "QualityPnlBranchV1::DenseDailyPnl",
            "dense_day_pnl_bytes",
            "traded_day_flags_bytes",
            "chronological_traded_day_compaction: true",
            "QualityPnlBranchV1::BoundedTradePnlFallback",
            "bounded_trade_pnl_bytes",
            "sealed_distinct_traded_day_count",
            "sealed_max_trade_count_per_active_chunk",
            "active_candidate_count",
            "active_chunk_count",
            "exact_logical_candidate_coverage",
        ],
    );
    require_all(
        &source,
        &[
            "sealed_distinct_traded_day_count >= QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1",
            "sealed_distinct_traded_day_count < QUALITY_BLOCK_BOOTSTRAP_MIN_DAYS_V1",
            "dense_day_pnl_bytes.checked_add(traded_day_flags_bytes)",
        ],
    );
    forbid_all(
        &source,
        &[
            "infer_traded_day_from_nonzero_pnl",
            "day_pnl != 0.0",
            "quality metric row extension",
            "quality_accumulator_bytes: 0",
        ],
    );
}

#[test]
fn parameter_mc_reuses_only_an_exact_sealed_metrics_workspace_without_outcomes() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "POPULATION_METRIC_ROW_BYTES_V1: u64 = 104",
            "POPULATION_SCENARIO_DESCRIPTOR_BYTES_V1: u64 = 56",
            "POPULATION_F64_BYTES_V1: u64 = 8",
            "ParameterMonteCarloWorkspaceChargeV1",
            "scenario_count.checked_mul(POPULATION_METRIC_ROW_BYTES_V1)",
            "scenario_count.checked_mul(month_capacity)",
            "monthly_pnl_bytes",
            "month_start_equity_bytes",
            "scenario_descriptor_bytes",
            "2_u64.checked_mul(month_capacity)",
            "checked_metrics_only_bytes_per_scenario_v1",
            "sealed_metrics_workspace_capacity",
            "exact_parameter_mc_scenario_count",
            "exact_sensitivity_scenario_count",
            "exact_cost_band_scenario_count",
            "exact_scenario_chunk_coverage",
            "outcome_bytes: 0",
            "accepted_trade_total_bytes: 0",
        ],
    );
    require_all(
        &source,
        &[
            "MetricsWorkspaceDispositionV1::ReuseInheritedExactCapacity",
            "MetricsWorkspaceDispositionV1::DedicatedContiguousCharge",
            "dedicated_metrics_workspace_bytes = required_metrics_workspace_bytes",
        ],
    );
    forbid_all(
        &source,
        &[
            "MAX_STAGED_CLONES",
            "MAX_TRADES_PER_CANDIDATE",
            "required_metrics_workspace_bytes - sealed_metrics_workspace_capacity",
            "accepted_trade_total_bytes > 0",
            "accepted_trade_total_bytes: POPULATION_F64_BYTES_V1",
        ],
    );
}

#[test]
fn compact_selected_trade_ledger_has_a_sealed_ceiling_and_exact_two_pass_charge() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "NEO_POPULATION_OUTCOME_BYTES_V1: u64 = 72",
            "CompactSelectedTradeLedgerCeilingV1",
            "selected_trade_count_bytes",
            "selected_trade_offset_bytes",
            "selected_trade_outcome_bytes",
            "selected_count.checked_add(1)",
            "exact_total_trade_count.checked_mul(NEO_POPULATION_OUTCOME_BYTES_V1)",
            "checked_compact_ledger_content_bytes_v1",
            "sealed_compact_ledger_byte_ceiling",
            "exact_compact_ledger_bytes <= sealed_compact_ledger_byte_ceiling",
            "SelectedLedgerPassV1::ExactTradeCount",
            "SelectedLedgerPassV1::CubExclusiveScan",
            "SelectedLedgerPassV1::CheckedBudget",
            "SelectedLedgerPassV1::CompactReplayWrite",
            "first_pass_count_sha256",
            "written_segment_length_sha256",
            "written_total_trade_count == exact_total_trade_count",
            "selected_gene_order_sha256",
            "settings_cost_conversion_sha256",
            "rng_counter_mapping_sha256",
            "native_build_sha256",
        ],
    );
    forbid_all(
        &source,
        &[
            "MAX_TRADES_PER_CANDIDATE",
            "8192",
            "sentinel",
            "truncate",
            "min(exact_total_trade_count",
        ],
    );
}

#[test]
fn cub_scratch_is_reused_only_as_one_sufficient_contiguous_region_or_fully_charged() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "CubScratchDispositionV1::ReuseInheritedGenerationScratch",
            "CubScratchDispositionV1::DedicatedContiguousPostGaScratch",
            "post_ga_required_cub_scratch_bytes",
            "inherited_generation_cub_scratch_bytes >= post_ga_required_cub_scratch_bytes",
            "dedicated_cub_scratch_bytes = post_ga_required_cub_scratch_bytes",
            "cub_query_receipt_sha256",
            "cuda_toolkit_build_sha256",
            "cccl_build_sha256",
            "same_admitted_stream_sha256",
        ],
    );
    forbid_all(
        &source,
        &[
            "post_ga_required_cub_scratch_bytes - inherited_generation_cub_scratch_bytes",
            "inherited_generation_cub_scratch_bytes + dedicated_cub_scratch_bytes",
            "caller_cub_scratch_bytes",
        ],
    );
}

#[test]
fn phase_arena_is_checked_max_only_after_typed_event_non_overlap_proof() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "ResidentPostGaPhaseKindV1::QualityAndMonteCarlo",
            "ResidentPostGaPhaseKindV1::PortfolioConstraints",
            "ResidentPostGaPhaseKindV1::SelectedLedgerAndValidation",
            "ResidentPostGaPhaseKindV1::RobustnessTail",
            "quality_and_monte_carlo_phase_bytes",
            "portfolio_constraints_phase_bytes",
            "selected_ledger_and_validation_phase_bytes",
            "robustness_tail_phase_bytes",
            "checked_max_phase_arena_bytes_v1",
            "ResidentPostGaEventLifetimeProofV1",
            "generation_ready_event_identity_sha256",
            "quality_complete_event_identity_sha256",
            "portfolio_complete_event_identity_sha256",
            "validation_complete_event_identity_sha256",
            "robustness_complete_event_identity_sha256",
            "producer_event_identity_sha256 == consumer_dependency_identity_sha256",
            "same_primary_context_identity_sha256",
            "same_run_stream_identity_sha256",
            "typed_non_overlap_proof",
            "phase_arena_device_bytes",
            "checked_add(always_resident_device_bytes, phase_arena_device_bytes)",
            "bounded_final_compact_readback_bytes",
        ],
    );
    forbid_all(
        &source,
        &[
            "quality_and_monte_carlo_phase_bytes\n            + portfolio_constraints_phase_bytes",
            "cudaEventSynchronize",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "wait()",
            "read_metrics",
            "per_phase_context",
            "per_phase_stream",
        ],
    );
}

#[test]
fn plan_is_checked_non_mintable_research_only_and_unwired_until_full_plan_integration() {
    let source = workspace_source();

    require_all(
        &source,
        &[
            "DEVICE_ALIGNMENT_BYTES_V1: u64 = 256",
            "checked_add",
            "checked_mul",
            "checked_ceil_div_v1",
            "checked_align_device_bytes_v1",
            "ResidentPostGaWorkspaceAuthorityV1::ResearchOnly",
            "ResidentPostGaPromotionEligibilityV1::NotPromotionEligible",
            "ResidentPostGaIntegrationStateV1::Unwired",
            "ResidentPostGaWorkspacePlanV1",
            "total_device_bytes",
            "workspace_plan_identity_sha256",
        ],
    );
    forbid_all(
        &source,
        &[
            "derive(Default)",
            "impl Default for ResidentPostGaWorkspacePlanV1",
            "Serialize",
            "Deserialize",
            "from_raw",
            "from_hash",
            "from_bytes",
            "saturating_add",
            "saturating_mul",
            "wrapping_add",
            "wrapping_mul",
            "cudaMemGetInfo",
            "mem_get_info",
            "free_memory_bytes",
            "GpuPreferred",
            "AllowCpu",
            "cpu_forced",
            "fallback_mode",
            "best_effort_fallback",
            "FALLBACK_PORTFOLIO_MAX",
        ],
    );

    let plan = section(
        &source,
        "pub(crate) struct ResidentPostGaWorkspacePlanV1 {",
        "\n}",
    );
    assert!(
        !plan.contains("pub "),
        "workspace plan authority fields must remain private"
    );
}
