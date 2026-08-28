use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn pipeline_source() -> String {
    let path = manifest_dir().join("src/gpu_full_discovery/gpu_post_ga_pipeline_v1.rs");
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
            "resident post-GA pipeline is missing {token:?}"
        );
    }
}

fn require_in_order(source: &str, required: &[&str]) {
    let mut cursor = 0;
    for token in required {
        let relative = source[cursor..]
            .find(token)
            .unwrap_or_else(|| panic!("post-GA pipeline is missing ordered token {token:?}"));
        cursor += relative + token.len();
    }
}

#[test]
fn exact_four_stage_plan_is_strict_gpu_and_identity_bound() {
    let source = pipeline_source();
    let stages = section(
        &source,
        "const GPU_POST_GA_STAGES_V1: [PipelineStage; 4] = [",
        "];",
    );
    for stage in [
        "PipelineStage::SignalAndMinTradeFilter",
        "PipelineStage::QualityScreen",
        "PipelineStage::MonteCarlo",
        "PipelineStage::SurvivorRanking",
    ] {
        assert_eq!(
            stages.matches(stage).count(),
            1,
            "post-GA plan must contain {stage} exactly once"
        );
    }
    assert_eq!(
        stages.matches("PipelineStage::").count(),
        4,
        "post-GA plan must contain exactly four stages"
    );

    require_all(
        &source,
        &[
            "StrictGpuOnlyFullDiscoveryPermitV2",
            "StageGpuCapability::StrictGpu",
            "selected_cuda_ordinal",
            "cuda_device_identity_sha256",
            "cuda_build_manifest_sha256",
            "canonical_search_input_receipt_sha256",
            "resident_input_content_sha256",
            "post_ga_plan_semantics_sha256",
            "compute_post_ga_stage_plan_identity_v1(",
        ],
    );
}

#[test]
fn post_ga_run_consumes_generation_authority_and_keeps_candidates_resident() {
    let source = pipeline_source();
    let run = section(&source, "pub struct GpuResidentPostGaRunV1 {", "\n}");
    require_all(
        run,
        &[
            "generation_outcome: SealedResidentGenerationOutcomeV1",
            "candidate_store: GpuResidentPostGaCandidateStoreV1",
            "metric_store: GpuResidentPostGaMetricStoreV1",
            "trade_state_store: GpuResidentTradeStateStoreV1",
            "cub_scratch_arena: CubScratchArenaV1",
            "run_identity_sha256: String",
        ],
    );
    assert!(
        !run.contains("pub "),
        "post-GA run fields must remain private and non-caller-mintable"
    );
    require_all(
        &source,
        &[
            "begin_gpu_resident_post_ga_run_v1(",
            "validate_generation_outcome_against_permit_v1(",
            "seal_resident_post_ga_outcome_v1(",
        ],
    );
}

#[test]
fn signal_filter_uses_fused_device_counts_and_official_segmented_selection() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "launch_fused_signal_and_trade_count_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "signal_semantics_sha256",
            "minimum_trade_filter_semantics_sha256",
            "full_signal_readback_count == 0",
        ],
    );
    for forbidden in [
        "CustomDeviceSegmentedReduce",
        "CustomDeviceSelect",
        "Vec<Vec<i8>>",
        "signals_for_gene_full",
    ] {
        assert!(
            !source.contains(forbidden),
            "signal filter replaces an official primitive or materializes host signals via {forbidden:?}"
        );
    }
}

#[test]
fn quality_and_monte_carlo_keep_trade_state_scenarios_and_reductions_on_device() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "launch_path_dependent_quality_semantics_v1(",
            "quality_semantics_sha256",
            "CounterRngAlgorithmV1::Philox4x32_10",
            "monte_carlo_counter_mapping_sha256",
            "launch_monte_carlo_scenarios_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScan",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSegmentedReduce",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode",
            "monte_carlo_metric_rows_readback_count == 0",
            "trade_rows_readback_count == 0",
        ],
    );
    for forbidden in [
        "simulate_trades_core",
        "classify_base_quality",
        "thread_rng",
        "StdRng",
        "CustomDeviceScan",
        "CustomDeviceReduce",
    ] {
        assert!(
            !source.contains(forbidden),
            "quality/Monte-Carlo retains host work or replaces official primitive via {forbidden:?}"
        );
    }
}

#[test]
fn survivor_ranking_uses_device_total_order_stable_ties_and_bounded_selection() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRadixSortPairs",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "canonical_f64_total_order_key_v1(",
            "stable_tie_break_candidate_identity_v1(",
            "resolved_portfolio_target",
            "serialized_byte_ceiling",
            "survivor_rank_semantics_sha256",
        ],
    );
    for forbidden in [
        "CustomDeviceRadixSort",
        "CustomDeviceSelect",
        "partial_cmp",
        "sort_by(",
        "sort_unstable",
    ] {
        assert!(
            !source.contains(forbidden),
            "survivor ranking replaces an official primitive or weakens ties via {forbidden:?}"
        );
    }
}

#[test]
fn strict_post_ga_execution_has_no_host_matrix_rayon_intermediate_readback_or_sync() {
    let source = pipeline_source();
    let execute = section(
        &source,
        "fn execute_gpu_resident_post_ga_pipeline_v1(",
        "\n}\n\n",
    );
    for forbidden in [
        "FeatureFrame",
        "DiscoveryResult",
        "Vec<StrategyMetrics>",
        "Vec<Trade>",
        "Vec<Vec<i8>>",
        "rayon",
        ".par_iter",
        "read_metrics",
        "copy_to_host",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        ".synchronize(",
    ] {
        assert!(
            !execute.contains(forbidden),
            "post-GA execution crosses to host via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "intermediate_metric_rows_readback_count == 0",
            "intermediate_explicit_synchronization_count == 0",
            "intermediate_host_decision_count == 0",
            "PostGaTransferAccountingMismatch",
        ],
    );
}

#[test]
fn only_bounded_compact_summaries_and_persisted_artifact_ids_can_cross() {
    let source = pipeline_source();
    let receipt = section(
        &source,
        "pub struct SealedBoundedPostGaGpuReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "selected_candidate_summaries: Vec<BoundedCandidateSummaryV1>",
            "resolved_portfolio_target: usize",
            "serialized_byte_ceiling: usize",
            "detailed_artifact_manifest_identity_sha256: String",
            "post_ga_receipt_identity_sha256: String",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "bounded post-GA receipt fields must remain private"
    );
    require_all(
        &source,
        &[
            "selected_candidate_summaries.len() <= resolved_portfolio_target",
            "serialized_len <= serialized_byte_ceiling",
            "compact_result_readback_count == 1",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "signals:",
        "trades:",
        "folds:",
        "metric_matrix",
        "FeatureFrame",
        "DiscoveryResult",
        "Deserialize",
        "pub fn from_hash",
    ] {
        assert!(
            !receipt.contains(forbidden),
            "bounded post-GA receipt leaks heavy/reconstructible field {forbidden:?}"
        );
    }
}

#[test]
fn card_present_post_ga_has_no_cpu_allow_cpu_or_fallback_path() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CardPresentCpuPostGaForbidden",
            "CardPresentAllowCpuPostGaForbidden",
            "ExactCudaOrdinalRequired",
        ],
    );
    for forbidden in [
        "cpu_forced",
        "cpu-forced",
        "GPU_PREFERRED",
        "FallbackPolicy::AllowCpu",
        "RecomputeOnCpu",
        "rayon::",
        "unwrap_or_else(cpu",
    ] {
        assert!(
            !source.contains(forbidden),
            "card-present post-GA retains CPU/fallback escape {forbidden:?}"
        );
    }
}

#[test]
fn selected_survivor_trade_ledger_is_exact_two_pass_compact_and_budget_checked() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "launch_selected_survivor_trade_count_pass_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScanExclusiveSum",
            "checked_selected_trade_total_v1(",
            "validate_selected_trade_ledger_budget_v1(",
            "launch_selected_survivor_compact_trade_write_pass_v1(",
            "CompactContiguousTradeOutcomeStoreV1",
            "selected_survivor_count",
            "exact_trade_count_per_survivor",
            "exact_trade_offsets",
            "exact_total_trade_count",
            "TradeCountOverflow",
            "TradeLedgerBudgetExceeded",
            "SelectedLedgerPassBindingMismatch",
            "SelectedLedgerSegmentLengthMismatch",
            "SelectedLedgerWrittenTotalMismatch",
            "destroy_unsealed_selected_ledger_v1(",
        ],
    );
    let binding = section(
        &source,
        "pub struct SealedSelectedSurvivorReplayBindingV1 {",
        "\n}",
    );
    require_all(
        binding,
        &[
            "sealed_parent_view_identity_sha256: String",
            "selected_gene_identities_sha256: String",
            "selected_gene_order_sha256: String",
            "settings_identity_sha256: String",
            "screening_cost_authority_sha256: String",
            "conversion_authority_sha256: String",
            "rng_algorithm_identity_sha256: String",
            "rng_counter_mapping_identity_sha256: String",
            "search_seed: u64",
            "kernel_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
        ],
    );
    assert!(
        !binding.contains("pub "),
        "selected-survivor replay binding fields must remain private"
    );
    let ledger = section(
        &source,
        "fn build_selected_survivor_trade_ledger_v1(",
        "\n}\n\n",
    );
    require_all(
        ledger,
        &[
            "count_pass_binding: &SealedSelectedSurvivorReplayBindingV1",
            "write_pass_binding: &SealedSelectedSurvivorReplayBindingV1",
            "validate_selected_ledger_pass_bindings_v1(",
            "validate_compact_segment_lengths_against_count_pass_v1(",
            "validate_total_written_against_scanned_total_v1(",
        ],
    );
    require_in_order(
        ledger,
        &[
            "seal_selected_survivor_replay_binding_v1(",
            "launch_selected_survivor_trade_count_pass_v1(",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceScanExclusiveSum",
            "checked_selected_trade_total_v1(",
            "validate_selected_trade_ledger_budget_v1(",
            "launch_selected_survivor_compact_trade_write_pass_v1(",
            "validate_compact_segment_lengths_against_count_pass_v1(",
            "validate_total_written_against_scanned_total_v1(",
            "seal_compact_selected_survivor_trade_ledger_v1(",
        ],
    );
    for forbidden in [
        "MAX_TRADES_PER_CANDIDATE",
        "8_192",
        "8192",
        "FixedTradeSlots",
        "sentinel",
        "truncate",
        "saturating_mul",
    ] {
        assert!(
            !ledger.contains(forbidden),
            "selected survivor ledger retains fixed/truncating allocation via {forbidden:?}"
        );
    }
}

#[test]
fn post_ga_memory_receipt_accounts_metrics_only_and_selected_ledger_separately() {
    let source = pipeline_source();
    let receipt = section(&source, "pub struct GpuPostGaMemoryPlanReceiptV1 {", "\n}");
    require_all(
        receipt,
        &[
            "metrics_only_population_bytes: u64",
            "selected_survivor_trade_ledger_bytes: u64",
            "selected_survivor_count: usize",
            "exact_total_trade_count: u64",
            "actual_plan_identity_sha256: String",
            "memory_receipt_identity_sha256: String",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "post-GA memory receipt fields must remain private"
    );
    require_all(
        &source,
        &[
            "validate_post_ga_memory_receipt_against_generation_plan_v1(",
            "selected_survivor_trade_ledger_bytes == checked_trade_ledger_bytes",
            "PostGaMemoryAccountingMismatch",
        ],
    );
}
