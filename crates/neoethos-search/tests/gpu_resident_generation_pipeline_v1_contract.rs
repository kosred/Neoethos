use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn pipeline_source() -> String {
    let path = manifest_dir().join("src/gpu_full_discovery/gpu_resident_generation_pipeline_v1.rs");
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
            "resident generation pipeline is missing {token:?}"
        );
    }
}

#[test]
fn one_run_owned_population_gene_metric_and_scratch_store_survives_all_generations() {
    let source = pipeline_source();
    let run = section(&source, "pub struct GpuResidentGenerationRunV1 {", "\n}");
    require_all(
        run,
        &[
            "population_store: GpuResidentPopulationStoreV1",
            "gene_store: GpuResidentGeneStoreV1",
            "metric_store: GpuResidentMetricStoreV1",
            "cub_scratch_arena: CubScratchArenaV1",
            "run_stream: GpuRunStreamV1",
            "run_identity_sha256: String",
        ],
    );
    assert!(
        !run.contains("pub "),
        "resident run fields must remain private and non-caller-mintable"
    );

    require_all(
        &source,
        &[
            "begin_gpu_resident_generation_run_v1(",
            "execute_resident_generation_v1(",
            "seal_resident_generation_outcome_v1(",
            "generation_store_allocation_count",
            "generation_store_allocation_count != 1",
        ],
    );
    for forbidden in [
        "impl Default for GpuResidentGenerationRunV1",
        "Deserialize",
        "pub fn new(",
        "global_generation_store",
        "static GENERATION",
        "OnceLock",
    ] {
        assert!(
            !source.contains(forbidden),
            "resident generation lifetime has forbidden escape {forbidden:?}"
        );
    }
}

#[test]
fn run_binds_exact_permit_device_build_input_gene_and_fitness_semantics() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "StrictGpuOnlyFullDiscoveryPermitV2",
            "selected_cuda_ordinal",
            "cuda_device_identity_sha256",
            "cuda_build_manifest_sha256",
            "canonical_search_input_receipt_sha256",
            "resident_input_content_sha256",
            "strategy_gene_schema_sha256",
            "fitness_ordering_semantics_sha256",
            "crossover_semantics_sha256",
            "mutation_semantics_sha256",
            "rng_mapping_identity_sha256",
            "validate_generation_plan_against_permit_v1(",
            "compute_generation_run_identity_v1(",
        ],
    );
    for forbidden in [
        "selected_cuda_ordinal: u32,\n    cuda_build_manifest_sha256: &str",
        "caller_gene_schema_sha256",
        "caller_fitness_semantics_sha256",
        "allow_cpu: bool",
    ] {
        assert!(
            !source.contains(forbidden),
            "caller can inject exact generation authority via {forbidden:?}"
        );
    }
}

#[test]
fn deterministic_counter_rng_maps_every_draw_without_host_rng_state() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CounterRngAlgorithmV1::Philox4x32_10",
            "search_seed",
            "run_identity_sha256",
            "generation_index",
            "candidate_identity",
            "genetic_operator_identity",
            "draw_index",
            "checked_counter_mapping_v1(",
            "rng_mapping_identity_sha256",
            "CounterMappingOverflow",
        ],
    );
    for forbidden in [
        "thread_rng",
        "SmallRng",
        "StdRng",
        "rand::random",
        "from_entropy",
        "system_time",
    ] {
        assert!(
            !source.contains(forbidden),
            "resident generation uses nondeterministic or host RNG {forbidden:?}"
        );
    }
}

#[test]
fn selection_dedup_and_rank_use_official_device_primitives_with_exact_ties() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRadixSortPairs",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceSelectFlagged",
            "OfficialGpuPrimitiveAuthorityV1::NvidiaCubDeviceRunLengthEncode",
            "launch_device_parent_selection_v1(",
            "launch_device_crossover_v1(",
            "launch_device_mutation_v1(",
            "launch_device_gene_hash_v1(",
            "canonical_f64_total_order_key_v1(",
            "stable_tie_break_gene_identity_v1(",
            "rank_semantics_identity_sha256",
        ],
    );
    for forbidden in [
        "CustomDeviceRadixSort",
        "CustomDeviceSelect",
        "CustomDeviceUnique",
        "partial_cmp",
        "sort_by(",
        "sort_unstable",
    ] {
        assert!(
            !source.contains(forbidden),
            "generation pipeline replaces an official primitive or weakens ordering via {forbidden:?}"
        );
    }
}

#[test]
fn generation_loop_has_no_host_collections_rayon_metric_readback_or_sync() {
    let source = pipeline_source();
    let execute = section(&source, "fn execute_resident_generation_v1(", "\n}\n\n");
    for forbidden in [
        "HashSet<",
        "Vec<HashSet",
        "rayon",
        ".par_iter",
        ".par_sort",
        "read_metrics",
        "copy_to_host",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        ".synchronize(",
        "Vec<StrategyMetrics>",
        "Vec<StrategyGene>",
    ] {
        assert!(
            !execute.contains(forbidden),
            "generation execution crosses to host via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "per_generation_metric_rows_readback_count == 0",
            "per_generation_explicit_synchronization_count == 0",
            "per_generation_host_decision_count == 0",
            "final_compact_readback_count",
            "GenerationTransferAccountingMismatch",
        ],
    );
}

#[test]
fn card_present_generation_has_no_cpu_allow_cpu_or_fallback_path() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "CardPresentCpuGenerationForbidden",
            "CardPresentAllowCpuGenerationForbidden",
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
            "card-present generation retains CPU/fallback escape {forbidden:?}"
        );
    }
}

#[test]
fn output_remains_opaque_and_gpu_resident_for_the_post_ga_pipeline() {
    let source = pipeline_source();
    require_all(
        &source,
        &[
            "pub struct SealedResidentGenerationOutcomeV1",
            "resident_generation_outcome_identity_sha256",
            "consume_in_gpu_post_ga_pipeline_v1(",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "pub genes: Vec<",
        "pub metrics: Vec<",
        "pub signals:",
        "pub trades:",
        "Deserialize",
        "impl Default for SealedResidentGenerationOutcomeV1",
        "pub fn from_hash",
    ] {
        assert!(
            !source.contains(forbidden),
            "resident generation output leaks or reconstructs authority via {forbidden:?}"
        );
    }
}

#[test]
fn generation_evaluator_is_metrics_only_with_null_outcomes_and_online_exact_accumulators() {
    let source = pipeline_source();
    let execute = section(&source, "fn execute_resident_generation_v1(", "\n}\n\n");
    require_all(
        execute,
        &[
            "GpuPopulationEvaluationModeV1::MetricsOnly",
            "outcomes_device_ptr: DevicePointerV1::Null",
            "accepted_trade_total_device_ptr: DevicePointerV1::Null",
            "outcome_seed_kernel_launch_count == 0",
            "outcome_write_count == 0",
            "accepted_trade_total_atomic_count == 0",
            "accepted_trade_total_d2h_count == 0",
            "OnlineStrategyMetricAccumulatorV1",
            "OnlineDailyMetricAccumulatorV1",
            "OnlineMonthlyMetricAccumulatorV1",
            "monthly_pnls_device_array",
            "month_start_equities_device_array",
            "MonthlyMeanVarianceOrderV1::ExactTwoPass",
        ],
    );
    for forbidden in [
        "MAX_TRADES_PER_CANDIDATE",
        "allocate_outcome_array",
        "seed_outcome_array",
        "write_outcome_array",
        "FixedTradeSlots",
        "allocate_accepted_trade_total",
        "atomic_add_accepted_trade_total",
        "read_accepted_trade_total",
        "copy_accepted_trade_total_to_host",
    ] {
        assert!(
            !execute.contains(forbidden),
            "metrics-only generation allocates or writes trade outcomes via {forbidden:?}"
        );
    }
    require_all(
        &source,
        &[
            "fn enqueue_default_production_metrics_only_v1(",
            "enqueue_population_metrics_only_on_run_stream_v1(",
            "GpuResidentMetricRowsHandleV1",
            "GpuResidentGenerationReadyEventV1",
            "rank_and_select_after_event_dependency_v1(",
            "per_generation_metric_rows_readback_count == 0",
            "per_generation_host_wait_count == 0",
            "trade_count_metric_index == 8",
            "reduce_final_generation_diagnostics_on_device_v1(",
            "FinalBoundedGenerationDiagnosticsReceiptV1",
            "GpuPopulationEvaluationModeV1::SelectedSurvivorDiagnostics",
        ],
    );
    let production_adapter = section(
        &source,
        "fn enqueue_default_production_metrics_only_v1(",
        "\n}\n\n",
    );
    for forbidden in [
        ".wait(",
        ".read_metrics(",
        "read_diagnostics",
        "accepted_trade_total",
        "seed_outcomes",
        "allocate_outcomes",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
    ] {
        assert!(
            !production_adapter.contains(forbidden),
            "default production adapter crosses into diagnostic trade state via {forbidden:?}"
        );
    }

    require_all(
        &source,
        &[
            "#[cfg(all(test, feature = \"gpu-b-native\"))]",
            "fn read_metrics_for_device_parity_test_only_v1(",
            "DeviceParityTestOnly",
            "validate_online_metrics_against_canonical_strategy_metrics_v1(",
            "ExactStrategyMetricsParityMismatch",
            "strategy_metrics_semantics_sha256",
        ],
    );
    let parity_helper = section(
        &source,
        "fn read_metrics_for_device_parity_test_only_v1(",
        "\n}\n\n",
    );
    require_all(parity_helper, &[".wait(", ".read_metrics("]);
    assert_eq!(
        source.matches(".wait(").count(),
        parity_helper.matches(".wait(").count(),
        "host wait escapes the device-gated parity helper"
    );
    assert_eq!(
        source.matches(".read_metrics(").count(),
        parity_helper.matches(".read_metrics(").count(),
        "metric-row D2H escapes the device-gated parity helper"
    );
    assert!(
        !source.contains("read_diagnostics("),
        "generation production module must never read diagnostic outcome rows"
    );
}

#[test]
fn memory_receipt_separates_metrics_and_survivor_ledger_and_proves_16384_capacity() {
    let source = pipeline_source();
    let receipt = section(
        &source,
        "pub struct GpuGenerationMemoryPlanReceiptV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "selected_cuda_ordinal: u32",
            "cuda_build_manifest_sha256: String",
            "actual_plan_identity_sha256: String",
            "metrics_only_fixed_bytes: u64",
            "metrics_only_bytes_per_candidate: u64",
            "monthly_pnls_bytes_per_candidate: u64",
            "month_start_equities_bytes_per_candidate: u64",
            "survivor_trade_ledger_bytes: u64",
            "planned_population_capacity: usize",
            "memory_receipt_identity_sha256: String",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "generation memory receipt fields must remain private"
    );
    require_all(
        &source,
        &[
            "const MIN_REQUIRED_RESIDENT_POPULATION_CAPACITY_V1: usize = 16_384;",
            "validate_metrics_only_capacity_from_actual_plan_v1(",
            "planned_population_capacity >= MIN_REQUIRED_RESIDENT_POPULATION_CAPACITY_V1",
            "MetricsOnlyPopulationCapacityBelow16384",
            "monthly_pnls_bytes_per_candidate",
            "month_start_equities_bytes_per_candidate",
            "checked_metrics_only_bytes_without_trade_outcomes_v1(",
            "checked_add",
            "checked_mul",
        ],
    );
    let capacity = section(
        &source,
        "fn validate_metrics_only_capacity_from_actual_plan_v1(",
        "\n}\n\n",
    );
    require_all(
        capacity,
        &[
            "metrics_only_fixed_bytes",
            "metrics_only_bytes_per_candidate",
            "monthly_pnls_bytes_per_candidate",
            "month_start_equities_bytes_per_candidate",
            "checked_sub",
        ],
    );
    for forbidden in [
        "MAX_TRADES_PER_CANDIDATE",
        "survivor_trade_ledger_bytes",
        "593_768",
        "593768",
        "593 768",
        "594 * 1024",
    ] {
        assert!(
            !capacity.contains(forbidden),
            "metrics-only capacity still includes legacy per-candidate outcome workspace via {forbidden:?}"
        );
    }
}
