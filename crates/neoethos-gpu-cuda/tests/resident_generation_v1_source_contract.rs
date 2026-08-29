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
            "resident generation V1 source is missing {token:?}"
        );
    }
}

#[test]
fn rust_owner_is_move_only_run_bound_and_fail_closed_while_work_is_in_flight() {
    let source = read_required("src/resident_generation_v1.rs");
    let owner = section(&source, "pub struct ResidentGenerationDeviceRunV1 {", "\n}");
    require_all(
        owner,
        &[
            "native: NonNull<NativeResidentGenerationRunV1>",
            "population_session_import: Option<ResidentGenerationPopulationSessionImportV1>",
            "state: ResidentGenerationRunStateV1",
            "selected_cuda_ordinal: u32",
            "primary_context_identity_sha256: [u8; 32]",
            "run_stream_identity_sha256: [u8; 32]",
            "cuda_build_manifest_sha256: [u8; 32]",
            "generation_semantics_sha256: [u8; 32]",
        ],
    );
    assert!(
        !owner.contains("pub "),
        "native owner fields must stay private"
    );
    require_all(
        &source,
        &[
            "#[must_use = \"resident generation work must be consumed on the admitted run stream\"]",
            "enum ResidentGenerationRunStateV1",
            "StrictIdle",
            "InFlight",
            "Sealed",
            "Poisoned",
            "bind_population_session_import_v1(",
            "pub(crate) generation_ready_event: *mut c_void",
            "raw.generation_ready_event.is_null()",
            "raw.generation_ready_event == raw.resident_parent_ready_event",
            "impl Drop for ResidentGenerationDeviceRunV1",
            "leak_live_native_generation_run_v1(",
        ],
    );
    for forbidden in [
        "impl Clone for ResidentGenerationDeviceRunV1",
        "impl Default for ResidentGenerationDeviceRunV1",
        "Deserialize",
        "pub fn raw_",
        "pub fn from_raw",
        "pub fn from_hash",
    ] {
        assert!(
            !source.contains(forbidden),
            "owner escape via {forbidden:?}"
        );
    }
}

#[test]
fn private_abi_uses_one_fixed_stride_normalized_gene_schema_and_checked_extents() {
    let header = read_required("native/resident_generation_v1_abi.cuh");
    require_all(
        &header,
        &[
            "constexpr std::uint32_t NEO_RESIDENT_GENERATION_ABI_V1 = 1;",
            "struct NeoResidentGenerationGeneScalarV1",
            "std::uint64_t gene_identity;",
            "std::uint64_t content_hash;",
            "std::uint32_t term_count;",
            "std::uint32_t smc_flags;",
            "double long_threshold;",
            "double short_threshold;",
            "double target_pips;",
            "double stop_pips;",
            "double stop_vol_multiplier;",
            "std::uint32_t generation;",
            "struct NeoResidentGenerationPlanV1",
            "std::uint64_t logical_population_count;",
            "std::uint64_t retained_evaluation_capacity;",
            "std::uint64_t feature_count;",
            "std::uint32_t max_terms_per_gene;",
            "std::uint32_t minimum_terms_per_gene;",
            "std::uint64_t generation_count;",
            "std::uint8_t generation_semantics_sha256[32];",
            "struct NeoResidentGenerationMetricRowV1",
            "std::uint64_t candidate_id;",
            "std::uint64_t scenario_id;",
            "double values[11];",
            "static_assert(sizeof(NeoResidentGenerationMetricRowV1) == 104",
        ],
    );
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &cuda,
        &[
            "normalize_fixed_stride_gene_v1(",
            "term_count <= plan.max_terms_per_gene",
            "indicator_index < plan.feature_count",
            "weight = clamp_f64_v1(weight, -5.0, 5.0)",
            "fabs(weight) > 1.0e-6",
            "deterministic_empty_gene_repair_v1(",
            "checked_gene_term_extent_v1(",
        ],
    );
}

#[test]
fn cuda_checked_byte_arithmetic_uses_exact_size_t_operands_on_lp64() {
    let cuda = read_required("native/resident_generation_v1.cu");

    require_all(
        &cuda,
        &[
            "std::size_t{5} * sizeof(std::uint64_t)",
            "std::size_t{3} * sizeof(std::uint64_t)",
            "std::size_t{2} * sizeof(std::uint32_t)",
            "std::size_t{12} * sizeof(std::uint64_t)",
        ],
    );
    for forbidden in [
        "5ull * sizeof(std::uint64_t)",
        "3ull * sizeof(std::uint64_t)",
        "2ull * sizeof(std::uint32_t)",
        "12ull * sizeof(std::uint64_t)",
        "static_cast<std::size_t>(5ull",
        "static_cast<std::size_t>(3ull",
        "static_cast<std::size_t>(2ull",
        "static_cast<std::size_t>(12ull",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "LP64 checked arithmetic still uses a mismatched or masking operand {forbidden:?}"
        );
    }
}

#[test]
fn device_reachable_maxima_are_exact_typed_expressions_without_relaxed_constexpr() {
    let cuda = read_required("native/resident_generation_v1.cu");
    let build = read_required("build.rs");

    assert_eq!(
        cuda.matches("std::numeric_limits<std::uint64_t>::max()")
            .count(),
        0,
        "device-reachable u64 maxima must not call the host standard-library constexpr"
    );
    assert_eq!(
        cuda.matches("std::numeric_limits<std::uint32_t>::max()")
            .count(),
        1,
        "only the host-side import sentinel may retain numeric_limits<u32>::max()"
    );
    assert!(
        cuda.matches("const std::uint64_t u64_max_v1 = ~std::uint64_t{0};")
            .count()
            >= 4,
        "every device-reachable overflow domain needs an exact typed u64 maximum"
    );
    require_all(
        &cuda,
        &["const std::uint32_t u32_max_v1 = ~std::uint32_t{0};"],
    );
    for source in [&cuda, &build] {
        assert!(
            !source.contains("expt-relaxed-constexpr"),
            "the native build must not weaken CUDA constexpr authority"
        );
    }
}

#[test]
fn strict_v1_admits_only_rank_weighted_and_refuses_every_other_policy() {
    let rust = read_required("src/resident_generation_v1.rs");
    let header = read_required("native/resident_generation_v1_abi.cuh");
    require_all(
        &rust,
        &[
            "DISCOVERY_GENERATION_SEMANTICS_V1",
            "SealedResidentGenerationPlanV1",
            "ParentSelectionPolicyV1::RankWeighted",
            "SurvivorSelectionPolicyV1::RankWeighted",
            "UnsupportedUniformSelection",
            "UnsupportedTournamentSelection",
            "UnsupportedSoftmaxSelection",
            "UnsupportedElitistSelection",
            "UnsupportedGenerationalSelection",
            "validate_rank_weighted_only_v1(",
        ],
    );
    require_all(
        &header,
        &[
            "NEO_RESIDENT_PARENT_RANK_WEIGHTED_V1",
            "NEO_RESIDENT_SURVIVOR_RANK_WEIGHTED_V1",
            "NEO_RESIDENT_STATUS_UNSUPPORTED_SELECTION_V1",
        ],
    );
    for forbidden in [
        "unwrap_or(ParentSelectionPolicyV1::RankWeighted)",
        "_ => ParentSelectionPolicyV1::RankWeighted",
        "allow_cpu",
        "gpu_preferred",
    ] {
        assert!(
            !rust.contains(forbidden),
            "strict selection silently remaps via {forbidden:?}"
        );
    }
}

#[test]
fn allocation_plan_charges_every_store_and_queries_one_reusable_cub_arena() {
    let rust = read_required("src/resident_generation_v1.rs");
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &rust,
        &[
            "pub struct ActualResidentGenerationAllocationPlanV1",
            "logical_gene_scalar_bytes",
            "logical_gene_index_bytes",
            "logical_gene_weight_bytes",
            "offspring_bytes",
            "metric_row_bytes",
            "rank_key_bytes",
            "selection_bytes",
            "dedup_hash_bytes",
            "cub_scratch_bytes",
            "retained_evaluation_workspace_bytes",
            "checked_add",
            "checked_mul",
            "same_context_free_bytes",
            "full_discovery_reserve_bytes",
        ],
    );
    require_all(
        &cuda,
        &[
            "query_cub_generation_scratch_bytes_v1(",
            "cub::DeviceRadixSort::SortPairs",
            "cub::DeviceSelect::Flagged",
            "cub::DeviceRunLengthEncode::Encode",
            "cudaMallocAsync",
            "cudaFreeAsync",
            "generation_store_allocation_count = 1",
            "sizeof(NeoResidentGenerationMetricRowV1)",
            "std::size_t{5} * sizeof(std::uint64_t)",
        ],
    );
    for forbidden in [
        "MAX_TRADES_PER_CANDIDATE",
        "outcome_bytes",
        "accepted_trade_total",
        "cudaMalloc(",
        "cudaFree(",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "generation allocation retains legacy or synchronizing storage via {forbidden:?}"
        );
    }
}

#[test]
fn philox4x32_10_has_exact_constants_counter_mapping_and_host_oracle_vectors() {
    let rust = read_required("src/resident_generation_v1.rs");
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &rust,
        &[
            "pub fn philox4x32_10_reference_v1(",
            "0xD251_1F53",
            "0xCD9E_8D57",
            "0x9E37_79B9",
            "0xBB67_AE85",
            "[0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]",
            "checked_philox_counter_mapping_v1(",
            "search_seed",
            "run_identity_sha256",
            "generation_index",
            "candidate_identity",
            "genetic_operator_identity",
            "draw_index",
        ],
    );
    require_all(
        &cuda,
        &[
            "philox4x32_10_v1(",
            "PHILOX_M0_V1 = 0xD2511F53u",
            "PHILOX_M1_V1 = 0xCD9E8D57u",
            "PHILOX_W0_V1 = 0x9E3779B9u",
            "PHILOX_W1_V1 = 0xBB67AE85u",
            "for (int round = 0; round < 10; ++round)",
            "NeoResidentPhiloxOperatorV1",
        ],
    );
    for forbidden in [
        "curandState",
        "curand_init",
        "clock64()",
        "std::random_device",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "stateful/nondeterministic RNG via {forbidden:?}"
        );
    }
}

#[test]
fn rank_parent_survivor_and_dedup_decisions_are_integer_and_device_resident() {
    let header = read_required("native/resident_generation_v1_abi.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");
    let plan = section(&header, "struct NeoResidentGenerationPlanV1 {", "\n};");
    require_all(plan, &["std::uint8_t cuda_build_manifest_sha256[32];"]);
    require_all(
        &cuda,
        &[
            "stable_gene_identity_tie_key_v1(",
            "resident_decision_keys_device",
            "cub::DeviceRadixSort::SortPairsDescending",
            "cub::DeviceRadixSort::SortPairs",
            "identity_equal_v1(import->cuda_build_manifest_sha256,",
            "plan->cuda_build_manifest_sha256)",
            "rank_weight_v1(",
            "return logical_population_count - rank;",
            "checked_rank_weight_total_v1(",
            "philox_uniform_below_without_modulo_bias_v1(",
            "select_rank_weighted_parents_kernel_v1",
            "select_rank_weighted_survivors_kernel_v1",
            "cub::DeviceRunLengthEncode::Encode",
            "full_fixed_stride_gene_equal_v1(",
            "gene_hash_collision_fault_device",
            "cub::DeviceSelect::Flagged",
        ],
    );
    for forbidden in [
        "partial_cmp",
        "exp(",
        "expf(",
        "curand",
        "host_rank",
        "std::sort",
        "thrust::sort",
        "fitness_total_order_key_v1(",
        "non_finite_fitness_is_worst_v1(",
        "cuda::execution::determinism::run_to_run",
        "require_run_to_run_determinism_v1",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "decision path uses forbidden {forbidden:?}"
        );
    }
}

#[test]
fn metric_rows_bind_identity_but_scoring_and_novelty_supply_the_sealed_u64_decision_keys() {
    let rust = read_required("src/resident_generation_v1.rs");
    let header = read_required("native/resident_generation_v1_abi.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");

    require_all(
        &header,
        &[
            "struct NeoResidentGenerationMetricRowV1",
            "const NeoResidentGenerationMetricRowV1* metric_rows_device;",
            "const std::uint64_t* resident_decision_keys_device;",
            "const std::uint64_t* expected_scenario_ids_device;",
            "cudaEvent_t scoring_novelty_ready_event;",
            "std::uint8_t metric_semantics_sha256[32];",
            "std::uint8_t scoring_semantics_sha256[32];",
            "std::uint8_t novelty_semantics_sha256[32];",
            "std::uint8_t scenario_order_semantics_sha256[32];",
        ],
    );
    require_all(
        &rust,
        &[
            "struct RawResidentGenerationMetricRowV1",
            "const _: [(); 104] = [(); std::mem::size_of::<RawResidentGenerationMetricRowV1>()];",
            "resident_decision_keys_device: *const u64",
            "expected_scenario_ids_device: *const u64",
            "scoring_semantics_sha256: [u8; 32]",
            "novelty_semantics_sha256: [u8; 32]",
            "scenario_order_semantics_sha256: [u8; 32]",
        ],
    );
    require_all(
        &cuda,
        &[
            "validate_and_import_scored_rows_kernel_v1",
            "row.candidate_id != scalars[logical_candidate].gene_identity",
            "row.scenario_id != expected_scenario_ids[item]",
            "destination_rows[logical_candidate] = row",
            "destination_decision_keys[logical_candidate] = source_decision_keys[item]",
            "identity_equal_v1(metrics->scoring_semantics_sha256,",
            "identity_equal_v1(metrics->novelty_semantics_sha256,",
            "identity_equal_v1(metrics->scenario_order_semantics_sha256,",
            "keys[position] = candidate_valid_flags[candidate] != 0",
            "? resident_decision_keys[candidate]",
        ],
    );

    let gather = section(
        &cuda,
        "gather_resident_decision_rank_keys_kernel_v1(",
        "\n}",
    );
    for forbidden in ["values[0]", "metric_rows", "double fitness", "isfinite"] {
        assert!(
            !gather.contains(forbidden),
            "generation rank inferred scoring from raw metrics via {forbidden:?}"
        );
    }
}

#[test]
fn initial_population_crossover_mutation_and_offspring_never_leave_the_device() {
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &cuda,
        &[
            "initialize_fixed_stride_population_kernel_v1",
            "launch_device_parent_selection_v1(",
            "launch_device_crossover_v1(",
            "launch_device_mutation_v1(",
            "launch_device_gene_hash_v1(",
            "rotate_resident_generation_stores_v1(",
            "offspring_gene_scalars_device",
            "offspring_gene_indices_device",
            "offspring_gene_weights_device",
            "mutation_intensity_q32",
            "threshold_ladder_bits",
            "stop_bounds_bits",
            "smc_probability_q32",
            "strict_generation_has_no_candidate_revival_v1",
        ],
    );
    for forbidden in [
        "best_effort",
        "fallback",
        "rescue_gene",
        "keeping portfolio",
        "std::vector",
        "std::unordered_set",
        "rayon",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "resident operators escape through {forbidden:?}"
        );
    }
}

#[test]
fn exact_chunks_cover_the_logical_population_without_padding_or_reallocation() {
    let rust = read_required("src/resident_generation_v1.rs");
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &rust,
        &[
            "checked_generation_chunk_count_v1(",
            "checked_generation_chunk_range_v1(",
            "retained_evaluation_capacity >= 1",
            "active_scenarios <= retained_evaluation_capacity",
            "covered_logical_population == logical_population_count",
        ],
    );
    require_all(
        &cuda,
        &[
            "active_scenarios <= run->retained_evaluation_capacity",
            "logical_offset + active_scenarios <= run->logical_population_count",
            "enqueue_exact_generation_chunk_v1(",
            "exact_chunk_coverage_device",
            "exact_candidate_identity_without_fillers_v1",
        ],
    );
    let chunk = section(&cuda, "enqueue_exact_generation_chunk_v1(", "\n}");
    for forbidden in ["cudaMalloc", "cudaFree", "padding", "dummy_candidate"] {
        assert!(
            !chunk.contains(forbidden),
            "chunk path mutates capacity via {forbidden:?}"
        );
    }
}

#[test]
fn every_operation_uses_the_imported_stream_and_only_event_dependencies_cross_stages() {
    let header = read_required("native/resident_generation_v1_abi.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");
    require_all(
        &header,
        &[
            "struct NeoResidentGenerationPopulationSessionImportV1",
            "cudaStream_t admitted_run_stream;",
            "cudaEvent_t generation_ready_event;",
            "std::uint8_t primary_context_identity_sha256[32];",
            "std::uint8_t run_stream_identity_sha256[32];",
            "std::uint8_t cuda_build_manifest_sha256[32];",
        ],
    );
    require_all(
        &cuda,
        &[
            "import->admitted_run_stream",
            "run->admitted_run_stream",
            "created->ready_event = import->generation_ready_event",
            "import->generation_ready_event != import->resident_parent_ready_event",
            "cudaStreamWaitEvent(run->admitted_run_stream",
            "cudaEventRecord",
            "consume_resident_generation_event_dependency_v1(",
            "same_stream_enqueue_count",
            "intermediate_host_wait_count = 0",
            "intermediate_readback_count = 0",
        ],
    );
    for forbidden in [
        "cudaStreamCreate",
        "cudaEventCreate",
        "cudaEventDestroy",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cudaMemcpy",
        "cudaMemcpyAsync",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "stream/transfer violation via {forbidden:?}"
        );
    }
}

#[test]
fn sealed_handoff_is_content_addressed_resident_and_not_promotion_authority() {
    let rust = read_required("src/resident_generation_v1.rs");
    let receipt = section(
        &rust,
        "pub struct SealedResidentGenerationDeviceOutcomeV1 {",
        "\n}",
    );
    require_all(
        receipt,
        &[
            "ready: ResidentGenerationReadyEventV1",
            "resident_gene_content: ResidentGenerationContentIdentityV1",
            "resident_metric_content: ResidentGenerationContentIdentityV1",
            "resident_generation_receipt: ResidentGenerationReceiptIdentityV1",
            "artifact_class: GenerationArtifactClassV1",
            "promotion_eligibility: GenerationPromotionEligibilityV1",
        ],
    );
    assert!(
        !receipt.contains("pub "),
        "sealed handoff fields must be private"
    );
    require_all(
        &rust,
        &[
            "pub struct ResidentGenerationReadyEventV1",
            "run: Option<ResidentGenerationDeviceRunV1>",
            "consume_into_post_ga_v1(",
            "GenerationArtifactClassV1::ResearchOnly",
            "GenerationPromotionEligibilityV1::NotPromotionEligible",
            "seal_content_identities_on_device_v1(",
            "generation_semantics_sha256",
            "selected_cuda_ordinal",
            "cuda_build_manifest_sha256",
            "ResidentGenerationContentIdentityV1",
            "ResidentGenerationReceiptIdentityV1",
            "final_compact_readback_count == 0",
        ],
    );
    for forbidden in [
        "impl Clone for SealedResidentGenerationDeviceOutcomeV1",
        "impl Default for SealedResidentGenerationDeviceOutcomeV1",
        "pub fn from_hash",
        "pub genes: Vec",
        "pub metrics: Vec",
    ] {
        assert!(
            !rust.contains(forbidden),
            "sealed handoff escape via {forbidden:?}"
        );
    }
}
