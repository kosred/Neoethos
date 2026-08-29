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

fn compact_whitespace(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn validate_slice2_wait_topology(
    generation_cuda: &str,
    scoring_cuda: &str,
    archive_cuda: &str,
) -> Result<(), String> {
    let generation_create = section(
        generation_cuda,
        "extern \"C\" std::int32_t create_resident_generation_run_from_import_v1(",
        "\n}",
    );
    let finite_scoring = section(
        scoring_cuda,
        "std::int32_t enqueue_resident_scoring_finite_objective_v2(",
        "\n}",
    );
    let wait = "cudaStreamWaitEvent(created->admitted_run_stream,\n                               created->resident_parent_ready_event, 0)";
    if generation_create.matches("cudaStreamWaitEvent").count() != 1
        || !generation_create.contains(wait)
    {
        return Err("generation create must retain the one exact parent-ready wait".to_owned());
    }
    if finite_scoring.contains("cudaStreamWaitEvent") {
        return Err("finite scoring must be same-stream eventless".to_owned());
    }
    if archive_cuda.contains("cudaStreamWaitEvent") {
        return Err("archive composition must not enqueue another wait".to_owned());
    }
    Ok(())
}

fn validate_slice2_terminal_event_advance(
    internal_header: &str,
    generation_cuda: &str,
) -> Result<(), String> {
    let lifecycle = section(
        internal_header,
        "class ResidentGenerationTerminalLifecycleV2 {",
        "\n};",
    );
    let borrow = section(
        generation_cuda,
        "borrow_resident_generation_terminal_lifecycle_v2(",
        "\n}",
    );
    let accept = section(
        generation_cuda,
        "accept_resident_generation_terminal_enqueue_v2(",
        "\n}",
    );
    for required in [
        "source_ready_receipt_v2() const",
        "resident_parent_ready_event_v2() const",
        "source_event_id_v2() const",
        "source_same_stream_enqueue_count_v2() const",
    ] {
        if !lifecycle.contains(required) {
            return Err(format!("terminal lifecycle is missing {required:?}"));
        }
    }
    for required in [
        "run->next_event_id == ~std::uint64_t{0}",
        "lifecycle->completion_event_identity_ = run->next_event_id + 1ull;",
    ] {
        if !borrow.contains(required) {
            return Err(format!("terminal borrow is missing {required:?}"));
        }
    }
    let compact_accept = compact_whitespace(accept);
    for required in [
        "lifecycle->completion_event_identity_v2() == generation->next_event_id + 1ull",
        "generation->next_event_id = lifecycle->completion_event_identity_v2();",
    ] {
        if !compact_accept.contains(required) {
            return Err(format!("terminal accept is missing {required:?}"));
        }
    }
    if compact_accept
        .matches("generation->next_event_id = lifecycle->completion_event_identity_v2();")
        .count()
        != 1
        || compact_accept.contains("++generation->next_event_id")
    {
        return Err("terminal accept must advance next_event_id exactly once".to_owned());
    }
    Ok(())
}

fn validate_slice2_publish_bounds_are_fail_closed(source: &str) -> Result<(), String> {
    let publish = section(
        source,
        "__device__ ResidentGenerationDevicePublishResultV2 publish_device_v2(",
        "\n  }\n\n private:",
    );
    let compact = compact_whitespace(publish);
    let generation_bound = "expected_next_generation_index_ <= 0xffffull";
    let epoch_bound = "expected_next_store_epoch_ <= 0x7fffffffull";
    let bound_fault = "!exact_device_identity || !packed_commit_bounds_v2";
    let combined_fault = "result.combined_fault =";
    let pre_mutation_return = "if (!packed_commit_bounds_v2) { return result; }";

    let position = |token: &str| {
        compact
            .find(token)
            .ok_or_else(|| format!("missing fail-closed publish token {token:?}"))
    };
    let generation_bound_at = position(generation_bound)?;
    let epoch_bound_at = position(epoch_bound)?;
    let bound_fault_at = position(bound_fault)?;
    let combined_fault_at = position(combined_fault)?;
    let pre_mutation_return_at = position(pre_mutation_return)?;
    let first_generation_mutation_at = [
        "seal->fault_code =",
        "seal->flags |=",
        "seal->current_store_index = expected_next_store_index_;",
        "seal->generation_index = expected_next_generation_index_;",
        "seal->store_epoch = expected_next_store_epoch_;",
        "control->fault_word =",
        "control->stop_requested =",
        "control->generation_index =",
        "control->executed_generations =",
        "control->current_store_index =",
    ]
    .into_iter()
    .filter_map(|token| compact.find(token))
    .min()
    .ok_or_else(|| "publish method has no observable generation mutation".to_owned())?;

    if generation_bound_at < epoch_bound_at
        && epoch_bound_at < bound_fault_at
        && bound_fault_at < combined_fault_at
        && combined_fault_at < pre_mutation_return_at
        && pre_mutation_return_at < first_generation_mutation_at
    {
        Ok(())
    } else {
        Err("packed generation/epoch bounds must fault and return before every seal/control mutation"
            .to_owned())
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

#[test]
fn slice2_uses_a_private_typed_generation_split_without_per_generation_host_completion() {
    let public_v2 = read_required("native/resident_generation_v2_abi.cuh");
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");

    require_all(
        &internal_v2,
        &[
            "namespace neoethos::resident_generation_v2_internal",
            "struct ResidentGenerationScoredRowsV2",
            "class ResidentGenerationPreparedAdvanceV2",
            "generation_owner_",
            "population_lifetime_owner_",
            "admitted_run_stream_",
            "device_seal_identity_",
            "expected_old_generation_index_",
            "expected_next_generation_index_",
            "expected_old_store_epoch_",
            "expected_next_store_epoch_",
            "expected_old_store_index_",
            "expected_next_store_index_",
            "same_stream_enqueue_count_",
            "publish_device_v2(",
            "enqueue_resident_generation_offspring_from_scored_rows_v2(",
            "accept_resident_generation_combined_publish_v2(",
        ],
    );
    for forbidden in [
        "extern \"C\"",
        "cudaMemcpy",
        "cudaEventRecord",
        "cudaEventQuery",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ] {
        assert!(
            !internal_v2.contains(forbidden),
            "private generation split header crossed a host boundary via {forbidden:?}"
        );
    }
    for private_name in [
        "ResidentGenerationScoredRowsV2",
        "ResidentGenerationPreparedAdvanceV2",
        "enqueue_resident_generation_offspring_from_scored_rows_v2",
        "accept_resident_generation_combined_publish_v2",
    ] {
        assert!(
            !public_v2.contains(private_name),
            "native-only generation authority escaped through the public ABI as {private_name:?}"
        );
    }

    let split_enqueue = section(
        &cuda,
        "enqueue_resident_generation_offspring_from_scored_rows_v2(",
        "\n}",
    );
    require_all(
        split_enqueue,
        &[
            "validate_and_import_scored_rows_kernel_v1<<<",
            "retained_view->expected_generation_index ==",
            "retained_view->expected_store_epoch ==",
            "launch_device_parent_selection_v1(",
            "launch_device_crossover_v1(",
            "launch_device_mutation_v1(",
            "launch_device_gene_hash_v1(",
        ],
    );
    for stale_authority in [
        "retained_view->expected_generation_index <=",
        "retained_view->expected_store_epoch <=",
    ] {
        assert!(
            !split_enqueue.contains(stale_authority),
            "split generation enqueue accepted stale authority via {stale_authority:?}"
        );
    }
    for forbidden in [
        "cudaMemcpyDeviceToHost",
        "cudaEventRecord",
        "cudaEventQuery",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ] {
        assert!(
            !split_enqueue.contains(forbidden),
            "split generation enqueue introduced per-generation completion via {forbidden:?}"
        );
    }

    let legacy = section(
        &cuda,
        "enqueue_full_population_scored_generation_advance_v2(",
        "\n}",
    );
    require_all(
        legacy,
        &[
            "enqueue_resident_generation_offspring_from_scored_rows_v2(",
            "publish_one_generation_commit_kernel_v2<<<",
            "cudaMemcpyAsync(generation->terminal_host_receipt_v2",
            "cudaEventRecord(generation->ready_event",
        ],
    );
}

#[test]
fn slice2_generation_finite_rows_seam_is_exact_and_eventless() {
    let public_v2 = read_required("native/resident_generation_v2_abi.cuh");
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");
    let symbol = "enqueue_resident_generation_offspring_from_finite_rows_v2";
    let compact_internal_v2 = compact_whitespace(&internal_v2);

    require_all(
        &compact_internal_v2,
        &[
            "#include \"resident_scoring_novelty_v2_internal.cuh\"",
            "enqueue_resident_generation_offspring_from_finite_rows_v2(",
            "const resident_scoring_novelty_v2_internal::ResidentScoringFiniteObjectiveRowsV2*",
            "const std::uint64_t* ranked_decision_keys_device",
            "resident_generation_v2::NeoResidentGenerationGeneViewV2*",
            "ResidentGenerationPreparedAdvanceV2*",
        ],
    );
    assert!(!public_v2.contains(symbol));

    let enqueue = section(&cuda, &format!("{symbol}("), "\n}");
    let compact_enqueue = compact_whitespace(enqueue);
    require_all(
        &compact_enqueue,
        &[
            "finite_rows->scoring_owner",
            "finite_rows->admitted_run_stream != generation->admitted_run_stream",
            "finite_rows->logical_population_count != generation->logical_population_count",
            "finite_rows->metric_rows_device",
            "finite_rows->expected_scenario_ids_device",
            "finite_rows->fitness_scores_device",
            "finite_rows->decision_keys_device",
            "finite_rows->device_seal",
            "ranked_decision_keys_device == nullptr",
            "retained_generation_view == nullptr",
            "finite_rows->metric_semantics_sha256",
            "finite_rows->scoring_semantics_sha256",
            "finite_rows->novelty_semantics_sha256",
            "finite_rows->scenario_order_semantics_sha256",
            "finite_rows->rank_semantics_sha256",
            "finite_rows->cuda_build_manifest_sha256",
            "finite_rows->cuda_math_flags_sha256",
            "validate_and_import_scored_rows_kernel_v1<<<",
            "launch_device_parent_selection_v1(",
            "launch_device_crossover_v1(",
            "launch_device_mutation_v1(",
            "launch_device_gene_hash_v1(",
        ],
    );
    for forbidden in [
        "scoring_novelty_ready_event",
        "cudaStreamWaitEvent",
        "cudaEventRecord",
        "cudaEventQuery",
        "cudaEventSynchronize",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaMemcpyDeviceToHost",
    ] {
        assert!(
            !enqueue.contains(forbidden),
            "finite-row generation seam crossed its eventless boundary via {forbidden:?}"
        );
    }
}

#[test]
fn slice2_has_one_pre_search_parent_wait_and_no_phase_local_waits() {
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let archive_cuda = read_required("native/resident_archive_knn_v2.cu");

    assert!(
        validate_slice2_wait_topology(&generation_cuda, &scoring_cuda, &archive_cuda).is_ok(),
        "the combined path must wait once at generation creation and never in finite scoring/archive phases"
    );

    let missing_parent_wait = generation_cuda.replacen(
        "cudaStreamWaitEvent(created->admitted_run_stream,\n                               created->resident_parent_ready_event, 0)",
        "cudaSuccess",
        1,
    );
    assert_ne!(missing_parent_wait, generation_cuda);
    assert!(
        validate_slice2_wait_topology(&missing_parent_wait, &scoring_cuda, &archive_cuda).is_err(),
        "the contract must kill removal of the sole parent-ready wait"
    );

    let duplicate_scoring_wait = scoring_cuda.replacen(
        "run->expected_scenario_ids_device = population->expected_scenario_ids_device;",
        "run->expected_scenario_ids_device = population->expected_scenario_ids_device;\n  cudaStreamWaitEvent(run->admitted_run_stream, population->metrics_ready_event, 0);",
        1,
    );
    assert_ne!(duplicate_scoring_wait, scoring_cuda);
    assert!(
        validate_slice2_wait_topology(&generation_cuda, &duplicate_scoring_wait, &archive_cuda)
            .is_err(),
        "the contract must kill a per-generation scoring wait"
    );
}

#[test]
fn slice2_initial_archive_dependency_is_exact_bound_authority() {
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let archive_cuda = read_required("native/resident_archive_knn_v2.cu");

    require_all(
        &internal_v2,
        &[
            "source_ready_receipt_v2() const",
            "resident_parent_ready_event_v2() const",
            "source_event_id_v2() const",
            "source_same_stream_enqueue_count_v2() const",
            "source_ready_receipt_",
            "resident_parent_ready_event_",
            "source_event_id_",
            "source_same_stream_enqueue_count_",
        ],
    );
    let borrow = section(
        &generation_cuda,
        "borrow_resident_generation_terminal_lifecycle_v2(",
        "\n}",
    );
    require_all(
        &compact_whitespace(borrow),
        &[
            "run->source_ready_receipt_token_v2",
            "run->source_event_id_v2",
            "run->source_same_stream_enqueue_count_v2",
            "run->resident_parent_ready_event",
            "lifecycle->source_ready_receipt_",
            "lifecycle->resident_parent_ready_event_",
            "lifecycle->source_event_id_",
            "lifecycle->source_same_stream_enqueue_count_",
        ],
    );
    let score = section(
        &archive_cuda,
        "extern \"C\" std::int32_t enqueue_resident_archive_score_and_rank_v2(",
        "\n}",
    );
    let compact_score = compact_whitespace(score);
    require_all(
        &compact_score,
        &[
            "dependency != owner->terminal_lifecycle.source_ready_receipt_v2()",
            "dependency->event_id != owner->terminal_lifecycle.source_event_id_v2()",
            "dependency->same_stream_enqueue_count != owner->terminal_lifecycle.source_same_stream_enqueue_count_v2()",
            "population->metrics_ready_event != owner->terminal_lifecycle.resident_parent_ready_event_v2()",
            "population->population_lifetime_owner != owner->terminal_lifecycle.population_lifetime_owner_v2()",
        ],
    );
    assert!(
        !score.contains("owner->same_stream_enqueue_count = dependency->same_stream_enqueue_count"),
        "an external receipt value must not overwrite the composite global enqueue count"
    );
    let scoring_helper = section(
        &scoring_cuda,
        "std::int32_t enqueue_resident_scoring_finite_objective_v2(",
        "\n}",
    );
    require_all(
        &compact_whitespace(scoring_helper),
        &[
            "run->population_lifetime_owner = population->population_lifetime_owner;",
            "run->metrics_ready_event = population->metrics_ready_event;",
        ],
    );
}

#[test]
fn slice2_generation_terminal_lifecycle_borrows_existing_resources_without_cuda_ops() {
    let public_v2 = read_required("native/resident_generation_v2_abi.cuh");
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");
    let cuda = read_required("native/resident_generation_v1.cu");

    let lifecycle = section(
        &internal_v2,
        "class ResidentGenerationTerminalLifecycleV2 {",
        "\n};",
    );
    let compact_lifecycle = compact_whitespace(lifecycle);
    require_all(
        &compact_lifecycle,
        &[
            " private:",
            "generation_owner_",
            "admitted_run_stream_",
            "completion_event_",
            "terminal_host_receipt_",
            "terminal_host_receipt_bytes_",
            "completion_event_identity_",
            "source_ready_receipt_",
            "resident_parent_ready_event_",
            "source_event_id_",
            "source_same_stream_enqueue_count_",
            "run_token_",
            "generation_index_",
            "store_epoch_",
            "current_store_index_",
            "same_stream_enqueue_count_",
            "generation_owner_v2() const",
            "population_lifetime_owner_v2() const",
            "admitted_run_stream_v2() const",
            "completion_event_v2() const",
            "terminal_host_receipt_v2() const",
            "terminal_host_receipt_bytes_v2() const",
            "completion_event_identity_v2() const",
            "source_ready_receipt_v2() const",
            "resident_parent_ready_event_v2() const",
            "source_event_id_v2() const",
            "source_same_stream_enqueue_count_v2() const",
            "run_token_v2() const",
            "generation_index_v2() const",
            "store_epoch_v2() const",
            "current_store_index_v2() const",
            "same_stream_enqueue_count_v2() const",
        ],
    );
    require_all(
        &internal_v2,
        &[
            "bool borrow_resident_generation_terminal_lifecycle_v2(",
            "std::uint64_t expected_terminal_host_receipt_bytes",
            "bool accept_resident_generation_terminal_enqueue_v2(",
            "std::uint64_t final_same_stream_enqueue_count",
        ],
    );
    for private_name in [
        "ResidentGenerationTerminalLifecycleV2",
        "borrow_resident_generation_terminal_lifecycle_v2",
        "accept_resident_generation_terminal_enqueue_v2",
    ] {
        assert!(!public_v2.contains(private_name));
    }

    let borrow = section(
        &cuda,
        "borrow_resident_generation_terminal_lifecycle_v2(",
        "\n}",
    );
    let compact_borrow = compact_whitespace(borrow);
    require_all(
        &compact_borrow,
        &[
            "expected_terminal_host_receipt_bytes != sizeof(NeoResidentSearchTerminalReceiptV2)",
            "run->admitted_run_stream",
            "run->population_lifetime_owner",
            "run->ready_event",
            "run->terminal_host_receipt_v2",
            "run->next_event_id",
            "run->run_token",
            "run->current_generation_index",
            "run->store_epoch_v2",
            "run->current_store_index_v2",
            "run->same_stream_enqueue_count",
        ],
    );
    let accept = section(
        &cuda,
        "accept_resident_generation_terminal_enqueue_v2(",
        "\n}",
    );
    let compact_accept = compact_whitespace(accept);
    require_all(
        &compact_accept,
        &[
            "lifecycle->generation_owner_v2()",
            "lifecycle->population_lifetime_owner_v2()",
            "lifecycle->admitted_run_stream_v2()",
            "lifecycle->completion_event_v2()",
            "lifecycle->terminal_host_receipt_v2()",
            "lifecycle->terminal_host_receipt_bytes_v2()",
            "lifecycle->completion_event_identity_v2()",
            "lifecycle->run_token_v2()",
            "lifecycle->generation_index_v2()",
            "lifecycle->store_epoch_v2()",
            "lifecycle->current_store_index_v2()",
            "lifecycle->same_stream_enqueue_count_v2() + 3ull",
            "generation->same_stream_enqueue_count = final_same_stream_enqueue_count;",
            "generation->next_event_id = lifecycle->completion_event_identity_v2();",
        ],
    );
    for body in [borrow, accept] {
        for forbidden in [
            "cudaMemcpy",
            "cudaEventRecord",
            "cudaEventQuery",
            "cudaEventSynchronize",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
        ] {
            assert!(!body.contains(forbidden));
        }
    }
    for forbidden_assignment in [
        "generation->current_generation_index =",
        "generation->store_epoch_v2 =",
        "generation->current_store_index_v2 =",
    ] {
        assert!(!compact_accept.contains(forbidden_assignment));
    }
}

#[test]
fn slice2_terminal_event_identity_advances_once_and_kills_reuse_mutants() {
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let archive_cuda = read_required("native/resident_archive_knn_v2.cu");
    assert!(
        validate_slice2_terminal_event_advance(&internal_v2, &generation_cuda).is_ok(),
        "terminal completion must reserve next_event_id + 1 and accept it exactly once"
    );

    let reused_event_id = generation_cuda.replacen(
        "lifecycle->completion_event_identity_ = run->next_event_id + 1ull;",
        "lifecycle->completion_event_identity_ = run->next_event_id;",
        1,
    );
    assert_ne!(reused_event_id, generation_cuda);
    assert!(
        validate_slice2_terminal_event_advance(&internal_v2, &reused_event_id).is_err(),
        "the contract must kill reuse of the source event identity"
    );

    let missing_accept_advance = generation_cuda.replacen(
        "generation->next_event_id = lifecycle->completion_event_identity_v2();",
        "generation->next_event_id = generation->next_event_id;",
        1,
    );
    assert_ne!(missing_accept_advance, generation_cuda);
    assert!(
        validate_slice2_terminal_event_advance(&internal_v2, &missing_accept_advance).is_err(),
        "the contract must kill failure to commit the reserved event identity"
    );

    let terminal_enqueue = section(
        &archive_cuda,
        "extern \"C\" std::int32_t enqueue_resident_archive_terminal_seal_v2(",
        "\n}",
    );
    require_all(
        &compact_whitespace(terminal_enqueue),
        &[
            "lifecycle.completion_event_identity_v2(), global_final_enqueue_count",
            "pending->completion_event_identity = lifecycle.completion_event_identity_v2();",
        ],
    );
    let terminal_complete = section(
        &archive_cuda,
        "extern \"C\" std::int32_t try_complete_resident_archive_terminal_v2(",
        "\n}",
    );
    require_all(
        &compact_whitespace(terminal_complete),
        &["committed_ready->event_id = pending->completion_event_identity;"],
    );
}

#[test]
fn slice2_combined_publish_rejects_unrepresentable_next_tuple_before_generation_mutation() {
    let internal_v2 = read_required("native/resident_generation_v2_internal.cuh");

    let maximum_generation = 0xffff_u64;
    let maximum_epoch = 0x7fff_ffff_u64;
    assert_eq!(maximum_generation.checked_add(1), Some(0x1_0000));
    assert_eq!(maximum_epoch.checked_add(1), Some(0x8000_0000));
    assert!(
        validate_slice2_publish_bounds_are_fail_closed(&internal_v2).is_ok(),
        "combined publish must reject the first unrepresentable generation or epoch before changing either generation seal or control"
    );

    let generation_increment_mutant = internal_v2.replacen(
        "expected_next_generation_index_ <= 0xffffull",
        "expected_next_generation_index_ <= 0x10000ull",
        1,
    );
    assert_ne!(generation_increment_mutant, internal_v2);
    assert!(
        validate_slice2_publish_bounds_are_fail_closed(&generation_increment_mutant).is_err(),
        "the contract must kill a mutant that admits generation 65,536"
    );

    let epoch_increment_mutant = internal_v2.replacen(
        "expected_next_store_epoch_ <= 0x7fffffffull",
        "expected_next_store_epoch_ <= 0x80000000ull",
        1,
    );
    assert_ne!(epoch_increment_mutant, internal_v2);
    assert!(
        validate_slice2_publish_bounds_are_fail_closed(&epoch_increment_mutant).is_err(),
        "the contract must kill a mutant that admits epoch 2^31"
    );

    let mutation_before_failure_mutant = internal_v2.replacen(
        "if (!packed_commit_bounds_v2)",
        "if (false && !packed_commit_bounds_v2)",
        1,
    );
    assert_ne!(mutation_before_failure_mutant, internal_v2);
    assert!(
        validate_slice2_publish_bounds_are_fail_closed(&mutation_before_failure_mutant).is_err(),
        "the contract must kill a mutant that bypasses the pre-mutation return"
    );
}
