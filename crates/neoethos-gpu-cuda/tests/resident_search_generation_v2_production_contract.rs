#![cfg(any(feature = "cuda", feature = "resident-search-slice2-compile-contract"))]

use std::fs;
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use neoethos_gpu_cuda::resident_search_v2::resident_search_v2_production_readiness;

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

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(
            source.contains(token),
            "missing required source token {token:?}"
        );
    }
}

fn forbid_all(source: &str, forbidden: &[&str]) {
    for token in forbidden {
        assert!(
            !source.contains(token),
            "forbidden source token remains present: {token:?}"
        );
    }
}

fn require_in_order(source: &str, required: &[&str]) {
    let mut cursor = 0_usize;
    for token in required {
        let relative = source[cursor..]
            .find(token)
            .unwrap_or_else(|| panic!("missing ordered source token {token:?}"));
        cursor += relative + token.len();
    }
}

fn braced_item<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing braced item {needle:?}"));
    let open_relative = source[start..]
        .find('{')
        .unwrap_or_else(|| panic!("missing opening brace after {needle:?}"));
    let open = start + open_relative;
    let mut depth = 0_i64;
    for (relative, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + relative];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated braced item {needle:?}");
}

fn validate_slice2_scoring_archive_cub_scratch_query(source: &str) -> Result<(), String> {
    let query = braced_item(source, "std::int32_t query_cub_reduce_scratch_bytes_v1(");
    let required_counts = [
        (
            "const int count = static_cast<int>(plan.logical_population_count);",
            1_usize,
        ),
        (
            "auto* rank_keys = static_cast<std::uint64_t*>(nullptr);",
            1_usize,
        ),
        (
            "auto* rank_values = static_cast<std::uint64_t*>(nullptr);",
            1_usize,
        ),
        ("cub::DeviceReduce::Min(", 1_usize),
        ("cub::DeviceReduce::Max(", 1_usize),
        ("cub::DeviceRadixSort::SortPairs(", 2_usize),
        ("cub::DeviceRadixSort::SortPairsDescending(", 1_usize),
        (
            "nullptr, candidate, rank_keys, rank_keys, rank_values, rank_values,\n      count, 0, 64, stream);",
            3_usize,
        ),
        (
            "maximum = candidate > maximum ? candidate : maximum;",
            4_usize,
        ),
        ("align_device_bytes_v1(maximum, scratch_bytes)", 1_usize),
    ];
    for (token, expected) in required_counts {
        let observed = query.matches(token).count();
        if observed != expected {
            return Err(format!(
                "Slice2 combined CUB scratch query requires {expected} occurrences of {token:?}, observed {observed}"
            ));
        }
    }
    for forbidden in ["65_536", "65'536", "65536"] {
        if query.contains(forbidden) {
            return Err(format!(
                "Slice2 combined CUB scratch query must not seal a literal fallback {forbidden:?}"
            ));
        }
    }
    Ok(())
}

fn replace_nth(source: &str, needle: &str, replacement: &str, occurrence: usize) -> String {
    let (offset, _) = source
        .match_indices(needle)
        .nth(occurrence)
        .unwrap_or_else(|| panic!("missing occurrence {occurrence} of {needle:?}"));
    let mut mutant = source.to_owned();
    mutant.replace_range(offset..offset + needle.len(), replacement);
    mutant
}

fn assert_cpp_symbol_not_fixture_gated(source: &str, symbol: &str) {
    let mut fixture_depth = 0_usize;
    let mut conditional_stack = Vec::<bool>::new();
    let mut found = false;

    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#if") {
            let fixture = trimmed.contains("NEOETHOS_CUDA_DEVICE_FIXTURES_V2");
            conditional_stack.push(fixture);
            if fixture {
                fixture_depth += 1;
            }
        }
        if line.contains(symbol) {
            found = true;
            assert_eq!(
                fixture_depth, 0,
                "production symbol {symbol:?} remains fixture-gated"
            );
        }
        if trimmed.starts_with("#endif") && conditional_stack.pop().unwrap_or(false) {
            fixture_depth -= 1;
        }
    }
    assert!(found, "production symbol {symbol:?} is absent");
}

fn assert_rust_ffi_not_fixture_gated(source: &str, symbol: &str) {
    let needle = format!("fn {symbol}(");
    let offset = source
        .find(&needle)
        .unwrap_or_else(|| panic!("missing Rust FFI declaration {symbol:?}"));
    let prefix = &source[offset.saturating_sub(256)..offset];
    assert!(
        !prefix.contains("cuda-device-fixtures"),
        "Rust FFI declaration {symbol:?} remains fixture-gated"
    );
}

fn assert_under_900_lines(name: &str, source: &str) {
    let lines = source.lines().count();
    assert!(
        lines < 900,
        "{name} is {lines} lines; bounded limit is <900"
    );
}

#[test]
fn a_scoring_v2_owner_and_symbols_are_rooted_in_normal_cuda() {
    // This read is the intentional first RED against the pre-production tree.
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let lib = read_required("src/lib.rs");
    let v2_header = read_required("native/resident_search_generation_v2_abi.cuh");
    let scoring_header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let build = read_required("build.rs");

    require_all(&lib, &["mod resident_scoring_v2;"]);
    require_all(
        &build,
        &[
            "native/resident_scoring_novelty_v1.cu",
            "native/resident_generation_v1.cu",
            "native/resident_search_generation_v2_abi.cuh",
        ],
    );

    for symbol in [
        "query_resident_scoring_admission_v2",
        "create_unbound_resident_scoring_run_v2",
        "bind_and_seal_resident_scoring_v2",
        "enqueue_resident_scoring_release_v2",
    ] {
        assert_cpp_symbol_not_fixture_gated(&scoring_header, symbol);
        assert_cpp_symbol_not_fixture_gated(&scoring_cuda, symbol);
        assert_rust_ffi_not_fixture_gated(&scoring_rust, symbol);
    }
    let advance = "enqueue_full_population_scored_generation_advance_v2";
    assert_cpp_symbol_not_fixture_gated(&v2_header, advance);
    assert_cpp_symbol_not_fixture_gated(&generation_cuda, advance);
    let search = read_required("src/resident_search_v2.rs");
    assert_rust_ffi_not_fixture_gated(&search, advance);

    require_all(
        &scoring_rust,
        &[
            "pub(crate) struct ResidentScoringRunV2",
            "pub(crate) struct SealedResidentScoringPlanV2",
            "pub(crate) struct SealedResidentSearchAdmissionV2",
        ],
    );
    assert_under_900_lines("src/resident_scoring_v2.rs", &scoring_rust);
    assert_under_900_lines("native/resident_search_generation_v2_abi.cuh", &v2_header);
}

#[test]
fn b_novelty_is_sealed_to_positive_zero_and_the_wrong_kernel_is_not_production() {
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");

    require_all(
        &scoring_rust,
        &[
            "ResidentScoringObjectiveV2",
            "PropFirmV4",
            "RiskyGrowthV5",
            "RESIDENT_NOVELTY_DISABLED_SEMANTICS_V2",
            "novelty_weight.to_bits() != 0_u64",
            "InvalidNoveltyWeight",
        ],
    );
    let seal = braced_item(&scoring_rust, "fn seal_resident_scoring_plan_v2(");
    require_all(seal, &["novelty_weight.to_bits()", "0_u64"]);

    let native_validation = braced_item(&scoring_cuda, "bool validate_scoring_admission_v2(");
    require_all(native_validation, &["plan->novelty_weight_bits", "0ull"]);

    let production_bind = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t bind_and_seal_resident_scoring_v2(",
    );
    require_all(
        production_bind,
        &[
            "score_canonical_metrics_kernel_v1",
            "encode_finite_objective_keys_kernel_v2",
            "seal_scoring_novelty_content_kernel_v1",
        ],
    );
    let key_encode = braced_item(
        &scoring_cuda,
        "__global__ void encode_finite_objective_keys_kernel_v2(",
    );
    require_all(
        key_encode,
        &[
            "device_fault_word",
            "!isfinite",
            "ordered_f64_decision_key_v2",
        ],
    );
    forbid_all(
        production_bind,
        &[
            "candidate_ordered_mean_jaccard_kernel_v1",
            "cub::DeviceReduce::Min",
            "cub::DeviceReduce::Max",
            "normalized_novelty",
        ],
    );
    assert_eq!(
        scoring_cuda
            .matches("candidate_ordered_mean_jaccard_kernel_v1")
            .count(),
        2,
        "legacy all-current novelty may remain defined and used by V1 only"
    );
}

#[test]
fn c_generation_and_scoring_allocations_are_sealed_before_the_first_kernel() {
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let search = read_required("src/resident_search_v2.rs");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let population_cuda = read_required("native/prototype_b_population.cu");

    let receipt = braced_item(
        &scoring_rust,
        "pub(crate) struct SealedResidentSearchAdmissionV2",
    );
    require_all(
        receipt,
        &[
            "generation_device_bytes",
            "scoring_device_bytes",
            "total_device_bytes",
            "same_context_free_bytes",
            "full_discovery_reserve_bytes",
            "generation_allocation_plan_sha256",
            "scoring_allocation_plan_sha256",
            "receipt_identity_sha256",
        ],
    );
    let seal = braced_item(&scoring_rust, "fn seal_combined_search_admission_v2(");
    require_all(
        seal,
        &[
            "checked_add",
            "checked_sub",
            "generation_device_bytes",
            "scoring_device_bytes",
            "full_discovery_reserve_bytes",
        ],
    );

    let begin = braced_item(&search, "fn begin_resident_search_sealed_v2(");
    require_in_order(
        begin,
        &[
            "neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2",
            "seal_resident_scoring_plan_v2",
            "neoethos_gpu_cuda_population_query_resident_search_combined_v2",
            "seal_combined_search_admission_v2",
            "neoethos_gpu_cuda_population_create_resident_search_combined_v2",
            "ffi_initialize_resident_generation_population_v1",
        ],
    );
    forbid_all(
        begin,
        &[
            "ResidentScoringRunV2::create_unbound_v2",
            "consume_into_resident_scoring_source_v2",
        ],
    );

    let query = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_query_resident_search_combined_v2(",
    );
    require_in_order(
        query,
        &[
            "runtime_facts_equal_v2",
            "cudaMemGetInfo",
            "calculate_resident_generation_allocation_v2",
            "calculate_resident_scoring_allocation_v2",
            "generation.total_device_bytes + scoring.total_device_bytes",
            "allocator_context_reserve_bytes",
            "free_memory_snapshot_count = 1u",
            "terminal_host_allocation_count = 1u",
        ],
    );
    assert_eq!(
        query.matches("cudaMemGetInfo").count(),
        1,
        "combined admission must own the sole coherent free-memory snapshot",
    );

    let create_scoring = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t create_unbound_resident_scoring_run_v2(",
    );
    require_all(
        create_scoring,
        &["cudaMallocAsync", "partition_allocation_v1"],
    );
    let create_generation = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t create_resident_generation_run_from_import_v1(",
    );
    require_all(
        create_generation,
        &["cudaMallocAsync", "partition_generation_allocation_v1"],
    );

    let calculate_scoring = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t calculate_resident_scoring_allocation_v2(",
    );
    let calculate_generation = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t calculate_resident_generation_allocation_v2(",
    );
    forbid_all(
        calculate_scoring,
        &[
            "cudaMemGetInfo",
            "cudaMalloc",
            "<<<",
            "cudaDeviceSynchronize",
        ],
    );
    forbid_all(
        calculate_generation,
        &[
            "cudaMemGetInfo",
            "cudaMalloc",
            "<<<",
            "cudaDeviceSynchronize",
        ],
    );

    let combined_create = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_resident_search_combined_v2(",
    );
    require_in_order(
        combined_create,
        &[
            "The combined receipt is already sealed here",
            "cudaEventCreateWithFlags",
            "cudaHostAlloc",
            "create_resident_generation_run_from_import_v1",
            "bind_resident_search_terminal_receipt_v2",
            "create_unbound_resident_scoring_run_v2",
        ],
    );
    forbid_all(combined_create, &["cudaMemGetInfo", "<<<"]);

    let bind = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t bind_and_seal_resident_scoring_v2(",
    );
    forbid_all(
        bind,
        &[
            "cudaMemGetInfo",
            "cudaMalloc",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyDeviceToHost",
        ],
    );
}

#[test]
fn d_metric_receipt_is_move_consumed_as_one_full_device_chunk() {
    let population = read_required("src/population.rs");
    let search = read_required("src/resident_search_v2.rs");
    let private_header = read_required("native/resident_search_generation_v2_abi.cuh");
    let public_header = read_required("native/neoethos_gpu_cuda.h");
    let population_cuda = read_required("native/prototype_b_population.cu");

    forbid_all(
        &population,
        &[
            "struct ResidentScoringPopulationSourceV2<'",
            "fn consume_into_resident_scoring_source_v2(",
        ],
    );
    let source_owner = braced_item(
        &population,
        "pub(crate) struct ResidentSearchPopulationCompletionLeaseV2",
    );
    require_all(
        source_owner,
        &[
            "session: Option<PopulationSession>",
            "receipt: Box<RawResidentPopulationMetricsHandleV1>",
            "raw: RawResidentScoringPopulationSourceV2",
            "consumed: bool",
        ],
    );
    let owned_enqueue = braced_item(&population, "fn enqueue_resident_gene_metrics_owned_v2(");
    require_all(
        owned_enqueue,
        &[
            "self",
            "ResidentSearchPopulationCompletionLeaseV2",
            "retained_evaluation_capacity != logical_population_count",
            "enqueue_resident_gene_metrics_v2",
            "Box::new(RawResidentPopulationMetricsHandleV1::default())",
            "export_resident_scoring_source_v2",
            "raw.population_lifetime_owner.is_null()",
            "raw.population_lifetime_owner != self.handle",
            "std::ptr::eq(\n                raw.population_lifetime_owner.cast_const(),\n                raw.receipt_token",
            "session: Some(self)",
        ],
    );
    let source_drop = braced_item(
        &population,
        "impl Drop for ResidentSearchPopulationCompletionLeaseV2",
    );
    require_all(source_drop, &["!self.consumed", "poison_without_reuse_v2"]);
    forbid_all(
        owned_enqueue,
        &[
            "consume_host_metrics_v1",
            "HostPopulationMetricsReceiptV1",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyDeviceToHost",
        ],
    );

    let advance = braced_item(&search, "fn advance_one_full_population_generation_v2(");
    require_in_order(
        advance,
        &[
            "ResidentSearchStateV2::Advancing",
            "enqueue_resident_gene_metrics_owned_v2",
            ".ready",
            ".take()",
            "enqueue_full_population_scored_generation_advance_v2",
            "ResidentSearchStateV2::AdvancePending",
            "ResidentSearchAdvancePendingV2",
        ],
    );
    require_all(
        advance,
        &[
            "source.raw_source_v2()",
            "dependency.as_ref()",
            "std::ptr::eq(pending.dependency_receipt_token, dependency.as_ref())",
            "completion: Some(source)",
            "dependency: Some(dependency)",
            "pending: Some(pending)",
        ],
    );
    forbid_all(
        advance,
        &[
            "consume_host_metrics_v1",
            "read_metrics",
            "read_diagnostics",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyDeviceToHost",
            "self.state = ResidentSearchStateV2::AdvancedOnce",
        ],
    );

    let native_source = braced_item(
        &private_header,
        "struct NeoResidentScoringPopulationSourceV2",
    );
    require_in_order(
        native_source,
        &[
            "const void* receipt_token;",
            "void* population_lifetime_owner;",
            "metric_rows_device;",
        ],
    );
    require_all(
        &private_header,
        &[
            "sizeof(NeoResidentScoringPopulationSourceV2) == 96",
            "alignof(NeoResidentScoringPopulationSourceV2) == 8",
            "population_lifetime_owner) == 40",
            "full_discovery_reserve_bytes) == 88",
        ],
    );
    let raw_source = braced_item(
        &population,
        "pub(crate) struct RawResidentScoringPopulationSourceV2",
    );
    require_in_order(
        raw_source,
        &[
            "receipt_token: *const c_void",
            "population_lifetime_owner: *mut c_void",
            "metric_rows_device: *const NeoPopulationMetricRow",
        ],
    );
    let raw_default = braced_item(
        &population,
        "impl Default for RawResidentScoringPopulationSourceV2",
    );
    require_all(
        raw_default,
        &["population_lifetime_owner: std::ptr::null_mut()"],
    );
    require_all(
        &population,
        &[
            "const _: [(); 96] = [(); std::mem::size_of::<RawResidentScoringPopulationSourceV2>()]",
            "layout!(RawResidentScoringPopulationSourceV2, 96, 8)",
            "RawResidentScoringPopulationSourceV2,\n                population_lifetime_owner",
            "RawResidentScoringPopulationSourceV2,\n                full_discovery_reserve_bytes",
        ],
    );
    let export = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_export_resident_scoring_source_v2(",
    );
    require_in_order(
        export,
        &[
            "source->receipt_token = resident_metrics;",
            "source->population_lifetime_owner = session;",
            "source->metric_rows_device = session->metric_rows;",
        ],
    );
    forbid_all(
        &public_header,
        &[
            "struct NeoResidentScoringPopulationSourceV2 {",
            "population_lifetime_owner",
        ],
    );
}

#[test]
fn e_nonfinite_scoring_fault_precedes_rank_consumers_and_conditional_commit() {
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");

    let score_kernel = braced_item(
        &scoring_cuda,
        "__global__ void score_canonical_metrics_kernel_v1(",
    );
    require_in_order(
        score_kernel,
        &[
            "all_metric_values_finite_v1",
            "atomicExch(device_fault_word",
            "score_prop_firm_ga_fitness_v4",
            "score_risky_ga_fitness_growth_v5",
            "!isfinite(score)",
            "atomicExch(device_fault_word",
        ],
    );

    let wrapper = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t enqueue_full_population_scored_generation_advance_v2(",
    );
    require_in_order(
        wrapper,
        &[
            "bind_and_seal_resident_scoring_v2",
            "export_current_resident_gene_view_v2",
            "cudaStreamWaitEvent",
            "enqueue_resident_generation_offspring_from_scored_rows_v2",
            "publish_one_generation_commit_kernel_v2",
        ],
    );
    let advance = braced_item(
        &generation_cuda,
        "std::int32_t enqueue_resident_generation_offspring_from_scored_rows_v2(",
    );
    require_in_order(
        advance,
        &[
            "promote_scoring_device_seal_kernel_v2",
            "validate_and_import_scored_rows_kernel_v1",
            "launch_device_parent_selection_v1",
            "launch_device_crossover_v1",
            "launch_device_mutation_v1",
            "launch_device_gene_hash_v1",
        ],
    );
    require_all(
        advance,
        &[
            "scoring_device_seal",
            "device_content_fault_device",
            "one_generation_advance_pending_v2",
        ],
    );
    forbid_all(
        wrapper,
        &[
            "rotate_resident_generation_stores_v1",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
        ],
    );
    assert_eq!(
        wrapper.matches("cudaMemcpyDeviceToHost").count(),
        1,
        "enqueue may copy only the compact terminal seal/fault receipt to host",
    );

    for kernel in [
        "__global__ void validate_and_import_scored_rows_kernel_v1(",
        "__global__ void select_rank_weighted_parents_kernel_v1(",
        "__global__ void crossover_resident_genes_kernel_v1(",
        "__global__ void mutate_resident_genes_kernel_v1(",
        "__global__ void verify_exact_chunk_coverage_kernel_v1(",
    ] {
        let body = braced_item(&generation_cuda, kernel);
        require_all(body, &["device_content_fault", "!= 0", "return"]);
    }
    let commit = braced_item(
        &generation_cuda,
        "__global__ void publish_one_generation_commit_kernel_v2(",
    );
    require_in_order(
        commit,
        &[
            "publish.combined_fault != 0u",
            "stop_requested = 1",
            "return",
            "NEO_RESIDENT_SEARCH_TERMINAL_COMMITTED_V2",
            "current_store_index = publish.current_store_index",
            "generation_index = publish.generation_index",
        ],
    );

    require_in_order(
        wrapper,
        &[
            "publish_one_generation_commit_kernel_v2<<<",
            "cudaMemcpyAsync(generation->terminal_host_receipt_v2",
            "cudaMemcpyDeviceToHost",
            "cudaEventRecord(generation->ready_event",
            "generation->one_generation_advance_pending_v2 = true",
        ],
    );
    let terminal = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t try_complete_resident_generation_advance_v2(",
    );
    require_in_order(
        terminal,
        &[
            "cudaEventQuery(run->ready_event)",
            "cudaErrorNotReady",
            "*terminal_copy = *run->terminal_host_receipt_v2",
            "NEO_RESIDENT_SEARCH_TERMINAL_FAULT_V2",
            "NEO_RESIDENT_SEARCH_TERMINAL_COMMITTED_V2",
            "if (!exact_fault_terminal && !exact_committed_terminal)",
            "run->terminal_event_proven_v2 = true",
            "if (exact_fault_terminal)",
            "return NEO_RESIDENT_STATUS_DEVICE_FAULT_V1",
            "rotate_resident_generation_stores_v1(run)",
            "run->current_generation_index = terminal_copy->generation_index",
            "run->terminal_committed_v2 = true",
        ],
    );
    forbid_all(
        terminal,
        &[
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpy",
        ],
    );
}

#[test]
fn f_rank_semantics_are_versioned_score_gene_then_original_ordinal() {
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");

    require_all(
        &scoring_rust,
        &[
            "RESIDENT_RANK_SEMANTICS_V2",
            "score-desc",
            "gene-identity-asc",
            "population-ordinal-asc",
            "stable-lsd",
            "rank_semantics_sha256",
        ],
    );
    let key = braced_item(&scoring_cuda, "ordered_f64_decision_key_v2(");
    require_all(key, &["isfinite", "value == 0.0 ? 0.0 : value"]);

    let rank = braced_item(
        &generation_cuda,
        "std::int32_t launch_device_parent_selection_v1(",
    );
    require_in_order(
        rank,
        &[
            "build_gene_identity_rank_keys_kernel_v1",
            "cub::DeviceRadixSort::SortPairs(",
            "gather_resident_decision_rank_keys_kernel_v1",
            "cub::DeviceRadixSort::SortPairsDescending(",
        ],
    );
    let identity_keys = braced_item(
        &generation_cuda,
        "__global__ void build_gene_identity_rank_keys_kernel_v1(",
    );
    require_all(
        identity_keys,
        &["stable_gene_identity_tie_key_v1", "values[index] = index"],
    );
    require_all(
        &generation_cuda,
        &["The population ordinal is the initial stable value"],
    );
}

#[test]
fn g_new_owners_are_move_only_private_and_fail_closed_after_enqueue() {
    let scoring = read_required("src/resident_scoring_v2.rs");
    let search = read_required("src/resident_search_v2.rs");
    let population = read_required("src/population.rs");

    for forbidden in [
        "impl Clone for ResidentScoringRunV2",
        "impl Copy for ResidentScoringRunV2",
        "impl Default for ResidentScoringRunV2",
        "Serialize for ResidentScoringRunV2",
        "Deserialize for ResidentScoringRunV2",
        "impl Clone for ResidentSearchAdvancePendingV2",
        "impl Copy for ResidentSearchAdvancePendingV2",
        "impl Clone for ResidentSearchPopulationCompletionLeaseV2",
        "impl Copy for ResidentSearchPopulationCompletionLeaseV2",
    ] {
        assert!(
            !scoring.contains(forbidden)
                && !search.contains(forbidden)
                && !population.contains(forbidden),
            "move-only escape {forbidden:?}"
        );
    }
    for (name, source) in [
        ("resident_scoring_v2.rs", scoring.as_str()),
        ("resident_search_v2.rs", search.as_str()),
        ("population.rs", population.as_str()),
    ] {
        for line in source.lines().filter(|line| line.contains("pub fn ")) {
            for forbidden in ["*mut ", "*const ", "c_void", "raw_", "native_handle"] {
                assert!(
                    !line.contains(forbidden),
                    "{name} public API exposes {forbidden:?}: {line}"
                );
            }
        }
    }

    let scoring_drop = braced_item(&scoring, "impl Drop for ResidentScoringRunV2");
    require_all(
        scoring_drop,
        &["release_v2", "ResidentScoringStateV2::Poisoned"],
    );
    let scoring_release = braced_item(&scoring, "fn release_v2(");
    require_all(
        scoring_release,
        &[
            "neoethos_gpu_cuda_population_release_resident_scoring_run_v2",
            "ResidentScoringStateV2::Poisoned",
        ],
    );
    let search_drop = braced_item(&search, "impl Drop for ResidentSearchRunV2");
    require_all(
        search_drop,
        &[
            "ResidentSearchStateV2::Advancing",
            "poison_resident_search_owner_v2",
        ],
    );
    require_all(
        &search,
        &[
            "ResidentSearchStateV2::AdvancePending",
            "ResidentSearchStateV2::AdvancedOnce",
            "one resident generation advance is one-shot",
        ],
    );
    let advance = braced_item(
        &search,
        "pub(crate) fn advance_one_full_population_generation_v2(",
    );
    forbid_all(
        advance,
        &[
            ".as_deref()",
            ".copied()",
            "&dependency,",
            "self.state = ResidentSearchStateV2::AdvancedOnce",
        ],
    );
    require_in_order(
        advance,
        &[
            "if !self.ready_receipt_address_is_stable_v2()",
            "whose receipt address native sealed",
            ".take()",
            "dependency.as_ref()",
            "std::ptr::eq(pending.dependency_receipt_token, dependency.as_ref())",
            "ResidentSearchStateV2::AdvancePending",
            "dependency: Some(dependency)",
        ],
    );

    let pending_owner = braced_item(&search, "pub struct ResidentSearchAdvancePendingV2");
    require_all(
        pending_owner,
        &[
            "run: Option<ResidentSearchRunV2>",
            "completion: Option<ResidentSearchPopulationCompletionLeaseV2>",
            "dependency: Option<Box<RawReadyEventV1>>",
            "pending: Option<Box<RawResidentSearchAdvancePendingReceiptV2>>",
            "consumed: bool",
        ],
    );
    let try_complete = braced_item(&search, "pub fn try_complete_one_generation_v2(");
    require_in_order(
        try_complete,
        &[
            "try_complete_resident_generation_advance_v2",
            "STATUS_NOT_READY_V2",
            "NotReady(self)",
            "STATUS_DEVICE_FAULT_V2",
            "poison_without_reuse_v2",
            "terminal.terminal_status != 1",
            "finish_device_consume_v2",
            "run.ready = Some(committed)",
            "run.terminal_receipt = Some",
            "run.state = ResidentSearchStateV2::AdvancedOnce",
            "run.refresh_current_gene_view_v2()",
        ],
    );
    forbid_all(
        try_complete,
        &[
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "consume_host_metrics_v1",
        ],
    );
    let pending_drop = braced_item(&search, "impl Drop for ResidentSearchAdvancePendingV2");
    require_all(
        pending_drop,
        &[
            "if self.consumed",
            "return",
            "completion.poison_without_reuse_v2()",
            "ResidentSearchStateV2::Poisoned",
        ],
    );
}

#[cfg(feature = "cuda")]
#[test]
fn h_implementation_patch_keeps_all_five_unproven_readiness_facts_false() {
    let readiness = resident_search_v2_production_readiness();
    assert!(!readiness.exact_generation_semantics());
    assert!(!readiness.device_resident_generation_advance());
    assert!(readiness.device_owned_search_control());
    assert!(!readiness.immutable_scenario_admission());
    assert!(!readiness.whole_workspace_preallocated());
    assert!(!readiness.unified_device_fault_authority());
    assert!(readiness.native_bridge_production_sealed());
    assert!(!readiness.terminal_cleanup_lease());
    assert!(!readiness.production_ready());
}

#[test]
fn i_real_rtx_oracle_covers_math_fault_order_and_exactly_one_advance() {
    let lib = read_required("src/lib.rs");
    let device = read_required("src/resident_search_generation_v2_device_tests.rs");

    require_all(
        &lib,
        &[
            "#[cfg(all(test, feature = \"cuda-device-fixtures\"))]",
            "mod resident_search_generation_v2_device_tests;",
        ],
    );
    require_all(
        &device,
        &[
            "fn resident_search_scores_and_advances_exactly_one_generation_on_real_cuda()",
            "NEOETHOS_REQUIRE_GPU",
            "const POPULATION: usize = 8",
            "ResidentScoringObjectiveV2::PropFirmV4",
            "ResidentScoringObjectiveV2::RiskyGrowthV5",
            "score_prop_firm_ga_fitness_v4",
            "score_risky_ga_fitness_growth_v5",
            "SCORING_CPU_ORACLE_TOLERANCE_V2",
            "(-0.0_f64)",
            "f64::NAN",
            "f64::INFINITY",
            "f64::NEG_INFINITY",
            "row.values.iter().all",
            "let identities = [50_u64, 10, 10, 40, 30, 20, 60, 0]",
            "assert_ne!(expected, (0..POPULATION as u64)",
            "population_ordinal",
            "philox4x32_10_reference_v1",
            "checked_philox_counter_mapping_v1",
            "checked_philox_rejection_draw_index_v1",
            "fn cpu_generation(",
            "fn assert_gene_exact(",
            "snapshot.parent_a",
            "snapshot.parent_b",
            "snapshot.selected_survivors",
            "snapshot.sorted_dedup_flags",
            "snapshot.candidate_valid_flags",
            "final_genes",
            "generation_index",
            "store_epoch",
            "let nonfinite_values = [",
            "for (metric_slot, nonfinite) in nonfinite_values.into_iter().enumerate()",
            "receipt.generation_index(), 0",
            "receipt.store_epoch(), 1",
            "set_duplicate_final_gene_content_fixture_v2(0, 1)",
            "duplicate full-gene content must fail closed",
            "OneGenerationAdvanceAlreadyEnqueued",
            "drop(pending)",
            "poisoned_pending_drop_count() > 0",
            "reused_in_flight_session_count(), 0",
            "assert_eq!(snapshot.population_counters.gene_upload_bytes, 0)",
            "assert_eq!(snapshot.population_counters.full_readback_bytes, 0)",
            "intermediate_host_wait_count",
            "intermediate_readback_count",
            "resident-search-v2 terminal objective=",
            "terminal",
        ],
    );
    let test = braced_item(
        &device,
        "fn resident_search_scores_and_advances_exactly_one_generation_on_real_cuda()",
    );
    require_in_order(
        test,
        &[
            "invalid_novelty in [-0.0_f64, 0.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]",
            "ResidentSearchStateV2::AdvancePending",
            "committed_gene_view_summary_v2().generation_index()",
            "0",
            "complete_pending(pending)",
            "ResidentSearchStateV2::AdvancedOnce",
            "assert_full_generation_oracle",
            "OneGenerationAdvanceAlreadyEnqueued",
            "let identities = [50_u64, 10, 10, 40, 30, 20, 60, 0]",
            "let nonfinite_values = [",
            "set_duplicate_final_gene_content_fixture_v2(0, 1)",
            "drop(pending)",
        ],
    );
    assert_under_900_lines("src/resident_search_generation_v2_device_tests.rs", &device);
}

#[test]
fn j_new_native_and_rust_scopes_remain_bounded() {
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let v2_header = read_required("native/resident_search_generation_v2_abi.cuh");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");

    assert_under_900_lines("src/resident_scoring_v2.rs", &scoring_rust);
    assert_under_900_lines("native/resident_search_generation_v2_abi.cuh", &v2_header);
    assert_under_900_lines(
        "native scoring V2 bind scope",
        braced_item(
            &scoring_cuda,
            "extern \"C\" std::int32_t bind_and_seal_resident_scoring_v2(",
        ),
    );
    assert_under_900_lines(
        "native generation V2 advance scope",
        braced_item(
            &generation_cuda,
            "extern \"C\" std::int32_t enqueue_full_population_scored_generation_advance_v2(",
        ),
    );
}

#[test]
fn k_private_representation_bridges_are_single_and_compile_time_ratcheted() {
    let population = read_required("native/prototype_b_population.cu");
    let generation = read_required("native/resident_generation_v1.cu");
    let generation_header = read_required("native/resident_generation_v1_abi.cuh");
    let scoring_header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let search_header = read_required("native/resident_search_generation_v2_abi.cuh");

    require_all(
        &generation_header,
        &[
            "using NeoResidentGenerationMetricRowV1 = ::NeoPopulationMetricRow",
            "sizeof(NeoResidentGenerationMetricRowV1) == 104",
        ],
    );
    require_all(
        &scoring_header,
        &[
            "using NeoResidentScoringNoveltyMetricRowV1 = ::NeoPopulationMetricRow",
            "sizeof(NeoResidentScoringNoveltyMetricRowV1) == 104",
        ],
    );
    require_all(
        &search_header,
        &["const resident_scoring_novelty_v1::NeoResidentScoringNoveltyMetricRowV1*"],
    );
    forbid_all(
        &generation,
        &[
            "reinterpret_cast<const NeoResidentScoringNoveltyMetricRowV1*>",
            "reinterpret_cast<const NeoResidentGenerationMetricRowV1*>",
        ],
    );

    let source_bridge = braced_item(
        &population,
        "neoethos_gpu_cuda_population_export_resident_scoring_source_v2(",
    );
    require_all(
        source_bridge,
        &[
            "static_assert(CHAR_BIT == 8)",
            "sizeof(unsigned long long) == sizeof(std::uint64_t)",
            "alignof(unsigned long long) == alignof(std::uint64_t)",
            "ULLONG_MAX == UINT64_MAX",
            "reinterpret_cast<const std::uint64_t*>",
            "one authority cast lives only at this private ABI bridge",
        ],
    );
    assert_eq!(
        population
            .matches("reinterpret_cast<const std::uint64_t*>(session->scenario_ids)")
            .count(),
        1
    );

    let stream_bridge = braced_item(&population, "read_resident_search_runtime_facts_v2(");
    require_all(
        stream_bridge,
        &[
            "sizeof(cudaStream_t) == sizeof(CUstream)",
            "alignof(cudaStream_t) == alignof(CUstream)",
            "reinterpret_cast<CUstream>(session->stream)",
            "cuStreamGetCtx",
            "cuStreamGetDevice",
            "cuCtxGetId",
            "cuStreamGetId",
        ],
    );
    assert_eq!(
        population
            .matches("reinterpret_cast<CUstream>(session->stream)")
            .count(),
        1,
        "runtime stream representation has one private bridge authority",
    );

    let generation_bridge = braced_item(
        &generation,
        "extern \"C\" std::int32_t enqueue_full_population_scored_generation_advance_v2(",
    );
    require_all(
        generation_bridge,
        &[
            "sizeof(GenerationGeneV1) == sizeof(ScoringGeneV1)",
            "alignof(GenerationGeneV1) == alignof(ScoringGeneV1)",
            "offsetof(GenerationGeneV1, gene_identity)",
            "offsetof(GenerationGeneV1, stop_vol_multiplier)",
            "offsetof(GenerationGeneV1, reserved)",
            "reinterpret_cast<const ScoringGeneV1*>",
            "single representation bridge here",
        ],
    );
    assert_eq!(
        generation
            .matches("reinterpret_cast<const ScoringGeneV1*>")
            .count(),
        1
    );
}

#[test]
fn l_runtime_identity_is_cuda_authoritative_and_bound_through_scoring() {
    let population = read_required("native/prototype_b_population.cu");
    let generation = read_required("native/resident_generation_v1.cu");
    let scoring = read_required("native/resident_scoring_novelty_v1.cu");
    let scoring_rust = read_required("src/resident_scoring_v2.rs");

    let facts = braced_item(&population, "read_resident_search_runtime_facts_v2(");
    require_all(
        facts,
        &[
            "cuCtxGetCurrent",
            "cuStreamGetCtx",
            "cuStreamGetDevice",
            "cuCtxGetId",
            "cuStreamGetId",
            "cudaDeviceGetMemPool",
            "cudaDeviceGetDefaultMemPool",
            "cudaMemPoolAttrReservedMemCurrent",
            "cudaMemPoolAttrUsedMemCurrent",
            "active_pool != default_pool",
            "device_uuid",
            "run_admission_ordinal",
            "allocator_context_reserve_bytes",
            "run_stream_process_token",
        ],
    );
    let rust_hashes = braced_item(&scoring_rust, "fn runtime_identity_hashes_v2(");
    require_all(
        rust_hashes,
        &[
            "selected_cuda_ordinal",
            "device_uuid",
            "compute_capability_major",
            "compute_capability_minor",
            "run_admission_ordinal",
            "primary_context_id",
            "run_stream_id",
            "pool_location_type",
            "pool_location_id",
            "pool_allocation_type",
            "pool_handle_types",
            "run_stream_process_token",
        ],
    );

    let generation_owner = braced_item(&generation, "struct NeoResidentGenerationRunV1");
    require_all(
        generation_owner,
        &[
            "cuda_device_identity_sha256[32]",
            "primary_context_identity_sha256[32]",
            "run_stream_identity_sha256[32]",
        ],
    );
    let create = braced_item(
        &generation,
        "extern \"C\" std::int32_t create_resident_generation_run_from_import_v1(",
    );
    require_all(
        create,
        &[
            "created->cuda_device_identity_sha256",
            "import->cuda_device_identity_sha256",
            "created->primary_context_identity_sha256",
            "import->primary_context_identity_sha256",
            "created->run_stream_identity_sha256",
            "import->run_stream_identity_sha256",
        ],
    );
    let enqueue = braced_item(
        &generation,
        "extern \"C\" std::int32_t enqueue_full_population_scored_generation_advance_v2(",
    );
    require_all(
        enqueue,
        &[
            "generation->cuda_device_identity_sha256",
            "generation->primary_context_identity_sha256",
            "generation->run_stream_identity_sha256",
        ],
    );
    forbid_all(
        enqueue,
        &[
            "import.cuda_device_identity_sha256,\n              generation->plan.run_identity_sha256",
            "import.primary_context_identity_sha256,\n              generation->plan.plan_identity_sha256",
            "import.run_stream_identity_sha256,\n              generation->plan.generation_semantics_sha256",
        ],
    );

    let bind = braced_item(
        &scoring,
        "extern \"C\" std::int32_t bind_and_seal_resident_scoring_v2(",
    );
    require_all(
        bind,
        &[
            "population->cuda_device_identity_sha256",
            "run->plan.cuda_device_identity_sha256",
            "population->primary_context_identity_sha256",
            "run->plan.primary_context_identity_sha256",
            "population->run_stream_identity_sha256",
            "run->plan.run_stream_identity_sha256",
        ],
    );
}

#[test]
fn m_new_reserve_and_stream_token_fields_have_full_cross_language_abi_ratchets() {
    let population_rust = read_required("src/population.rs");
    let layout_cpp = read_required("native/layout_asserts.cpp");
    let public_header = read_required("native/neoethos_gpu_cuda.h");
    let population_cuda = read_required("native/prototype_b_population.cu");

    require_all(
        &population_rust,
        &[
            "layout!(RawResidentFeatureStoreBindV3, 256, 8)",
            "offset_of!(\n                    RawResidentFeatureStoreBindV3,\n                    allocator_context_reserve_bytes",
            "offset_of!(RawResidentFeatureStoreBindV3, run_stream_process_token_v3)",
            "hasher.update(resident.allocator_context_reserve_bytes.to_le_bytes())",
            "hasher.update(resident.run_stream_process_token_v3)",
        ],
    );
    for cpp in [&layout_cpp, &public_header] {
        require_all(
            cpp,
            &[
                "sizeof(NeoPopulationResidentFeatureStoreV3) == 256",
                "alignof(NeoPopulationResidentFeatureStoreV3) == 8",
                "allocator_context_reserve_bytes) == 216",
                "run_stream_process_token_v3) == 224",
                "run_stream_process_token_v3) + 32 ==",
                "sizeof(NeoPopulationResidentFeatureStoreV3)",
            ],
        );
    }
    require_all(
        &layout_cpp,
        &[
            "allocator_context_reserve_bytes) ==\n              offsetof(NeoPopulationResidentFeatureStoreV3,\n                       canonical_content_merkle) + 32",
            "run_stream_process_token_v3) ==\n              offsetof(NeoPopulationResidentFeatureStoreV3,\n                       allocator_context_reserve_bytes) + sizeof(std::uint64_t)",
        ],
    );
    let bind = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_bind_resident_feature_store_v3(",
    );
    require_all(
        bind,
        &[
            "resident->allocator_context_reserve_bytes == 0ull",
            "resident->run_stream_process_token_v3",
            "session->allocator_context_reserve_bytes_v3 =",
            "session->run_stream_process_token_v3",
        ],
    );
}

#[test]
fn n_full_discovery_reserve_is_preserved_from_admission_through_scoring_bind() {
    let population_rust = read_required("src/population.rs");
    let search_rust = read_required("src/resident_search_v2.rs");
    let population_cuda = read_required("native/prototype_b_population.cu");
    let generation_cuda = read_required("native/resident_generation_v1.cu");

    let export = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_export_resident_scoring_source_v2(",
    );
    require_all(
        export,
        &[
            "session->allocator_context_reserve_bytes_v3 == 0ull",
            "source->full_discovery_reserve_bytes =\n      session->allocator_context_reserve_bytes_v3",
        ],
    );
    forbid_all(export, &["source->full_discovery_reserve_bytes = 0"]);

    let owned = braced_item(
        &population_rust,
        "pub(crate) fn enqueue_resident_gene_metrics_owned_v2(",
    );
    require_all(
        owned,
        &[
            "expected_full_discovery_reserve_bytes: u64",
            "expected_full_discovery_reserve_bytes == 0",
            "raw.full_discovery_reserve_bytes != expected_full_discovery_reserve_bytes",
        ],
    );
    let advance = braced_item(
        &search_rust,
        "pub(crate) fn advance_one_full_population_generation_v2(",
    );
    require_all(
        advance,
        &[
            ".admission",
            "full_discovery_reserve_bytes",
            "enqueue_resident_gene_metrics_owned_v2",
        ],
    );
    let native_advance = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t enqueue_full_population_scored_generation_advance_v2(",
    );
    require_all(
        native_advance,
        &[
            "population->full_discovery_reserve_bytes == 0ull",
            "population->full_discovery_reserve_bytes !=\n          generation->allocation.full_discovery_reserve_bytes",
            "import.full_discovery_reserve_bytes =\n      population->full_discovery_reserve_bytes",
        ],
    );
}

#[test]
fn o_not_ready_poll_never_writes_the_inflight_pinned_d2h_destination() {
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let run = braced_item(&generation_cuda, "struct NeoResidentGenerationRunV1");
    require_all(
        run,
        &[
            "completion_event_query_count_v2",
            "terminal_event_proven_v2",
        ],
    );
    let complete = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t try_complete_resident_generation_advance_v2(",
    );
    require_in_order(
        complete,
        &[
            "cudaEventQuery(run->ready_event)",
            "++run->completion_event_query_count_v2",
            "cudaErrorNotReady",
            "return resident_generation_v2::NEO_RESIDENT_SEARCH_NOT_READY_V2",
            "*terminal_copy = *run->terminal_host_receipt_v2",
            "terminal_copy->completion_event_query_count =\n      run->completion_event_query_count_v2",
            "const bool exact_terminal",
            "const bool exact_fault_terminal",
            "const bool exact_committed_terminal",
            "if (!exact_fault_terminal && !exact_committed_terminal)",
            "run->terminal_event_proven_v2 = true",
        ],
    );
    forbid_all(
        complete,
        &[
            "++run->terminal_host_receipt_v2->completion_event_query_count",
            "run->terminal_host_receipt_v2->completion_event_query_count =",
        ],
    );
}

#[test]
fn p_every_unconditional_cub_input_has_a_total_deterministic_fault_producer() {
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let generation_cuda = read_required("native/resident_generation_v1.cu");

    require_all(
        &scoring_rust,
        &["defined-sentinel-cub-inputs", "fault-gated-semantic-commit"],
    );
    forbid_all(&scoring_rust, &["nonfinite-fault-before-consumer"]);

    let hash = braced_item(&generation_cuda, "__global__ void gene_hash_kernel_v1(");
    require_all(
        hash,
        &[
            "candidate >= plan.logical_population_count",
            "values[candidate] = candidate",
            "hashes[candidate] = RESIDENT_CUB_FAULT_SENTINEL_KEY_V2",
        ],
    );
    let dedup = braced_item(
        &generation_cuda,
        "__global__ void verify_sorted_gene_dedup_kernel_v1(",
    );
    require_all(
        dedup,
        &[
            "position >= plan.logical_population_count",
            "sorted_unique_flags[position] = 0",
            "candidate_valid_flags[position] = 0",
        ],
    );
    let identity = braced_item(
        &generation_cuda,
        "__global__ void build_gene_identity_rank_keys_kernel_v1(",
    );
    require_all(
        identity,
        &[
            "index >= count",
            "keys[index]",
            "RESIDENT_CUB_FAULT_SENTINEL_KEY_V2",
            "values[index] = index",
        ],
    );
    let gather = braced_item(
        &generation_cuda,
        "__global__ void gather_resident_decision_rank_keys_kernel_v1(",
    );
    require_all(
        gather,
        &[
            "position >= count",
            "keys[position]",
            "RESIDENT_CUB_FAULT_SENTINEL_KEY_V2",
            "values[position] = position",
        ],
    );
    let survivors = braced_item(
        &generation_cuda,
        "__global__ void select_rank_weighted_survivors_kernel_v1(",
    );
    require_all(
        survivors,
        &[
            "selected < plan.survivor_count",
            "selected_rank_indices[selected] = selected",
        ],
    );

    let rank = braced_item(
        &generation_cuda,
        "std::int32_t launch_device_parent_selection_v1(",
    );
    require_in_order(
        rank,
        &[
            "build_gene_identity_rank_keys_kernel_v1",
            "cub::DeviceRadixSort::SortPairs(",
            "gather_resident_decision_rank_keys_kernel_v1",
            "cub::DeviceRadixSort::SortPairsDescending(",
            "select_rank_weighted_survivors_kernel_v1",
            "cub::DeviceRadixSort::SortKeys(",
        ],
    );
    let gene_hash = braced_item(&generation_cuda, "std::int32_t launch_device_gene_hash_v1(");
    require_in_order(
        gene_hash,
        &[
            "gene_hash_kernel_v1",
            "cub::DeviceRadixSort::SortPairs(",
            "cub::DeviceRunLengthEncode::Encode(",
            "verify_sorted_gene_dedup_kernel_v1",
            "cub::DeviceSelect::Flagged(",
        ],
    );
}

#[test]
fn q_combined_create_is_transactional_and_unwinds_every_known_stage() {
    let population_cuda = read_required("native/prototype_b_population.cu");
    let combined = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_resident_search_combined_v2(",
    );
    require_all(
        combined,
        &[
            "created_generation",
            "created_scoring",
            "created_terminal_host_receipt",
            "created_generation_ready_event",
            "created_scoring_ready_event",
            "unwind_combined_create",
            "enqueue_resident_scoring_release_v2",
            "enqueue_resident_generation_release_v1",
            "cudaFreeHost",
            "cudaEventDestroy",
            "session->resident_generation_run_v2 = created_generation",
            "session->resident_scoring_run_v2 = created_scoring",
            "*generation = created_generation",
            "*scoring = created_scoring",
        ],
    );
    require_in_order(
        combined,
        &[
            "created_generation_ready_event",
            "created_scoring_ready_event",
            "created_terminal_host_receipt",
            "created_generation",
            "created_scoring",
            "session->resident_generation_run_v2 = created_generation",
        ],
    );
    forbid_all(
        combined,
        &[
            "session->resident_generation_run_v2 = *generation",
            "session->resident_scoring_run_v2 = *scoring",
        ],
    );
}

#[test]
fn r_event_ready_fault_consumes_and_destroys_without_publishing_generation_one() {
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let population_rust = read_required("src/population.rs");
    let search_rust = read_required("src/resident_search_v2.rs");
    let device = read_required("src/resident_search_generation_v2_device_tests.rs");

    let release = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t enqueue_resident_generation_release_v1(",
    );
    require_all(
        release,
        &[
            "run->one_generation_advance_pending_v2",
            "run->poisoned_v2 && !run->terminal_event_proven_v2",
            "cudaFreeAsync",
        ],
    );
    let finish = braced_item(&population_rust, "pub(crate) fn finish_device_consume_v2(");
    require_in_order(
        finish,
        &[
            "neoethos_gpu_cuda_population_finish_resident_scoring_source_v2",
            "StrictResidentSessionStateV1::StrictIdle",
            "authorize_resident_session_destroy_v3",
        ],
    );
    let terminal_cleanup = braced_item(
        &search_rust,
        "fn release_terminal_proven_fault_resources_v2(",
    );
    require_in_order(
        terminal_cleanup,
        &[
            "scoring.release_v2()",
            "neoethos_gpu_cuda_population_release_resident_generation_run_v2",
            "self.generation = None",
        ],
    );
    let complete = braced_item(&search_rust, "pub fn try_complete_one_generation_v2(");
    require_in_order(
        complete,
        &[
            "STATUS_DEVICE_FAULT_V2",
            "finish_device_consume_v2",
            "release_terminal_proven_fault_resources_v2",
            "DeviceTerminalFault",
        ],
    );
    require_all(
        &device,
        &[
            "terminal_fault_cleanup_count",
            "terminal_session_destroy_count",
            "receipt.generation_index(), 0",
            "receipt.store_epoch(), 1",
        ],
    );
}

#[test]
fn s_start_failure_and_terminal_cleanup_retain_exact_lifecycle_authority() {
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let population_cuda = read_required("native/prototype_b_population.cu");
    let generation_rust = read_required("src/resident_generation_v1.rs");
    let population_rust = read_required("src/population.rs");
    let search_rust = read_required("src/resident_search_v2.rs");
    let feature_rust = read_required("src/resident_feature_store_v3.rs");
    let feature_device = read_required("src/resident_population_session_v3_device_tests.rs");

    let generation_create = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t create_resident_generation_run_from_import_v1(",
    );
    require_all(
        generation_create,
        &[
            "cudaFreeAsync",
            "retire_generation_allocation_identity_v2(created)",
            "NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
        ],
    );
    assert_eq!(
        generation_create.matches("*run = created").count(),
        1,
        "generation creator may publish its owner only on the success path"
    );
    let scoring_create = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t create_unbound_resident_scoring_run_v2(",
    );
    require_all(
        scoring_create,
        &[
            "cudaFreeAsync",
            "retire_scoring_allocation_identity_v2(created)",
            "NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
        ],
    );
    assert_eq!(
        scoring_create.matches("*run = created").count(),
        1,
        "scoring creator may publish its owner only on the success path"
    );
    let generation_bind = braced_item(
        &generation_rust,
        "pub(crate) fn bind_population_session_import_v1(",
    );
    require_in_order(
        generation_bind,
        &["if status != STATUS_OK_V1", "native_error_v1("],
    );
    forbid_all(
        generation_bind,
        &["ffi_enqueue_resident_generation_release_v1(native)"],
    );

    let combined = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_resident_search_combined_v2(",
    );
    require_all(
        combined,
        &[
            "PopulationStrictExecutionStateV1::StrictIdle",
            "created_generation != nullptr",
            "created_scoring != nullptr",
            "stream_state_unknown",
            "*generation = nullptr",
            "*scoring = nullptr",
            "PopulationStrictExecutionStateV1::Poisoned",
        ],
    );

    let begin = braced_item(&search_rust, "fn begin_resident_search_sealed_v2(");
    require_in_order(
        begin,
        &[
            "self.authorize_resident_session_destroy_v3()",
            "if smc_weights",
            "neoethos_gpu_cuda_population_create_resident_search_combined_v2",
            "self.arm_resident_session_leak_only_v3()",
            "NonNull::new(generation)",
        ],
    );
    let v3_start = braced_item(
        &feature_rust,
        "pub(crate) fn consume_into_resident_search_run_v2(",
    );
    require_all(
        v3_start,
        &[
            "match population_session.begin_resident_search_from_plan_v2",
            "resident_import.record_consumer_completion()",
            "ResidentFeatureStoreSearchStartErrorV2::Search",
            "cleanup_lease",
            "ResidentFeatureStoreSearchStartErrorV2::CleanupEvent",
        ],
    );
    forbid_all(v3_start, &["smc_gate_disabled,\n        )?"]);
    require_all(
        &feature_device,
        &[
            "resident_store_v3_search_start_failure_returns_event_owned_recovery_carrier",
            "[0.0; SMC_SLOTS]",
            "ResidentFeatureStoreSearchStartErrorV2::Search",
            ".into_cleanup_lease()",
            "while !lease.completion_is_ready()?",
            "lease.rows(), ROWS",
        ],
    );

    let generation_release = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_release_resident_generation_run_v2(",
    );
    require_in_order(
        generation_release,
        &[
            "detach_resident_search_terminal_receipt_v2",
            "cudaFreeHost",
            "enqueue_resident_generation_release_v1(run)",
            "session->resident_generation_run_v2 = nullptr",
        ],
    );
    let checked_destroy = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_destroy_terminal_checked_v2(",
    );
    require_in_order(
        checked_destroy,
        &[
            "session->resident_generation_run_v2 != nullptr",
            "session->resident_scoring_run_v2 != nullptr",
            "session->release_terminal_checked_v2()",
            "delete session",
            "return NEO_POPULATION_STATUS_OK",
        ],
    );
    let rust_destroy = braced_item(
        &population_rust,
        "pub(crate) fn destroy_terminal_proven_resident_search_v2(",
    );
    require_in_order(
        rust_destroy,
        &[
            "neoethos_gpu_cuda_population_destroy_terminal_checked_v2(self.handle)",
            "if status != STATUS_OK",
            "self.handle = std::ptr::null_mut()",
            "TERMINAL_SEARCH_SESSION_DESTROY_COUNT_V2.fetch_add",
        ],
    );
}

#[test]
fn t_stream_ordered_allocation_identity_is_one_way_retired_on_every_outcome() {
    let generation_header = read_required("native/resident_generation_v1_abi.cuh");
    let scoring_header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let population_header = read_required("native/neoethos_gpu_cuda.h");
    let generation_cuda = read_required("native/resident_generation_v1.cu");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let population_cuda = read_required("native/prototype_b_population.cu");
    let population_rust = read_required("src/population.rs");
    let generation_rust = read_required("src/resident_generation_v1.rs");
    let scoring_rust = read_required("src/resident_scoring_v2.rs");
    let search_rust = read_required("src/resident_search_v2.rs");

    require_all(
        &generation_header,
        &[
            "NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 = -13",
            "NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2 = -14",
        ],
    );
    require_all(
        &scoring_header,
        &[
            "NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 = -9",
            "NEO_SCORING_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2 = -10",
        ],
    );
    require_all(
        &population_header,
        &[
            "NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN (-48)",
            "NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN (-49)",
        ],
    );

    // CUDA's stream-ordered allocator contract is deliberately repeated at
    // both ownership authorities: a non-success may report an earlier async
    // fault even though the free was inserted, so status cannot restore the
    // pointer to the owner or authorize a retry/query.
    for source in [&generation_cuda, &scoring_cuda] {
        require_all(
            source,
            &[
                "cudaFreeAsync may surface a",
                "prior asynchronous error",
                "retired before",
                "invocation and is never queried, accessed, or freed again",
                "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html",
            ],
        );
    }

    require_all(
        &generation_cuda,
        &[
            "allocation_free_issued_v2",
            "free_outcome_unknown_deliberate_leak_v2",
            "retire_generation_allocation_identity_v2",
            "run->allocation_base = nullptr",
            "run->gene_scalars_device = nullptr",
            "run->offspring_gene_scalars_device = nullptr",
            "run->metric_rows_device = nullptr",
            "run->rank_keys_a_device = nullptr",
            "run->gene_hashes_a_device = nullptr",
            "run->device_seal_v2 = nullptr",
            "run->resident_control_device_v2 = nullptr",
            "run->cub_scratch_device = nullptr",
            "run->terminal_device_receipt_v2 = nullptr",
        ],
    );
    assert_eq!(
        generation_cuda.matches("cudaFreeAsync(").count(),
        3,
        "generation must retain exactly the two creator-unwind frees and one release free"
    );
    let generation_create = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t create_resident_generation_run_from_import_v1(",
    );
    require_in_order(
        generation_create,
        &[
            "void* attempted_allocation = nullptr",
            "cudaMallocAsync(",
            "&attempted_allocation",
            "if (status != cudaSuccess)",
            "NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2",
            "created->allocation_base = attempted_allocation",
            "retire_generation_allocation_identity_v2(created)",
            "cudaFreeAsync(",
            "retire_generation_allocation_identity_v2(created)",
            "cudaFreeAsync(",
        ],
    );
    forbid_all(
        generation_create,
        &["cudaFreeAsync(created->allocation_base"],
    );
    assert_eq!(generation_create.matches("*run = created").count(), 1);
    let generation_release = braced_item(
        &generation_cuda,
        "extern \"C\" std::int32_t enqueue_resident_generation_release_v1(",
    );
    require_in_order(
        generation_release,
        &[
            "run->allocation_free_issued_v2",
            "retire_generation_allocation_identity_v2(run)",
            "cudaFreeAsync(",
            "free_outcome_unknown_deliberate_leak_v2 = true",
            "NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
        ],
    );
    forbid_all(generation_release, &["cudaFreeAsync(run->allocation_base"]);

    require_all(
        &scoring_cuda,
        &[
            "allocation_free_issued_v2",
            "free_outcome_unknown_deliberate_leak_v2",
            "retire_scoring_allocation_identity_v2",
            "run->allocation_base = nullptr",
            "run->set_words_device = nullptr",
            "run->fitness_scores_device = nullptr",
            "run->novelty_scores_device = nullptr",
            "run->decision_keys_device = nullptr",
            "run->cub_scratch_device = nullptr",
            "run->device_fault_word = nullptr",
            "run->device_seal = nullptr",
        ],
    );
    assert_eq!(
        scoring_cuda.matches("cudaFreeAsync(").count(),
        5,
        "scoring must retain exactly two V1-create, one V2-create, one Slice2-create, and one release free"
    );
    for creator in [
        "extern \"C\" std::int32_t create_resident_scoring_novelty_run_v1(",
        "extern \"C\" std::int32_t create_unbound_resident_scoring_run_v2(",
        "std::int32_t create_slice2_combined_scoring_archive_run_v2(",
    ] {
        let body = braced_item(&scoring_cuda, creator);
        require_all(
            body,
            &[
                "void* attempted_allocation = nullptr",
                "cudaMallocAsync(",
                "&attempted_allocation",
                "NEO_SCORING_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2",
                "created->allocation_base = attempted_allocation",
                "retire_scoring_allocation_identity_v2(created)",
                "cudaFreeAsync(",
            ],
        );
        forbid_all(body, &["cudaFreeAsync(created->allocation_base"]);
        assert_eq!(body.matches("*run = created").count(), 1);
    }
    let scoring_release = braced_item(
        &scoring_cuda,
        "extern \"C\" std::int32_t enqueue_resident_scoring_novelty_release_v1(",
    );
    require_in_order(
        scoring_release,
        &[
            "run->allocation_free_issued_v2",
            "retire_scoring_allocation_identity_v2(run)",
            "cudaFreeAsync(",
            "free_outcome_unknown_deliberate_leak_v2 = true",
            "NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
        ],
    );
    forbid_all(scoring_release, &["cudaFreeAsync(run->allocation_base"]);

    let combined = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_resident_search_combined_v2(",
    );
    require_all(
        combined,
        &[
            "NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN",
            "NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN",
            "attempted_generation_ready_event",
            "attempted_scoring_ready_event",
            "attempted_terminal_host_receipt",
        ],
    );
    forbid_all(
        combined,
        &[
            "status == NEO_RESIDENT_STATUS_CUDA_ERROR_V1 ||\n                    created_generation != nullptr",
            "status, created_scoring != nullptr",
        ],
    );
    let standalone_generation = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_resident_generation_run_v2(",
    );
    require_in_order(
        standalone_generation,
        &[
            "if (status != NEO_RESIDENT_STATUS_OK_V1)",
            "if (*run != nullptr)",
            "NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
            "NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2",
            "PopulationStrictExecutionStateV1::Poisoned",
            "return population_status_from_generation_v2(status)",
        ],
    );
    forbid_all(
        standalone_generation,
        &["enqueue_resident_generation_release_v1(*run)"],
    );
    let standalone_scoring = braced_item(
        &population_cuda,
        "neoethos_gpu_cuda_population_create_unbound_resident_scoring_run_v2(",
    );
    require_in_order(
        standalone_scoring,
        &[
            "if (status != NEO_SCORING_STATUS_OK_V1)",
            "if (*run != nullptr)",
            "NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2",
            "NEO_SCORING_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2",
            "PopulationStrictExecutionStateV1::Poisoned",
            "return population_status_from_scoring_v2(status)",
        ],
    );
    forbid_all(
        standalone_scoring,
        &["enqueue_resident_scoring_release_v2(*run)"],
    );
    for source in [&population_rust, &generation_rust, &scoring_rust] {
        require_all(
            source,
            &[
                "AsyncFreeOutcomeUnknownDeliberateLeak",
                "AsyncAllocationOutcomeUnknownDeliberateLeak",
            ],
        );
    }
    let generation_bind = braced_item(
        &generation_rust,
        "pub(crate) fn bind_population_session_import_v1(",
    );
    forbid_all(
        generation_bind,
        &[
            "ffi_enqueue_resident_generation_release_v1(native)",
            "enqueue_resident_generation_release_v1_after_failed_create",
        ],
    );
    let search_begin = braced_item(&search_rust, "fn begin_resident_search_sealed_v2(");
    require_in_order(
        search_begin,
        &[
            "neoethos_gpu_cuda_population_create_resident_search_combined_v2(",
            "if status != STATUS_OK",
            "return Err(native_error(",
            "self.arm_resident_session_leak_only_v3()",
            "NonNull::new(generation)",
        ],
    );
    let search_native_error = braced_item(
        &search_rust,
        "fn native_error(operation: &'static str, status: i32)",
    );
    require_all(
        search_native_error,
        &[
            "STATUS_ASYNC_FREE_OUTCOME_UNKNOWN",
            "STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN",
            "CudaPopulationError::native",
        ],
    );
}

#[test]
fn u_slice2_scoring_owns_one_combined_arena_and_exports_only_native_same_stream_authority() {
    let public_header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let internal_header = read_required("native/resident_scoring_novelty_v2_internal.cuh");
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");

    require_all(
        &internal_header,
        &[
            "namespace neoethos::resident_scoring_novelty_v2_internal",
            "struct ResidentScoringArenaAccessV2",
            "cudaStream_t admitted_run_stream;",
            "void* allocation_base;",
            "std::uint64_t allocation_bytes;",
            "struct ResidentScoringFiniteObjectiveRowsV2",
            "NeoResidentScoringNoveltyRunV1* scoring_owner;",
            "const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;",
            "const std::uint64_t* expected_scenario_ids_device;",
            "double* fitness_scores_device;",
            "std::uint64_t* decision_keys_device;",
            "const NeoResidentScoringNoveltyDeviceSealV1* device_seal;",
            "std::uint8_t metric_semantics_sha256[32];",
            "std::uint8_t scoring_semantics_sha256[32];",
            "std::uint8_t novelty_semantics_sha256[32];",
            "std::uint8_t scenario_order_semantics_sha256[32];",
            "std::uint8_t rank_semantics_sha256[32];",
            "std::uint8_t cuda_build_manifest_sha256[32];",
            "std::uint8_t cuda_math_flags_sha256[32];",
            "std::int32_t create_slice2_combined_scoring_archive_run_v2(",
            "std::int32_t borrow_resident_scoring_archive_arena_v2(",
            "std::int32_t enqueue_resident_scoring_finite_objective_v2(",
            "NEO_RESIDENT_SCORING_SLICE2_CONTROL_BYTES_V2 = 64",
            "NEO_RESIDENT_SCORING_SLICE2_ARCHIVE_CONTROL_OFFSET_V2 = 64",
        ],
    );
    let arena_access = braced_item(&internal_header, "struct ResidentScoringArenaAccessV2");
    require_all(
        arena_access,
        &[
            "cudaStream_t admitted_run_stream;",
            "void* allocation_base;",
            "std::uint64_t allocation_bytes;",
            "std::uint64_t same_stream_enqueue_count;",
        ],
    );
    forbid_all(
        &internal_header,
        &[
            "extern \"C\"",
            "cudaMemcpy",
            "cudaEventRecord",
            "cudaEventQuery",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaHostAlloc",
        ],
    );
    for private_name in [
        "ResidentScoringArenaAccessV2",
        "ResidentScoringFiniteObjectiveRowsV2",
        "create_slice2_combined_scoring_archive_run_v2",
        "borrow_resident_scoring_archive_arena_v2",
        "enqueue_resident_scoring_finite_objective_v2",
    ] {
        assert!(
            !public_header.contains(private_name),
            "native-only Slice2 scoring authority escaped through the public ABI as {private_name:?}"
        );
    }

    let create = braced_item(
        &scoring_cuda,
        "std::int32_t create_slice2_combined_scoring_archive_run_v2(",
    );
    require_all(
        create,
        &[
            "validate_slice2_combined_binding_v2",
            "binding->total_device_bytes",
            "cudaMallocAsync(",
            "partition_slice2_combined_arena_v2",
            "created->slice2_combined_arena_v2 = true",
            "created->retained_slice2_binding_v2 = *binding",
            "*run = created",
        ],
    );
    assert_eq!(
        create.matches("cudaMallocAsync(").count(),
        1,
        "the combined ScoringArchiveArena creator must issue exactly one device allocation"
    );
    forbid_all(
        create,
        &[
            "cudaMemGetInfo",
            "cudaHostAlloc",
            "cudaEventCreate",
            "cudaEventRecord",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
        ],
    );

    let partition = braced_item(&scoring_cuda, "bool partition_slice2_combined_arena_v2(");
    require_all(
        partition,
        &[
            "binding.fitness_scores",
            "binding.decision_keys",
            "binding.cub_scratch",
            "binding.current_population_signatures",
            "binding.novelty_scores",
            "binding.archive_control_and_seal",
            "run->fitness_scores_device",
            "run->decision_keys_device",
            "run->cub_scratch_device",
            "run->device_fault_word",
            "run->device_seal",
        ],
    );

    let borrow = braced_item(
        &scoring_cuda,
        "std::int32_t borrow_resident_scoring_archive_arena_v2(",
    );
    require_all(
        borrow,
        &[
            "run->slice2_combined_arena_v2",
            "same_slice2_binding_v2(run->retained_slice2_binding_v2, *binding)",
            "access->admitted_run_stream = run->admitted_run_stream",
            "access->allocation_base = run->allocation_base",
            "access->allocation_bytes = run->allocation.total_device_bytes",
            "access->same_stream_enqueue_count = run->same_stream_enqueue_count",
        ],
    );
    forbid_all(
        borrow,
        &[
            "cudaMalloc",
            "cudaFree",
            "cudaMemGetInfo",
            "cudaMemcpy",
            "cudaEvent",
        ],
    );

    let score = braced_item(
        &scoring_cuda,
        "std::int32_t enqueue_resident_scoring_finite_objective_v2(",
    );
    require_in_order(
        score,
        &[
            "validate_slice2_population_source_v2",
            "cudaMemsetAsync(run->device_fault_word",
            "cudaMemsetAsync(run->device_seal",
            "score_canonical_metrics_kernel_v1<<<",
            "encode_finite_objective_keys_kernel_v2<<<",
            "seal_finite_objective_content_kernel_v2<<<",
            "rows->fitness_scores_device = run->fitness_scores_device",
            "rows->device_seal = run->device_seal",
        ],
    );
    forbid_all(
        score,
        &[
            "candidate_ordered_mean_jaccard_kernel_v1",
            "blend_and_encode_decision_keys_kernel_v1",
            "cudaMemcpy",
            "cudaEventRecord",
            "cudaEventQuery",
            "cudaStreamWaitEvent",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaMalloc",
            "cudaFree",
        ],
    );

    let objective_seal = braced_item(
        &scoring_cuda,
        "__global__ void seal_finite_objective_content_kernel_v2(",
    );
    require_all(
        objective_seal,
        &[
            "const double* fitness_scores",
            "f64_bits_v1(fitness_scores[candidate])",
            "plan.metric_semantics_sha256",
            "plan.scoring_semantics_sha256",
            "plan.cuda_math_flags_sha256",
        ],
    );
    forbid_all(objective_seal, &["decision_keys"]);
}

#[test]
fn u_slice2_combined_cub_scratch_query_covers_reduce_and_all_three_rank_passes() {
    let scoring_cuda = read_required("native/resident_scoring_novelty_v1.cu");
    validate_slice2_scoring_archive_cub_scratch_query(&scoring_cuda)
        .expect("combined arena scratch must cover scoring reductions and archive tuple rank");

    let query = braced_item(
        &scoring_cuda,
        "std::int32_t query_cub_reduce_scratch_bytes_v1(",
    );
    let mut mutants = Vec::new();
    for occurrence in 0..2 {
        mutants.push(replace_nth(
            query,
            "cub::DeviceRadixSort::SortPairs(",
            "cub::DeviceRadixSort::SortKeys(",
            occurrence,
        ));
    }
    mutants.push(replace_nth(
        query,
        "cub::DeviceRadixSort::SortPairsDescending(",
        "cub::DeviceRadixSort::SortPairs(",
        0,
    ));
    for occurrence in 1..4 {
        mutants.push(replace_nth(
            query,
            "maximum = candidate > maximum ? candidate : maximum;",
            "candidate = maximum;",
            occurrence,
        ));
    }
    for mutant in mutants {
        assert!(
            validate_slice2_scoring_archive_cub_scratch_query(&mutant).is_err(),
            "source contract must kill removal of every rank query and its maximum fold"
        );
    }
}
