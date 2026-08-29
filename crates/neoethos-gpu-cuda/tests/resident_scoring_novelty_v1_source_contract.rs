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
            "resident scoring/novelty V1 source is missing {token:?}"
        );
    }
}

#[test]
fn canonical_search_scoring_and_novelty_source_is_pinned_before_the_cuda_port() {
    let named = read_required("../neoethos-search/src/scoring/named.rs");
    let evolution = read_required("../neoethos-search/src/genetic/evolution_math.rs");
    let search = read_required("../neoethos-search/src/genetic/search_engine.rs");

    require_all(
        &named,
        &[
            "pub const SCORING_VERSION_CURRENT: ScoringVersion = ScoringVersion(5);",
            "pub fn ga_fitness(metrics: &[f64; 11]) -> f64",
            "if trades_f < 1.0 {",
            "return -100.0;",
            "let activity = (trades_f / 30.0).clamp(0.0, 1.0);",
            "let hit = monthly_hit.clamp(0.0, 1.0) * 0.45;",
            "let ret = (net / 20_000.0).clamp(-2.0, 2.0) * 0.15;",
            "let daily_dd_pen = max_daily_dd.clamp(0.0, 1.0) * 10.0;",
            "pub fn ga_fitness_growth(metrics: &[f64; 11]) -> f64",
            "let p = win_rate.clamp(0.0, 0.99);",
            "let pf = profit_factor.clamp(0.0, 10.0);",
            "let f = (f_star * 0.5).clamp(0.0, 0.25);",
            "p * (1.0 + rr * f).ln() + (1.0 - p) * (1.0 - f).ln()",
            "growth * 10.0 + edge_gradient",
        ],
    );
    require_all(
        &evolution,
        &[
            "crate::scoring::ga_fitness_growth(m)",
            "crate::scoring::ga_fitness(m)",
        ],
    );
    require_all(
        &search,
        &[
            "if novelty_weight > 0.0 && scored.len() > 1",
            "g.indices.iter().copied().collect()",
            "for (j, sig_j) in index_sets.iter().enumerate()",
            "let intersection = sig_i.intersection(sig_j).count() as f64;",
            "let union = sig_i.union(sig_j).count() as f64;",
            "dist_sum / (n_pop as f64 - 1.0)",
            "let fit_range = (max_fit - min_fit).max(1e-9);",
            "let norm_fit = (scored[i].0 - min_fit) / fit_range;",
            "let norm_nov = novelty_scores[i] / max_nov;",
            "(1.0 - novelty_weight) * norm_fit + novelty_weight * norm_nov",
            ".then_with(|| a.1.cmp(&b.1))",
        ],
    );
}

#[test]
fn rust_authority_is_move_only_build_bound_and_leaks_on_ambiguous_drop() {
    let rust = read_required("src/resident_scoring_novelty_v1.rs");
    let run = section(
        &rust,
        "pub struct ResidentScoringNoveltyDeviceRunV1 {",
        "\n}",
    );
    require_all(
        run,
        &[
            "native: NonNull<NativeResidentScoringNoveltyRunV1>",
            "population_import: Option<ResidentScoringNoveltyPopulationImportV1>",
            "state: ResidentScoringNoveltyRunStateV1",
            "selected_cuda_ordinal: u32",
            "primary_context_identity_sha256: [u8; 32]",
            "run_stream_identity_sha256: [u8; 32]",
            "cuda_build_manifest_sha256: [u8; 32]",
            "cuda_math_flags_sha256: [u8; 32]",
        ],
    );
    assert!(
        !run.contains("pub "),
        "native owner fields must stay private"
    );
    require_all(
        &rust,
        &[
            "#[must_use = \"resident scoring/novelty work must be consumed by the next device stage\"]",
            "enum ResidentScoringNoveltyRunStateV1",
            "StrictIdle",
            "InFlight",
            "Sealed",
            "Poisoned",
            "impl Drop for ResidentScoringNoveltyDeviceRunV1",
            "leak_live_native_scoring_novelty_run_v1(",
        ],
    );
    for forbidden in [
        "impl Clone for ResidentScoringNoveltyDeviceRunV1",
        "impl Default for ResidentScoringNoveltyDeviceRunV1",
        "pub fn from_raw",
        "pub fn raw_",
        "Deserialize",
    ] {
        assert!(
            !rust.contains(forbidden),
            "authority escape via {forbidden:?}"
        );
    }
}

#[test]
fn cuda_checked_feature_word_arithmetic_uses_exact_u64_operand_on_lp64() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");

    require_all(
        &cuda,
        &["checked_add_v1(feature_count, std::uint64_t{63}, &expanded)"],
    );
    for forbidden in [
        "checked_add_v1(feature_count, 63ull, &expanded)",
        "static_cast<std::uint64_t>(63ull)",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "LP64 checked arithmetic still uses a mismatched or masking operand {forbidden:?}"
        );
    }
}

#[test]
fn private_abi_binds_exact_metric_gene_scenario_stream_and_preowned_events() {
    let header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let import = section(
        &header,
        "struct NeoResidentScoringNoveltyPopulationImportV1 {",
        "\n};",
    );
    let plan = section(&header, "struct NeoResidentScoringNoveltyPlanV1 {", "\n};");
    require_all(
        &header,
        &[
            "struct NeoResidentScoringNoveltyMetricRowV1",
            "std::uint64_t candidate_id;",
            "std::uint64_t scenario_id;",
            "double values[11];",
            "static_assert(sizeof(NeoResidentScoringNoveltyMetricRowV1) == 104",
        ],
    );
    require_all(
        import,
        &[
            "cudaStream_t admitted_run_stream;",
            "cudaEvent_t metrics_ready_event;",
            "cudaEvent_t scoring_novelty_ready_event;",
            "const NeoResidentScoringNoveltyMetricRowV1* metric_rows_device;",
            "const NeoResidentScoringNoveltyGeneScalarV1* gene_scalars_device;",
            "const std::uint64_t* gene_indices_device;",
            "const std::uint64_t* expected_scenario_ids_device;",
            "std::uint64_t logical_population_count;",
            "std::uint64_t feature_count;",
            "std::uint32_t max_terms_per_gene;",
            "std::uint8_t metric_semantics_sha256[32];",
            "std::uint8_t gene_schema_sha256[32];",
            "std::uint8_t scenario_order_semantics_sha256[32];",
            "std::uint8_t cuda_build_manifest_sha256[32];",
        ],
    );
    require_all(
        plan,
        &[
            "std::uint8_t cuda_device_identity_sha256[32];",
            "std::uint8_t primary_context_identity_sha256[32];",
            "std::uint8_t run_stream_identity_sha256[32];",
        ],
    );
}

#[test]
fn plan_admits_only_the_two_canonical_objectives_and_checked_novelty_weight() {
    let rust = read_required("src/resident_scoring_novelty_v1.rs");
    let header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    require_all(
        &rust,
        &[
            "pub const RESIDENT_SCORING_NOVELTY_SEMANTICS_V1",
            "CanonicalPropFirmGaFitnessV4 = 1",
            "CanonicalRiskyGaFitnessGrowthV5 = 2",
            "const SCORING_VERSION_V1: u32 = 5;",
            "novelty_weight_bits: u64",
            "let novelty_weight = f64::from_bits(input.novelty_weight_bits);",
            "!novelty_weight.is_finite()",
            "!(0.0..=1.0).contains(&novelty_weight)",
            "identity_is_zero_v1(&input.scoring_semantics_sha256)",
            "identity_is_zero_v1(&input.novelty_semantics_sha256)",
            "identity_is_zero_v1(&input.rank_semantics_sha256)",
            "identity_is_zero_v1(&input.cuda_build_manifest_sha256)",
            "identity_is_zero_v1(&input.cuda_math_flags_sha256)",
        ],
    );
    require_all(
        &header,
        &[
            "NEO_RESIDENT_SCORING_PROPFIRM_V4 = 1",
            "NEO_RESIDENT_SCORING_RISKY_GROWTH_V5 = 2",
            "std::uint32_t scoring_version;",
            "std::uint64_t novelty_weight_bits;",
            "std::uint8_t cuda_math_flags_sha256[32];",
            "std::uint8_t cuda_device_identity_sha256[32];",
            "std::uint8_t primary_context_identity_sha256[32];",
            "std::uint8_t run_stream_identity_sha256[32];",
        ],
    );
}

#[test]
fn cuda_ports_current_ga_fitness_v4_branch_constants_and_guards() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let score = section(&cuda, "score_prop_firm_ga_fitness_v4(", "\n}");
    require_all(
        score,
        &[
            "metrics[0]",
            "metrics[1]",
            "metrics[3]",
            "metrics[4]",
            "metrics[5]",
            "metrics[7]",
            "metrics[8]",
            "metrics[9]",
            "metrics[10]",
            "trades < 1.0",
            "return -100.0;",
            "clamp_f64_v1(trades / 30.0, 0.0, 1.0)",
            "0.3 + 0.7 * activity",
            "clamp_f64_v1(monthly_hit, 0.0, 1.0) * 0.45",
            "clamp_f64_v1(net / 20000.0, -2.0, 2.0) * 0.15",
            "ga_pf_component_v1(profit_factor)",
            "profit_factor >= 1.0 ? 0.15 : 0.25",
            "drawdown_penalty_v1(max_drawdown)",
            "clamp_f64_v1(max_daily_drawdown, 0.0, 1.0) * 10.0",
        ],
    );
}

#[test]
fn cuda_ports_current_growth_v5_log_order_and_edge_gradient() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    let score = section(&cuda, "score_risky_ga_fitness_growth_v5(", "\n}");
    require_all(
        score,
        &[
            "metrics[0]",
            "metrics[1]",
            "metrics[4]",
            "metrics[5]",
            "metrics[8]",
            "trades < 1.0",
            "return -100.0;",
            "clamp_f64_v1(win_rate, 0.0, 0.99)",
            "clamp_f64_v1(profit_factor, 0.0, 10.0)",
            "p * (pf - 1.0) / pf",
            "clamp_f64_v1(f_star * 0.5, 0.0, 0.25)",
            "p * log(1.0 + rr * f) + (1.0 - p) * log(1.0 - f)",
            "growth * 10.0 + edge_gradient",
        ],
    );
}

#[test]
fn unsorted_duplicate_terms_become_checked_set_bitmaps_before_jaccard() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &cuda,
        &[
            "checked_feature_word_count_v1(",
            "build_checked_gene_set_bitmap_kernel_v1",
            "term_count > plan.max_terms_per_gene",
            "feature_index >= plan.feature_count",
            "set_words[candidate * feature_word_count + word] |= bit;",
            "__popcll(left & right)",
            "__popcll(left | right)",
            "for (std::uint64_t other = 0; other < plan.logical_population_count; ++other)",
            "dist_sum += 1.0 - static_cast<double>(intersection) /",
            "dist_sum / static_cast<double>(plan.logical_population_count - 1)",
        ],
    );
    for forbidden in ["assume_sorted", "assume_unique", "binary_search"] {
        assert!(
            !cuda.contains(forbidden),
            "set semantics bypass via {forbidden:?}"
        );
    }
}

#[test]
fn global_normalization_and_blend_stay_resident_and_candidate_ordered() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &cuda,
        &[
            "cub::DeviceReduce::Min",
            "cub::DeviceReduce::Max",
            "fit_range = max_fitness - min_fitness",
            "fit_range < 1.0e-9 ? 1.0e-9 : fit_range",
            "max_novelty < 1.0e-9 ? 1.0e-9 : max_novelty",
            "(1.0 - novelty_weight) * normalized_fitness +",
            "novelty_weight * normalized_novelty",
            "ordered_f64_decision_key_v1(",
            "value == 0.0 ? 0.0 : value",
            "metric_rows[candidate].candidate_id",
        ],
    );
    for forbidden in ["thrust::", "std::sort", "partial_cmp", "atomicAdd"] {
        assert!(
            !cuda.contains(forbidden),
            "decision-order drift via {forbidden:?}"
        );
    }
}

#[test]
fn nonfinite_or_malformed_inputs_invalidate_the_device_seal() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &cuda,
        &[
            "all_metric_values_finite_v1(",
            "atomicExch(device_fault_word, 1u)",
            "if (!isfinite(score))",
            "decision_keys[candidate] = 0;",
            "seal->valid = *device_fault_word == 0 ? 1u : 0u;",
            "seal->valid == 0",
            "opaque device seal must be checked by the same-stream consumer",
        ],
    );
    assert!(
        !cuda.contains("partial_cmp"),
        "non-transitive CPU NaN comparator must not be reproduced"
    );
}

#[test]
fn allocation_is_checked_charged_once_and_preserves_full_discovery_reserve() {
    let header = read_required("native/resident_scoring_novelty_v1_abi.cuh");
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &header,
        &[
            "set_bitmap_bytes",
            "fitness_score_bytes",
            "novelty_score_bytes",
            "decision_key_bytes",
            "cub_scratch_bytes",
            "device_control_bytes",
            "total_device_bytes",
            "same_context_free_bytes",
            "full_discovery_reserve_bytes",
        ],
    );
    require_all(
        &cuda,
        &[
            "checked_mul_v1(",
            "checked_add_v1(",
            "cudaMemGetInfo",
            "cudaMallocAsync",
            "cudaFreeAsync",
            "scoring_store_allocation_count = 1",
            "full_discovery_reserve_bytes > same_context_free_bytes",
        ],
    );
    assert_eq!(cuda.matches("cudaMallocAsync(").count(), 1);
    for forbidden in ["cudaMalloc(", "cudaFree(", "new double[", "std::vector"] {
        assert!(
            !cuda.contains(forbidden),
            "unplanned allocation via {forbidden:?}"
        );
    }
}

#[test]
fn same_admitted_stream_and_preowned_events_are_the_only_ordering_authority() {
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &cuda,
        &[
            "import->admitted_run_stream",
            "import->metrics_ready_event",
            "import->scoring_novelty_ready_event",
            "import->metrics_ready_event != import->scoring_novelty_ready_event",
            "cudaStreamWaitEvent(created->admitted_run_stream,",
            "cudaEventRecord(run->scoring_novelty_ready_event, run->admitted_run_stream)",
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
            "ordering/transfer escape via {forbidden:?}"
        );
    }
}

#[test]
fn output_is_opaque_research_only_and_binds_math_build_and_semantic_identities() {
    let rust = read_required("src/resident_scoring_novelty_v1.rs");
    let output = section(
        &rust,
        "pub struct SealedResidentScoringNoveltyDecisionRowsV1 {",
        "\n}",
    );
    require_all(
        output,
        &[
            "run: Option<ResidentScoringNoveltyDeviceRunV1>",
            "raw: RawScoredDecisionRowsV1",
            "artifact_class: ScoringNoveltyArtifactClassV1",
            "promotion_eligibility: ScoringNoveltyPromotionEligibilityV1",
        ],
    );
    assert!(
        !output.contains("pub "),
        "sealed output fields must stay private"
    );
    require_all(
        &rust,
        &[
            "ScoringNoveltyArtifactClassV1::ResearchOnly",
            "ScoringNoveltyPromotionEligibilityV1::NotPromotionEligible",
            "metric_semantics_sha256",
            "scoring_semantics_sha256",
            "novelty_semantics_sha256",
            "rank_semantics_sha256",
            "cuda_build_manifest_sha256",
            "cuda_math_flags_sha256",
            "device_seal: *const RawDeviceSealV1",
            "final_compact_readback_count == 0",
        ],
    );
    for forbidden in [
        "pub decision_keys",
        "pub metrics",
        "pub fn raw",
        "pub fn read",
        "impl Clone for SealedResidentScoringNoveltyDecisionRowsV1",
        "Deserialize",
    ] {
        assert!(
            !rust.contains(forbidden),
            "sealed-output escape via {forbidden:?}"
        );
    }
}

#[test]
fn build_math_authority_is_explicit_and_no_cpu_or_f32_path_exists() {
    let rust = read_required("src/resident_scoring_novelty_v1.rs");
    let cuda = read_required("native/resident_scoring_novelty_v1.cu");
    require_all(
        &rust,
        &[
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "CPU/GPU golden parity is required before strict full-discovery authority",
        ],
    );
    require_all(
        &cuda,
        &[
            "identity_equal_v1(import->cuda_build_manifest_sha256,",
            "plan->cuda_build_manifest_sha256)",
            "identity_equal_v1(import->cuda_math_flags_sha256,",
            "plan->cuda_math_flags_sha256)",
            "identity_equal_v1(import->cuda_device_identity_sha256,",
            "plan->cuda_device_identity_sha256)",
            "identity_equal_v1(import->primary_context_identity_sha256,",
            "plan->primary_context_identity_sha256)",
            "identity_equal_v1(import->run_stream_identity_sha256,",
            "plan->run_stream_identity_sha256)",
            "identity_equal_v1(import->metric_semantics_sha256,",
            "plan->metric_semantics_sha256)",
            "identity_equal_v1(import->gene_schema_sha256,",
            "plan->gene_schema_sha256)",
            "identity_equal_v1(import->scenario_order_semantics_sha256,",
            "plan->scenario_order_semantics_sha256)",
        ],
    );
    for forbidden in [
        "AllowCpu",
        "cpu_forced",
        "fallback",
        "std::thread",
        "rayon",
        "float ",
        "__float",
        "cudaMemcpy",
    ] {
        assert!(!rust.contains(forbidden), "Rust escape via {forbidden:?}");
        assert!(!cuda.contains(forbidden), "CUDA escape via {forbidden:?}");
    }
}
