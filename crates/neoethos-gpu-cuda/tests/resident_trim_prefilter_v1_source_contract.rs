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
            "resident trim/prefilter native V1 source is missing {token:?}"
        );
    }
}

#[test]
fn rust_owner_is_move_only_same_run_and_leaks_on_ambiguous_drop() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let run = section(
        &rust,
        "pub struct ResidentTrimPrefilterDeviceRunV1 {",
        "\n}",
    );
    require_all(
        run,
        &[
            "native: NonNull<NativeResidentTrimPrefilterRunV1>",
            "parent_import: Option<ResidentTrimPrefilterParentImportV1>",
            "sealed_schema: Option<SealedResidentColumnClassificationV1>",
            "full_admission: Option<ResidentTrimPrefilterFullDiscoveryAdmissionV1>",
            "state: ResidentTrimPrefilterRunStateV1",
            "selected_cuda_ordinal: u32",
            "primary_context_identity_sha256: [u8; 32]",
            "run_stream_identity_sha256: [u8; 32]",
            "cuda_build_manifest_sha256: [u8; 32]",
        ],
    );
    assert!(
        !run.contains("pub "),
        "native owner fields must remain private"
    );
    require_all(
        &rust,
        &[
            "#[must_use = \"resident trim/prefilter work must be consumed by the same GPU run\"]",
            "ResidentTrimPrefilterRunStateV1::StrictIdle",
            "ResidentTrimPrefilterRunStateV1::InFlight",
            "ResidentTrimPrefilterRunStateV1::Sealed",
            "ResidentTrimPrefilterRunStateV1::Poisoned",
            "impl Drop for ResidentTrimPrefilterDeviceRunV1",
            "leak_ambiguous_resident_trim_prefilter_run_v1(",
        ],
    );
    for forbidden in [
        "impl Clone for ResidentTrimPrefilterDeviceRunV1",
        "impl Default for ResidentTrimPrefilterDeviceRunV1",
        "pub fn from_raw",
        "pub fn raw_",
        "pub fn wait",
        "pub fn read",
        "Deserialize",
    ] {
        assert!(
            !rust.contains(forbidden),
            "authority escape via {forbidden:?}"
        );
    }
}

#[test]
fn private_abi_binds_parent_schema_scope_build_stream_and_preowned_events() {
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    let import = section(&header, "struct NeoResidentTrimPrefilterImportV1 {", "\n};");
    require_all(
        import,
        &[
            "cudaStream_t admitted_run_stream;",
            "cudaEvent_t parent_ready_event;",
            "cudaEvent_t schema_ready_event;",
            "cudaEvent_t trim_prefilter_ready_event;",
            "const double* indicators_bar_major;",
            "const unsigned char* indicators_validity_u4;",
            "const double* close;",
            "const double* high;",
            "const double* low;",
            "const unsigned char* column_class_flags_device;",
            "const std::uint32_t* timeframe_group_ids_device;",
            "const unsigned char* template_force_keep_flags_device;",
            "std::uint8_t canonical_content_merkle_sha256[32];",
            "std::uint8_t ordered_feature_schema_sha256[32];",
            "std::uint8_t column_classification_content_sha256[32];",
            "std::uint8_t primary_context_identity_sha256[32];",
            "std::uint8_t run_stream_identity_sha256[32];",
            "std::uint8_t cuda_build_manifest_sha256[32];",
            "std::uint8_t cuda_math_flags_sha256[32];",
        ],
    );
    require_all(
        &header,
        &[
            "static_assert(sizeof(void*) == 8",
            "static_assert(sizeof(NeoResidentTrimPrefilterImportV1) == 560",
            "static_assert(sizeof(NeoResidentTrimPrefilterPlanV1) == 608",
            "static_assert(sizeof(NeoResidentTrimPrefilterAllocationReceiptV1) == 200",
            "static_assert(sizeof(NeoResidentTrimPrefilterViewsV1) == 344",
            "NeoResidentTrimPrefilterPlanV1",
            "NeoResidentTrimPrefilterAllocationReceiptV1",
            "NeoResidentTrimPrefilterDeviceSealV1",
            "NeoResidentTrimPrefilterViewsV1",
        ],
    );
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    require_all(
        &rust,
        &[
            "const _: [(); 560] = [(); mem::size_of::<RawResidentTrimPrefilterImportV1>()]",
            "const _: [(); 608] = [(); mem::size_of::<RawResidentTrimPrefilterPlanV1>()]",
            "mem::size_of::<RawResidentTrimPrefilterAllocationReceiptV1>()",
            "const _: [(); 344] = [(); mem::size_of::<RawResidentTrimPrefilterViewsV1>()]",
        ],
    );
    for forbidden in ["cudaEventCreate", "cudaStreamCreate", "cudaSetDevice"] {
        assert!(
            !header.contains(forbidden),
            "ABI creates route state via {forbidden:?}"
        );
    }
}

#[test]
fn sealed_views_are_revalidated_against_every_retained_identity_and_event() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    require_all(
        &rust,
        &[
            "struct ResidentTrimPrefilterExpectedViewsV1 {",
            "expected_views: ResidentTrimPrefilterExpectedViewsV1",
            "views.trim_prefilter_ready_event",
            "run.expected_views.trim_prefilter_ready_event.as_ptr()",
            "views.parent_row_count != run.expected_views.parent_row_count",
            "views.parent_column_count != run.expected_views.parent_column_count",
            "views.selection_row_start != run.expected_views.selection_row_start",
            "views.selection_row_end != run.expected_views.selection_row_end",
            "views.holdout_row_start != run.expected_views.holdout_row_start",
            "views.holdout_row_end != run.expected_views.holdout_row_end",
            "views.plan_identity_sha256 != run.expected_views.plan_identity_sha256",
            "views.view_semantics_sha256 != run.expected_views.view_semantics_sha256",
            "views.canonical_content_merkle_sha256",
            "run.expected_views.canonical_content_merkle_sha256",
            "views.ordered_feature_schema_sha256",
            "run.expected_views.ordered_feature_schema_sha256",
            "views.cuda_device_identity_sha256",
            "run.expected_views.cuda_device_identity_sha256",
            "ready.same_stream_enqueue_count != expected_ready_enqueue_count",
        ],
    );
}

#[test]
fn exact_outer_split_suffix_trim_and_view_ranges_never_copy_values() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &cuda,
        &[
            "resolve_absolute_view_ranges_v1(",
            "floor(0.8 * static_cast<double>(parent_row_count))",
            "outer_split_at < 64U",
            "const std::uint64_t row_cap =",
            "min_nonzero_v1(global_row_cap, timeframe_row_cap)",
            "const std::uint64_t selection_row_start =",
            "outer_split_at - retained_selection_rows",
            "selection_row_end = outer_split_at",
            "holdout_row_start = outer_split_at",
            "holdout_row_end = parent_row_count",
            "same_selected_column_map_for_holdout = 1U",
        ],
    );
    for forbidden in [
        "cudaMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost",
        "upload_dataset",
        "transpose",
        "feature_major",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "resident view copies data via {forbidden:?}"
        );
    }
}

#[test]
fn first_passage_labels_match_directional_cost_geometry_and_fail_closed() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    let label = section(
        &cuda,
        "__global__ void first_passage_labels_kernel_v1(",
        "\n}",
    );
    require_all(
        label,
        &[
            "rolling_atr_simple_finite_mean_v1(",
            "long_take = entry + take_distance + round_trip_cost_price",
            "long_stop = entry - stop_distance + round_trip_cost_price",
            "short_take = entry - take_distance - round_trip_cost_price",
            "short_stop = entry + stop_distance - round_trip_cost_price",
            "remaining_horizon = selection_rows - 1U - row",
            "max_hold_bars < remaining_horizon ? max_hold_bars : remaining_horizon",
            "horizon_end = row + horizon_step",
            "long_take_hit && long_stop_hit",
            "short_take_hit && short_stop_hit",
            "long_labels[row] = long_label",
            "short_labels[row] = short_label",
        ],
    );
    require_all(
        &cuda,
        &[
            "MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1",
            "max_u64_v1(decided_long, decided_short) <",
            "MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1",
            "NEO_TRIM_PREFILTER_FAULT_INSUFFICIENT_LABELS_V1",
            "device_seal->valid = 0U",
        ],
    );
    for forbidden in ["forward_return", "best_effort", "fallback_mode"] {
        assert!(
            !cuda.contains(forbidden),
            "label target changes through {forbidden:?}"
        );
    }
}

#[test]
fn cpcv_and_prefix_windows_match_current_order_and_geometry() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &cuda,
        &[
            "combination_count_checked_v1(",
            "gcd_u64_v1(",
            "const std::uint64_t value_divisor = gcd_u64_v1(value, divisor)",
            "remaining_divisor > 1U && factor % remaining_divisor != 0U",
            "value > U64_MAX_V1 / reduced_factor",
            "lexicographic_test_group_combination_v1(",
            "cpcv_training_group_range_v1(",
            "cpcv_combination_has_training_rows_v1(",
            "valid_available_combinations",
            "sampled_valid_combination_rank",
            "descriptor.available_combinations = valid_available_combinations",
            "group_size = capped_rows / split_count",
            "const std::uint64_t group_end =",
            "query_group + 1U == split_count ? capped_rows",
            ": (query_group + 1U) * group_size",
            "const std::uint64_t purge_rows =",
            "ceil_fraction_rows_v1(capped_rows, purge_fraction)",
            "const std::uint64_t embargo_rows =",
            "ceil_fraction_rows_v1(capped_rows, embargo_fraction)",
            "const std::uint64_t step =",
            "valid_available_combinations, MAXIMUM_REFIT_FOLDS_V1",
            "target_valid_rank = next_fold * step",
            "fit_tail_offset = selection_rows - capped_rows",
            "std::uint64_t prefix_train_end = static_cast<std::uint64_t>(",
            "floor(insample_fraction * static_cast<double>(selection_rows)))",
            "prefix_exclusive_end = prefix_train_end - 1U",
        ],
    );
}

#[test]
fn pairwise_correlation_is_exact_ordered_two_pass_f64_not_a_parallel_reduction() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    let kernel = section(
        &cuda,
        "__global__ void pairwise_two_pass_correlation_kernel_v1(",
        "\n}",
    );
    require_all(
        kernel,
        &[
            "if (threadIdx.x != 0U)",
            "pairwise_two_pass_one_direction_v1(",
            "worst = fmin(worst, direction_score)",
            "best = fmax(best, direction_score)",
        ],
    );
    require_all(
        &cuda,
        &[
            "for (std::uint64_t row = 0U; row < selection_rows; ++row)",
            "validity_code_v1(indicators_validity_u4, cell) == 0U",
            "used < MINIMUM_PAIRWISE_SAMPLES_V1",
            "sum_x += x",
            "sum_y += y",
            "sxx += dx * dx",
            "syy += dy * dy",
            "sxy += dx * dy",
            "sqrt(sxx * syy)",
        ],
    );
    for forbidden in [
        "cub::DeviceReduce",
        "cub::BlockReduce",
        "atomicAdd",
        "float sum_",
        "--use_fast_math",
    ] {
        assert!(
            !kernel.contains(forbidden),
            "decision math reorders through {forbidden:?}"
        );
    }
}

#[test]
fn official_cub_sort_and_select_preserve_score_ties_and_parent_order() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &cuda,
        &[
            "cub::DeviceSelect::Flagged(",
            "cub::DeviceRadixSort::SortPairsDescending(",
            "monotone_nonnegative_f64_key_v1(",
            "input_parent_indices_are_ascending",
            "stable_equal_keys_preserve_parent_index_order",
            "finalize_state_template_timeframe_quota_kernel_v1",
            "select_ascending_parent_map_kernel_v1",
            "selected_compact_to_parent_columns_device",
            "selected_column_count_device",
        ],
    );
    for forbidden in [
        "thrust::sort",
        "std::sort",
        "partial_sort",
        "CustomDeviceRadixSort",
        "cublas",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "ranking authority drifts via {forbidden:?}"
        );
    }
}

#[test]
fn allocation_is_checked_same_context_and_charges_every_buffer() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    require_all(
        &rust,
        &[
            "impl ResidentTrimPrefilterNativeScratchBytesV1",
            "pub fn query_from_same_run(",
            "query_resident_trim_prefilter_allocation_v1(",
            "same_context_free_bytes",
            "full_discovery_reserve_bytes",
            "trim_prefilter_reserved_bytes",
            "AllocationReceiptMismatch",
            "ArithmeticOverflow",
            "fields.max_hold_bars == 0",
            "fields.parent_column_count > MAX_GRID_X_V1",
            "selection_rows > MAX_GRID_X_V1 * LAUNCH_THREADS_V1",
        ],
    );
    require_all(
        &header,
        &[
            "long_labels_bytes;",
            "short_labels_bytes;",
            "label_census_bytes;",
            "fold_descriptor_bytes;",
            "column_score_bytes;",
            "column_instability_bytes;",
            "column_rankability_bytes;",
            "radix_key_ping_pong_bytes;",
            "radix_index_ping_pong_bytes;",
            "timeframe_group_counter_bytes;",
            "selected_column_map_bytes;",
            "selected_column_count_bytes;",
            "cub_select_scratch_bytes;",
            "cub_radix_sort_scratch_bytes;",
            "device_seal_bytes;",
            "retained_device_bytes;",
            "peak_device_bytes;",
        ],
    );
}

#[test]
fn allocation_and_plan_hashes_bind_every_semantic_and_memory_component() {
    let search =
        fs::read_to_string(manifest_dir().join(
            "../neoethos-search/src/gpu_full_discovery/gpu_resident_trim_prefilter_view_v1.rs",
        ))
        .expect("read additive Search resident trim/prefilter source");
    let allocation_hash = section(&search, "let allocation_plan_sha256 = sha256_v1(&[", "]);");
    require_all(
        allocation_hash,
        &[
            "&long_labels_bytes.to_le_bytes()",
            "&short_labels_bytes.to_le_bytes()",
            "&label_census_bytes.to_le_bytes()",
            "&fold_descriptor_bytes.to_le_bytes()",
            "&column_score_bytes.to_le_bytes()",
            "&column_instability_bytes.to_le_bytes()",
            "&column_rankability_bytes.to_le_bytes()",
            "&state_template_timeframe_metadata_bytes.to_le_bytes()",
            "&radix_key_ping_pong_bytes.to_le_bytes()",
            "&radix_index_ping_pong_bytes.to_le_bytes()",
            "&timeframe_group_counter_bytes.to_le_bytes()",
            "&selected_column_map_bytes.to_le_bytes()",
            "&selected_column_count_bytes.to_le_bytes()",
            "&cub_select_scratch_bytes.to_le_bytes()",
            "&cub_radix_sort_scratch_bytes.to_le_bytes()",
            "&device_seal_bytes.to_le_bytes()",
            "&retained_device_bytes.to_le_bytes()",
            "&peak_device_bytes.to_le_bytes()",
            "&full_discovery_reserve_bytes.to_le_bytes()",
        ],
    );
    let plan_hash = section(&search, "fn compute_resolved_plan_identity_v1(", "\n}");
    require_all(
        plan_hash,
        &[
            "&plan.semantics.parent_column_count.to_le_bytes()",
            "&plan.semantics.configured_top_k.to_le_bytes()",
            "&plan.semantics.resolved_top_k.to_le_bytes()",
            "&plan.semantics.minimum_per_timeframe.to_le_bytes()",
            "&plan.semantics.insample_fraction_bits.to_le_bytes()",
            "&plan.semantics.max_hold_bars.to_le_bytes()",
            "&plan.semantics.atr_period.to_le_bytes()",
            "&plan.semantics.stop_atr_multiplier_bits.to_le_bytes()",
            "&plan.semantics.reward_risk_ratio_bits.to_le_bytes()",
            "&plan.semantics.round_trip_cost_price_bits.to_le_bytes()",
            "&plan.semantics.cpcv_split_count.to_le_bytes()",
            "&plan.semantics.cpcv_test_group_count.to_le_bytes()",
            "&plan.semantics.cpcv_embargo_fraction_bits.to_le_bytes()",
            "&plan.semantics.cpcv_purge_fraction_bits.to_le_bytes()",
            "&plan.semantics.cpcv_max_rows.to_le_bytes()",
            "&plan.selected_cuda_ordinal.to_le_bytes()",
        ],
    );
}

#[test]
fn async_release_failure_never_publishes_or_destroys_ambiguous_ownership() {
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &cuda,
        &[
            "cudaError_t release_one_async_v1(",
            "cudaError_t release_intermediate_buffers_async_v1(",
            "cudaError_t release_all_buffers_async_v1(",
            "status = release_intermediate_buffers_async_v1(run)",
            "if (release_all_buffers_async_v1(run) != cudaSuccess)",
            "return NEO_TRIM_PREFILTER_STATUS_CUDA_ERROR_V1",
        ],
    );
    let release = section(
        &cuda,
        "extern \"C\" std::int32_t enqueue_resident_trim_prefilter_release_v1(",
        "\n}",
    );
    let failure = release
        .find("if (release_all_buffers_async_v1(run) != cudaSuccess)")
        .expect("explicit release checks every async free");
    let deletion = release
        .find("delete run")
        .expect("successful release deletes run");
    assert!(
        failure < deletion,
        "native owner was deleted before async-free success was known"
    );
}

#[test]
fn enqueue_is_stream_ordered_with_zero_host_transfer_wait_or_sync() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &cuda,
        &[
            "cudaStreamWaitEvent(run->admitted_run_stream,",
            "run->parent_ready_event, 0U)",
            "run->schema_ready_event, 0U)",
            "cudaEventRecord(run->trim_prefilter_ready_event,",
            "run->admitted_run_stream)",
            "same_stream_enqueue_count",
            "intermediate_host_wait_count = 0U",
            "intermediate_readback_count = 0U",
            "host_to_device_transfer_count = 0U",
            "device_to_host_transfer_count = 0U",
            "explicit_synchronization_count = 0U",
        ],
    );
    require_all(
        &rust,
        &[
            "intermediate_host_wait_count",
            "intermediate_readback_count",
            "host_to_device_transfer_count",
            "device_to_host_transfer_count",
            "explicit_synchronization_count",
        ],
    );
    for forbidden in [
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cudaDeviceSynchronize",
        "cudaMemcpy(",
        "cudaMemcpyAsync(",
        "cudaEventCreate",
        "cudaEventDestroy",
        "cudaStreamCreate",
        "cudaSetDevice",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "same-run pipeline escapes via {forbidden:?}"
        );
    }
}

#[test]
fn opaque_device_seal_and_views_carry_identity_without_host_selected_count() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let header = read_required("native/resident_trim_prefilter_v1_abi.cuh");
    require_all(
        &header,
        &[
            "const std::uint32_t* selected_compact_to_parent_columns_device;",
            "const std::uint64_t* selected_column_count_device;",
            "const NeoResidentTrimPrefilterDeviceSealV1* device_seal;",
            "cudaEvent_t trim_prefilter_ready_event;",
            "std::uint8_t plan_identity_sha256[32];",
            "std::uint8_t view_semantics_sha256[32];",
        ],
    );
    require_all(
        &rust,
        &[
            "pub struct SealedResidentTrimPrefilterDeviceViewsV1",
            "selected_compact_to_parent_columns_device(&self) -> bool",
            "selected_column_count_device(&self) -> bool",
            "same_selected_column_map_for_holdout(&self) -> bool",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    for forbidden in [
        "pub selected_column_count: u64",
        "pub selected_columns:",
        "Vec<u32>",
        "Vec<usize>",
        "copy_selected",
    ] {
        assert!(
            !rust.contains(forbidden),
            "opaque result leaks through {forbidden:?}"
        );
    }
}

#[test]
fn strict_math_build_identity_and_device_parity_boundary_are_explicit() {
    let rust = read_required("src/resident_trim_prefilter_v1.rs");
    let cuda = read_required("native/resident_trim_prefilter_v1.cu");
    require_all(
        &rust,
        &[
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "cuda_math_flags_sha256",
            "cuda_build_manifest_sha256",
            "MissingExactSelectedIndexDeviceParity",
            "ResearchOnly",
            "NotPromotionEligible",
        ],
    );
    require_all(
        &cuda,
        &[
            "#include <limits>",
            "static_assert(sizeof(double) == 8",
            "constexpr double F64_INFINITY_V1 = std::numeric_limits<double>::infinity();",
            "column_scores[column] = F64_INFINITY_V1",
            "double worst = F64_INFINITY_V1",
            "column_scores[column] = -F64_INFINITY_V1",
            "isfinite(",
            "NEO_TRIM_PREFILTER_FAULT_NONFINITE_DECISION_V1",
        ],
    );
    assert_eq!(
        cuda.matches("std::numeric_limits<double>::infinity()")
            .count(),
        1,
        "positive infinity must have one portable exact definition"
    );
    assert!(
        !cuda.contains("CUDART_INF"),
        "CUDART_INF is not a portable CUDA runtime constant"
    );
    for forbidden in [
        "MINIMUM_IN_SAMPLE_ROWS_V1",
        "diag_suppress",
        "diagnostic ignored",
        "--diag-suppress",
        "-Wno-unused",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "warning-denied native source must remove dead code instead of using {forbidden:?}"
        );
    }
    assert!(!cuda.contains("--expt-relaxed-constexpr"));
}

#[test]
fn cuda_feature_builds_and_exports_trim_prefilter_without_stub_authority() {
    let lib = read_required("src/lib.rs");
    let build = read_required("build.rs");
    let stub = read_required("native/stub.cpp");

    require_all(
        &lib,
        &["#[cfg(feature = \"cuda\")]\npub mod resident_trim_prefilter_v1;"],
    );
    require_all(
        &build,
        &[
            "const DEVICE_SOURCES: [&str; 8] = [",
            "\"native/resident_trim_prefilter_v1.cu\",",
            "cargo:rerun-if-changed=native/resident_trim_prefilter_v1_abi.cuh",
        ],
    );
    for forbidden in [
        "begin_resident_trim_prefilter_device_run_v1",
        "enqueue_first_passage_labels_v1",
        "seal_resident_trim_prefilter_device_views_v1",
    ] {
        assert!(
            !stub.contains(forbidden),
            "no-CUDA stub fabricated resident trim/prefilter authority via {forbidden:?}"
        );
    }
    for source in [&lib, &build] {
        for forbidden in ["allow(dead_code)", "expect(dead_code)", "diag_suppress"] {
            assert!(
                !source.contains(forbidden),
                "production wiring suppresses its warning frontier via {forbidden:?}"
            );
        }
    }
}
